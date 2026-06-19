#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ void sincos_wrapper(scalar_t phase, scalar_t* sin_val, scalar_t* cos_val);

template <>
__device__ __forceinline__ void sincos_wrapper<float>(float phase, float* sin_val, float* cos_val) {
    sincosf(phase, sin_val, cos_val);
}

template <>
__device__ __forceinline__ void sincos_wrapper<double>(double phase, double* sin_val, double* cos_val) {
    sincos(phase, sin_val, cos_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_freq(int idx, int n, scalar_t d) {
    int shifted_idx = idx - n / 2;
    return static_cast<scalar_t>(shifted_idx) / (static_cast<scalar_t>(n) * d);
}

template <typename scalar_t>
__global__ void bandlimited_propagation_forward_kernel(
    const scalar_t* __restrict__ field_f_real,
    const scalar_t* __restrict__ field_f_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    scalar_t wavelength,
    scalar_t distance,
    int nx,
    int ny,
    scalar_t aperture_size,
    scalar_t pixel_pitch,
    scalar_t fx_max,
    scalar_t fy_max,
    scalar_t offset_x,
    scalar_t offset_y,
    scalar_t k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nx || idy >= ny) return;
    
    const int index = idy * nx + idx;
    
    const scalar_t fx_val = compute_freq(idx, nx, pixel_pitch);
    const scalar_t fy_val = compute_freq(idy, ny, pixel_pitch);
    const scalar_t abs_fx = fabsf(fx_val);
    const scalar_t abs_fy = fabsf(fy_val);
    
    if ((abs_fx >= fx_max) | (abs_fy >= fy_max)) {
        output_real[index] = 0;
        output_imag[index] = 0;
        return;
    }
    
    const scalar_t two_pi = static_cast<scalar_t>(6.283185307179586);
    const scalar_t two_pi_fx = two_pi * fx_val;
    const scalar_t two_pi_fy = two_pi * fy_val;
    const scalar_t kz_squared = k * k - two_pi_fx * two_pi_fx - two_pi_fy * two_pi_fy;
    
    if (kz_squared <= 0) {
        output_real[index] = 0;
        output_imag[index] = 0;
        return;
    }
    
    const scalar_t in_real = __ldg(&field_f_real[index]);
    const scalar_t in_imag = __ldg(&field_f_imag[index]);
    
    const scalar_t kz = sqrtf(kz_squared);
    scalar_t sin_val, cos_val;
    sincos_wrapper(distance * kz, &sin_val, &cos_val);
    
    scalar_t out_real = in_real * cos_val - in_imag * sin_val;
    scalar_t out_imag = in_real * sin_val + in_imag * cos_val;
    
    if (aperture_size > 0) {
        const scalar_t dx = static_cast<scalar_t>(idx) - offset_x;
        const scalar_t dy = static_cast<scalar_t>(idy) - offset_y;
        const scalar_t r_sq = dx * dx + dy * dy;
        if (r_sq >= aperture_size * aperture_size) {
            out_real = 0;
            out_imag = 0;
        }
    }
    
    output_real[index] = out_real;
    output_imag[index] = out_imag;
}

template <typename scalar_t>
__global__ void bandlimited_propagation_backward_kernel(
    const scalar_t* __restrict__ grad_output_real,
    const scalar_t* __restrict__ grad_output_imag,
    scalar_t* __restrict__ grad_field_real,
    scalar_t* __restrict__ grad_field_imag,
    scalar_t wavelength,
    scalar_t distance,
    int nx,
    int ny,
    scalar_t aperture_size,
    scalar_t pixel_pitch,
    scalar_t fx_max,
    scalar_t fy_max,
    scalar_t offset_x,
    scalar_t offset_y,
    scalar_t k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nx || idy >= ny) return;
    
    const int index = idy * nx + idx;
    
    const scalar_t fx_val = compute_freq(idx, nx, pixel_pitch);
    const scalar_t fy_val = compute_freq(idy, ny, pixel_pitch);
    const scalar_t abs_fx = fabsf(fx_val);
    const scalar_t abs_fy = fabsf(fy_val);
    
    if ((abs_fx >= fx_max) | (abs_fy >= fy_max)) {
        grad_field_real[index] = 0;
        grad_field_imag[index] = 0;
        return;
    }
    
    const scalar_t two_pi = static_cast<scalar_t>(6.283185307179586);
    const scalar_t two_pi_fx = two_pi * fx_val;
    const scalar_t two_pi_fy = two_pi * fy_val;
    const scalar_t kz_squared = k * k - two_pi_fx * two_pi_fx - two_pi_fy * two_pi_fy;
    
    if (kz_squared <= 0) {
        grad_field_real[index] = 0;
        grad_field_imag[index] = 0;
        return;
    }
    
    const scalar_t grad_real = __ldg(&grad_output_real[index]);
    const scalar_t grad_imag = __ldg(&grad_output_imag[index]);
    
    const scalar_t kz = sqrtf(kz_squared);
    scalar_t sin_val, cos_val;
    sincos_wrapper(-distance * kz, &sin_val, &cos_val);
    
    scalar_t out_real = grad_real * cos_val - grad_imag * sin_val;
    scalar_t out_imag = grad_real * sin_val + grad_imag * cos_val;
    
    if (aperture_size > 0) {
        const scalar_t dx = static_cast<scalar_t>(idx) - offset_x;
        const scalar_t dy = static_cast<scalar_t>(idy) - offset_y;
        const scalar_t r_sq = dx * dx + dy * dy;
        if (r_sq >= aperture_size * aperture_size) {
            out_real = 0;
            out_imag = 0;
        }
    }
    
    grad_field_real[index] = out_real;
    grad_field_imag[index] = out_imag;
}

void bandlimited_propagation_forward_cuda(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    torch::Tensor& output_real,
    torch::Tensor& output_imag,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch,
    int nx,
    int ny) {
    
    const float two_pi = 2.0f * M_PI;
    const float k = two_pi / wavelength;
    const float inv_lambda = 1.0f / wavelength;
    const float inv_x = 1.0f / (pixel_pitch * nx);
    const float inv_y = 1.0f / (pixel_pitch * ny);
    const float two_dist_inv_x = 2.0f * distance * inv_x;
    const float two_dist_inv_y = 2.0f * distance * inv_y;
    
    const float fx_max = inv_lambda * rsqrtf(two_dist_inv_x * two_dist_inv_x + 1.0f);
    const float fy_max = inv_lambda * rsqrtf(two_dist_inv_y * two_dist_inv_y + 1.0f);
    
    const float offset_x = nx * 0.5f - 0.5f;
    const float offset_y = ny * 0.5f - 0.5f;
    
    int bx = 16, by = 16;
    if (nx >= 512 || ny >= 512) {
        bx = 32; by = 32;
    }
    
    dim3 threads(bx, by);
    dim3 blocks((nx + bx - 1) / bx, (ny + by - 1) / by);
    
    AT_DISPATCH_FLOATING_TYPES(field_f_real.scalar_type(), "bandlimited_propagation_forward", ([&] {
        bandlimited_propagation_forward_kernel<scalar_t><<<blocks, threads>>>(
            field_f_real.data_ptr<scalar_t>(),
            field_f_imag.data_ptr<scalar_t>(),
            output_real.data_ptr<scalar_t>(),
            output_imag.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            nx,
            ny,
            static_cast<scalar_t>(aperture_size),
            static_cast<scalar_t>(pixel_pitch),
            static_cast<scalar_t>(fx_max),
            static_cast<scalar_t>(fy_max),
            static_cast<scalar_t>(offset_x),
            static_cast<scalar_t>(offset_y),
            static_cast<scalar_t>(k));
    }));
}

void bandlimited_propagation_backward_cuda(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    torch::Tensor& grad_field_real,
    torch::Tensor& grad_field_imag,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch,
    int nx,
    int ny) {
    
    const float two_pi = 2.0f * M_PI;
    const float k = two_pi / wavelength;
    const float inv_lambda = 1.0f / wavelength;
    const float inv_x = 1.0f / (pixel_pitch * nx);
    const float inv_y = 1.0f / (pixel_pitch * ny);
    const float two_dist_inv_x = 2.0f * distance * inv_x;
    const float two_dist_inv_y = 2.0f * distance * inv_y;
    
    const float fx_max = inv_lambda * rsqrtf(two_dist_inv_x * two_dist_inv_x + 1.0f);
    const float fy_max = inv_lambda * rsqrtf(two_dist_inv_y * two_dist_inv_y + 1.0f);
    
    const float offset_x = nx * 0.5f - 0.5f;
    const float offset_y = ny * 0.5f - 0.5f;
    
    int bx = 16, by = 16;
    if (nx >= 512 || ny >= 512) {
        bx = 32; by = 32;
    }
    
    dim3 threads(bx, by);
    dim3 blocks((nx + bx - 1) / bx, (ny + by - 1) / by);
    
    AT_DISPATCH_FLOATING_TYPES(grad_output_real.scalar_type(), "bandlimited_propagation_backward", ([&] {
        bandlimited_propagation_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_real.data_ptr<scalar_t>(),
            grad_output_imag.data_ptr<scalar_t>(),
            grad_field_real.data_ptr<scalar_t>(),
            grad_field_imag.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            nx,
            ny,
            static_cast<scalar_t>(aperture_size),
            static_cast<scalar_t>(pixel_pitch),
            static_cast<scalar_t>(fx_max),
            static_cast<scalar_t>(fy_max),
            static_cast<scalar_t>(offset_x),
            static_cast<scalar_t>(offset_y),
            static_cast<scalar_t>(k));
    }));
}