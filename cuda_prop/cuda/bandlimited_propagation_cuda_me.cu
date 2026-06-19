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
__global__ void bandlimited_propagation_me_forward_kernel(
    const scalar_t* __restrict__ field_f_real,
    const scalar_t* __restrict__ field_f_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    scalar_t wavelength,
    scalar_t distance,
    scalar_t aperture_size,
    scalar_t pixel_pitch,
    int nx,
    int ny
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nx || idy >= ny) return;
    
    const int index = idy * nx + idx;
    
    const scalar_t fx = compute_freq(idx, nx, pixel_pitch);
    const scalar_t fy = compute_freq(idy, ny, pixel_pitch);
    
    const scalar_t Lx = pixel_pitch * static_cast<scalar_t>(nx);
    const scalar_t Ly = pixel_pitch * static_cast<scalar_t>(ny);
    const scalar_t two_dist_inv_x = static_cast<scalar_t>(2) * distance / Lx;
    const scalar_t two_dist_inv_y = static_cast<scalar_t>(2) * distance / Ly;
    
    const scalar_t fx_max = static_cast<scalar_t>(1) / (wavelength * sqrtf(two_dist_inv_x * two_dist_inv_x + static_cast<scalar_t>(1)));
    const scalar_t fy_max = static_cast<scalar_t>(1) / (wavelength * sqrtf(two_dist_inv_y * two_dist_inv_y + static_cast<scalar_t>(1)));
    
    const bool bandlimit = (fabsf(fx) < fx_max) && (fabsf(fy) < fy_max);
    
    if (!bandlimit) {
        output_real[index] = static_cast<scalar_t>(0);
        output_imag[index] = static_cast<scalar_t>(0);
        return;
    }
    
    const scalar_t two_pi = static_cast<scalar_t>(6.283185307179586);
    const scalar_t k = two_pi / wavelength;
    const scalar_t r2 = fx * fx + fy * fy;
    const scalar_t kz_sq = k * k - (two_pi * two_pi) * r2;
    
    scalar_t kz;
    if (kz_sq > static_cast<scalar_t>(0)) {
        kz = sqrtf(kz_sq);
    } else {
        kz = static_cast<scalar_t>(0);
    }
    
    const scalar_t phase = distance * kz;
    scalar_t sin_val, cos_val;
    sincos_wrapper(phase, &sin_val, &cos_val);
    
    const scalar_t f_real = __ldg(&field_f_real[index]);
    const scalar_t f_imag = __ldg(&field_f_imag[index]);
    
    output_real[index] = f_real * cos_val - f_imag * sin_val;
    output_imag[index] = f_real * sin_val + f_imag * cos_val;
}

template <typename scalar_t>
__global__ void bandlimited_propagation_me_apply_aperture_kernel(
    scalar_t* __restrict__ field_real,
    scalar_t* __restrict__ field_imag,
    scalar_t aperture_size,
    int nx,
    int ny
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nx || idy >= ny) return;
    
    const int index = idy * nx + idx;
    
    if (aperture_size > static_cast<scalar_t>(0)) {
        const scalar_t offset_x = static_cast<scalar_t>(nx) * static_cast<scalar_t>(0.5) - static_cast<scalar_t>(0.5);
        const scalar_t offset_y = static_cast<scalar_t>(ny) * static_cast<scalar_t>(0.5) - static_cast<scalar_t>(0.5);
        const scalar_t dx = static_cast<scalar_t>(idx) - offset_x;
        const scalar_t dy = static_cast<scalar_t>(idy) - offset_y;
        const scalar_t r_sq = dx * dx + dy * dy;
        
        if (r_sq >= aperture_size * aperture_size) {
            field_real[index] = static_cast<scalar_t>(0);
            field_imag[index] = static_cast<scalar_t>(0);
        }
    }
}

template <typename scalar_t>
__global__ void bandlimited_propagation_me_backward_kernel(
    const scalar_t* __restrict__ grad_real,
    const scalar_t* __restrict__ grad_imag,
    scalar_t* __restrict__ output_real,
    scalar_t* __restrict__ output_imag,
    scalar_t wavelength,
    scalar_t distance,
    scalar_t pixel_pitch,
    int nx,
    int ny
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nx || idy >= ny) return;
    
    const int index = idy * nx + idx;
    
    const scalar_t fx = compute_freq(idx, nx, pixel_pitch);
    const scalar_t fy = compute_freq(idy, ny, pixel_pitch);
    
    const scalar_t Lx = pixel_pitch * static_cast<scalar_t>(nx);
    const scalar_t Ly = pixel_pitch * static_cast<scalar_t>(ny);
    const scalar_t two_dist_inv_x = static_cast<scalar_t>(2) * distance / Lx;
    const scalar_t two_dist_inv_y = static_cast<scalar_t>(2) * distance / Ly;
    
    const scalar_t fx_max = static_cast<scalar_t>(1) / (wavelength * sqrtf(two_dist_inv_x * two_dist_inv_x + static_cast<scalar_t>(1)));
    const scalar_t fy_max = static_cast<scalar_t>(1) / (wavelength * sqrtf(two_dist_inv_y * two_dist_inv_y + static_cast<scalar_t>(1)));
    
    const bool bandlimit = (fabsf(fx) < fx_max) && (fabsf(fy) < fy_max);
    
    if (!bandlimit) {
        output_real[index] = static_cast<scalar_t>(0);
        output_imag[index] = static_cast<scalar_t>(0);
        return;
    }
    
    const scalar_t two_pi = static_cast<scalar_t>(6.283185307179586);
    const scalar_t k = two_pi / wavelength;
    const scalar_t r2 = fx * fx + fy * fy;
    const scalar_t kz_sq = k * k - (two_pi * two_pi) * r2;
    
    scalar_t kz;
    if (kz_sq > static_cast<scalar_t>(0)) {
        kz = sqrtf(kz_sq);
    } else {
        kz = static_cast<scalar_t>(0);
    }
    
    const scalar_t phase = -distance * kz;
    scalar_t sin_val, cos_val;
    sincos_wrapper(phase, &sin_val, &cos_val);
    
    const scalar_t g_real = __ldg(&grad_real[index]);
    const scalar_t g_imag = __ldg(&grad_imag[index]);
    
    output_real[index] = g_real * cos_val - g_imag * sin_val;
    output_imag[index] = g_real * sin_val + g_imag * cos_val;
}

void bandlimited_propagation_me_forward_cuda(
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
    
    int bx = 16, by = 16;
    if (nx >= 512 || ny >= 512) {
        bx = 32; by = 32;
    }
    
    dim3 threads(bx, by);
    dim3 blocks((nx + bx - 1) / bx, (ny + by - 1) / by);
    
    AT_DISPATCH_FLOATING_TYPES(field_f_real.scalar_type(), "bandlimited_propagation_me_forward", ([&] {
        bandlimited_propagation_me_forward_kernel<scalar_t><<<blocks, threads>>>(
            field_f_real.data_ptr<scalar_t>(),
            field_f_imag.data_ptr<scalar_t>(),
            output_real.data_ptr<scalar_t>(),
            output_imag.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            static_cast<scalar_t>(aperture_size),
            static_cast<scalar_t>(pixel_pitch),
            nx,
            ny);
    }));
}

void bandlimited_propagation_me_apply_aperture_cuda(
    torch::Tensor& field_real,
    torch::Tensor& field_imag,
    float aperture_size,
    int nx,
    int ny) {
    
    int bx = 16, by = 16;
    if (nx >= 512 || ny >= 512) {
        bx = 32; by = 32;
    }
    
    dim3 threads(bx, by);
    dim3 blocks((nx + bx - 1) / bx, (ny + by - 1) / by);
    
    AT_DISPATCH_FLOATING_TYPES(field_real.scalar_type(), "bandlimited_propagation_me_apply_aperture", ([&] {
        bandlimited_propagation_me_apply_aperture_kernel<scalar_t><<<blocks, threads>>>(
            field_real.data_ptr<scalar_t>(),
            field_imag.data_ptr<scalar_t>(),
            static_cast<scalar_t>(aperture_size),
            nx,
            ny);
    }));
}

void bandlimited_propagation_me_backward_cuda(
    const torch::Tensor& grad_real,
    const torch::Tensor& grad_imag,
    torch::Tensor& output_real,
    torch::Tensor& output_imag,
    float wavelength,
    float distance,
    float pixel_pitch,
    int nx,
    int ny) {
    
    int bx = 16, by = 16;
    if (nx >= 512 || ny >= 512) {
        bx = 32; by = 32;
    }
    
    dim3 threads(bx, by);
    dim3 blocks((nx + bx - 1) / bx, (ny + by - 1) / by);
    
    AT_DISPATCH_FLOATING_TYPES(grad_real.scalar_type(), "bandlimited_propagation_me_backward", ([&] {
        bandlimited_propagation_me_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_real.data_ptr<scalar_t>(),
            grad_imag.data_ptr<scalar_t>(),
            output_real.data_ptr<scalar_t>(),
            output_imag.data_ptr<scalar_t>(),
            static_cast<scalar_t>(wavelength),
            static_cast<scalar_t>(distance),
            static_cast<scalar_t>(pixel_pitch),
            nx,
            ny);
    }));
}