#include "gaussian_2d.h"
#include "config_2d.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace gaussian_2d_cuda {

// Optimized backward kernel using tile-based approach (simplified)
template <int CHANNELS>
__global__ void renderBackwardTileKernel(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means_2d,
    const float* __restrict__ inv_cov_2d,
    const float* __restrict__ colours,
    const float* __restrict__ phase,
    const float* __restrict__ opacities,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ grad_output_real,
    const float* __restrict__ grad_output_imag,
    float* __restrict__ grad_means_2d,
    float* __restrict__ grad_inv_cov_2d,
    float* __restrict__ grad_colours,
    float* __restrict__ grad_phase,
    float* __restrict__ grad_opacities,
    int N, int num_channels,
    int W, int H)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    // Optimized shared memory layout
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float3 collected_inv_cov[BLOCK_SIZE];
    __shared__ float collected_opacities[BLOCK_SIZE];
    __shared__ float collected_colors[CHANNELS * BLOCK_SIZE];
    __shared__ float collected_phases[CHANNELS * BLOCK_SIZE];

    // Gradient computation parameters
    const float ddelx_dx = 1.0f;
    const float ddely_dy = 1.0f;

    // Load gradients for this pixel
    float dL_dpixel_real[CHANNELS];
    float dL_dpixel_imag[CHANNELS];
    if (inside) {
        for (int i = 0; i < num_channels; i++) {
            int idx = (i * H * W) + pix_id;
            dL_dpixel_real[i] = grad_output_real[idx];
            dL_dpixel_imag[i] = grad_output_imag[idx];
        }
    }

    const float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;

    uint32_t contributor = toDo;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_rec_real[CHANNELS] = {0};
    float accum_rec_imag[CHANNELS] = {0};
    float last_alpha = 0;
    float last_color_real[CHANNELS] = {0};
    float last_color_imag[CHANNELS] = {0};

    // Main rendering loop
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            const int coll_id = point_list[range.y - progress - 1];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = means_2d[coll_id];
            
            // Load inverse covariance
            const float* inv_cov_ptr = &inv_cov_2d[coll_id * 3];
            collected_inv_cov[block.thread_rank()] = make_float3(inv_cov_ptr[0], inv_cov_ptr[1], inv_cov_ptr[2]);
            
            collected_opacities[block.thread_rank()] = opacities[coll_id];
            
            // Load color and phase
            for (int ch = 0; ch < min(num_channels, CHANNELS); ch++) {
                collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colours[coll_id * num_channels + ch];
                collected_phases[ch * BLOCK_SIZE + block.thread_rank()] = phase[coll_id * num_channels + ch];
            }
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor--;
            if (contributor >= last_contributor) continue;

            const int global_id = collected_id[j];
            const float2 xy = collected_xy[j];
            const float opacity = collected_opacities[j];
            
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            
            // Get inverse covariance
            const float3 inv_cov = collected_inv_cov[j];
            const float inv_00 = inv_cov.x;
            const float inv_01 = inv_cov.y;
            const float inv_11 = inv_cov.z;
            
            const float power = -0.5f * (d.x * (inv_00 * d.x + inv_01 * d.y) +
                                        d.y * (inv_01 * d.x + inv_11 * d.y));
            if (power > 0.0f) continue;

            const float G = expf(power);
            const float alpha = min(0.99f, opacity * G);
            if (alpha < 1.0f / 255.0f) continue;

            T = T / (1.0f - alpha);
            const float dchannel_dcolor = alpha * T;

            float dL_dalpha = 0.0f;
            
            // Process channels
            for (int ch = 0; ch < min(num_channels, CHANNELS); ch++) {
                const float color = collected_colors[ch * BLOCK_SIZE + j];
                const float ph = collected_phases[ch * BLOCK_SIZE + j];
                
                float cos_ph, sin_ph;
                __sincosf(ph, &sin_ph, &cos_ph);
                
                const float color_real = color * cos_ph;
                const float color_imag = color * sin_ph;

                accum_rec_real[ch] = last_alpha * last_color_real[ch] + (1.0f - last_alpha) * accum_rec_real[ch];
                accum_rec_imag[ch] = last_alpha * last_color_imag[ch] + (1.0f - last_alpha) * accum_rec_imag[ch];
                last_color_real[ch] = color_real;
                last_color_imag[ch] = color_imag;

                const float dL_dchannel_real = dL_dpixel_real[ch];
                const float dL_dchannel_imag = dL_dpixel_imag[ch];
                
                dL_dalpha += (color_real - accum_rec_real[ch]) * dL_dchannel_real;
                dL_dalpha += (color_imag - accum_rec_imag[ch]) * dL_dchannel_imag;

                // Gradient w.r.t. color
                float dL_dcolor = dchannel_dcolor * (cos_ph * dL_dchannel_real + sin_ph * dL_dchannel_imag);
                atomicAdd(&grad_colours[global_id * num_channels + ch], dL_dcolor);

                // Gradient w.r.t. phase
                float dL_dphase = dchannel_dcolor * color * (-sin_ph * dL_dchannel_real + cos_ph * dL_dchannel_imag);
                atomicAdd(&grad_phase[global_id * num_channels + ch], dL_dphase);
            }

            dL_dalpha *= T;
            last_alpha = alpha;

            // Background contribution
            float bg_dot_dpixel = 0.0f;
            dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;

            // Gradient computation
            const float dL_dG = opacity * dL_dalpha;
            const float gdx = G * d.x;
            const float gdy = G * d.y;
            const float dG_ddelx = -gdx * inv_00 - gdy * inv_01;
            const float dG_ddely = -gdy * inv_11 - gdx * inv_01;

            // Update gradients w.r.t. 2D mean position
            atomicAdd(&grad_means_2d[global_id * 2], dL_dG * dG_ddelx * ddelx_dx);
            atomicAdd(&grad_means_2d[global_id * 2 + 1], dL_dG * dG_ddely * ddely_dy);

            // Inverse covariance gradients
            float dL_dinv_00 = -0.5f * gdx * d.x * dL_dG;
            float dL_dinv_01 = -0.5f * (gdx * d.y + gdy * d.x) * dL_dG;
            float dL_dinv_11 = -0.5f * gdy * d.y * dL_dG;

            atomicAdd(&grad_inv_cov_2d[global_id * 3], dL_dinv_00);
            atomicAdd(&grad_inv_cov_2d[global_id * 3 + 1], dL_dinv_01);
            atomicAdd(&grad_inv_cov_2d[global_id * 3 + 2], dL_dinv_11);

            // Update gradient w.r.t. opacity
            atomicAdd(&grad_opacities[global_id], G * dL_dalpha);
        }
    }
}

void BACKWARD_2D::render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    const float2* means_2d,
    const float* inv_cov_2d,
    const float* colours,
    const float* phase,
    const float* opacities,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* grad_output_real,
    const float* grad_output_imag,
    float* grad_means_2d,
    float* grad_inv_cov_2d,
    float* grad_colours,
    float* grad_phase,
    float* grad_opacities,
    int N, int num_channels,
    int W, int H)
{
    // Use template specialization for optimal performance
    switch (num_channels) {
        case 1:
            renderBackwardTileKernel<1><<<grid, block>>>(
                ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                final_Ts, n_contrib, grad_output_real, grad_output_imag,
                grad_means_2d, grad_inv_cov_2d, grad_colours, grad_phase, grad_opacities,
                N, num_channels, W, H);
            break;
        case 3:
            renderBackwardTileKernel<3><<<grid, block>>>(
                ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                final_Ts, n_contrib, grad_output_real, grad_output_imag,
                grad_means_2d, grad_inv_cov_2d, grad_colours, grad_phase, grad_opacities,
                N, num_channels, W, H);
            break;
        default:
            if (num_channels <= MAX_CHANNELS) {
                renderBackwardTileKernel<MAX_CHANNELS><<<grid, block>>>(
                    ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                    final_Ts, n_contrib, grad_output_real, grad_output_imag,
                    grad_means_2d, grad_inv_cov_2d, grad_colours, grad_phase, grad_opacities,
                    N, num_channels, W, H);
            }
            break;
    }
}

// Optimized backward pass for 2D covariance inversion
__global__ void invert_2d_covariance_backward_kernel(
    const float* __restrict__ grad_inv_cov,
    const float* __restrict__ cov_2d,
    float* __restrict__ grad_cov_2d,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load covariance matrix elements (2x2 symmetric matrix)
    const float a = cov_2d[idx * 4 + 0]; // (0,0)
    const float b = cov_2d[idx * 4 + 1]; // (0,1)
    const float c = cov_2d[idx * 4 + 2]; // (1,0) = b for symmetric
    const float d = cov_2d[idx * 4 + 3]; // (1,1)
    
    // Load gradients of inverse covariance (3 elements: inv_00, inv_01, inv_11)
    const float grad_inv_00 = grad_inv_cov[idx * 3 + 0];
    const float grad_inv_01 = grad_inv_cov[idx * 3 + 1];
    const float grad_inv_11 = grad_inv_cov[idx * 3 + 2];
    
    // Compute determinant
    const float det = a * d - b * c;
    const float inv_det = 1.0f / det;

    // Chain rule: dL/dA = dL/dA^-1 * dA^-1/dA
    // For 2x2 matrix inversion gradient (using formula for symmetric case)
    const float ddet_da = d;
    const float ddet_db = -2.0f * b;  // Since b = c for symmetric matrix
    const float ddet_dd = a;
    
    // Gradients w.r.t. covariance elements
    float grad_a = 0.0f;
    float grad_b = 0.0f;
    float grad_d = 0.0f;
    
    // Gradient through inv_00 = d/det
    grad_a += grad_inv_00 * (-d * inv_det * inv_det * ddet_da);
    grad_b += grad_inv_00 * (-d * inv_det * inv_det * ddet_db);
    grad_d += grad_inv_00 * (inv_det - d * inv_det * inv_det * ddet_dd);
    
    // Gradient through inv_01 = -b/det
    grad_a += grad_inv_01 * (b * inv_det * inv_det * ddet_da);
    grad_b += grad_inv_01 * (-inv_det + b * inv_det * inv_det * ddet_db);
    grad_d += grad_inv_01 * (b * inv_det * inv_det * ddet_dd);
    
    // Gradient through inv_11 = a/det
    grad_a += grad_inv_11 * (inv_det - a * inv_det * inv_det * ddet_da);
    grad_b += grad_inv_11 * (-a * inv_det * inv_det * ddet_db);
    grad_d += grad_inv_11 * (-a * inv_det * inv_det * ddet_dd);
    
    // Write gradients (symmetric matrix)
    grad_cov_2d[idx * 4 + 0] = grad_a;  // (0,0)
    grad_cov_2d[idx * 4 + 1] = grad_b;  // (0,1)
    grad_cov_2d[idx * 4 + 2] = grad_b;  // (1,0) = (0,1)
    grad_cov_2d[idx * 4 + 3] = grad_d;  // (1,1)
}

__global__ void compute_2d_covariance_backward_kernel(
    const float* __restrict__ grad_cov_2d,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    float* __restrict__ grad_scales,
    float* __restrict__ grad_rotations,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const float sx = scales[idx * 2];
    const float sy = scales[idx * 2 + 1];
    const float theta = rotations[idx];
    
    float cos_theta, sin_theta;
    __sincosf(theta, &sin_theta, &cos_theta);
    
    // Load gradient of covariance matrix
    const float grad_c00 = grad_cov_2d[idx * 4 + 0];
    const float grad_c01 = grad_cov_2d[idx * 4 + 1];
    const float grad_c10 = grad_cov_2d[idx * 4 + 2];
    const float grad_c11 = grad_cov_2d[idx * 4 + 3];
    
    // Precompute common terms
    const float sx2 = sx * sx;
    const float sy2 = sy * sy;
    const float cos2 = cos_theta * cos_theta;
    const float sin2 = sin_theta * sin_theta;
    const float cos_sin = cos_theta * sin_theta;
    
    // Compute gradients using chain rule
    // Covariance matrix: C = R * S * S^T * R^T where R is rotation, S is scale
    
    // Gradients w.r.t. sx
    float grad_sx = 0.0f;
    grad_sx += grad_c00 * (2.0f * sx * cos2);
    grad_sx += grad_c01 * (2.0f * sx * cos_sin);
    grad_sx += grad_c10 * (2.0f * sx * cos_sin);
    grad_sx += grad_c11 * (2.0f * sx * sin2);
    
    // Gradients w.r.t. sy
    float grad_sy = 0.0f;
    grad_sy += grad_c00 * (2.0f * sy * sin2);
    grad_sy += grad_c01 * (-2.0f * sy * cos_sin);
    grad_sy += grad_c10 * (-2.0f * sy * cos_sin);
    grad_sy += grad_c11 * (2.0f * sy * cos2);
    
    // Gradients w.r.t. theta
    float grad_theta = 0.0f;
    grad_theta += grad_c00 * (2.0f * (sy2 - sx2) * cos_sin);
    grad_theta += grad_c01 * ((sx2 - sy2) * (cos2 - sin2));
    grad_theta += grad_c10 * ((sx2 - sy2) * (cos2 - sin2));
    grad_theta += grad_c11 * (2.0f * (sx2 - sy2) * cos_sin);
    
    // Write outputs
    grad_scales[idx * 2] = grad_sx;
    grad_scales[idx * 2 + 1] = grad_sy;
    grad_rotations[idx] = grad_theta;
}

std::vector<torch::Tensor> invert_2d_covariance_backward(
    const torch::Tensor& grad_inv_cov,
    const torch::Tensor& cov_2d)
{
    const int N = cov_2d.size(0);
    auto grad_cov_2d = torch::zeros_like(cov_2d);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    invert_2d_covariance_backward_kernel<<<blocks, threads>>>(
        grad_inv_cov.data_ptr<float>(),
        cov_2d.data_ptr<float>(),
        grad_cov_2d.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in invert_2d_covariance_backward: %s\n", cudaGetErrorString(err));
    }
    
    return {grad_cov_2d};
}

std::vector<torch::Tensor> compute_2d_covariance_backward(
    const torch::Tensor& grad_cov_2d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations)
{
    const int N = scales.size(0);
    auto grad_scales = torch::zeros_like(scales);
    auto grad_rotations = torch::zeros_like(rotations);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    compute_2d_covariance_backward_kernel<<<blocks, threads>>>(
        grad_cov_2d.data_ptr<float>(),
        scales.data_ptr<float>(),
        rotations.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in compute_2d_covariance_backward: %s\n", cudaGetErrorString(err));
    }

    return {grad_scales, grad_rotations};
}

} // namespace gaussian_2d_cuda