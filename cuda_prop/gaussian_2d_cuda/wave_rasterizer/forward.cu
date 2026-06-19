#include "gaussian_2d.h"
#include "config_2d.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cmath>

namespace cg = cooperative_groups;

namespace gaussian_2d_cuda {

// Helper function to compute tile bounds for a Gaussian
__device__ void getRect(const float2 pos, float radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
    const int tile_x = (int)((pos.x - radius) / BLOCK_X);
    const int tile_y = (int)((pos.y - radius) / BLOCK_Y);
    const int tile_x_max = (int)((pos.x + radius) / BLOCK_X);
    const int tile_y_max = (int)((pos.y + radius) / BLOCK_Y);
    
    rect_min.x = max(0, tile_x);
    rect_min.y = max(0, tile_y);
    rect_max.x = min((int)grid.x, tile_x_max + 1);
    rect_max.y = min((int)grid.y, tile_y_max + 1);
}

// CUDA kernel for computing 2D covariance matrices
template <typename scalar_t>
__global__ void compute_2d_covariance_kernel(
    const scalar_t* __restrict__ scales,      // (N, 2)
    const scalar_t* __restrict__ rotations,   // (N,)
    scalar_t* __restrict__ cov_2d,            // (N, 2, 2)
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const scalar_t sx = scales[idx * 2 + 0];
    const scalar_t sy = scales[idx * 2 + 1];
    const scalar_t rotation = rotations[idx];

    const scalar_t cos_r = cosf(rotation);
    const scalar_t sin_r = sinf(rotation);

    const scalar_t sx2 = sx * sx;
    const scalar_t sy2 = sy * sy;
    const scalar_t cos_r2 = cos_r * cos_r;
    const scalar_t sin_r2 = sin_r * sin_r;
    const scalar_t cos_sin = cos_r * sin_r;

    scalar_t* cov = &cov_2d[idx * 4];
    cov[0] = sx2 * cos_r2 + sy2 * sin_r2 + 0.1f;      // cov_00
    cov[1] = (sx2 - sy2) * cos_sin;                    // cov_01
    cov[2] = (sx2 - sy2) * cos_sin;                    // cov_10 (symmetric)
    cov[3] = sx2 * sin_r2 + sy2 * cos_r2 + 0.1f;      // cov_11
}

// CUDA kernel for inverting 2D covariance matrices and computing radius
template <typename scalar_t>
__global__ void invert_2d_covariance_kernel(
    const scalar_t* __restrict__ cov_2d,     // (N, 2, 2)
    scalar_t* __restrict__ inv_cov_2d,       // (N, 3) - [inv_00, inv_01, inv_11]
    int* __restrict__ radii,                 // (N,) - computed radii
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    radii[idx] = 0;

    const scalar_t* cov = &cov_2d[idx * 4];
    const scalar_t cov_00 = cov[0];
    const scalar_t cov_01 = cov[1];
    const scalar_t cov_11 = cov[3];

    const scalar_t det = cov_00 * cov_11 - cov_01 * cov_01;
    if (fabsf(det) < 1e-10f) return;
    
    const scalar_t inv_det = 1.0f / det;

    scalar_t* inv_cov = &inv_cov_2d[idx * 3];
    inv_cov[0] = cov_11 * inv_det;      // inv_00
    inv_cov[1] = -cov_01 * inv_det;     // inv_01
    inv_cov[2] = cov_00 * inv_det;      // inv_11

    // Compute radius using eigenvalues
    const scalar_t mid = 0.5f * (cov_00 + cov_11);
    const scalar_t disc = fmaxf(0.1f, mid * mid - det);
    const scalar_t lambda1 = mid + sqrtf(disc);
    const scalar_t lambda2 = mid - sqrtf(disc);
    
    const float radius = ceilf(3.0f * sqrtf(fmaxf(lambda1, lambda2)));
    radii[idx] = (int)radius;
}

// Preprocessing kernel to compute tile coverage
__global__ void preprocessGaussiansKernel(
    int N,
    const float2* __restrict__ means_2d,
    const int* __restrict__ radii,
    uint32_t* __restrict__ tiles_touched,
    dim3 grid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    tiles_touched[idx] = 0;
    
    const int radius = radii[idx];
    if (radius == 0) return;
    
    const float2 pos = means_2d[idx];
    
    uint2 rect_min, rect_max;
    getRect(pos, radius, rect_min, rect_max, grid);
    
    const uint32_t count = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    tiles_touched[idx] = count;
}

// Generate key-value pairs for tile-Gaussian associations
__global__ void duplicateWithKeysKernel(
    int N,
    const float2* __restrict__ means_2d,
    const int* __restrict__ radii,
    const uint32_t* __restrict__ point_offsets,
    uint64_t* __restrict__ keys_unsorted,
    uint32_t* __restrict__ values_unsorted,
    dim3 grid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    const int radius = radii[idx];
    if (radius == 0) return;
    
    const uint32_t offset = (idx == 0) ? 0 : point_offsets[idx - 1];
    
    uint2 rect_min, rect_max;
    getRect(means_2d[idx], radius, rect_min, rect_max, grid);
    
    // Use Gaussian index as depth for ordering (front-to-back)
    uint32_t depth_bits = idx;
    
    uint32_t current_offset = offset;
    for (uint32_t y = rect_min.y; y < rect_max.y; y++) {
        for (uint32_t x = rect_min.x; x < rect_max.x; x++) {
            uint64_t key = (uint64_t)(y * grid.x + x) << 32;
            key |= depth_bits;
            
            keys_unsorted[current_offset] = key;
            values_unsorted[current_offset] = idx;
            current_offset++;
        }
    }
}

// Identify tile ranges kernel
__global__ void identifyTileRangesKernel(
    int num_pairs,
    const uint64_t* __restrict__ keys_sorted,
    uint2* __restrict__ ranges)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    const uint32_t tile_id = keys_sorted[idx] >> 32;
    
    if (idx == 0) {
        ranges[tile_id].x = idx;
    } else if (tile_id != (keys_sorted[idx-1] >> 32)) {
        const uint32_t prev_tile = keys_sorted[idx-1] >> 32;
        ranges[prev_tile].y = idx;
        ranges[tile_id].x = idx;
    }
    
    if (idx == num_pairs - 1) {
        ranges[tile_id].y = idx + 1;
    }
}

// Main tile-based rendering kernel (simplified)
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderTileKernel(
    const uint2* __restrict__ tile_ranges,
    const uint32_t* __restrict__ gaussian_indices_sorted,
    const float2* __restrict__ means_2d,
    const float* __restrict__ inv_cov_2d,   // (N, 3) format
    const float* __restrict__ colours,
    const float* __restrict__ phase,
    const float* __restrict__ opacities,
    float* __restrict__ output_real,
    float* __restrict__ output_imag,
    float* __restrict__ final_Ts,
    uint32_t* __restrict__ n_contrib,
    int W, int H, int N, int channels)
{
    const uint32_t tile_x = blockIdx.x;
    const uint32_t tile_y = blockIdx.y;
    const uint32_t horizontal_tiles = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t tile_id = tile_y * horizontal_tiles + tile_x;
    
    const uint32_t pix_x = tile_x * BLOCK_X + threadIdx.x;
    const uint32_t pix_y = tile_y * BLOCK_Y + threadIdx.y;
    const bool inside = pix_x < W && pix_y < H;
    
    const uint2 range = tile_ranges[tile_id];
    const int toDo = range.y - range.x;
    
    if (toDo <= 0) return;
    
    const float2 pixf = { (float)pix_x, (float)pix_y };
    
    if (inside) {
        const uint32_t pix_id = pix_y * W + pix_x;
        
        float T = 1.0f;
        uint32_t position = 0;
        uint32_t last_position = 0;
        
        float accum_real[CHANNELS] = {0.0f};
        float accum_imag[CHANNELS] = {0.0f};
        
        const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
        bool done = false;
        
        // Shared memory for batch processing
        __shared__ uint32_t collected_ids[BLOCK_SIZE];
        __shared__ float2 collected_means[BLOCK_SIZE];
        __shared__ float collected_inv_cov[BLOCK_SIZE * 3];
        __shared__ float collected_opacities[BLOCK_SIZE];
        
        for (int i = 0; i < rounds; i++) {
            __syncthreads();
            
            int num_done = __syncthreads_count(done);
            if (num_done == BLOCK_X * BLOCK_Y) break;
            
            // Load batch data into shared memory
            const int progress = i * BLOCK_SIZE + threadIdx.x + threadIdx.y * blockDim.x;
            if (range.x + progress < range.y) {
                const uint32_t g_idx = gaussian_indices_sorted[range.x + progress];
                collected_ids[threadIdx.x + threadIdx.y * blockDim.x] = g_idx;
                collected_means[threadIdx.x + threadIdx.y * blockDim.x] = means_2d[g_idx];
                
                // Load inverse covariance (3 elements)
                const int shared_idx = threadIdx.x + threadIdx.y * blockDim.x;
                collected_inv_cov[shared_idx * 3 + 0] = inv_cov_2d[g_idx * 3 + 0];
                collected_inv_cov[shared_idx * 3 + 1] = inv_cov_2d[g_idx * 3 + 1];
                collected_inv_cov[shared_idx * 3 + 2] = inv_cov_2d[g_idx * 3 + 2];
                
                collected_opacities[shared_idx] = opacities[g_idx];
            }
            __syncthreads();
            
            const int batch_size = min(BLOCK_SIZE, toDo - i * BLOCK_SIZE);
            for (int j = 0; j < batch_size && !done; j++) {
                position++;
                
                const uint32_t g_idx = collected_ids[j];
                const float2 xy = collected_means[j];
                const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                
                // Get inverse covariance elements from shared memory
                const float inv_00 = collected_inv_cov[j * 3 + 0];
                const float inv_01 = collected_inv_cov[j * 3 + 1];
                const float inv_11 = collected_inv_cov[j * 3 + 2];
                
                const float power = -0.5f * (d.x * (inv_00 * d.x + inv_01 * d.y) +
                                            d.y * (inv_01 * d.x + inv_11 * d.y));
                if (power > 0.0f) continue;

                const float g_opacity = collected_opacities[j];
                const float gauss_exp = expf(power);
                
                const float alpha = fminf(0.99f, g_opacity * gauss_exp);
                
                float test_T = T * (1.0f - alpha);
                if (test_T < 0.0001f) done = true;
                
                // Process all channels
                #pragma unroll
                for (int c = 0; c < CHANNELS; c++) {
                    const float color = colours[g_idx * channels + c];
                    const float ph = phase[g_idx * channels + c];
                    
                    const float scale = color * alpha * T;
                    
                    float sin_val, cos_val;
                    __sincosf(ph, &sin_val, &cos_val);
                    
                    accum_real[c] += scale * cos_val;
                    accum_imag[c] += scale * sin_val;
                }
                
                T = test_T;
                last_position = position;
                
                if (done) break;
            }
        }
        
        // Write results
        #pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
            const int idx = (c * H * W) + pix_id;
            output_real[idx] = accum_real[c];
            output_imag[idx] = accum_imag[c];
        }
        
        final_Ts[pix_id] = T;
        n_contrib[pix_id] = last_position;
    }
}

// Forward pass implementations
torch::Tensor compute_2d_covariance_forward(
    const torch::Tensor& scales,
    const torch::Tensor& rotations)
{
    const int N = scales.size(0);
    auto cov_2d = torch::empty({N, 2, 2}, scales.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(scales.scalar_type(), "compute_2d_covariance_forward", ([&] {
        compute_2d_covariance_kernel<scalar_t><<<blocks, threads>>>(
            scales.data_ptr<scalar_t>(),
            rotations.data_ptr<scalar_t>(),
            cov_2d.data_ptr<scalar_t>(),
            N
        );
    }));

    return cov_2d;
}

std::tuple<torch::Tensor, torch::Tensor> invert_2d_covariance_forward(
    const torch::Tensor& cov_2d)
{
    const int N = cov_2d.size(0);
    auto inv_cov_2d = torch::empty({N, 3}, cov_2d.options());
    auto radii = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(cov_2d.device()));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(cov_2d.scalar_type(), "invert_2d_covariance_forward", ([&] {
        invert_2d_covariance_kernel<scalar_t><<<blocks, threads>>>(
            cov_2d.data_ptr<scalar_t>(),
            inv_cov_2d.data_ptr<scalar_t>(),
            radii.data_ptr<int>(),
            N
        );
    }));

    return std::make_tuple(inv_cov_2d, radii);
}

void FORWARD_2D::preprocess(
    int N,
    const float2* means_2d,
    const int* radii,
    uint32_t* tiles_touched,
    const dim3 grid)
{
    const int block_size = 256;
    const dim3 blocks((N + block_size - 1) / block_size);
    
    preprocessGaussiansKernel<<<blocks, block_size>>>(
        N, means_2d, radii, tiles_touched, grid
    );
}

void FORWARD_2D::duplicateWithKeys(
    int N,
    const float2* means_2d,
    const int* radii,
    const uint32_t* point_offsets,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    const dim3 grid)
{
    const int block_size = 256;
    const dim3 blocks((N + block_size - 1) / block_size);
    
    duplicateWithKeysKernel<<<blocks, block_size>>>(
        N, means_2d, radii, point_offsets, keys_unsorted, values_unsorted, grid
    );
}

void FORWARD_2D::identifyTileRanges(
    int num_pairs,
    const uint64_t* keys_sorted,
    uint2* ranges)
{
    identifyTileRangesKernel<<<(num_pairs + 255) / 256, 256>>>(
        num_pairs, keys_sorted, ranges
    );
}

void FORWARD_2D::render(
    const dim3 grid, dim3 block,
    const uint2* tile_ranges,
    const uint32_t* point_list,
    const float2* means_2d,
    const float* inv_cov_2d,
    const float* colours,
    const float* phase,
    const float* opacities,
    float* output_real,
    float* output_imag,
    float* final_Ts,
    uint32_t* n_contrib,
    int W, int H, int N,
    int channels)
{
    switch (channels) {
        case 1:
            renderTileKernel<1><<<grid, block>>>(
                tile_ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                output_real, output_imag, final_Ts, n_contrib, W, H, N, channels);
            break;
        case 3:
            renderTileKernel<3><<<grid, block>>>(
                tile_ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                output_real, output_imag, final_Ts, n_contrib, W, H, N, channels);
            break;
        default:
            if (channels <= MAX_CHANNELS) {
                renderTileKernel<MAX_CHANNELS><<<grid, block>>>(
                    tile_ranges, point_list, means_2d, inv_cov_2d, colours, phase, opacities,
                    output_real, output_imag, final_Ts, n_contrib, W, H, N, channels);
            }
            break;
    }
}

} // namespace gaussian_2d_cuda