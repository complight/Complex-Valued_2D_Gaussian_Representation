#include "gaussian_2d.h"
#include "config_2d.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <iostream>

namespace gaussian_2d_cuda {

// Helper function for memory management
template<typename T>
size_t required(size_t P) {
    char* size_ptr = nullptr;
    T::fromChunk(size_ptr, P);
    return ((size_t)size_ptr) + 128;
}

// Overload for ImageState which needs different parameters
template<>
size_t required<ImageState>(size_t P) {
    // This is a dummy implementation since ImageState needs more parameters
    return P * (sizeof(uint2) + sizeof(float) + sizeof(uint32_t)) + 128;
}

// Memory manager class
class CudaMemoryManager {
private:
    char* geom_buffer = nullptr;
    char* binning_buffer = nullptr;
    char* img_buffer = nullptr;
    size_t geom_buffer_size = 0;
    size_t binning_buffer_size = 0;
    size_t img_buffer_size = 0;

public:
    static CudaMemoryManager& getInstance() {
        static CudaMemoryManager instance;
        return instance;
    }

    char* getGeometryBuffer(size_t size);
    char* getBinningBuffer(size_t size);
    char* getImageBuffer(size_t size);
    void freeAll();
};

char* CudaMemoryManager::getGeometryBuffer(size_t size) {
    if (geom_buffer != nullptr && geom_buffer_size < size) {
        cudaFree(geom_buffer);
        geom_buffer = nullptr;
    }
    
    if (geom_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&geom_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate geometry buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        geom_buffer_size = size;
    }
    
    return geom_buffer;
}

char* CudaMemoryManager::getBinningBuffer(size_t size) {
    if (binning_buffer != nullptr && binning_buffer_size < size) {
        cudaFree(binning_buffer);
        binning_buffer = nullptr;
    }
    
    if (binning_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&binning_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate binning buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        binning_buffer_size = size;
    }
    
    return binning_buffer;
}

char* CudaMemoryManager::getImageBuffer(size_t size) {
    if (img_buffer != nullptr && img_buffer_size < size) {
        cudaFree(img_buffer);
        img_buffer = nullptr;
    }
    
    if (img_buffer == nullptr) {
        cudaError_t err = cudaMalloc(&img_buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate image buffer: " 
                      << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        img_buffer_size = size;
    }
    
    return img_buffer;
}

void CudaMemoryManager::freeAll() {
    if (geom_buffer != nullptr) {
        cudaFree(geom_buffer);
        geom_buffer = nullptr;
        geom_buffer_size = 0;
    }
    
    if (binning_buffer != nullptr) {
        cudaFree(binning_buffer);
        binning_buffer = nullptr;
        binning_buffer_size = 0;
    }
    
    if (img_buffer != nullptr) {
        cudaFree(img_buffer);
        img_buffer = nullptr;
        img_buffer_size = 0;
    }
}

// Memory state structure implementations
GeometryState GeometryState::fromChunk(char*& chunk, size_t N) {
    GeometryState state;
    auto obtain = [&](auto& ptr, size_t count, size_t alignment) {
        size_t offset = ((size_t)chunk + alignment - 1) & ~(alignment - 1);
        ptr = (typename std::remove_reference<decltype(ptr)>::type)(offset);
        chunk = (char*)(offset + count * sizeof(*ptr));
    };
    
    obtain(state.means_2d, N, 128);
    obtain(state.inv_cov_2d, N * 3, 128);
    obtain(state.radii, N, 128);
    obtain(state.tiles_touched, N, 128);
    obtain(state.point_offsets, N, 128);
    
    cub::DeviceScan::InclusiveSum(nullptr, state.scan_size, state.tiles_touched, state.tiles_touched, N);
    obtain(state.scanning_space, state.scan_size, 128);
    
    return state;
}

BinningState BinningState::fromChunk(char*& chunk, size_t total_pairs) {
    BinningState state;
    auto obtain = [&](auto& ptr, size_t count, size_t alignment) {
        size_t offset = ((size_t)chunk + alignment - 1) & ~(alignment - 1);
        ptr = (typename std::remove_reference<decltype(ptr)>::type)(offset);
        chunk = (char*)(offset + count * sizeof(*ptr));
    };
    
    obtain(state.keys_unsorted, total_pairs, 128);
    obtain(state.values_unsorted, total_pairs, 128);
    obtain(state.keys_sorted, total_pairs, 128);
    obtain(state.values_sorted, total_pairs, 128);
    
    cub::DeviceRadixSort::SortPairs(
        nullptr, state.sorting_size,
        state.keys_unsorted, state.keys_sorted,
        state.values_unsorted, state.values_sorted, total_pairs);
    obtain(state.sorting_space, state.sorting_size, 128);
    
    return state;
}

ImageState ImageState::fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels) {
    ImageState state;
    auto obtain = [&](auto& ptr, size_t count, size_t alignment) {
        size_t offset = ((size_t)chunk + alignment - 1) & ~(alignment - 1);
        ptr = (typename std::remove_reference<decltype(ptr)>::type)(offset);
        chunk = (char*)(offset + count * sizeof(*ptr));
    };
    
    obtain(state.ranges, num_tiles, 128);
    obtain(state.final_Ts, num_pixels, 128);
    obtain(state.n_contrib, num_pixels, 128);
    
    return state;
}

// Fill kernel for initializing transmittance
__global__ void fillOnesKernel(float* data, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

// Main interface functions
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
render_2d_gaussians_cuda(
    const torch::Tensor& means_2d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& colours,
    const torch::Tensor& phase,
    const torch::Tensor& opacities,
    int width, int height,
    int num_channels)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int device_id = means_2d.device().index();
    at::cuda::CUDAGuard device_guard(device_id);
    
    const int N = means_2d.size(0);
    
    // Configure tile grid
    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    const int num_tiles = tile_grid.x * tile_grid.y;
    
    // Create output tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(means_2d.device());
    
    auto output_real = torch::zeros({num_channels, height, width}, options);
    auto output_imag = torch::zeros({num_channels, height, width}, options);
    
    // Step 1: Compute covariance matrices
    auto cov_2d = compute_2d_covariance_forward(scales, rotations);
    
    // Step 2: Invert covariance matrices and get radii
    auto invert_result = invert_2d_covariance_forward(cov_2d);
    auto inv_cov_2d = std::get<0>(invert_result);
    auto radii = std::get<1>(invert_result);
    
    // Allocate memory using memory manager
    size_t geom_chunk_size = required<GeometryState>(N);
    char* geom_chunk = CudaMemoryManager::getInstance().getGeometryBuffer(geom_chunk_size);
    GeometryState geom = GeometryState::fromChunk(geom_chunk, N);
    
    // Copy data to device buffers
    cudaMemcpyAsync(geom.means_2d, means_2d.data_ptr<float>(), N * sizeof(float2), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(geom.inv_cov_2d, inv_cov_2d.data_ptr<float>(), N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(geom.radii, radii.data_ptr<int>(), N * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    
    // Preprocess Gaussians
    FORWARD_2D::preprocess(N, geom.means_2d, geom.radii, geom.tiles_touched, tile_grid);
    
    // Compute prefix sum
    cub::DeviceScan::InclusiveSum(geom.scanning_space, geom.scan_size, geom.tiles_touched, geom.point_offsets, N, stream);
    
    // Get total number of key-value pairs
    int total_pairs = 0;
    cudaMemcpyAsync(&total_pairs, geom.point_offsets + (N - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (total_pairs == 0) {
        auto output = torch::complex(output_real, output_imag);
        auto final_Ts = torch::ones({height, width}, options);
        auto n_contrib = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device()));
        auto ranges = torch::zeros({num_tiles, 2}, torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device()));
        auto point_list = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device()));
        
        return std::make_tuple(
            output, 
            std::make_tuple(final_Ts, n_contrib, point_list, ranges)
        );
    }
    
    // Allocate binning state
    size_t binning_chunk_size = required<BinningState>(total_pairs);
    char* binning_chunk = CudaMemoryManager::getInstance().getBinningBuffer(binning_chunk_size);
    BinningState binning = BinningState::fromChunk(binning_chunk, total_pairs);
    
    // Create key-value pairs
    FORWARD_2D::duplicateWithKeys(N, geom.means_2d, geom.radii, geom.point_offsets, 
                                  binning.keys_unsorted, binning.values_unsorted, tile_grid);
    
    // Sort key-value pairs
    cub::DeviceRadixSort::SortPairs(
        binning.sorting_space, binning.sorting_size,
        binning.keys_unsorted, binning.keys_sorted,
        binning.values_unsorted, binning.values_sorted,
        total_pairs, 0, 64, stream);
    
    // Allocate image state
    size_t img_chunk_size = num_tiles * sizeof(uint2) + height * width * (sizeof(float) + sizeof(uint32_t)) + 128;
    char* img_chunk = CudaMemoryManager::getInstance().getImageBuffer(img_chunk_size);
    ImageState img = ImageState::fromChunk(img_chunk, num_tiles, height * width);
    
    // Reset range values
    cudaMemsetAsync(img.ranges, 0, num_tiles * sizeof(uint2), stream);
    
    // Identify ranges for each tile
    FORWARD_2D::identifyTileRanges(total_pairs, binning.keys_sorted, img.ranges);
    
    // Initialize transmittance and contribution count
    cudaMemsetAsync(img.n_contrib, 0, height * width * sizeof(uint32_t), stream);
    
    const int fill_block_size = 256;
    const int fill_grid_size = (height * width + fill_block_size - 1) / fill_block_size;
    fillOnesKernel<<<fill_grid_size, fill_block_size, 0, stream>>>(img.final_Ts, height * width);
    
    // Render tiles
    const dim3 render_block(BLOCK_X, BLOCK_Y);
    
    FORWARD_2D::render(
        tile_grid, render_block,
        img.ranges, binning.values_sorted,
        geom.means_2d, geom.inv_cov_2d,
        colours.data_ptr<float>(),
        phase.data_ptr<float>(),
        opacities.data_ptr<float>(),
        output_real.data_ptr<float>(),
        output_imag.data_ptr<float>(),
        img.final_Ts, img.n_contrib,
        width, height, N, num_channels
    );
    
    // Create output
    auto output = torch::complex(output_real, output_imag);
    
    // Create tensors for backward pass
    auto final_Ts_tensor = torch::from_blob(img.final_Ts, {height, width}, options).clone();
    auto n_contrib_tensor = torch::from_blob(img.n_contrib, {height, width}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device())).clone();
    auto point_list_tensor = torch::from_blob(binning.values_sorted, {total_pairs}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device())).clone();
    auto ranges_tensor = torch::from_blob(img.ranges, {num_tiles, 2}, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device())).clone();
    
    return std::make_tuple(
        output, 
        std::make_tuple(final_Ts_tensor, n_contrib_tensor, point_list_tensor, ranges_tensor)
    );
}

std::vector<torch::Tensor> render_2d_gaussians_cuda_backward(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    const torch::Tensor& means_2d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& colours,
    const torch::Tensor& phase,
    const torch::Tensor& opacities,
    const torch::Tensor& final_Ts,
    const torch::Tensor& n_contrib,
    const torch::Tensor& point_list,
    const torch::Tensor& ranges,
    int width, int height,
    int num_channels)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int device_id = means_2d.device().index();
    at::cuda::CUDAGuard device_guard(device_id);
    
    const int N = means_2d.size(0);
    
    // Create gradient tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(means_2d.device());
    
    auto grad_means_2d = torch::zeros({N, 2}, options);
    auto grad_scales = torch::zeros({N, 2}, options);
    auto grad_rotations = torch::zeros({N}, options);
    auto grad_colours = torch::zeros({N, num_channels}, options);
    auto grad_phase = torch::zeros({N, num_channels}, options);
    auto grad_opacities = torch::zeros({N}, options);
    auto grad_inv_cov_2d = torch::zeros({N, 3}, options);
    
    // Recompute forward pass intermediate results
    auto cov_2d = compute_2d_covariance_forward(scales, rotations);
    auto invert_result = invert_2d_covariance_forward(cov_2d);
    auto inv_cov_2d = std::get<0>(invert_result);
    
    // Configure kernel dimensions
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch backward render kernel
    BACKWARD_2D::render(
        grid, block,
        (uint2*)ranges.data_ptr<int>(),
        (uint32_t*)point_list.data_ptr<int>(),
        (float2*)means_2d.data_ptr<float>(),
        inv_cov_2d.data_ptr<float>(),
        colours.data_ptr<float>(),
        phase.data_ptr<float>(),
        opacities.data_ptr<float>(),
        final_Ts.data_ptr<float>(),
        (uint32_t*)n_contrib.data_ptr<int>(),
        grad_output_real.data_ptr<float>(),
        grad_output_imag.data_ptr<float>(),
        grad_means_2d.data_ptr<float>(),
        grad_inv_cov_2d.data_ptr<float>(),
        grad_colours.data_ptr<float>(),
        grad_phase.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        N, num_channels, width, height
    );
    
    // Transform gradients through covariance computation chain
    cudaStreamSynchronize(stream);
    
    auto grad_cov_2d_grads = invert_2d_covariance_backward(grad_inv_cov_2d, cov_2d);
    auto cov_grads = compute_2d_covariance_backward(grad_cov_2d_grads[0], scales, rotations);
    
    grad_scales.add_(cov_grads[0]);
    grad_rotations.add_(cov_grads[1]);
    
    return {grad_means_2d, grad_scales, grad_rotations, grad_colours, 
            grad_phase, grad_opacities};
}

} // namespace gaussian_2d_cuda