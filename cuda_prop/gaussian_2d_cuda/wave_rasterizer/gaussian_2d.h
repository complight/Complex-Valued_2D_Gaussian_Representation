#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>

namespace gaussian_2d_cuda {

// Forward declarations for structures
struct GeometryState {
    float2* means_2d;
    float* inv_cov_2d;
    int* radii;
    uint32_t* tiles_touched;
    uint32_t* point_offsets;
    char* scanning_space;
    size_t scan_size;
    
    static GeometryState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
    uint64_t* keys_unsorted;
    uint32_t* values_unsorted;
    uint64_t* keys_sorted;
    uint32_t* values_sorted;
    char* sorting_space;
    size_t sorting_size;
    
    static BinningState fromChunk(char*& chunk, size_t total_pairs);
};

struct ImageState {
    uint2* ranges;
    float* final_Ts;
    uint32_t* n_contrib;
    
    static ImageState fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels);
};

// Forward pass class
class FORWARD_2D {
public:
    static void preprocess(
        int N,
        const float2* means_2d,
        const int* radii,
        uint32_t* tiles_touched,
        const dim3 grid);
    
    static void duplicateWithKeys(
        int N,
        const float2* means_2d,
        const int* radii,
        const uint32_t* point_offsets,
        uint64_t* keys_unsorted,
        uint32_t* values_unsorted,
        const dim3 grid);
    
    static void identifyTileRanges(
        int num_pairs,
        const uint64_t* keys_sorted,
        uint2* ranges);
    
    static void render(
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
        int num_channels);
};

// Backward pass class
class BACKWARD_2D {
public:
    static void render(
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
        int W, int H);
};

// Function declarations
torch::Tensor compute_2d_covariance_forward(
    const torch::Tensor& scales,
    const torch::Tensor& rotations);

std::tuple<torch::Tensor, torch::Tensor> invert_2d_covariance_forward(
    const torch::Tensor& cov_2d);

std::vector<torch::Tensor> invert_2d_covariance_backward(
    const torch::Tensor& grad_inv_cov,
    const torch::Tensor& cov_2d);

std::vector<torch::Tensor> compute_2d_covariance_backward(
    const torch::Tensor& grad_cov_2d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations);

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
    int num_channels);

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
    int num_channels);

} // namespace gaussian_2d_cuda