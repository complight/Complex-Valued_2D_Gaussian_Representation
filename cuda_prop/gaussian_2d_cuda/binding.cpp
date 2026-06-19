#include <torch/extension.h>
#include "wave_rasterizer/gaussian_2d.h"

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_2d_gaussians_cuda", &gaussian_2d_cuda::render_2d_gaussians_cuda, "Render 2D Gaussians CUDA forward");
    m.def("render_2d_gaussians_cuda_backward", &gaussian_2d_cuda::render_2d_gaussians_cuda_backward, "Render 2D Gaussians CUDA backward");
}