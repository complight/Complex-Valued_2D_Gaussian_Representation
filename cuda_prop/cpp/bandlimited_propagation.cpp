#include <torch/extension.h>
#include <vector>

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
    int ny);

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
    int ny);

std::vector<torch::Tensor> bandlimited_propagation_forward(
    const torch::Tensor& field_f_real,
    const torch::Tensor& field_f_imag,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch) 
{
    auto output_real = torch::zeros_like(field_f_real);
    auto output_imag = torch::zeros_like(field_f_imag);
    
    int nx = field_f_real.size(0);
    int ny = field_f_real.size(1);

    bandlimited_propagation_forward_cuda(
        field_f_real, field_f_imag,
        output_real, output_imag,
        wavelength, distance, aperture_size, pixel_pitch, nx, ny
    );

    return {output_real, output_imag};
}

std::vector<torch::Tensor> bandlimited_propagation_backward(
    const torch::Tensor& grad_output_real,
    const torch::Tensor& grad_output_imag,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch) 
{
    auto grad_field_real = torch::zeros_like(grad_output_real);
    auto grad_field_imag = torch::zeros_like(grad_output_imag);
    
    int nx = grad_output_real.size(0);
    int ny = grad_output_real.size(1);

    bandlimited_propagation_backward_cuda(
        grad_output_real, grad_output_imag,
        grad_field_real, grad_field_imag,
        wavelength, distance, aperture_size, pixel_pitch, nx, ny
    );

    return {grad_field_real, grad_field_imag};
}