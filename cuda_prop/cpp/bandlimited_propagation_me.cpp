#include <torch/extension.h>
#include <vector>

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
    int ny);

void bandlimited_propagation_me_apply_aperture_cuda(
    torch::Tensor& field_real,
    torch::Tensor& field_imag,
    float aperture_size,
    int nx,
    int ny);

void bandlimited_propagation_me_backward_cuda(
    const torch::Tensor& grad_real,
    const torch::Tensor& grad_imag,
    torch::Tensor& output_real,
    torch::Tensor& output_imag,
    float wavelength,
    float distance,
    float pixel_pitch,
    int nx,
    int ny);

std::vector<torch::Tensor> bandlimited_propagation_me_forward(
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

    bandlimited_propagation_me_forward_cuda(
        field_f_real, field_f_imag,
        output_real, output_imag,
        wavelength, distance, aperture_size, pixel_pitch, nx, ny
    );

    return {output_real, output_imag};
}

void bandlimited_propagation_me_apply_aperture(
    torch::Tensor& field_real,
    torch::Tensor& field_imag,
    float aperture_size)
{
    int nx = field_real.size(0);
    int ny = field_real.size(1);

    bandlimited_propagation_me_apply_aperture_cuda(
        field_real, field_imag,
        aperture_size, nx, ny
    );
}

std::vector<torch::Tensor> bandlimited_propagation_me_backward(
    const torch::Tensor& grad_real,
    const torch::Tensor& grad_imag,
    float wavelength,
    float distance,
    float aperture_size,
    float pixel_pitch)
{
    auto output_real = torch::zeros_like(grad_real);
    auto output_imag = torch::zeros_like(grad_imag);
    
    int nx = grad_real.size(0);
    int ny = grad_real.size(1);

    bandlimited_propagation_me_backward_cuda(
        grad_real, grad_imag,
        output_real, output_imag,
        wavelength, distance, pixel_pitch, nx, ny
    );

    return {output_real, output_imag};
}