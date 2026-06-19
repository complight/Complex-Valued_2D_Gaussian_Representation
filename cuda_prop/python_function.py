import math
import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple
import odak
import os
from odak.learn.tools import zero_pad, crop_center, circular_binary_mask

try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    compile_dir = os.path.join(current_dir, 'compile')
    if compile_dir not in sys.path:
        sys.path.append(compile_dir)
    
    from cuda_modules import *
    print("Successfully loaded pre-compiled CUDA module.")
    cuda_module = sys.modules['cuda_modules']
except ImportError:
    print("Pre-compiled CUDA module not found. Compiling now...")
    from .build_cuda_extension import build_cuda_module
    cuda_module = build_cuda_module()
    print("CUDA module compilation completed.")

def calculate_padding(original_size: int, target_size: int) -> Tuple[int, int]:
    pad_size = target_size - original_size
    pad_left = pad_size // 2
    pad_right = pad_size - pad_left
    return (pad_left, pad_right)

def bandlimited_angular_spectrum_propagation(field, wavelength, pixel_pitch, distance, size, aperture_size = -1):
    size = [i * 2 for i in size]
    field = odak.learn.tools.zero_pad(field, size)
    aperture = circular_binary_mask(
                                        size[0],
                                        size[1],
                                        aperture_size,
                                    ).to(field.device) * 1.
    field_f = torch.fft.fftshift(torch.fft.fft2(field))

    Nx, Ny = field.shape
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=pixel_pitch)).to(field.device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=pixel_pitch)).to(field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    
    x = torch.tensor(pixel_pitch * float(Nx), device=field.device)
    y = torch.tensor(pixel_pitch * float(Ny), device=field.device)
    distance = torch.tensor(distance, device=field.device)
    wavelength = torch.tensor(wavelength, device=field.device)
    
    fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
    fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
    bandlimit_mask = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max))
    
    k = 2 * torch.pi / wavelength
    kz = torch.sqrt(k**2 - (2*torch.pi*FX)**2 - (2*torch.pi*FY)**2)
    kz = torch.where(torch.isnan(kz), 0, kz)
    
    H = torch.exp(1j * distance * kz) * bandlimit_mask
    field_propagated_f = field_f * H  * aperture
    field_propagated = torch.fft.ifft2(torch.fft.ifftshift(field_propagated_f))
    field_propagated = crop_center(field_propagated)
    return field_propagated


class BatchMatrixMultiplicationFunction(Function):
    @staticmethod
    def forward(ctx, diff: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
        diff = diff.contiguous()
        cov_inv = cov_inv.contiguous()
        ctx.save_for_backward(diff, cov_inv)
        
        output = cuda_module.batch_matrix_multiplication_forward(
            diff, cov_inv)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff, cov_inv = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        grad_diff, grad_cov_inv = cuda_module.batch_matrix_multiplication_backward(
            grad_output, diff, cov_inv)
        
        return grad_diff, grad_cov_inv

class BandlimitedPropagationFunction(Function):
    @staticmethod
    def forward(ctx, field: torch.Tensor, wavelength: float, pixel_pitch: float, 
                distance: float, size: Tuple[int, int], aperture_size: float = -1.0) -> torch.Tensor:
        doubled_size = [int(i * 2) for i in size]

        ctx.wavelength = wavelength
        ctx.pixel_pitch = pixel_pitch
        ctx.distance = distance
        ctx.input_size = field.shape[-2:]
        ctx.aperture_size = aperture_size
        
        padded_field = odak.learn.tools.zero_pad(field, doubled_size)
        field_f = torch.fft.fftshift(torch.fft.fft2(padded_field))
        
        output_real, output_imag = cuda_module.bandlimited_propagation_forward(
            field_f.real.contiguous(), field_f.imag.contiguous(), 
            wavelength, distance, aperture_size, pixel_pitch)
        
        propagated_field = torch.fft.ifft2(torch.fft.ifftshift(
            torch.complex(output_real, output_imag)))
        propagated_field = odak.learn.tools.crop_center(propagated_field, size)
        
        return propagated_field

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, None]:
        doubled_size = [int(i * 2) for i in ctx.input_size]
        
        padded_grad = odak.learn.tools.zero_pad(grad_output, doubled_size)
        grad_f = torch.fft.fftshift(torch.fft.fft2(padded_grad))

        grad_field_real, grad_field_imag = cuda_module.bandlimited_propagation_backward(
            grad_f.real.contiguous(), grad_f.imag.contiguous(),
            ctx.wavelength, ctx.distance, ctx.aperture_size, ctx.pixel_pitch)
        
        grad_field = torch.fft.ifft2(torch.fft.ifftshift(
            torch.complex(grad_field_real, grad_field_imag)))
        grad_field = odak.learn.tools.crop_center(grad_field, ctx.input_size)
        
        return grad_field, None, None, None, None, None

class BandlimitedPropagationMEFunction(Function):
    @staticmethod
    def forward(ctx, field: torch.Tensor, wavelength: float, pixel_pitch: float, 
                distance: float, size: Tuple[int, int], aperture_size: float = -1.0) -> torch.Tensor:
        if not torch.is_complex(field):
            field = field.to(torch.complex64)
        
        Ny, Nx = size
        if list(field.shape[-2:]) != [Ny, Nx]:
            field = zero_pad(field, (Ny, Nx))
            field = crop_center(field, (Ny, Nx))

        ctx.wavelength = wavelength
        ctx.pixel_pitch = pixel_pitch
        ctx.distance = distance
        ctx.size = size
        ctx.aperture_size = aperture_size
        
        field_f = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        
        output_real, output_imag = cuda_module.bandlimited_propagation_me_forward(
            field_f.real.contiguous(), field_f.imag.contiguous(),
            wavelength, distance, aperture_size, pixel_pitch)
        
        propagated_field_f = torch.complex(output_real, output_imag)
        propagated_field = torch.fft.ifft2(torch.fft.ifftshift(propagated_field_f, dim=(-2, -1)))
        
        return propagated_field

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, None]:
        grad_f = torch.fft.fftshift(torch.fft.fft2(grad_output), dim=(-2, -1))
        
        grad_field_real, grad_field_imag = cuda_module.bandlimited_propagation_me_backward(
            grad_f.real.contiguous(), grad_f.imag.contiguous(),
            ctx.wavelength, ctx.distance, ctx.aperture_size, ctx.pixel_pitch)
        
        grad_field_f = torch.complex(grad_field_real, grad_field_imag)
        grad_field = torch.fft.ifft2(torch.fft.ifftshift(grad_field_f, dim=(-2, -1)))
        
        return grad_field, None, None, None, None, None

def bandlimited_angular_spectrum_propagation_me(field: torch.Tensor,
                                                wavelength: float,
                                                pixel_pitch: float,
                                                distance: float,
                                                size: Tuple[int, int],
                                                aperture_size: float = -1.0) -> torch.Tensor:
    """
    Memory-efficient BL-ASM:
      • avoids meshgrid (uses separable broadcasting),
      • avoids size-doubling pad by default (keeps accuracy via rectangular bandlimit),
      • minimizes intermediate copies and keeps complex dtype.
    Returns a field cropped back to `size`.
    """
    if not torch.is_complex(field):
        field = field.to(torch.complex64)
    device = field.device
    cdtype = field.dtype

    Ny, Nx = size
    if list(field.shape[-2:]) != [Ny, Nx]:
        field = zero_pad(field, (Ny, Nx))
        field = crop_center(field, (Ny, Nx))

    U = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))

    dx = float(pixel_pitch)
    dy = float(pixel_pitch)
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=dx), dim=0).to(device=device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=dy), dim=0).to(device=device)

    wl = torch.as_tensor(wavelength, device=device, dtype=torch.float64)
    z  = torch.as_tensor(distance,  device=device, dtype=torch.float64)
    k  = (2.0 * torch.pi / wl)

    Lx = dx * Nx
    Ly = dy * Ny
    fx_max = 1.0 / (torch.sqrt((2.0 * z * (1.0 / Lx))**2 + 1.0) * wl)
    fy_max = 1.0 / (torch.sqrt((2.0 * z * (1.0 / Ly))**2 + 1.0) * wl)
    mask_x = (torch.abs(fx) < fx_max)
    mask_y = (torch.abs(fy) < fy_max)
    band = mask_y[:, None] & mask_x[None, :]

    two_pi = 2.0 * torch.pi
    r2 = (fx[None, :]**2 + fy[:, None]**2).to(torch.float64)
    kz = torch.sqrt(torch.clamp(k**2 - (two_pi**2) * r2, min=0.0))
    H  = torch.exp(1j * (z * kz)).to(cdtype)
    H  = H * band

    U = U * H

    u = torch.fft.ifft2(torch.fft.ifftshift(U, dim=(-2, -1)))

    if aperture_size is not None and aperture_size > 0:
        ap = circular_binary_mask(Ny, Nx, aperture_size).to(device=device)
        u = u * ap

    return u


def BandlimitedPropagation(field: torch.Tensor, wavelength: float, pixel_pitch: float, 
                          distance: float, size: Tuple[int, int], use_cuda: bool, use_cuda_gaussian_version: bool, aperture_size: float = -1.0) -> torch.Tensor:
    #  both cases are cuda version of BLASM, but one of them is tailored just for gaussians hologram
    if use_cuda_gaussian_version:
        #  this BLASM is designed just for complex-valued 2D gaussian
        if use_cuda:
            return BandlimitedPropagationMEFunction.apply(field, wavelength, pixel_pitch, distance, size, aperture_size)
        else:
            # python version in parallel for gradient check
            return bandlimited_angular_spectrum_propagation_me(field, wavelength, pixel_pitch, distance, size, aperture_size)
    else:
        #  this is regular BLASM.
        if use_cuda:
            return BandlimitedPropagationFunction.apply(field, wavelength, pixel_pitch, distance, size, aperture_size)
        else:
            # python version in parallel for gradient check
            return bandlimited_angular_spectrum_propagation(field, wavelength, pixel_pitch, distance, size, aperture_size)
        
def compute_bmm_cuda(diff: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    return BatchMatrixMultiplicationFunction.apply(diff, cov_inv)

class SumLastDimFunction(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = input.shape
        ctx.device = input.device
        ctx.dtype = input.dtype
        
        output = cuda_module.sum_last_dim_forward(input)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        batch_size, hw_size, _ = ctx.input_shape
        
        dummy_input = torch.empty(ctx.input_shape, device=ctx.device, dtype=ctx.dtype)
        grad_input = cuda_module.sum_last_dim_backward(
            grad_output, dummy_input)
        
        return grad_input

def sum_last_dim_cuda(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or x.shape[-1] != 2:
        raise ValueError(f"Input must be (N, HW, 2), but got {x.shape}")

    return SumLastDimFunction.apply(x.contiguous())

class ElementWiseMultiplicationFunction(Function):
    @staticmethod
    def forward(ctx, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        
        ctx.save_for_backward(input1, input2)
        
        return cuda_module.element_wise_multiplication_forward(
            input1, input2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input1, input2 = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        grad_input1, grad_input2 = cuda_module.element_wise_multiplication_backward(
            grad_output, input1, input2)
        
        return grad_input1, grad_input2

def element_wise_multiplication_cuda(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.dim() == 3 and input1.shape[-1] == 2, "input1 must be (N, HW, 2)"
    assert input2.dim() == 3 and input2.shape[-1] == 2, "input2 must be (N, HW, 2)"
    assert input1.shape[0] == input2.shape[0] and input1.shape[1] == input2.shape[1], "input1 and input2 dimensions must match"
    
    return ElementWiseMultiplicationFunction.apply(input1, input2)

