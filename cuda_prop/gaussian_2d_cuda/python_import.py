import os
import sys
import subprocess
from torch.autograd import Function
import torch
import numpy as np
from typing import Tuple

# Try to import pre-compiled module, only build if it is unavailable
try:
    import gaussian_2d_cuda
    print("Successfully imported pre-compiled gaussian_2d_cuda module.")
except ImportError:
    print("Building gaussian_2d_cuda module...")

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Run build command
    build_command = [sys.executable, "setup.py", "install"]
    subprocess.check_call(build_command, cwd=current_dir)

    # Import the CUDA extension
    import gaussian_2d_cuda
    print("Successfully imported gaussian_2d_cuda module.")

class Render2DGaussiansFunction(Function):
    @staticmethod
    def forward(ctx, means_2d, scales, rotations, colours, phase, opacities,
                width, height, num_channels):
        """
        Forward pass for 2D Gaussian tile-based rendering (simplified version)
        
        Args:
            means_2d: (N, 2) - activated 2D positions
            scales: (N, 2) - activated scaling factors
            rotations: (N,) - rotation angles
            colours: (N, C) - RGB colors for each channel
            phase: (N, C) - phase values for each channel
            opacities: (N,) - activated opacity values
            width, height: image dimensions
            num_channels: number of color channels
        
        Returns:
            output_complex: (C, H, W) - complex field
            forward_info: tuple of (final_Ts, n_contrib, point_list, ranges)
        """
        
        # Input validation
        N = means_2d.size(0)
        
        assert means_2d.shape == (N, 2), f"means_2d shape mismatch: {means_2d.shape} vs ({N}, 2)"
        assert scales.shape == (N, 2), f"scales shape mismatch: {scales.shape} vs ({N}, 2)"
        assert rotations.shape == (N,), f"rotations shape mismatch: {rotations.shape} vs ({N},)"
        assert colours.shape == (N, num_channels), f"colours shape mismatch: {colours.shape} vs ({N}, {num_channels})"
        assert phase.shape == (N, num_channels), f"phase shape mismatch: {phase.shape} vs ({N}, {num_channels})"
        assert opacities.shape == (N,), f"opacities shape mismatch: {opacities.shape} vs ({N},)"
        
        # Call CUDA function with tile-based processing
        output_complex, forward_info = gaussian_2d_cuda.render_2d_gaussians_cuda(
            means_2d, scales, rotations, colours, phase, opacities,
            width, height, num_channels
        )
        
        # Save for backward pass
        ctx.save_for_backward(means_2d, scales, rotations, colours, phase, opacities)
        ctx.width = width
        ctx.height = height
        ctx.num_channels = num_channels
        ctx.forward_info = forward_info
        
        return output_complex

    @staticmethod
    def backward(ctx, grad_output_complex, grad_forward_info=None):
        """
        Backward pass for 2D Gaussian tile-based rendering (simplified version)
        
        Args:
            grad_output_complex: (C, H, W) - gradients w.r.t. complex output
            grad_forward_info: ignored
        
        Returns:
            Gradients for all input parameters
        """
        
        # Retrieve saved tensors
        means_2d, scales, rotations, colours, phase, opacities = ctx.saved_tensors
        final_Ts, n_contrib, point_list, ranges = ctx.forward_info
        
        width = ctx.width
        height = ctx.height
        num_channels = ctx.num_channels
        
        # Split complex gradients
        grad_output_real = torch.real(grad_output_complex).contiguous()
        grad_output_imag = torch.imag(grad_output_complex).contiguous()
        
        # Call CUDA backward function
        gradients = gaussian_2d_cuda.render_2d_gaussians_cuda_backward(
            grad_output_real, grad_output_imag, 
            means_2d, scales, rotations, colours, phase, opacities,
            final_Ts, n_contrib, point_list, ranges,
            width, height, num_channels
        )
        
        # Unpack gradients
        grad_means_2d = gradients[0]
        grad_scales = gradients[1]
        grad_rotations = gradients[2]
        grad_colours = gradients[3]
        grad_phase = gradients[4]
        grad_opacities = gradients[5]
        
        # Return gradients in the same order as forward inputs
        return (grad_means_2d, grad_scales, grad_rotations, grad_colours, 
                grad_phase, grad_opacities, None, None, None)


def cuda_render_scene_2d(scales, rotation, phase, opacities, means_2d, colours, 
                        img_size: Tuple[int, int], num_planes: int, num_channels: int):
    """
    Main interface function for CUDA-accelerated 2D Gaussian tile-based rendering (simplified version)
    
    Args:
        scales: (N, 2) activated scaling factors
        rotation: (N,) rotation angles
        phase: (N, C) phase values
        opacities: (N,) activated opacity values
        means_2d: (N, 2) activated 2D positions
        colours: (N, C) color values
        img_size: (width, height) tuple
        num_planes: ignored in simplified version
        num_channels: number of color channels
        
    Returns:
        output_complex: (C, H, W) complex tensor
    """
    
    width, height = img_size
    
    # Convert single rotation values to proper tensor shape
    if rotation.dim() == 1:
        rotations = rotation
    else:
        rotations = rotation.squeeze()
    
    # Call the CUDA function through autograd
    output_complex = Render2DGaussiansFunction.apply(
        means_2d, scales, rotations, colours, phase, opacities,
        width, height, num_channels
    )
    
    return output_complex