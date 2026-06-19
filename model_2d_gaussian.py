import math
import sys
from argparse import Namespace
from typing import List, Optional, Tuple

import numpy as np
import odak
import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_prop import BandlimitedPropagation
from cuda_prop.gaussian_2d_cuda.python_import import cuda_render_scene_2d


class Gaussians2D(torch.nn.Module):
    def __init__(
        self,
        num_points: int,
        img_size: Tuple[int, int],
        device: str,
        args_prop: Namespace,
        merge_opacity: bool = False,
    ):
        super(Gaussians2D, self).__init__()

        self.device = device
        self.img_size = img_size
        self.merge_opacity = merge_opacity

        # Initialize 2D Gaussians - removed plane assignment
        data = self._initialize_2d(num_points, img_size)

        # Register parameters (6 instead of 7, no plane assignment)
        self.register_parameter(
            "means_2d", torch.nn.Parameter(data["means_2d"], requires_grad=False)
        )
        self.register_parameter(
            "pre_act_scales",
            torch.nn.Parameter(data["pre_act_scales"], requires_grad=False),
        )
        self.register_parameter(
            "pre_act_rotation",
            torch.nn.Parameter(data["pre_act_rotation"], requires_grad=False),
        )
        self.register_parameter(
            "colours", torch.nn.Parameter(data["colours"], requires_grad=False)
        )
        self.register_parameter(
            "pre_act_phase",
            torch.nn.Parameter(data["pre_act_phase"], requires_grad=False),
        )
        self.register_parameter(
            "pre_act_opacities",
            torch.nn.Parameter(data["pre_act_opacities"], requires_grad=False),
        )

        # Optimization caches
        self._activation_cache = None
        self._cache_dirty = True
        self._covariance_cache = None

        self.to(device)

    def _initialize_2d(self, num_points: int, img_size: Tuple[int, int]):
        W, H = img_size
        data = {}

        # Improved position initialization with tanh constraint (GaussianImage paper)
        data["means_2d"] = torch.rand((num_points, 2), dtype=torch.float32)
        data["means_2d"][:, 0] = data["means_2d"][:, 0] * W
        data["means_2d"][:, 1] = data["means_2d"][:, 1] * H

        # Convert to tanh space for better optimization stability
        normalized_means = data["means_2d"].clone()
        normalized_means[:, 0] = (
            normalized_means[:, 0] / W
        ) * 2 - 1  # Convert to [-1, 1]
        normalized_means[:, 1] = (normalized_means[:, 1] / H) * 2 - 1
        # Apply atanh but clamp to avoid infinities
        normalized_means = torch.clamp(normalized_means, -0.999, 0.999)
        data["means_2d"] = torch.atanh(normalized_means)

        # Initialize colors with slight bias towards middle values to avoid saturation
        data["colours"] = torch.rand((num_points, 3), dtype=torch.float32)

        # Initialize scales with minimum size to prevent dot artifacts
        min_init_scale = 1.5  # Minimum initial scale in pixels
        max_init_scale = 5.0  # Maximum initial scale in pixels
        scale_range = max_init_scale - min_init_scale
        data["pre_act_scales"] = torch.log(
            torch.rand((num_points, 2), dtype=torch.float32) * scale_range
            + min_init_scale
        )

        # Initialize rotation uniformly
        data["pre_act_rotation"] = (
            torch.rand((num_points,), dtype=torch.float32) * 2 * np.pi - np.pi
        )

        # Initialize phase with smaller variance for stability
        data["pre_act_phase"] = torch.zeros((num_points, 3), dtype=torch.float32)
        # Initialize opacities to be slightly transparent to avoid over-saturation
        # This helps with bright area artifacts
        data["pre_act_opacities"] = (
            torch.zeros((num_points,), dtype=torch.float32) - 0.5
        )  # Will sigmoid to ~0.38

        print(f"Initialized {num_points} 2D Gaussians for image size {img_size}")
        print(
            f"Initial scale range: [{min_init_scale:.2f}, {max_init_scale:.2f}] pixels"
        )
        return data

    def invalidate_cache(self):
        """Call this when parameters are updated during training"""
        self._cache_dirty = True
        self._covariance_cache = None

    def apply_activations(self):
        """Optimized activation computation with caching"""
        # Position activation with tanh constraint (GaussianImage paper approach)
        means_tanh = torch.tanh(self.means_2d)  # Constrain to (-1, 1)
        means_2d = torch.zeros_like(means_tanh)
        means_2d[:, 0] = (
            (means_tanh[:, 0] + 1) * 0.5 * self.img_size[0]
        )  # Scale to [0, W]
        means_2d[:, 1] = (
            (means_tanh[:, 1] + 1) * 0.5 * self.img_size[1]
        )  # Scale to [0, H]

        # Enhanced scale activation with stability constant
        scales = (
            torch.exp(self.pre_act_scales) + 0.1
        )  # Add small constant for stability

        rotation = self.pre_act_rotation
        phase = self.pre_act_phase % (2.0 * np.pi)
        if self.merge_opacity:
            opacities = torch.ones(
                self.pre_act_opacities.shape[0], device=self.pre_act_opacities.device
            )
        else:
            opacities = torch.sigmoid(self.pre_act_opacities)

        self._activation_cache = (scales, rotation, phase, opacities, means_2d)
        self._cache_dirty = False

        return scales, rotation, phase, opacities, means_2d

    @staticmethod
    def invert_cov_2D(cov_00, cov_01, cov_11):
        """Optimized 2x2 matrix inversion for covariance matrices"""
        det = cov_00 * cov_11 - cov_01 * cov_01
        det = det.clamp(min=1e-10)  # Numerical stability
        inv_det = 1.0 / det

        inv_00 = cov_11 * inv_det
        inv_01 = -cov_01 * inv_det
        inv_11 = cov_00 * inv_det

        return inv_00, inv_01, inv_11

    def compute_2d_covariance_elements(self, scales, rotation):
        """Optimized 2D covariance computation with Cholesky-like approach"""
        if (
            self._covariance_cache is not None
            and not self._cache_dirty
            and self._covariance_cache["scales"].shape == scales.shape
        ):
            return self._covariance_cache["cov_elements"]

        N = scales.shape[0]

        # Precompute trigonometric functions
        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)

        # Extract scales
        sx, sy = scales[:, 0], scales[:, 1]
        sx2, sy2 = sx * sx, sy * sy

        # Optimized covariance matrix computation
        # Following R * S^2 * R^T pattern but more efficient
        cos_r2 = cos_r * cos_r
        sin_r2 = sin_r * sin_r
        cos_sin = cos_r * sin_r

        cov_00 = sx2 * cos_r2 + sy2 * sin_r2 + 0.1  # Add regularization
        cov_01 = (sx2 - sy2) * cos_sin
        cov_11 = sx2 * sin_r2 + sy2 * cos_r2 + 0.1  # Add regularization

        # Cache the results
        self._covariance_cache = {
            "scales": scales.clone(),
            "cov_elements": (cov_00, cov_01, cov_11),
        }

        return cov_00, cov_01, cov_11

    def save_gaussians(self, save_path: str):
        """Enhanced save with optimization metadata"""
        state_dict = {
            "means_2d": self.means_2d.cpu(),
            "pre_act_scales": self.pre_act_scales.cpu(),
            "pre_act_rotation": self.pre_act_rotation.cpu(),
            "colours": self.colours.cpu(),
            "pre_act_phase": self.pre_act_phase.cpu(),
            "pre_act_opacities": self.pre_act_opacities.cpu(),
            "img_size": self.img_size,
            "merge_opacity": self.merge_opacity,
        }
        torch.save(state_dict, save_path)
        print(f"2D Gaussians saved to {save_path}")

    def load_gaussians(self, load_path: str):
        """Load saved gaussians and invalidate caches"""
        state_dict = torch.load(load_path, map_location=self.device)
        self.means_2d.data.copy_(state_dict["means_2d"].to(self.device))
        self.pre_act_scales.data.copy_(state_dict["pre_act_scales"].to(self.device))
        self.pre_act_rotation.data.copy_(state_dict["pre_act_rotation"].to(self.device))
        self.colours.data.copy_(state_dict["colours"].to(self.device))
        self.pre_act_phase.data.copy_(state_dict["pre_act_phase"].to(self.device))
        self.pre_act_opacities.data.copy_(
            state_dict["pre_act_opacities"].to(self.device)
        )
        self.merge_opacity = state_dict.get("merge_opacity", False)
        self.invalidate_cache()
        print(f"2D Gaussians loaded from {load_path}")

    def __len__(self):
        return len(self.means_2d)

    def opacity_regularization(self, decrease_amount=0.003):
        if self.merge_opacity:
            return 0
        # Get current opacities
        opacities = torch.sigmoid(self.pre_act_opacities)
        denominator = opacities * (1 - opacities)
        denominator = torch.clamp(denominator, min=1e-6)

        # Calculate how much to decrease pre_act_opacities by
        delta = decrease_amount / denominator

        # Apply the decrease to pre_act_opacities
        self.pre_act_opacities.data = self.pre_act_opacities.data - delta

        # Count the affected Gaussians (those not already near zero opacity)
        affected_count = (opacities > decrease_amount).sum().item()

        print(
            f"Opacity regularization applied: decreased all opacities by ~{decrease_amount}"
        )
        print(f"Affected {affected_count} out of {len(opacities)} Gaussians")

        return affected_count


class Scene2D:
    def __init__(self, gaussians: Gaussians2D, args_prop):
        self.gaussians = gaussians
        self.args_prop = args_prop
        self.device = gaussians.device
        self.wavelengths = torch.tensor(
            args_prop.wavelengths, dtype=torch.float32, device=self.device
        )

        # Optimization caches
        self._pixel_grid_cache = {}

    def get_pixel_grid(self, img_size: Tuple[int, int]):
        """Cached pixel grid computation"""
        W, H = img_size
        cache_key = f"{W}x{H}"

        if cache_key not in self._pixel_grid_cache:
            xs, ys = torch.meshgrid(
                torch.arange(W, device=self.device, dtype=torch.float32),
                torch.arange(H, device=self.device, dtype=torch.float32),
                indexing="xy",
            )
            self._pixel_grid_cache[cache_key] = torch.stack([xs, ys], dim=-1).reshape(
                1, H * W, 2
            )

        return self._pixel_grid_cache[cache_key]

    def evaluate_gaussians_2d_optimized(
        self, points_2d, means_2d, inv_00, inv_01, inv_11
    ):
        """
        Highly optimized vectorized evaluation of 2D Gaussians

        Args:
            points_2d: (1, H*W, 2) pixel coordinates
            means_2d: (N, 2) gaussian centers
            inv_00, inv_01, inv_11: (N,) inverse covariance elements

        Returns:
            power: (N, H*W) gaussian values at each pixel
        """
        # Reshape for efficient broadcasting
        means_expanded = means_2d.unsqueeze(1)  # (N, 1, 2)
        diff = points_2d - means_expanded  # (N, H*W, 2)

        # Extract components for vectorized computation
        dx = diff[..., 0]  # (N, H*W)
        dy = diff[..., 1]  # (N, H*W)

        # Reshape inverse elements for broadcasting
        inv_00_exp = inv_00.view(-1, 1)  # (N, 1)
        inv_01_exp = inv_01.view(-1, 1)  # (N, 1)
        inv_11_exp = inv_11.view(-1, 1)  # (N, 1)

        # Vectorized Mahalanobis distance computation
        mahal_dist = (
            dx * dx * inv_00_exp + 2 * dx * dy * inv_01_exp + dy * dy * inv_11_exp
        )

        # Gaussian evaluation with numerical stability
        power = -0.5 * mahal_dist
        power = torch.clamp(power, min=-50)  # Prevent underflow

        return power

    def render_hologram_direct(self, img_size: Tuple[int, int], cuda_render):
        """Render Gaussians directly to hologram plane as complex field"""
        W, H = img_size
        device = self.device

        colours = self.gaussians.colours
        # Apply activations with caching
        scales, rotation, phase, opacities, means_2d = (
            self.gaussians.apply_activations()
        )
        if cuda_render:
            hologram_field = cuda_render_scene_2d(
                scales,
                rotation,
                phase,
                opacities,
                means_2d,
                colours,
                img_size,
                self.args_prop.num_planes,
                len(self.wavelengths),
            )
            hologram_field = odak.learn.tools.zero_pad(
                hologram_field, self.args_prop.pad_size
            )
            return hologram_field

        # Get cached pixel grid
        points_2d = self.get_pixel_grid(img_size)  # (1, H*W, 2), coords in pixel space
        # Get cached covariance elements
        cov_00, cov_01, cov_11 = self.gaussians.compute_2d_covariance_elements(
            scales, rotation
        )

        # Compute inverse covariance elements
        inv_00, inv_01, inv_11 = self.gaussians.invert_cov_2D(cov_00, cov_01, cov_11)

        # Initialize hologram field
        hologram_field = torch.zeros(
            (len(self.wavelengths), H, W), dtype=torch.complex64, device=device
        )

        # Evaluate all Gaussians at once for better cache locality
        gaussian_powers = self.evaluate_gaussians_2d_optimized(
            points_2d, means_2d, inv_00, inv_01, inv_11
        )  # (N, H*W)

        # Keep smooth Gaussian amplitude (no modulation)
        gaussian_values = torch.exp(gaussian_powers).view(len(self.gaussians), H, W)

        # Process in batches for memory efficiency
        batch_size = 5000
        n_gaussians = len(self.gaussians)

        for batch_start in range(0, n_gaussians, batch_size):
            batch_end = min(batch_start + batch_size, n_gaussians)
            batch_indices = torch.arange(batch_start, batch_end, device=device)

            # Get batch data
            batch_gaussian_values = gaussian_values[batch_indices]  # (B, H, W)
            batch_opacities = opacities[batch_indices].view(-1, 1, 1)  # (B, 1, 1)
            batch_colours = colours[batch_indices]  # (B, 3)
            batch_phase = phase[batch_indices]  # (B, 3)
            batch_means = means_2d[batch_indices]  # (B, 2)

            # Compute alpha values (smooth amplitude)
            alphas = batch_opacities * batch_gaussian_values  # (B, H, W)

            # Vectorized complex field computation for all wavelengths
            for c in range(len(self.wavelengths)):
                color_c = batch_colours[:, c].view(-1, 1, 1)  # (B, 1, 1)
                phase_c = batch_phase[:, c].view(-1, 1, 1)  # (B, 1, 1)
                total_phase = phase_c

                # Complex wave: smooth amplitude × complex phase (with spatial variation)
                complex_contrib = (
                    color_c * alphas * torch.exp(1j * total_phase)
                )  # (B, H, W)

                # Accumulate contributions
                hologram_field[c] += complex_contrib.sum(dim=0)

        hologram_field = odak.learn.tools.zero_pad(
            hologram_field, self.args_prop.pad_size
        )
        return hologram_field

    def render(self, img_size: Tuple[int, int], cuda_render=True):
        """Main rendering function - directly render hologram"""
        # Invalidate Gaussian caches when new rendering starts
        self.gaussians.invalidate_cache()

        # Render 2D Gaussians directly to hologram
        hologram_complex = self.render_hologram_direct(img_size, cuda_render)

        return hologram_complex


def make_trainable_2d(gaussians):
    """Make 2D Gaussian parameters trainable - removed plane assignment"""
    gaussians.means_2d.requires_grad_()
    gaussians.pre_act_scales.requires_grad_()
    gaussians.pre_act_rotation.requires_grad_()
    gaussians.colours.requires_grad_()
    gaussians.pre_act_phase.requires_grad_()
    if not gaussians.merge_opacity:
        gaussians.pre_act_opacities.requires_grad_()

    # Hook to invalidate cache when parameters are updated
    def invalidate_hook(grad):
        gaussians.invalidate_cache()
        return grad

    gaussians.means_2d.register_hook(invalidate_hook)
    gaussians.pre_act_scales.register_hook(invalidate_hook)
    gaussians.pre_act_rotation.register_hook(invalidate_hook)
    gaussians.colours.register_hook(invalidate_hook)
    gaussians.pre_act_phase.register_hook(invalidate_hook)
    if not gaussians.merge_opacity:
        gaussians.pre_act_opacities.register_hook(invalidate_hook)
