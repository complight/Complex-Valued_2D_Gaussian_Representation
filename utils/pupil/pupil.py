"""Pupil-aware sampling utilities for gaze-contingent hologram supervision."""

import math
import random
import time
from random import Random

import torch
import torch.nn.functional as F

# Separate RNG for pupil sampling — seeded from wall-clock time each call
# so positions are non-repeating, without affecting the global random state.
_pupil_rng = Random()


def sample_pupil_params(
    img_size, aperture_size, pupil_ratio=0.3, num_pupils=1, offset_scale=1.0
):
    """Sample random pupil centers within (and optionally beyond) the eye-box.

    Parameters
    ----------
    img_size      : int — spatial size of the propagated field (Ny = Nx)
    aperture_size : float — BL-ASM aperture radius (in doubled-grid pixels)
    pupil_ratio   : float, or list/tuple of 1-2 floats —
                    Single value = fixed ratio.
                    Two values = [min, max] range, randomly sampled each call.
    num_pupils    : int — number of pupil positions to sample
    offset_scale  : float — multiplier for max pupil offset.
                    1.0 = pupils stay within eyebox (default).
                    >1.0 = pupils can go beyond eyebox boundary, covering
                    the aperture edge region for better boundary supervision.

    Returns
    -------
    centers : list of (center_y, center_x) tuples
    radius  : float — shared pupil radius in pixels
    """
    # Reseed with current time so positions differ every call
    _pupil_rng.seed(time.time_ns())

    # Resolve pupil_ratio: fixed or random from range
    if isinstance(pupil_ratio, (list, tuple)):
        if len(pupil_ratio) == 2:
            ratio = _pupil_rng.uniform(pupil_ratio[0], pupil_ratio[1])
        else:
            ratio = pupil_ratio[0]
    else:
        ratio = pupil_ratio

    pupil_r = ratio * aperture_size
    eyebox_aperture = img_size / 2.0 / 1.4
    # Allow free movement within the eyebox, matching pupil_scan_gif behavior.
    # The pupil mask operates on the 2×-padded FFT grid, so pupil extending
    # beyond the original grid boundary is fine (no clipping needed).
    max_offset = offset_scale * eyebox_aperture
    mid_y = img_size / 2.0
    mid_x = img_size / 2.0

    centers = []
    for _ in range(num_pupils):
        angle = _pupil_rng.uniform(0, 2 * math.pi)
        r = math.sqrt(_pupil_rng.uniform(0, 1)) * max_offset
        centers.append((mid_y + r * math.sin(angle), mid_x + r * math.cos(angle)))

    return centers, pupil_r


def apply_stochastic_pupil(
    field,
    min_pupil_size=0.3,
    max_pupil_size=0.4,
    pupil_range=(0.4, 0.4),
    num_pupils=4,
):
    """Apply random sub-pupil filters to the complex field in Fourier domain.

    Samples num_pupils random positions, reconstructs through each, and averages
    the results. Averaging multiple pupils per iteration gives the optimizer a
    more uniform gradient signal across the eyebox.

    Args:
        field: complex tensor with last 2 dims being (Ny, Nx).
               e.g. (num_frames, num_depth, num_channels, Ny, Nx).
        min_pupil_size: float — min pupil radius in normalized coords [-1,1].
        max_pupil_size: float — max pupil radius in normalized coords [-1,1].
        pupil_range: (float, float) — max offset range for pupil center.
        num_pupils: int — number of random pupil positions to average.

    Returns:
        Complex tensor of same shape — filtered through random sub-pupil(s).
        When num_pupils > 1, returns the average of all pupil reconstructions.
    """
    # Get spatial dimensions
    original_shape = field.shape
    ny, nx = original_shape[-2:]
    pad_ny, pad_nx = 2 * ny, 2 * nx

    # Flatten all leading dimensions for processing
    field_flat = field.reshape(-1, ny, nx)

    # Pad each item in batch
    pad_field = F.pad(
        field_flat, (nx // 2, nx // 2, ny // 2, ny // 2), mode="constant", value=0
    )

    # FFT to eyebox (shared across all pupils)
    eyebox = torch.fft.fftshift(torch.fft.fft2(pad_field, dim=(-2, -1)), dim=(-2, -1))

    # Create coordinate grids (shared)
    iy = torch.linspace(-1, 1, pad_ny, device=field.device)
    ix = torch.linspace(-1, 1, pad_nx, device=field.device)
    Y, X = torch.meshgrid(iy, ix, indexing="ij")

    recon_accum = None
    for _ in range(num_pupils):
        # Random pupil parameters per sample
        pupil_rad = random.uniform(min_pupil_size, max_pupil_size)
        pupil_pos = (
            random.uniform(-pupil_range[0], pupil_range[0]),
            random.uniform(-pupil_range[1], pupil_range[1]),
        )

        # Circular pupil mask
        pupil_mask = ((Y - pupil_pos[0]) ** 2 + (X - pupil_pos[1]) ** 2) < (
            pupil_rad**2
        )
        pupil_mask = pupil_mask.float()

        # Apply pupil and IFFT
        filtered = eyebox * pupil_mask.unsqueeze(0)
        recon_flat = torch.fft.ifft2(
            torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
        )

        # Crop back to original size
        start_y, start_x = ny // 2, nx // 2
        recon_flat = recon_flat[:, start_y : start_y + ny, start_x : start_x + nx]

        if recon_accum is None:
            recon_accum = recon_flat
        else:
            recon_accum = recon_accum + recon_flat

    # Average across pupils
    recon = (recon_accum / num_pupils).reshape(original_shape)
    return recon
