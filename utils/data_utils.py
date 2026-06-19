import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
import odak
import math
import sys
import random
from torch.nn.functional import mse_loss
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from pytorch_msssim import SSIM, ms_ssim

CMAP_JET = plt.get_cmap("jet")
CMAP_MIN_NORM, CMAP_MAX_NORM = 5.0, 7.0


@contextlib.contextmanager
def console_only_print():
    original_stdout = sys.stdout
    if original_stdout != sys.__stdout__:
        sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdout = original_stdout


def GaussianLoss(pred, target, lambda_ssim=0.025, lambda_l2=10, lambda_flip=0.025):
    """
    Calculate combined Gaussian loss with SSIM and L2 components.

    Args:
        pred: Predicted image [3, H, W] or [B, 3, H, W]
        target: Target image [3, H, W, 3] or [B, 3, H, W]
        lambda_ssim: Weight for SSIM loss component (default: 0.02)
        lambda_l2: Weight for L2 loss component (default: 10)

    Returns:
        total_loss: Combined weighted loss
    """
    # Add batch dimension if not present
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    ssim_loss = (1 - ssim(pred, target)) * 0.2
    total_loss = lambda_ssim * ssim_loss
    return total_loss


def multiplane_loss(target_image, target_depth, args_prop):
    from .propagator import multiplane_loss_odak
    loss_function = multiplane_loss_odak(
                        target_image = target_image,
                        target_depth = target_depth,
                        target_blur_size = 20,
                        number_of_planes = args_prop.num_planes,
                        blur_ratio = 8,
                        weights = [1.0, 1.0, 1.0, 0.0],
                        scheme = "defocus",
                        reduction = "mean",
                        split_ratio = args_prop.split_ratio,  
                        device = "cuda"
    )

    targets, mask, quantized_depth = loss_function.get_targets()
    return targets, loss_function, mask

def count_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params) / 1e6:.2f}M)")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
    
    
def complex_to_phase_encode(phase_low, phase_high, DPAC=False):
    """Apply double phase encoding to combine 6 phase channels into 3"""
    """Or complex encoding to convert complex 6 channels into phase-only 3 channels"""
    if not DPAC:
        phase_encoded = torch.zeros_like(phase_low)
        phase_encoded[..., 0::2, 0::2] = phase_low[..., 0::2, 0::2]
        phase_encoded[..., 1::2, 1::2] = phase_low[..., 1::2, 1::2]
        phase_encoded[..., 0::2, 1::2] = phase_high[..., 0::2, 1::2]
        phase_encoded[..., 1::2, 0::2] = phase_high[..., 1::2, 0::2]
        phase_encoded = phase_encoded - phase_encoded.mean()
        return phase_encoded
    else:
        amplitude = phase_low
        phase = phase_high
        eps=1e-4
        steep=4.0
        A_soft = (1 - 2*eps) * torch.sigmoid(steep * (amplitude - 0.5)) + eps  # shape = phase
        delta  = torch.acos(A_soft)  # d/dA ~ -1/sqrt(1-A^2): finite because A∈(eps,1-eps)

        alpha = phase + delta
        beta  = phase - delta

        # Interleave α,β in a 2×2 checkerboard (no in-place on views that would break autograd graph)
        phase_encoded = torch.empty_like(phase)
        phase_encoded[..., 0::2, 0::2] = alpha[..., 0::2, 0::2]
        phase_encoded[..., 1::2, 1::2] = alpha[..., 1::2, 1::2]
        phase_encoded[..., 0::2, 1::2] = beta [..., 0::2, 1::2]
        phase_encoded[..., 1::2, 0::2] = beta [..., 1::2, 0::2]
        return phase_encoded
