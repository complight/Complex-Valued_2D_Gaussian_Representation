import argparse
import gc
import os
import random
import sys
import time
from argparse import Namespace

import imageio

import imageio.v2 as imageio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import odak
import torch
import torch.nn as nn
import torch.nn.functional as F
from cuda_prop import BandlimitedPropagation
from odak.learn.tools import multi_scale_total_variation_loss
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from utils import (
    complex_to_phase_encode,
    GaussianLoss,
    multiplane_loss,
    propagator,
    set_seed,
)

result_dir = os.path.join("./result_2d_numerical")
os.makedirs(result_dir, exist_ok=True)

set_seed(100)


def load_target_image(image_path, img_size):
    """Load and preprocess target image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size, Image.LANCZOS)

    img_array = np.array(img) / 255.0
    target_tensor = torch.from_numpy(img_array).float()
    target_tensor = target_tensor.permute(2, 0, 1)  # (C, H, W)

    print(f"Loaded target image: {image_path}, size: {img_size}")
    return target_tensor


def load_depth_image(depth_path, img_size):
    """Load and preprocess depth image"""
    if not os.path.exists(depth_path):
        return None

    depth_img = Image.open(depth_path).convert("L")
    depth_img = depth_img.resize(img_size, Image.LANCZOS)

    depth_array = np.array(depth_img) / 255.0
    depth_tensor = torch.from_numpy(depth_array).float()
    depth_tensor = depth_tensor.unsqueeze(0)  # (1, H, W)

    print(f"Loaded depth image: {depth_path}, size: {img_size}")
    return depth_tensor


class HologramOptimizer(nn.Module):
    def __init__(self, hologram_size, device="cuda"):
        super().__init__()
        h, w = hologram_size

        self.phase = nn.Parameter(torch.zeros(3, h, w, device=device))

    def forward(self):
        amplitude = torch.ones_like(self.phase)
        return amplitude, self.phase


def run_training_hologram(args, args_prop):
    img_size = tuple(args.img_size)
    lpips_fn = lpips.LPIPS(net="vgg").cuda()

    target_image = load_target_image(args.target_image_path, img_size).cuda()
    C, H, W = target_image.shape
    print(f"Target image shape: {target_image.shape}")

    depth_image = (
        load_depth_image(args.depth_path, img_size) if args.depth_path else None
    )
    has_depth = depth_image is not None

    if has_depth:
        depth_image = depth_image.cuda()
        print(f"Depth image shape: {depth_image.shape}")
    else:
        depth_image = torch.ones((1, H, W), device="cuda")
        print("Using default depth (all ones)")

    print(f"Has depth: {has_depth}")

    if not has_depth and args_prop.num_planes > 1:
        raise ValueError("Depth is required for supervision more than 1 plane.")

    model = HologramOptimizer(args_prop.pad_size, device=args.device).cuda()
    print(f"Using direct hologram optimization with phase-only hologram")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_itrs, eta_min=1e-3
    )

    targets, loss_function, mask = multiplane_loss(
        target_image=target_image, target_depth=depth_image, args_prop=args_prop
    )
    for idx, target in enumerate(targets):
        odak.learn.tools.save_image(f"{result_dir}/target_{idx}.png", target, cmin=0.0, cmax=1.0)

    running_losses = []
    running_psnrs = []
    best_psnr = 0.0

    pbar = tqdm(range(args.num_itrs), desc="Optimizing")

    for itr in pbar:
        optimizer.zero_grad()

        amplitude, phase_map = model()
        phase_map = phase_map % (2 * odak.pi)

        reconstruction_intensities_sum = propagator.reconstruct(
            phase_map, amplitude=amplitude, no_grad=False
        )
        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)

        loss = 0.0
        loss += multi_scale_total_variation_loss(phase_map, levels=6) * 0.01
        random.seed(int(time.time()))
        num_planes = len(reconstruction_intensities)
        selected_plane_idx = random.randint(0, num_planes - 1)
        for idx, (reconstruction_intensity, target) in enumerate(
            zip(reconstruction_intensities, targets)
        ):
            pred_cropped = odak.learn.tools.crop_center(
                reconstruction_intensity, size=(H, W)
            )
            pred_cropped = torch.clamp(pred_cropped, min=0.0, max=1.0)

            loss += loss_function(pred_cropped, target, idx, inject_noise=True)
            ssim_loss = GaussianLoss(pred_cropped, target)
            loss += ssim_loss

            if idx == 0:
                current_psnr = peak_signal_noise_ratio(
                    target.detach().cpu().numpy(), pred_cropped.detach().cpu().numpy()
                )
                running_psnrs.append(current_psnr)

        running_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        mean_loss = sum(running_losses[-50:]) / min(len(running_losses), 50)
        mean_psnr = sum(running_psnrs[-50:]) / min(len(running_psnrs), 50)
        current_lr = scheduler.get_last_lr()[0]

        pbar.set_postfix(
            {
                "Loss": f"{mean_loss:.6f}",
                "PSNR": f"{mean_psnr:.2f}",
                "LR": f"{current_lr:.2e}",
            }
        )

        if args.viz_freq != -1 and itr % args.viz_freq == 0:
            with torch.no_grad():
                for plane_idx in range(args_prop.num_planes):
                    if plane_idx < len(reconstruction_intensities):
                        recon = reconstruction_intensities[plane_idx]
                        recon = torch.clamp(recon, min=0.0, max=1.0)
                        recon = odak.learn.tools.crop_center(recon, size=(H, W))

                        plane_suffix = (
                            f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                        )
                        odak.learn.tools.save_image(
                            f"{result_dir}/recon_{itr:06d}{plane_suffix}.png",
                            recon,
                            cmin=0.0,
                            cmax=1.0,
                        )

                phase_cropped = odak.learn.tools.crop_center(phase_map, size=(H, W))
                amp_cropped = odak.learn.tools.crop_center(amplitude, size=(H, W))

                odak.learn.tools.save_image(
                    f"{result_dir}/phase_{itr:06d}.png",
                    phase_cropped,
                    cmin=0.0,
                    cmax=2 * odak.pi,
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/amp_{itr:06d}.png", amp_cropped, cmin=0.0, cmax=1.0
                )

        if (itr % args.eval_freq == 0 and itr > 0) or (itr == args.num_itrs - 1):
            with torch.no_grad():
                targets, _, _ = multiplane_loss(
                    target_image=target_image,
                    target_depth=depth_image,
                    args_prop=args_prop,
                )

                amplitude, phase_map = model()
                phase_map = phase_map % (2 * odak.pi)

                reconstruction_intensities_sum = propagator.reconstruct(
                    phase_map, amplitude=amplitude, no_grad=False
                )
                reconstruction_intensities = torch.sum(
                    reconstruction_intensities_sum, dim=0
                )

                all_psnr_vals = []
                all_ssim_vals = []
                all_lpips_vals = []

                for plane_idx in range(
                    min(len(reconstruction_intensities), len(targets))
                ):
                    recon = reconstruction_intensities[plane_idx]
                    recon = torch.clamp(recon, min=0.0, max=1.0)
                    recon = odak.learn.tools.crop_center(recon, size=(H, W))
                    target = targets[plane_idx]

                    if recon.dim() == 3:
                        recon_np = recon.permute(1, 2, 0).detach().cpu().numpy()
                    else:
                        recon_np = recon.detach().cpu().numpy()

                    if target.dim() == 3:
                        target_np = target.permute(1, 2, 0).detach().cpu().numpy()
                    else:
                        target_np = target.detach().cpu().numpy()

                    psnr_val = peak_signal_noise_ratio(
                        target_np, recon_np, data_range=1.0
                    )
                    ssim_val = structural_similarity(
                        target_np,
                        recon_np,
                        data_range=1.0,
                        channel_axis=2 if recon_np.ndim == 3 else None,
                    )

                    if recon.dim() == 2:
                        recon_3ch = recon.unsqueeze(0).repeat(3, 1, 1)
                    elif recon.shape[0] != 3:
                        recon_3ch = (
                            recon.repeat(3, 1, 1) if recon.shape[0] == 1 else recon[:3]
                        )
                    else:
                        recon_3ch = recon

                    if target.dim() == 2:
                        target_3ch = target.unsqueeze(0).repeat(3, 1, 1)
                    elif target.shape[0] != 3:
                        target_3ch = (
                            target.repeat(3, 1, 1)
                            if target.shape[0] == 1
                            else target[:3]
                        )
                    else:
                        target_3ch = target

                    recon_norm = 2 * recon_3ch.unsqueeze(0) - 1
                    target_norm = 2 * target_3ch.unsqueeze(0) - 1
                    lpips_val = lpips_fn(recon_norm, target_norm).item()

                    all_psnr_vals.append(psnr_val)
                    all_ssim_vals.append(ssim_val)
                    all_lpips_vals.append(lpips_val)

                primary_plane_idx = (
                    min(1, args_prop.num_planes - 1) if args_prop.num_planes > 1 else 0
                )
                primary_recon = (
                    reconstruction_intensities[primary_plane_idx]
                    if len(reconstruction_intensities) > primary_plane_idx
                    else reconstruction_intensities[0]
                )
                primary_recon = torch.clamp(primary_recon, min=0.0, max=1.0)
                primary_recon = odak.learn.tools.crop_center(primary_recon, size=(H, W))
                primary_target = (
                    targets[primary_plane_idx]
                    if len(targets) > primary_plane_idx
                    else targets[0]
                )

                contrast_window = 3
                eps = 1e-8
                pred_bg = F.avg_pool2d(
                    primary_recon.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )
                target_bg = F.avg_pool2d(
                    primary_target.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )

                pred_weber = (primary_recon.unsqueeze(0) - pred_bg) / (
                    torch.abs(pred_bg) + eps
                )
                target_weber = (primary_target.unsqueeze(0) - target_bg) / (
                    torch.abs(target_bg) + eps
                )

                pred_weber_mean = pred_weber.mean().item()
                target_weber_mean = target_weber.mean().item()
                pred_weber_std = pred_weber.std().item()
                target_weber_std = target_weber.std().item()

                pred_max = F.max_pool2d(
                    primary_recon.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )
                pred_min = -F.max_pool2d(
                    -primary_recon.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )
                target_max = F.max_pool2d(
                    primary_target.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )
                target_min = -F.max_pool2d(
                    -primary_target.unsqueeze(0),
                    contrast_window,
                    stride=1,
                    padding=contrast_window // 2,
                )

                pred_michelson = (pred_max - pred_min) / (pred_max + pred_min + eps)
                target_michelson = (target_max - target_min) / (
                    target_max + target_min + eps
                )

                pred_michelson_mean = pred_michelson.mean().item()
                target_michelson_mean = target_michelson.mean().item()
                pred_michelson_std = pred_michelson.std().item()
                target_michelson_std = target_michelson.std().item()

                print(f"\nEvaluation at iteration {itr}:")
                for plane_idx in range(len(all_psnr_vals)):
                    print(
                        f"Plane {plane_idx}: PSNR: {all_psnr_vals[plane_idx]:.3f}, SSIM: {all_ssim_vals[plane_idx]:.3f}, LPIPS: {all_lpips_vals[plane_idx]:.3f}"
                    )

                mean_psnr = sum(all_psnr_vals) / len(all_psnr_vals)
                mean_ssim = sum(all_ssim_vals) / len(all_ssim_vals)
                mean_lpips = sum(all_lpips_vals) / len(all_lpips_vals)
                print(
                    f"Mean: PSNR: {mean_psnr:.3f}, SSIM: {mean_ssim:.3f}, LPIPS: {mean_lpips:.3f}"
                )
                print(
                    f"Weber Contrast - Pred: {pred_weber_mean:.3f}±{pred_weber_std:.3f}, Target: {target_weber_mean:.3f}±{target_weber_std:.3f}"
                )
                print(
                    f"Michelson Contrast - Pred: {pred_michelson_mean:.3f}±{pred_michelson_std:.3f}, Target: {target_michelson_mean:.3f}±{target_michelson_std:.3f}"
                )
                print(f"Current LR: {current_lr:.2e}")

                primary_psnr = (
                    all_psnr_vals[primary_plane_idx]
                    if len(all_psnr_vals) > primary_plane_idx
                    else all_psnr_vals[0]
                )
                if primary_psnr > best_psnr:
                    best_psnr = primary_psnr
                    best_model_path = f"{result_dir}/best_model_{itr}.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")
                else:
                    latest_model_path = f"{result_dir}/latest_model_{itr}.pth"
                    torch.save(model.state_dict(), latest_model_path)
                sys.stdout.flush()

    torch.save(model.state_dict(), f"{result_dir}/final.pth")
    print("[*] Training Completed.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_image_path",
        default="./images/071.png",
        type=str,
        help="Path to the target image to overfit",
    )
    parser.add_argument(
        "--depth_path",
        default="./images/d_071.png",
        type=str,
        help="Path to the depth image (optional). If not provided, depth will be all ones.",
    )
    parser.add_argument(
        "--num_itrs",
        default=2001,
        type=int,
        help="Number of iterations to train the model.",
    )
    parser.add_argument(
        "--viz_freq",
        default=200,
        type=int,
        help="Frequency with which visualization should be performed.",
    )
    parser.add_argument(
        "--eval_freq", default=1000, type=int, help="Frequency with evaluation process."
    )
    parser.add_argument("--lr", default=0.025, type=float, help="Learning Rate")
    parser.add_argument(
        "--img_size",
        nargs=2,
        default=[1024, 640],
        type=int,
        help="Target image resolution",
    )
    parser.add_argument(
        "--split_ratio",
        default=1.0,
        type=float,
        help="split_ratio for depth in multiplane loss",
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args_prop = Namespace(
        wavelengths=[639e-9, 532e-9, 473e-9],
        pixel_pitch=3.74e-6,
        volume_depth=4e-3,
        d_val=3e-3,
        pad_size=[max(args.img_size), max(args.img_size)],
        aperture_size=int(sum(args.img_size) / 2.0),
        num_planes=2,
        split_ratio=args.split_ratio,
    )

    if args_prop.num_planes > 1:
        args_prop.distances = (
            torch.linspace(
                -args_prop.volume_depth / 2.0,
                args_prop.volume_depth / 2.0,
                args_prop.num_planes,
            )
            + args_prop.d_val
        )
    else:
        args_prop.distances = [args_prop.d_val]

    print("Distance: ", args_prop.distances)

    propagator = propagator(
        resolution=args_prop.pad_size,
        wavelengths=args_prop.wavelengths,
        pixel_pitch=args_prop.pixel_pitch,
        number_of_frames=3,
        number_of_depth_layers=args_prop.num_planes,
        volume_depth=args_prop.volume_depth,
        image_location_offset=args_prop.d_val,
        propagation_type="Bandlimited Angular Spectrum",
        propagator_type="forward",
        laser_channel_power=torch.eye(3),
        aperture_size=args_prop.aperture_size,
        aperture=None,
        method="conventional",
        device="cuda",
    )

    run_training_hologram(args, args_prop)
