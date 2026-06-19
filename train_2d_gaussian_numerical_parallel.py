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
from model_2d_gaussian import Gaussians2D, make_trainable_2d, Scene2D
from odak.learn.tools import multi_scale_total_variation_loss
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from utils import (
    Adan,
    console_only_print,
    GaussianLoss,
    multiplane_loss,
    propagator,
    set_seed,
)
from utils.pupil import apply_stochastic_pupil, eyebox_with_subpupil, pupil_scan_gif

result_dir = os.path.join("./result_2d_parallel")
checkpoint_dir = os.path.join(result_dir, "checkpoints")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
logsave = result_dir

log_debug = True
if log_debug:
    sys.stdout = open(os.path.join(logsave, "log.txt"), "w")

set_seed(100)


def load_target_image(image_path, img_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size, Image.LANCZOS)
    img_array = np.array(img) / 255.0
    target_tensor = torch.from_numpy(img_array).float()
    target_tensor = target_tensor.permute(2, 0, 1)
    print(f"Loaded target image: {image_path}, size: {img_size}")
    return target_tensor


def load_depth_image(depth_path, img_size):
    if not os.path.exists(depth_path):
        return None
    depth_img = Image.open(depth_path).convert("L")
    depth_img = depth_img.resize(img_size, Image.LANCZOS)
    depth_array = np.array(depth_img) / 255.0
    depth_tensor = torch.from_numpy(depth_array).float()
    depth_tensor = depth_tensor.unsqueeze(0)
    print(f"Loaded depth image: {depth_path}, size: {img_size}")
    return depth_tensor


def calculate_psnr(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np)


def setup_optimizer_parallel(
    gaussians, numerical_phase, num_itrs, lr=0.01, lr_p=0.025, freeze_gaussians=False
):
    param_groups = []

    if not freeze_gaussians:
        param_groups.extend(
            [
                {"params": [gaussians.means_2d], "lr": lr, "name": "means"},
                {"params": [gaussians.pre_act_scales], "lr": 0.005, "name": "scales"},
                {
                    "params": [gaussians.colours, gaussians.pre_act_phase],
                    "lr": 0.0025,
                    "name": "amplitude_phase",
                },
                {
                    "params": [gaussians.pre_act_opacities],
                    "lr": 0.025,
                    "name": "opacity",
                },
                {
                    "params": [gaussians.pre_act_rotation],
                    "lr": 0.001,
                    "name": "rotation",
                },
            ]
        )

    param_groups.append(
        {"params": [numerical_phase], "lr": lr_p, "name": "numerical_phase"}
    )

    optimizer = Adan(param_groups, lr=0, eps=1e-8)

    trainable_params = []
    for param_group in param_groups:
        trainable_params.extend(param_group["params"])

    decay_group_indices = []
    decay_params_list = []
    decay_group_names = (
        ["means", "numerical_phase"] if not freeze_gaussians else ["numerical_phase"]
    )

    for i, group in enumerate(optimizer.param_groups):
        if group.get("name") in decay_group_names:
            decay_group_indices.append(i)
            decay_params_list.append({"params": group["params"], "lr": group["lr"]})

    temp_optimizer = torch.optim.SGD(decay_params_list)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        temp_optimizer, T_max=num_itrs, eta_min=1e-3
    )

    class CustomScheduler:
        def __init__(self, optimizer, decay_scheduler, decay_group_indices):
            self.optimizer = optimizer
            self.decay_scheduler = decay_scheduler
            self.decay_group_indices = decay_group_indices
            self.last_epoch = decay_scheduler.last_epoch
            self._last_lr = (
                decay_scheduler._last_lr
                if hasattr(decay_scheduler, "_last_lr")
                else None
            )

        def step(self):
            self.decay_scheduler.step()
            for i, group_idx in enumerate(self.decay_group_indices):
                self.optimizer.param_groups[group_idx][
                    "lr"
                ] = self.decay_scheduler.get_last_lr()[i]
            self.last_epoch = self.decay_scheduler.last_epoch
            self._last_lr = self.decay_scheduler.get_last_lr()

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    scheduler = CustomScheduler(optimizer, decay_scheduler, decay_group_indices)
    return optimizer, scheduler, trainable_params


def run_training_parallel(args, args_prop):
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

    gaussians = Gaussians2D(
        num_points=args.num_gaussians,
        img_size=img_size,
        device=args.device,
        args_prop=args_prop,
    )

    scene = Scene2D(gaussians, args_prop)
    make_trainable_2d(gaussians)

    if args.load_gaussians:
        gaussians.load_gaussians(args.gaussian_weights_path)
        gaussians.means_2d.requires_grad_(False)
        gaussians.pre_act_scales.requires_grad_(False)
        gaussians.colours.requires_grad_(False)
        gaussians.pre_act_phase.requires_grad_(False)
        gaussians.pre_act_opacities.requires_grad_(False)
        gaussians.pre_act_rotation.requires_grad_(False)
        print(f"Loaded and froze Gaussian weights from {args.gaussian_weights_path}")

    pad_h, pad_w = args_prop.pad_size
    # numerical_phase = nn.Parameter(
    #     torch.rand(3, pad_h, pad_w, device=args.device) * 2 * np.pi,
    #     requires_grad=True
    # )
    numerical_phase = nn.Parameter(
        torch.zeros(3, pad_h, pad_w, device=args.device), requires_grad=True
    )
    optimizer, scheduler, _ = setup_optimizer_parallel(
        gaussians=gaussians,
        numerical_phase=numerical_phase,
        num_itrs=args.num_itrs,
        lr=args.lr,
        lr_p=args.lr_p,
        freeze_gaussians=args.load_gaussians,
    )

    targets, loss_function, _ = multiplane_loss(
        target_image=target_image, target_depth=depth_image, args_prop=args_prop
    )
    for idx, target in enumerate(targets):
        odak.learn.tools.save_image(f"{result_dir}/target_{idx}.png", target, cmin=0.0, cmax=1.0)

    running_losses = []
    running_psnrs = []
    best_psnr = 0.0
    viz_psnr_list = []
    viz_iterations = []

    pbar = tqdm(range(args.num_itrs), desc="Training Parallel")
    for itr in pbar:
        optimizer.zero_grad()

        if not has_depth and args_prop.num_planes > 1:
            raise ValueError("Depth is required for supervision more than 1 plane.")

        hologram_complex_g = scene.render(img_size)
        phase_g = odak.learn.wave.calculate_phase(hologram_complex_g) % (2 * odak.pi)
        amplitude_g = odak.learn.wave.calculate_amplitude(hologram_complex_g)
        amplitude_g = torch.clamp(amplitude_g, min=0.0, max=1.0)

        propagator.use_cuda_blasm_gaussian_version = True
        recon_g, complex_g = propagator.reconstruct(
            phase_g, amplitude=amplitude_g, no_grad=False, get_complex=True
        )
        recon_intensities_g = torch.sum(recon_g, dim=0)

        # propagator.use_cuda_blasm_gaussian_version = False
        # phase_p = numerical_phase % (2 * odak.pi)
        # recon_p, complex_p = propagator.reconstruct(phase_p, amplitude=None, no_grad=False, get_complex=True)
        # recon_intensities_p = torch.sum(recon_p, dim=0)

        propagator.use_cuda_blasm_gaussian_version = False
        phase_p = numerical_phase % (2 * odak.pi)
        recon_p, complex_p = propagator.reconstruct(
            phase_p, amplitude=None, no_grad=False, get_complex=True
        )
        recon_intensities_p = torch.sum(recon_p, dim=0)

        if args.pupil_aware:
            # Also compute pupil-filtered reconstruction for combined supervision
            complex_p_pupil = apply_stochastic_pupil(
                complex_p,
                min_pupil_size=args.pupil_ratio[0],
                max_pupil_size=args.pupil_ratio[-1],
                pupil_range=(0.4, 0.4),
                num_pupils=args.num_pupils,
            )
            recon_p_pupil = torch.abs(complex_p_pupil) ** 2
            recon_intensities_p_pupil = torch.sum(recon_p_pupil, dim=0)

        loss = 0.0
        if not args.load_gaussians:
            for idx, (recon_g_plane, target) in enumerate(
                zip(recon_intensities_g, targets)
            ):
                pred_g = odak.learn.tools.crop_center(recon_g_plane, size=(H, W))
                pred_g = torch.clamp(pred_g, min=0.0, max=1.0)
                loss += loss_function(pred_g, target, idx, inject_noise=True)
                ssim_loss_g = GaussianLoss(pred_g, target)
                loss += ssim_loss_g

        # Full-aperture recon loss for POH
        for idx, (recon_p_plane, target) in enumerate(
            zip(recon_intensities_p, targets)
        ):
            pred_p = odak.learn.tools.crop_center(recon_p_plane, size=(H, W))
            pred_p = torch.clamp(pred_p, min=0.0, max=1.0)
            loss += loss_function(pred_p, target, idx, inject_noise=True)
            ssim_loss_p = GaussianLoss(pred_p, target)
            loss += ssim_loss_p

            if idx == 0:
                psnr_p = calculate_psnr(pred_p, target)
                running_psnrs.append(psnr_p)

        # Pupil-aware recon loss for POH (combined with full-aperture above)
        if args.pupil_aware:
            for idx, (recon_pupil_plane, target) in enumerate(
                zip(recon_intensities_p_pupil, targets)
            ):
                pred_pupil = odak.learn.tools.crop_center(
                    recon_pupil_plane, size=(H, W)
                )
                pred_pupil = torch.clamp(pred_pupil, min=0.0, max=1.0)
                loss += args.lambda_pupil * loss_function(
                    pred_pupil, target, idx, inject_noise=True
                )
                loss += args.lambda_pupil * GaussianLoss(pred_pupil, target)

            # DC suppression: penalize the DC (center) energy in the eyebox
            # to spread energy uniformly across the aperture
            for plane_idx in range(args_prop.num_planes):
                cf_rgb = torch.stack(
                    [complex_p[c, plane_idx, c] for c in range(3)], dim=0
                )  # (3, Ny, Nx)
                ny, nx = cf_rgb.shape[-2:]
                pad_cf = F.pad(cf_rgb, (nx // 2, nx // 2, ny // 2, ny // 2))
                eyebox = torch.fft.fftshift(
                    torch.fft.fft2(pad_cf, dim=(-2, -1)), dim=(-2, -1)
                )
                eyebox_amp = eyebox.abs()
                # DC bin is at center of the 2× grid
                cy, cx = eyebox_amp.shape[-2] // 2, eyebox_amp.shape[-1] // 2
                dc_region = eyebox_amp[:, cy - 2 : cy + 3, cx - 2 : cx + 3]
                dc_energy = dc_region.mean()
                total_energy = eyebox_amp.mean() + 1e-10
                loss += args.lambda_dc * (dc_energy / total_energy)

        for recon_g_plane, recon_p_plane in zip(
            recon_intensities_g, recon_intensities_p
        ):
            pred_g = odak.learn.tools.crop_center(recon_g_plane, size=(H, W))
            pred_p = odak.learn.tools.crop_center(recon_p_plane, size=(H, W))
            loss += args.recon_weight * F.mse_loss(pred_g, pred_p)

        # loss += args.complex_weight * (
        #     F.l1_loss(complex_g.real, complex_p.real) +
        #     F.l1_loss(complex_g.imag, complex_p.imag)
        # )
        running_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
            if not args.load_gaussians
            else optimizer.param_groups[-1]["lr"]
        )
        current_lr_p = next(
            g["lr"] for g in optimizer.param_groups if g["name"] == "numerical_phase"
        )
        mean_loss = sum(running_losses[-50:]) / min(len(running_losses), 50)
        mean_psnr = sum(running_psnrs[-50:]) / min(len(running_psnrs), 50)

        mode_str = "Phase-Only" if args.load_gaussians else "Full"
        pbar.set_postfix(
            {
                "Loss": f"{mean_loss:.6f}",
                "PSNR": f"{mean_psnr:.2f}",
                "LR": f"{current_lr:.2e}",
                "LR_P": f"{current_lr_p:.2e}",
                "G": f"{len(gaussians)}",
                "Mode": mode_str,
            }
        )

        if args.viz_freq != -1 and itr % args.viz_freq == 0:
            # if itr in [0, 10, 50, 200, 500, 1000, 1500, 2000]:
            with torch.no_grad():
                for plane_idx in range(args_prop.num_planes):
                    if plane_idx < len(recon_intensities_g):
                        recon = odak.learn.tools.crop_center(
                            recon_intensities_g[plane_idx], size=(H, W)
                        )
                        recon = torch.clamp(recon, min=0.0, max=1.0)
                        plane_suffix = (
                            f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                        )
                        odak.learn.tools.save_image(
                            f"{result_dir}/recon_g_{itr:06d}{plane_suffix}.png",
                            recon,
                            cmin=0.0,
                            cmax=1.0,
                        )

                for plane_idx in range(args_prop.num_planes):
                    if plane_idx < len(recon_intensities_p):
                        recon = odak.learn.tools.crop_center(
                            recon_intensities_p[plane_idx], size=(H, W)
                        )
                        recon = torch.clamp(recon, min=0.0, max=1.0)
                        plane_suffix = (
                            f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                        )
                        odak.learn.tools.save_image(
                            f"{result_dir}/recon_p_{itr:06d}{plane_suffix}.png",
                            recon,
                            cmin=0.0,
                            cmax=1.0,
                        )

                phase_g_cropped = odak.learn.tools.crop_center(
                    phase_g.squeeze(0), size=(H, W)
                )
                phase_p_cropped = odak.learn.tools.crop_center(
                    phase_p.squeeze(0), size=(H, W)
                )
                amplitude_g_cropped = odak.learn.tools.crop_center(
                    amplitude_g.squeeze(0), size=(H, W)
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/phase_g_{itr:06d}.png",
                    phase_g_cropped,
                    cmin=0.0,
                    cmax=2 * odak.pi,
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/phase_p_{itr:06d}.png",
                    phase_p_cropped,
                    cmin=0.0,
                    cmax=2 * odak.pi,
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/amp_g_{itr:06d}.png",
                    amplitude_g_cropped,
                    cmin=0.0,
                    cmax=1.0,
                )
                # fourier_intensity_g_log = torch.log10(fourier_intensity_g + 1e-8)
                # fourier_intensity_g_normalized = (fourier_intensity_g_log - fourier_intensity_g_log.min()) / (fourier_intensity_g_log.max() - fourier_intensity_g_log.min() + 1e-8)
                # odak.learn.tools.save_image(
                #     f"{result_dir}/fourier_g_{itr:06d}.png",
                #     fourier_intensity_g_normalized, cmin=0., cmax=1.0
                # )
                # fourier_intensity_p_log = torch.log10(fourier_intensity_p + 1e-8)
                # fourier_intensity_p_normalized = (fourier_intensity_p_log - fourier_intensity_p_log.min()) / (fourier_intensity_p_log.max() - fourier_intensity_p_log.min() + 1e-8)
                # odak.learn.tools.save_image(
                #     f"{result_dir}/fourier_p_{itr:06d}.png",
                #     fourier_intensity_p_normalized, cmin=0., cmax=1.0
                # )

                if args.viz_freq == 1:
                    primary_plane_idx = 0
                    recon = recon_intensities_p[primary_plane_idx]
                    recon = torch.clamp(recon, min=0.0, max=1.0)
                    recon = odak.learn.tools.crop_center(recon, size=(H, W))
                    target = targets[primary_plane_idx]
                    psnr_val = calculate_psnr(recon, target)
                    viz_psnr_list.append(psnr_val)
                    viz_iterations.append(itr)

                # Save eyebox + sub-pupil recon visualization for POH
                if args.pupil_aware:
                    for plane_idx in range(args_prop.num_planes):
                        cf_rgb = torch.stack(
                            [complex_p[c, plane_idx, c] for c in range(3)], dim=0
                        )  # (3, Ny, Nx)
                        plane_suffix = (
                            f"_{plane_idx+1}" if args_prop.num_planes > 1 else ""
                        )
                        eyebox_with_subpupil(
                            cf_rgb,
                            aperture_size=args_prop.aperture_size,
                            img_size=(H, W),
                            out_path=f"{result_dir}/eyebox_p_{itr:06d}{plane_suffix}.png",
                            pupil_rad=args.pupil_rad,
                            scan_radius=args.scan_radius,
                            pupil_rolloff=args.pupil_rolloff,
                        )

        if (itr % args.eval_freq == 0 and itr > 0) or (itr == args.num_itrs - 1):
            with torch.no_grad():
                print(f"\nEvaluation at iteration {itr}:")

                all_psnr_g = []
                all_ssim_g = []
                all_lpips_g = []
                for plane_idx in range(min(len(recon_intensities_g), len(targets))):
                    recon = odak.learn.tools.crop_center(
                        recon_intensities_g[plane_idx], size=(H, W)
                    )
                    recon = torch.clamp(recon, min=0.0, max=1.0)
                    target = targets[plane_idx]

                    recon_np = (
                        recon.permute(1, 2, 0).detach().cpu().numpy()
                        if recon.dim() == 3
                        else recon.detach().cpu().numpy()
                    )
                    target_np = (
                        target.permute(1, 2, 0).detach().cpu().numpy()
                        if target.dim() == 3
                        else target.detach().cpu().numpy()
                    )
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

                    all_psnr_g.append(psnr_val)
                    all_ssim_g.append(ssim_val)
                    all_lpips_g.append(lpips_val)

                all_psnr_p = []
                all_ssim_p = []
                all_lpips_p = []
                for plane_idx in range(min(len(recon_intensities_p), len(targets))):
                    recon = odak.learn.tools.crop_center(
                        recon_intensities_p[plane_idx], size=(H, W)
                    )
                    recon = torch.clamp(recon, min=0.0, max=1.0)
                    target = targets[plane_idx]

                    recon_np = (
                        recon.permute(1, 2, 0).detach().cpu().numpy()
                        if recon.dim() == 3
                        else recon.detach().cpu().numpy()
                    )
                    target_np = (
                        target.permute(1, 2, 0).detach().cpu().numpy()
                        if target.dim() == 3
                        else target.detach().cpu().numpy()
                    )
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

                    all_psnr_p.append(psnr_val)
                    all_ssim_p.append(ssim_val)
                    all_lpips_p.append(lpips_val)

                print("Gaussian hologram:")
                for plane_idx in range(len(all_psnr_g)):
                    print(
                        f"  Plane {plane_idx}: PSNR: {all_psnr_g[plane_idx]:.3f}, SSIM: {all_ssim_g[plane_idx]:.3f}, LPIPS: {all_lpips_g[plane_idx]:.3f}"
                    )
                mean_psnr_g = sum(all_psnr_g) / len(all_psnr_g)
                mean_ssim_g = sum(all_ssim_g) / len(all_ssim_g)
                mean_lpips_g = sum(all_lpips_g) / len(all_lpips_g)
                print(
                    f"  Mean: PSNR: {mean_psnr_g:.3f}, SSIM: {mean_ssim_g:.3f}, LPIPS: {mean_lpips_g:.3f}"
                )

                print("Numerical phase hologram:")
                for plane_idx in range(len(all_psnr_p)):
                    print(
                        f"  Plane {plane_idx}: PSNR: {all_psnr_p[plane_idx]:.3f}, SSIM: {all_ssim_p[plane_idx]:.3f}, LPIPS: {all_lpips_p[plane_idx]:.3f}"
                    )
                mean_psnr_p = sum(all_psnr_p) / len(all_psnr_p)
                mean_ssim_p = sum(all_ssim_p) / len(all_ssim_p)
                mean_lpips_p = sum(all_lpips_p) / len(all_lpips_p)
                print(
                    f"  Mean: PSNR: {mean_psnr_p:.3f}, SSIM: {mean_ssim_p:.3f}, LPIPS: {mean_lpips_p:.3f}"
                )

                primary_plane_idx = (
                    min(1, args_prop.num_planes - 1) if args_prop.num_planes > 1 else 0
                )
                primary_psnr = (
                    all_psnr_g[primary_plane_idx]
                    if len(all_psnr_g) > primary_plane_idx
                    else all_psnr_g[0]
                )

                if primary_psnr > best_psnr:
                    best_psnr = primary_psnr
                    best_model_path = os.path.join(
                        checkpoint_dir, f"best_gaussians_parallel_{itr}.pth"
                    )
                    gaussians.save_gaussians(best_model_path)
                    torch.save(
                        numerical_phase,
                        os.path.join(checkpoint_dir, f"best_numerical_phase_{itr}.pth"),
                    )
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")
                sys.stdout.flush()
                del (
                    recon_intensities_g,
                    recon_intensities_p,
                    recon_g,
                    recon_p,
                    complex_g,
                    complex_p,
                    recon,
                    target,
                    recon_np,
                    target_np,
                )
                gc.collect()
                torch.cuda.empty_cache()

    final_model_path = os.path.join(
        checkpoint_dir, f"final_gaussians_parallel_{args.num_itrs}.pth"
    )
    if args.viz_freq == 1:
        torch.save(
            {"iterations": viz_iterations, "psnr": viz_psnr_list},
            f"{result_dir}/viz_metrics.pth",
        )
    gaussians.save_gaussians(final_model_path)
    torch.save(
        numerical_phase,
        os.path.join(checkpoint_dir, f"final_numerical_phase_{args.num_itrs}.pth"),
    )
    print("[*] Training Completed.")

    # ---- Pupil scanning GIF (eyebox visualization) ----
    if args.pupil_aware:
        with torch.no_grad():
            phase_p_final = numerical_phase % (2 * odak.pi)
            propagator.use_cuda_blasm_gaussian_version = False
            _, complex_p_final = propagator.reconstruct(
                phase_p_final, amplitude=None, no_grad=True, get_complex=True
            )
            # complex_p_final: (num_frames, num_depth, num_channels, Ny, Nx)
            for plane_idx in range(args_prop.num_planes):
                cf_rgb = torch.stack(
                    [complex_p_final[c, plane_idx, c] for c in range(3)], dim=0
                )
                plane_suffix = f"_{plane_idx + 1}" if args_prop.num_planes > 1 else ""
                pupil_scan_gif(
                    cf_rgb,
                    aperture_size=args_prop.aperture_size,
                    out_path=f"{result_dir}/pupil_scan{plane_suffix}.gif",
                    img_size=(H, W),
                    pupil_rad=args.pupil_rad,
                    scan_radius=args.scan_radius,
                    n_frames=args.pupil_n_frames,
                    pupil_rolloff=args.pupil_rolloff,
                )

    if log_debug:
        sys.stdout.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_image_path", default="./images/071.png", type=str)
    parser.add_argument("--depth_path", default="./images/d_071.png", type=str)
    parser.add_argument("--compression_ratio", default=0.2, type=float)
    parser.add_argument("--num_itrs", default=2001, type=int)
    parser.add_argument("--viz_freq", default=200, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument(
        "--lr", default=0.01, type=float, help="lr for complex gaussians"
    )
    parser.add_argument(
        "--lr_p", default=0.025, type=float, help="lr for numerical_phase"
    )
    parser.add_argument("--img_size", nargs=2, default=[1024, 640], type=int)
    parser.add_argument("--split_ratio", default=1.0, type=float)
    parser.add_argument(
        "--recon_weight",
        default=0.1,
        type=float,
        help="Weight for MSE loss between reconstructions",
    )
    parser.add_argument(
        "--complex_weight",
        default=0.01,
        type=float,
        help="Weight for L1 loss between complex fields",
    )
    parser.add_argument(
        "--gaussian_weights_path",
        default="",
        type=str,
        help="Path to pre-trained Gaussian weights (.pth file)",
    )
    # parser.add_argument("--gaussian_weights_path", default="/hy-tmp/GaussianImage_Holo/result_2d/checkpoints/final_gaussians_2d_full-complex_2001_1024_640_dragon.pth", type=str, help="Path to pre-trained Gaussian weights (.pth file)")
    # parser.add_argument("--gaussian_weights_path", default="/hy-tmp/GaussianImage_Holo/result_2d/checkpoints/final_gaussians_2d_full-complex_2001_1024_640_flower.pth", type=str, help="Path to pre-trained Gaussian weights (.pth file)")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    # Pupil-aware supervision
    parser.add_argument(
        "--pupil_aware",
        action="store_true",
        default=False,
        help="Enable pupil-aware supervision on the POH numerical phase hologram.",
    )
    parser.add_argument(
        "--pupil_ratio",
        nargs="+",
        default=[0.3, 0.65],
        type=float,
        help="Pupil radius range in normalized coords [-1,1]. Two values = [min, max].",
    )
    parser.add_argument(
        "--lambda_pupil",
        default=2.0,
        type=float,
        help="Weight for pupil-aware recon loss.",
    )
    parser.add_argument(
        "--lambda_dc",
        default=1e-5,
        type=float,
        help="Weight for DC suppression loss on the eyebox. Penalizes energy concentration at the center.",
    )
    parser.add_argument(
        "--num_pupils",
        default=2,
        type=int,
        help="Number of random pupil positions to average per iteration. More = more uniform eyebox.",
    )
    # Pupil scanning GIF params
    parser.add_argument(
        "--pupil_rad",
        default=0.65,
        type=float,
        help="Pupil radius for scan GIF (1.0 = full aperture).",
    )
    parser.add_argument(
        "--scan_radius",
        default=0.3,
        type=float,
        help="Orbit radius for pupil scan circle.",
    )
    parser.add_argument(
        "--pupil_n_frames",
        default=12,
        type=int,
        help="Number of frames in the pupil scan GIF.",
    )
    parser.add_argument(
        "--pupil_rolloff",
        default=0.15,
        type=float,
        help="Soft-edge rolloff at pupil boundary.",
    )
    args = parser.parse_args()

    args.load_gaussians = os.path.exists(args.gaussian_weights_path)
    if args.load_gaussians:
        print(
            f"Gaussian weights file found at {args.gaussian_weights_path}. Will load and freeze Gaussians."
        )
    else:
        print(
            f"Gaussian weights file not found at {args.gaussian_weights_path}. Will train from scratch."
        )

    return args


if __name__ == "__main__":
    args = get_args()
    args.num_gaussians = int(
        ((args.img_size[0] * args.img_size[1] * 6) / 12) * args.compression_ratio
    )
    args_prop = Namespace(
        wavelengths=[639e-9, 532e-9, 473e-9],
        pixel_pitch=3.74e-6,
        volume_depth=4e-3,
        d_val=3e-3,
        # pad_size=[1920, 1920],
        # aperture_size=1920,
        pad_size=[max(args.img_size), max(args.img_size)],
        aperture_size=int(
            sum(args.img_size) / 2.0 / (1.4 if args.pupil_aware else 1.0)
        ),
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

    run_training_parallel(args, args_prop)
