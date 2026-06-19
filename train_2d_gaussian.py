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
    complex_to_phase_encode,
    console_only_print,
    GaussianLoss,
    multiplane_loss,
    propagator,
    set_seed,
    visualize_gaussian_positions,
)

log_debug = True

set_seed(100)


def load_target_image(image_path, img_size):
    """Load and preprocess target image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size, Image.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    img_array = np.array(img) / 255.0
    target_tensor = torch.from_numpy(img_array).float()
    target_tensor = target_tensor.permute(2, 0, 1)  # (C, H, W)

    print(f"Loaded target image: {image_path}, size: {img_size}")
    return target_tensor


def load_depth_image(depth_path, img_size):
    """Load and preprocess depth image"""
    if not os.path.exists(depth_path):
        return None

    # Load depth image
    depth_img = Image.open(depth_path).convert("L")  # Convert to grayscale
    depth_img = depth_img.resize(img_size, Image.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    depth_array = np.array(depth_img) / 255.0
    depth_tensor = torch.from_numpy(depth_array).float()
    depth_tensor = depth_tensor.unsqueeze(0)  # (1, H, W)

    print(f"Loaded depth image: {depth_path}, size: {img_size}")
    return depth_tensor


def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target tensors."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np)


def setup_optimizer_simplified(
    gaussians,
    num_itrs,
    lr=0.01,
    current_iter=0,
    prev_optimizer=None,
    prev_scheduler=None,
):
    """
    Simplified optimizer setup without plane assignment parameters
    """
    g = gaussians.module if hasattr(gaussians, "module") else gaussians

    # Parameter groups without plane assignment
    param_groups = [
        {
            "params": [gaussians.means_2d],
            "lr": lr,  # This will be subject to cosine annealing
            "name": "means",
        },
        {"params": [gaussians.pre_act_scales], "lr": 0.005, "name": "scales"},
        {
            "params": [gaussians.colours, gaussians.pre_act_phase],
            "lr": 0.0025,
            "name": "amplitude_phase",
        },
        {"params": [gaussians.pre_act_rotation], "lr": 0.001, "name": "rotation"},
    ]

    if not g.merge_opacity:
        param_groups.append(
            {"params": [gaussians.pre_act_opacities], "lr": 0.025, "name": "opacity"}
        )

    # Create new optimizer
    optimizer = Adan(param_groups, lr=0, eps=1e-8)

    # Extract all trainable parameters from parameter groups for gradient clipping
    trainable_params = []
    for param_group in param_groups:
        trainable_params.extend(param_group["params"])

    # Transfer state from previous optimizer if available
    if prev_optimizer is not None:
        # Extract parameter names from previous optimizer
        prev_param_groups = {
            pg["name"]: i
            for i, pg in enumerate(prev_optimizer.param_groups)
            if "name" in pg
        }

        # For each parameter group in the new optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            if "name" in param_group and param_group["name"] in prev_param_groups:
                # Get the corresponding parameter group from the previous optimizer
                prev_group_idx = prev_param_groups[param_group["name"]]
                prev_group = prev_optimizer.param_groups[prev_group_idx]

                # Transfer learning rate and other hyperparameters
                param_group["lr"] = prev_group["lr"]

                # For each parameter in this group
                for param in param_group["params"]:
                    param_state = {}

                    # Find corresponding parameter in previous optimizer
                    found = False
                    for prev_param in prev_group["params"]:
                        if prev_param in prev_optimizer.state:
                            # Transfer state (momentum, etc.)
                            try:
                                # Copy state for parameters with matching shapes
                                if param.shape == prev_param.shape:
                                    param_state = prev_optimizer.state[prev_param]
                                    found = True
                                    break
                            except:
                                # If shapes don't match, we can't transfer state
                                pass

                    # If we found matching state, add to the new optimizer's state
                    if found and param_state:
                        optimizer.state[param] = param_state

    # Create a custom scheduler that only updates specific parameter groups
    remaining_iters = max(1, num_itrs - current_iter)

    # Find the indices of the parameter groups we want to apply the scheduler to
    decay_group_indices = []
    decay_params_list = []
    decay_group_names = ["means"]  # Only means will have LR decay

    for i, group in enumerate(optimizer.param_groups):
        if group.get("name") in decay_group_names:
            decay_group_indices.append(i)
            decay_params_list.append({"params": group["params"], "lr": group["lr"]})

    # Create a cosine annealing scheduler only for selected parameters
    temp_optimizer = torch.optim.SGD(decay_params_list)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        temp_optimizer, T_max=remaining_iters, eta_min=1e-3
    )

    # If we have a previous scheduler, try to restore its state
    if prev_scheduler is not None:
        try:
            decay_scheduler.last_epoch = prev_scheduler.last_epoch
            if hasattr(prev_scheduler, "_last_lr"):
                decay_scheduler._last_lr = prev_scheduler._last_lr
        except:
            pass

    # Create a custom scheduler class that only updates specific learning rates
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
            # Step the decay scheduler
            self.decay_scheduler.step()

            # Update only the specified parameter groups' learning rates
            for i, group_idx in enumerate(self.decay_group_indices):
                self.optimizer.param_groups[group_idx][
                    "lr"
                ] = self.decay_scheduler.get_last_lr()[i]

            # Update scheduler state
            self.last_epoch = self.decay_scheduler.last_epoch
            self._last_lr = self.decay_scheduler.get_last_lr()

        def get_last_lr(self):
            return [group["lr"] for group in self.optimizer.param_groups]

    # Create our custom scheduler
    scheduler = CustomScheduler(optimizer, decay_scheduler, decay_group_indices)

    return optimizer, scheduler, trainable_params


def POH(phase_map, amplitude=None):
    phase_map = complex_to_phase_encode(amplitude, phase_map)
    phase_map = phase_map % (2 * odak.pi)
    amplitude = torch.ones_like(phase_map)
    return phase_map, amplitude


def run_training_2d(args, args_prop, result_dir, checkpoint_dir):

    # Image size
    img_size = tuple(args.img_size)

    # Initialize LPIPS metric
    lpips_fn = lpips.LPIPS(net="vgg").cuda()

    # Load target image
    target_image = load_target_image(args.target_image_path, img_size).cuda()

    C, H, W = target_image.shape
    print(f"Target image shape: {target_image.shape}")

    # Load depth image if available
    depth_image = (
        load_depth_image(args.depth_path, img_size) if args.depth_path else None
    )
    has_depth = depth_image is not None

    if has_depth:
        depth_image = depth_image.cuda()
        print(f"Depth image shape: {depth_image.shape}")
    else:
        # Create depth with all ones if no depth file
        depth_image = torch.ones((1, H, W), device="cuda")
        print("Using default depth (all ones)")

    print(f"Has depth: {has_depth}")
    print(f"Using hologram type: {args.hologram_type}")

    # Initialize 2D Gaussians (simplified - no plane assignment)
    gaussians = Gaussians2D(
        num_points=args.num_gaussians,
        img_size=img_size,
        device=args.device,
        args_prop=args_prop,
        merge_opacity=args.merge_opacity,
    )

    scene = Scene2D(gaussians, args_prop)
    make_trainable_2d(gaussians)

    # Setup simplified optimizer
    optimizer, scheduler, trainable_params = setup_optimizer_simplified(
        gaussians=gaussians,
        num_itrs=args.num_itrs,
        lr=args.lr,  # Base learning rate for means group
        current_iter=0,
        prev_optimizer=None,
        prev_scheduler=None,
    )
    # Setup multiplane targets
    targets, loss_function, mask = multiplane_loss(
        target_image=target_image, target_depth=depth_image, args_prop=args_prop
    )
    for idx, target in enumerate(targets):
        odak.learn.tools.save_image(f"{result_dir}/target_{idx}.png", target, cmin=0.0, cmax=1.0)

    # Training loop
    running_losses = []
    running_ssim_losses = []
    running_psnrs = []
    running_lrs = []  # Track learning rate changes
    best_psnr = 0.0
    viz_psnr_list = []
    viz_iterations = []

    pbar = tqdm(range(args.num_itrs), desc="Training 2D Gaussians")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for itr in pbar:
        optimizer.zero_grad()
        loss_ssim = GaussianLoss

        if not has_depth and args_prop.num_planes > 1:
            raise ValueError("Depth is required for supervision more than 1 plane.")

        # Render hologram directly
        starter.record()
        hologram_complex = scene.render(img_size)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"--forward {curr_time}")
        # Reconstruct using propagator
        starter.record()
        phase_map = odak.learn.wave.calculate_phase(hologram_complex)
        amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)

        if args.hologram_type == "phase-only":
            phase_map, amplitude = POH(phase_map, amplitude)
        else:
            phase_map = phase_map % (2 * odak.pi)
            amplitude = torch.clamp(amplitude, min=0.0, max=1.0)
        reconstruction_intensities_sum = propagator.reconstruct(
            phase_map, amplitude=amplitude, no_grad=False
        )
        reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)
        # # === BACKPROP SPEED TEST: Disable propagator ===
        # # Get output shape without computing gradients
        # with torch.no_grad():
        #     shape_template = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad=True)

        # # Create dummy tensor with minimal gradient connection to phase_map and amplitude
        # # This preserves gradient flow but avoids expensive propagator backprop
        # reconstruction_intensities_sum = torch.randn_like(shape_template) + \
        #     (phase_map.sum() + amplitude.sum()) * 1e-12  # Tiny connection to maintain gradient graph
        # reconstruction_intensities = torch.sum(reconstruction_intensities_sum, dim=0)
        # # === END BACKPROP SPEED TEST ===
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"recon {curr_time}")

        # Calculate loss using multiplane loss function - restored from original
        loss = 0.0
        starter.record()
        random.seed(int(time.time()))
        num_planes = len(reconstruction_intensities)
        selected_plane_idx = random.randint(0, num_planes - 1)
        for idx, (reconstruction_intensity, target) in enumerate(
            zip(reconstruction_intensities, targets)
        ):
            # if idx == selected_plane_idx:
            pred_cropped = odak.learn.tools.crop_center(
                reconstruction_intensity, size=(H, W)
            )
            pred_cropped = torch.clamp(pred_cropped, min=0.0, max=1.0)

            # Use multiplane loss function
            loss += loss_function(pred_cropped, target, idx, inject_noise=True)
            # loss += F.mse_loss(pred_cropped, target_image)
            # SSIM loss
            ssim_loss = loss_ssim(pred_cropped, target)
            loss += ssim_loss

            if idx == 0:  # Calculate PSNR for first channel
                current_psnr = calculate_psnr(pred_cropped, target)
                running_psnrs.append(current_psnr)

        running_losses.append(loss.item())
        running_ssim_losses.append(ssim_loss.item())

        del reconstruction_intensities

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        with console_only_print():
            print(f"back {curr_time}")

        current_lr = optimizer.param_groups[0]["lr"]
        running_lrs.append(current_lr)

        # Calculate moving averages
        mean_loss = sum(running_losses[-50:]) / min(len(running_losses), 50)
        mean_ssim = sum(running_ssim_losses[-50:]) / min(len(running_ssim_losses), 50)
        mean_psnr = sum(running_psnrs[-50:]) / min(len(running_psnrs), 50)

        pbar.set_postfix(
            {
                "Loss": f"{mean_loss:.6f}",
                "PSNR": f"{mean_psnr:.2f}",
                "SSIM": f"{mean_ssim:.6f}",
                "LR": f"{current_lr:.2e}",
                "G": f"{len(gaussians)}",
                "Type": args.hologram_type,
            }
        )

        # Visualization
        if args.viz_freq != -1 and itr % args.viz_freq == 0:
            # if itr in [0, 10, 50, 200, 500, 1000, 1500, 2000]:
            # Save visualization
            with torch.no_grad():
                hologram_complex = scene.render(img_size)
                phase_map = odak.learn.wave.calculate_phase(hologram_complex)
                amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)

                if args.hologram_type == "phase-only":
                    phase_map, amplitude = POH(phase_map, amplitude)
                else:
                    phase_map = phase_map % (2 * odak.pi)
                    amplitude = torch.clamp(amplitude, min=0.0, max=1.0)
                reconstruction_intensities_sum = propagator.reconstruct(
                    phase_map, amplitude=amplitude, no_grad=False
                )
                reconstruction_intensities = torch.sum(
                    reconstruction_intensities_sum, dim=0
                )

                # Save reconstruction for each plane
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

                # Save phase and amplitude
                phase_cropped = odak.learn.tools.crop_center(
                    phase_map.squeeze(0), size=(H, W)
                )
                amp_cropped = odak.learn.tools.crop_center(
                    amplitude.squeeze(0), size=(H, W)
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/phase_{itr:06d}.png",
                    phase_cropped,
                    cmin=0.0,
                    cmax=2 * odak.pi,
                )
                odak.learn.tools.save_image(
                    f"{result_dir}/amp_{itr:06d}.png", amp_cropped, cmin=0.0, cmax=1.0
                )
                visualize_gaussian_positions(
                    gaussians, gaussians, img_size, itr, result_dir
                )

                if args.viz_freq == 1:
                    primary_plane_idx = 0
                    recon = reconstruction_intensities[primary_plane_idx]
                    recon = torch.clamp(recon, min=0.0, max=1.0)
                    recon = odak.learn.tools.crop_center(recon, size=(H, W))
                    target = targets[primary_plane_idx]
                    psnr_val = calculate_psnr(recon, target)
                    viz_psnr_list.append(psnr_val)
                    viz_iterations.append(itr)

        # Evaluation
        if (itr % args.eval_freq == 0 and itr > 0) or (itr == args.num_itrs - 1):
            with torch.no_grad():
                # Prepare targets for evaluation (same as training)
                targets, _, _ = multiplane_loss(
                    target_image=target_image,
                    target_depth=depth_image,
                    args_prop=args_prop,
                )

                hologram_complex = scene.render(img_size)
                phase_map = odak.learn.wave.calculate_phase(hologram_complex)
                amplitude = odak.learn.wave.calculate_amplitude(hologram_complex)

                if args.hologram_type == "phase-only":
                    phase_map, amplitude = POH(phase_map, amplitude)
                else:
                    phase_map = phase_map % (2 * odak.pi)
                    amplitude = torch.clamp(amplitude, min=0.0, max=1.0)
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

                    # Convert to numpy and transpose to (H, W, C) for skimage
                    if recon.dim() == 3:  # (C, H, W) -> (H, W, C)
                        recon_np = recon.permute(1, 2, 0).detach().cpu().numpy()
                    else:  # Single channel
                        recon_np = recon.detach().cpu().numpy()

                    if target.dim() == 3:  # (C, H, W) -> (H, W, C)
                        target_np = target.permute(1, 2, 0).detach().cpu().numpy()
                    else:  # Single channel
                        target_np = target.detach().cpu().numpy()

                    # Calculate PSNR and SSIM
                    psnr_val = peak_signal_noise_ratio(
                        target_np, recon_np, data_range=1.0
                    )
                    ssim_val = structural_similarity(
                        target_np,
                        recon_np,
                        data_range=1.0,
                        channel_axis=2 if recon_np.ndim == 3 else None,
                    )

                    # LPIPS - ensure 3 channels and batch dimension
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

                # Use primary plane for contrast calculations and model saving
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

                # Calculate Weber contrast
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

                # Calculate Michelson contrast
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

                # Save model
                primary_psnr = (
                    all_psnr_vals[primary_plane_idx]
                    if len(all_psnr_vals) > primary_plane_idx
                    else all_psnr_vals[0]
                )
                if primary_psnr > best_psnr:
                    best_psnr = primary_psnr
                    best_model_path = os.path.join(
                        checkpoint_dir,
                        f"best_gaussians_2d_{args.hologram_type}_{itr}.pth",
                    )
                    gaussians.save_gaussians(best_model_path)
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")
                else:
                    latest_model_path = os.path.join(
                        checkpoint_dir,
                        f"latest_gaussians_2d_{args.hologram_type}_{itr}.pth",
                    )
                    gaussians.save_gaussians(latest_model_path)
                sys.stdout.flush()
                del all_psnr_vals, all_ssim_vals, all_lpips_vals
                del (
                    reconstruction_intensities,
                    reconstruction_intensities_sum,
                    recon,
                    target,
                    recon_np,
                    target_np,
                )
                gc.collect()
                torch.cuda.empty_cache()

    # Final save
    final_model_path = os.path.join(
        checkpoint_dir, f"final_gaussians_2d_{args.hologram_type}_{args.num_itrs}.pth"
    )
    if args.viz_freq == 1:
        torch.save(
            {"iterations": viz_iterations, "psnr": viz_psnr_list},
            f"{result_dir}/viz_metrics.pth",
        )
    gaussians.save_gaussians(final_model_path)
    print("[*] Training Completed.")

    if log_debug:
        sys.stdout.close()


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
        "--hologram_type",
        default="full-complex",
        type=str,
        choices=["full-complex", "phase-only"],
        help="Hologram type: 'full-complex' optimizes both amplitude and phase, 'phase-only' optimizes only phase with fixed amplitude",
    )
    parser.add_argument(
        "--compression_ratio",
        default=0.5,
        type=float,
        help="Number of 2D Gaussians to initialize",
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
    parser.add_argument("--lr", default=0.01, type=float, help="Learning Rate")
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
    parser.add_argument(
        "--overwrite_saving",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="If True, use default result_2d folder. If False, create unique folder with image name, resolution, and plane number",
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    parser.add_argument(
        "--merge_opacity",
        action="store_true",
        default=False,
        help="Merge opacity into per-channel color (ablation: remove separate alpha)",
    )

    args = parser.parse_args()
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
        # pad_size=[2560, 1440][::-1],
        # aperture_size=1920,
        pad_size=[max(args.img_size), max(args.img_size)],
        aperture_size=int(sum(args.img_size) / 2.0),
        num_planes=2,
        split_ratio=args.split_ratio,
    )

    # Setup distances
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

    # Create result directory based on overwrite_saving parameter
    if args.overwrite_saving:
        result_dir = os.path.join("./result_2d")
    else:
        image_name = os.path.splitext(os.path.basename(args.target_image_path))[0]
        resolution = f"{args.img_size[0]}x{args.img_size[1]}"
        plane_number = args_prop.num_planes
        result_dir = os.path.join(
            f"./result_2d/{image_name}_{resolution}_plane_num_{plane_number}"
        )

    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if log_debug:
        sys.stdout = open(os.path.join(result_dir, "log.txt"), "w")

    print("Distance: ", args_prop.distances)
    print(f"Using hologram type: {args.hologram_type}")
    print(f"Result directory: {result_dir}")

    # Initialize propagator
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
    propagator.use_cuda_blasm_gaussian_version = True
    if args.hologram_type == "phase-only":
        propagator.use_cuda_blasm_gaussian_version = False
    run_training_2d(args, args_prop, result_dir, checkpoint_dir)
