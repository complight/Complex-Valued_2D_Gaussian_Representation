"""Pupil scanning GIF utility for holographic eyebox visualization."""

import math

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pupil_scan_gif(
    complex_fields,
    aperture_size,
    out_path,
    img_size,
    pupil_rad=0.5,
    scan_radius=0.3,
    n_frames=12,
    fps=4,
    pupil_rolloff=0.15,
):
    """Generate a GIF of pupil scanning around the eyebox.

    Follows the same approach as holographic-parallax/prop_models.py:view_from_pupil():
    pad field to 2×, FT to eyebox, apply shifted pupil mask, IFT back.
    Supports time-multiplexing: averages |recon|² across TM frames then takes sqrt
    (incoherent RMS, same as holographic-parallax).

    Left panel: eyebox amplitude with red pupil ring.
    Right panel: reconstruction seen through that pupil position.

    Args:
        complex_fields: (TM, 3, H, W) or (3, H, W) complex tensor — propagated fields.
        aperture_size: int — aperture radius in pixels (on the doubled/padded grid).
        out_path: str — output GIF path.
        img_size: (H_img, W_img) — original image size for cropping.
        pupil_rad: float — pupil radius in normalized coords (1.0 = full aperture).
        scan_radius: float — orbit radius in normalized coords.
        n_frames: int — number of frames in the GIF.
        fps: int — frames per second.
        pupil_rolloff: float — width of the soft cosine rolloff at the pupil edge,
            as a fraction of the pupil radius. 0 = hard edge, 0.15 = 15% taper.
            Apodization suppresses ringing artifacts from the hard pupil boundary.
    """
    # Handle single frame vs TM
    if complex_fields.dim() == 3:
        complex_fields = complex_fields.unsqueeze(0)  # (1, 3, H, W)
    TM, C, ny, nx = complex_fields.shape
    device = complex_fields.device

    # Pad each TM frame
    eyebox_fields = []
    for t in range(TM):
        pad_field = F.pad(
            complex_fields[t],
            (nx // 2, nx // 2, ny // 2, ny // 2),
            mode="constant",
            value=0,
        )
        ef = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(pad_field, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )
        eyebox_fields.append(ef)
    eyebox_fields = torch.stack(eyebox_fields, dim=0)  # (TM, 3, 2*ny, 2*nx)
    pny, pnx = eyebox_fields.shape[-2:]

    # Eyebox for visualization: TM-averaged amplitude (RMS)
    eyebox_amp = (
        (eyebox_fields.abs().square().mean(dim=0)).sqrt().float()
    )  # (3, pny, pnx)
    for c in range(C):
        eyebox_amp[c] = eyebox_amp[c] / (eyebox_amp[c].max() + 1e-10)
        # Gamma correction to reveal aperture circle structure
        # (without this, DC energy dominates and everything else looks black)
        eyebox_amp[c] = eyebox_amp[c] ** 0.3
    eyebox_np = eyebox_amp.cpu().permute(1, 2, 0).numpy()

    # Normalized coordinates
    ix = torch.linspace(-pnx / 2, pnx / 2, pnx, device=device)
    iy = torch.linspace(-pny / 2, pny / 2, pny, device=device)
    YY, XX = torch.meshgrid(iy, ix, indexing="ij")
    XX = XX / XX.abs().max()
    YY = YY / YY.abs().max()

    aperture_norm = aperture_size / max(ny, nx)

    frames = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        pcy = scan_radius * aperture_norm * math.sin(angle)
        pcx = scan_radius * aperture_norm * math.cos(angle)

        dist2 = (YY - pcy) ** 2 + (XX - pcx) ** 2
        pr = pupil_rad * aperture_norm
        # Soft-edge pupil (apodization): cosine rolloff at the boundary.
        r = torch.sqrt(dist2) / pr
        if pupil_rolloff > 0:
            t_val = torch.clamp((1.0 - r) / pupil_rolloff, 0.0, 1.0)
            pupil_mask = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - t_val)))
        else:
            pupil_mask = (r < 1.0).float()

        # For each TM frame: apply pupil → IFFT → |recon|²
        # Then average across TM frames (incoherent sum) and take sqrt
        recon_intensities = []
        for t in range(TM):
            filtered = eyebox_fields[t] * pupil_mask.unsqueeze(0)  # (C, pny, pnx)
            recon = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(filtered, dim=(-2, -1)),
                    dim=(-2, -1),
                    norm="ortho",
                ),
                dim=(-2, -1),
            )
            recon = recon[:, ny // 2 : ny // 2 + ny, nx // 2 : nx // 2 + nx]
            recon_intensities.append(recon.abs().square())

        # TM average intensity: (|E_1|² + |E_2|² + ...)/TM
        recon_intensity = torch.stack(recon_intensities, dim=0).mean(dim=0)

        crop_y = (ny - img_size[0]) // 2
        crop_x = (nx - img_size[1]) // 2
        recon_cropped = recon_intensity[
            :, crop_y : crop_y + img_size[0], crop_x : crop_x + img_size[1]
        ]
        recon_np = recon_cropped.cpu().permute(1, 2, 0).numpy()
        recon_np = np.clip(recon_np, 0, 1)

        # === Build side-by-side frame ===
        H_img, W_img = img_size

        # Left: eyebox resized + red pupil circle
        eyebox_resized = np.array(
            Image.fromarray((np.clip(eyebox_np, 0, 1) * 255).astype(np.uint8)).resize(
                (W_img, H_img), Image.BILINEAR
            )
        )
        # Map pupil coords to resized image coords
        # normalized [-1,1] → pixel [0, W_img-1]
        circle_cx = int((pcx + 1) / 2 * W_img)
        circle_cy = int((pcy + 1) / 2 * H_img)
        circle_r = int(pr / 2 * max(W_img, H_img))

        # Draw red circle
        for theta_idx in range(720):
            th = 2 * math.pi * theta_idx / 720
            for dr in range(-1, 2):
                ry = int(circle_cy + (circle_r + dr) * math.sin(th))
                rx = int(circle_cx + (circle_r + dr) * math.cos(th))
                if 0 <= ry < H_img and 0 <= rx < W_img:
                    eyebox_resized[ry, rx] = [255, 0, 0]

        # Right: reconstruction
        recon_rgb = (np.clip(recon_np, 0, 1) * 255).astype(np.uint8)

        frame = np.concatenate([eyebox_resized, recon_rgb], axis=1)
        frames.append(frame)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"[pupil scan] saved {n_frames}-frame GIF to {out_path}")


def eyebox_with_subpupil(
    complex_field,
    aperture_size,
    img_size,
    out_path,
    pupil_rad=0.5,
    scan_radius=0.3,
    pupil_rolloff=0.15,
):
    """Save a side-by-side PNG: eyebox amplitude (left) + sub-pupil reconstruction (right).

    Uses a single pupil position at angle=0 (rightward shift).

    Args:
        complex_field: (3, H, W) complex tensor — propagated field for one plane.
        aperture_size: int — aperture radius in pixels.
        img_size: (H_img, W_img) — original image size for cropping.
        out_path: str — output PNG path.
        pupil_rad: float — pupil radius in normalized coords.
        scan_radius: float — pupil offset in normalized coords.
        pupil_rolloff: float — soft edge rolloff fraction.
    """
    C, ny, nx = complex_field.shape
    device = complex_field.device

    pad_cf = F.pad(complex_field, (nx // 2, nx // 2, ny // 2, ny // 2))
    eyebox = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(pad_cf, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )

    # Eyebox amplitude with gamma correction
    eyebox_amp = eyebox.abs().float()
    for c in range(C):
        eyebox_amp[c] = eyebox_amp[c] / (eyebox_amp[c].max() + 1e-10)
        eyebox_amp[c] = eyebox_amp[c] ** 0.3

    # Sub-pupil reconstruction at angle=0
    pny, pnx = eyebox.shape[-2:]
    ix = torch.linspace(-pnx / 2, pnx / 2, pnx, device=device)
    iy = torch.linspace(-pny / 2, pny / 2, pny, device=device)
    YY, XX = torch.meshgrid(iy, ix, indexing="ij")
    XX = XX / XX.abs().max()
    YY = YY / YY.abs().max()

    aperture_norm = aperture_size / max(ny, nx)
    pcx = scan_radius * aperture_norm
    pcy = 0.0
    pr = pupil_rad * aperture_norm
    r = torch.sqrt((YY - pcy) ** 2 + (XX - pcx) ** 2) / pr

    if pupil_rolloff > 0:
        t_val = torch.clamp((1.0 - r) / pupil_rolloff, 0.0, 1.0)
        pupil_mask = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - t_val)))
    else:
        pupil_mask = (r < 1.0).float()

    filtered = eyebox * pupil_mask.unsqueeze(0)
    recon_sub = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(filtered, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )
    recon_sub = recon_sub[:, ny // 2 : ny // 2 + ny, nx // 2 : nx // 2 + nx]
    recon_sub = recon_sub.abs().square()

    H_img, W_img = img_size
    crop_y = (ny - H_img) // 2
    crop_x = (nx - W_img) // 2
    recon_sub = recon_sub[:, crop_y : crop_y + H_img, crop_x : crop_x + W_img]
    recon_sub = torch.clamp(recon_sub, 0.0, 1.0)

    # Build side-by-side image
    eyebox_np = eyebox_amp.cpu().permute(1, 2, 0).numpy()
    eyebox_resized = np.array(
        Image.fromarray((np.clip(eyebox_np, 0, 1) * 255).astype(np.uint8)).resize(
            (W_img, H_img), Image.BILINEAR
        )
    )

    recon_rgb = (recon_sub.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    side_by_side = np.concatenate([eyebox_resized, recon_rgb], axis=1)
    Image.fromarray(side_by_side).save(out_path)
