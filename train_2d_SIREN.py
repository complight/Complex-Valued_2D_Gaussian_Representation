import os
import sys
from argparse import ArgumentParser, Namespace

import lpips
import numpy as np
import odak
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from utils import count_param, propagator

result_dir = os.path.join("./result_siren")
os.makedirs(result_dir, exist_ok=True)


def get_mgrid(sidelen, dim=2):
    tensors = [torch.linspace(-1, 1, steps=sidelen[i]) for i in range(dim)]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        output_activation=None,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)
        self.output_activation = output_activation

    def forward(self, coords):
        output = self.net(coords)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


def positional_encoding(coords, L=10):
    encoded = [coords]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encoded.append(fn(2.0**i * np.pi * coords))
    return torch.cat(encoded, dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        output_activation=None,
        pos_enc_L=10,
    ):
        super().__init__()

        self.pos_enc_L = pos_enc_L
        encoded_dim = in_features * (2 * pos_enc_L + 1)

        layers = []
        layers.append(nn.Linear(encoded_dim, hidden_features))
        layers.append(nn.ReLU())

        for i in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, coords):
        encoded_coords = positional_encoding(coords, self.pos_enc_L)
        output = self.net(encoded_coords)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


def load_target_image(image_path, img_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    H, W = img_size
    img = img.resize((W, H), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    target_tensor = torch.from_numpy(img_array).float()
    target_tensor = target_tensor.permute(2, 0, 1)

    print(f"Loaded target image: {image_path}, size: (H={H}, W={W})")
    return target_tensor


def train_siren(
    target_image_path,
    img_size=(512, 768),
    num_itrs=2000,
    lr=1e-3,
    hidden_features=256,
    hidden_layers=3,
    viz_freq=200,
    tag="",
    is_phase=False,
    model_type="siren",
    pos_enc_L=10,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    target_image = load_target_image(target_image_path, img_size).to(device)
    C, H, W = target_image.shape
    print(f"Target image shape: {target_image.shape}")

    pixels = target_image.permute(1, 2, 0).reshape(-1, C)

    coords = get_mgrid([H, W], 2).to(device)

    output_activation = nn.Sigmoid() if is_phase else None

    if model_type == "siren":
        model = Siren(
            in_features=2,
            out_features=C,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            outermost_linear=True,
            output_activation=output_activation,
        ).to(device)
    elif model_type == "mlp":
        model = MLP(
            in_features=2,
            out_features=C,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            output_activation=output_activation,
            pos_enc_L=pos_enc_L,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'siren' or 'mlp'.")
    count_param(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_itrs, eta_min=0
    )

    pbar = tqdm(range(num_itrs), desc=f"Training {model_type.upper()} {tag}")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    for itr in pbar:
        optimizer.zero_grad()
        # starter.record()
        model_output = model(coords)
        # ender.record()
        # torch.cuda.synchronize()
        # curr_time = starter.elapsed_time(ender)
        # print(f"--forward {curr_time}")
        if is_phase:
            model_output_scaled = model_output * 2 * odak.pi
            pixels_scaled = pixels * 2 * odak.pi
            loss = ((model_output_scaled - pixels_scaled) ** 2).mean()
        else:
            loss = ((model_output - pixels) ** 2).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        psnr_current = -10 * torch.log10(loss).item()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.6f}",
                "PSNR": f"{psnr_current:.2f}dB",
                "LR": f"{current_lr:.6f}",
            }
        )

        if viz_freq != -1 and itr % viz_freq == 0:
            with torch.no_grad():
                recon = model_output.reshape(H, W, C).permute(2, 0, 1)
                recon = torch.clamp(recon, 0.0, 1.0)
                odak.learn.tools.save_image(
                    f"{result_dir}/recon_{tag}_{itr:06d}.png", recon, cmin=0.0, cmax=1.0
                )

    with torch.no_grad():
        model_output = model(coords)
        recon = model_output.reshape(H, W, C).permute(2, 0, 1)
        recon = torch.clamp(recon, 0.0, 1.0)

        recon_np = recon.permute(1, 2, 0).cpu().numpy()
        target_np = target_image.permute(1, 2, 0).cpu().numpy()

        psnr_val = peak_signal_noise_ratio(target_np, recon_np, data_range=1.0)
        ssim_val = structural_similarity(
            target_np, recon_np, data_range=1.0, channel_axis=2
        )

        recon_norm = 2 * recon.unsqueeze(0) - 1
        target_norm = 2 * target_image.unsqueeze(0) - 1
        lpips_val = lpips_fn(recon_norm, target_norm).item()

        odak.learn.tools.save_image(
            f"{result_dir}/final_recon_{tag}.png", recon, cmin=0.0, cmax=1.0
        )

        print(f"\n[*] Training {tag} Completed.")
        print(f"Final PSNR: {psnr_val:.3f} dB")
        print(f"Final SSIM: {ssim_val:.4f}")
        print(f"Final LPIPS: {lpips_val:.4f}")

    torch.save(model.state_dict(), f"{result_dir}/{model_type}_{tag}.pth")
    return model, psnr_val, ssim_val, lpips_val


def load_siren_output(model, img_size, device="cuda"):
    H, W = img_size
    coords = get_mgrid([H, W], 2).to(device)
    with torch.no_grad():
        model_output = model(coords)
        output = model_output.reshape(H, W, 3).permute(2, 0, 1)
    return output


def compare_reconstructions(
    gt_amp_path,
    gt_phase_path,
    siren_amp_model,
    siren_phase_model,
    img_size,
    prop,
    args_prop,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="vgg").cuda()

    H, W = img_size

    gt_amp = load_target_image(gt_amp_path, (H, W)).to(device)
    gt_phase = load_target_image(gt_phase_path, (H, W)).to(device)

    odak.learn.tools.save_image(
        f"{result_dir}/gt_amplitude.png", gt_amp, cmin=0.0, cmax=1.0
    )
    odak.learn.tools.save_image(
        f"{result_dir}/gt_phase.png", gt_phase, cmin=0.0, cmax=1.0
    )

    siren_amp = load_siren_output(siren_amp_model, img_size, device)
    siren_phase = load_siren_output(siren_phase_model, img_size, device)

    gt_phase_scaled = gt_phase * 2 * np.pi
    siren_phase_scaled = siren_phase * 2 * np.pi

    print("\n[*] Reconstructing with GT amplitude and phase...")
    gt_recon_sum = prop.reconstruct(gt_phase_scaled, amplitude=gt_amp, no_grad=False)
    gt_recon = torch.sum(gt_recon_sum, dim=0)

    print("[*] Reconstructing with SIREN amplitude and phase...")
    siren_recon_sum = prop.reconstruct(
        siren_phase_scaled, amplitude=siren_amp, no_grad=False
    )
    siren_recon = torch.sum(siren_recon_sum, dim=0)

    print(f"\n[*] Comparing reconstructions per plane (Total planes: {len(gt_recon)}):")
    for plane_idx in range(len(gt_recon)):
        gt_plane = torch.clamp(gt_recon[plane_idx], 0.0, 1.0)
        siren_plane = torch.clamp(siren_recon[plane_idx], 0.0, 1.0)

        gt_plane = odak.learn.tools.crop_center(gt_plane, size=(H, W))
        siren_plane = odak.learn.tools.crop_center(siren_plane, size=(H, W))

        gt_np = gt_plane.permute(1, 2, 0).cpu().numpy()
        siren_np = siren_plane.permute(1, 2, 0).cpu().numpy()

        psnr_val = peak_signal_noise_ratio(gt_np, siren_np, data_range=1.0)
        ssim_val = structural_similarity(
            gt_np, siren_np, data_range=1.0, channel_axis=2
        )

        gt_norm = 2 * gt_plane.unsqueeze(0) - 1
        siren_norm = 2 * siren_plane.unsqueeze(0) - 1
        lpips_val = lpips_fn(siren_norm, gt_norm).item()

        print(
            f"  Plane {plane_idx}: PSNR={psnr_val:.3f}dB, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}"
        )

        odak.learn.tools.save_image(
            f"{result_dir}/gt_recon_plane{plane_idx}.png", gt_plane, cmin=0.0, cmax=1.0
        )
        odak.learn.tools.save_image(
            f"{result_dir}/siren_recon_plane{plane_idx}.png",
            siren_plane,
            cmin=0.0,
            cmax=1.0,
        )

    sys.stdout.flush()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train SIREN/MLP on hologram amplitude and phase"
    )

    parser.add_argument(
        "--base_dir", type=str, default="./result_2d", help="Base directory for images"
    )
    parser.add_argument(
        "--amp_name",
        type=str,
        default="amp_blue_cat.png",
        help="Amplitude image filename",
    )
    parser.add_argument(
        "--phase_name",
        type=str,
        default="phase_blue_cat.png",
        help="Phase image filename",
    )
    parser.add_argument(
        "--img_size", type=int, nargs=2, default=[1024, 640], help="Image size (H W)"
    )

    parser.add_argument(
        "--num_itrs", type=int, default=2000, help="Number of training iterations"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--viz_freq",
        type=int,
        default=200,
        help="Visualization frequency (-1 to disable)",
    )

    parser.add_argument(
        "--hidden_features", type=int, default=256, help="Number of hidden features"
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=4, help="Number of hidden layers"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="siren",
        choices=["siren", "mlp"],
        help="Model type",
    )
    parser.add_argument(
        "--pos_enc_L", type=int, default=30, help="Positional encoding levels (for MLP)"
    )

    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs=3,
        default=[639e-9, 532e-9, 473e-9],
        help="Wavelengths (RGB)",
    )
    parser.add_argument(
        "--pixel_pitch", type=float, default=3.74e-6, help="Pixel pitch"
    )
    parser.add_argument("--volume_depth", type=float, default=4e-3, help="Volume depth")
    parser.add_argument("--d_val", type=float, default=3e-3, help="Distance value")
    parser.add_argument("--num_planes", type=int, default=2, help="Number of planes")

    args = parser.parse_args()

    amp_path = f"{args.base_dir}/{args.amp_name}"
    phase_path = f"{args.base_dir}/{args.phase_name}"
    img_size = tuple(args.img_size[::-1])

    print("\n" + "=" * 60)
    print("[*] Training SIREN on Phase")
    print("=" * 60)
    phase_model, phase_psnr, phase_ssim, phase_lpips = train_siren(
        target_image_path=phase_path,
        img_size=img_size,
        num_itrs=args.num_itrs,
        lr=args.lr,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        viz_freq=args.viz_freq,
        tag="phase",
        is_phase=True,
        model_type=args.model_type,
        pos_enc_L=args.pos_enc_L,
    )

    print("=" * 60)
    print("[*] Training SIREN on Amplitude")
    print("=" * 60)
    amp_model, amp_psnr, amp_ssim, amp_lpips = train_siren(
        target_image_path=amp_path,
        img_size=img_size,
        num_itrs=args.num_itrs,
        lr=args.lr,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        viz_freq=args.viz_freq,
        tag="amplitude",
        is_phase=False,
        model_type=args.model_type,
        pos_enc_L=args.pos_enc_L,
    )

    args_prop = Namespace(
        wavelengths=args.wavelengths,
        pixel_pitch=args.pixel_pitch,
        volume_depth=args.volume_depth,
        d_val=args.d_val,
        pad_size=[max(img_size), max(img_size)],
        aperture_size=int(sum(img_size) / 2.0),
        num_planes=args.num_planes,
    )

    args_prop.distances = (
        torch.linspace(
            -args_prop.volume_depth / 2.0,
            args_prop.volume_depth / 2.0,
            args_prop.num_planes,
        )
        + args_prop.d_val
    )

    print("\n" + "=" * 60)
    print("[*] Initializing Propagator")
    print("=" * 60)
    prop = propagator(
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

    print("\n" + "=" * 60)
    print("[*] Comparing Reconstructions")
    print("=" * 60)
    compare_reconstructions(
        gt_amp_path=amp_path,
        gt_phase_path=phase_path,
        siren_amp_model=amp_model,
        siren_phase_model=phase_model,
        img_size=img_size,
        prop=prop,
        args_prop=args_prop,
    )
