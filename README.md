<div align="center">

# Complex-Valued 2D Gaussian Representation for Computer-Generated Holography

[![DOI](https://img.shields.io/badge/DOI-10.1145/3804450-green)](#)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/complight/Complex-Valued_2D_Gaussian_Representation)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b)](https://arxiv.org/abs/2511.15022)
[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://complightlab.com/publications/complex_valued_2d_gaussians/)
<!-- TODO: fill in the DOI badge and link once available -->

**European Conference on Computer Vision (ECCV) 2026**

[Yicheng Zhan](https://albertgary.github.io/)<sup>1</sup> · [Xiangjun Gao](https://gaoxiangjun.github.io/)<sup>2</sup> · [Long Quan](https://home.cse.ust.hk/~quan/)<sup>2</sup> · [Kaan Akşit](https://www.kaanaksit.com/)<sup>1</sup>

<sup>1</sup> University College London (UCL) &emsp; <sup>2</sup> Hong Kong University of Science and Technology (HKUST)
<!-- TODO: confirm the author order and affiliations -->

</div>



## Overview

This repository optimizes a **2D hologram represented by complex-valued Gaussian primitives** and reconstructs it onto multiple depth planes through a differentiable wave-optics propagator. Given a single RGB image (and an optional depth map), the Gaussians jointly encode amplitude and phase; the resulting hologram is numerically propagated to each depth plane and supervised against the per-plane targets.

Unlike the [*Complex-Valued Holographic Radiance Field*](https://github.com/complight/Complex_Valued_Holographic_Radiance_Fields) and [*Gaussian Wave Splatting*](https://github.com/computational-imaging/hsplat)  (3D, multi-view methods), our method follows a **normal (per-plane) propagation** scheme: the hologram is propagated to each plane separately and optimized directly on 2D image targets.

The repository also includes several baselines for comparison:

| Script | Representation |
|---|---|
| `train_2d_gaussian.py` | 2D complex-valued Gaussians (main method) |
| `train_2d_numerical.py` | Direct per-pixel phase-only hologram |
| `train_2d_SIREN.py` | SIREN / MLP coordinate network |
| `train_2d_gaussian_numerical_parallel.py` | Structural Guidance Optimization |



## Installation

### 1. Create the Conda Environment

```bash
git clone https://github.com/complight/Complex-Valued_2D_Gaussian_Representation.git
cd Complex-Valued_2D_Gaussian_Representation
conda create -n gholo python=3.10 -y
conda activate gholo
```

### 2. Install Dependencies

```bash
# PyTorch (match your CUDA toolkit; CUDA 12.x tested)
pip install torch torchvision

# Core dependencies
pip install odak lpips imageio scikit-image matplotlib numpy pillow tqdm
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: a CUDA-enabled build with True
```

> The light propagator is built on [odak](https://github.com/kaanaksit/odak). `lpips` downloads pretrained VGG weights on first use.

### 3. Compile the CUDA Extensions

Two CUDA extensions are required and are built from the project root:

```bash
# Step 1: the 2D Gaussian wave rasterizer (gaussian_2d_cuda)
cd cuda_prop/gaussian_2d_cuda && python install.py
cd ../..

# Step 2: the band-limited propagation kernels (cuda_modules)
cd cuda_prop && python build_cuda_extension.py
cd ..
```

Step 1 automatically downloads the GLM library. Both extensions are compiled for `sm_80` (A100) and `sm_86` (RTX 3090 / A6000) by default. To target other GPUs, uncomment the matching lines in `cuda_prop/gaussian_2d_cuda/setup.py` and `cuda_prop/build_cuda_extension.py`:

```python
# '--generate-code=arch=compute_89,code=sm_89',  # RTX 4090
# '--generate-code=arch=compute_90,code=sm_90'   # H100
```

> **Note:** the rasterizer is precompiled and imported lazily — `cuda_prop/gaussian_2d_cuda/python_import.py` only rebuilds it if `import gaussian_2d_cuda` fails, so a working `.so` is reused without recompiling on every run.



## Dataset

Sample RGB images and their depth maps are provided under `images/`. Each scene is a pair of an RGB file and its matching depth map:

```
images/
├── dragon.jpg        +  dragon_depth.png
├── tiger.jpg         +  tiger_depth.png
├── burger.jpg        +  burger_depth.png
└── ...
```

To use your own data, supply any RGB image via `--target_image_path` and a single-channel depth map via `--depth_path`. If no depth map is given, a constant (all-ones) depth is used and only a single plane can be supervised.



## Light Propagation Parameters

The propagator parameters are defined in the `args_prop` namespace at the bottom of each training script. Defaults:

| Parameter | Default Value | Description |
|---|---|---|
| `wavelengths` | `[639, 532, 473]e-9` | RGB laser wavelengths in meters (red, green, blue) |
| `pixel_pitch` | `3.74e-6` | SLM pixel pitch in meters |
| `volume_depth` | `4e-3` | Total depth of the reconstruction volume in meters |
| `d_val` | `3e-3` | Image location offset (center of the volume) in meters |
| `pad_size` | `[max(img_size), max(img_size)]` | Zero-padding size for the propagation kernel |
| `aperture_size` | `sum(img_size)/2` | Aperture size in pixels |
| `num_planes` | `2` | Number of depth planes (requires a depth map when `> 1`) |

`distances` are derived automatically by spreading `num_planes` evenly across `volume_depth`, centered at `d_val`.



## Training

### Main method — 2D Gaussian hologram

```bash
python train_2d_gaussian.py \
    --target_image_path ./images/tiger.jpg \
    --depth_path ./images/tiger_depth.png \
    --hologram_type full-complex \
    --compression_ratio 0.2 \
    --num_itrs 2001 \
    --eval_freq 2000 \
    --viz_freq -1 \
    --lr 0.01 \
    --img_size 1024 640 \
    --split_ratio 1.0 \
    --device cuda
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--hologram_type` | `full-complex` | `full-complex` optimizes amplitude + phase; `phase-only` optimizes phase with fixed amplitude |
| `--compression_ratio` | `0.5` | Controls the number of Gaussians relative to pixel count |
| `--img_size` | `1024 640` | Target resolution (width height) |
| `--num_itrs` | `2001` | Number of optimization iterations |
| `--viz_freq` | `200` | Save reconstructions every N iters (`-1` disables) |
| `--eval_freq` | `1000` | Run PSNR / SSIM / LPIPS evaluation every N iters |
| `--overwrite_saving` | `True` | If `False`, results go to a per-scene subfolder |

Outputs (reconstructions, phase/amplitude maps, checkpoints, and `log.txt`) are written to `./result_2d/`, with checkpoints under `./result_2d/checkpoints/`.

### Baseline — direct numerical hologram

```bash
python train_2d_numerical.py \
    --target_image_path ./images/dragon.jpg \
    --depth_path ./images/dragon_depth.png \
    --num_itrs 2001 \
    --eval_freq 1000 \
    --lr 0.025 \
    --img_size 1024 640 \
    --device cuda
```

Results are saved to `./result_2d_numerical/`.

### Baseline — SIREN / MLP

```bash
python train_2d_SIREN.py \
    --base_dir ./result_2d \
    --amp_name amp_blue_cat.png \
    --phase_name phase_blue_cat.png \
    --img_size 1024 640 \
    --num_itrs 2000 \
    --lr 0.001 \
    --model_type siren   # or: mlp
```

Ready-to-run example scripts for every variant are provided under `scripts/`.



## Acknowledgements

The authors thank Maoye for the early suggestions on this paper.



## Citation

```bibtex
@inproceedings{zhan2026complexgaussian,
  title     = {Complex-Valued 2D Gaussian Representation for Computer-Generated Holography},
  author    = {Zhan, Yicheng and Gao, Xiangjun and Quan, Long and Ak{\c{s}}it, Kaan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
<!-- TODO: update with the final citation once published -->



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## Contact

For questions about the code or methodology:

**Yicheng Zhan:** yicheng_zhan2001@outlook.com
