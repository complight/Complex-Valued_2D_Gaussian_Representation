import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def visualize_gaussian_positions(gaussians_amp_fixed, gaussians_phase_fixed, img_size, itr, result_dir):
    """
    Visualize the positions of two sets of Gaussians and evaluate overlap
    """
    with torch.no_grad():
        means_amp = gaussians_amp_fixed.means_2d.detach().cpu().numpy()
        means_phase = gaussians_phase_fixed.means_2d.detach().cpu().numpy()
        
        H, W = img_size
        scale_x, scale_y = W / 2, H / 2
        
        means_amp_pixel = np.empty_like(means_amp)
        means_amp_pixel[:, 0] = (means_amp[:, 0] + 1) * scale_x
        means_amp_pixel[:, 1] = (means_amp[:, 1] + 1) * scale_y
        
        means_phase_pixel = np.empty_like(means_phase)
        means_phase_pixel[:, 0] = (means_phase[:, 0] + 1) * scale_x
        means_phase_pixel[:, 1] = (means_phase[:, 1] + 1) * scale_y
        
        tree = cKDTree(means_phase_pixel)
        min_distances, _ = tree.query(means_amp_pixel, k=1)
        
        threshold = 5.0
        overlapping_amp = np.sum(min_distances < threshold)
        overlap_percentage = (overlapping_amp / len(means_amp)) * 100
        
        mean_min_dist = min_distances.mean()
        median_min_dist = np.median(min_distances)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].scatter(means_amp_pixel[:, 0], means_amp_pixel[:, 1], 
                       c='blue', s=1, alpha=0.5, rasterized=True)
        axes[0].set_xlim(0, W)
        axes[0].set_ylim(0, H)
        axes[0].set_title(f'Set 1: Amplitude Fixed (Phase Trainable)\nN={len(means_amp)}')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(means_phase_pixel[:, 0], means_phase_pixel[:, 1], 
                       c='red', s=1, alpha=0.5, rasterized=True)
        axes[1].set_xlim(0, W)
        axes[1].set_ylim(0, H)
        axes[1].set_title(f'Set 2: Phase Fixed (Amplitude Trainable)\nN={len(means_phase)}')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].scatter(means_amp_pixel[:, 0], means_amp_pixel[:, 1], 
                       c='blue', s=1, alpha=0.3, rasterized=True)
        axes[2].scatter(means_phase_pixel[:, 0], means_phase_pixel[:, 1], 
                       c='red', s=1, alpha=0.3, rasterized=True)
        axes[2].set_xlim(0, W)
        axes[2].set_ylim(0, H)
        axes[2].set_title(f'Overlay - Overlap Analysis\nOverlap: {overlap_percentage:.1f}% (<{threshold}px)')
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{result_dir}/gaussian_positions_{itr:06d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nGaussian Position Analysis (Iteration {itr}):")
        print(f"  Set 1 (Amp Fixed): {len(means_amp)} Gaussians")
        print(f"  Set 2 (Phase Fixed): {len(means_phase)} Gaussians")
        print(f"  Overlap (<{threshold}px): {overlap_percentage:.2f}% ({overlapping_amp}/{len(means_amp)})")
        print(f"  Mean minimum distance: {mean_min_dist:.2f} pixels")
        print(f"  Median minimum distance: {median_min_dist:.2f} pixels")