"""
evaluate_pix2pix.py — Quantitative evaluation of trained Pix2Pix model.

Computes SSIM, PSNR, MAE (L1), and per-sample metrics on the Cavalcanti
validation set (paired CT+Seg → US ground truth).

Usage
-----
    python evaluate_pix2pix.py \
        --data_root ./data/cavalcanti_processed \
        --checkpoint ./runs/pix2pix/best_model.pth \
        --output_dir ./runs/pix2pix/eval

Produces:
    - eval_metrics.json         : aggregate SSIM, PSNR, MAE
    - eval_visual_grid.png      : side-by-side grid (CT | Bone | GT US | Pred US | Error)
    - per_sample_metrics.csv    : per-sample breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add model directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))

from dataset import CTSimUSDataset, normalise_simus, SIMUS_MAX, SIMUS_MIN
from pix2pix.model import UNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ssim_2d(pred: np.ndarray, target: np.ndarray,
            window_size: int = 11, C1: float = 0.01**2,
            C2: float = 0.03**2) -> float:
    """Compute SSIM between two 2D images in [0, 1]."""
    from scipy.ndimage import uniform_filter
    mu_p = uniform_filter(pred, size=window_size)
    mu_t = uniform_filter(target, size=window_size)
    sigma_pp = uniform_filter(pred * pred, size=window_size) - mu_p * mu_p
    sigma_tt = uniform_filter(target * target, size=window_size) - mu_t * mu_t
    sigma_pt = uniform_filter(pred * target, size=window_size) - mu_p * mu_t

    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


def psnr(pred: np.ndarray, target: np.ndarray,
         max_val: float = 1.0) -> float:
    """Compute PSNR between two images in [0, max_val]."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(max_val**2 / mse))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pix2pix_generator(checkpoint_path: str, device: torch.device,
                           base_features: int = 64,
                           dropout: float = 0.1) -> torch.nn.Module:
    """Load trained Pix2Pix generator."""
    model = UNet(in_channels=2, out_channels=1,
                 base_features=base_features, dropout=dropout)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("G", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Pix2Pix on Cavalcanti validation set")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to cavalcanti_processed directory")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_model.pth")
    p.add_argument("--output_dir", type=str, default="./runs/pix2pix/eval",
                   help="Where to save evaluation results")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_vis", type=int, default=8,
                   help="Number of samples to show in the visual grid")
    p.add_argument("--val_subjects", type=str, default="auto",
                   help="Comma-separated val subject IDs, or 'auto' for meta.json")
    return p.parse_args()


def resolve_val_subjects(data_root: str, val_arg: str) -> list[str]:
    """Resolve validation subject list from meta.json or CLI."""
    if val_arg == "auto":
        meta_path = os.path.join(data_root, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("val_subjects", meta.get("val", []))
        else:
            raise FileNotFoundError(f"meta.json not found at {meta_path}")
    return [s.strip() for s in val_arg.split(",") if s.strip()]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Load model ---
    log.info("Loading model from %s", args.checkpoint)
    model = load_pix2pix_generator(args.checkpoint, device)
    log.info("Model loaded successfully")

    # --- Load validation dataset ---
    val_subjects = resolve_val_subjects(args.data_root, args.val_subjects)
    log.info("Validation subjects: %s", val_subjects)

    val_ds = CTSimUSDataset(args.data_root, val_subjects, is_pix2pix=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    log.info("Validation samples: %d", len(val_ds))

    # --- Run inference and compute metrics ---
    all_ssim = []
    all_psnr = []
    all_mae = []
    vis_samples = []  # Store samples for visualization

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)    # (B, 2, H, W) in [-1, 1]
            targets = targets.to(device)  # (B, 1, H, W) in [0, 1]

            # Generate predictions (Tanh output → [-1, 1])
            preds = model(inputs)
            # Rescale from [-1, 1] to [0, 1]
            preds_01 = (preds + 1.0) / 2.0
            preds_01 = preds_01.clamp(0.0, 1.0)

            # Move to numpy for metric computation
            preds_np = preds_01.cpu().numpy()
            targets_np = targets.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            for i in range(preds_np.shape[0]):
                pred_img = preds_np[i, 0]     # (H, W) in [0, 1]
                gt_img = targets_np[i, 0]     # (H, W) in [0, 1]

                s = ssim_2d(pred_img, gt_img)
                p = psnr(pred_img, gt_img)
                m = mae(pred_img, gt_img)

                all_ssim.append(s)
                all_psnr.append(p)
                all_mae.append(m)

                # Collect visualization samples (evenly spaced)
                global_idx = batch_idx * args.batch_size + i
                if len(vis_samples) < args.n_vis:
                    step = max(1, len(val_ds) // args.n_vis)
                    if global_idx % step == 0:
                        # CT input channel (rescale from [-1,1] to [0,1])
                        ct_vis = (inputs_np[i, 0] + 1.0) / 2.0
                        # Seg input channel (rescale from [-1,1] to [0,1])
                        seg_vis = (inputs_np[i, 1] + 1.0) / 2.0
                        vis_samples.append({
                            "ct": ct_vis,
                            "seg": seg_vis,
                            "gt": gt_img,
                            "pred": pred_img,
                            "error": np.abs(pred_img - gt_img),
                            "ssim": s,
                            "psnr": p,
                        })

    # --- Aggregate metrics ---
    metrics = {
        "n_samples": len(all_ssim),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
    }

    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info("  Samples evaluated : %d", metrics["n_samples"])
    log.info("  SSIM              : %.4f ± %.4f", metrics["ssim_mean"], metrics["ssim_std"])
    log.info("  PSNR              : %.2f ± %.2f dB", metrics["psnr_mean"], metrics["psnr_std"])
    log.info("  MAE (L1)          : %.4f ± %.4f", metrics["mae_mean"], metrics["mae_std"])
    log.info("=" * 60)

    # --- Save metrics ---
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", out_dir / "eval_metrics.json")

    # --- Save per-sample CSV ---
    with open(out_dir / "per_sample_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "ssim", "psnr_dB", "mae"])
        for i, (s, p, m) in enumerate(zip(all_ssim, all_psnr, all_mae)):
            writer.writerow([i, f"{s:.4f}", f"{p:.2f}", f"{m:.4f}"])
    log.info("Per-sample CSV saved → %s", out_dir / "per_sample_metrics.csv")

    # --- Save visual grid ---
    n_vis = len(vis_samples)
    if n_vis > 0:
        fig, axes = plt.subplots(n_vis, 5, figsize=(20, 3.5 * n_vis))
        if n_vis == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["CT Input", "Bone Mask", "Ground Truth US", "Predicted US", "Absolute Error"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=12, fontweight="bold")

        for i, sample in enumerate(vis_samples):
            axes[i, 0].imshow(sample["ct"], cmap="gray", vmin=0, vmax=1)
            axes[i, 1].imshow(sample["seg"], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(sample["gt"], cmap="gray", vmin=0, vmax=1)
            axes[i, 3].imshow(sample["pred"], cmap="gray", vmin=0, vmax=1)
            axes[i, 4].imshow(sample["error"], cmap="hot", vmin=0, vmax=0.5)

            # Add SSIM/PSNR labels
            axes[i, 3].set_xlabel(
                f"SSIM={sample['ssim']:.3f}  PSNR={sample['psnr']:.1f}dB",
                fontsize=9)

            for ax in axes[i]:
                ax.axis("off")

        fig.suptitle("Pix2Pix Evaluation — Cavalcanti Validation Set", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "eval_visual_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Visual grid saved → %s", out_dir / "eval_visual_grid.png")

    log.info("Evaluation complete!")


if __name__ == "__main__":
    main()
