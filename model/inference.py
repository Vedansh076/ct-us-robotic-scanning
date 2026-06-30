"""
inference.py — Run a trained CT → SimUS model on new data.

Modes
-----
1. Single file   : --ct_path slice.npy  →  saves pred_simus.npy + PNG
2. Directory     : --ct_dir  ct/        →  saves all predictions in --out_dir
3. Evaluation    : --ct_dir + --simus_dir  →  computes MAE / SSIM over pairs

Usage examples
--------------
    # Single file
    python inference.py --checkpoint runs/exp/best_model.pth \
                        --ct_path dataset/ct/tcga-qq-asvc_00042.npy \
                        --out_dir preds/

    # Whole directory
    python inference.py --checkpoint runs/exp/best_model.pth \
                        --ct_dir dataset/ct/ \
                        --out_dir preds/

    # Evaluation with metrics
    python inference.py --checkpoint runs/exp/best_model.pth \
                        --ct_dir dataset/ct/ \
                        --simus_dir dataset/simus/ \
                        --out_dir preds/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from dataset import normalise_ct, normalise_simus, denormalise_simus
from model import UNet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Simplified single-channel SSIM (Wang et al. 2004)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = pred.mean(), gt.mean()
    sigma1  = pred.std() ** 2
    sigma2  = gt.std()   ** 2
    sigma12 = float(np.mean((pred - mu1) * (gt - mu2)))
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return float(ssim)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CT → SimUS inference")
    p.add_argument("--checkpoint",    required=True,  help="Path to best_model.pth")
    p.add_argument("--ct_path",       default="",     help="Single CT .npy file")
    p.add_argument("--ct_dir",        default="",     help="Directory of CT .npy files")
    p.add_argument("--simus_dir",     default="",
                   help="Optional: directory of GT SimUS files for evaluation metrics")
    p.add_argument("--out_dir",       default="./inference_out")
    p.add_argument("--base_features", type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.0,
                   help="Set to 0 for deterministic inference")
    p.add_argument("--save_npy",      action="store_true", default=True,
                   help="Save predictions as .npy files")
    p.add_argument("--save_png",      action="store_true", default=True,
                   help="Save side-by-side PNG visualisations")
    p.add_argument("--device",        default="auto",
                   help="'auto', 'cpu', or 'cuda'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_slice(
    model: torch.nn.Module,
    ct_arr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict SimUS from a raw (un-normalised) CT numpy array.

    Parameters
    ----------
    ct_arr : (H, W)  raw CT slice in HU
    Returns : (H, W) predicted SimUS normalised to [0, 1]
    """
    ct_norm = normalise_ct(ct_arr)                          # [-1, 1]
    ct_t    = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    ct_t    = ct_t.to(device)

    model.eval()
    with autocast(enabled=device.type == "cuda"):
        pred = model(ct_t)                                  # (1,1,H,W) in [0,1]

    return pred.squeeze().cpu().float().numpy()             # (H, W)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_png(
    ct_raw: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray | None,
    out_path: Path,
    stem: str,
) -> None:
    n_cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    titles = ["CT input", "Pred SimUS"]
    images = [ct_raw, pred]
    if gt is not None:
        titles.append("GT SimUS")
        images.append(gt)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", origin="upper")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Device -----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # ----- Load model -----
    model = UNet(base_features=args.base_features, dropout=args.dropout).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')},  "
             f"best val {ckpt.get('best_val_loss', float('nan')):.5f})")

    # ----- Collect files -----
    if args.ct_path:
        ct_files = [Path(args.ct_path)]
    elif args.ct_dir:
        ct_files = sorted(Path(args.ct_dir).glob("*.npy"))
    else:
        raise ValueError("Provide --ct_path or --ct_dir")

    log.info(f"Processing {len(ct_files)} file(s) …")

    # ----- Metrics accumulators -----
    mae_list, ssim_list = [], []

    for ct_path in ct_files:
        stem = ct_path.stem
        ct_raw = np.load(ct_path)
        if ct_raw.ndim == 3:
            ct_raw = ct_raw.squeeze()

        pred = predict_slice(model, ct_raw, device)        # [0, 1]

        # Ground truth (optional)
        gt_norm = None
        if args.simus_dir:
            gt_path = Path(args.simus_dir) / ct_path.name
            if gt_path.exists():
                gt_raw  = np.load(gt_path)
                if gt_raw.ndim == 3:
                    gt_raw = gt_raw.squeeze()
                gt_norm = normalise_simus(gt_raw)
                mae_list.append(compute_mae(pred, gt_norm))
                ssim_list.append(compute_ssim(pred, gt_norm))

        if args.save_npy:
            np.save(out_dir / f"{stem}_pred.npy", pred)

        if args.save_png:
            save_comparison_png(
                ct_raw, pred, gt_norm,
                out_dir / f"{stem}_compare.png",
                stem,
            )

    # ----- Aggregate metrics -----
    if mae_list:
        log.info(
            f"Evaluation over {len(mae_list)} paired samples:\n"
            f"  MAE  (norm [0,1]): {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f}\n"
            f"  SSIM             : {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}"
        )

    log.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
