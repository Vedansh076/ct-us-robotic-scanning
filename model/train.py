"""
train.py — Supervised CT → SimUS translation training.

Usage
-----
    python train.py --data_root /path/to/dataset --output_dir ./runs/exp1

Key design decisions
--------------------
* Subject-based train/val split to avoid data leakage.
* L1 loss — more robust to outliers than L2 and produces sharper predictions
  for medical images compared to MSE.
* Adam with a cosine-annealing LR schedule for smooth convergence.
* Mixed-precision (torch.amp) on CUDA for ~2× throughput at no accuracy cost.
* Gradient clipping (max_norm=1.0) guards against rare gradient explosions.
* Image comparisons saved every `--vis_every` epochs (CT | GT SimUS | Pred).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / headless runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import (
    CTSimUSDataset,
    denormalise_ct,
    denormalise_simus,
)
from model import UNet

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default subject IDs
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_SUBJECTS = ["tcga-qq-a8vg", "tcga-qq-asv2"]
DEFAULT_VAL_SUBJECTS   = ["tcga-qq-asvc"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CT → SimUS U-Net trainer")

    # Paths
    p.add_argument("--data_root",  type=str, required=True,
                   help="Root of the dataset directory (contains ct/, simus/, poses/)")
    p.add_argument("--output_dir", type=str, default="./runs/exp",
                   help="Where to save checkpoints, plots, and sample images")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Per-GPU batch size.  8 is safe for 8 GB VRAM at 256²")
    p.add_argument("--lr",           type=float, default=2e-4,
                   help="Initial learning rate for Adam")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")

    # Model
    p.add_argument("--base_features", type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.1)

    # Data loading
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory",  action="store_true", default=True)

    # Reporting
    p.add_argument("--vis_every",  type=int, default=5,
                   help="Save image comparison grids every N epochs")
    p.add_argument("--n_vis",      type=int, default=4,
                   help="Number of samples to show per comparison grid")

    # Resume
    p.add_argument("--resume", type=str, default="",
                   help="Path to checkpoint to resume from")

    # Subject lists (Cavalcanti support)
    p.add_argument("--train_subjects", type=str, default="",
                   help="Comma-separated subject IDs, or 'auto' to read meta.json")
    p.add_argument("--val_subjects",   type=str, default="",
                   help="Comma-separated subject IDs, or 'auto' to read meta.json")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    history: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float, dict]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    log.info(f"Resumed from epoch {ckpt['epoch']}  (best val {ckpt['best_val_loss']:.6f})")
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt.get("history", {"train": [], "val": []})


@torch.no_grad()
def save_visual_comparison(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    n_samples: int = 4,
    epoch: int = 0,
) -> None:
    """Save a grid of [CT | GT SimUS | Pred SimUS] for n_samples examples."""
    model.eval()

    ct_list, gt_list, pred_list = [], [], []
    for ct, sim in loader:
        ct = ct.to(device)
        with autocast(enabled=device.type == "cuda"):
            pred = model(ct)
        ct_list.append(ct.cpu())
        gt_list.append(sim)
        pred_list.append(pred.cpu())
        if sum(c.shape[0] for c in ct_list) >= n_samples:
            break

    ct_all   = torch.cat(ct_list)[:n_samples]
    gt_all   = torch.cat(gt_list)[:n_samples]
    pred_all = torch.cat(pred_list)[:n_samples]

    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    cols = ["CT input", "Seg input", "GT US", "Pred US"]
    
    for row_idx in range(n_samples):
        # 1. CT input (channel 0)
        axes[row_idx, 0].imshow(ct_all[row_idx, 0].numpy(), cmap="gray", origin="upper")
        axes[row_idx, 0].axis("off")
        
        # 2. Seg input (channel 1)
        axes[row_idx, 1].imshow(ct_all[row_idx, 1].numpy(), cmap="gray", origin="upper")
        axes[row_idx, 1].axis("off")
        
        # 3. GT US
        axes[row_idx, 2].imshow(gt_all[row_idx, 0].numpy(), cmap="gray", origin="upper")
        axes[row_idx, 2].axis("off")
        
        # 4. Pred US
        axes[row_idx, 3].imshow(pred_all[row_idx, 0].numpy(), cmap="gray", origin="upper")
        axes[row_idx, 3].axis("off")
        
    for col_idx, title in enumerate(cols):
        axes[0, col_idx].set_title(title, fontsize=11)

    fig.suptitle(f"Epoch {epoch}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_losses(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train"], label="Train L1", linewidth=1.5)
    ax.plot(epochs, history["val"],   label="Val L1",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("CT → SimUS — Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Loss plot saved → {out_path}")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    training: bool,
) -> float:
    model.train(training)
    total_loss = 0.0

    for ct, sim in loader:
        ct  = ct.to(device, non_blocking=True)
        sim = sim.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            pred = model(ct)
            loss = criterion(pred, sim)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * ct.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Subject resolution (auto / CLI / default)
# ---------------------------------------------------------------------------

def _resolve_subjects(args: argparse.Namespace):
    """
    Determine train and val subject lists.

    Priority:
      1. --train_subjects auto  → read meta.json in data_root
      2. --train_subjects s1,s2 → parse comma-separated list
      3. (empty)                → fall back to DEFAULT_*_SUBJECTS
    """
    train_s = args.train_subjects.strip()
    val_s   = args.val_subjects.strip()

    if train_s.lower() == "auto" or val_s.lower() == "auto":
        meta_path = Path(args.data_root) / "meta.json"
        if not meta_path.exists():
            log.error("--train_subjects auto requires meta.json in %s\n"
                      "Run prepare_cavalcanti.py first.", args.data_root)
            raise FileNotFoundError(str(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        t = meta.get("train_subjects", [])
        v = meta.get("val_subjects", [])
        log.info("Loaded auto-split from meta.json: %d train / %d val subjects",
                 len(t), len(v))
        return t, v

    if train_s:
        t = [s.strip() for s in train_s.split(",") if s.strip()]
    else:
        t = DEFAULT_TRAIN_SUBJECTS

    if val_s:
        v = [s.strip() for s in val_s.split(",") if s.strip()]
    else:
        v = DEFAULT_VAL_SUBJECTS

    return t, v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # Save config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(f"Device: {device} | AMP: {use_amp}")

    # ----- Resolve subject lists -----
    train_subjects, val_subjects = _resolve_subjects(args)
    log.info(f"Train subjects: {train_subjects}")
    log.info(f"Val   subjects: {val_subjects}")

    # ----- Datasets -----
    train_ds = CTSimUSDataset(args.data_root, train_subjects)
    val_ds   = CTSimUSDataset(args.data_root, val_subjects)
    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and use_amp,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and use_amp,
        persistent_workers=args.num_workers > 0,
    )

    # ----- Model, loss, optimiser -----
    model = UNet(
        in_channels=2,
        out_channels=1,
        base_features=args.base_features,
        dropout=args.dropout,
    ).to(device)
    log.info(model)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    scaler = GradScaler(enabled=use_amp)

    # ----- Resume -----
    start_epoch = 0
    best_val_loss = float("inf")
    history: dict = {"train": [], "val": []}

    if args.resume:
        start_epoch, best_val_loss, history = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        # Restore scheduler state
        for _ in range(start_epoch):
            scheduler.step()

    # ----- Training loop -----
    log.info(f"Starting training for {args.epochs} epochs …")
    t0 = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.grad_clip, training=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, None, scaler,
            device, args.grad_clip, training=False
        )
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}  elapsed={elapsed/60:.1f}min"
        )

        # Save latest checkpoint (always)
        save_checkpoint(
            out_dir / "latest_checkpoint.pth",
            model, optimizer, scaler, epoch, best_val_loss, history
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                out_dir / "best_model.pth",
                model, optimizer, scaler, epoch, best_val_loss, history
            )
            log.info(f"  ↳ New best val loss: {best_val_loss:.6f}")

        # Visual comparison
        if epoch % args.vis_every == 0 or epoch == args.epochs:
            save_visual_comparison(
                model, val_loader, device,
                vis_dir / f"comparison_epoch{epoch:04d}.png",
                n_samples=args.n_vis,
                epoch=epoch,
            )

        # Loss plot (updated every epoch)
        plot_losses(history, out_dir / "loss_curve.png")

    log.info(f"Training complete. Best val L1: {best_val_loss:.6f}")
    log.info(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
