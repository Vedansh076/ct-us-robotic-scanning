"""
train_ultrabones.py — Train UNet model for bone surface segmentation on UltraBones100k.

Usage:
------
    python model/train_ultrabones.py --data_root /path/to/UltraBones100k --output_dir ./runs/ultrabones_exp1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ultrabones_dataset import UltraBonesDataset
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
# Dice Loss Helper
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1. - dice


class SegmentLoss(nn.Module):
    """Combines BCE and Dice Loss for robust boundary segmentation."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp predictions to avoid numerical instability (log(0)) in BCE
        preds_clamped = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(preds_clamped, targets)
        dice = self.dice_loss(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UltraBones100k UNet segmentation trainer")

    # Paths
    p.add_argument("--data_root",  type=str, required=True,
                   help="Root of the UltraBones100k dataset (contains specimen01/, specimen02/, etc.)")
    p.add_argument("--output_dir", type=str, default="./runs/ultrabones_exp1",
                   help="Where to save checkpoints, plots, and sample images")

    # Training Hyperparameters
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Per-GPU batch size. 8 is recommended for 8 GB VRAM at 256x256")
    p.add_argument("--lr",           type=float, default=2e-4,
                   help="Initial learning rate for Adam")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")

    # Data config
    p.add_argument("--mask_type",    type=str,   choices=["Labels", "Labels_full"], default="Labels",
                   help="Which mask annotation directory to train on: 'Labels' or 'Labels_full'")
    p.add_argument("--img_size",     type=int,   default=256)
    p.add_argument("--train_specimens", type=str, default="1,2,3,4,5,6,7,8,9,10,11",
                   help="Specimen numbers to use for training (comma-separated list)")
    p.add_argument("--val_specimens",   type=str, default="12,13,14",
                   help="Specimen numbers to use for validation (comma-separated list)")

    # Model Hyperparameters
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
    """Save a grid of [Ultrasound Image | GT Mask | Pred Mask] for n_samples examples."""
    model.eval()

    us_list, gt_list, pred_list = [], [], []
    for us, mask in loader:
        us = us.to(device)
        with autocast(enabled=device.type == "cuda"):
            pred = model(us)
        us_list.append(us.cpu())
        gt_list.append(mask)
        pred_list.append(pred.cpu())
        if sum(u.shape[0] for u in us_list) >= n_samples:
            break

    us_all   = torch.cat(us_list)[:n_samples]
    gt_all   = torch.cat(gt_list)[:n_samples]
    pred_all = torch.cat(pred_list)[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    cols = ["Ultrasound (US)", "GT Mask", "Pred Mask"]
    for col_idx, (title, imgs) in enumerate(
        zip(cols, [us_all, gt_all, pred_all])
    ):
        for row_idx in range(n_samples):
            img = imgs[row_idx, 0].numpy()
            # Pred Mask will show probabilities; we can display as heatmap or thresholded
            axes[row_idx, col_idx].imshow(img, cmap="gray" if col_idx < 2 else "hot", origin="upper", vmin=0.0, vmax=1.0)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=11)

    fig.suptitle(f"Epoch {epoch}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_losses(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val"],   label="Val Loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.set_title("UltraBones100k — Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training epoch
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

    for us, mask in loader:
        us   = us.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            pred = model(us)
            loss = criterion(pred, mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * us.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main Training Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # Save configs for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(f"Device: {device} | AMP: {use_amp}")

    # Parse train/val specimens lists
    train_specs = [int(x.strip()) for x in args.train_specimens.split(",") if x.strip()]
    val_specs = [int(x.strip()) for x in args.val_specimens.split(",") if x.strip()]

    # ----- Datasets -----
    train_ds = UltraBonesDataset(
        data_root=args.data_root,
        specimen_ids=train_specs,
        mask_type=args.mask_type,
        img_size=args.img_size,
        augment=True,
    )
    val_ds = UltraBonesDataset(
        data_root=args.data_root,
        specimen_ids=val_specs,
        mask_type=args.mask_type,
        img_size=args.img_size,
        augment=False,
    )
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

    # ----- Model, loss, optimizer -----
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_features=args.base_features,
        dropout=args.dropout,
    ).to(device)
    log.info(model)

    # Combine BCE & Dice Loss
    criterion = SegmentLoss(bce_weight=0.5, dice_weight=0.5)

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
        for _ in range(start_epoch):
            scheduler.step()

    # ----- Training Loop -----
    log.info(f"Starting training for {args.epochs} epochs ...")
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

        # Save checkpoints
        save_checkpoint(
            out_dir / "latest_checkpoint.pth",
            model, optimizer, scaler, epoch, best_val_loss, history
        )

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

        # Loss curve plotting
        plot_losses(history, out_dir / "loss_curve.png")

    log.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    log.info(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
