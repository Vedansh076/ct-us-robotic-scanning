"""
train_pix2pix.py — CT→SimUS Pix2Pix Training
=============================================
Built by extending the original train.py.  Every structural pattern —
logging, arg parsing, checkpoint format, visualisation, DataLoader
construction, subject lists, AMP usage — is preserved.  Only the parts
that must change for Pix2Pix are changed, and each change is documented.

Usage (mirrors original train.py)
----------------------------------
    python train_pix2pix.py \
        --data_root /path/to/dataset \
        --output_dir ./runs/pix2pix

    # Resume
    python train_pix2pix.py \
        --data_root /path/to/dataset \
        --output_dir ./runs/pix2pix \
        --resume ./runs/pix2pix/latest_checkpoint.pth

COMPLETE CHANGE LOG vs train.py
---------------------------------

[CHANGE 1] Two models + two optimisers
  Original: one UNet, one Adam.
  Now: UNet generator G and PatchGANDiscriminator D, each with its own Adam.
  Rationale: G and D have separate loss functions and must not share a
  gradient tape.  Their Adam beta1 is lowered from the original 0.9 to 0.5,
  the standard GAN setting that reduces oscillation from high momentum.

[CHANGE 2] SimUS re-normalisation  [0,1] → [-1,1] inside the training loop
  Original: dataset.py returns SimUS in [0, 1] (Sigmoid head).
  Now: generator head is Tanh → output in [-1, 1].
  Reconciliation: real SimUS tensors are mapped  sim * 2 - 1  before they
  enter either the discriminator or the L1 loss.  dataset.py is NOT touched.
  Visualisation rescales back:  (tensor + 1) / 2  before imshow.

[CHANGE 3] Two-step per-batch update
  Original: single forward → L1 → backward.
  Now:
    Step A: update D on (real pair) and (fake pair with detach).
    Step B: update G using GAN loss + λ·L1 on the same fake.
  The .detach() on the fake during Step A prevents G gradients from leaking
  into D's backward pass.

[CHANGE 4] GAN loss function
  BCEWithLogitsLoss applied to PatchGAN logit maps.
  Label smoothing: real target = 0.9 (not 1.0) to prevent D overconfidence.
  The discriminator loss is halved (× 0.5) to slow D relative to G.

[CHANGE 5] Two GradScalers (one per backward pass)
  Original: one GradScaler for a single backward.
  AMP requires separate scalers when two independent backward passes occur
  within the same batch step.  scaler_G and scaler_D are fully independent.

[CHANGE 6] Dual checkpoint format
  Original checkpoint keys: model_state, optimizer_state, scaler_state,
                            best_val_loss, history, epoch.
  Extended keys added:      G_state, D_state, opt_G_state, opt_D_state,
                            scaler_G_state, scaler_D_state.
  The original keys are also saved so that a generator checkpoint can be
  loaded directly by inference.py (which expects model_state).

[CHANGE 7] Visualisation: 4-column grid
  Original: CT | GT SimUS | Pred SimUS  (3 columns).
  Now:       CT | GT SimUS | Fake SimUS | D patch map  (4 columns).
  The patch map (sigmoid of D logits) is upsampled to image size for display.
  All images are rescaled to [0, 1] for matplotlib (CHANGE 2 reversal).

[CHANGE 8] Loss history extended
  history dict now tracks: train_G, train_D, val_G (L1 component on val set).
  The loss plot shows all three curves.

UNCHANGED from original train.py
----------------------------------
  - TRAIN_SUBJECTS / VAL_SUBJECTS and CTSimUSDataset call signature
  - DataLoader construction (same flags: drop_last, persistent_workers, etc.)
  - Logging format and log calls
  - Cosine-annealing LR schedule (applied to G; D uses a constant rate)
  - Gradient clipping (applied to G only — D rarely explodes)
  - config.json save for reproducibility
  - Loss curve plot function (extended, but same matplotlib style)
  - vis_dir structure under output_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from pix2pix.model import UNet
from pix2pix.discriminator import PatchGANDiscriminator
from dataset import CTSimUSDataset


# ---------------------------------------------------------------------------
# Logging  (UNCHANGED)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subject lists  (UNCHANGED)
# ---------------------------------------------------------------------------
TRAIN_SUBJECTS = ["tcga-qq-a8vg", "tcga-qq-asv2"]
VAL_SUBJECTS   = ["tcga-qq-asvc"]


# ---------------------------------------------------------------------------
# Argument parsing  (UNCHANGED flags + Pix2Pix additions)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CT → SimUS Pix2Pix trainer")

    # Paths  (UNCHANGED)
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./runs/pix2pix")

    # Training  (UNCHANGED except note on lr/beta)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="4 for 8 GB VRAM; 8 for 16 GB")
    p.add_argument("--lr",           type=float, default=2e-4,
                   help="LR for both G and D (Pix2Pix default)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="0 is the GAN convention; original used 1e-5")
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Applied to G only, same as original")

    # Model  (UNCHANGED)
    p.add_argument("--base_features",    type=int,   default=64)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--lambda_l1",        type=float, default=100.0,
                   help="Weight on L1 reconstruction loss (Pix2Pix default)")

    # Data loading  (UNCHANGED)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory",  action="store_true", default=True)

    # Reporting  (UNCHANGED)
    p.add_argument("--vis_every",  type=int, default=5)
    p.add_argument("--n_vis",      type=int, default=4)

    # Resume  (UNCHANGED flag name)
    p.add_argument("--resume", type=str, default="",
                   help="Path to latest_checkpoint.pth to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# CHANGE 2 helper — SimUS range conversion
# ---------------------------------------------------------------------------

def simus_to_tanh_range(sim: torch.Tensor) -> torch.Tensor:
    """Map real SimUS from [0, 1] (dataset.py) to [-1, 1] (Tanh generator)."""
    return sim * 2.0 - 1.0


def tanh_range_to_display(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] tensor back to [0, 1] for matplotlib / imshow."""
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# CHANGE 4 — GAN loss helpers
# ---------------------------------------------------------------------------

def _bce(pred: torch.Tensor, target_val: float) -> torch.Tensor:
    """BCEWithLogitsLoss with a scalar target (broadcast to pred shape)."""
    target = torch.full_like(pred, target_val)
    return F.binary_cross_entropy_with_logits(pred, target)


def d_loss_real(logits: torch.Tensor) -> torch.Tensor:
    """D on real pairs: target = 0.9 (label smoothing)."""
    return _bce(logits, 0.9)


def d_loss_fake(logits: torch.Tensor) -> torch.Tensor:
    """D on fake pairs: target = 0.0."""
    return _bce(logits, 0.0)


def g_loss_gan(logits: torch.Tensor) -> torch.Tensor:
    """G wants D to output 1.0 for fakes."""
    return _bce(logits, 1.0)


# ---------------------------------------------------------------------------
# CHANGE 6 — Checkpoint helpers (extended format)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    scaler_G: GradScaler,
    scaler_D: GradScaler,
    epoch: int,
    best_val_loss: float,
    history: dict,
) -> None:
    torch.save(
        {
            "epoch":          epoch,
            "best_val_loss":  best_val_loss,
            "history":        history,
            # Generator keys — also stored as model_state / optimizer_state /
            # scaler_state so inference.py can load this file unchanged.
            "model_state":    G.state_dict(),
            "optimizer_state": opt_G.state_dict(),
            "scaler_state":   scaler_G.state_dict(),
            # Explicit Pix2Pix keys for clean resume
            "G_state":        G.state_dict(),
            "D_state":        D.state_dict(),
            "opt_G_state":    opt_G.state_dict(),
            "opt_D_state":    opt_D.state_dict(),
            "scaler_G_state": scaler_G.state_dict(),
            "scaler_D_state": scaler_D.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    scaler_G: GradScaler,
    scaler_D: GradScaler,
    device: torch.device,
) -> tuple[int, float, dict]:
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    scaler_G.load_state_dict(ckpt["scaler_G_state"])
    scaler_D.load_state_dict(ckpt["scaler_D_state"])
    log.info(
        f"Resumed from epoch {ckpt['epoch']}  "
        f"(best val G loss {ckpt['best_val_loss']:.6f})"
    )
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt.get("history", {"train_G": [], "train_D": [], "val_G": []}),
    )


# ---------------------------------------------------------------------------
# CHANGE 7 — Visualisation (4-column grid)
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_visual_comparison(
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    n_samples: int = 4,
    epoch: int = 0,
) -> None:
    """Save CT | GT SimUS | Fake SimUS | D patch map (4 columns)."""
    G.eval()
    D.eval()

    ct_list, gt_list, fake_list, patch_list = [], [], [], []

    for ct, sim in loader:
        ct  = ct.to(device)
        sim = sim.to(device)
        sim_tanh = simus_to_tanh_range(sim)      # [0,1] → [-1,1] for D

        with autocast(enabled=device.type == "cuda"):
            fake     = G(ct)                             # [-1, 1]
            patch    = torch.sigmoid(D(ct, fake))        # probability map

        # Upsample patch map to image size for side-by-side display
        patch_up = F.interpolate(patch, size=ct.shape[-2:], mode="nearest")
        patch_up = patch_up.expand_as(ct)                # broadcast to 1 ch

        ct_list.append(ct.cpu())
        gt_list.append(sim.cpu())           # already [0,1] from dataset
        fake_list.append(fake.cpu())
        patch_list.append(patch_up.cpu())

        if sum(c.shape[0] for c in ct_list) >= n_samples:
            break

    n = n_samples
    ct_all    = torch.cat(ct_list)[:n]
    gt_all    = torch.cat(gt_list)[:n]
    fake_all  = torch.cat(fake_list)[:n]
    patch_all = torch.cat(patch_list)[:n]

    # Rescale everything to [0,1] for imshow
    ct_disp   = tanh_range_to_display(ct_all)   # CT was in [-1,1]
    fake_disp = tanh_range_to_display(fake_all) # fake in [-1,1]
    # gt_all already [0,1]; patch_all already [0,1] (sigmoid)

    cols  = ["CT input", "GT SimUS", "Fake SimUS", "D patch (P(real))"]
    imgs  = [ct_disp, gt_all, fake_disp, patch_all]

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col_idx, (title, batch) in enumerate(zip(cols, imgs)):
        for row_idx in range(n):
            img = batch[row_idx, 0].numpy()
            axes[row_idx, col_idx].imshow(img, cmap="gray", origin="upper",
                                          vmin=0.0, vmax=1.0)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=10)

    fig.suptitle(f"Epoch {epoch}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    G.train()
    D.train()


# ---------------------------------------------------------------------------
# CHANGE 8 — Loss plot (extended)
# ---------------------------------------------------------------------------

def plot_losses(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_G"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, history["train_G"], label="Train loss_G", linewidth=1.5)
    ax.plot(epochs, history["train_D"], label="Train loss_D", linewidth=1.5)
    ax.plot(epochs, history["val_G"],   label="Val L1 (G)",   linewidth=1.5,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CT → SimUS Pix2Pix — Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Loss plot saved → {out_path}")


# ---------------------------------------------------------------------------
# CHANGE 3 — Per-epoch training (two-step G / D update)
# ---------------------------------------------------------------------------

def run_train_epoch(
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    scaler_G: GradScaler,
    scaler_D: GradScaler,
    l1_criterion: nn.Module,
    lambda_l1: float,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
) -> tuple[float, float]:
    """
    One full training epoch.  Returns (mean_loss_G, mean_loss_D).

    Step A — Discriminator update
      Compute D loss on (real pair) and (fake pair, G output detached).
      Scale by 0.5 to slow D relative to G.

    Step B — Generator update
      Recompute fake (with grad), compute GAN + λ·L1 loss, backprop into G.
    """
    G.train()
    D.train()
    total_G = total_D = 0.0

    for ct, sim in loader:
        ct  = ct.to(device, non_blocking=True)
        sim = sim.to(device, non_blocking=True)
        # CHANGE 2: rescale real SimUS to [-1,1] for D and L1
        sim_tanh = simus_to_tanh_range(sim)

        # ── Step A: Discriminator ───────────────────────────────────────
        with autocast(enabled=use_amp):
            fake       = G(ct).detach()          # no gradient into G here
            pred_real  = D(ct, sim_tanh)
            pred_fake  = D(ct, fake)
            loss_D     = (d_loss_real(pred_real) + d_loss_fake(pred_fake)) * 0.5

        opt_D.zero_grad(set_to_none=True)
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # ── Step B: Generator ───────────────────────────────────────────
        with autocast(enabled=use_amp):
            fake      = G(ct)                    # re-run with grad
            pred_fake = D(ct, fake)
            loss_G_gan = g_loss_gan(pred_fake)
            loss_G_l1  = l1_criterion(fake, sim_tanh) * lambda_l1
            loss_G     = loss_G_gan + loss_G_l1

        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward()
        if grad_clip > 0:
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
        scaler_G.step(opt_G)
        scaler_G.update()

        total_G += loss_G.item() * ct.size(0)
        total_D += loss_D.item() * ct.size(0)

    n = len(loader.dataset)
    return total_G / n, total_D / n


def run_val_epoch(
    G: nn.Module,
    loader: DataLoader,
    l1_criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Validation: L1 loss only (no D needed).  MATCHES original run_epoch logic."""
    G.eval()
    total = 0.0
    with torch.no_grad():
        for ct, sim in loader:
            ct  = ct.to(device, non_blocking=True)
            sim = sim.to(device, non_blocking=True)
            sim_tanh = simus_to_tanh_range(sim)      # same space as G output
            with autocast(enabled=use_amp):
                pred = G(ct)
                loss = l1_criterion(pred, sim_tanh)
            total += loss.item() * ct.size(0)
    return total / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    # Config save  (UNCHANGED)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device  (UNCHANGED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(f"Device: {device} | AMP: {use_amp}")

    # ----- Datasets (UNCHANGED call signature) -----
    train_ds = CTSimUSDataset(args.data_root, TRAIN_SUBJECTS)
    val_ds   = CTSimUSDataset(args.data_root, VAL_SUBJECTS)
    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ----- DataLoaders (UNCHANGED flags) -----
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

    # ----- Models (CHANGE 1) -----
    G = UNet(
        in_channels=1,
        out_channels=1,
        base_features=args.base_features,
        dropout=args.dropout,
        pix2pix_dropout=True,
    ).to(device)
    D = PatchGANDiscriminator(in_channels=1, out_channels=1).to(device)
    log.info(G)
    log.info(D)

    # ----- Optimisers (CHANGE 1: two optimisers, beta1=0.5) -----
    opt_G = torch.optim.Adam(
        G.parameters(), lr=args.lr, betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    opt_D = torch.optim.Adam(
        D.parameters(), lr=args.lr, betas=(0.5, 0.999),
    )

    # Cosine-anneal G LR exactly as original; D uses constant LR
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ----- AMP scalers (CHANGE 5: two scalers) -----
    scaler_G = GradScaler(enabled=use_amp)
    scaler_D = GradScaler(enabled=use_amp)

    l1_criterion = nn.L1Loss()

    # ----- Resume (CHANGE 6: extended loader) -----
    start_epoch   = 0
    best_val_loss = float("inf")
    history: dict = {"train_G": [], "train_D": [], "val_G": []}

    if args.resume:
        start_epoch, best_val_loss, history = load_checkpoint(
            args.resume, G, D, opt_G, opt_D, scaler_G, scaler_D, device
        )
        for _ in range(start_epoch):
            scheduler_G.step()

    # ----- Training loop -----
    log.info(f"Starting Pix2Pix training for {args.epochs} epochs …")
    t0 = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):

        loss_G, loss_D = run_train_epoch(
            G, D, train_loader, opt_G, opt_D,
            scaler_G, scaler_D, l1_criterion, args.lambda_l1,
            device, args.grad_clip, use_amp,
        )
        val_loss = run_val_epoch(G, val_loader, l1_criterion, device, use_amp)
        scheduler_G.step()

        history["train_G"].append(loss_G)
        history["train_D"].append(loss_D)
        history["val_G"].append(val_loss)

        elapsed = time.time() - t0
        lr_now  = scheduler_G.get_last_lr()[0]
        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss_G={loss_G:.5f}  loss_D={loss_D:.5f}  "
            f"val_L1={val_loss:.5f}  lr={lr_now:.2e}  "
            f"elapsed={elapsed/60:.1f}min"
        )

        # ----- Save latest checkpoint (always) -----
        save_checkpoint(
            out_dir / "latest_checkpoint.pth",
            G, D, opt_G, opt_D, scaler_G, scaler_D,
            epoch, best_val_loss, history,
        )

        # ----- Save best generator checkpoint -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                out_dir / "best_model.pth",
                G, D, opt_G, opt_D, scaler_G, scaler_D,
                epoch, best_val_loss, history,
            )
            log.info(f"  ↳ New best val L1: {best_val_loss:.6f}")

        # ----- Visual comparison -----
        if epoch % args.vis_every == 0 or epoch == args.epochs:
            save_visual_comparison(
                G, D, val_loader, device,
                vis_dir / f"comparison_epoch{epoch:04d}.png",
                n_samples=args.n_vis,
                epoch=epoch,
            )

        # ----- Loss plot -----
        plot_losses(history, out_dir / "loss_curve.png")

    log.info(f"Training complete.  Best val L1: {best_val_loss:.6f}")
    log.info(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()


# ===========================================================================
# HYPERPARAMETER RECOMMENDATIONS FOR 8–16 GB GPU
# ===========================================================================
#
# Batch size
#   8 GB  → batch_size = 2–4   (G + D + activations for both)
#   16 GB → batch_size = 6–8
#
# Learning rate  (both G and D share --lr)
#   2e-4 is the Pix2Pix paper default and works well.
#   If loss_D collapses to 0 in the first 5–10 epochs, halve --lr to 1e-4
#   for D specifically (split into opt_D lr in the code above).
#
# lambda_l1
#   100 (default) maintains structural accuracy.
#   Increase to 150 if predictions lose fine-scale bone / shadow detail.
#
# Epochs + LR schedule
#   200 epochs with CosineAnnealingLR down to 2e-6 works well.
#   For faster iteration, 100 epochs is sufficient to see GAN sharpening.
#
# Healthy training indicators
#   loss_D ≈ 0.3–0.7  →  D is competitive but not winning
#   loss_G decreasing over 50–100 epochs  →  G is learning from D
#   val_L1 converges to roughly the same value as pure L1 training
#     but with qualitatively sharper textures in the saved grids
#
# Warning signs
#   loss_D → 0 early   : D dominates; reduce D lr or reduce lambda_l1
#   loss_G → constant  : check .detach() on fake in Step A; check AMP scaler
#   checkerboard artefacts in fakes : add spectral norm to D (optional)
# ===========================================================================