"""
inference_pix2pix.py — CT→SimUS Pix2Pix Inference
==================================================
Run on a single image:
    python inference_pix2pix.py \
        --generator_ckpt ./runs/pix2pix/generator_epoch0200.pth \
        --input_path /path/to/ct_slice.png \
        --output_path ./output/simulated_us.png

Run on a directory of CT slices:
    python inference_pix2pix.py \
        --generator_ckpt ./runs/pix2pix/generator_epoch0200.pth \
        --input_dir /path/to/ct_slices/ \
        --output_dir ./output/

CHANGES FROM ORIGINAL U-NET INFERENCE
--------------------------------------
1. **Generator-only loading** (SIMPLIFIED):
   Only the generator checkpoint is needed at inference. The discriminator
   is not loaded — it was only required during training.

2. **tanh → [0, 1] rescaling** (CHANGED):
   The generator now outputs values in [-1, 1] (tanh). Inference rescales
   to [0, 1] via  `output = (output + 1) / 2`  before saving.
   Original sigmoid output was already in [0, 1].

3. **Test-Time Augmentation (TTA)** (NEW, optional):
   `--tta` averages predictions from the original image and its horizontal
   flip. This typically reduces boundary artefacts at no training cost.

4. **Dropout is kept active by default** (NOTE):
   The Pix2Pix paper evaluates with dropout ON (the generator uses MC
   dropout during inference). Pass `--deterministic` to disable it if you
   prefer deterministic predictions.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

from pix2pix.model import UNetGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pix2Pix inference: CT → SimUS")

    p.add_argument("--generator_ckpt", required=True,
                   help="Path to generator_epochXXXX.pth or latest.pth")

    # Input: single file or directory
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_path", help="Single CT image path")
    g.add_argument("--input_dir",  help="Directory of CT images (.png/.tif/.jpg)")

    # Output
    p.add_argument("--output_path", default=None, help="Output file (single mode)")
    p.add_argument("--output_dir",  default="./inference_out",
                   help="Output directory (batch mode)")

    # Image settings
    p.add_argument("--img_size",     type=int, default=256)
    p.add_argument("--in_channels",  type=int, default=1)
    p.add_argument("--out_channels", type=int, default=1)

    # Behaviour
    p.add_argument("--tta",          action="store_true",
                   help="Test-time augmentation (horizontal flip average)")
    p.add_argument("--deterministic", action="store_true",
                   help="Set model to eval mode (disables dropout)")
    p.add_argument("--save_comparison", action="store_true",
                   help="Save CT | SimUS side-by-side grid (requires --input_dir)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generator(ckpt_path: str, in_ch: int, out_ch: int,
                   device: torch.device) -> UNetGenerator:
    G = UNetGenerator(in_channels=in_ch, out_channels=out_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both stand-alone generator checkpoint and combined 'latest.pth'
    state = ckpt.get("G", ckpt)
    G.load_state_dict(state)
    print(f"✓ Generator loaded from {ckpt_path}")
    return G


# ---------------------------------------------------------------------------
# Pre/post-processing
# ---------------------------------------------------------------------------

def preprocess(img_path: str, img_size: int) -> torch.Tensor:
    """Load a grayscale CT slice and normalise to [-1, 1]."""
    img = Image.open(img_path).convert("L")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),                          # [0, 1]
        T.Normalize(mean=[0.5], std=[0.5]),    # [-1, 1]
    ])
    return transform(img).unsqueeze(0)         # (1, 1, H, W)


def postprocess(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale tanh output [-1, 1] → [0, 1]."""
    return (tensor.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(G: UNetGenerator, ct: torch.Tensor,
            device: torch.device, tta: bool = False) -> torch.Tensor:
    """
    Run generator on a (1, 1, H, W) CT tensor.
    Returns (1, 1, H, W) SimUS tensor in [0, 1].
    """
    ct = ct.to(device)

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        fake = G(ct)
        if tta:
            ct_flip = torch.flip(ct, dims=[-1])          # horizontal flip
            fake_flip = G(ct_flip)
            fake_flip = torch.flip(fake_flip, dims=[-1]) # flip back
            fake = (fake + fake_flip) / 2                # average

    return postprocess(fake.cpu())


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def run_batch(args: argparse.Namespace, G: UNetGenerator, device: torch.device):
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not paths:
        raise FileNotFoundError(f"No supported images found in {in_dir}")

    print(f"Processing {len(paths)} images …")
    grids = []

    for img_path in paths:
        ct   = preprocess(str(img_path), args.img_size)
        fake = predict(G, ct, device, tta=args.tta)

        out_path = out_dir / img_path.name
        vutils.save_image(fake, out_path)

        if args.save_comparison:
            ct_01 = postprocess(ct)
            grids.append(torch.cat([ct_01, fake], dim=-1))  # side-by-side

    if args.save_comparison and grids:
        comparison = torch.cat(grids, dim=0)                # stack all pairs
        vutils.save_image(comparison, out_dir / "_comparison_grid.png", nrow=1)
        print(f"  ✓ Comparison grid → {out_dir / '_comparison_grid.png'}")

    print(f"✓ Done. Outputs saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G = load_generator(args.generator_ckpt, args.in_channels, args.out_channels, device)

    # Pix2Pix evaluates with dropout ON by default (the stochastic generator).
    # Pass --deterministic to use eval mode instead.
    if args.deterministic:
        G.eval()
        print("  Note: dropout disabled (deterministic mode)")
    else:
        G.train()   # keeps dropout active
        print("  Note: dropout active (stochastic inference — Pix2Pix default)")

    if args.input_path:
        # ── Single image ──────────────────────────────────────────────────
        ct   = preprocess(args.input_path, args.img_size)
        fake = predict(G, ct, device, tta=args.tta)
        out  = Path(args.output_path or "simulated_us.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        vutils.save_image(fake, out)
        print(f"✓ Saved → {out}")

    else:
        # ── Batch ─────────────────────────────────────────────────────────
        run_batch(args, G, device)


if __name__ == "__main__":
    main()
