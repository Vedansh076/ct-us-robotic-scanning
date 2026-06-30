"""
discriminator.py — 70×70 PatchGAN Discriminator for CT→SimUS Pix2Pix
======================================================================
Style follows the original model.py conventions:
  - Module-level building-block classes
  - InstanceNorm (not BatchNorm) — consistent with the generator
  - DCGAN weight init matching the updated model.py _init_weights
  - Docstrings in the same numpy-style format

WHY PATCHGAN?
-------------
A standard global discriminator collapses high-frequency texture judgements
into one scalar, which lets the generator cheat with plausible low-frequency
content while producing blurry fine detail.  PatchGAN instead classifies
each overlapping 70×70 patch independently and averages the patch logits for
the loss.  This directly penalises local blurriness — the exact pathology in
the original U-Net predictions.

WHY CONDITIONAL (concat CT)?
------------------------------
The discriminator receives (CT, SimUS) concatenated along the channel axis.
Without the condition, D only asks "does this look like an ultrasound?"
With the condition, D asks "does this SimUS look *physically plausible given
this CT slice*?".  The generator is therefore forced to maintain spatial
correspondence, not just produce generic-looking ultrasound textures.

WHY INSTANCENORM HERE (NOT BATCHNORM)?
---------------------------------------
The original pipeline uses InstanceNorm throughout, which is correct for
small batch sizes and image-translation tasks (per-image normalisation avoids
coupling statistics across subjects).  The Pix2Pix paper uses BatchNorm in
the discriminator, but InstanceNorm is a safe drop-in that is more robust
at batch_size ≤ 4.  It is set with affine=True to retain representational
capacity, matching the generator's ConvBlock convention.

WHY BCEWithLogitsLoss (NO SIGMOID IN FORWARD)?
------------------------------------------------
Numerical stability: BCEWithLogitsLoss fuses the sigmoid and binary-cross-
entropy into a single log-sum-exp operation, avoiding floating-point overflow
at the extremes of the logit range.  This is especially important with AMP
(fp16).  The sigmoid is therefore NOT part of the discriminator forward pass;
it is applied only during visualisation in the training script.

INPUT / OUTPUT SHAPES (for 256×256 inputs)
--------------------------------------------
  condition   : (B, 1, 256, 256)  — CT slice,       normalised to [-1, 1]
  target      : (B, 1, 256, 256)  — real or fake SimUS, normalised to [-1, 1]
  concat input: (B, 2, 256, 256)
  output      : (B, 1, 30, 30)    — patch logit map (not probabilities)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class DiscConvBlock(nn.Module):
    """
    Conv2d → InstanceNorm2d → LeakyReLU(0.2)

    The same stride-2 / same-padding structure as the Pix2Pix discriminator.
    InstanceNorm is omitted on the first layer (paper convention; no statistics
    to normalise when the input is raw pixels).

    Parameters
    ----------
    in_ch      : input channels
    out_ch     : output channels
    stride     : 2 for downsampling blocks, 1 for the pre-output block
    normalize  : False for the first block only
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                      padding=1, bias=not normalize),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 70×70 PatchGAN
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator for conditioned image translation.

    Parameters
    ----------
    in_channels  : int  — channels of the condition image (CT), default 1
    out_channels : int  — channels of the target image (SimUS), default 1
    features     : int  — base feature width, default 64

    Forward signature
    -----------------
    forward(condition, target) → logit_map (B, 1, ~H/8, ~W/8)

    Pass the logit map directly to BCEWithLogitsLoss — no sigmoid here.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
    ) -> None:
        super().__init__()

        combined = in_channels + out_channels   # 2

        # Block 1: no InstanceNorm on raw pixel input
        # 256 → 128
        self.block1 = DiscConvBlock(combined,      features,     stride=2, normalize=False)
        # Block 2: 128 → 64
        self.block2 = DiscConvBlock(features,      features * 2, stride=2)
        # Block 3: 64 → 32
        self.block3 = DiscConvBlock(features * 2,  features * 4, stride=2)
        # Block 4: stride=1 — spatial resolution held at ~32 before final conv
        self.block4 = DiscConvBlock(features * 4,  features * 8, stride=1)
        # Output: 1-channel logit map (no activation — BCEWithLogitsLoss used externally)
        self.output = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """DCGAN weight init — mirrors model.py _init_weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias.data)

    # ------------------------------------------------------------------

    def forward(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        condition : (B, 1, H, W) — CT slice in [-1, 1]
        target    : (B, 1, H, W) — SimUS (real or generated) in [-1, 1]

        Returns
        -------
        (B, 1, ~H/8, ~W/8) — raw logits; high value = D thinks 'real'
        """
        x = torch.cat([condition, target], dim=1)   # (B, 2, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"PatchGANDiscriminator(params={self.count_parameters():,})"