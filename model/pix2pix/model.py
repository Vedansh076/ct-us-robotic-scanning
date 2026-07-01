"""
model.py — U-Net Generator for CT→SimUS Pix2Pix
=================================================
This file is a minimal adaptation of the original model.py.
Every original design decision is preserved unless it conflicts with the
Pix2Pix objective.  Three changes are made and documented below.

ARCHITECTURE SUMMARY (unchanged)
----------------------------------
Encoder : 4 × DownBlock  (ConvBlock + MaxPool, LeakyReLU, InstanceNorm)
          channels: 1 → 64 → 128 → 256 → 512
Bottleneck : ConvBlock  512 → 512
Decoder : 4 × UpBlock  (bilinear upsample + skip concat + ConvBlock, ReLU, InstanceNorm)
Head    : 1×1 Conv → [activation]

CHANGE 1 — Output activation: Sigmoid → Tanh
----------------------------------------------
Original: self.out_act = nn.Sigmoid()   # output in [0, 1]
Changed : self.out_act = nn.Tanh()      # output in [-1, 1]

Why: the discriminator receives both real and generated SimUS images.
If real images are in [0, 1] (as dataset.py produces with normalise_simus)
but fake images are in [-1, 1], the discriminator trivially separates them
by output range alone — the GAN signal becomes meaningless.

Resolution adopted here: keep Tanh on the generator, and rescale real SimUS
from [0, 1] to [-1, 1] in the training loop before passing to the
discriminator.  Predictions are rescaled back to [0, 1] at inference time.
The L1 loss also operates in [-1, 1] space (both fake and rescaled real).

This is consistent, keeps dataset.py untouched, and matches the Pix2Pix
convention of feeding both modalities in [-1, 1] to the network.

CHANGE 2 — Dropout schedule: single shared drop → per-block control
---------------------------------------------------------------------
Original: self.drop = nn.Dropout2d(p) applied only to bottleneck and s4.
Changed : a boolean `pix2pix_dropout` flag activates Dropout2d(0.5) inside
          the first three UpBlocks (dec4, dec3, dec2), while keeping the
          original lightweight dropout on the bottleneck/deepest skip.

Why: the Pix2Pix paper applies dropout to the top three decoder layers as
the stochastic source that prevents mode collapse.  The original pipeline
used a single Dropout2d for mild regularisation; both effects are retained.

The `dropout` float argument remains and controls the bottleneck/deepest-skip
dropout exactly as before.  `pix2pix_dropout=True` (default when used as the
Pix2Pix generator) additionally activates the decoder dropout.

CHANGE 3 — Weight initialisation: Kaiming → DCGAN normal
----------------------------------------------------------
Original: nn.init.kaiming_normal_ for Conv layers.
Changed : nn.init.normal_(mean=0, std=0.02) for Conv/ConvTranspose;
          nn.init.normal_(mean=1, std=0.02) for InstanceNorm weight.

Why: GAN training with Adam(beta1=0.5) is sensitive to the initial gradient
landscape.  Kaiming init is optimal for networks trained with a single
supervised loss; DCGAN init (Radford et al. 2015) is empirically more stable
when an adversarial objective is added.  InstanceNorm bias stays at 0.

NOTE: this only affects training from scratch.  When fine-tuning from a
pre-trained U-Net checkpoint, load the checkpoint *after* constructing the
model — the loaded weights override this init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks  (UNCHANGED from original)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two convolutional layers, each followed by InstanceNorm + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            act(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            act(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2×2 MaxPool (halves spatial dims).  UNCHANGED."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, act=nn.LeakyReLU)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Bilinear upsample, concat skip, then ConvBlock.  UNCHANGED."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net generator for CT → SimUS Pix2Pix.

    Identical to the original UNet class except for the three documented
    changes above (output activation, dropout schedule, weight init).

    Parameters
    ----------
    in_channels : int     — input channels (1 for single-channel CT)
    out_channels : int    — output channels (1 for single-channel SimUS)
    base_features : int   — encoder width at level 1 (default 64)
    dropout : float       — bottleneck / deepest-skip dropout prob (original behaviour)
    pix2pix_dropout : bool
                          — also activate Dropout2d(0.5) on decoder blocks
                            dec4, dec3, dec2 (Pix2Pix paper, Section 6.1.1)
    """

    ENCODER_CHANNELS = (1, 2, 4, 8)

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 64,
        dropout: float = 0.1,           # unchanged: bottleneck regularisation
        pix2pix_dropout: bool = True,   # NEW: decoder dropout for GAN
    ) -> None:
        super().__init__()
        f = base_features
        enc_ch = [f * m for m in self.ENCODER_CHANNELS]
        bot_ch = enc_ch[-1]

        # ----- Encoder (UNCHANGED) -----
        self.enc1 = DownBlock(in_channels, enc_ch[0])
        self.enc2 = DownBlock(enc_ch[0],   enc_ch[1])
        self.enc3 = DownBlock(enc_ch[1],   enc_ch[2])
        self.enc4 = DownBlock(enc_ch[2],   enc_ch[3])

        # ----- Bottleneck (UNCHANGED) -----
        self.bottleneck = ConvBlock(enc_ch[3], bot_ch)

        # ----- Original dropout (UNCHANGED behaviour) -----
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ----- Pix2Pix decoder dropout (CHANGE 2) -----
        # Applied inside the forward pass on the outputs of dec4, dec3, dec2
        self.dec_drop = nn.Dropout2d(0.5) if pix2pix_dropout else nn.Identity()

        # ----- Decoder (UNCHANGED) -----
        self.dec4 = UpBlock(bot_ch,    enc_ch[3], enc_ch[2])
        self.dec3 = UpBlock(enc_ch[2], enc_ch[2], enc_ch[1])
        self.dec2 = UpBlock(enc_ch[1], enc_ch[1], enc_ch[0])
        self.dec1 = UpBlock(enc_ch[0], enc_ch[0], enc_ch[0] // 2)

        # ----- Head (CHANGE 1: Tanh instead of Sigmoid) -----
        self.head    = nn.Conv2d(enc_ch[0] // 2, out_channels, kernel_size=1)
        self.out_act = nn.Tanh()    # was nn.Sigmoid() — see module docstring

        # ----- Weight init (CHANGE 3: DCGAN normal init) -----
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """DCGAN / Pix2Pix weight initialisation (CHANGE 3)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias.data)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) — CT normalised to [-1, 1]

        Returns
        -------
        (B, 1, H, W) — predicted SimUS in [-1, 1]  (CHANGE 1: was [0, 1])
        """
        # Encoder (UNCHANGED)
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        # Bottleneck + original dropout (UNCHANGED)
        b = self.bottleneck(x4)

        # Decoder — original dropout on bottleneck/deepest skip (UNCHANGED),
        # plus Pix2Pix dropout on the output of the top three decoder blocks
        d4 = self.dec_drop(self.dec4(self.drop(b),  self.drop(s4)))  # CHANGE 2
        d3 = self.dec_drop(self.dec3(d4, s3))                         # CHANGE 2
        d2 = self.dec_drop(self.dec2(d3, s2))                         # CHANGE 2
        d1 = self.dec1(d2, s1)                                        # unchanged

        return self.out_act(self.head(d1))

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        p = self.drop.p if isinstance(self.drop, nn.Dropout2d) else 0
        d = isinstance(self.dec_drop, nn.Dropout2d)
        return (
            f"UNet(params={self.count_parameters():,}, "
            f"bottleneck_dropout={p}, pix2pix_decoder_dropout={d})"
        )