"""
model.py — U-Net for paired CT-to-SimUS image translation.

Architecture choices
--------------------
* Encoder: 5 levels of (Conv → InstanceNorm → LeakyReLU) × 2 + MaxPool
* Bottleneck: 2 × (Conv → InstanceNorm → LeakyReLU)
* Decoder: bilinear upsample + concat skip + (Conv → InstanceNorm → ReLU) × 2
* Final: 1×1 Conv → Sigmoid  (output in [0, 1] to match SimUS normalisation)

InstanceNorm is preferred over BatchNorm for image-translation tasks:
it normalises per-image statistics, which is more robust when the batch
size is small and avoids coupling statistics across subjects.

The model is fully convolutional → supports any input resolution, but was
designed and tested for 256 × 256 inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
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
    """ConvBlock followed by 2×2 MaxPool (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, act=nn.LeakyReLU)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip          # down → next encoder level, skip → decoder


class UpBlock(nn.Module):
    """Bilinear upsample, concat skip, then ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for 1-channel CT → 1-channel SimUS translation.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 for single-channel CT slices).
    out_channels : int
        Number of output channels (1 for single-channel SimUS slices).
    base_features : int
        Channel width at the first encoder level.  Doubles at each level.
        Default 64 → levels are [64, 128, 256, 512, 512(bottleneck)].
    dropout : float
        Dropout probability applied *before* each UpBlock.  0 disables it.
        Useful when training data is limited (~9 k samples).
    """

    ENCODER_CHANNELS = (1, 2, 4, 8)   # multipliers of base_features

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        f = base_features
        enc_ch = [f * m for m in self.ENCODER_CHANNELS]   # [64,128,256,512]
        bot_ch = enc_ch[-1]                                # 512  (bottleneck)

        # ----- Encoder -----
        self.enc1 = DownBlock(in_channels, enc_ch[0])
        self.enc2 = DownBlock(enc_ch[0],   enc_ch[1])
        self.enc3 = DownBlock(enc_ch[1],   enc_ch[2])
        self.enc4 = DownBlock(enc_ch[2],   enc_ch[3])

        # ----- Bottleneck -----
        self.bottleneck = ConvBlock(enc_ch[3], bot_ch)

        # ----- Decoder -----
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.dec4 = UpBlock(bot_ch,    enc_ch[3], enc_ch[2])
        self.dec3 = UpBlock(enc_ch[2], enc_ch[2], enc_ch[1])
        self.dec2 = UpBlock(enc_ch[1], enc_ch[1], enc_ch[0])
        self.dec1 = UpBlock(enc_ch[0], enc_ch[0], enc_ch[0] // 2)

        # ----- Head -----
        self.head = nn.Conv2d(enc_ch[0] // 2, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()   # SimUS target is normalised to [0,1]

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, H, W) — normalised CT + Seg in [-1, 1] / [0, 1]


        Returns
        -------
        (B, 1, H, W) — predicted SimUS in [0, 1]
        """
        # Encoder
        x1, s1 = self.enc1(x)    # s1: (B, 64,  H,   W  )
        x2, s2 = self.enc2(x1)   # s2: (B, 128, H/2, W/2)
        x3, s3 = self.enc3(x2)   # s3: (B, 256, H/4, W/4)
        x4, s4 = self.enc4(x3)   # s4: (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(x4)   # (B, 512, H/16, W/16)

        # Decoder (dropout applied to bottleneck / deepest skip for regularisation)
        d4 = self.dec4(self.drop(b),  self.drop(s4))
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_act(self.head(d1))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UNet(params={self.count_parameters():,}, "
            f"dropout={self.drop.p if isinstance(self.drop, nn.Dropout2d) else 0})"
        )
