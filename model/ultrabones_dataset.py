"""
ultrabones_dataset.py — PyTorch Dataset loader for the UltraBones100k dataset.

Loads B-mode ultrasound images (grayscale PNG) and matching bone surface masks.
Enforces subject/specimen-based split to avoid data leakage.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UltraBonesDataset(Dataset):
    """
    Loads matched ultrasound and bone surface mask frames from the UltraBones100k dataset.

    Directory structure expected:
    data_root/
    ├── specimen01/
    │   └── ultrasound_records/
    │       └── tibia_sweep1/
    │           ├── UltrasoundImages/
    │           │   ├── frame_0000.png
    │           │   └── ...
    │           ├── Labels/
    │           │   ├── frame_0000.png
    │           │   └── ...
    │           └── Labels_full/
    │               ├── frame_0000.png
    │               └── ...
    """

    def __init__(
        self,
        data_root: str | Path,
        specimen_ids: list[int],
        mask_type: str = "Labels",  # "Labels" (directly visible) or "Labels_full" (includes shadow)
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.specimen_ids = set(specimen_ids)
        self.mask_type = mask_type
        self.img_size = img_size
        self.augment = augment

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {self.data_root}")

        self.samples: list[dict[str, Path]] = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for specimen IDs {specimen_ids} under {self.data_root}."
            )

    def _collect_samples(self) -> list[dict[str, Path]]:
        samples = []
        # Find specimen folders (e.g., specimen01 to specimen14)
        for specimen_dir in sorted(self.data_root.iterdir()):
            if not specimen_dir.is_dir():
                continue
            
            # Match folder name 'specimenNN'
            m = re.match(r"^specimen(\d+)$", specimen_dir.name)
            if not m:
                continue
            
            specimen_num = int(m.group(1))
            if specimen_num not in self.specimen_ids:
                continue

            records_dir = specimen_dir / "ultrasound_records"
            if not records_dir.exists():
                continue

            # Walk through each sweep/record folder
            for sweep_dir in sorted(records_dir.iterdir()):
                if not sweep_dir.is_dir():
                    continue

                us_dir = sweep_dir / "UltrasoundImages"
                label_dir = sweep_dir / self.mask_type

                if not us_dir.exists() or not label_dir.exists():
                    continue

                # Collect all images and match with their labels
                for img_path in sorted(us_dir.glob("*.png")):
                    lbl_path = label_dir / img_path.name
                    if lbl_path.exists():
                        samples.append({
                            "img": img_path,
                            "label": lbl_path,
                            "specimen": specimen_num,
                            "sweep": sweep_dir.name
                        })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_paths = self.samples[idx]

        # Load image & mask as PIL Grayscale Images
        img = Image.open(sample_paths["img"]).convert("L")
        lbl = Image.open(sample_paths["label"]).convert("L")

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        lbl = lbl.resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert to numpy
        img_arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        lbl_arr = np.array(lbl, dtype=np.float32) / 255.0  # [0, 1]

        # Apply augmentation (if active)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_arr = np.fliplr(img_arr).copy()
                lbl_arr = np.fliplr(lbl_arr).copy()

        # Convert to PyTorch Tensors and add channel dimension (C, H, W)
        img_t = torch.from_numpy(img_arr).unsqueeze(0)
        lbl_t = torch.from_numpy(lbl_arr).unsqueeze(0)

        # Binarize label just in case resizing/interpolation introduced floating points
        lbl_t = (lbl_t > 0.5).float()

        return img_t, lbl_t

    def __repr__(self) -> str:
        return (
            f"UltraBonesDataset(root='{self.data_root}', "
            f"specimens={sorted(self.specimen_ids)}, "
            f"mask_type='{self.mask_type}', "
            f"n_samples={len(self)})"
        )
