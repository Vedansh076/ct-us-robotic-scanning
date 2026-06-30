"""
dataset.py — Paired CT / SimUS slice dataset for supervised image translation.

Each sample is a tuple of:
    ct_tensor   : float32 tensor [1, H, W], normalised to [-1, 1]
    simus_tensor: float32 tensor [1, H, W], normalised to [ 0, 1]
    pose        : float32 tensor (optional pose vector)

Subject-based train/val split is enforced at construction time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Normalisation constants — tuned to the known intensity ranges.
# CT  : approximate bone window covering [-150, 1250] HU → mapped to [-1, 1]
# SimUS: [0, 220] → mapped to [0, 1]          (values already non-negative)
# ---------------------------------------------------------------------------
CT_MIN: float = -200.0
CT_MAX: float = 300.0

SIMUS_MIN: float = 0.0
SIMUS_MAX: float = 220.0


def normalise_ct(arr: np.ndarray) -> np.ndarray:
    """Clip-and-scale CT HU values to [-1, 1]."""
    arr = arr.astype(np.float32)
    arr = np.clip(arr, CT_MIN, CT_MAX)
    arr = (arr - CT_MIN) / (CT_MAX - CT_MIN)   # [0, 1]
    arr = arr * 2.0 - 1.0                        # [-1, 1]
    return arr


def normalise_simus(arr: np.ndarray) -> np.ndarray:
    """Clip-and-scale SimUS values to [0, 1]."""
    arr = arr.astype(np.float32)
    arr = np.clip(arr, SIMUS_MIN, SIMUS_MAX)
    arr = (arr - SIMUS_MIN) / (SIMUS_MAX - SIMUS_MIN)  # [0, 1]
    return arr


def denormalise_ct(tensor: torch.Tensor) -> torch.Tensor:
    """Invert CT normalisation back to original HU range."""
    arr = (tensor + 1.0) / 2.0                          # [0, 1]
    arr = arr * (CT_MAX - CT_MIN) + CT_MIN              # HU
    return arr


def denormalise_simus(tensor: torch.Tensor) -> torch.Tensor:
    """Invert SimUS normalisation back to [0, 220]."""
    return tensor * (SIMUS_MAX - SIMUS_MIN) + SIMUS_MIN


def _subject_from_filename(name: str) -> str:
    """Extract subject ID prefix from a filename like 'tcga-qq-a8vg_00000'.

    Returns the portion before the last underscore-and-digits block, e.g.
    'tcga-qq-a8vg'.  Falls back to the full stem if the pattern is not found.
    """
    m = re.match(r"^(.+?)_\d+$", name)
    return m.group(1) if m else name


class CTSimUSDataset(Dataset):
    """
    Loads paired (CT, SimUS) .npy slices from disk.

    Parameters
    ----------
    root : str | Path
        Root directory that contains subdirectories ``ct/``, ``simus/``, and
        optionally ``poses/``.
    subject_ids : list[str]
        Subjects whose files should be included (e.g. ``["tcga-qq-a8vg"]``).
    ct_transform : callable, optional
        Additional augmentation applied to the CT numpy array *before*
        normalisation.  Should return a numpy array of the same shape.
    simus_transform : callable, optional
        Same but for SimUS.
    load_poses : bool
        Whether to load the corresponding pose .npy file and return it as the
        third element of each sample.  Defaults to False.
    """

    def __init__(
        self,
        root: str | Path,
        subject_ids: list[str],
        ct_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        simus_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        load_poses: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.subject_ids = set(subject_ids)
        self.ct_transform = ct_transform
        self.simus_transform = simus_transform
        self.load_poses = load_poses

        self.ct_dir = self.root / "ct"
        self.simus_dir = self.root / "simus"
        self.pose_dir = self.root / "poses"

        self._samples: list[str] = self._collect_samples()

        if len(self._samples) == 0:
            raise RuntimeError(
                f"No samples found for subjects {subject_ids} under {self.root}. "
                "Check directory structure and subject prefixes."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_samples(self) -> list[str]:
        """Return sorted list of stem names that belong to the wanted subjects."""
        stems: list[str] = []
        for fname in sorted(os.listdir(self.ct_dir)):
            if not fname.endswith(".npy"):
                continue
            stem = fname[: -len(".npy")]
            subject = _subject_from_filename(stem)
            if subject not in self.subject_ids:
                continue
            # Verify matching SimUS file exists
            if not (self.simus_dir / fname).exists():
                raise FileNotFoundError(
                    f"CT file '{fname}' has no matching SimUS counterpart."
                )
            stems.append(stem)
        return stems

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        stem = self._samples[idx]

        ct = np.load(self.ct_dir / f"{stem}.npy")       # (H, W) or (1,H,W)
        sim = np.load(self.simus_dir / f"{stem}.npy")

        # Ensure 2-D
        if ct.ndim == 3:
            ct = ct.squeeze()
        if sim.ndim == 3:
            sim = sim.squeeze()

        # Optional augmentation (e.g. random flips) on raw values
        if self.ct_transform is not None:
            ct = self.ct_transform(ct)
        if self.simus_transform is not None:
            sim = self.simus_transform(sim)

        # Normalise
        ct = normalise_ct(ct)
        sim = normalise_simus(sim)

        # Add channel dim → [1, H, W]
        ct_t = torch.from_numpy(ct).unsqueeze(0)
        sim_t = torch.from_numpy(sim).unsqueeze(0)

        if self.load_poses:
            pose_path = self.pose_dir / f"{stem}.npy"
            pose = np.load(pose_path).astype(np.float32)
            pose_t = torch.from_numpy(pose)
            return ct_t, sim_t, pose_t

        return ct_t, sim_t

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CTSimUSDataset(root='{self.root}', "
            f"subjects={sorted(self.subject_ids)}, "
            f"n_samples={len(self)})"
        )
