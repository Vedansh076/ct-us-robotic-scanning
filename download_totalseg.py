#!/usr/bin/env python3
"""
download_totalseg.py
====================
Downloads the TotalSegmentator small dataset (102 subjects) from Zenodo
and extracts only the first N subjects to save disk space.

Each extracted subject contains:
  s{id}/ct.nii.gz                  — full abdominal CT volume
  s{id}/segmentations/body.nii.gz  — body/skin mask (for mesh)
  s{id}/segmentations/vertebrae_*.nii.gz, rib_*.nii.gz, etc. — bone masks

Usage:
  python download_totalseg.py --output-dir totalseg_patients --n-subjects 5
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests

ZENODO_ZIP_URL = (
    "https://zenodo.org/api/records/10047263/files/"
    "Totalsegmentator_dataset_small_v201.zip/content"
)

# Bone structure names we want to merge into bone_label volume
BONE_STRUCTURES = [
    # Vertebrae
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
    "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
    # Ribs
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
    "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10",
    "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5",
    "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10",
    "rib_right_11", "rib_right_12",
    # Pelvis / sacrum
    "sacrum", "hip_left", "hip_right",
    # Sternum / shoulders
    "sternum", "clavicula_left", "clavicula_right",
    "scapula_left", "scapula_right",
]


def download_zip_streaming(url: str, output_dir: Path, n_subjects: int) -> None:
    """
    Stream the Zenodo ZIP and extract only the first n_subjects subject folders.
    Uses zipfile streaming so we never load the whole archive into memory.
    """
    print(f"[download] Streaming from:\n  {url}\n")
    print(f"[download] Extracting first {n_subjects} subjects to: {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # We must download the full ZIP first because Python's zipfile needs
    # random access for the central directory. Stream to a temp file.
    tmp_zip = output_dir / "_totalseg_download.zip"

    if tmp_zip.exists():
        print(f"[download] Found cached ZIP at {tmp_zip}, skipping download.")
    else:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1 << 20  # 1 MB chunks

        with open(tmp_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / total * 100 if total else 0
                    mb = downloaded / 1e6
                    print(f"\r[download] {mb:.0f} MB / {total/1e6:.0f} MB  ({pct:.1f}%)", end="", flush=True)
        print(f"\n[download] Download complete: {tmp_zip}")

    print("[extract] Scanning ZIP for subject folders …")
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        all_names = zf.namelist()

        # Find all top-level subject folders (s0001/, s0002/, etc.)
        subjects_seen = set()
        for name in all_names:
            parts = name.split("/")
            if parts[0].startswith("s") and parts[0][1:].isdigit():
                subjects_seen.add(parts[0])

        subjects_sorted = sorted(subjects_seen)[:n_subjects]
        print(f"[extract] Found {len(subjects_seen)} subjects total. Extracting: {subjects_sorted}")

        for subj in subjects_sorted:
            subj_out = output_dir / subj
            if subj_out.exists() and (subj_out / "ct.nii.gz").exists():
                print(f"[extract] {subj} already extracted, skipping.")
                continue
            subj_out.mkdir(parents=True, exist_ok=True)

            # Extract only files belonging to this subject
            subj_files = [n for n in all_names if n.startswith(subj + "/")]
            for member in subj_files:
                # Strip the top-level subject folder prefix so files land in subj_out
                rel = "/".join(member.split("/")[1:])
                if not rel:
                    continue
                dest = subj_out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            print(f"[extract] ✓ {subj}  ({len(subj_files)} files)")

    print(f"\n[done] {n_subjects} subjects extracted to: {output_dir}")
    print(f"[info]  Temporary ZIP kept at {tmp_zip} (delete manually to free space)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TotalSegmentator small dataset.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("totalseg_patients"),
        help="Directory to extract subjects into (default: totalseg_patients/)"
    )
    parser.add_argument(
        "--n-subjects", type=int, default=5,
        help="Number of subjects to extract (default: 5)"
    )
    parser.add_argument(
        "--url", type=str, default=ZENODO_ZIP_URL,
        help="Override download URL"
    )
    args = parser.parse_args()

    download_zip_streaming(args.url, args.output_dir, args.n_subjects)


if __name__ == "__main__":
    main()
