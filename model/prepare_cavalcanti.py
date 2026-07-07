#!/usr/bin/env python3
"""
prepare_cavalcanti.py — Preprocess the Cavalcanti Robotic Lumbar Spine Dataset.

Performs full 3D oblique reslicing of CT volumes to extract slices that
correspond to tracked robotic ultrasound frames.  Only robotic sweeps
(R1–R3) are used, following the constant 5 N force protocol.

Pipeline per volunteer
----------------------
1. Load DICOM CT volume → 3D HU array + voxel-to-world affine
2. Threshold bone mask (HU > 300)
3. Load STL bone meshes (in CT patient coordinates)
4. Parse RUS_pose.txt  → 4×4 end-effector transforms
5. Parse body_marker.csv → breathing-compensation transforms
6. Register tracked positions to CT space (PCA pre-align + ICP)
7. Per US frame (with --stride subsampling):
     a. T_CT = T_reg · T_breath · T_robot   → probe pose in CT coords
     b. Oblique reslice CT volume and bone mask along imaging plane
     c. Load and preprocess US PNG frame
     d. Resize all to 256×256, save as .npy

Output structure
----------------
    output_dir/
      ct/       URS01_R1_00000.npy   (float32, HU)
      labels/   URS01_R1_00000.npy   (float32, binary 0/1)
      simus/    URS01_R1_00000.npy   (float32, scaled to [0,220])
      diagnostics/   sample alignment PNGs
      meta.json      {volunteers, n_samples, split}

Usage
-----
    # Discover dataset structure
    python prepare_cavalcanti.py --data_root ../data/Cavalcanti --discover

    # Full preprocessing
    python prepare_cavalcanti.py \\
        --data_root ../data/Cavalcanti \\
        --output_dir ../data/cavalcanti_processed \\
        --stride 5

    # Process specific volunteers
    python prepare_cavalcanti.py \\
        --data_root ../data/Cavalcanti \\
        --output_dir ../data/cavalcanti_processed \\
        --volunteers URS01 URS02 URS03
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import struct
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

try:
    import pydicom
except ImportError:
    sys.exit("ERROR: pydicom not installed.  Run:  pip install pydicom")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cavalcanti")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BONE_HU_THRESHOLD = 300          # HU above which voxel is classified bone
DEFAULT_IMG_SIZE  = 256
DEFAULT_STRIDE    = 5            # process every Nth US frame
SIMUS_SCALE       = 220.0 / 255.0  # map uint8 [0,255] → [0,220] for dataset.py

# US transducer field of view (typical lumbar spine linear probe, mm)
US_FOV_WIDTH_MM = 50.0
US_FOV_DEPTH_MM = 60.0

# Approximate skin-to-bone depth offset (mm) to improve ICP initialisation.
# Tracked probe positions sit on the skin surface; STL meshes are bone.
DEPTH_OFFSET_MM  = 25.0

# Euler angle auto-detection threshold (radians vs degrees)
EULER_DEG_THRESH = 2.0 * np.pi


# ═══════════════════════════════════════════════════════════════════════════
# Geometry utilities
# ═══════════════════════════════════════════════════════════════════════════

def euler_to_T(x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Position + Euler XYZ (rad) → 4×4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3]  = [x, y, z]
    return T


def quat_to_T(x: float, y: float, z: float,
              qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Position + quaternion (x,y,z,w) → 4×4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3]  = [x, y, z]
    return T


def _pca_pre_align(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Centroid + PCA axis alignment.  Returns (4,4) rigid transform."""
    sc = src.mean(0)
    tc = tgt.mean(0)
    _, _, Vs = np.linalg.svd(src - sc, full_matrices=False)
    _, _, Vt = np.linalg.svd(tgt - tc, full_matrices=False)
    R = Vt.T @ Vs
    if np.linalg.det(R) < 0:          # fix reflection
        Vt[-1] *= -1
        R = Vt.T @ Vs
    t = tc - R @ sc
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def icp(source: np.ndarray, target: np.ndarray,
        init: Optional[np.ndarray] = None,
        max_iter: int = 120, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    ICP point-to-point registration.

    Parameters
    ----------
    source  (N,3) in source frame
    target  (M,3) in target frame
    init    optional (4,4) initial transform

    Returns
    -------
    T_total  (4,4) source → target
    rmse     final root-mean-square residual
    """
    src = source.copy()
    if init is not None:
        src = (init[:3, :3] @ src.T).T + init[:3, 3]
        T_total = init.copy()
    else:
        T_total = np.eye(4)

    tree = KDTree(target)
    prev_err = np.inf

    for _ in range(max_iter):
        dists, idx = tree.query(src)
        matched = target[idx]

        sc = src.mean(0);     tc = matched.mean(0)
        H  = (src - sc).T @ (matched - tc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tc - R @ sc

        step = np.eye(4); step[:3, :3] = R; step[:3, 3] = t
        src = (R @ src.T).T + t
        T_total = step @ T_total

        err = float(np.sqrt(np.mean(dists ** 2)))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    rmse = float(np.sqrt(np.mean(tree.query(src)[0] ** 2)))
    return T_total, rmse


# ═══════════════════════════════════════════════════════════════════════════
# File I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_inner(d: str) -> str:
    """Navigate the nested dir: URS01_CT_raw/URS1_CT_raw/..."""
    subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    return os.path.join(d, subs[0]) if len(subs) == 1 else d


def load_stl(path: str) -> np.ndarray:
    """Load unique vertices from an STL file (binary or ASCII).  → (N,3)."""
    with open(path, "rb") as f:
        header = f.read(80)

    # ASCII heuristic: starts with 'solid' and NOT followed by binary data
    is_ascii = False
    try:
        txt = header.decode("ascii", errors="ignore").strip().lower()
        if txt.startswith("solid"):
            with open(path, "r") as f:
                first_lines = [f.readline() for _ in range(3)]
            if any("facet" in l.lower() or "vertex" in l.lower()
                   for l in first_lines):
                is_ascii = True
    except Exception:
        pass

    verts: list = []
    if is_ascii:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("vertex"):
                    parts = line.split()
                    verts.append([float(parts[1]),
                                  float(parts[2]),
                                  float(parts[3])])
    else:
        with open(path, "rb") as f:
            f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            for _ in range(n_tri):
                f.read(12)                     # normal
                for _ in range(3):
                    verts.append(struct.unpack("<3f", f.read(12)))
                f.read(2)                      # attrib

    arr = np.array(verts, dtype=np.float32)
    return np.unique(arr, axis=0)              # deduplicate shared vertices


def load_all_stl_vertices(stl_dir: str) -> np.ndarray:
    """Load and concatenate all STL files in a directory.  → (N,3)."""
    all_verts: list = []
    for f in sorted(os.listdir(stl_dir)):
        if f.lower().endswith(".stl"):
            fp = os.path.join(stl_dir, f)
            try:
                v = load_stl(fp)
                all_verts.append(v)
                log.debug("  STL %s → %d vertices", f, len(v))
            except Exception as e:
                log.warning("  Failed to load %s: %s", fp, e)
    if not all_verts:
        raise FileNotFoundError(f"No valid STL files in {stl_dir}")
    combined = np.concatenate(all_verts, axis=0)
    return np.unique(combined, axis=0)


# ---------------------------------------------------------------------------
# Pose / marker parsing
# ---------------------------------------------------------------------------

def parse_robot_poses(pose_file: str) -> List[np.ndarray]:
    """
    Parse RUS_pose.txt → list of 4×4 transforms.
    Expected format per line: x  y  z  roll  pitch  yaw
    Auto-detects degrees vs radians.
    """
    rows: list = []
    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("x"):
                continue                              # header / comment
            parts = re.split(r"[,\s\t]+", line)
            parts = [p for p in parts if p]
            if len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts[:6]]
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        return []

    # Auto-detect degrees: if any angle > 2π, assume degrees
    angles = np.array([r[3:6] for r in rows])
    if np.any(np.abs(angles) > EULER_DEG_THRESH):
        log.info("  Detected Euler angles in DEGREES; converting to radians")
        for r in rows:
            r[3] = np.deg2rad(r[3])
            r[4] = np.deg2rad(r[4])
            r[5] = np.deg2rad(r[5])

    return [euler_to_T(*r) for r in rows]


def parse_body_markers(marker_file: str) -> List[Tuple[np.ndarray, float]]:
    """
    Parse body_marker.csv → [(4×4 transform, timestamp), ...].
    Format: x, y, z, qx, qy, qz, qw, timestamp
    """
    markers: list = []
    with open(marker_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip header lines
            if len(row) < 7:
                continue
            try:
                vals = [float(v) for v in row[:8]]
            except ValueError:
                continue
            if len(vals) >= 8:
                T = quat_to_T(*vals[:7])
                markers.append((T, vals[7]))
            elif len(vals) >= 7:
                T = quat_to_T(*vals[:7])
                markers.append((T, float(len(markers))))
    return markers


def breathing_compensation(markers: List[Tuple[np.ndarray, float]],
                           n_frames: int) -> List[np.ndarray]:
    """
    Compute per-frame breathing compensation transforms.

    Returns a list of (4,4) transforms that undo body motion relative
    to the first frame.  If markers are empty, returns identity transforms.
    """
    if not markers:
        return [np.eye(4)] * n_frames

    T_ref = markers[0][0]
    T_ref_inv = np.linalg.inv(T_ref)

    # If we have exactly as many markers as frames, use 1:1 mapping
    if len(markers) == n_frames:
        return [T_ref_inv @ m[0] for m, _ in zip(markers, range(n_frames))]

    # Otherwise, spread evenly
    comps: list = []
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1) * (len(markers) - 1)
        idx  = int(frac)
        idx  = min(idx, len(markers) - 1)
        T_cur = markers[idx][0]
        comps.append(T_ref_inv @ T_cur)
    return comps


# ---------------------------------------------------------------------------
# DICOM volume loading
# ---------------------------------------------------------------------------

def load_dicom_volume(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DICOM series and return the CT volume in HU plus the affine.

    Returns
    -------
    volume : (n_slices, n_rows, n_cols) float32 — Hounsfield units
    affine : (4,4) — maps voxel index (col, row, slice) → world mm
    """
    dcm_list: list = []
    for fname in sorted(os.listdir(dicom_dir)):
        fp = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, force=True)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "pixel_array"):
                dcm_list.append(ds)
        except Exception:
            continue

    if not dcm_list:
        raise FileNotFoundError(f"No DICOM images in {dicom_dir}")

    dcm_list.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    n = len(dcm_list)
    rows = dcm_list[0].Rows
    cols = dcm_list[0].Columns

    volume = np.zeros((n, rows, cols), dtype=np.float32)
    for i, ds in enumerate(dcm_list):
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[i] = arr * slope + intercept

    # ---- affine ----
    ds0 = dcm_list[0]
    iop = np.array(ds0.ImageOrientationPatient, dtype=np.float64)
    row_dir = iop[:3]          # increasing column index
    col_dir = iop[3:]          # increasing row index
    ps      = np.array(ds0.PixelSpacing, dtype=np.float64)  # [row_sp, col_sp]
    ipp0    = np.array(ds0.ImagePositionPatient, dtype=np.float64)

    if n > 1:
        ipp_last  = np.array(dcm_list[-1].ImagePositionPatient, dtype=np.float64)
        slice_vec = (ipp_last - ipp0) / (n - 1)
    else:
        sl_thick  = float(getattr(ds0, "SliceThickness", 1.0))
        slice_vec = np.cross(row_dir, col_dir)
        slice_vec = slice_vec / np.linalg.norm(slice_vec) * sl_thick

    affine = np.eye(4)
    affine[:3, 0] = row_dir * ps[1]     # Δcol → world
    affine[:3, 1] = col_dir * ps[0]     # Δrow → world
    affine[:3, 2] = slice_vec            # Δslice → world
    affine[:3, 3] = ipp0                 # origin

    log.info("  DICOM volume: %s  spacing: %.2f×%.2f×%.2f mm",
             volume.shape,
             ps[1], ps[0], np.linalg.norm(slice_vec))
    return volume, affine


def select_ct_sequence(inner_ct_dir: str) -> str:
    """Pick the best CT reconstruction from available subdirectories."""
    seq_dirs = [d for d in sorted(os.listdir(inner_ct_dir))
                if os.path.isdir(os.path.join(inner_ct_dir, d))
                and d.startswith("CT_Seq")]

    if not seq_dirs:
        raise FileNotFoundError(f"No CT_Seq.* directories in {inner_ct_dir}")

    # Priority: thin slice (0,60), sharp kernel (Br60), large FOV (Torso)
    for keywords in [
        ("0,60", "Br60", "Torso"),
        ("0,60", "Br60"),
        ("0,60", "Torso"),
        ("0,60",),
    ]:
        for d in seq_dirs:
            if all(k in d for k in keywords):
                # Skip topograms, sagittals, coronals, patient protocol, 3D
                skip = any(s in d.lower() for s in
                           ["topogram", "sag", "cor", "protocol", "1003"])
                if not skip:
                    return d

    # Last resort: first non-special sequence
    for d in seq_dirs:
        skip = any(s in d.lower() for s in
                   ["topogram", "sag", "cor", "protocol", "1003"])
        if not skip:
            return d

    return seq_dirs[0]


# ---------------------------------------------------------------------------
# Oblique reslicing
# ---------------------------------------------------------------------------

def oblique_reslice(volume: np.ndarray, affine: np.ndarray,
                    center_world: np.ndarray,
                    right_dir: np.ndarray, down_dir: np.ndarray,
                    fov_w: float, fov_d: float,
                    size: int) -> np.ndarray:
    """
    Sample a 2-D oblique slice from a 3-D volume.

    Parameters
    ----------
    volume       (D, H, W) — the 3-D volume
    affine       (4, 4)    — maps (col, row, slice) → world mm
    center_world (3,)      — plane centre in world mm
    right_dir    (3,)      — unit vector, left→right in the slice
    down_dir     (3,)      — unit vector, top→bottom  (into tissue)
    fov_w, fov_d           — field of view in mm (width, depth)
    size                   — output pixel size  (size × size)

    Returns
    -------
    (size, size) float32 slice
    """
    inv_aff = np.linalg.inv(affine)

    u = np.linspace(-fov_w / 2, fov_w / 2, size)   # lateral
    v = np.linspace(0,           fov_d,     size)   # depth
    uu, vv = np.meshgrid(u, v)                      # (size, size)

    # World coordinates of every pixel in the slice plane
    world = (center_world[None, None, :]
             + uu[..., None] * right_dir[None, None, :]
             + vv[..., None] * down_dir[None, None, :])

    pts = world.reshape(-1, 3)
    homo = np.column_stack([pts, np.ones(len(pts))])
    vox  = (inv_aff @ homo.T).T[:, :3]              # (N, 3) = (col, row, slice)

    # map_coordinates expects (axis0=slice, axis1=row, axis2=col)
    coords = np.array([vox[:, 2], vox[:, 1], vox[:, 0]])
    result = map_coordinates(volume, coords, order=1,
                             mode="constant", cval=0.0)
    return result.reshape(size, size).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Discovery mode
# ═══════════════════════════════════════════════════════════════════════════

def discover(data_root: str) -> None:
    """Print a structured summary of the Cavalcanti dataset."""
    root = Path(data_root)
    ct_root  = root / "computed_tomography"
    us_root  = root / "ultrasound"
    val_root = root / "us_spatial_tracking_validation"

    print("=" * 70)
    print("CAVALCANTI DATASET DISCOVERY")
    print("=" * 70)
    print(f"Root: {root}\n")

    # -- top-level files --
    top_files = [f for f in os.listdir(root) if os.path.isfile(root / f)]
    print(f"Top-level files: {top_files}\n")

    # -- demographics --
    demo = root / "demographics.csv"
    if demo.exists():
        with open(demo) as f:
            lines = f.readlines()
        print(f"demographics.csv: {len(lines)-1} entries")
        print(f"  Header: {lines[0].strip()}\n")

    # -- scan availability --
    avail = root / "us_scantype_availability.csv"
    if avail.exists():
        with open(avail) as f:
            lines = f.readlines()
        print(f"us_scantype_availability.csv: {len(lines)-1} volunteers")
        print(f"  Header: {lines[0].strip()}\n")

    # -- CT --
    if ct_root.exists():
        ct_dirs = sorted(os.listdir(ct_root))
        vol_ids = sorted(set(re.match(r"(URS\d+)", d).group(1)
                             for d in ct_dirs if re.match(r"URS\d+", d)))
        print(f"CT directories: {len(ct_dirs)}  ({len(vol_ids)} volunteers)")
        # show first volunteer's structure
        if ct_dirs:
            first = ct_dirs[0]
            inner = _find_inner(str(ct_root / first))
            print(f"  Example: {first}/")
            for d in sorted(os.listdir(inner)):
                n_files = len(os.listdir(os.path.join(inner, d))) \
                    if os.path.isdir(os.path.join(inner, d)) else ""
                print(f"    {d}/  ({n_files} files)" if n_files else
                      f"    {d}")
        print()

    # -- US --
    if us_root.exists():
        us_dirs = sorted(os.listdir(us_root))
        robotic = [d for d in us_dirs if re.search(r"_R\d", d)]
        handheld = [d for d in us_dirs if re.search(r"_H\d", d)]
        demo_sw  = [d for d in us_dirs if re.search(r"_D\d", d)]
        print(f"US directories: {len(us_dirs)}  "
              f"(R: {len(robotic)}, H: {len(handheld)}, D: {len(demo_sw)})")
        # show first robotic sweep's structure
        if robotic:
            first = robotic[0]
            inner = _find_inner(str(us_root / first))
            print(f"  Example: {first}/")
            for item in sorted(os.listdir(inner)):
                fp = os.path.join(inner, item)
                if os.path.isdir(fp):
                    n = len(os.listdir(fp))
                    print(f"    {item}/  ({n} files)")
                else:
                    sz = os.path.getsize(fp)
                    print(f"    {item}  ({sz} bytes)")
                    # Show first few lines of text files
                    if item.endswith((".txt", ".csv")):
                        with open(fp) as fh:
                            for i, line in enumerate(fh):
                                if i >= 3:
                                    break
                                print(f"      | {line.rstrip()}")
        print()

    # -- Validation --
    if val_root.exists():
        val_items = sorted(os.listdir(val_root))
        print(f"Validation directory: {len(val_items)} items")
        for item in val_items[:10]:
            fp = val_root / item
            if fp.is_dir():
                n = len(list(fp.rglob("*")))
                print(f"  {item}/  ({n} files)")
            else:
                print(f"  {item}  ({fp.stat().st_size} bytes)")
    print()

    # -- STL --
    stl_dirs = [d for d in (sorted(os.listdir(ct_root)) if ct_root.exists()
                            else []) if "stl" in d.lower()]
    if stl_dirs:
        first_stl = ct_root / stl_dirs[0]
        inner = _find_inner(str(first_stl))
        stl_files = [f for f in os.listdir(inner) if f.lower().endswith(".stl")]
        print(f"STL directories: {len(stl_dirs)}")
        print(f"  Example: {stl_dirs[0]}/ → {len(stl_files)} STL files")
        if stl_files:
            print(f"  Files: {stl_files[:5]}{'...' if len(stl_files)>5 else ''}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════════════════

def get_robotic_sweeps(us_root: str, vol_id: str) -> List[str]:
    """Return list of robotic sweep directory names for a volunteer."""
    sweeps: list = []
    for d in sorted(os.listdir(us_root)):
        m = re.match(rf"{vol_id}_R(\d+)", d, re.IGNORECASE)
        if m:
            sweeps.append(d)
    return sweeps


def process_sweep(
    vol_id: str,
    sweep_name: str,
    ct_vol: np.ndarray,
    bone_mask: np.ndarray,
    affine: np.ndarray,
    T_reg: np.ndarray,
    us_inner_dir: str,
    output_dir: Path,
    img_size: int,
    stride: int,
    fov_w: float,
    fov_d: float,
) -> int:
    """
    Process one robotic sweep. Returns number of saved triplets.
    """
    sweep_tag = re.sub(r"^URS\d+_", "", sweep_name)  # R1, R2, R3

    # --- Locate files inside the (possibly nested) sweep directory ---
    pose_candidates = ["RUS_pose.txt", "MUS_pose.txt", "DUS_pose.txt"]
    pose_file = None
    us_frame_dir = None
    marker_file  = None

    for name in sorted(os.listdir(us_inner_dir)):
        fp = os.path.join(us_inner_dir, name)
        low = name.lower()
        if low.endswith("_pose.txt") or low in [p.lower() for p in pose_candidates]:
            pose_file = fp
        elif low == "body_marker.csv":
            marker_file = fp
        elif os.path.isdir(fp) and low in ("rus", "mus", "dus"):
            us_frame_dir = fp

    if pose_file is None:
        log.warning("  [%s/%s] No pose file found — skipping", vol_id, sweep_name)
        return 0
    if us_frame_dir is None:
        log.warning("  [%s/%s] No US frame directory — skipping", vol_id, sweep_name)
        return 0

    # --- Parse poses ---
    poses = parse_robot_poses(pose_file)
    if not poses:
        log.warning("  [%s/%s] Empty pose file — skipping", vol_id, sweep_name)
        return 0

    # --- US frame files (sorted) ---
    frame_files = sorted([
        f for f in os.listdir(us_frame_dir)
        if f.lower().endswith((".png", ".jpg", ".bmp", ".tif", ".tiff"))
    ])
    if not frame_files:
        log.warning("  [%s/%s] No US images found — skipping", vol_id, sweep_name)
        return 0

    n_frames = min(len(poses), len(frame_files))
    if len(poses) != len(frame_files):
        log.warning("  [%s/%s] Pose count (%d) ≠ frame count (%d); using min=%d",
                    vol_id, sweep_name, len(poses), len(frame_files), n_frames)

    # --- Breathing compensation ---
    markers: list = []
    if marker_file and os.path.exists(marker_file):
        markers = parse_body_markers(marker_file)
    breath_comps = breathing_compensation(markers, n_frames)

    # --- Output directories ---
    ct_out     = output_dir / "ct"
    label_out  = output_dir / "labels"
    simus_out  = output_dir / "simus"
    ct_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    simus_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(0, n_frames, stride):
        T_robot  = poses[i]
        T_breath = breath_comps[i]

        # Probe pose in CT world coordinates
        T_ct = T_reg @ T_breath @ T_robot

        # Imaging plane vectors
        position  = T_ct[:3, 3]
        right_dir = T_ct[:3, 0]   # probe X axis → lateral
        down_dir  = T_ct[:3, 2]   # probe Z axis → into patient (depth)

        # Normalise directions
        right_dir = right_dir / np.linalg.norm(right_dir)
        down_dir  = down_dir / np.linalg.norm(down_dir)

        # --- Oblique reslice CT and bone mask ---
        ct_slice   = oblique_reslice(ct_vol,   affine, position,
                                     right_dir, down_dir,
                                     fov_w, fov_d, img_size)
        bone_slice = oblique_reslice(bone_mask, affine, position,
                                     right_dir, down_dir,
                                     fov_w, fov_d, img_size)
        bone_slice = (bone_slice > 0.5).astype(np.float32)   # re-binarise

        # --- Load US frame ---
        us_path = os.path.join(us_frame_dir, frame_files[i])
        us_img  = cv2.imread(us_path, cv2.IMREAD_GRAYSCALE)
        if us_img is None:
            log.warning("  [%s/%s] Could not read frame %s", vol_id, sweep_name,
                        frame_files[i])
            continue
        us_img = cv2.resize(us_img, (img_size, img_size),
                            interpolation=cv2.INTER_AREA)
        us_arr = us_img.astype(np.float32) * SIMUS_SCALE   # → [0, 220]

        # --- Save ---
        stem = f"{vol_id}_{sweep_tag}_{i:05d}"
        np.save(ct_out    / f"{stem}.npy", ct_slice)
        np.save(label_out / f"{stem}.npy", bone_slice)
        np.save(simus_out / f"{stem}.npy", us_arr)
        saved += 1

    return saved


def process_volunteer(
    vol_id: str,
    data_root: Path,
    output_dir: Path,
    img_size: int,
    stride: int,
    fov_w: float,
    fov_d: float,
    depth_offset: float,
) -> int:
    """Process all robotic sweeps for one volunteer. Returns sample count."""
    ct_root = data_root / "computed_tomography"
    us_root = data_root / "ultrasound"

    # ---------- CT ----------
    ct_outer = ct_root / f"{vol_id}_CT_raw"
    if not ct_outer.exists():
        log.warning("[%s] CT directory not found — skipping", vol_id)
        return 0
    ct_inner = _find_inner(str(ct_outer))
    seq_name = select_ct_sequence(ct_inner)
    dicom_dir = os.path.join(ct_inner, seq_name)
    log.info("[%s] CT sequence: %s", vol_id, seq_name)

    try:
        ct_vol, affine = load_dicom_volume(dicom_dir)
    except Exception as e:
        log.error("[%s] DICOM load failed: %s", vol_id, e)
        return 0

    # ---------- Bone mask ----------
    bone_mask = (ct_vol > BONE_HU_THRESHOLD).astype(np.float32)
    log.info("[%s] Bone voxels: %d / %d (%.1f%%)", vol_id,
             int(bone_mask.sum()), bone_mask.size,
             100.0 * bone_mask.sum() / bone_mask.size)

    # ---------- STL for registration ----------
    stl_outer = ct_root / f"{vol_id}stl"
    stl_verts = None
    if stl_outer.exists():
        stl_inner = _find_inner(str(stl_outer))
        try:
            stl_verts = load_all_stl_vertices(stl_inner)
            log.info("[%s] STL vertices: %d", vol_id, len(stl_verts))
        except Exception as e:
            log.warning("[%s] STL load failed: %s", vol_id, e)

    # ---------- Robotic sweeps ----------
    sweeps = get_robotic_sweeps(str(us_root), vol_id)
    if not sweeps:
        log.warning("[%s] No robotic sweeps found — skipping", vol_id)
        return 0

    # ---------- Collect probe positions from ALL sweeps for registration ----------
    all_probe_pos: list = []
    all_probe_dirs: list = []
    sweep_poses_map: Dict[str, List[np.ndarray]] = {}

    for sw in sweeps:
        sw_outer = us_root / sw
        sw_inner = _find_inner(str(sw_outer))
        pose_file = None
        for name in os.listdir(sw_inner):
            if name.lower().endswith("_pose.txt"):
                pose_file = os.path.join(sw_inner, name)
                break
        if pose_file is None:
            continue
        poses = parse_robot_poses(pose_file)
        sweep_poses_map[sw] = poses
        for T in poses:
            all_probe_pos.append(T[:3, 3])
            all_probe_dirs.append(T[:3, 2])   # Z-axis (into patient)

    if not all_probe_pos:
        log.warning("[%s] No valid poses across sweeps — skipping", vol_id)
        return 0

    probe_positions = np.array(all_probe_pos)
    probe_z_dirs    = np.array(all_probe_dirs)

    # ---------- Registration ----------
    if stl_verts is not None and len(stl_verts) > 50:
        # Shift probe positions inward along probe Z to approximate bone depth
        shifted_pos = probe_positions.copy()
        for j in range(len(shifted_pos)):
            z_dir = probe_z_dirs[j]
            z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-12)
            shifted_pos[j] += z_dir * depth_offset

        # Subsample STL for speed (max 10k points)
        if len(stl_verts) > 10000:
            idx = np.random.default_rng(42).choice(
                len(stl_verts), 10000, replace=False)
            stl_sub = stl_verts[idx]
        else:
            stl_sub = stl_verts

        # PCA pre-alignment + ICP
        T_init = _pca_pre_align(shifted_pos, stl_sub)
        T_reg, rmse = icp(shifted_pos, stl_sub, init=T_init, max_iter=150)
        log.info("[%s] Registration RMSE: %.2f mm", vol_id, rmse)

        if rmse > 50.0:
            log.warning("[%s] Registration RMSE > 50 mm — results may be poor",
                        vol_id)
    else:
        log.warning("[%s] No STL mesh — using identity registration "
                    "(results may be misaligned)", vol_id)
        T_reg = np.eye(4)

    # ---------- Process each sweep ----------
    total = 0
    for sw in sweeps:
        sw_outer = us_root / sw
        sw_inner = _find_inner(str(sw_outer))
        log.info("[%s] Processing sweep %s …", vol_id, sw)
        n = process_sweep(
            vol_id, sw, ct_vol, bone_mask, affine, T_reg,
            sw_inner, output_dir, img_size, stride, fov_w, fov_d,
        )
        log.info("[%s/%s] Saved %d triplets", vol_id, sw, n)
        total += n

    # ---------- Save sample diagnostic images ----------
    try:
        _save_diagnostics(vol_id, output_dir, img_size)
    except Exception:
        pass   # non-critical

    return total


def _save_diagnostics(vol_id: str, output_dir: Path, img_size: int) -> None:
    """Save a PNG grid of the first 4 triplets for visual QA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    ct_dir    = output_dir / "ct"
    label_dir = output_dir / "labels"
    simus_dir = output_dir / "simus"

    files = sorted([f for f in os.listdir(ct_dir)
                    if f.startswith(vol_id) and f.endswith(".npy")])[:4]
    if not files:
        return

    fig, axes = plt.subplots(len(files), 3, figsize=(9, 3 * len(files)))
    if len(files) == 1:
        axes = axes[np.newaxis, :]

    for i, fname in enumerate(files):
        ct  = np.load(ct_dir / fname)
        lbl = np.load(label_dir / fname)
        us  = np.load(simus_dir / fname)

        axes[i, 0].imshow(ct,  cmap="gray"); axes[i, 0].set_title("CT slice")
        axes[i, 1].imshow(lbl, cmap="gray"); axes[i, 1].set_title("Bone mask")
        axes[i, 2].imshow(us,  cmap="gray"); axes[i, 2].set_title("US frame")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(f"{vol_id} — Sample Alignment Check", fontsize=13)
    fig.tight_layout()
    fig.savefig(diag_dir / f"{vol_id}_alignment.png", dpi=120)
    plt.close(fig)
    log.info("[%s] Diagnostic image saved → %s", vol_id,
             diag_dir / f"{vol_id}_alignment.png")


# ═══════════════════════════════════════════════════════════════════════════
# Auto-split helper
# ═══════════════════════════════════════════════════════════════════════════

def auto_train_val_split(output_dir: Path,
                         val_fraction: float = 0.2) -> Dict[str, list]:
    """
    Scan the output directory and split subjects 80 / 20 by volunteer.
    Saves meta.json with split information.
    """
    ct_dir = output_dir / "ct"
    if not ct_dir.exists():
        return {"train": [], "val": []}

    # Collect all stems
    stems = sorted(set(
        f[:-4] for f in os.listdir(ct_dir) if f.endswith(".npy")
    ))

    # Extract volunteer IDs from stems like "URS01_R1_00000"
    vol_to_subjects: Dict[str, set] = {}
    for s in stems:
        m = re.match(r"(URS\d+)", s)
        if m:
            vid = m.group(1)
            # Subject = everything before the last _DIGITS  (e.g. URS01_R1)
            subj_m = re.match(r"^(.+?)_\d+$", s)
            subj = subj_m.group(1) if subj_m else s
            vol_to_subjects.setdefault(vid, set()).add(subj)

    vols = sorted(vol_to_subjects.keys())
    n_val = max(1, int(len(vols) * val_fraction))
    val_vols   = vols[-n_val:]
    train_vols = vols[:-n_val]

    train_subjects = sorted(set().union(*(vol_to_subjects[v] for v in train_vols)))
    val_subjects   = sorted(set().union(*(vol_to_subjects[v] for v in val_vols)))

    meta = {
        "n_volunteers": len(vols),
        "n_train_volunteers": len(train_vols),
        "n_val_volunteers": len(val_vols),
        "train_volunteers": train_vols,
        "val_volunteers": val_vols,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "n_total_samples": len(stems),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Auto-split: %d train / %d val volunteers  (%d / %d subjects)",
             len(train_vols), len(val_vols),
             len(train_subjects), len(val_subjects))
    log.info("Meta saved → %s", meta_path)
    return meta


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess Cavalcanti Robotic Lumbar Spine Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_root",  type=str, required=True,
                   help="Path to extracted Cavalcanti dataset root")
    p.add_argument("--output_dir", type=str, default="./data/cavalcanti_processed",
                   help="Output directory for .npy triplets")
    p.add_argument("--img_size",   type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--stride",     type=int, default=DEFAULT_STRIDE,
                   help="Frame stride — process every Nth US frame (default 5)")
    p.add_argument("--volunteers", nargs="+", default=None,
                   help="Process only these volunteer IDs (e.g. URS01 URS02)")
    p.add_argument("--fov_width",  type=float, default=US_FOV_WIDTH_MM,
                   help="US field of view width in mm")
    p.add_argument("--fov_depth",  type=float, default=US_FOV_DEPTH_MM,
                   help="US field of view depth in mm")
    p.add_argument("--depth_offset", type=float, default=DEPTH_OFFSET_MM,
                   help="Estimated skin-to-bone depth for ICP initialisation (mm)")
    p.add_argument("--discover",   action="store_true",
                   help="Print dataset structure and exit")
    args = p.parse_args()

    # ---- Discovery mode ----
    if args.discover:
        discover(args.data_root)
        return

    # ---- Full preprocessing ----
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    ct_root    = data_root / "computed_tomography"
    us_root    = data_root / "ultrasound"

    if not ct_root.exists() or not us_root.exists():
        log.error("Expected computed_tomography/ and ultrasound/ in %s", data_root)
        sys.exit(1)

    # Discover volunteer IDs that have both CT and robotic US
    ct_vols = set()
    for d in os.listdir(ct_root):
        m = re.match(r"(URS\d+)_CT_raw", d)
        if m:
            ct_vols.add(m.group(1))

    us_vols = set()
    for d in os.listdir(us_root):
        m = re.match(r"(URS\d+)_R\d", d)
        if m:
            us_vols.add(m.group(1))

    paired = sorted(ct_vols & us_vols)

    if args.volunteers:
        paired = [v for v in paired if v in args.volunteers]

    log.info("Volunteers with paired CT + robotic US: %d", len(paired))
    log.info("  IDs: %s", paired)

    # ---- Process ----
    grand_total = 0
    for vol_id in paired:
        log.info("━" * 60)
        log.info("Processing %s …", vol_id)
        try:
            n = process_volunteer(
                vol_id, data_root, output_dir,
                args.img_size, args.stride,
                args.fov_width, args.fov_depth,
                args.depth_offset,
            )
            grand_total += n
            log.info("[%s] Total saved: %d", vol_id, n)
        except Exception:
            log.error("[%s] FAILED:\n%s", vol_id, traceback.format_exc())

    log.info("━" * 60)
    log.info("DONE — %d total triplets saved to %s", grand_total, output_dir)

    # ---- Auto-split ----
    if grand_total > 0:
        auto_train_val_split(output_dir)


if __name__ == "__main__":
    main()
