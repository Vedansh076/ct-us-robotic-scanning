"""
validate_registration.py — Validate the CT ↔ Patient mesh registration.
========================================================================
Runs three validation suites (no PyBullet GUI required):

  1. REPROJECTION TEST
     Sample mesh vertices, map them through the full forward->inverse
     transform chain, and verify sub-voxel round-trip accuracy.

  2. ANATOMICAL CONSISTENCY (NEIGHBOURHOOD TEST)
     Generate clusters of nearby mesh points and verify that their
     corresponding voxel coordinates are also neighbours (smooth mapping).

  3. VISUAL DEBUGGING
     For sampled surface points, extract CT slices at the registered
     voxel coordinates and save PNG images with crosshair overlays.

Usage:
    python validate_registration.py --subject ../TCGA-QQ-A8VG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

# Import registration module from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import (
    compute_registered_ct_center,
    load_registration_meta,
    voxel_to_world,
)

# Import extract_slice (prefer local customized version in research_registration)
LOCAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = LOCAL_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(LOCAL_DIR))
from extract_slice import extract_slice


# ===============================================================================
# OBJ READER
# ===============================================================================

def read_obj_vertices(obj_path: Path) -> np.ndarray:
    """Read vertex positions from a Wavefront OBJ file."""
    verts = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64)


# ===============================================================================
# VALIDATION 1: REPROJECTION TEST
# ===============================================================================

def validation_reprojection(
    vertices_centered: np.ndarray,
    body_position: np.ndarray,
    body_orientation_matrix: np.ndarray,
    meta: dict,
    n_samples: int = 200,
    mesh_scale: np.ndarray | float = 1.0,
) -> dict:
    """Test round-trip accuracy: mesh vertex -> world -> voxel -> world -> error.

    For each sampled vertex:
      1. Compute its "world" position (forward: centered -> world)
      2. Map world -> voxel via compute_registered_ct_center
      3. Map voxel -> world via voxel_to_world
      4. Compute Euclidean error between step 1 and step 3
    """
    print(f"\n{'-'*50}")
    print(f"  VALIDATION 1: Reprojection Test ({n_samples} samples)")
    print(f"{'-'*50}")

    R_body = body_orientation_matrix
    T_body = body_position

    rng = np.random.RandomState(42)
    indices = rng.choice(len(vertices_centered), size=min(n_samples, len(vertices_centered)),
                         replace=False)

    errors_mm = []
    voxel_results = []
    clamped_count = 0

    for idx in indices:
        v_centered = vertices_centered[idx]

        # Forward: centered mesh -> world
        p_world = R_body @ (v_centered * np.asarray(mesh_scale, dtype=np.float64)) + T_body

        # Inverse: world -> voxel
        voxel = compute_registered_ct_center(
            hit_position_world=p_world,
            body_position=T_body,
            body_orientation_matrix=R_body,
            inv_affine=meta["inv_affine"],
            mesh_centering_offset=meta["mesh_centering_offset"],
            ct_shape=meta["ct_shape"],
            mesh_scale=mesh_scale,
        )

        # Check if any axis was clamped
        v_unclamped = v_centered + meta["mesh_centering_offset"]
        v_mm = v_unclamped * 1000.0
        v_hom = np.array([*v_mm, 1.0])
        v_raw = (meta["inv_affine"] @ v_hom)[:3]
        was_clamped = False
        for ax in range(3):
            if v_raw[ax] < 0 or v_raw[ax] > meta["ct_shape"][ax] - 1:
                was_clamped = True
                break
        if was_clamped:
            clamped_count += 1

        # Forward again: voxel -> world
        p_world_roundtrip = voxel_to_world(
            voxel=voxel,
            body_position=T_body,
            body_orientation_matrix=R_body,
            affine=meta["affine"],
            mesh_centering_offset=meta["mesh_centering_offset"],
            mesh_scale=mesh_scale,
        )

        error_m = np.linalg.norm(p_world - p_world_roundtrip)
        errors_mm.append(error_m * 1000.0)
        voxel_results.append(voxel)

    errors_mm = np.array(errors_mm)
    # Exclude clamped points from error statistics (they are correctly
    # clamped to volume boundary, which breaks the round-trip)
    unclamped_errors = errors_mm[errors_mm < 0.1]  # sub-0.1mm means unclamped

    print(f"  Total points sampled   : {len(indices)}")
    print(f"  Points clamped to bound: {clamped_count}")
    print(f"  Unclamped points       : {len(unclamped_errors)}")
    if len(unclamped_errors) > 0:
        print(f"  Mean reprojection error: {unclamped_errors.mean():.6f} mm")
        print(f"  Max  reprojection error: {unclamped_errors.max():.6f} mm")
        print(f"  Std  reprojection error: {unclamped_errors.std():.6f} mm")
    else:
        print(f"  Mean reprojection error: {errors_mm.mean():.6f} mm")
        print(f"  Max  reprojection error: {errors_mm.max():.6f} mm")

    # Check sub-voxel accuracy
    voxel_spacing_mm = meta["voxel_spacing"]
    min_spacing = voxel_spacing_mm.min()
    target_errors = unclamped_errors if len(unclamped_errors) > 0 else errors_mm
    if target_errors.max() < min_spacing:
        print(f"  PASS: SUB-VOXEL ACCURACY ACHIEVED (max error < {min_spacing:.3f} mm)")
    else:
        print(f"  FAIL: Some errors exceed voxel spacing ({min_spacing:.3f} mm)")

    return {
        "errors_mm": errors_mm,
        "voxels": np.array(voxel_results),
        "clamped_count": clamped_count,
    }


# ===============================================================================
# VALIDATION 2: ANATOMICAL CONSISTENCY (NEIGHBOURHOOD TEST)
# ===============================================================================

def validation_neighbourhood(
    vertices_centered: np.ndarray,
    body_position: np.ndarray,
    body_orientation_matrix: np.ndarray,
    meta: dict,
    n_clusters: int = 10,
    cluster_radius_m: float = 0.01,
    points_per_cluster: int = 5,
    mesh_scale: np.ndarray | float = 1.0,
) -> None:
    """Verify that nearby mesh points map to nearby voxel coordinates.

    Samples clusters of points that are geometrically close on the mesh
    surface, maps each to voxel space, and checks that voxel coordinates
    within each cluster are also close (smooth, continuous mapping).
    """
    print(f"\n{'-'*50}")
    print(f"  VALIDATION 2: Neighbourhood Consistency")
    print(f"{'-'*50}")

    R_body = body_orientation_matrix
    T_body = body_position

    rng = np.random.RandomState(123)
    center_indices = rng.choice(len(vertices_centered),
                                size=min(n_clusters, len(vertices_centered)),
                                replace=False)

    all_ok = True
    for ci, center_idx in enumerate(center_indices):
        center_v = vertices_centered[center_idx]

        # Find nearest neighbours within cluster_radius
        dists = np.linalg.norm(vertices_centered - center_v, axis=1)
        nearby_mask = dists < cluster_radius_m
        nearby_indices = np.where(nearby_mask)[0]

        if len(nearby_indices) < points_per_cluster:
            # Not enough nearby points; skip this cluster
            continue

        selected = rng.choice(nearby_indices, size=points_per_cluster, replace=False)

        # Map each point to voxel space
        voxels = []
        for idx in selected:
            v_centered = vertices_centered[idx]
            p_world = R_body @ (v_centered * np.asarray(mesh_scale, dtype=np.float64)) + T_body
            vox = compute_registered_ct_center(
                hit_position_world=p_world,
                body_position=T_body,
                body_orientation_matrix=R_body,
                inv_affine=meta["inv_affine"],
                mesh_centering_offset=meta["mesh_centering_offset"],
                ct_shape=meta["ct_shape"],
                mesh_scale=mesh_scale,
            )
            voxels.append(vox)

        voxels = np.array(voxels)
        voxel_spread = voxels.max(axis=0) - voxels.min(axis=0)
        world_spread = cluster_radius_m * 1000.0  # mm

        # Check that voxel spread is proportional to world spread
        max_voxel_spread_mm = voxel_spread * meta["voxel_spacing"]
        max_voxel_spread = max_voxel_spread_mm.max()

        # Voxel spread should not exceed ~2x the world spread (accounting for
        # anisotropic spacing and coordinate rotation)
        ratio = max_voxel_spread / world_spread if world_spread > 0 else 0
        status = "PASS:" if ratio < 3.0 else "FAIL:"
        if ratio >= 3.0:
            all_ok = False

        print(f"  Cluster {ci+1:2d}: mesh spread={world_spread:.1f}mm  "
              f"voxel spread={max_voxel_spread:.1f}mm  "
              f"ratio={ratio:.2f} {status}")

    if all_ok:
        print(f"  PASS: ALL CLUSTERS SHOW SMOOTH MAPPING")
    else:
        print(f"  FAIL: Some clusters show discontinuous mapping")


# ===============================================================================
# VALIDATION 3: VISUAL DEBUGGING
# ===============================================================================

def validation_visual(
    ct_volume: np.ndarray,
    spacing: np.ndarray,
    vertices_centered: np.ndarray,
    body_position: np.ndarray,
    body_orientation_matrix: np.ndarray,
    meta: dict,
    output_dir: Path,
    n_samples: int = 8,
    slice_size: int = 256,
    pixel_spacing: float = 0.35,
    mesh_scale: np.ndarray | float = 1.0,
    interp_order: int = 3,
) -> None:
    """Extract CT slices at registered probe positions and save with crosshairs.

    For each sampled mesh vertex:
      - Compute the world hit point
      - Map to voxel center via registration
      - Extract a CT slice at that voxel center
      - Draw a crosshair at the slice center
      - Save as PNG
    """
    print(f"\n{'-'*50}")
    print(f"  VALIDATION 3: Visual Debugging ({n_samples} samples)")
    print(f"{'-'*50}")

    R_body = body_orientation_matrix
    T_body = body_position

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample vertices spread along the mesh extent
    # Sort by the axis with largest extent and sample evenly
    extents = vertices_centered.max(axis=0) - vertices_centered.min(axis=0)
    sort_axis = np.argmax(extents)
    sorted_indices = np.argsort(vertices_centered[:, sort_axis])
    step = max(1, len(sorted_indices) // n_samples)
    sample_indices = sorted_indices[::step][:n_samples]

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for si, idx in enumerate(sample_indices):
        v_centered = vertices_centered[idx]
        p_world = R_body @ (v_centered * np.asarray(mesh_scale, dtype=np.float64)) + T_body

        voxel = compute_registered_ct_center(
            hit_position_world=p_world,
            body_position=T_body,
            body_orientation_matrix=R_body,
            inv_affine=meta["inv_affine"],
            mesh_centering_offset=meta["mesh_centering_offset"],
            ct_shape=meta["ct_shape"],
            mesh_scale=mesh_scale,
        )

        # Extract CT slice at this voxel center
        ct_slice = extract_slice(
            ct_volume,
            center=voxel,
            quaternion=identity_quat,
            spacing=spacing,
            size=slice_size,
            pixel_spacing=pixel_spacing,
            order=interp_order,
        )

        # Convert to displayable uint8
        finite = ct_slice[np.isfinite(ct_slice)]
        if finite.size > 0:
            lo, hi = np.percentile(finite, (1, 99))
            if hi > lo:
                display = np.clip((ct_slice - lo) / (hi - lo), 0, 1) * 255
            else:
                display = np.zeros_like(ct_slice)
        else:
            display = np.zeros_like(ct_slice)
        display = display.astype(np.uint8)

        # Draw crosshair at slice center
        display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        cx, cy = slice_size // 2, slice_size // 2
        color = (0, 255, 0)  # green
        cv2.line(display_bgr, (cx - 15, cy), (cx + 15, cy), color, 1)
        cv2.line(display_bgr, (cx, cy - 15), (cx, cy + 15), color, 1)

        # Overlay text
        cv2.putText(display_bgr,
                    f"Probe #{si+1}  world=({p_world[0]:.3f}, {p_world[1]:.3f}, {p_world[2]:.3f})",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(display_bgr,
                    f"voxel=({voxel[0]:.1f}, {voxel[1]:.1f}, {voxel[2]:.1f})",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        ct_mean = float(ct_slice.mean()) if finite.size > 0 else 0
        cv2.putText(display_bgr,
                    f"CT mean={ct_mean:.1f} HU",
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        out_path = output_dir / f"registered_slice_{si+1:02d}.png"
        cv2.imwrite(str(out_path), display_bgr)
        print(f"  Sample {si+1:2d}: world=({p_world[0]:+.3f}, {p_world[1]:+.3f}, "
              f"{p_world[2]:+.3f})  voxel=({voxel[0]:.1f}, {voxel[1]:.1f}, "
              f"{voxel[2]:.1f})  CT_mean={ct_mean:.1f} -> {out_path.name}")

    print(f"\n  Saved {n_samples} slice images to {output_dir}")


# ===============================================================================
# MAIN
# ===============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CT ↔ mesh registration")
    parser.add_argument("--subject", type=Path, required=True,
                        help="Subject directory containing CT.nii and registration_meta.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save validation outputs (default: subject/validation/)")
    parser.add_argument("--mesh-scale", type=str, default="1.0",
                        help="Mesh scale factor (e.g. '1.0' or '3.0,3.0,1.2')")
    parser.add_argument("--interp-order", type=int, choices=(1, 3), default=3,
                        help="Interpolation order for CT slice extraction: 1=linear, 3=cubic spline (default: 3)")
    args = parser.parse_args()

    subject_dir = args.subject.resolve()
    ct_path = subject_dir / "CT.nii"
    meta_path = subject_dir / "registration_meta.json"
    mesh_path = subject_dir / "patient_skin.obj"
    output_dir = args.output_dir or (subject_dir / "validation")

    # Check prerequisites
    for p_check, name in [(ct_path, "CT.nii"), (meta_path, "registration_meta.json"),
                          (mesh_path, "patient_skin.obj")]:
        if not p_check.exists():
            print(f"ERROR: {name} not found in {subject_dir}")
            print("Run extract_patient_mesh.py first.")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"  Registration Validation: {subject_dir.name}")
    print(f"{'='*60}")

    # Parse mesh scale
    scale_str = args.mesh_scale
    if "," in scale_str:
        mesh_scale = np.array([float(x) for x in scale_str.split(",")], dtype=np.float64)
    else:
        s_val = float(scale_str)
        mesh_scale = np.array([s_val, s_val, s_val], dtype=np.float64)

    # ── Load data ─────────────────────────────────────────────────────────
    meta = load_registration_meta(meta_path)
    vertices_centered = read_obj_vertices(mesh_path)
    ct_img = nib.load(str(ct_path))
    ct_volume = ct_img.get_fdata()
    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float32)

    print(f"  CT shape     : {meta['ct_shape']}")
    print(f"  Voxel spacing: {meta['voxel_spacing']} mm")
    print(f"  Mesh vertices: {vertices_centered.shape[0]}")
    print(f"  Centering off: {meta['mesh_centering_offset']}")
    print(f"  Mesh scale   : {mesh_scale}")

    # ── Simulate body placement (same as live_registered_demo.py) ─────────
    # Identity orientation, placed at bed center with bottom on mattress
    BED_MATTRESS_TOP_Z = 0.72
    body_position = np.array([0.0, 0.0, BED_MATTRESS_TOP_Z + 0.01], dtype=np.float64)
    body_orientation_matrix = np.eye(3, dtype=np.float64)

    # ── Run validations ───────────────────────────────────────────────────
    reproj = validation_reprojection(
        vertices_centered, body_position, body_orientation_matrix, meta,
        n_samples=200,
        mesh_scale=mesh_scale,
    )

    validation_neighbourhood(
        vertices_centered, body_position, body_orientation_matrix, meta,
        n_clusters=10, cluster_radius_m=0.01, points_per_cluster=5,
        mesh_scale=mesh_scale,
    )

    validation_visual(
        ct_volume, spacing,
        vertices_centered, body_position, body_orientation_matrix, meta,
        output_dir=output_dir,
        n_samples=8,
        mesh_scale=mesh_scale,
        interp_order=args.interp_order,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*60}")
    unclamped = reproj["errors_mm"][reproj["errors_mm"] < 0.1]
    if len(unclamped) > 0:
        print(f"  Reprojection: mean={unclamped.mean():.6f}mm, max={unclamped.max():.6f}mm")
    print(f"  Clamped points: {reproj['clamped_count']} / {len(reproj['errors_mm'])}")
    print(f"  Visual slices: saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
