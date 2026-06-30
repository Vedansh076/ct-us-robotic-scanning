"""
extract_patient_mesh.py — Generate a patient-specific body surface mesh from CT.nii.
=====================================================================================
Reads a NIfTI CT volume, segments the outer body boundary via HU thresholding,
runs Marching Cubes to extract the isosurface, transforms the mesh vertices from
voxel indices to physical scanner coordinates (meters) via the NIfTI affine,
and exports the result as an OBJ file ready for PyBullet.

Outputs (saved alongside CT.nii in the subject directory):
    patient_skin.obj        — triangulated surface mesh, vertices in meters
    registration_meta.json  — affine, inverse affine, centering offset, CT shape

Usage:
    python extract_patient_mesh.py --subject ../TCGA-QQ-A8VG
    python extract_patient_mesh.py --subject ../TCGA-QQ-A8VG --threshold -300 --step-size 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure


# ═══════════════════════════════════════════════════════════════════════════════
# OBJ WRITER (no trimesh dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to Wavefront OBJ format.

    Parameters
    ----------
    path : Path
        Output file path.
    vertices : (N, 3) float
        Vertex positions.
    faces : (M, 3) int
        Triangle indices (0-based; converted to 1-based for OBJ).
    """
    with open(path, "w") as f:
        f.write(f"# Patient skin mesh — {vertices.shape[0]} vertices, "
                f"{faces.shape[0]} faces\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            # OBJ uses 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MESH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mesh(
    ct_path: Path,
    threshold_hu: float = -200.0,
    step_size: int = 2,
    smooth_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Extract a skin surface mesh from a CT NIfTI volume.

    Returns
    -------
    vertices_m : (N, 3) — mesh vertices in physical meters (centered)
    faces : (M, 3) — triangle indices
    affine : (4, 4) — NIfTI affine (voxel → physical mm)
    centering_offset : (3,) — offset subtracted from physical-meter coords
    ct_shape : tuple of 3 ints — volume dimensions in voxels
    """
    print(f"  Loading CT: {ct_path}")
    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine.astype(np.float64)
    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)
    ct_shape = ct_data.shape

    print(f"  CT shape: {ct_shape}")
    print(f"  Voxel spacing (mm): {spacing}")
    print(f"  Affine matrix:\n{affine}")
    print(f"  HU range in volume: [{ct_data.min():.1f}, {ct_data.max():.1f}]")

    # ── 1. Segment body from air ──────────────────────────────────────────
    print(f"  Segmenting body at threshold {threshold_hu} HU ...")
    binary = ct_data > threshold_hu

    # Morphological cleanup: fill internal cavities, smooth boundary
    binary = ndimage.binary_fill_holes(binary)
    if smooth_iterations > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        binary = ndimage.binary_closing(binary, structure=struct,
                                        iterations=smooth_iterations)
        binary = ndimage.binary_opening(binary, structure=struct,
                                        iterations=1)

    # Keep only the largest connected component (the body)
    labelled, n_components = ndimage.label(binary)
    if n_components > 1:
        component_sizes = ndimage.sum(binary, labelled,
                                      range(1, n_components + 1))
        largest = np.argmax(component_sizes) + 1
        binary = labelled == largest
        print(f"  Kept largest component ({int(component_sizes[largest-1])} "
              f"voxels) out of {n_components} components")

    body_fraction = binary.sum() / binary.size
    print(f"  Body voxels: {binary.sum()} ({body_fraction*100:.1f}% of volume)")

    # ── 2. Marching Cubes in voxel coordinates ────────────────────────────
    print(f"  Running Marching Cubes (step_size={step_size}) ...")
    t0 = time.time()
    # spacing=(1,1,1) so vertices are in voxel-index units
    vertices_voxel, faces, normals, values = measure.marching_cubes(
        binary.astype(np.float32),
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )
    elapsed = time.time() - t0
    print(f"  Marching Cubes: {vertices_voxel.shape[0]} vertices, "
          f"{faces.shape[0]} faces ({elapsed:.2f}s)")

    # ── 3. Transform voxel → physical mm via NIfTI affine ─────────────────
    # Homogeneous coordinates: [i, j, k, 1]
    ones = np.ones((vertices_voxel.shape[0], 1), dtype=np.float64)
    verts_hom = np.hstack([vertices_voxel.astype(np.float64), ones])
    verts_mm = (affine @ verts_hom.T).T[:, :3]

    # ── 4. Convert mm → meters ────────────────────────────────────────────
    verts_m = verts_mm / 1000.0

    # ── 5. Center the mesh for PyBullet placement ─────────────────────────
    # Use bounding-box center for reproducible centering
    bbox_min = verts_m.min(axis=0)
    bbox_max = verts_m.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Centering offset: we subtract this from physical meters
    # to center the mesh horizontally and place bottom at z=0
    centering_offset = np.array([
        bbox_center[0],
        bbox_center[1],
        bbox_min[2],     # bottom of mesh goes to z=0
    ], dtype=np.float64)

    verts_centered = verts_m - centering_offset

    mesh_extent = bbox_max - bbox_min
    print(f"  Mesh extent (m): x={mesh_extent[0]:.3f}, "
          f"y={mesh_extent[1]:.3f}, z={mesh_extent[2]:.3f}")
    print(f"  Centering offset (m): {centering_offset}")

    return verts_centered, faces, affine, centering_offset, ct_shape


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract patient-specific body surface mesh from CT.nii"
    )
    parser.add_argument(
        "--subject", type=Path, required=True,
        help="Path to subject directory containing CT.nii",
    )
    parser.add_argument(
        "--threshold", type=float, default=-200.0,
        help="HU threshold for skin segmentation (default: -200)",
    )
    parser.add_argument(
        "--step-size", type=int, default=2,
        help="Marching Cubes step size (higher = fewer triangles, default: 2)",
    )
    parser.add_argument(
        "--smooth", type=int, default=3,
        help="Morphological closing iterations (default: 3)",
    )
    args = parser.parse_args()

    subject_dir = args.subject.resolve()
    ct_path = subject_dir / "CT.nii"
    if not ct_path.exists():
        print(f"ERROR: CT.nii not found in {subject_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Patient Mesh Extraction: {subject_dir.name}")
    print(f"{'='*60}")

    vertices, faces, affine, centering_offset, ct_shape = extract_mesh(
        ct_path,
        threshold_hu=args.threshold,
        step_size=args.step_size,
        smooth_iterations=args.smooth,
    )

    # ── Save mesh ─────────────────────────────────────────────────────────
    obj_path = subject_dir / "patient_skin.obj"
    write_obj(obj_path, vertices, faces)
    print(f"\n  Saved mesh: {obj_path}")

    # ── Save registration metadata ────────────────────────────────────────
    inv_affine = np.linalg.inv(affine)
    spacing = np.array(nib.load(str(ct_path)).header.get_zooms()[:3],
                       dtype=np.float64)

    meta = {
        "affine": affine.tolist(),
        "inv_affine": inv_affine.tolist(),
        "ct_shape": list(ct_shape),
        "voxel_spacing": spacing.tolist(),
        "mesh_centering_offset": centering_offset.tolist(),
        "mesh_threshold_hu": args.threshold,
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
    }

    meta_path = subject_dir / "registration_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved registration metadata: {meta_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Subject          : {subject_dir.name}")
    print(f"  CT shape         : {ct_shape}")
    print(f"  Voxel spacing    : {spacing} mm")
    print(f"  Threshold        : {args.threshold} HU")
    print(f"  Vertices         : {vertices.shape[0]}")
    print(f"  Faces            : {faces.shape[0]}")
    print(f"  Centering offset : {centering_offset}")
    print(f"  Output mesh      : {obj_path}")
    print(f"  Output meta      : {meta_path}")

    # Quick validation: check that a voxel round-trip works
    vol_center_vox = np.array(ct_shape, dtype=np.float64) / 2.0
    vox_hom = np.array([*vol_center_vox, 1.0])
    center_mm = (affine @ vox_hom)[:3]
    center_m = center_mm / 1000.0
    # Inverse
    back_hom = np.array([*center_mm, 1.0])
    back_vox = (inv_affine @ back_hom)[:3]
    roundtrip_err = np.linalg.norm(back_vox - vol_center_vox)
    print(f"\n  Affine round-trip error: {roundtrip_err:.2e} voxels")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
