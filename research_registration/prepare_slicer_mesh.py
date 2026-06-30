"""
prepare_slicer_mesh.py — Prepare a 3D Slicer exported mesh for PyBullet.
========================================================================
Loads a Slicer-exported OBJ mesh (where vertices are in mm in RAS/LPS physical space),
converts physical coordinates to meters, centers the mesh horizontally and places
its bottom at z=0, and exports "patient_skin.obj" and "registration_meta.json"
ready for the registration pipeline and live_registered_demo.py.

Usage:
    python prepare_slicer_mesh.py --subject ../TCGA-QQ-A8VG --slicer-mesh Segmentation.obj
"""

import argparse
import json
import sys
from pathlib import Path
import nibabel as nib
import numpy as np


def parse_obj(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse vertices and faces from an OBJ file."""
    vertices = []
    faces = []
    
    print(f"Reading OBJ: {obj_path}")
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                # Face lines can contain texture/normal indices (e.g., f v1/vt1/vn1)
                # We only need the vertex index (first part before /)
                parts = line.split()[1:4]
                face_idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(face_idx)
                
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a mesh to Wavefront OBJ format."""
    print(f"Writing OBJ: {path}")
    with open(path, "w") as f:
        f.write(f"# Processed Slicer mesh — {vertices.shape[0]} vertices, "
                f"{faces.shape[0]} faces\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a Slicer-exported OBJ mesh for PyBullet."
    )
    parser.add_argument(
        "--subject", type=Path, required=True,
        help="Path to subject directory containing CT.nii",
    )
    parser.add_argument(
        "--slicer-mesh", type=Path, required=True,
        help="Path to the Segmentation.obj file exported from 3D Slicer",
    )
    args = parser.parse_args()

    subject_dir = args.subject.resolve()
    ct_path = subject_dir / "CT.nii"
    slicer_mesh_path = args.slicer_mesh.resolve()

    if not ct_path.exists():
        print(f"ERROR: CT.nii not found in {subject_dir}")
        sys.exit(1)
        
    if not slicer_mesh_path.exists():
        print(f"ERROR: Slicer mesh not found at {slicer_mesh_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Preparing Slicer Mesh for Subject: {subject_dir.name}")
    print(f"{'='*60}")

    # ── 1. Load CT image to get NIfTI headers ──────────────────────────────
    print(f"Loading CT to read geometry: {ct_path}")
    ct_img = nib.load(str(ct_path))
    affine = ct_img.affine.astype(np.float64)
    inv_affine = np.linalg.inv(affine)
    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)
    ct_shape = ct_img.shape

    print(f"  CT Shape         : {ct_shape}")
    print(f"  Voxel Spacing    : {spacing} mm")
    print(f"  NIfTI Affine     :\n{affine}")

    # ── 2. Parse the Slicer OBJ ───────────────────────────────────────────
    verts_mm, faces = parse_obj(slicer_mesh_path)
    print(f"  Parsed {verts_mm.shape[0]} vertices and {faces.shape[0]} faces.")

    # Convert LPS to RAS (3D Slicer exports OBJ in LPS physical coordinates,
    # but NIfTI affine uses RAS physical coordinates)
    print("  Converting Slicer LPS coordinates to NIfTI RAS space (negating X and Y)...")
    verts_mm[:, 0] = -verts_mm[:, 0]
    verts_mm[:, 1] = -verts_mm[:, 1]

    # ── 3. Convert mm → meters for PyBullet ────────────────────────────────
    verts_m = verts_mm / 1000.0

    # ── 4. Center the mesh horizontally and place bottom at z=0 ───────────
    bbox_min = verts_m.min(axis=0)
    bbox_max = verts_m.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Subtract this offset to center x, y, and align z-min to 0
    centering_offset = np.array([
        bbox_center[0],
        bbox_center[1],
        bbox_min[2]
    ], dtype=np.float64)

    verts_centered = verts_m - centering_offset
    
    mesh_extent = bbox_max - bbox_min
    print(f"  Mesh Extent (m)  : x={mesh_extent[0]:.3f}, y={mesh_extent[1]:.3f}, z={mesh_extent[2]:.3f}")
    print(f"  Centering Offset : {centering_offset}")

    # ── 5. Save processed mesh and metadata in subject directory ──────────
    obj_out = subject_dir / "patient_skin.obj"
    write_obj(obj_out, verts_centered, faces)

    meta = {
        "affine": affine.tolist(),
        "inv_affine": inv_affine.tolist(),
        "ct_shape": list(ct_shape),
        "voxel_spacing": spacing.tolist(),
        "mesh_centering_offset": centering_offset.tolist(),
        "mesh_threshold_hu": "Slicer-segmentation",
        "mesh_vertex_count": int(verts_mm.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
    }

    meta_out = subject_dir / "registration_meta.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved registration metadata: {meta_out}")

    # ── 6. Simple validation check ────────────────────────────────────────
    print(f"\nChecking affine round-trip error ...")
    vol_center_vox = np.array(ct_shape, dtype=np.float64) / 2.0
    vox_hom = np.array([*vol_center_vox, 1.0])
    center_mm = (affine @ vox_hom)[:3]
    back_vox = (inv_affine @ np.array([*center_mm, 1.0]))[:3]
    roundtrip_err = np.linalg.norm(back_vox - vol_center_vox)
    print(f"  Voxel round-trip error: {roundtrip_err:.2e} voxels")
    print(f"{'='*60}")
    print(f"  SUCCESS: Mesh prepared and saved in {subject_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
