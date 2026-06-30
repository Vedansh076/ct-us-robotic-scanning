"""
registration.py — Rigorous affine-based world→voxel coordinate registration.
=============================================================================
This module provides the mathematically exact mapping between PyBullet world
coordinates and CT voxel indices.

The transform chain is:

    FORWARD (voxel → world):
        voxel (i,j,k)
        → NIfTI affine → physical scanner mm (x,y,z)
        → /1000 → physical meters
        → −centering_offset → centered mesh coords
        → R_body @ (·) + T_body → PyBullet world

    INVERSE (world → voxel):
        PyBullet world (x,y,z)
        → R_body^T @ (· − T_body) → centered mesh coords
        → +centering_offset → physical meters
        → ×1000 → physical scanner mm
        → inv(affine) → voxel (i,j,k)

No bounding-box normalization, no heuristic scaling, no hit_scale parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_registration_meta(json_path: str | Path) -> dict:
    """Load registration metadata saved by extract_patient_mesh.py.

    Returns a dict with numpy arrays for 'affine', 'inv_affine',
    'mesh_centering_offset', 'ct_shape', 'voxel_spacing', etc.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    meta = {
        "affine": np.array(raw["affine"], dtype=np.float64),
        "inv_affine": np.array(raw["inv_affine"], dtype=np.float64),
        "ct_shape": tuple(raw["ct_shape"]),
        "voxel_spacing": np.array(raw["voxel_spacing"], dtype=np.float64),
        "mesh_centering_offset": np.array(
            raw["mesh_centering_offset"], dtype=np.float64
        ),
        "mesh_threshold_hu": raw["mesh_threshold_hu"],
        "mesh_vertex_count": raw.get("mesh_vertex_count", 0),
        "mesh_face_count": raw.get("mesh_face_count", 0),
    }
    return meta


def compute_registered_ct_center(
    hit_position_world: np.ndarray | None,
    body_position: np.ndarray,
    body_orientation_matrix: np.ndarray,
    inv_affine: np.ndarray,
    mesh_centering_offset: np.ndarray,
    ct_shape: tuple[int, ...],
    mesh_scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Map a PyBullet world hit point to the exact CT voxel coordinate.

    Parameters
    ----------
    hit_position_world : (3,) or None
        Raycast hit point in PyBullet world coordinates (meters).
        If None, returns the volume center as fallback.
    body_position : (3,)
        PyBullet base position of the patient mesh body.
    body_orientation_matrix : (3, 3)
        Rotation matrix of the patient mesh body in PyBullet.
    inv_affine : (4, 4)
        Inverse of the NIfTI affine matrix (physical mm → voxel).
    mesh_centering_offset : (3,)
        The offset that was subtracted from physical-meter vertices
        during mesh export to center them for PyBullet placement.
    ct_shape : tuple of 3 ints
        Shape of the CT volume, used for clamping.
    mesh_scale : (3,) or float, default 1.0
        Scaling factor applied to the mesh in PyBullet.

    Returns
    -------
    voxel_center : (3,) float32
        The voxel coordinate in the CT volume.
    """
    if hit_position_world is None:
        # Fallback: return volume center
        return (np.array(ct_shape, dtype=np.float32) - 1.0) / 2.0

    hit = np.asarray(hit_position_world, dtype=np.float64)
    body_pos = np.asarray(body_position, dtype=np.float64)
    R_body = np.asarray(body_orientation_matrix, dtype=np.float64)

    # ── Step 1: World → centered mesh coordinates ─────────────────────
    # Undo the PyBullet placement: P_centered = R^T @ (P_world − T)
    p_centered = R_body.T @ (hit - body_pos)

    # Undo the PyBullet mesh scale (element-wise division for non-isotropic scaling)
    p_centered_unscaled = p_centered / np.asarray(mesh_scale, dtype=np.float64)

    # ── Step 2: Undo centering → physical meters ─────────────────────
    p_meters = p_centered_unscaled + mesh_centering_offset

    # ── Step 3: Meters → millimeters (scanner physical coords) ────────
    p_mm = p_meters * 1000.0

    # ── Step 4: Physical mm → voxel via inverse affine ────────────────
    p_mm_hom = np.array([p_mm[0], p_mm[1], p_mm[2], 1.0], dtype=np.float64)
    voxel_hom = inv_affine @ p_mm_hom
    voxel = voxel_hom[:3]

    # ── Step 5: Clamp to valid CT volume bounds ──────────────────────
    for axis in range(3):
        voxel[axis] = max(0.0, min(voxel[axis], ct_shape[axis] - 1.0))

    return voxel.astype(np.float32)


def voxel_to_world(
    voxel: np.ndarray,
    body_position: np.ndarray,
    body_orientation_matrix: np.ndarray,
    affine: np.ndarray,
    mesh_centering_offset: np.ndarray,
    mesh_scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Forward transform: voxel → PyBullet world (for validation).

    This is the exact mathematical inverse of compute_registered_ct_center.
    """
    voxel = np.asarray(voxel, dtype=np.float64)
    body_pos = np.asarray(body_position, dtype=np.float64)
    R_body = np.asarray(body_orientation_matrix, dtype=np.float64)

    # Voxel → physical mm via affine
    vox_hom = np.array([voxel[0], voxel[1], voxel[2], 1.0], dtype=np.float64)
    p_mm = (affine @ vox_hom)[:3]

    # mm → meters
    p_meters = p_mm / 1000.0

    # Apply centering
    p_centered = p_meters - mesh_centering_offset

    # Apply PyBullet mesh scale
    p_centered_scaled = p_centered * np.asarray(mesh_scale, dtype=np.float64)

    # Apply body placement
    p_world = R_body @ p_centered_scaled + body_pos

    return p_world.astype(np.float64)
