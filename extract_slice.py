import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def extract_slice(volume,
                  center,
                  quaternion,
                  spacing,
                  size=256,
                  pixel_spacing=1.0,
                  order=1,
                  cval=None,
                  inv_affine=None,
                  mesh_scale=None,
                  body_orientation_matrix=None):
    """Extract a 2D slice from a 3D volume at a given center and orientation.

    Parameters
    ----------
    volume : 3D ndarray
        The voxel volume.
    center : (3,)
        The center coordinates of the slice in voxel index space.
    quaternion : (4,)
        The rotation quaternion [x, y, z, w] representing the orientation.
    spacing : (3,)
        Voxel spacing of the volume in mm.
    size : int, default 256
        The size of the output slice (size x size pixels).
    pixel_spacing : float, default 1.0
        The physical resolution of the extracted slice pixels in mm.
    order : int, default 1
        Spline interpolation order. 1 is bilinear, 3 is cubic spline.
    cval : float, optional
        Fill value for points outside the volume. Defaults to volume.min().
    inv_affine : (4,4) ndarray, optional
        Inverse NIfTI affine matrix (physical mm → voxel). When provided
        together with mesh_scale and body_orientation_matrix, the probe's
        direction vectors are transformed through the full registration chain:
            PyBullet world → body-centered → unscale → mm → voxel
        This ensures the slice plane aligns perfectly with the CT anatomy
        regardless of mesh scale or coordinate space differences.
    mesh_scale : (3,) or float, optional
        Non-isotropic mesh scale applied in PyBullet (e.g. [2, 4, 1]).
        Required when inv_affine is provided.
    body_orientation_matrix : (3,3) ndarray, optional
        Rotation matrix of the patient mesh body in PyBullet.
        Required when inv_affine is provided.

    Notes
    -----
    WITHOUT registration context (inv_affine=None):
        The classic extraction is used: direction vectors are rotated by the
        probe quaternion in world space and offsets are divided by voxel
        spacing. This is correct only if CT voxel axes are perfectly aligned
        with world axes (no mesh rotation, no non-isotropic scaling).

    WITH registration context (inv_affine provided):
        Each pixel offset along u/v (in PyBullet world meters) is mapped
        through the full inverse chain into true voxel-space offsets:
            u_world (m)
            → R_body^T @ u_world           (undo body orientation)
            → / mesh_scale                  (undo non-isotropic scaling)
            × 1000                          (meters → millimeters)
            → inv_affine_3x3 @ (·)          (mm → voxel, incl. affine rotation)
        This produces direction vectors that are perfectly aligned with the
        CT data, eliminating the diagonal cropping artifact.
    """
    rot = R.from_quat(quaternion)
    R_probe = rot.as_matrix()

    # Probe local axes in PyBullet world space (dimensionless unit vectors)
    u_world = R_probe[:, 0]  # Width axis
    w_world = -R_probe[:, 2]  # Depth axis (pointing down along the beam)

    coords_u = np.arange(size) - size / 2
    coords_w = np.arange(size)  # 0 to size-1 representing depth
    ii, jj = np.meshgrid(coords_u, coords_w)

    if inv_affine is not None:
        # ── Full registration-aware direction vector transform ─────────────
        # Convert the probe's world-space unit vectors into voxel-space
        # offset vectors, compensating for body rotation, mesh scale, and
        # the NIfTI affine (which may contain its own rotation/flip).
        #
        # For a single pixel at physical distance `pixel_spacing` mm along u:
        #   displacement_world = (pixel_spacing / 1000) m  × u_world
        # Transform chain:
        #   → R_body^T @ displacement_world     (undo body rotation → mesh space)
        #   → / mesh_scale                       (undo non-isotropic stretch → m)
        #   → × 1000                             (m → mm)
        #   → inv_affine_3x3 @ (·)               (mm → voxel, incl. affine rot)

        R_body = np.eye(3, dtype=np.float64)
        if body_orientation_matrix is not None:
            R_body = np.asarray(body_orientation_matrix, dtype=np.float64)

        scale = np.ones(3, dtype=np.float64)
        if mesh_scale is not None:
            scale = np.asarray(mesh_scale, dtype=np.float64).ravel()
            if scale.size == 1:
                scale = np.full(3, scale[0])

        # Upper 3×3 of the inverse NIfTI affine (pure mm→voxel linear part)
        M_inv = np.asarray(inv_affine, dtype=np.float64)[:3, :3]

        def world_to_voxel_direction(u_w):
            """Convert a PyBullet world unit vector to a voxel-space direction
            scaled by pixel_spacing (so one grid step = one voxel offset)."""
            # pixel_spacing is in mm; convert to meters for the world→mesh step
            u_mesh  = R_body.T @ (u_w * pixel_spacing / 1000.0)
            u_phys  = u_mesh / scale          # undo non-isotropic scaling (m)
            u_mm    = u_phys * 1000.0         # metres → millimetres
            u_voxel = M_inv @ u_mm            # mm → voxel space
            return u_voxel

        u_vox = world_to_voxel_direction(u_world)
        w_vox = world_to_voxel_direction(w_world)

        # Build voxel-space offsets directly (no further spacing division needed)
        offsets_voxel = (
            ii[..., None] * u_vox +
            jj[..., None] * w_vox
        )
    else:
        # ── Legacy path: classic spacing-divide (world axes ≈ voxel axes) ───
        offsets_mm = (
            ii[..., None] * pixel_spacing * u_world +
            jj[..., None] * pixel_spacing * w_world
        )
        offsets_voxel = offsets_mm / spacing

    points = center + offsets_voxel

    # Use PyTorch grid_sample path if input is a PyTorch Tensor
    if _TORCH_OK and isinstance(volume, torch.Tensor):
        # volume shape is (D, H, W)
        D, H, W = volume.shape[-3:]
        
        # Normalize coordinates to [-1, 1]
        z_norm = (points[..., 0] / (D - 1)) * 2.0 - 1.0
        y_norm = (points[..., 1] / (H - 1)) * 2.0 - 1.0
        x_norm = (points[..., 2] / (W - 1)) * 2.0 - 1.0
        
        # PyTorch grid_sample expects (x, y, z) coordinates order
        grid = np.stack([x_norm, y_norm, z_norm], axis=-1)
        grid_t = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float().to(volume.device)
        
        vol_t = volume.unsqueeze(0).unsqueeze(0)
        mode = "nearest" if order == 0 else "bilinear"
        
        slice_t = F.grid_sample(
            vol_t, grid_t,
            mode=mode,
            padding_mode="zeros",
            align_corners=True
        )
        return slice_t.squeeze().cpu().numpy()

    # ── Fallback path: SciPy map_coordinates (CPU-only, no PyTorch dependence) ──
    sample_coords = [
        points[..., 0].ravel(),
        points[..., 1].ravel(),
        points[..., 2].ravel()
    ]

    if cval is None:
        cval = float(volume.min())

    slice_img = map_coordinates(
        volume,
        sample_coords,
        order=order,
        mode='constant',
        cval=cval
    )

    return slice_img.reshape(size, size)