import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

def compare_interpolation():
    print("Testing coordinate mapping equivalence...")
    D, H, W = 100, 120, 140
    volume = np.random.rand(D, H, W).astype(np.float32)
    
    # Create test points (size=128)
    size = 128
    z_center, y_center, x_center = 50.0, 60.0, 70.0
    
    # Simple meshgrid
    coords_u = np.arange(size) - size / 2
    coords_w = np.arange(size)
    ii, jj = np.meshgrid(coords_u, coords_w)
    
    # Arbitrary direction vectors
    u_vox = np.array([0.1, 0.2, 0.3])
    w_vox = np.array([0.4, -0.1, 0.2])
    
    offsets_voxel = (
        ii[..., None] * u_vox +
        jj[..., None] * w_vox
    )
    center = np.array([z_center, y_center, x_center])
    points = center + offsets_voxel  # (128, 128, 3)
    
    # 1. SciPy Map Coordinates (Nearest)
    sample_coords = [
        points[..., 0].ravel(),
        points[..., 1].ravel(),
        points[..., 2].ravel()
    ]
    
    t0 = time.perf_counter()
    scipy_out = map_coordinates(
        volume,
        sample_coords,
        order=0,
        mode='constant',
        cval=0.0
    ).reshape(size, size)
    scipy_time = (time.perf_counter() - t0) * 1000
    
    # 2. PyTorch Grid Sample (Nearest, CPU)
    # volume shape: (D, H, W) -> PyTorch (1, 1, D, H, W)
    vol_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
    
    # points shape: (H_out, W_out, 3) -> grid shape (1, 1, H_out, W_out, 3)
    # voxel order in points is (z, y, x). PyTorch expects (x, y, z) normalized to [-1, 1]
    z_vox = points[..., 0]
    y_vox = points[..., 1]
    x_vox = points[..., 2]
    
    z_norm = (z_vox / (D - 1)) * 2.0 - 1.0
    y_norm = (y_vox / (H - 1)) * 2.0 - 1.0
    x_norm = (x_vox / (W - 1)) * 2.0 - 1.0
    
    grid_t = torch.stack([
        torch.from_numpy(x_norm),
        torch.from_numpy(y_norm),
        torch.from_numpy(z_norm)
    ], dim=-1).unsqueeze(0).unsqueeze(0).float() # (1, 1, size, size, 3) and cast to float32
    
    t0 = time.perf_counter()
    torch_out_cpu_t = F.grid_sample(
        vol_t, grid_t,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    torch_out_cpu = torch_out_cpu_t.squeeze().numpy()
    torch_cpu_time = (time.perf_counter() - t0) * 1000
    
    # 3. PyTorch Grid Sample (Nearest, GPU) - BYPASSED FOR CPU ONLY
    torch_gpu_time = None
    torch_out_gpu = None
        
    print(f"SciPy CPU Time  : {scipy_time:.3f} ms")
    print(f"PyTorch CPU Time: {torch_cpu_time:.3f} ms")
        
    # Check max difference
    diff_cpu = np.max(np.abs(scipy_out - torch_out_cpu))
    print(f"Max Diff (SciPy vs PyTorch CPU): {diff_cpu}")

if __name__ == "__main__":
    compare_interpolation()
