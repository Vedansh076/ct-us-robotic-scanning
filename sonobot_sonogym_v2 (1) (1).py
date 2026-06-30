#!/usr/bin/env python3
"""
sonobot_sonogym.py
==================
True SonoGym Architecture: Real CT Data + CycleGAN Neural Rendering
=====================================================================

PIPELINE (per env.step() call):
  PyBullet 6-DOF probe pose  (position m, quaternion w,x,y,z)
          │
          ▼
  ┌─────────────────────────────────────────────────────┐
  │  MedicalVolumeLoader                                │
  │  nibabel → NIfTI (.nii.gz) → HU volume (float32)   │
  │  Affine: PyBullet world m  →  CT voxel ijk          │
  └─────────────────────────┬───────────────────────────┘
                            │ 3D HU volume + world→voxel transform
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  VolumeSlicer                                        │
  │  Probe 6-DOF → 2D imaging plane                     │
  │  128×128 grid of world points                       │
  │  → map_coordinates (trilinear) → 128×128 HU slice   │
  └─────────────────────────┬───────────────────────────┘
                            │ 128×128 float32 HU slice
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  NeuralUSRenderer                                    │
  │  HU → windowed & normalised [-1, 1]                 │
  │  → CycleGAN ResNet Generator (PyTorch)              │
  │  → tanh output → uint8 US image (128×128)           │
  └─────────────────────────┬───────────────────────────┘
                            │ 128×128 uint8 ultrasound image
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  ImageQualityAnalyzer + SonoSimEnv                  │
  │  Returns obs = {'image', 'quality', 'metrics'}      │
  └─────────────────────────────────────────────────────┘

FALLBACK: If CT file or model weights are absent, the system automatically
falls back to the physics-based FanBeamUSRenderer so the BO loop
always runs — even without real data.

REQUIRED FILES (place alongside this script):
  patient_ct.nii.gz       — abdominal CT scan  (e.g. CHAOS, LiTS, TCGA)
  cyclegan_generator.pt   — trained generator weights

RECOMMENDED FREE DATASETS:
  CHAOS  : https://chaos.grand-challenge.org   (liver CT, Creative Commons)
  LiTS   : https://competitions.codalab.org/competitions/17094
  TCGA   : https://www.cancerimagingarchive.net

QUICK START (Colab):
  !pip install nibabel torch torchvision
  # Download a CT:
  # from chaos_downloader import download; download('CT', 'patient_ct.nii.gz')
  # Train (or load) the CycleGAN:
  # python train_cyclegan.py --dataroot ./data --name ct2us --model cycle_gan
"""

import os, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2

# ─── Optional heavy dependencies (graceful fallback if absent) ────────────────
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel not found. Install with: pip install nibabel. "
                  "Falling back to synthetic anatomy.", stacklevel=2)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    warnings.warn("PyTorch not found. Install with: pip install torch. "
                  "Falling back to physics renderer.", stacklevel=2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CT_PATH    = Path("patient_ct.nii.gz")       # real CT scan
MODEL_PATH = Path("cyclegan_generator.pt")   # trained CycleGAN weights

# Patient body centre in PyBullet world frame (metres)
BODY_CENTER_M = np.array([0.45, 0.0, 0.32])

# Probe imaging plane dimensions
PROBE_LAT_CM   = 5.0    # lateral width of the 2D slice (cm)
PROBE_DEPTH_CM = 10.0   # imaging depth (cm)
SLICE_SIZE     = 128    # output image pixels (square)

# HU windowing for abdominal soft tissue (standard clinical window)
HU_WINDOW_CENTER = 40.0    # Hounsfield units
HU_WINDOW_WIDTH  = 400.0   # Hounsfield units
# → effective HU range: [-160, 240]

# Axis mapping: PyBullet world → CT RAS (for supine patient)
# PyBullet: X=head-to-foot, Y=lateral(right), Z=up-from-bed
# CT RAS:   X=right,        Y=anterior,       Z=superior(head)
# Rotation matrix R: body_offset_mm → RAS_offset_mm
R_PYBULLET_TO_RAS = np.array([
    [ 0.,  1.,  0.],   # RAS X (right)    ← PyBullet Y
    [ 0.,  0., -1.],   # RAS Y (anterior) ← –PyBullet Z
    [ 1.,  0.,  0.],   # RAS Z (superior) ← PyBullet X
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MEDICAL VOLUME LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class MedicalVolumeLoader:
    """
    Load a NIfTI CT scan and provide a fast world→voxel coordinate transform.

    Attributes
    ----------
    hu_volume   : np.ndarray (float32), shape (I, J, K)  — Hounsfield Units
    voxel_size  : np.ndarray (float64), shape (3,)        — voxel spacing (mm)
    affine      : np.ndarray (float64), shape (4,4)       — voxel→world mm
    affine_inv  : np.ndarray (float64), shape (4,4)       — world mm→voxel
    shape       : tuple (I, J, K)

    Coordinate contract
    -------------------
    The loader exposes  world_m_to_voxel(pts_m)  which converts PyBullet
    world positions (in metres, shape (3, N)) to voxel indices (float, (3, N)).
    Indices outside [0, dim-1] are valid inputs to map_coordinates (clamped
    or extrapolated depending on the `mode` argument you pass downstream).
    """

    def __init__(self, nifti_path: Path,
                 body_center_m: np.ndarray = BODY_CENTER_M,
                 r_pybullet_to_ras: np.ndarray = R_PYBULLET_TO_RAS):

        if not HAS_NIBABEL:
            raise RuntimeError("nibabel is required for MedicalVolumeLoader. "
                               "pip install nibabel")
        if not nifti_path.exists():
            raise FileNotFoundError(f"CT file not found: {nifti_path}\n"
                                    "Download a free abdominal CT (see header docstring).")

        print(f"  Loading CT: {nifti_path} … ", end='', flush=True)
        t0 = time.time()

        img = nib.load(str(nifti_path))
        # Load as float32 immediately to avoid repeated dtype conversions
        self.hu_volume   = np.asarray(img.get_fdata(dtype=np.float32))
        self.affine      = img.affine.astype(np.float64)   # voxel → mm (RAS)
        self.affine_inv  = np.linalg.inv(self.affine)       # mm → voxel
        self.voxel_size  = np.abs(np.diag(self.affine)[:3]) # mm per voxel
        self.shape       = self.hu_volume.shape

        # Registration: body-frame offset (mm, RAS) → CT voxel
        # We map the patient body centre to the CT volume centre.
        self._body_center_m       = body_center_m.astype(np.float64)
        self._r_pb_to_ras         = r_pybullet_to_ras.astype(np.float64)
        self._ct_center_mm        = self._compute_ct_center_mm()

        elapsed = time.time() - t0
        hu_min, hu_max = self.hu_volume.min(), self.hu_volume.max()
        print(f"done ({elapsed:.1f}s)  shape={self.shape}  "
              f"HU=[{hu_min:.0f},{hu_max:.0f}]  "
              f"voxel={self.voxel_size.round(2)} mm")

    def _compute_ct_center_mm(self) -> np.ndarray:
        """CT volume centre in mm (RAS), computed from affine."""
        center_vox = np.array(self.shape, dtype=np.float64) / 2.0
        return (self.affine @ np.append(center_vox, 1.0))[:3]

    def world_m_to_voxel(self, pts_world_m: np.ndarray) -> np.ndarray:
        """
        Convert PyBullet world positions to CT voxel indices.

        Parameters
        ----------
        pts_world_m : (3, N) or (3,) float64 — world positions in metres

        Returns
        -------
        voxel_ijk   : (3, N) or (3,) float64 — fractional voxel indices
                      (suitable for scipy.ndimage.map_coordinates)
        """
        scalar = pts_world_m.ndim == 1
        pts = pts_world_m.reshape(3, -1)                      # (3, N)

        # Step 1: PyBullet world m  →  body-relative offset mm
        body_offset_mm = (pts - self._body_center_m[:, None]) * 1000.0  # (3, N)

        # Step 2: body-relative mm (PyBullet axes)  →  RAS mm offset
        ras_offset_mm = self._r_pb_to_ras @ body_offset_mm               # (3, N)

        # Step 3: RAS offset  →  absolute RAS mm  →  voxel ijk
        ras_abs_mm = self._ct_center_mm[:, None] + ras_offset_mm          # (3, N)
        hom        = np.vstack([ras_abs_mm, np.ones((1, ras_abs_mm.shape[1]))])  # (4, N)
        vox_hom    = self.affine_inv @ hom                                 # (4, N)
        voxels     = vox_hom[:3]                                           # (3, N)

        return voxels[:, 0] if scalar else voxels

    def get_hu_stats(self) -> dict:
        """Return summary stats about the loaded volume (useful for debugging)."""
        return dict(shape=self.shape,
                    voxel_mm=self.voxel_size.tolist(),
                    hu_min=float(self.hu_volume.min()),
                    hu_max=float(self.hu_volume.max()),
                    hu_mean=float(self.hu_volume.mean()))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VOLUME SLICER
# ═══════════════════════════════════════════════════════════════════════════════
class VolumeSlicer:
    """
    Extract a 2D imaging plane from a 3D CT volume given a probe 6-DOF pose.

    The imaging plane is defined by the probe's coordinate frame:
      • Lateral axis  : probe X-axis  (scans left ↔ right)
      • Depth axis    : probe Z-axis  (beam direction, into body)
      • Slice plane   : spanned by lateral × depth

    A 2D grid of (SLICE_SIZE × SLICE_SIZE) sample points is placed on this
    plane, spanning PROBE_LAT_CM laterally and PROBE_DEPTH_CM in depth.
    Voxel values are retrieved by trilinear interpolation.

    Speed notes
    -----------
    • The probe rotation matrix is cached; recomputed only when quaternion changes.
    • All numpy operations are vectorised — no Python loops.
    • Typical latency: 2–8 ms per slice on CPU.
    """

    def __init__(self, loader: MedicalVolumeLoader,
                 slice_size: int          = SLICE_SIZE,
                 lat_cm:     float        = PROBE_LAT_CM,
                 depth_cm:   float        = PROBE_DEPTH_CM):

        self.loader    = loader
        self.size      = slice_size
        self.lat_cm    = lat_cm
        self.depth_cm  = depth_cm

        # Pre-compute the 2D grid in the probe's local frame (in cm)
        # Shape: (2, size, size) — (lat, depth) coordinates
        lats   = np.linspace(-lat_cm / 2., lat_cm / 2., slice_size)
        depths = np.linspace(0., depth_cm, slice_size)
        L, D   = np.meshgrid(lats, depths, indexing='ij')   # (size, size)
        # Stack as (2, size*size) for easy dot products
        self._ld_flat = np.stack([L.ravel(), D.ravel()], axis=0)  # (2, N)

        # Cache last quaternion to avoid redundant R recomputation
        self._last_q  = None
        self._last_R  = None

    def _quat_to_R(self, q_wxyz: np.ndarray) -> np.ndarray:
        """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
        w, x, y, z = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-9)
        return np.array([
            [1-2*y*y-2*z*z,  2*x*y-2*w*z,   2*x*z+2*w*y],
            [2*x*y+2*w*z,    1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y,    2*y*z+2*w*x,   1-2*x*x-2*y*y],
        ], dtype=np.float64)

    def extract(self, probe_pos_m:    np.ndarray,
                       probe_quat_wxyz: np.ndarray,
                       interp_order:    int = 1) -> np.ndarray:
        """
        Extract a 2D HU slice from the CT volume.

        Parameters
        ----------
        probe_pos_m     : (3,) world position of the probe face (metres)
        probe_quat_wxyz : (4,) probe orientation quaternion (w,x,y,z)
        interp_order    : 0=nearest, 1=linear (default), 3=cubic

        Returns
        -------
        hu_slice : (SLICE_SIZE, SLICE_SIZE) float32  — Hounsfield Units
                   Values outside the CT volume are filled with -1000 HU (air).
        """
        pos = np.asarray(probe_pos_m,    dtype=np.float64)
        q   = np.asarray(probe_quat_wxyz, dtype=np.float64)

        # Cache rotation matrix
        if self._last_q is None or not np.allclose(q, self._last_q, atol=1e-6):
            self._last_R = self._quat_to_R(q)
            self._last_q = q.copy()

        R = self._last_R
        # Probe axes in world frame (column vectors of R)
        lat_dir  = R[:, 0]   # probe X-axis (lateral direction)
        beam_dir = R[:, 2]   # probe Z-axis (beam / depth direction)

        # World positions of all sample points (m): (3, size*size)
        # pts_world = pos + lat*lat_dir + depth*beam_dir   [lat,depth in cm → m]
        pts_world_m = (pos[:, None]
                       + lat_dir[:, None]  * (self._ld_flat[0:1] / 100.)
                       + beam_dir[:, None] * (self._ld_flat[1:2] / 100.))  # (3, N)

        # Convert world positions → CT voxel indices
        vox_coords = self.loader.world_m_to_voxel(pts_world_m)   # (3, N)

        # Trilinear interpolation (map_coordinates expects (3, N) coords)
        hu_flat = map_coordinates(
            self.loader.hu_volume,
            vox_coords,
            order=interp_order,
            mode='constant',
            cval=-1000.0,    # outside volume → air HU
        )

        return hu_flat.reshape(self.size, self.size).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CYCLEGAN GENERATOR (ResNet Architecture)
# ═══════════════════════════════════════════════════════════════════════════════
class _ResnetBlock(nn.Module):
    """
    Standard ResNet block used in CycleGAN / pix2pix generators.
    Padding: ReflectionPad2d → avoids checkerboard artefacts.
    Norm: InstanceNorm2d   → standard for image-translation tasks.
    """
    def __init__(self, channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)   # residual connection


class CycleGANGenerator(nn.Module):
    """
    ResNet-based generator for CT→Ultrasound image translation.
    Matches the architecture used by pytorch-CycleGAN-and-pix2pix
    (Zhu et al., ICCV 2017) with 1-channel I/O for grayscale images.

    Architecture (for 128×128 input, n_blocks=6):
      Encoder:     7×7 conv, 2× stride-2 conv  → 32×32, 256ch
      Transformer: 6 ResNet blocks              → 32×32, 256ch
      Decoder:     2× stride-2 deconv           → 128×128, 64ch
      Output:      7×7 conv + Tanh              → 128×128, 1ch

    Parameters
    ----------
    input_nc   : int  — input channels  (1 for grayscale CT)
    output_nc  : int  — output channels (1 for grayscale US)
    ngf        : int  — base filter count (64)
    n_blocks   : int  — ResNet blocks (6 for 128px, 9 for 256px)
    use_dropout: bool — dropout in ResBlocks (False for inference)
    """

    def __init__(self,
                 input_nc:    int  = 1,
                 output_nc:   int  = 1,
                 ngf:         int  = 64,
                 n_blocks:    int  = 6,
                 use_dropout: bool = False):
        super().__init__()
        assert n_blocks >= 0

        # ── Encoder ──────────────────────────────────────────────────────────
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Two downsampling layers: 128→64→32
        for i in range(2):
            mult = 2 ** i                                  # 1, 2
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # ── Transformer (ResBlocks) ───────────────────────────────────────────
        mult    = 4                   # channels = ngf * 4 = 256
        resblks = [_ResnetBlock(ngf * mult, use_dropout) for _ in range(n_blocks)]

        # ── Decoder ──────────────────────────────────────────────────────────
        decoder = []
        for i in range(2):
            mult = 2 ** (2 - i)      # 4, 2
            decoder += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *resblks, *decoder)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEURAL US RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
class NeuralUSRenderer:
    """
    PyTorch inference wrapper: CT HU slice → synthetic US image.

    Steps per call:
      1. Apply HU windowing  (soft-tissue: C=40, W=400)
      2. Normalise to [-1, 1]
      3. Run CycleGAN generator (torch.no_grad)
      4. Rescale tanh output to [0, 255] uint8

    Speed optimisations:
      • Model loaded once at init; stays on GPU/CPU throughout
      • Input tensor pre-allocated and reused (avoid alloc overhead)
      • torch.inference_mode() context manager (faster than no_grad)
      • Optional FP16 (half) for ~2× speedup on GPU with minor quality loss
    """

    def __init__(self,
                 weights_path: Path = MODEL_PATH,
                 device:       "torch.device" = None,
                 use_fp16:     bool = False,
                 ngf:          int  = 64,
                 n_blocks:     int  = 6):

        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for NeuralUSRenderer. "
                               "pip install torch")

        self.device   = device or DEVICE
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')

        # Instantiate generator
        self.generator = CycleGANGenerator(input_nc=1, output_nc=1,
                                            ngf=ngf, n_blocks=n_blocks)

        # Load weights
        if not weights_path.exists():
            raise FileNotFoundError(
                f"CycleGAN weights not found: {weights_path}\n"
                "Options:\n"
                "  1. Train: python train_cyclegan.py --dataroot ./ct_us_pairs\n"
                "  2. Download pre-trained weights (see README)\n"
                "  3. Omit path → use physics fallback in SonoSimEnv")

        print(f"  Loading CycleGAN weights: {weights_path} … ", end='', flush=True)
        t0       = time.time()
        state    = torch.load(str(weights_path), map_location=self.device)
        # Support both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict) and 'generator' in state:
            state = state['generator']
        self.generator.load_state_dict(state, strict=True)
        self.generator.to(self.device)
        self.generator.eval()
        if self.use_fp16:
            self.generator.half()
        print(f"done ({time.time()-t0:.2f}s)  device={self.device}  "
              f"fp16={self.use_fp16}")

        # Pre-allocate input buffer: (1, 1, size, size)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self._buf = torch.zeros(1, 1, SLICE_SIZE, SLICE_SIZE,
                                dtype=dtype, device=self.device)

    def _apply_window(self, hu: np.ndarray) -> np.ndarray:
        """
        Apply a clinical HU window and normalise to [-1, 1].

        Window centre = 40 HU, width = 400 HU
        → effective range: [-160 HU, +240 HU]
        Tissues below -160 HU are clipped to -1 (air→dark)
        Tissues above +240 HU are clipped to +1 (bone→bright)
        """
        lo  = HU_WINDOW_CENTER - HU_WINDOW_WIDTH / 2.
        hi  = HU_WINDOW_CENTER + HU_WINDOW_WIDTH / 2.
        clipped = np.clip(hu, lo, hi)
        return ((clipped - lo) / (hi - lo) * 2. - 1.).astype(np.float32)

    def render(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        Convert a 128×128 HU slice to a 128×128 uint8 ultrasound image.

        Parameters
        ----------
        hu_slice : (128, 128) float32 — Hounsfield Units

        Returns
        -------
        us_image : (128, 128) uint8 — synthetic B-mode ultrasound
        """
        # Window + normalise → [-1, 1]
        norm = self._apply_window(hu_slice)

        # Copy into pre-allocated tensor (zero-copy if same shape)
        self._buf[0, 0].copy_(torch.from_numpy(norm))

        # Generator inference
        with torch.inference_mode():
            out = self.generator(self._buf)     # (1, 1, 128, 128), tanh ∈[-1,1]

        # Tanh → [0, 255] uint8
        out_np = out[0, 0].cpu().float().numpy()
        us     = np.clip((out_np + 1.) / 2. * 255., 0., 255.).astype(np.uint8)
        return us

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.generator.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS FALLBACK CLASSES (kept from previous version)
# ═══════════════════════════════════════════════════════════════════════════════
class Tissue:
    AIR=0;SKIN=1;FAT=2;MUSCLE=3;FASCIA=4;LIVER=5;LIVER_VES=6;GALLBLADDER=7
    KIDNEY_COR=8;KIDNEY_MED=9;BOWEL_WALL=10;BOWEL_GAS=11;AORTA=12;IVC=13
    SPINE=14;BLADDER_W=15;BLADDER_F=16
    Z = {0:0.0004,1:1.68,2:1.38,3:1.70,4:1.72,5:1.65,6:1.62,7:1.51,
         8:1.62,9:1.58,10:1.68,11:0.0004,12:1.62,13:1.62,14:7.80,15:1.65,16:1.52}
    ECHO = {0:0.00,1:0.15,2:0.12,3:0.22,4:0.08,5:0.25,6:0.00,7:0.00,
            8:0.22,9:0.08,10:0.20,11:0.85,12:0.00,13:0.00,14:0.55,15:0.08,16:0.00}
    ATTEN = {0:10.0,1:0.35,2:0.48,3:0.57,4:0.80,5:0.50,6:0.18,7:0.05,
             8:0.55,9:0.55,10:0.60,11:8.00,12:0.18,13:0.18,14:2.50,15:0.40,16:0.08}
    POST = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:+1,8:0,9:0,10:0,11:-1,12:+1,13:+1,14:-1,15:0,16:+1}

class SyntheticTorsoVolume:
    DX,NX=15.,150; DY,NY=18.,180; DZ,NZ=12.,120
    def __init__(self, verbose=True):
        if verbose: print("  Building synthetic anatomy … ", end='', flush=True)
        t0=time.time()
        x=np.linspace(-self.DX/2,self.DX/2,self.NX)
        y=np.linspace(-self.DY/2,self.DY/2,self.NY)
        z=np.linspace(0,self.DZ,self.NZ)
        X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
        self.vol=self._build(X,Y,Z)
        self.x_ax,self.y_ax,self.z_ax=x,y,z
        self.Zvol   =self._lut(Tissue.Z)
        self.echvol =self._lut(Tissue.ECHO)
        self.attvol =self._lut(Tissue.ATTEN)
        self.postvol=np.array([Tissue.POST.get(i,0) for i in range(17)],dtype=np.float32)[self.vol]
        if verbose: print(f"done ({time.time()-t0:.1f}s)")
    def _lut(self,d):
        return np.array([d.get(i,0.) for i in range(17)],dtype=np.float32)[self.vol]
    @staticmethod
    def _ell(X,Y,Z,cx,cy,cz,rx,ry,rz):
        return ((X-cx)**2/rx**2+(Y-cy)**2/ry**2+(Z-cz)**2/rz**2)<1.
    @staticmethod
    def _cyl(X,Y,Z,cx,cz,r,y0,y1):
        return ((X-cx)**2+(Z-cz)**2<r**2)&(Y>y0)&(Y<y1)
    def _build(self,X,Y,Z):
        T=Tissue; v=np.zeros((self.NX,self.NY,self.NZ),dtype=np.uint8)
        v[Z<0.4]=T.SKIN; v[(Z>=0.4)&(Z<2.5)&(np.abs(X)<7.)]=T.FAT
        for xc in [-1.8,1.8]: v[(Z>=2.5)&(Z<5.)&(np.abs(X-xc)<1.5)&(Y>-8.)&(Y<7.)]=T.MUSCLE
        for s in [-1,1]: v[(Z>=1.5)&(Z<4.5)&(s*X>2.5)&(s*X<7.)]=T.MUSCLE
        v[(Z>=2.5)&(Z<5.5)&(np.abs(X)<0.3)&(Y>-8.)&(Y<7.)]=T.FASCIA
        v[(np.abs(Z-2.5)<0.25)&(np.abs(X)<5.)&(Y>-8.)&(Y<7.)]=T.FASCIA
        liv=self._ell(X,Y,Z,-3.,-3.5,7.5,5.5,5.,4.)|self._ell(X,Y,Z,1.,-4.,6.5,2.5,3.5,3.)
        v[liv&(Z>4.5)]=T.LIVER
        v[self._cyl(X,Y,Z,-3.,7.,0.55,-6.,-1.)&(v==T.LIVER)]=T.LIVER_VES
        v[self._cyl(X,Y,Z,-1.5,6.5,.4,-8.,-2.)&(v==T.LIVER)]=T.LIVER_VES
        v[self._ell(X,Y,Z,-4.5,-1.5,7.,1.8,1.5,1.8)]=T.GALLBLADDER
        v[self._ell(X,Y,Z,-5.,-1.,9.5,2.5,2.8,2.)]=T.KIDNEY_COR
        v[self._ell(X,Y,Z,-5.,-1.,9.5,1.4,1.7,1.1)]=T.KIDNEY_MED
        v[self._ell(X,Y,Z, 5., .5,9.,2.3,2.6,1.9)]=T.KIDNEY_COR
        v[self._ell(X,Y,Z, 5., .5,9.,1.3,1.5,1.1)]=T.KIDNEY_MED
        v[self._cyl(X,Y,Z,.8,9.5,.8,-9.,9.)]=T.AORTA
        v[self._cyl(X,Y,Z,-.8,9.2,.65,-9.,9.)]=T.IVC
        rng=np.random.RandomState(42)
        for _ in range(8):
            bx=rng.uniform(-3,3);by=rng.uniform(-1,4);bz=rng.uniform(5,7.5);br=rng.uniform(.6,1.)
            loop=(np.sqrt((X-bx)**2+(Z-bz)**2)<br)&(np.abs(Y-by)<1.2)
            inner=(np.sqrt((X-bx)**2+(Z-bz)**2)<br*.5)&(np.abs(Y-by)<1.)
            v[loop&(v==0)]=T.BOWEL_WALL; v[inner&(v==T.BOWEL_WALL)]=T.BOWEL_GAS
        v[self._ell(X,Y,Z,0.,6.5,7.,3.,2.,3.)]=T.BLADDER_W
        v[self._ell(X,Y,Z,0.,6.5,7.,2.6,1.6,2.6)]=T.BLADDER_F
        v[(np.abs(X)<2.)&(Z>10.)]=T.SPINE; v[(Z<0.15)&(v>T.FAT)]=T.SKIN
        return v

class FanBeamUSRenderer:
    """Physics fallback renderer (fan-beam + impedance boundaries + TGC)."""
    N_LINES=192; N_DEPTH=512; DEPTH_CM=15.; FAN_DEG=60.; FREQ=3.5; IM=512; DYN_DB=60.
    def __init__(self,volume,seed=0):
        self.vol=volume; self.rng=np.random.RandomState(seed)
        self.FH=np.radians(self.FAN_DEG/2.)
        self.angles=np.linspace(-self.FH,self.FH,self.N_LINES)
        self.depths=np.linspace(0.,self.DEPTH_CM,self.N_DEPTH)
        self.dz=self.DEPTH_CM/(self.N_DEPTH-1.)
        self.spk_sigma=max(1.5,0.44/(self.DEPTH_CM*10./self.N_DEPTH))
        self._build_sc_lut()
    def _build_sc_lut(self):
        FH=self.FH; DC=self.DEPTH_CM; IM=self.IM
        half_w=DC*np.sin(FH)
        px=np.arange(IM,dtype=np.float32); py=np.arange(IM,dtype=np.float32)
        PX,PY=np.meshgrid(px,py,indexing='ij')
        lat=(PX/(IM-1.)-0.5)*2.*half_w; dep=(PY/(IM-1.))*DC
        r=np.sqrt(lat**2+dep**2); theta=np.arctan2(lat,dep+1e-6)
        self._mask=(np.abs(theta)<=FH)&(r<=DC)&(dep>=0.4)
        self._sc_line=np.clip((theta+FH)/(2.*FH)*(self.N_LINES-1.),0.,self.N_LINES-1.).astype(np.float32)
        self._sc_depth=np.clip(r/DC*(self.N_DEPTH-1.),0.,self.N_DEPTH-1.).astype(np.float32)
    def _interp(self,prop,wx,wy,wz):
        V=self.vol
        ix=np.clip((wx-V.x_ax[0])/(V.x_ax[-1]-V.x_ax[0])*(V.NX-1),0.,V.NX-1.)
        iy=np.clip((wy-V.y_ax[0])/(V.y_ax[-1]-V.y_ax[0])*(V.NY-1),0.,V.NY-1.)
        iz=np.clip((wz-V.z_ax[0])/(V.z_ax[-1]-V.z_ax[0])*(V.NZ-1),0.,V.NZ-1.)
        return map_coordinates(prop,[ix.ravel(),iy.ravel(),iz.ravel()],order=1,mode='constant',cval=0.).reshape(wx.shape)
    def _speckle(self,shape):
        xi=gaussian_filter(self.rng.randn(*shape).astype(np.float32),(0.,self.spk_sigma))
        xq=gaussian_filter(self.rng.randn(*shape).astype(np.float32),(0.,self.spk_sigma))
        env=np.sqrt(xi**2+xq**2); return env/(env.mean()+1e-8)
    def render(self,probe_pos_cm,probe_R):
        NL,ND=self.N_LINES,self.N_DEPTH
        sin_a=np.sin(self.angles); cos_a=np.cos(self.angles)
        dirs_body=np.stack([sin_a,np.zeros_like(sin_a),cos_a],axis=1)@probe_R.T
        pos=(probe_pos_cm[:,None,None]+dirs_body.T[:,:,None]*self.depths[None,None,:])
        wx,wy,wz=pos[0],pos[1],pos[2]
        Z_s=self._interp(self.vol.Zvol,wx,wy,wz); echo_s=self._interp(self.vol.echvol,wx,wy,wz)
        att_s=self._interp(self.vol.attvol,wx,wy,wz); post_s=self._interp(self.vol.postvol,wx,wy,wz)
        Z_n=np.roll(Z_s,-1,1); Z_p=np.roll(Z_s,1,1)
        boundary=np.clip(np.abs(Z_n-Z_p)/(Z_n+Z_p+1e-6)/0.65,0.,1.)
        raw=0.70*boundary+0.30*echo_s*self._speckle((NL,ND))
        raw*=np.exp(-np.cumsum(att_s,axis=1)*self.dz*self.FREQ/20.)
        raw*=np.power(10.,(0.50*self.FREQ*self.depths[None,:])/20.)
        cum_p=np.cumsum(post_s,axis=1)
        raw*=(1.+0.5*np.tanh(np.clip(cum_p,0.,None)*.4))*np.exp(-.7*np.clip(-cum_p,0.,None))
        nf=max(2,int(0.6/self.dz)); raw[:,nf:nf+nf]+=np.exp(-np.arange(nf)/(nf*.25))[None,:]*0.15
        raw=np.clip(raw,0.,None); top=np.percentile(raw[raw>0.001],99) if (raw>0.001).any() else 1.
        raw/=(top+1e-8)
        scan_data=np.clip(np.log1p(raw*999.)/np.log(1000.),0.,1.).astype(np.float32)
        mask=self._mask
        coords=np.array([self._sc_line[mask].ravel(),self._sc_depth[mask].ravel()])
        vals=map_coordinates(scan_data,coords,order=1,mode='constant',cval=0.)
        img=np.zeros((self.IM,self.IM),dtype=np.float32); img[mask]=vals
        img=1./(1.+np.exp(-7.*(img-0.42))); img[~mask]=0.
        img=gaussian_filter(img,(0.8,0.)); img[~mask]=0.
        return np.clip(img*255.,0.,255.).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════
class ImageQualityAnalyzer:
    """Score image quality from content alone — no ground-truth needed."""
    _T = {'liver':(45,130),'gallbladder':(0,35),'kidney':(40,135),'bladder':(0,30),'vessels':(0,35)}
    def __init__(self,target='liver'): self.target=target
    def score(self,image):
        if image.mean()<5.: return dict(quality=0.,visibility=0.,contrast=0.,confidence=0.,sharpness=0.)
        f=image.astype(np.float32)/255.
        d=np.exp(-3.*np.linspace(0,1,image.shape[1])); l=np.exp(-4.*np.linspace(-1,1,image.shape[0])**2)
        conf=l[:,None]*d[None,:]; conf*=0.4+0.6*(f>0.04)
        q_conf=float(conf.mean())
        gx,gy=np.gradient(f); q_sharp=float(np.clip(np.sum(np.hypot(gx,gy)*conf)/(conf.sum()+1e-8)*12.,0.,1.))
        lo,hi=self._T.get(self.target,(40,180))
        tpx=(image>=lo)&(image<=hi); q_vis=float(np.clip(tpx.mean()*8.,0.,1.))
        q_cont=float(np.clip(abs(f[tpx].mean()-f[~tpx].mean())/(f[tpx].mean()+f[~tpx].mean()+1e-8)*2.,0.,1.)) if tpx.any() and (~tpx).any() else 0.
        return dict(quality=float(0.30*q_conf+0.25*q_sharp+0.25*q_vis+0.20*q_cont),
                    visibility=q_vis,contrast=q_cont,confidence=q_conf,sharpness=q_sharp)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SonoSimEnv — Updated Gym Interface
# ═══════════════════════════════════════════════════════════════════════════════
class SonoSimEnv:
    """
    Gym-style ultrasound simulation environment.
    Automatically selects the best available renderer:

      Priority 1 (best):   Real CT + CycleGAN neural rendering
      Priority 2 (good):   Physics-based fan-beam (synthetic anatomy)
      Priority 3 (fallback): Error — should not reach here

    API is UNCHANGED:
        obs = env.step(probe_pos_m, probe_quat_wxyz)
        obs['image']   → (128, 128) uint8 ultrasound image
        obs['quality'] → float in [0, 1]
        obs['metrics'] → dict with sub-scores
        obs['renderer']→ str: 'neural' or 'physics'

    Parameters
    ----------
    ct_path      : Path to .nii.gz CT scan (None → physics fallback)
    model_path   : Path to CycleGAN .pt weights (None → physics fallback)
    target       : Quality metric target organ ('liver', 'gallbladder', etc.)
    use_fp16     : Use FP16 inference (GPU only, ~2× speedup)
    verbose      : Print initialisation info
    """

    BODY_CENTER_M = BODY_CENTER_M   # patient body centre in PyBullet (m)

    def __init__(self,
                 ct_path:    Path   = CT_PATH,
                 model_path: Path   = MODEL_PATH,
                 target:     str    = 'liver',
                 use_fp16:   bool   = False,
                 verbose:    bool   = True):

        self.target   = target
        self.quality  = ImageQualityAnalyzer(target=target)
        self.history  = []
        self._mode    = None   # set below

        if verbose:
            print("=" * 55)
            print("  SonoSimEnv initialisation")
            print("=" * 55)

        # ── Try real CT + neural renderer ─────────────────────────────────────
        self._loader   = None
        self._slicer   = None
        self._neural   = None
        self._phys_vol = None
        self._phys_ren = None

        neural_ok = self._try_init_neural(ct_path, model_path, use_fp16, verbose)

        if neural_ok:
            self._mode = 'neural'
            if verbose:
                print("  ✓ Neural renderer active (CT + CycleGAN)")
        else:
            # Fallback to physics renderer
            if verbose:
                print("  ⚠  Neural renderer unavailable — using physics fallback")
            self._phys_vol = SyntheticTorsoVolume(verbose=verbose)
            self._phys_ren = FanBeamUSRenderer(self._phys_vol)
            self._mode = 'physics'
            if verbose:
                print("  ✓ Physics renderer active")

        if verbose:
            print(f"  Mode: {self._mode}  |  Target: {target}")
            print("=" * 55)

    def _try_init_neural(self, ct_path, model_path, use_fp16, verbose) -> bool:
        """Attempt to init the CT loader + neural renderer. Return False on any error."""
        try:
            if not HAS_NIBABEL:
                if verbose: print("  ℹ nibabel not installed → physics fallback")
                return False
            if not HAS_TORCH:
                if verbose: print("  ℹ PyTorch not installed → physics fallback")
                return False
            if not Path(ct_path).exists():
                if verbose: print(f"  ℹ CT file not found: {ct_path} → physics fallback")
                return False
            if not Path(model_path).exists():
                if verbose: print(f"  ℹ Model weights not found: {model_path} → physics fallback")
                return False

            self._loader = MedicalVolumeLoader(ct_path,
                                               body_center_m=self.BODY_CENTER_M,
                                               r_pybullet_to_ras=R_PYBULLET_TO_RAS)
            self._slicer = VolumeSlicer(self._loader)
            self._neural = NeuralUSRenderer(Path(model_path), use_fp16=use_fp16)
            return True

        except Exception as exc:
            if verbose: print(f"  ⚠  Neural init failed ({exc.__class__.__name__}: {exc})")
            return False

    # ── Core step ─────────────────────────────────────────────────────────────
    def step(self, probe_pos_m, probe_quat_wxyz) -> dict:
        """
        Simulate a US image for the given probe pose.

        Parameters
        ----------
        probe_pos_m      : array-like (3,) — probe tip position, world metres
        probe_quat_wxyz  : array-like (4,) — orientation quaternion (w,x,y,z)

        Returns
        -------
        obs : dict
            'image'    : (128, 128) uint8 — ultrasound image
            'quality'  : float [0,1]
            'metrics'  : dict
            'renderer' : 'neural' | 'physics'
            'pos_m'    : probe position used
            'q_wxyz'   : quaternion used
        """
        pos_m = np.asarray(probe_pos_m,    dtype=np.float64)
        q     = np.asarray(probe_quat_wxyz, dtype=np.float64)
        q    /= np.linalg.norm(q) + 1e-9

        if self._mode == 'neural':
            image = self._step_neural(pos_m, q)
        else:
            image = self._step_physics(pos_m, q)

        metrics = self.quality.score(image)
        obs = dict(image=image, quality=metrics['quality'],
                   metrics=metrics, renderer=self._mode,
                   pos_m=pos_m, q_wxyz=q)
        self.history.append(obs)
        return obs

    def _step_neural(self, pos_m: np.ndarray, q: np.ndarray) -> np.ndarray:
        """CT slice + CycleGAN → US image."""
        hu_slice = self._slicer.extract(pos_m, q, interp_order=1)
        return self._neural.render(hu_slice)

    def _step_physics(self, pos_m: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Physics fan-beam renderer → US image."""
        diff = pos_m - self.BODY_CENTER_M
        pos_cm = np.array([diff[1]*100., diff[0]*100., max(0., -diff[2]*100.+0.5)])
        w,x,y,z = q
        R_world = np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                             [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                             [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])
        R_wb    = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        R_body  = R_wb @ R_world
        return self._phys_ren.render(pos_cm, R_body)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def reset(self):
        self.history.clear()

    @property
    def mode(self) -> str:
        return self._mode

    def get_hu_slice(self, probe_pos_m, probe_quat_wxyz) -> np.ndarray:
        """
        Return raw HU slice (neural mode only) for debugging / training.
        Raises RuntimeError if not in neural mode.
        """
        if self._mode != 'neural':
            raise RuntimeError("get_hu_slice only available in neural mode.")
        return self._slicer.extract(
            np.asarray(probe_pos_m, float),
            np.asarray(probe_quat_wxyz, float))

    def benchmark(self, n_calls: int = 20) -> dict:
        """Time the env.step() pipeline over n_calls frames."""
        import time as _time
        pos = BODY_CENTER_M + np.array([-0.03, 0., 0.])
        q   = np.array([0., 1., 0., 0.])
        times = []
        for _ in range(n_calls):
            t0 = _time.perf_counter()
            self.step(pos, q)
            times.append(_time.perf_counter() - t0)
        times = np.array(times[2:])  # skip 2 warm-up
        return dict(mean_ms=float(times.mean()*1000.),
                    std_ms=float(times.std()*1000.),
                    min_ms=float(times.min()*1000.),
                    max_ms=float(times.max()*1000.),
                    fps=float(1./times.mean()),
                    mode=self._mode)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DISPLAY (PyBullet loop integration)
# ═══════════════════════════════════════════════════════════════════════════════
class LiveUSDisplay:
    """Real-time US display using OpenCV (drop-in for PyBullet loop)."""
    def __init__(self, win_name="SonoBot US"):
        self._ok=False
        try:
            cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name,520,550); self._ok=True; self._w=win_name
        except Exception: pass

    def update(self, image, quality, renderer='physics', metrics=None):
        if not self._ok: return
        d = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        if d.shape[0] != 520:
            d = cv2.resize(d,(520,520),interpolation=cv2.INTER_LINEAR)
        bw=int(quality*d.shape[1])
        col=(0,int(255*quality),255-int(255*quality))
        cv2.rectangle(d,(0,d.shape[0]-22),(bw,d.shape[0]),col,-1)
        mode_color=(0,255,128) if renderer=='neural' else (0,180,255)
        cv2.putText(d,f"Q={quality:.3f} [{renderer}]",(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.60,mode_color,2)
        if metrics:
            for i,(k,v) in enumerate(metrics.items()):
                if k!='quality':
                    cv2.putText(d,f"{k[:5]}={v:.2f}",(8,50+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        cv2.imshow(self._w,d); cv2.waitKey(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES (CycleGAN training script template)
# ═══════════════════════════════════════════════════════════════════════════════
def print_training_guide():
    """Print instructions for training the CycleGAN CT→US translator."""
    guide = """
╔══════════════════════════════════════════════════════════════════════╗
║          CycleGAN CT → Ultrasound  Training Guide                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STEP 1 — Get free abdominal CT data                                 ║
║    CHAOS: https://chaos.grand-challenge.org  (CC BY licence)         ║
║    Format: DICOM or NIfTI, abdominal CT, 20 patients                 ║
║                                                                      ║
║  STEP 2 — Get US images (unpaired is fine for CycleGAN)              ║
║    CAMUS:  https://www.creatis.insa-lyon.fr/Challenge/camus          ║
║    CLUST:  https://clust.grand-challenge.org                         ║
║    Any public abdominal B-mode dataset works                         ║
║                                                                      ║
║  STEP 3 — Preprocess                                                 ║
║    • Convert CT to axial PNG slices (128×128, soft-tissue window)    ║
║    • Resize US images to 128×128                                     ║
║    • Place in:  ./data/ct2us/trainA/  (CT slices)                    ║
║                 ./data/ct2us/trainB/  (US images)                    ║
║                                                                      ║
║  STEP 4 — Train with pytorch-CycleGAN-and-pix2pix                   ║
║    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix║
║    cd pytorch-CycleGAN-and-pix2pix                                   ║
║    python train.py \\                                                 ║
║        --dataroot ../data/ct2us \\                                    ║
║        --name ct2us_128 \\                                            ║
║        --model cycle_gan \\                                           ║
║        --input_nc 1 --output_nc 1 \\                                  ║
║        --crop_size 128 --load_size 128 \\                             ║
║        --n_epochs 100 --n_epochs_decay 100                           ║
║                                                                      ║
║  STEP 5 — Export weights                                             ║
║    cp checkpoints/ct2us_128/latest_net_G_A.pth cyclegan_generator.pt║
║                                                                      ║
║  STEP 6 — Drop files next to sonobot_sonogym.py                      ║
║    patient_ct.nii.gz   (any subject from CHAOS)                      ║
║    cyclegan_generator.pt                                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(guide)


def export_generator_weights(ckpt_path: str, out_path: str = "cyclegan_generator.pt"):
    """
    Helper: extract generator weights from a pytorch-CycleGAN checkpoint.
    The CycleGAN repo saves the full model; this extracts just the state dict.

    Usage:
        export_generator_weights('checkpoints/ct2us/latest_net_G_A.pth')
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pytorch-CycleGAN-and-pix2pix saves the raw state dict in .pth files
    # If it's wrapped, unwrap it
    if hasattr(ckpt, 'state_dict'):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    torch.save(sd, out_path)
    print(f"Saved generator weights → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_demo():
    """Demonstrate the environment and print a benchmark."""
    import argparse, matplotlib
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description='SonoBot SonoGym Demo')
    parser.add_argument('--ct',    default=str(CT_PATH),    help='CT NIfTI path')
    parser.add_argument('--model', default=str(MODEL_PATH), help='CycleGAN weights path')
    parser.add_argument('--fp16',  action='store_true',     help='Use FP16 inference')
    parser.add_argument('--guide', action='store_true',     help='Print training guide')
    args = parser.parse_args()

    if args.guide:
        print_training_guide(); return

    # Build environment (neural if files present, physics otherwise)
    env = SonoSimEnv(ct_path=Path(args.ct), model_path=Path(args.model),
                     target='liver', use_fp16=args.fp16)

    # Benchmark
    print("\nRunning benchmark (20 calls) …")
    bm = env.benchmark(n_calls=20)
    print(f"  Mean: {bm['mean_ms']:.1f} ms/frame  "
          f"({bm['fps']:.1f} FPS)  ±{bm['std_ms']:.1f} ms  mode={bm['mode']}")

    # Generate a few demo images
    BODY = BODY_CENTER_M
    q0   = np.array([0., 1., 0., 0.])

    positions = [
        (BODY+np.array([-0.03, 0., 0.]),   "Liver"),
        (BODY+np.array([-0.01,-0.04, 0.]), "Gallbladder"),
        (BODY+np.array([ 0.00, 0., 0.]),   "Aorta/IVC"),
        (BODY+np.array([ 0.06, 0., 0.]),   "Bladder"),
    ]

    fig, axes = plt.subplots(1, len(positions), figsize=(14, 4))
    for ax, (pos, title) in zip(axes.flat, positions):
        obs = env.step(pos, q0)
        ax.imshow(obs['image'], cmap='gray', vmin=0, vmax=255, aspect='equal')
        ax.set_title(f"{title}\nQ={obs['quality']:.3f} [{obs['renderer']}]", fontsize=9)
        ax.axis('off')

    plt.suptitle(f"SonoSimEnv Demo — renderer: {env.mode}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = 'figures/sonogym_demo.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nDemo figure saved → {out}")

    # Summarise available components
    print("\n  Component Status:")
    print(f"    nibabel   : {'✓' if HAS_NIBABEL else '✗ (pip install nibabel)'}")
    print(f"    torch     : {'✓' if HAS_TORCH else '✗ (pip install torch)'}")
    print(f"    CT file   : {'✓' if Path(args.ct).exists() else '✗ (see --guide)'}")
    print(f"    GAN model : {'✓' if Path(args.model).exists() else '✗ (see --guide)'}")
    print(f"    Renderer  : {env.mode}")
    if env.mode != 'neural':
        print("\n  To enable neural rendering:")
        print("    python sonobot_sonogym.py --guide")


if __name__ == '__main__':
    run_demo()
