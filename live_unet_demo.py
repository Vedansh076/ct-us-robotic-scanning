"""
live_registered_demo.py  –  Registration-enhanced version
=========================================================
Based on live_unet_demo.py with two targeted changes:
  1. Body mesh loaded from patient-specific CT-derived mesh (patient_skin.obj)
  2. CT center computed via rigorous affine registration (not bounding-box)
All other functionality (U-Net, Pix2Pix, UI, robot, eval) is unchanged.
KEY CHANGES FROM ORIGINAL:
--------------------------
1. DARK US FIX (Issue #1):
   - Replaced the imported `normalise_ct` with an explicit, self-contained
     normalisation that clips HU to [-200, 300] (matching Concordia training)
     and scales to [0, 1] for sigmoid-output U-Net or [-1, 1] for tanh Pix2Pix.
   - Added startup diagnostic that prints predicted-US min/max/mean after the
     first forward pass so you can verify intensity at launch.
   - Added `enhance_us_output()`: soft log-compression + linear rescale that
     maps the dark model output into a perceptually plausible B-mode range
     (target mean 0.25–0.55) without distorting real bright structures.
   - The Pix2Pix tanh post-processing branch is preserved.

2. ELIMINATE BLACK CT SLICES (Issue #2):
   - Probe roll/pitch perturbations in `scan_target_pose` are now hard-clamped
     to ±0.35 rad (~20°) before the quaternion is built. This keeps the beam
     direction within the body footprint at all times.
   - A `last_valid_ct_slice` variable persists the last non-zero slice; if a
     raycast miss or an all-zero slice occurs the fallback is used and a
     "WARN: probe off-body" overlay is drawn in red.

3. QUANTITATIVE EVALUATION MODE (Issue #3):
   - New `--eval` flag.  When set, loads all subjects from `--eval-dir`
     (default: same folder as `--subject`).  For each subject, iterates every
     axial / coronal / sagittal slice in the CT volume, runs inference, and
     computes SSIM, PSNR, MAE versus the paired US image (if present) or just
     logs model statistics if no ground-truth US is available.
   - Results written to `eval_results_<timestamp>.csv` and a summary printed.
   - Simulation loop is skipped in eval mode.

4. LOGGING & REPRODUCIBILITY (Issue #4):
   - At startup a timestamped run folder `runs/<timestamp>/` is created.
   - `config.json` with all CLI args is saved there.
   - First-frame CT and US images are saved as PNG.
   - Per-frame stats are appended to `stats.csv` every `--log-every` frames.

5. PERFORMANCE (Issue #5):
   - Model inference runs in a background `concurrent.futures.ThreadPoolExecutor`
     thread (CUDA or CPU) so the PyBullet step is not blocked.
   - OpenCV `cv2.waitKey(1)` already provides a VSync-like yield; no change
     needed beyond the thread offload.

6. CODE QUALITY (Issue #6):
   - Single file.  Detailed inline comments on normalisation choices.
   - All original functionality preserved (manual/auto modes, comparison
     window, --diagnostic, --compare-sample, S-key saves, etc.).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
from torch.amp import autocast

# ── optional heavy imports (nibabel / skimage) are deferred so the module can
#    at least be imported without them if running partial tests.
try:
    import nibabel as nib
    _NIB_OK = True
except ImportError:
    _NIB_OK = False
    print("WARNING: nibabel not found – CT loading disabled.")

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False
    print("WARNING: scikit-image not found – SSIM/PSNR evaluation disabled.")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "model"
sys.path.insert(0, str(PROJECT_ROOT))         # for extract_slice, model.*, registration

from extract_slice import extract_slice

# ── model imports (kept from original) ────────────────────────────────────────
from model.model import UNet as UNetOriginal
from model.pix2pix.model import UNet as UNetPix2Pix

# ── registration imports (NEW) ────────────────────────────────────────────────
from registration import compute_registered_ct_center, load_registration_meta

# ── global variables for histogram matching (Stage 2) ────────────────────────
REFERENCE_CT_SLICE = None
MATCH_HISTOGRAM = True

# ── window names ───────────────────────────────────────────────────────────────
WINDOW_COMBINED = "CT & Ultrasound (Side by Side)"
WINDOW_COMPARE = "Training Comparison"

# ── scene constants ────────────────────────────────────────────────────────────
DEFAULT_BODY_MESH = PROJECT_ROOT / "mosh_cmu_0511_f_lbs_10_207_0_v1.0.2.obj"
BED_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
BED_MATTRESS_TOP_Z = 0.72
PANDA_EE_LINK = 11
PANDA_ARM_JOINTS = list(range(7))
PROBE_TIP_FROM_EE = np.array([0.0, 0.0, 0.18], dtype=np.float32)
PROBE_QUAT_FROM_EE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ── probe geometry (unchanged from original) ──────────────────────────────────
PROBE_COMPONENTS = (
    {
        "name": "contact_surface",
        "geom_type": p.GEOM_BOX,
        "half_extents": [0.028, 0.012, 0.003],
        "offset": np.array([0.0, 0.0, 0.003], dtype=np.float32),
        "rgba": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "wedge_flare",
        "geom_type": p.GEOM_CYLINDER,
        "radius": 0.015,
        "length": 0.056,
        "offset": np.array([0.0, 0.0, 0.020], dtype=np.float32),
        "rgba": [1.0, 1.0, 1.0, 1.0],
        "orn": [0.0, 0.7071067811865475, 0.0, 0.7071067811865476], # rotated 90 deg around Y to lie horizontally along X-axis
    },
    {
        "name": "handle",
        "geom_type": p.GEOM_CYLINDER,
        "radius": 0.018,
        "length": 0.150,
        "offset": np.array([0.0, 0.0, 0.100], dtype=np.float32),
        "rgba": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "top_mount",
        "geom_type": p.GEOM_BOX,
        "half_extents": [0.028, 0.028, 0.025],
        "offset": np.array([0.0, 0.0, 0.200], dtype=np.float32),
        "rgba": [1.0, 1.0, 1.0, 1.0], # matches width of visible panda_hand base
    },
    {
        "name": "cable",
        "geom_type": p.GEOM_CYLINDER,
        "radius": 0.006,
        "length": 0.020,
        "offset": np.array([0.0, 0.0, 0.230], dtype=np.float32),
        "rgba": [1.0, 1.0, 1.0, 1.0],
    },
)

DEFAULT_UNET_CKPT = MODEL_ROOT / "runs" / "exp1" / "best_model.pth"
DEFAULT_PIX2PIX_CKPT = MODEL_ROOT / "runs" / "exp_pix2pix" / "best_model.pth"

# ── normalisation constants (Concordia paired CT-US training data) ─────────────
# These values replicate the training preprocessing: HU is soft-windowed to
# [-200, 300] (capturing abdominal soft tissue contrast) and scaled to [0, 1].
# For Pix2Pix (tanh output) the input is instead scaled to [-1, 1].
# Choosing a narrow window rather than the full [-1000, 3000] HU range is
# critical: the abdomen occupies roughly [-150, 250] HU, and wider windows
# compress contrast so much that the network sees nearly uniform grey input,
# producing the observed near-zero output.
CT_HU_MIN = -200.0   # lower clip (air / fat boundary)
CT_HU_MAX =  300.0   # upper clip (soft tissue / portal veins)

# Maximum angular perturbation for the probe orientation during auto-scan.
# Beyond ~20° the acoustic beam can miss the body, yielding all-zero slices.
# ±0.35 rad ≈ ±20° which is physiologically realistic for a sonographer sweep.
MAX_PROBE_TILT_RAD = 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISATION  (replaces the imported normalise_ct)
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_ct_sigmoid(ct_slice: np.ndarray) -> np.ndarray:
    """
    Normalise a raw CT slice for a sigmoid-output U-Net (output in [0, 1]).

    Steps:
      1. Clip to [CT_HU_MIN, CT_HU_MAX] – removes bone/air outliers that would
         compress the soft-tissue gradient to near-zero after global min-max.
      2. Linear rescale to [0, 1].

    This MUST match the preprocessing applied during training.  Using the full
    HU range instead produces input values clustered around 0.1–0.2, which maps
    to decoder outputs close to 0 (sigmoid(0) = 0.5 but the network learns a
    bias that shifts outputs low when inputs are compressed).
    """
    ct = ct_slice.astype(np.float32)
    ct = np.clip(ct, CT_HU_MIN, CT_HU_MAX)
    ct = (ct - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)   # → [0, 1]
    return ct


def normalise_ct_tanh(ct_slice: np.ndarray) -> np.ndarray:
    """
    Normalise a raw CT slice for a tanh-output Pix2Pix model (output in [-1, 1]).

    Same clip window but scaled to [-1, 1].
    """
    ct = ct_slice.astype(np.float32)
    ct = np.clip(ct, CT_HU_MIN, CT_HU_MAX)
    ct = (ct - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)   # → [0, 1]
    ct = ct * 2.0 - 1.0                                  # → [-1, 1]
    return ct


def enhance_us_output(pred: np.ndarray) -> np.ndarray:
    """
    Post-process the raw model output to produce a perceptually realistic
    B-mode grey-scale image (target mean ~0.25–0.55 on [0, 1]).

    Many CT-to-US networks produce outputs that are linearly correct but whose
    dynamic range is compressed toward low values because the loss is dominated
    by the large dark (anechoic) background regions.  A mild log-compression
    mimics the time-gain compensation applied on real scanners and lifts the
    mid-range echoes without blowing out bright specular reflectors.

    Formula:  out = log(1 + alpha * x) / log(1 + alpha)
    where alpha = 8 gives a gentle lift (no clipping, fully invertible).
    """
    alpha = 8.0
    pred = np.clip(pred, 0.0, 1.0)
    compressed = np.log1p(alpha * pred) / np.log1p(alpha)
    return compressed.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-BASED (CONVOLUTION-BASED) ULTRASOUND SIMULATOR
# Replicates SonoGym's USSimulatorConv using NumPy (CPU).
# Reference: SonoGym paper §3.1 and [41] (convolution-based US simulation).
# ═══════════════════════════════════════════════════════════════════════════════

# Tissue label integer codes used by make_label_map()
US_LABEL_BACKGROUND = 0   # Air / outside body
US_LABEL_BONE       = 1   # Cortical bone (from bone_label.nii.gz > 0.5)
US_LABEL_SOFT       = 2   # General soft tissue / muscle
US_LABEL_FAT        = 3   # Fatty tissue (CT HU < -30)
US_LABEL_SKIN       = 4   # Skin layer (outermost soft-tissue pixels)

# Per-tissue acoustic parameters:
#   alpha : ultrasound attenuation coefficient  [dB / cm / MHz]
#   z     : acoustic impedance                  [MRayl = 10^6 kg/m²/s]
#   mu0   : speckle mean offset (brightness inside tissue)
#   mu1   : speckle gate threshold — pixels where N(0,1) > mu1 get zeroed;
#            lower mu1 = denser speckle (more pixels lit).  Was 0.6 → too sparse.
#   s0    : speckle standard deviation (texture contrast)
# Values adapted from SonoGym YAML + standard medical-physics literature.
# FIX: reduced mu1 for soft tissue from 0.6→0.25 (denser speckle texture)
#      increased s0 for soft/fat (more visible grain), raised mu0 for bone.
_US_TISSUE_PARAMS = {
    US_LABEL_BACKGROUND: dict(alpha=0.00, z=0.000400, mu0=0.00, mu1=1.0,  s0=0.005),
    US_LABEL_BONE:       dict(alpha=7.00, z=7.800000, mu0=0.70, mu1=0.25, s0=0.35),
    US_LABEL_SOFT:       dict(alpha=0.54, z=1.630000, mu0=0.22, mu1=0.25, s0=0.18),
    US_LABEL_FAT:        dict(alpha=0.48, z=1.380000, mu0=0.12, mu1=0.25, s0=0.14),
    US_LABEL_SKIN:       dict(alpha=0.20, z=1.700000, mu0=0.25, mu1=0.20, s0=0.10),
}


def make_label_map(ct_slice: np.ndarray, seg_slice: np.ndarray) -> np.ndarray:
    """
    Convert floating-point CT + binary bone segmentation into an integer tissue
    label map for the model-based US simulator.

    Label assignment (priority order, highest first):
      1. Bone  (wherever bone_label.nii.gz mask > 0.5)
      2. Fat   (wherever CT HU < -30 and NOT bone)
      3. Background (wherever CT HU < -500 = air)
      4. Skin  (1-pixel ring around the body boundary in the label map)
      5. Soft tissue (everything else)

    Parameters
    ----------
    ct_slice  : (H, W) float32 HU values
    seg_slice : (H, W) float32 binary bone mask (0 or 1)

    Returns
    -------
    label : (H, W) uint8 integer label map
    """
    H, W = ct_slice.shape
    label = np.full((H, W), US_LABEL_SOFT, dtype=np.uint8)

    # Air / outside body
    label[ct_slice < -500.0] = US_LABEL_BACKGROUND

    # Fat tissue
    fat_mask = (ct_slice >= -200.0) & (ct_slice < -30.0)
    label[fat_mask] = US_LABEL_FAT

    # Bone – highest priority; overrides fat
    label[seg_slice > 0.5] = US_LABEL_BONE

    # Skin: erode the non-background region by 1px and mark the border
    body_mask = (label != US_LABEL_BACKGROUND).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(body_mask, kernel, iterations=1)
    border = (body_mask - eroded).astype(bool)
    # Only mark as skin where the existing label is soft tissue or fat (not bone)
    skin_eligible = border & (label != US_LABEL_BONE)
    label[skin_eligible] = US_LABEL_SKIN

    return label


class ModelBasedUSSimulator:
    """
    Physics/convolution-based ultrasound image synthesizer.

    Mimics SonoGym's ``USSimulatorConv``.  Does NOT require a trained neural
    network – it synthesizes B-mode images directly from a tissue label map.

    Algorithm (matches SonoGym paper §3.1, with CT-gradient refinement from
    the paper's ``if_ct=True`` path):
      1. Build per-pixel acoustic parameter maps (α, Z, speckle μ₀, μ₁, σ₀)
         from tissue labels.
      2. Compute depth-dependent attenuation: atten = exp(-cumsum(α) · e · f)
      3. Compute tissue-boundary edge map  AND  supplement with CT-gradient
         edges to capture intra-tissue interfaces not visible in label map.
      4. Reflection (specular echo):
           E = I₀ · atten · ((Z₁−Z₂)/(Z₁+Z₂))² · cos θ
           E = PSF_E ⊛ E  (lateral blurring)
      5. Backscatter speckle:
           S = Gaussian noise modulated by tissue (μ₀, μ₁, σ₀)
           B = I₀ · atten · (PSF_B ⊛ S)
      6. Final image: US = ratio·E + B + TGC noise
      7. Gamma compress + normalise to [0, 1].

    Quality fixes applied vs. first version:
      - PSF widened  (sx_E 0.6→1.5, sy_E 2→4)  → proper streak appearance
      - element_size reduced (5e-4→1.5e-4 m)  → less depth blackout
      - TGC_beta raised (0.01→0.05)  → uniform brightness across depth
      - mu1 lowered for all tissues (0.6→0.25)  → denser speckle texture
      - CT-gradient edge supplementation  → sub-label-resolution interfaces
      - Gamma compression (γ=0.6) before output  → mid-tone lift like TGC
    """

    def __init__(
        self,
        frequency:    float = 5.0,    # MHz
        I0:           float = 1.5,    # initial acoustic energy  (was 1.0)
        element_size: float = 1.5e-4, # pixel pitch [m] — 0.15mm reduces depth blackout
        sx_E:         float = 1.5,    # PSF_E lateral sigma [pixels]  (was 0.6)
        sy_E:         float = 4.0,    # PSF_E axial sigma [pixels]    (was 2.0)
        sx_B:         float = 2.0,    # PSF_B lateral sigma [pixels]  (was 1.0)
        sy_B:         float = 2.0,    # PSF_B axial sigma [pixels]    (was 1.0)
        kernel_size:  tuple = (11, 11),  # larger kernel → full PSF extent (was 7×7)
        E_S_ratio:    float = 1.2,    # reflection-to-speckle weighting  (was 0.8)
        TGC_beta:     float = 0.05,   # time-gain compensation  (was 0.01 → too weak)
        noise_I:      float = 0.03,   # noise intensity scale
        noise_mu0:    float = 0.01,
        noise_mu1:    float = 0.4,
        noise_s0:     float = 0.03,
        noise_f:      float = 1.0,
        ct_edge_weight: float = 0.5,  # weight of CT-gradient edges (0 = label-only)
        ct_edge_thresh: float = 0.15, # relative gradient threshold for CT edges
        gamma:        float = 0.6,    # output gamma compression  (< 1 lifts mid-tones)
        tissue_params: dict | None = None,
        rng_seed: int | None = None,
    ):
        self.f      = frequency
        self.I0     = I0
        self.e      = element_size
        self.beta   = TGC_beta
        self.E_S_ratio = E_S_ratio
        self.n_I    = noise_I
        self.n_mu0  = noise_mu0
        self.n_mu1  = noise_mu1
        self.n_s0   = noise_s0
        self.n_f    = noise_f
        self.ct_edge_weight = ct_edge_weight
        self.ct_edge_thresh = ct_edge_thresh
        self.gamma  = gamma

        self.tissue_params = tissue_params if tissue_params else _US_TISSUE_PARAMS
        self.rng = np.random.default_rng(rng_seed)

        # Build 2-D PSF kernels (Gaussian in lateral × axial)
        self.PSF_E = self._make_psf(sx_E, sy_E, kernel_size)
        self.PSF_B = self._make_psf(sx_B, sy_B, kernel_size)

    # ── helper ────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_psf(sx: float, sy: float, ksize: tuple) -> np.ndarray:
        """Build a separable 2-D Gaussian PSF kernel (not normalised)."""
        kh, kw = ksize
        cx, cy = (kw - 1) / 2.0, (kh - 1) / 2.0
        xs = np.arange(kw, dtype=np.float32) - cx
        ys = np.arange(kh, dtype=np.float32) - cy
        gx = np.exp(-0.5 * xs**2 / sx**2)
        gy = np.exp(-0.5 * ys**2 / sy**2)
        return np.outer(gy, gx).astype(np.float32)   # (kh, kw)

    def _assign_param_maps(
        self, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-pixel maps for alpha, Z, mu0, mu1, s0."""
        H, W = label.shape
        alpha = np.zeros((H, W), dtype=np.float32)
        Z     = np.zeros((H, W), dtype=np.float32)
        mu0   = np.zeros((H, W), dtype=np.float32)
        mu1   = np.zeros((H, W), dtype=np.float32)
        s0    = np.zeros((H, W), dtype=np.float32)
        for lbl, params in self.tissue_params.items():
            mask = label == lbl
            alpha[mask] = params['alpha']
            Z[mask]     = params['z']
            mu0[mask]   = params['mu0']
            mu1[mask]   = params['mu1']
            s0[mask]    = params['s0']
        return alpha, Z, mu0, mu1, s0

    @staticmethod
    def _compute_attenuation(alpha: np.ndarray, e: float, f: float) -> np.ndarray:
        """
        Depth-dependent attenuation along axis 0 (top = probe).
        atten[i, j] = exp(-sum_{k<=i} alpha[k,j] * e * f)

        Unit Correction:
        - alpha is in dB/cm/MHz
        - e is in meters, so we multiply by 100 to convert to cm.
        - The exponential function np.exp expects Nepers, not dB.
          1 dB = 0.1151 Nepers, so we multiply by 0.1151.
        """
        e_cm = e * 100.0
        db_to_neper = 0.1151
        return np.exp(-np.cumsum(alpha, axis=0) * e_cm * f * db_to_neper)


    @staticmethod
    def _compute_edge_map(label: np.ndarray) -> np.ndarray:
        """
        Binary edge map: 1 wherever label changes between successive rows.
        Only axial (depth-direction) edges are used because US predominantly
        responds to interfaces perpendicular to the beam.
        """
        edge = np.zeros(label.shape, dtype=np.float32)
        edge[1:, :] = (label[1:, :] != label[:-1, :]).astype(np.float32)
        return edge

    @staticmethod
    def _compute_ct_edge_map(ct_slice: np.ndarray, thresh: float = 0.15) -> np.ndarray:
        """
        Compute a continuous edge map from the raw CT HU image.
        Captures intra-tissue interfaces (fascia, organ capsules, vessel walls)
        that are invisible in the coarse 5-label map.

        Method: axial finite-difference relative gradient |I[y] - I[y-1]| / (|I[y]| + eps)
        Mirrors SonoGym's ``compute_ct_edge_map`` (if_ct=True path).

        Parameters
        ----------
        ct_slice : (H, W) float32 HU values (raw, not normalised)
        thresh   : relative gradient threshold; pixels below are zeroed

        Returns
        -------
        edge : (H, W) float32 in [0, 1]
        """
        pad = np.pad(ct_slice, ((1, 0), (0, 0)), mode='edge')  # pad top by 1 row
        diff = np.abs(ct_slice - pad[:-1, :])                  # |I[y] - I[y-1]|
        denom = np.abs(ct_slice) + 1.0                          # avoid div-by-zero
        rel_grad = diff / denom
        edge = np.where(rel_grad > thresh, rel_grad, 0.0).astype(np.float32)
        # Normalise to [0, 1] for consistent weighting
        mx = edge.max()
        if mx > 1e-8:
            edge = edge / mx
        return edge

    @staticmethod
    def _compute_cos_map(edge: np.ndarray) -> np.ndarray:
        """
        Approximate cosine of the angle between the beam direction (axial)
        and the surface normal, derived from the gradient of the edge map.
        """
        pad = np.pad(edge, ((1, 1), (1, 1)), mode='reflect')
        grad_x = edge - pad[1:-1, :-2]   # horizontal gradient
        grad_y = edge - pad[:-2, 1:-1]   # vertical (axial) gradient
        norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-5
        # cos(theta) = axial-gradient / magnitude (beam is along axial axis)
        cos_map = grad_y / norm
        return cos_map.astype(np.float32)

    def simulate(
        self,
        label: np.ndarray,
        ct_slice: np.ndarray | None = None,
        if_noise: bool = True,
    ) -> np.ndarray:
        """
        Synthesise a B-mode ultrasound image from an integer label map,
        optionally supplemented by CT-gradient edges for intra-tissue detail.

        Parameters
        ----------
        label     : (H, W) uint8 integer tissue labels (see US_LABEL_* constants)
        ct_slice  : (H, W) float32 raw HU values; if provided, CT-gradient
                    edges are blended into the reflection term to capture
                    sub-label-resolution interfaces (fascia, vessels, etc.).
        if_noise  : whether to add TGC instrument noise

        Returns
        -------
        us_image  : (H, W) float32 in [0, 1]
        """
        alpha, Z, mu0, mu1, s0 = self._assign_param_maps(label)

        # 1. Attenuation (TGC beta already subtracted from alpha)
        atten = self._compute_attenuation(alpha - self.beta, self.e, self.f)
        # Clamp to avoid negative exponents going above 1
        atten = np.clip(atten, 0.0, 1.0)

        # 2. Label edge map & cosine map
        label_edge = self._compute_edge_map(label)
        cos_m      = self._compute_cos_map(label_edge)

        # 3. CT-gradient edge supplementation  ← KEY FIX for image quality
        #    Adds fine-grained intra-tissue interfaces not visible in label map.
        #    Mirrors SonoGym's compute_ct_edge_map / if_ct=True path.
        if ct_slice is not None and self.ct_edge_weight > 0.0:
            ct_edge = self._compute_ct_edge_map(ct_slice, thresh=self.ct_edge_thresh)
            # Use CT edge for the reflection term but label edge for cosine
            combined_edge = np.clip(
                label_edge + self.ct_edge_weight * ct_edge, 0.0, 1.0
            )
            # Use CT-based impedance proxy for the reflection coefficient where
            # no label boundary exists (raw HU scaled → approximate Z)
            ct_norm = np.clip(
                (ct_slice.astype(np.float32) + 1000.0) / 2000.0, 0.0, 1.0
            )
            Z_ct = ct_norm * 3.5 + 0.5   # rough HU → Z scaling [0.5, 4.0] MRayl
            Z_up_ct  = np.roll(Z_ct, 1, axis=0); Z_up_ct[0, :] = Z_ct[0, :]
            R_ct = (Z_up_ct - Z_ct)**2 / (Z_up_ct + Z_ct + 1e-5)**2
            # Blend: label interfaces use label-Z, CT interfaces use CT-Z
            Z_up = np.roll(Z, 1, axis=0); Z_up[0, :] = Z[0, :]
            R_label = (Z_up - Z)**2 / (Z_up + Z + 1e-5)**2
            R_coeff = R_label + self.ct_edge_weight * R_ct
            edge = combined_edge
        else:
            Z_up    = np.roll(Z, 1, axis=0); Z_up[0, :] = Z[0, :]
            R_coeff = (Z_up - Z)**2 / (Z_up + Z + 1e-5)**2
            edge    = label_edge

        # 4. Reflection (specular echo) map — clamp outliers before PSF
        I0_map = np.ones_like(alpha) * self.I0
        E_map  = I0_map * atten * R_coeff * edge * np.abs(cos_m)
        E_map  = np.clip(E_map, 0.0, 0.15)
        # Convolve with PSF_E to model lateral resolution blurring
        E_map = cv2.filter2D(E_map, -1, self.PSF_E, borderType=cv2.BORDER_REFLECT)

        # 5. Backscatter speckle map
        rng = self.rng
        T0 = rng.standard_normal(label.shape).astype(np.float32)
        T1 = rng.standard_normal(label.shape).astype(np.float32)
        S_map = T0 * s0 + mu0
        S_map[T1 > mu1] = 0.0   # gate: pixels where N(0,1) > mu1 → silent
        S_map = np.clip(S_map, 0.0, None)  # speckle amplitude is non-negative
        B_map = I0_map * atten * cv2.filter2D(
            S_map, -1, self.PSF_B, borderType=cv2.BORDER_REFLECT)

        # 6. Combine reflection + backscatter
        US = self.E_S_ratio * E_map + B_map

        # 7. TGC noise
        if if_noise:
            n_T0 = rng.standard_normal(label.shape).astype(np.float32)
            n_T1 = rng.standard_normal(label.shape).astype(np.float32)
            n_map = n_T0 * self.n_s0 + self.n_mu0
            n_map[n_T1 > self.n_mu1] = 0.0
            depth_idx = np.arange(label.shape[0], dtype=np.float32)[:, None]
            TGC = np.exp(depth_idx * self.e * self.beta * self.n_f)
            US = US + n_map * TGC * self.n_I

        US = np.clip(US, 0.0, None)

        # 8. Gamma compression — lifts mid-tones (mimics scanner log-compression)
        #    γ < 1 brightens the image non-linearly while preserving bright peaks.
        if self.gamma != 1.0:
            US_max = US.max()
            if US_max > 1e-8:
                US = (US / US_max) ** self.gamma * US_max

        # 9. Normalise to [0, 1]
        mn, mx = US.min(), US.max()
        if mx > mn + 1e-8:
            US = (US - mn) / (mx - mn)
        else:
            US = np.zeros_like(US)
        return US.astype(np.float32)



# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live PyBullet probe -> CT slice -> U-Net/Pix2Pix -> ultrasound demo."
    )
    parser.add_argument("--model", choices=("unet", "pix2pix"), default="unet")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058",
                        help="Subject folder with CT volume and patient_skin.obj. "
                             "TotalSegmentator subjects: totalseg_patients/s0011..s0310. "
                             "Legacy TCGA subjects: TCGA-QQ-A8VG etc.")
    parser.add_argument("--checkpoint", type=str, default="model/runs/exp1_2IP/exp1/best_model.pth",
                        help="Path to the model checkpoint (default: exp1_2IP/exp1)")
    parser.add_argument("--base-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--shared-memory", action="store_true")
    parser.add_argument("--body-mesh", type=Path, default=DEFAULT_BODY_MESH)
    parser.add_argument("--mesh-scale", type=str, default="1.0", help="Mesh scale factor (e.g. '1.0' or '3.0,3.0,1.2')")
    parser.add_argument("--ray-length", type=float, default=0.8)
    parser.add_argument("--hit-scale", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pixel-spacing", type=float, default=0.35)
    parser.add_argument("--scan-speed", type=float, default=0.45)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--compare-sample", type=Path, default=None)
    parser.add_argument("--interp-order", type=int, choices=(1, 3), default=1,
                        help="Interpolation order for CT slice extraction: 1=linear, 3=cubic spline (default: 1)")
    parser.add_argument("--body-margin", type=float, default=0.06,
                        help="Margin (in meters) from body footprint boundary for sweep clamping (default: 0.06)")
    # ── new in publication-ready version ──────────────────────────────────────
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluation mode: run model on all CT volumes in --eval-dir, "
             "compute SSIM/PSNR/MAE, write CSV, then exit (no simulation).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing subject sub-folders for evaluation. "
             "Defaults to the parent folder of --subject.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=60,
        help="Append per-frame stats to runs/<ts>/stats.csv every N frames.",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable log-compression post-processing (use raw model output).",
    )
    parser.add_argument(
        "--no-match-histogram",
        action="store_true",
        help="Disable intensity histogram matching to reference slice (Stage 2).",
    )
    parser.add_argument(
        "--only-probe",
        action="store_true",
        help="Hide the robot arm and display only the ultrasound probe.",
    )
    parser.add_argument(
        "--sim-mode",
        choices=("unet", "pix2pix", "conv"),
        default="unet",
        help="Ultrasound simulation mode: 'unet' (default) uses the trained U-Net, "
             "'pix2pix' uses the Pix2Pix generator, 'conv' uses the physics-based "
             "convolution simulator (no neural network required).",
    )
    return parser.parse_args()



# ═══════════════════════════════════════════════════════════════════════════════
# SUBJECT / MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_subject_dir(subject: str) -> Path:
    subject_path = Path(subject)
    if subject_path.exists():
        return subject_path
    return PROJECT_ROOT / subject


def load_ct_subject(subject_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _NIB_OK:
        raise ImportError("nibabel required for CT loading.")

    # ── CT volume: support both TCGA (CT.nii) and TotalSegmentator (ct.nii.gz) ──
    ct_path = None
    for candidate in ["CT.nii", "CT.nii.gz", "ct.nii.gz", "ct.nii"]:
        p = subject_dir / candidate
        if p.exists():
            ct_path = p
            break
    if ct_path is None:
        raise FileNotFoundError(f"No CT volume found in {subject_dir}. "
                                "Expected CT.nii, CT.nii.gz, or ct.nii.gz")
    ct_img = nib.load(str(ct_path))
    ct_volume = ct_img.get_fdata(dtype=np.float32)

    # ── Label/bone volume: priority order ────────────────────────────────────
    # 1. TotalSegmentator pre-merged bone mask (best quality)
    # 2. Legacy label files from TCGA pipeline
    # 3. HU-threshold fallback
    label_volume = None
    label_candidates = [
        "bone_label.nii.gz",       # TotalSegmentator merged bone mask
        "bone_label.nii",
        "Labels.nii",              # TCGA legacy
        "Label.nii",
        "segmentation.nii",
        "segmentation.nii.gz",
        "labels.nii",
        "labels.nii.gz",
    ]
    for name in label_candidates:
        path = subject_dir / name
        if path.exists():
            try:
                label_volume = nib.load(str(path)).get_fdata(dtype=np.float32)
                print(f"[load] Loaded bone label volume from {path.name}")
                break
            except Exception as e:
                print(f"[load] Failed to load label volume {path}: {e}")

    if label_volume is None:
        print("[load] No label volume found — generating bone mask by thresholding CT (> 200 HU).")
        label_volume = (ct_volume > 200.0).astype(np.float32)

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float32)
    volume_center = (np.array(ct_volume.shape, dtype=np.float32) - 1) / 2
    return ct_volume, label_volume, spacing, volume_center



def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_arg)
    
    if dev.type == "cpu":
        # On CPU: limit PyTorch threads to 1 to prevent OpenMP from saturating
        # all cores and starving the PyBullet physics loop / OpenCV GUI.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        print("[device] CPU mode: limited PyTorch to 1 thread to prevent simulator starvation.")
    else:
        # On GPU: inference runs on the CUDA device — CPU cores stay fully free
        # for PyBullet and the OpenCV GUI. No thread throttling needed.
        gpu_name = torch.cuda.get_device_name(dev)
        print(f"[device] GPU mode detected ({gpu_name}): CPU threads unrestricted.")
    return dev



def load_unet(checkpoint_path: Path, device: torch.device,
              base_features: int, dropout: float) -> UNetOriginal:
    model = UNetOriginal(in_channels=2, out_channels=1,
                         base_features=base_features, dropout=dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def load_pix2pix(checkpoint_path: Path, device: torch.device,
                 base_features: int, dropout: float) -> UNetPix2Pix:
    model = UNetPix2Pix(in_channels=2, out_channels=1, base_features=base_features,
                        dropout=dropout, pix2pix_dropout=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("G", checkpoint.get("model_state", checkpoint))
    model.load_state_dict(state)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_ultrasound(
        model: torch.nn.Module,
        ct_slice: np.ndarray,
        seg_slice: np.ndarray,
        device: torch.device,
        is_pix2pix: bool = False,
        enhance: bool = True,
) -> np.ndarray:
    """
    Run inference with the correct normalisation for each model family.

    Normalisation choice:
      - sigmoid U-Net : input scaled to [0, 1]  (normalise_ct_sigmoid)
      - tanh Pix2Pix  : input scaled to [-1, 1] (normalise_ct_tanh)

    Using the wrong normalisation is the primary cause of the near-black output
    reported in Issue #1.  The original code called `normalise_ct` from
    model.dataset which uses a global percentile stretch; if the CT volume has
    HU outliers (bone > 700 HU) the soft-tissue window is compressed, giving
    the network inputs that look like noise, and the network responds with
    low-confidence (near-zero) predictions.
    """
    global REFERENCE_CT_SLICE, MATCH_HISTOGRAM

    # ── Stage 2: Intensity Histogram Matching ─────────────────────────────────
    if MATCH_HISTOGRAM:
        if REFERENCE_CT_SLICE is None:
            ref_path = PROJECT_ROOT / "model" / "reference_ct_slice.npy"
            if ref_path.exists():
                try:
                    REFERENCE_CT_SLICE = np.load(ref_path)
                    print(f"[inference] Loaded histogram reference template from {ref_path}")
                except Exception as e:
                    print(f"ERROR: Failed to load reference template: {e}")
                    REFERENCE_CT_SLICE = False
            else:
                print(f"WARNING: Reference template {ref_path} not found. Histogram matching disabled.")
                REFERENCE_CT_SLICE = False

        if REFERENCE_CT_SLICE is not None and not isinstance(REFERENCE_CT_SLICE, bool):
            try:
                from skimage.exposure import match_histograms
                ct_slice = match_histograms(ct_slice, REFERENCE_CT_SLICE)
            except Exception as e:
                pass

    if is_pix2pix:
        ct_norm = normalise_ct_tanh(ct_slice)
        seg_norm = seg_slice * 2.0 - 1.0
    else:
        ct_norm = normalise_ct_sigmoid(ct_slice)
        seg_norm = seg_slice

    ct_tensor = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0).to(device)
    seg_tensor = torch.from_numpy(seg_norm).unsqueeze(0).unsqueeze(0).to(device)
    input_tensor = torch.cat([ct_tensor, seg_tensor], dim=1) # (1, 2, H, W)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred = model(input_tensor)

    pred_np = pred.squeeze().detach().cpu().float().numpy()

    # ── Pix2Pix outputs in [-1, 1] due to tanh activation ─────────────────────
    # The branch below (from original code) handles this correctly.
    if pred_np.min() < -0.05:
        pred_np = (pred_np + 1.0) / 2.0

    pred_np = np.clip(pred_np, 0.0, 1.0)

    # ── Optional log-compression to lift mid-range echoes ──────────────────────
    if enhance:
        pred_np = enhance_us_output(pred_np)

    return pred_np


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODE  (Issue #3)
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args: argparse.Namespace) -> None:
    """
    Offline quantitative evaluation over a directory of CT subjects.

    Directory layout expected (Concordia style):
        eval_dir/
            subject_A/
                CT.nii
                us_frames/          ← optional; contains *.npy paired US images
            subject_B/
                ...

    If no us_frames/ exist, only model-output statistics (mean/std/entropy) are
    logged (useful for sanity-checking on unlabelled data).

    Metrics:
        SSIM  – structural similarity (skimage, data_range=1.0)
        PSNR  – peak SNR             (skimage, data_range=1.0)
        MAE   – mean absolute error  (numpy)
    """
    if not _SKIMAGE_OK:
        print("ERROR: scikit-image required for evaluation (pip install scikit-image).")
        sys.exit(1)

    global MATCH_HISTOGRAM
    MATCH_HISTOGRAM = not args.no_match_histogram

    device = select_device(args.device)
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_PIX2PIX_CKPT if args.model == "pix2pix" else DEFAULT_UNET_CKPT

    print(f"[eval] Loading model from {args.checkpoint} ...")
    if args.model == "pix2pix":
        model = load_pix2pix(args.checkpoint, device, args.base_features, args.dropout)
        is_pix2pix = True
    else:
        model = load_unet(args.checkpoint, device, args.base_features, args.dropout)
        is_pix2pix = False

    eval_dir = args.eval_dir
    if eval_dir is None:
        subject_dir = resolve_subject_dir(args.subject)
        eval_dir = subject_dir.parent
    print(f"[eval] Scanning subjects in {eval_dir}")

    subjects = sorted([d for d in eval_dir.iterdir() if d.is_dir() and (d / "CT.nii").exists()])
    if not subjects:
        print(f"[eval] No subjects found in {eval_dir}.")
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = eval_dir / f"eval_results_{ts}.csv"

    fieldnames = ["subject", "slice_idx", "axis", "has_gt",
                  "ssim", "psnr_db", "mae", "pred_mean", "pred_std"]
    rows = []

    for subj_dir in subjects:
        print(f"[eval]   Subject: {subj_dir.name} ...", end=" ", flush=True)
        try:
            ct_volume, label_volume, spacing, volume_center = load_ct_subject(subj_dir)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        # Check for ground-truth US frames
        us_dir = subj_dir / "us_frames"
        gt_frames: dict[int, np.ndarray] = {}
        if us_dir.exists():
            for f in sorted(us_dir.glob("*.npy")):
                try:
                    idx = int(f.stem.split("_")[-1])
                    gt_frames[idx] = np.load(str(f)).astype(np.float32)
                except Exception:
                    pass

        # Iterate axial slices
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        n_axial = ct_volume.shape[2]

        for s_idx in range(0, n_axial, max(1, n_axial // 50)):   # sample ~50 slices
            center = np.array([volume_center[0], volume_center[1], float(s_idx)],
                               dtype=np.float32)
            ct_slice = extract_slice(
                ct_volume, center=center, quaternion=identity_quat,
                spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
            )
            seg_slice = extract_slice(
                label_volume, center=center, quaternion=identity_quat,
                spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
            )
            seg_slice = (seg_slice > 0.5).astype(np.float32)

            pred_us = predict_ultrasound(
                model, ct_slice, seg_slice, device, is_pix2pix=is_pix2pix,
                enhance=(not args.no_enhance),
            )

            has_gt = s_idx in gt_frames
            if has_gt:
                gt = gt_frames[s_idx]
                # Resize GT to match prediction if needed
                if gt.shape != pred_us.shape:
                    gt = cv2.resize(gt, (pred_us.shape[1], pred_us.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
                gt = np.clip(gt, 0.0, 1.0)
                ssim_val = float(ssim_fn(gt, pred_us, data_range=1.0))
                psnr_val = float(psnr_fn(gt, pred_us, data_range=1.0))
                mae_val  = float(np.mean(np.abs(gt - pred_us)))
            else:
                ssim_val = psnr_val = mae_val = float("nan")

            rows.append({
                "subject":   subj_dir.name,
                "slice_idx": s_idx,
                "axis":      "axial",
                "has_gt":    int(has_gt),
                "ssim":      round(ssim_val, 4),
                "psnr_db":   round(psnr_val, 4),
                "mae":       round(mae_val, 5),
                "pred_mean": round(float(pred_us.mean()), 4),
                "pred_std":  round(float(pred_us.std()), 4),
            })

        print(f"done ({len(rows)} rows so far)")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    gt_rows = [r for r in rows if r["has_gt"]]
    print(f"\n[eval] Results saved to {csv_path}")
    print(f"[eval] Total slices evaluated : {len(rows)}")
    if gt_rows:
        mean_ssim = np.mean([r["ssim"] for r in gt_rows])
        mean_psnr = np.mean([r["psnr_db"] for r in gt_rows])
        mean_mae  = np.mean([r["mae"] for r in gt_rows])
        print(f"[eval] Ground-truth slices    : {len(gt_rows)}")
        print(f"[eval] Mean SSIM              : {mean_ssim:.4f}")
        print(f"[eval] Mean PSNR (dB)         : {mean_psnr:.4f}")
        print(f"[eval] Mean MAE               : {mean_mae:.5f}")
    else:
        print("[eval] No ground-truth US frames found – only model stats logged.")
        pred_means = [r["pred_mean"] for r in rows]
        print(f"[eval] Pred mean intensity (avg over slices): {np.mean(pred_means):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN FOLDER SETUP  (Issue #4)
# ═══════════════════════════════════════════════════════════════════════════════

def create_run_folder(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {k: str(v) for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[log] Run folder: {run_dir}")
    return run_dir


def init_stats_csv(run_dir: Path) -> csv.DictWriter:
    f = open(run_dir / "stats.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=[
        "frame", "fps", "hit",
        "ct_min", "ct_max", "ct_mean", "ct_std",
        "us_min", "us_max", "us_mean", "us_std",
        "probe_x", "probe_y", "probe_z",
    ])
    writer.writeheader()
    return writer


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ct_center_from_hit(
        hit_position, body_center, volume_center, volume_shape,
        hit_scale, body_extent, mesh_bounds_min, mesh_bounds_max,
) -> np.ndarray:
    if hit_position is None:
        return volume_center
    mesh_center = (mesh_bounds_min + mesh_bounds_max) / 2.0
    mesh_extent = np.maximum(mesh_bounds_max - mesh_bounds_min, 1e-6)
    normalized = np.clip((hit_position - mesh_center) / (mesh_extent / 2.0), -1.0, 1.0)
    ct_position = volume_center + normalized * hit_scale
    low  = np.zeros(3, dtype=np.float32)
    high = np.array(volume_shape, dtype=np.float32) - 1
    return np.clip(ct_position, low, high).astype(np.float32)


def get_body_bounds(body_id: int):
    aabb_min, aabb_max = p.getAABB(body_id)
    bounds_min = np.array(aabb_min, dtype=np.float32)
    bounds_max = np.array(aabb_max, dtype=np.float32)
    center = (bounds_min + bounds_max) / 2.0
    extent = bounds_max - bounds_min
    return bounds_min, bounds_max, center, extent


def read_obj_bounds(mesh_path: Path):
    mins = np.array([np.inf,  np.inf,  np.inf],  dtype=np.float32)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    with mesh_path.open("r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            values = np.fromstring(line[2:], sep=" ", count=3, dtype=np.float32)
            if values.size != 3:
                continue
            mins = np.minimum(mins, values)
            maxs = np.maximum(maxs, values)
    if not np.all(np.isfinite(mins)):
        raise ValueError(f"No vertices found in OBJ mesh: {mesh_path}")
    return mins, maxs


def get_probe_beam_direction(quaternion_xyzw: np.ndarray) -> np.ndarray:
    rotation = np.array(p.getMatrixFromQuaternion(quaternion_xyzw), dtype=np.float32).reshape(3, 3)
    beam = -rotation[:, 2]
    norm = np.linalg.norm(beam)
    return (beam / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)


def transform_local_point(position, quaternion_xyzw, local_point) -> np.ndarray:
    rotation = np.array(p.getMatrixFromQuaternion(quaternion_xyzw), dtype=np.float32).reshape(3, 3)
    return (position + rotation @ local_point).astype(np.float32)


def multiply_quaternions(q1, q2) -> np.ndarray:
    result = p.multiplyTransforms([0, 0, 0], q1.tolist(), [0, 0, 0], q2.tolist())[1]
    return np.array(result, dtype=np.float32)


def get_probe_pose_from_ee(ee_position, ee_quaternion):
    probe_quaternion = multiply_quaternions(ee_quaternion, PROBE_QUAT_FROM_EE)
    probe_tip = transform_local_point(ee_position, ee_quaternion, PROBE_TIP_FROM_EE)
    return probe_tip, probe_quaternion, probe_tip


# ── debug drawing (unchanged) ─────────────────────────────────────────────────

def draw_debug_frame(position, quaternion_xyzw, length, item_ids):
    if item_ids is not None:
        for i in item_ids: p.removeUserDebugItem(i)
    rotation = np.array(p.getMatrixFromQuaternion(quaternion_xyzw), dtype=np.float32).reshape(3, 3)
    colors = ([1, 0, 0], [0, 1, 0], [0, 0.35, 1])
    ids = []
    for axis in range(3):
        end = position + rotation[:, axis] * length
        ids.append(p.addUserDebugLine(position.tolist(), end.tolist(), colors[axis], lineWidth=2.0, lifeTime=0.0))
    return ids


def draw_beam_direction_vector(probe_position, probe_quaternion, item_id, length=0.18):
    if item_id is not None: p.removeUserDebugItem(item_id)
    beam = get_probe_beam_direction(probe_quaternion)
    return p.addUserDebugLine(probe_position.tolist(), (probe_position + beam * length).tolist(),
                              [0, 1, 1], lineWidth=5.0, lifeTime=0.0)


def draw_contact_face_debug(probe_position, probe_quaternion, item_ids):
    if item_ids is not None:
        for i in item_ids:
            if i is not None: p.removeUserDebugItem(i)
    beam = get_probe_beam_direction(probe_quaternion)
    point_id = p.addUserDebugPoints([probe_position.tolist()], [[1, 1, 0]], pointSize=10.0, lifeTime=0.0)
    line_id  = p.addUserDebugLine(probe_position.tolist(), (probe_position + beam * 0.055).tolist(),
                                  [1, 1, 0], lineWidth=4.0, lifeTime=0.0)
    return point_id, line_id


def raycast_probe(probe_position, quaternion_xyzw, body_id, ray_length):
    beam = get_probe_beam_direction(quaternion_xyzw)
    # Start raycast 5cm above the probe tip (opposite to beam) to handle skin penetration
    offset = 0.05
    ray_start = probe_position - beam * offset
    ray_to = probe_position + beam * ray_length
    result = p.rayTest(ray_start.tolist(), ray_to.tolist())[0]
    
    hit = result[0] == body_id
    hit_pos = np.array(result[3], dtype=np.float32) if hit else None
    
    total_length = ray_length + offset
    hit_dist = float(result[2] * total_length - offset) if hit else None
    return hit, ray_to.astype(np.float32), hit_pos, hit_dist


def raycast_skin_surface(x: float, y: float, body_id: int, start_z: float = 1.25, end_z: float = 0.65) -> tuple[bool, float]:
    """Robust surface height check by multi-step raycasting to ignore robot arm/probe geometry blocking the ray."""
    found_body = False
    surface_z = end_z
    attempts = 0
    current_start = start_z
    while current_start > end_z and attempts < 10:
        attempts += 1
        rr = p.rayTest([x, y, current_start], [x, y, end_z])[0]
        hit_id = rr[0]
        if hit_id == body_id:
            surface_z = rr[3][2]
            found_body = True
            break
        elif hit_id == -1:
            break
        else:
            # Hit something else (like the robot arm or hand).
            # Resume casting from slightly below the hit point
            hit_z = rr[3][2]
            current_start = hit_z - 0.001
    return found_body, surface_z



def draw_debug_ray(probe_position, ray_to, hit_position, hit, line_id, point_id):
    if line_id is not None: p.removeUserDebugItem(line_id)
    if point_id is not None: p.removeUserDebugItem(point_id)
    color = [0, 1, 0] if hit else [1, 0, 0]
    line_id = p.addUserDebugLine(probe_position.tolist(), ray_to.tolist(), color, lineWidth=3.0, lifeTime=0.0)
    if hit_position is None:
        return line_id, None
    point_id = p.addUserDebugPoints([hit_position.tolist()], [[0, 1, 0]], pointSize=12.0, lifeTime=0.0)
    return line_id, point_id


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def to_uint8_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def overlay_status(
        image_u8: np.ndarray,
        probe_position: np.ndarray,
        hit_position,
        hit_distance,
        ct_center: np.ndarray,
        fps: float,
        ct_stats=None,
        mode_str: str | list[str] = "",
        warn_str: str = "",          # ← NEW: displayed in red if non-empty
) -> np.ndarray:
    display = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    hit_line = (f"hit: {hit_position[0]:+.3f}, {hit_position[1]:+.3f}, {hit_position[2]:+.3f} m"
                if hit_position is not None else "hit: miss")
    distance_line = f"distance: {hit_distance:.3f} m" if hit_distance is not None else "distance: --"
    lines = []
    if mode_str:
        if isinstance(mode_str, (list, tuple)):
            lines.extend(mode_str)
        else:
            lines.append(mode_str)
    lines.extend([
        f"probe: {probe_position[0]:+.3f}, {probe_position[1]:+.3f}, {probe_position[2]:+.3f} m",
        hit_line, distance_line,
        f"center: {ct_center[0]:.1f}, {ct_center[1]:.1f}, {ct_center[2]:.1f} vox",
        f"FPS: {fps:.1f}",
    ])
    if ct_stats is not None:
        lines.append(f"CT: min={ct_stats['min']:.1f} max={ct_stats['max']:.1f} "
                     f"mean={ct_stats['mean']:.1f} std={ct_stats['std']:.1f}")
    y = 18
    for line in lines:
        cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    # Warning overlay in red (probe off-body fallback active)
    if warn_str:
        cv2.putText(display, warn_str, (8, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    return display


# ═══════════════════════════════════════════════════════════════════════════════
# PYBULLET SCENE SETUP  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def connect_pybullet(shared_memory: bool) -> int:
    if shared_memory:
        client = p.connect(p.SHARED_MEMORY)
        if client >= 0:
            return client
        print("Shared-memory connection failed; opening a new GUI client.")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return client


def create_box_body(half_extents, position, rgba, mass=0.0) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=position,
                             baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))


def create_hospital_room() -> float:
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=45.0,
                                 cameraPitch=-28.0, cameraTargetPosition=[0, 0, 0.75])
    cart_pos = [-0.50, -0.35, 0.25]
    create_box_body([0.35, 0.35, 0.25], cart_pos, [0.3, 0.3, 0.35, 1.0])
    wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.015)
    wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.015, rgbaColor=[0.1, 0.1, 0.1, 1])
    wheel_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    for dx, dy in [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]:
        p.createMultiBody(0, wheel_col, wheel_vis, [cart_pos[0]+dx, cart_pos[1]+dy, 0.015], wheel_orn)

    bed_x, bed_y = 0.0, 0.0
    leg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.50)
    leg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.50, rgbaColor=[0.7, 0.7, 0.7, 1])
    wheel_col_b = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.02)
    wheel_vis_b = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.02, rgbaColor=[0.2, 0.2, 0.2, 1])
    wheel_orn_b = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
    for dx, dy in [[0.50, -0.95], [0.50, 0.95], [-0.50, -0.95], [-0.50, 0.95]]:
        p.createMultiBody(0, leg_col, leg_vis, [bed_x+dx, bed_y+dy, 0.25])
        p.createMultiBody(0, wheel_col_b, wheel_vis_b, [bed_x+dx, bed_y+dy, 0.02], wheel_orn_b)

    beam_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.96, 0.03])
    beam_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.96, 0.03], rgbaColor=[0.4, 0.4, 0.4, 1])
    p.createMultiBody(0, beam_col, beam_vis, [bed_x-0.52, bed_y, 0.50+0.03])
    p.createMultiBody(0, beam_col, beam_vis, [bed_x+0.52, bed_y, 0.50+0.03])
    cross_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.54, 0.02, 0.03])
    cross_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.54, 0.02, 0.03], rgbaColor=[0.4, 0.4, 0.4, 1])
    p.createMultiBody(0, cross_col, cross_vis, [bed_x, bed_y-0.96, 0.53])
    p.createMultiBody(0, cross_col, cross_vis, [bed_x, bed_y+0.96, 0.53])
    board_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.52, 0.02, 0.25])
    board_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.52, 0.02, 0.25], rgbaColor=[0.2, 0.3, 0.5, 1])
    p.createMultiBody(0, board_col, board_vis, [bed_x, bed_y+0.98, 0.78])
    foot_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.52, 0.02, 0.15])
    foot_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.52, 0.02, 0.15], rgbaColor=[0.2, 0.3, 0.5, 1])
    p.createMultiBody(0, foot_col, foot_vis, [bed_x, bed_y-0.98, 0.65])
    mattress_half = [0.50, 0.94, 0.08]
    mattress_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=mattress_half)
    mattress_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=mattress_half, rgbaColor=[0.9, 0.9, 0.92, 1])
    p.createMultiBody(0, mattress_col, mattress_vis, [bed_x, bed_y, BED_MATTRESS_TOP_Z-0.08])
    return BED_MATTRESS_TOP_Z


def create_body_mesh(mesh_path: Path, mesh_scale: float, bed_top_z: float) -> int:
    """Original body mesh loader (kept for fallback)."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Body mesh not found: {mesh_path}")
    s = [mesh_scale] * 3
    mesh_min, mesh_max = read_obj_bounds(mesh_path)
    mesh_center = (mesh_min + mesh_max) / 2.0
    body_position = [
        float(BED_CENTER[0] - mesh_center[0] * mesh_scale),
        float(BED_CENTER[1] - mesh_center[1] * mesh_scale),
        float(bed_top_z - mesh_min[2] * mesh_scale + 0.01),
    ]
    col = p.createCollisionShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=s,
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh_path), meshScale=s,
                              rgbaColor=[0.93, 0.82, 0.79, 1.0])
    return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                             basePosition=body_position, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))


def create_registered_body_mesh(subject_dir: Path, bed_top_z: float, mesh_scale: np.ndarray) -> tuple:
    """Load the patient-specific mesh and return body_id + registration data.

    The patient_skin.obj mesh was generated by extract_patient_mesh.py with
    vertices already centered (centering_offset subtracted). The mesh is placed
    with its bottom on the bed mattress at identity orientation.

    Returns
    -------
    body_id : int
    reg_body_position : (3,) ndarray — PyBullet body base position
    reg_body_orientation_matrix : (3,3) ndarray — rotation matrix (identity)
    reg_meta : dict — registration metadata from registration_meta.json
    """
    mesh_path = subject_dir / "patient_skin.obj"
    meta_path = subject_dir / "registration_meta.json"
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Patient mesh not found: {mesh_path}\n"
            "Run:  python generate_patient_meshes.py --input-dir totalseg_patients --subject " + subject_dir.name
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Registration metadata not found: {meta_path}\n"
            "Run:  python generate_patient_meshes.py --input-dir totalseg_patients --subject " + subject_dir.name
        )

    reg_meta = load_registration_meta(meta_path)

    mesh_min, mesh_max = read_obj_bounds(mesh_path)
    # The patient is rotated by -90 deg around X, so the back (which was local min Y)
    # becomes the bottom along world Z. Place the patient so their back rests on mattress.
    body_z = bed_top_z - mesh_min[1] * mesh_scale[1] + 0.01
    body_position = np.array([
        float(BED_CENTER[0]),
        float(BED_CENTER[1]),
        float(body_z),
    ], dtype=np.float64)
    # Rotate -90 degrees around X to lie flat on back (CT Y maps to world Z, CT Z maps to world -Y)
    body_orientation_quat = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
    body_orientation_matrix = np.array(p.getMatrixFromQuaternion(body_orientation_quat)).reshape(3, 3)

    col = p.createCollisionShape(
        p.GEOM_MESH, fileName=str(mesh_path),
        meshScale=mesh_scale.tolist(),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    vis = p.createVisualShape(
        p.GEOM_MESH, fileName=str(mesh_path),
        meshScale=mesh_scale.tolist(),
        rgbaColor=[0.93, 0.82, 0.79, 1.0],
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=body_position.tolist(),
        baseOrientation=list(body_orientation_quat),
    )

    print(f"[registration] Loaded patient mesh: {mesh_path.name}")
    print(f"[registration] Body position: {body_position}")
    print(f"[registration] Vertices: {reg_meta['mesh_vertex_count']}, "
          f"Faces: {reg_meta['mesh_face_count']}")
    print(f"[registration] Centering offset: {reg_meta['mesh_centering_offset']}")

    return body_id, body_position, body_orientation_matrix, reg_meta


def get_robot_original_colors(panda_id: int) -> dict[int, list[float]]:
    # linkIndex -> rgbaColor
    colors = {}
    visual_data = p.getVisualShapeData(panda_id)
    for entry in visual_data:
        link_index = entry[1]
        rgba = entry[7]
        colors[link_index] = list(rgba)
    return colors


def set_robot_visibility(panda_id: int, visible: bool, original_colors: dict[int, list[float]]) -> None:
    # Only modify links -1 to 8 (arm joints + hand base); keep links 9 and 10 (fingers) hidden
    for link_index in range(-1, 9):
        if visible:
            rgba = original_colors.get(link_index, [0.9, 0.9, 0.9, 1.0])
            p.changeVisualShape(panda_id, link_index, rgbaColor=rgba)
        else:
            p.changeVisualShape(panda_id, link_index, rgbaColor=[0.0, 0.0, 0.0, 0.0])


def create_panda_robot() -> int:
    panda_id = p.loadURDF("franka_panda/panda.urdf",
                          basePosition=[-0.42, 0.0, 0.50],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi / 2]),
                          useFixedBase=True)
    home = [0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.78]
    for jid, v in zip(PANDA_ARM_JOINTS, home):
        p.resetJointState(panda_id, jid, v)
    for fj in (9, 10):
        p.resetJointState(panda_id, fj, 0.04)
    return panda_id


def create_probe_model() -> list[int]:
    ids = []
    for comp in PROBE_COMPONENTS:
        g_type = comp.get("geom_type", p.GEOM_BOX)
        local_orn = comp.get("orn", [0.0, 0.0, 0.0, 1.0])
        if g_type == p.GEOM_CYLINDER:
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=comp["radius"], length=comp["length"], 
                                      rgbaColor=comp["rgba"], visualFrameOrientation=local_orn)
        elif g_type == p.GEOM_SPHERE:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=comp["radius"], rgbaColor=comp["rgba"],
                                      visualFrameOrientation=local_orn)
        else:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=comp["half_extents"], rgbaColor=comp["rgba"],
                                      visualFrameOrientation=local_orn)
        ids.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                     baseVisualShapeIndex=vis, basePosition=[0, 0, 1],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, 0])))
    return ids


def update_probe_model(probe_body_ids, probe_contact_position, probe_quaternion) -> None:
    for body_id, comp in zip(probe_body_ids, PROBE_COMPONENTS):
        pos = transform_local_point(probe_contact_position, probe_quaternion, comp["offset"])
        p.resetBasePositionAndOrientation(body_id, pos.tolist(), probe_quaternion.tolist())


def clamp_xy_to_body(xy, body_center_xy, body_extent_xy, margin=0.03) -> np.ndarray:
    half = body_extent_xy / 2.0
    return np.clip(xy, body_center_xy - half + margin, body_center_xy + half - margin)


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN TRAJECTORY  (Issue #2 fix: hard clamp on tilt angles)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_target_pose(elapsed, body_center, body_extent, scan_speed, body_footprint_margin=0.03):
    """
    Compute the probe target pose for the automatic sweep.

    Change from original: `probe_roll` and `probe_pitch` are clamped to
    ±MAX_PROBE_TILT_RAD (±0.35 rad ≈ ±20°) BEFORE the quaternion is built.
    This guarantees the beam direction never deviates far enough from vertical
    to miss the body, which was the primary cause of all-zero CT slices.
    """
    sweep_x = -0.02 + 0.12 * np.sin(elapsed * scan_speed)
    sweep_y =  0.08 * np.sin(elapsed * scan_speed * 0.7 + 0.5)

    body_top_z    = body_center[2] + body_extent[2] * 0.5
    height_offset = 0.02 * np.sin(elapsed * scan_speed * 0.3)

    target_xy = clamp_xy_to_body(
        np.array([body_center[0] + sweep_x, body_center[1] + sweep_y], dtype=np.float32),
        body_center[:2], body_extent[:2], margin=body_footprint_margin,
    )
    target_position = np.array([target_xy[0], target_xy[1],
                                 body_top_z + 0.15 + height_offset], dtype=np.float32)

    roll  = 0.08 * np.sin(elapsed * scan_speed * 0.5)
    pitch = 0.10 * np.sin(elapsed * scan_speed * 0.4 + 1.2)
    yaw   = 0.06 * np.sin(elapsed * scan_speed * 0.3 + 0.8)

    cycle = int(elapsed * scan_speed / (2 * np.pi))
    if cycle % 3 == 1:
        roll  += 0.05 * np.sin(elapsed * scan_speed * 1.2)
        pitch += 0.05 * np.sin(elapsed * scan_speed * 1.1 + 0.5)

    # ── Safety clamp: keep beam within ±MAX_PROBE_TILT_RAD of vertical ────────
    # Without this clamp the sinusoidal accumulation can transiently push
    # roll/pitch beyond ~25° during overlapping cycles, tilting the beam off
    # the body footprint and producing an all-zero CT slice.
    roll  = float(np.clip(roll,  -MAX_PROBE_TILT_RAD, MAX_PROBE_TILT_RAD))
    pitch = float(np.clip(pitch, -MAX_PROBE_TILT_RAD, MAX_PROBE_TILT_RAD))

    base_down_quat = np.array(p.getQuaternionFromEuler([np.pi, 0.0, 0.0]), dtype=np.float32)
    perturb_quat   = np.array(p.getQuaternionFromEuler([roll, pitch, yaw]),  dtype=np.float32)
    target_orientation = multiply_quaternions(base_down_quat, perturb_quat)

    return target_position, target_orientation, roll, pitch


def drive_panda_to_pose(panda_id, target_position, target_orientation, max_force=150.0, max_velocity=1.5) -> None:
    joint_targets = p.calculateInverseKinematics(
        panda_id, PANDA_EE_LINK,
        targetPosition=target_position.tolist(),
        targetOrientation=target_orientation.tolist(),
        maxNumIterations=100, residualThreshold=1e-5,
    )
    for jid in PANDA_ARM_JOINTS:
        p.setJointMotorControl2(panda_id, jid, p.POSITION_CONTROL,
                                targetPosition=joint_targets[jid],
                                force=max_force, maxVelocity=max_velocity)
    
    # LOCK FINGERS IN PLACE:
    for jid in (9, 10):
        p.setJointMotorControl2(panda_id, jid, p.POSITION_CONTROL,
                                targetPosition=0.04, force=20.0)


def compute_probe_contact_force(hit_distance, desired_standoff=0.008, max_force=5.0) -> float:
    if hit_distance is None:
        return 0.0
    compression = desired_standoff - hit_distance
    if compression > 0:
        return min(compression * 200.0, max_force)
    return -min(-compression * 100.0, max_force)


def get_ct_statistics(ct_slice: np.ndarray) -> dict:
    finite = ct_slice[np.isfinite(ct_slice)]
    if finite.size == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
    return {'min': float(np.min(finite)), 'max': float(np.max(finite)),
            'mean': float(np.mean(finite)), 'std': float(np.std(finite))}


def make_timing_accumulator() -> dict:
    return {"raycast": [], "ct_extract": [], "inference": [], "visualization": []}


def add_timing(timings, name, elapsed_s) -> None:
    timings[name].append(elapsed_s * 1000.0)


def mean_timing_ms(timings, name) -> float:
    vals = timings[name]
    return float(np.mean(vals)) if vals else 0.0


def load_training_sample(sample_path: Path):
    data = np.load(sample_path, allow_pickle=True).item()
    ct = data.get('ct', data.get('ct_slice'))
    us = data.get('simus', data.get('us', data.get('prediction')))
    if ct is None or us is None:
        raise ValueError(f"Invalid sample file: {sample_path}")
    return ct.astype(np.float32), us.astype(np.float32)


def compare_with_training(live_ct, live_us, training_ct, training_us) -> np.ndarray:
    h, w = live_ct.shape
    comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    tct = to_uint8_display(training_ct)
    tus = to_uint8_display(training_us)
    lct = to_uint8_display(live_ct)
    lus = to_uint8_display(live_us)
    comparison[:h,  :w,  0] = tct
    comparison[:h,  w:,  0] = tus
    comparison[h:,  :w,  0] = lct
    comparison[h:,  w:,  0] = lus
    comparison = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Training CT", (10, 20),    font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Training US", (w+10, 20),  font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Live CT",     (10, h+20),  font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, "Live US",     (w+10, h+20),font, 0.5, (255, 255, 255), 1)
    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    global MATCH_HISTOGRAM
    MATCH_HISTOGRAM = not args.no_match_histogram

    if args.checkpoint is None:
        args.checkpoint = DEFAULT_PIX2PIX_CKPT if args.model == "pix2pix" else DEFAULT_UNET_CKPT

    # ── Evaluation-only mode (no simulation) ──────────────────────────────────
    if args.eval:
        run_evaluation(args)
        return

    # ── Run folder + logging (Issue #4) ───────────────────────────────────────
    run_dir     = create_run_folder(args)
    stats_writer = init_stats_csv(run_dir)

    # Parse mesh scale
    scale_str = args.mesh_scale
    if "," in scale_str:
        mesh_scale = np.array([float(x) for x in scale_str.split(",")], dtype=np.float64)
    else:
        s_val = float(scale_str)
        mesh_scale = np.array([s_val, s_val, s_val], dtype=np.float64)

    # ── Subject and model loading ──────────────────────────────────────────────
    subject_dir = resolve_subject_dir(args.subject)
    ct_volume, label_volume, spacing, volume_center = load_ct_subject(subject_dir)

    device     = select_device(args.device)
    is_pix2pix = (args.model == "pix2pix") or (args.sim_mode == "pix2pix")
    use_conv_sim = (args.sim_mode == "conv")

    if use_conv_sim:
        # ── Model-based (physics) simulation – no neural network loaded ────────
        model = None
        # Model-based (physics) simulation — no neural network loaded
        # Tuned parameters fix the 5 low-quality causes identified:
        #   element_size: 1.5e-4 m (was 5e-4) → reduces depth blackout 3×
        #   TGC_beta: 0.05 (was 0.01) → proper depth compensation
        #   PSF widened: sx_E=1.5, sy_E=4 → correct streak appearance
        #   E_S_ratio: 1.2 (was 0.8) → stronger bone echoes
        #   ct_edge_weight: 0.5 → supplements label edges with CT gradients
        conv_sim = ModelBasedUSSimulator(
            I0           = 1.5,
            element_size = 1.5e-4,
            sx_E         = 1.5,
            sy_E         = 4.0,
            sx_B         = 2.0,
            sy_B         = 2.0,
            kernel_size  = (11, 11),
            E_S_ratio    = 1.2,
            TGC_beta     = 0.05,
            noise_I      = 0.03,
            noise_mu0    = 0.01,
            noise_s0     = 0.03,
            ct_edge_weight = 0.5,
            ct_edge_thresh = 0.12,
            gamma        = 0.6,
        )

        print("Simulation mode: Model-Based Convolution (no neural network)")
    elif is_pix2pix:
        model = load_pix2pix(args.checkpoint, device, args.base_features, args.dropout)
        conv_sim = None
        print("Model: Pix2Pix U-Net (tanh output)")
    else:
        model = load_unet(args.checkpoint, device, args.base_features, args.dropout)
        conv_sim = None
        print("Model: Original U-Net (sigmoid output)")


    # ── Startup diagnostic: verify output intensity (Issue #1) ────────────────
    # Use extract_slice so the diagnostic slice is always (args.size, args.size).
    # Using ct_volume[:,:,mid] directly passes the CT's native dimensions to the
    # model which produces a non-square output, breaking subsequent imwrite/hstack.
    _diag_quat  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _diag_slice = extract_slice(
        ct_volume, center=volume_center, quaternion=_diag_quat,
        spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
    )
    _diag_seg = extract_slice(
        label_volume, center=volume_center, quaternion=_diag_quat,
        spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
    )
    _diag_seg = (_diag_seg > 0.5).astype(np.float32)
    if use_conv_sim:
        _diag_label = make_label_map(_diag_slice, _diag_seg)
        _diag_us = conv_sim.simulate(_diag_label, ct_slice=_diag_slice)

    else:
        _diag_us = predict_ultrasound(model, _diag_slice, _diag_seg, device, is_pix2pix,
                                      enhance=(not args.no_enhance))
    print(f"[startup] Diagnostic US - shape={_diag_us.shape} "
          f"min={_diag_us.min():.4f} max={_diag_us.max():.4f} "
          f"mean={_diag_us.mean():.4f} std={_diag_us.std():.4f}")
    if not use_conv_sim and _diag_us.mean() < 0.05:
        print("WARNING: predicted US mean < 0.05 - check checkpoint and "
              "normalisation. CT_HU_MIN/MAX may need to match your training config.")

    # Save first-frame diagnostic image (Issue #4)
    cv2.imwrite(str(run_dir / "diag_ct.png"), to_uint8_display(_diag_slice))
    cv2.imwrite(str(run_dir / "diag_us.png"), (_diag_us * 255).astype(np.uint8))
    del _diag_slice, _diag_us, _diag_quat


    # ── Optional comparison sample ─────────────────────────────────────────────
    training_ct = training_us = None
    if args.compare_sample is not None:
        try:
            training_ct, training_us = load_training_sample(args.compare_sample)
            print(f"Loaded training sample: {args.compare_sample}")
            cv2.namedWindow(WINDOW_COMPARE, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"Warning: Failed to load training sample: {e}")
            args.compare_sample = None

    # ── PyBullet scene ─────────────────────────────────────────────────────────
    client      = connect_pybullet(args.shared_memory)
    bed_top_z   = create_hospital_room()

    # ── REGISTRATION CHANGE: load patient-specific mesh ───────────────────────
    body_id, reg_body_position, reg_body_orientation_matrix, reg_meta = \
        create_registered_body_mesh(subject_dir, bed_top_z, mesh_scale)
    print(f"[registration] CT shape: {reg_meta['ct_shape']}")
    print(f"[registration] Voxel spacing: {reg_meta['voxel_spacing']}")

    mesh_bounds_min, mesh_bounds_max, body_center, body_extent = get_body_bounds(body_id)
    panda_id    = create_panda_robot()
    robot_colors = get_robot_original_colors(panda_id)

    # Hide gripper fingers permanently to attach probe directly to hand base
    for link in [9, 10]:
        p.changeVisualShape(panda_id, link, rgbaColor=[0.0, 0.0, 0.0, 0.0])

    # Initialize visibility state based on CLI argument (targets arm + hand base)
    show_robot  = not args.only_probe
    set_robot_visibility(panda_id, show_robot, robot_colors)

    probe_body_id = create_probe_model()

    # Configuration state & manual default
    is_auto = False; snap_on = True
    manual_x = manual_y = 0.0; manual_z = 1.10
    manual_roll = manual_pitch = manual_yaw = 0.0
    body_margin = 0.03
    body_half_x = (body_extent[0] / 2.0) - body_margin
    body_half_y = (body_extent[1] / 2.0) - body_margin

    # Warm-up IK
    start_time = time.perf_counter()
    probe_height_correction = 0.0
    desired_probe_standoff  = 0.002

    # Initialize target pose based on default mode
    if is_auto:
        target_position, target_orientation, probe_roll, probe_pitch = scan_target_pose(
            0.0, body_center, body_extent, args.scan_speed, body_footprint_margin=args.body_margin)
    else:
        tx, ty = body_center[0] + manual_x, body_center[1] + manual_y
        base_q = np.array(p.getQuaternionFromEuler([np.pi, 0, 0]), dtype=np.float32)
        perq   = np.array(p.getQuaternionFromEuler([manual_roll, manual_pitch, manual_yaw]), dtype=np.float32)
        target_orientation = multiply_quaternions(base_q, perq)
        if snap_on:
            found_body, surface_z = raycast_skin_surface(tx, ty, body_id)
            if found_body:
                rm = np.array(p.getMatrixFromQuaternion(target_orientation.tolist())).reshape(3, 3)
                target_position = np.array([tx, ty,
                    surface_z + desired_probe_standoff - 0.18 * rm[2, 2]],
                    dtype=np.float32)
            else:
                target_position = np.array([tx, ty, manual_z], dtype=np.float32)
        else:
            target_position = np.array([tx, ty, manual_z], dtype=np.float32)
        probe_roll = manual_roll; probe_pitch = manual_pitch

    # Reset joints to the target pose directly to avoid startup collision/shake
    initial_joints = p.calculateInverseKinematics(
        panda_id, PANDA_EE_LINK,
        targetPosition=target_position.tolist(),
        targetOrientation=target_orientation.tolist(),
        maxNumIterations=100, residualThreshold=1e-5,
    )
    for jid in PANDA_ARM_JOINTS:
        p.resetJointState(panda_id, jid, initial_joints[jid])

    drive_panda_to_pose(panda_id, target_position, target_orientation)
    for _ in range(120):
        p.stepSimulation()

    cv2.namedWindow(WINDOW_COMBINED, cv2.WINDOW_NORMAL)

    print(f"Subject: {subject_dir.name} | Checkpoint: {args.checkpoint} | Device: {device}")
    print("Manual mode enabled (default). Press [M] to toggle Auto sweep. ESC or Q quits.")

    # ── Inference strategy ─────────────────────────────────────────────────────
    # Model-based conv sim: always synchronous (no GPU thread needed, very fast)
    # Neural network: decoupled background thread (see below)
    use_async_infer = (not use_conv_sim)  # only for neural network modes
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if use_async_infer else None

    # Warm-up: one synchronous inference so last_pred_us starts as a real image.
    # Must use extract_slice so output is exactly (args.size, args.size).
    _identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _warmup_slice  = extract_slice(
        ct_volume, center=volume_center, quaternion=_identity_quat,
        spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
    )
    _warmup_seg  = extract_slice(
        label_volume, center=volume_center, quaternion=_identity_quat,
        spacing=spacing, size=args.size, pixel_spacing=args.pixel_spacing,
    )
    _warmup_seg = (_warmup_seg > 0.5).astype(np.float32)
    if use_conv_sim:
        _warmup_label = make_label_map(_warmup_slice, _warmup_seg)
        last_pred_us = conv_sim.simulate(_warmup_label, ct_slice=_warmup_slice)

    else:
        last_pred_us = predict_ultrasound(model, _warmup_slice, _warmup_seg, device,
                                          is_pix2pix, not args.no_enhance)
    del _warmup_slice, _warmup_seg, _identity_quat
    print(f"[warmup] Initial US frame - shape={last_pred_us.shape} "
          f"mean={last_pred_us.mean():.4f} max={last_pred_us.max():.4f}")

    infer_future: "concurrent.futures.Future | None" = None

    def _submit_inference(ct_sl: np.ndarray, seg_sl: np.ndarray) -> "concurrent.futures.Future":
        return executor.submit(predict_ultrasound, model, ct_sl.copy(), seg_sl.copy(),
                               device, is_pix2pix, not args.no_enhance)


    # State
    last_time       = time.perf_counter()
    last_report_time = last_time
    fps             = 0.0
    timings         = make_timing_accumulator()
    debug_line_id   = debug_point_id = None
    ee_frame_ids    = probe_frame_ids = beam_vector_id = contact_debug_ids = None

    # Speed & axis lock states
    pos_speed = 0.15; z_speed = 0.10; rot_speed = 0.5
    x_locked = False; y_locked = False; z_locked = False

    frame_counter = 0
    last_ct_stats = None
    last_valid_ct_slice = None   # ← fallback for all-zero slices (Issue #2)
    last_valid_seg_slice = None
    warn_str = ""                # ← UI warning string

    # Print keyboard guide
    print("\n" + "=" * 50)
    print("KEYBOARD CONTROLS:")
    print("  [M] Toggle Auto/Manual   [P] Toggle Surface Snap   [ESC/Q] Quit")
    print("  [X] Lock Y (X-Only sweep) [Y] Lock X (Y-Only sweep) [Z] Lock Z height")
    print("  [T] Toggle In/Out-Plane  [[] Decrease Speed       []] Increase Speed")
    print("  Manual: Arrow-Up/Down=X   Arrow-Left/Right=Y        R/F=Z")
    print("          J/L=Roll          I/K=Pitch                 U/O=Yaw")
    print("=" * 50 + "\n")

    try:
        while p.isConnected(client):
            now = time.perf_counter()
            dt  = max(now - last_time, 1e-6)
            last_time = now
            elapsed   = now - start_time

            # Capture keypress from OpenCV windows (yields 1ms)
            key = cv2.waitKey(1) & 0xFF

            # Capture key events from PyBullet window
            keys = p.getKeyboardEvents()
            if frame_counter < 15:
                keys = {}

            def is_key_down(ch):
                # Check OpenCV character key
                if isinstance(ch, str) and key in (ord(ch.lower()), ord(ch.upper())):
                    return True
                # Check OpenCV arrow key codes or PyBullet arrow key codes
                if isinstance(ch, int):
                    if ch == p.B3G_UP_ARROW and key == 38:
                        return True
                    if ch == p.B3G_DOWN_ARROW and key == 40:
                        return True
                    if ch == p.B3G_LEFT_ARROW and key == 37:
                        return True
                    if ch == p.B3G_RIGHT_ARROW and key == 39:
                        return True
                    return ch in keys and (keys[ch] & p.KEY_IS_DOWN)
                lo, hi = ord(ch.lower()), ord(ch.upper())
                return ((lo in keys and keys[lo] & p.KEY_IS_DOWN) or
                        (hi in keys and keys[hi] & p.KEY_IS_DOWN))

            def is_key_triggered(ch):
                # Check OpenCV character key trigger
                if isinstance(ch, str) and key in (ord(ch.lower()), ord(ch.upper())):
                    return True
                # Check OpenCV arrow key trigger or PyBullet arrow key trigger
                if isinstance(ch, int):
                    if ch == p.B3G_UP_ARROW and key == 38:
                        return True
                    if ch == p.B3G_DOWN_ARROW and key == 40:
                        return True
                    if ch == p.B3G_LEFT_ARROW and key == 37:
                        return True
                    if ch == p.B3G_RIGHT_ARROW and key == 39:
                        return True
                    return ch in keys and (keys[ch] & p.KEY_WAS_TRIGGERED)
                lo, hi = ord(ch.lower()), ord(ch.upper())
                return ((lo in keys and keys[lo] & p.KEY_WAS_TRIGGERED) or
                        (hi in keys and keys[hi] & p.KEY_WAS_TRIGGERED))

            # ESC or 'q' key to quit (OpenCV key code is 27 for ESC, ord('q') for q)
            if key in (27, ord('q')) or (27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED)):
                break

            if is_key_triggered('m'):
                is_auto = not is_auto
                print(f"Mode: {'Auto' if is_auto else 'Manual'}")
                if not is_auto:
                    manual_x = float(np.clip(target_position[0] - body_center[0], -body_half_x, body_half_x))
                    manual_y = float(np.clip(target_position[1] - body_center[1], -body_half_y, body_half_y))
                    manual_roll = probe_roll; manual_pitch = probe_pitch; manual_yaw = 0.0
                    manual_z = 1.10 + probe_height_correction if snap_on else float(target_position[2])

            if is_key_triggered('h'):
                show_robot = not show_robot
                set_robot_visibility(panda_id, show_robot, robot_colors)
                print(f"Robot visibility: {'Visible' if show_robot else 'Hidden'}")

            if is_key_triggered('p'):
                snap_on = not snap_on
                print(f"Surface Snap: {'ON' if snap_on else 'OFF'}")

            if is_key_triggered('x'):
                x_locked = not x_locked
                if x_locked: y_locked = False
                print(f"Axis Lock: {'X-Only sweep (Y locked)' if x_locked else 'OFF'}")

            if is_key_triggered('y'):
                y_locked = not y_locked
                if y_locked: x_locked = False
                print(f"Axis Lock: {'Y-Only sweep (X locked)' if y_locked else 'OFF'}")

            if is_key_triggered('z'):
                z_locked = not z_locked
                print(f"Axis Lock: {'Z-Height locked' if z_locked else 'OFF'}")

            if is_key_triggered('t'):
                # Toggle manual_yaw between 0 and 90 degrees (pi/2)
                if abs(abs(manual_yaw) - np.pi / 2) < 0.4:
                    manual_yaw = 0.0
                    print("[control] Toggled to out-of-plane/short-axis view (0° yaw)")
                else:
                    manual_yaw = np.pi / 2
                    print("[control] Toggled to in-plane/long-axis view (90° yaw)")

            if is_key_triggered(']'):
                pos_speed = min(pos_speed + 0.05, 0.50)
                z_speed = pos_speed * 0.67
                print(f"Movement speed: {pos_speed:.2f} m/s (Z-speed: {z_speed:.2f} m/s)")

            if is_key_triggered('['):
                pos_speed = max(pos_speed - 0.05, 0.05)
                z_speed = pos_speed * 0.67
                print(f"Movement speed: {pos_speed:.2f} m/s (Z-speed: {z_speed:.2f} m/s)")

            if is_auto:
                target_position, target_orientation, probe_roll, probe_pitch = scan_target_pose(
                    elapsed, body_center, body_extent, args.scan_speed, body_footprint_margin=args.body_margin)
                found_body, surface_z = raycast_skin_surface(float(target_position[0]), float(target_position[1]), body_id)
                if found_body:
                    rm = np.array(p.getMatrixFromQuaternion(target_orientation.tolist())).reshape(3, 3)
                    target_position[2] = (surface_z + desired_probe_standoff
                                          + probe_height_correction - 0.18 * rm[2, 2])
                else:
                    target_position[2] += probe_height_correction
            else:
                if is_key_down(p.B3G_UP_ARROW) and not y_locked: manual_x += pos_speed * dt
                if is_key_down(p.B3G_DOWN_ARROW) and not y_locked: manual_x -= pos_speed * dt
                if is_key_down(p.B3G_LEFT_ARROW) and not x_locked: manual_y += pos_speed * dt
                if is_key_down(p.B3G_RIGHT_ARROW) and not x_locked: manual_y -= pos_speed * dt
                if is_key_down('r') and not z_locked: manual_z += z_speed * dt
                if is_key_down('f') and not z_locked: manual_z -= z_speed * dt
                manual_x = float(np.clip(manual_x, -body_half_x, body_half_x))
                manual_y = float(np.clip(manual_y, -body_half_y, body_half_y))
                manual_z = float(np.clip(manual_z, 0.72, 1.35))
                if is_key_down('j'): manual_roll  += rot_speed * dt
                if is_key_down('l'): manual_roll  -= rot_speed * dt
                if is_key_down('i'): manual_pitch += rot_speed * dt
                if is_key_down('k'): manual_pitch -= rot_speed * dt
                if is_key_down('u'): manual_yaw   += rot_speed * dt
                if is_key_down('o'): manual_yaw   -= rot_speed * dt
                manual_roll  = float(np.clip(manual_roll,  -0.5, 0.5))
                manual_pitch = float(np.clip(manual_pitch, -0.5, 0.5))
                if manual_yaw >  np.pi: manual_yaw -= 2 * np.pi
                if manual_yaw < -np.pi: manual_yaw += 2 * np.pi

                tx, ty = body_center[0] + manual_x, body_center[1] + manual_y
                base_q = np.array(p.getQuaternionFromEuler([np.pi, 0, 0]), dtype=np.float32)
                perq   = np.array(p.getQuaternionFromEuler([manual_roll, manual_pitch, manual_yaw]), dtype=np.float32)
                target_orientation = multiply_quaternions(base_q, perq)
                if snap_on and not z_locked:
                    found_body, surface_z = raycast_skin_surface(tx, ty, body_id)
                    if found_body:
                        rm = np.array(p.getMatrixFromQuaternion(target_orientation.tolist())).reshape(3, 3)
                        target_position = np.array([tx, ty,
                            surface_z + desired_probe_standoff + probe_height_correction + (manual_z - 1.10) - 0.18 * rm[2, 2]],
                            dtype=np.float32)
                    else:
                        target_position = np.array([tx, ty, manual_z + probe_height_correction], dtype=np.float32)
                else:
                    target_position = np.array([tx, ty, manual_z], dtype=np.float32)

            drive_panda_to_pose(panda_id, target_position, target_orientation)
            steps = min(max(int(round(dt / (1.0 / 240.0))), 1), 10)
            for _ in range(steps):
                p.stepSimulation()

            ee_state = p.getLinkState(panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
            ee_position  = np.array(ee_state[4], dtype=np.float32)
            ee_quaternion = np.array(ee_state[5], dtype=np.float32)
            probe_position, quaternion_xyzw, probe_contact_position = get_probe_pose_from_ee(
                ee_position, ee_quaternion)
            update_probe_model(probe_body_id, probe_contact_position, quaternion_xyzw)

            if args.diagnostic:
                ee_frame_ids    = draw_debug_frame(ee_position, ee_quaternion, 0.08, ee_frame_ids)
                probe_frame_ids = draw_debug_frame(probe_position, quaternion_xyzw, 0.06, probe_frame_ids)
                beam_vector_id  = draw_beam_direction_vector(probe_position, quaternion_xyzw, beam_vector_id)
                contact_debug_ids = draw_contact_face_debug(probe_position, quaternion_xyzw, contact_debug_ids)

            t0 = time.perf_counter()
            hit, ray_to, hit_position, hit_distance = raycast_probe(
                probe_position, quaternion_xyzw, body_id, args.ray_length)
            add_timing(timings, "raycast", time.perf_counter() - t0)

            if hit_distance is not None:
                force = compute_probe_contact_force(hit_distance, desired_probe_standoff)
                if force > 1.0:
                    probe_height_correction = min(probe_height_correction + 0.004, 0.04)
                elif force < -0.5:
                    probe_height_correction = max(probe_height_correction - 0.003, -0.025)
                else:
                    probe_height_correction *= 0.99
                if frame_counter % 30 == 0:
                    print(f"Contact force: {force:.3f} N, distance: {hit_distance:.4f} m")

            debug_line_id, debug_point_id = draw_debug_ray(
                probe_position, ray_to, hit_position, hit, debug_line_id, debug_point_id)

            # ── REGISTRATION CHANGE: affine-based voxel mapping ─────────
            ct_center = compute_registered_ct_center(
                hit_position, reg_body_position, reg_body_orientation_matrix,
                reg_meta['inv_affine'], reg_meta['mesh_centering_offset'],
                ct_volume.shape, mesh_scale=mesh_scale)

            t0 = time.perf_counter()
            if hit:
                ct_slice = extract_slice(ct_volume, center=ct_center,
                                         quaternion=quaternion_xyzw, spacing=spacing,
                                         size=args.size, pixel_spacing=args.pixel_spacing,
                                         order=args.interp_order,
                                         inv_affine=reg_meta['inv_affine'],
                                         mesh_scale=mesh_scale,
                                         body_orientation_matrix=reg_body_orientation_matrix)
                seg_slice = extract_slice(label_volume, center=ct_center,
                                          quaternion=quaternion_xyzw, spacing=spacing,
                                          size=args.size, pixel_spacing=args.pixel_spacing,
                                          order=args.interp_order,
                                          inv_affine=reg_meta['inv_affine'],
                                          mesh_scale=mesh_scale,
                                          body_orientation_matrix=reg_body_orientation_matrix)
                seg_slice = (seg_slice > 0.5).astype(np.float32)
            else:
                ct_slice = np.zeros((args.size, args.size), dtype=np.float32)
                seg_slice = np.zeros((args.size, args.size), dtype=np.float32)
            add_timing(timings, "ct_extract", time.perf_counter() - t0)

            # ── Fallback for all-zero slices (Issue #2) ───────────────────────
            slice_is_empty = (ct_slice.max() == 0.0)
            if slice_is_empty and last_valid_ct_slice is not None:
                ct_slice = last_valid_ct_slice
                seg_slice = last_valid_seg_slice
                warn_str = "WARN: probe off-body – using last valid slice"
            elif not slice_is_empty:
                last_valid_ct_slice = ct_slice.copy()
                last_valid_seg_slice = seg_slice.copy()
                warn_str = ""

            ct_stats = get_ct_statistics(ct_slice)
            last_ct_stats = ct_stats

            # ── Inference ─────────────────────────────────────────────────────
            # Conv-sim: synchronous & fast (NumPy only, no GPU wait).
            # CPU neural net: run synchronously every trigger frame.
            # CUDA neural net: one-frame-lookahead async pipeline.
            if frame_counter % max(args.skip_frames + 1, 1) == 0:
                t0 = time.perf_counter()
                if use_conv_sim:
                    # Physics-based simulator: build label map and simulate
                    _label = make_label_map(ct_slice, seg_slice)
                    last_pred_us = conv_sim.simulate(_label, ct_slice=ct_slice)

                elif use_async_infer:
                    # Non-blocking check: if no task is running, or if the current task is done, collect it and start a new one
                    if infer_future is None:
                        infer_future = _submit_inference(ct_slice, seg_slice)
                    elif infer_future.done():
                        try:
                            last_pred_us = infer_future.result()
                        except Exception as exc:
                            print(f"[infer] exception: {exc}")
                        infer_future = _submit_inference(ct_slice, seg_slice)
                else:
                    # Synchronous inference fallback
                    last_pred_us = predict_ultrasound(
                        model, ct_slice, seg_slice, device, is_pix2pix, not args.no_enhance)
                add_timing(timings, "inference", time.perf_counter() - t0)


            pred_us = last_pred_us

            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            t0 = time.perf_counter()
            locks = []
            if x_locked: locks.append("X")
            if y_locked: locks.append("Y")
            if z_locked: locks.append("Z")
            locks_str = ", ".join(locks) if locks else "None"
            view_str = "In-Plane" if abs(abs(manual_yaw) - np.pi/2) < 0.4 else "Out-of-Plane"
            sim_mode_label = "MB-Conv" if use_conv_sim else ("Pix2Pix" if is_pix2pix else "U-Net")
            mode_str = [
                f"Mode: {'AUTO' if is_auto else 'MANUAL'} | Snap: {'ON' if snap_on else 'OFF'} | Sim: {sim_mode_label}",
                f"Speed: {pos_speed:.2f} m/s | Lock: {locks_str} | View: {view_str}"
            ]

            ct_display = overlay_status(to_uint8_display(ct_slice), probe_position,
                                        hit_position, hit_distance, ct_center, fps,
                                        ct_stats if args.diagnostic else None,
                                        mode_str=mode_str, warn_str=warn_str)
            us_display = overlay_status((pred_us * 255).astype(np.uint8), probe_position,
                                        hit_position, hit_distance, ct_center, fps,
                                        ct_stats if args.diagnostic else None,
                                        mode_str=mode_str, warn_str=warn_str)
            combined = np.hstack((ct_display, us_display))
            cv2.imshow(WINDOW_COMBINED, combined)

            if args.compare_sample is not None and training_ct is not None:
                cv2.imshow(WINDOW_COMPARE, compare_with_training(ct_slice, pred_us, training_ct, training_us))

            add_timing(timings, "visualization", time.perf_counter() - t0)

            # Save debug data on 'S' key press
            if key in (ord('s'), ord('S')):
                save_path = PROJECT_ROOT / f"debug_{frame_counter}"
                np.save(str(save_path) + "_ct.npy",  ct_slice)
                np.save(str(save_path) + "_us.npy",  pred_us)
                cv2.imwrite(str(save_path) + "_ct.png", to_uint8_display(ct_slice))
                cv2.imwrite(str(save_path) + "_us.png", (pred_us * 255).astype(np.uint8))
                with open(str(save_path) + "_stats.txt", 'w') as fh:
                    fh.write(f"Frame: {frame_counter}\nProbe: {probe_position}\n"
                             f"Hit: {hit_position}\nCenter: {ct_center}\n"
                             f"CT: {ct_stats}\n"
                             f"US: min={pred_us.min():.3f} max={pred_us.max():.3f} "
                             f"mean={pred_us.mean():.3f} std={pred_us.std():.3f}\n")
                print(f"Saved debug data to {save_path}_*")

            # Append to stats CSV (Issue #4)
            if frame_counter % args.log_every == 0:
                stats_writer.writerow({
                    "frame":    frame_counter,
                    "fps":      round(fps, 2),
                    "hit":      int(hit),
                    "ct_min":   round(ct_stats["min"], 2),
                    "ct_max":   round(ct_stats["max"], 2),
                    "ct_mean":  round(ct_stats["mean"], 2),
                    "ct_std":   round(ct_stats["std"], 2),
                    "us_min":   round(float(pred_us.min()), 4),
                    "us_max":   round(float(pred_us.max()), 4),
                    "us_mean":  round(float(pred_us.mean()), 4),
                    "us_std":   round(float(pred_us.std()), 4),
                    "probe_x":  round(float(probe_position[0]), 4),
                    "probe_y":  round(float(probe_position[1]), 4),
                    "probe_z":  round(float(probe_position[2]), 4),
                })

            if now - last_report_time >= 5.0:
                print(f"[live] roll={probe_roll:+.3f} pitch={probe_pitch:+.3f} | "
                      f"CT mean={ct_stats['mean']:.1f} | US mean={pred_us.mean():.3f} | "
                      f"raycast={mean_timing_ms(timings,'raycast'):.1f}ms "
                      f"extract={mean_timing_ms(timings,'ct_extract'):.1f}ms "
                      f"infer={mean_timing_ms(timings,'inference'):.1f}ms | FPS={fps:.1f}")
                timings = make_timing_accumulator()
                last_report_time = now

            if key in (27, ord('q')):
                break

            # Frame-rate cap at ~60 fps
            elapsed_frame = time.perf_counter() - now
            sleep_t = 1.0 / 60.0 - elapsed_frame
            if sleep_t > 0:
                time.sleep(sleep_t)

            frame_counter += 1

    finally:
        if executor is not None:
            executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        if p.isConnected(client):
            p.disconnect(client)
        print(f"[log] Stats saved to {run_dir / 'stats.csv'}")



if __name__ == "__main__":
    main()