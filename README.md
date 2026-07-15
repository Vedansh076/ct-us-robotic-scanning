# CT-to-Ultrasound Robotic Scanning Simulation

> A PyBullet-based simulation environment where a Franka Emika Panda robot arm equipped with a curved ultrasound probe autonomously scans a patient torso, synthesizing realistic B-mode ultrasound images from CT volumes in real-time using deep learning and physics-based rendering.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements & Dependencies](#2-system-requirements--dependencies)
3. [Setup & Installation](#3-setup--installation)
4. [Data Pipeline](#4-data-pipeline)
5. [Running the Interactive Simulation](#5-running-the-interactive-simulation)
6. [Ultrasound Synthesis Modes](#6-ultrasound-synthesis-modes)
7. [Training Autonomous Scanning Agents](#7-training-autonomous-scanning-agents)
8. [Evaluating Trained Policies](#8-evaluating-trained-policies)
9. [Imitation Learning from Real Robot Data](#9-imitation-learning-from-real-robot-data)
10. [Gymnasium Environment Reference](#10-gymnasium-environment-reference)
11. [Cloud Training on Google Colab](#11-cloud-training-on-google-colab)
12. [Repository Structure](#12-repository-structure)
13. [Documentation Map](#13-documentation-map)
14. [Git Workflow](#14-git-workflow)

---

## 1. Project Overview

This project builds an end-to-end pipeline for **autonomous robotic ultrasound scanning**:

1. **Patient Mesh Generation** — 3D CT volumes from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) are processed into watertight skin meshes (`.obj`) and binary bone labels (`.nii.gz`) using marching cubes.

2. **4 Ultrasound Synthesis Modes** — B-mode ultrasound images are synthesized from CT data using four interchangeable methods:
   - **U-Net** — 2-channel deep learning (CT + bone mask → simulated US)
   - **Pix2Pix GAN** — Conditional adversarial network
   - **Physics Convolution** — Rayleigh speckle + tissue attenuation + PSF (no GPU required)
   - **Ray-Tracing** — Snell's law refraction at tissue boundaries (no GPU required)

3. **PyBullet Physics Simulation** — Franka Panda robot arm, curved convex-array probe, patient mesh, and hospital bed loaded in a full physics scene with gravity and contact forces.

4. **Coordinate Registration** — Mathematically exact affine transforms map the probe's contact position in PyBullet world coordinates to CT voxel indices, accounting for NIfTI LPS conventions, mesh centering, body placement, and non-isotropic scaling.

5. **OpenAI Gymnasium Environment** — Standard `gymnasium.Env` interface compatible with Stable-Baselines3 RL algorithms.

6. **5 Trained Autonomous Scanning Algorithms:**

   | Algorithm | Type | Peak Reward | Checkpoint |
   |-----------|------|-------------|------------|
   | **SAC** | Off-policy RL | **+317.0** (best) | `sac_checkpoints/sac_final_model.zip` |
   | **A2C** | On-policy RL | +264.1 | `a2c_checkpoints/a2c_final_model.zip` |
   | **PPO** | On-policy RL | -237.0 | `ppo_checkpoints/` (server) |
   | **BC** | Imitation Learning | loss 2.0 | `bc_checkpoints/bc_policy.zip` |
   | **GAIL** | Adversarial IL | — | `gail_checkpoints/gail_policy.zip` |

---

## 2. System Requirements & Dependencies

| Component | Version / Notes |
|---|---|
| Python | 3.9 – 3.11 |
| PyTorch | ≥ 2.0 (CPU or CUDA 11.8+) |
| PyBullet | ≥ 3.2 |
| Gymnasium | ≥ 0.29 |
| Stable-Baselines3 | ≥ 2.0 |
| nibabel | ≥ 5.0 |
| scikit-image | ≥ 0.21 (for `marching_cubes`, `match_histograms`) |
| scipy | ≥ 1.11 |
| OpenCV | ≥ 4.8 |
| NumPy | ≥ 1.24 |
| imitation | ≥ 1.0 (for BC/GAIL training only) |

Install all Python dependencies:

```bash
pip install torch torchvision pybullet gymnasium stable-baselines3[extra] \
            nibabel scikit-image scipy opencv-python trimesh tqdm imitation
```

---

## 3. Setup & Installation

### 3.1 Clone the Repository

```bash
git clone https://github.com/Vedansh076/ct-us-robotic-scanning.git
cd ct-us-robotic-scanning
```

### 3.2 Download TotalSegmentator Patient Data

The `download_totalseg.py` script streams 5 pre-selected subjects from Zenodo (~3.24 GB compressed) and extracts them into `totalseg_patients/`:

```bash
python download_totalseg.py
```

Subject IDs downloaded: **s0011, s0058, s0223, s0250, s0310**.  
The ZIP is cached locally after the first download.

> **Note:** If you have pre-downloaded TotalSegmentator data, you can manually copy subject folders (e.g. `s0058/`) into `totalseg_patients/`. Each folder must contain `ct.nii.gz` and a `segmentations/` subfolder with the per-structure NIfTI masks.

### 3.3 Generate Patient Meshes, Bone Labels & Registration Metadata

For each subject, run the mesh generator. This produces three files per subject that the simulation requires:

| Output File | Description |
|---|---|
| `patient_skin.obj` | Watertight, Laplacian-smoothed 3D skin surface mesh for PyBullet collision physics |
| `bone_label.nii.gz` | Binary volume merging 49 bone structures (vertebrae, ribs, pelvis, sternum, shoulders) |
| `registration_meta.json` | NIfTI affine, inverse affine, voxel spacing, and mesh centering offset for the registration pipeline |

```bash
# Process all subjects at once
python generate_patient_meshes.py --input-dir totalseg_patients --smooth-iter 10

# Or process a single subject
python generate_patient_meshes.py --input-dir totalseg_patients --subject s0058
```

**Key implementation details:**
- The body binary volume is zero-padded before marching cubes to guarantee fully closed (manifold) meshes with no open borders.
- Triangle winding order is inverted so all normals point outward — this prevents PyBullet's back-face culling from making the mesh appear transparent/hollow.

---

## 4. Data Pipeline

```
totalseg_patients/s0058/
├── ct.nii.gz              ← Raw 3D CT volume (TotalSegmentator format)
├── segmentations/         ← Per-structure NIfTI masks (body.nii.gz, vertebrae_*, rib_*, ...)
│
│   [generate_patient_meshes.py produces ↓]
│
├── patient_skin.obj       ← Smoothed skin surface mesh for PyBullet
├── bone_label.nii.gz      ← Merged binary bone mask (for 2-channel U-Net input)
└── registration_meta.json ← Spatial registration metadata (affines, spacing, centroid offset)
```

The **registration_meta.json** stores:
- `affine` / `inv_affine`: The NIfTI affine matrix and its inverse for voxel↔physical-mm conversion.
- `voxel_spacing`: The CT voxel size in mm (e.g. `[0.7, 0.7, 1.5]`).
- `mesh_centering_offset`: The 3D offset subtracted from physical-meter vertices to center the mesh at the origin before PyBullet placement.
- `mesh_vertex_count`, `mesh_face_count`: Mesh size stats for reference.

---

## 5. Running the Interactive Simulation

Launch the real-time PyBullet GUI demo:

```bash
# Use the default patient (s0058) with U-Net synthesis
python live_unet_demo.py

# Use a different subject
python live_unet_demo.py --subject totalseg_patients/s0223

# Enable intensity histogram matching (reduces domain gap between CT and training data)
python live_unet_demo.py --match-histogram

# Skip neural network — display the raw bone mask overlay
python live_unet_demo.py --skip-unet
```

### Keyboard Controls

| Key | Action |
|---|---|
| `↑ ↓ ← →` Arrow Keys | Translate probe Forward / Backward / Left / Right |
| `R` / `F` | Translate probe Up / Down |
| `J` / `L` | Roll rotation (Left / Right) |
| `I` / `K` | Pitch rotation (Forward / Backward) |
| `U` / `O` | Yaw rotation (Clockwise / Counter-Clockwise) |
| `[` / `]` | Decrease / Increase translational scanning speed |
| `X` / `Y` / `Z` | Lock movement to X, Y, or Z axis (Z-lock also disables surface snapping) |
| `T` | Toggle In-Plane (longitudinal) ↔ Out-of-Plane (transverse) B-mode scan view |
| `M` | Toggle between Auto-sweep and Manual scanning modes |
| `P` | Toggle skin surface contact snapping |
| `S` | Save a debug snapshot (ultrasound image, probe stats, CT slice) |
| `ESC` / `Q` | Quit the simulation |

---

## 6. Ultrasound Synthesis Modes

The simulation supports 4 interchangeable B-mode ultrasound synthesis methods:

```bash
python live_unet_demo.py --sim-mode unet       # Deep learning U-Net (default, needs GPU)
python live_unet_demo.py --sim-mode pix2pix     # Pix2Pix conditional GAN (needs GPU)
python live_unet_demo.py --sim-mode conv        # Physics convolution (CPU only)
python live_unet_demo.py --sim-mode ray         # Ray-tracing with Snell's law (CPU only)
```

| Mode | Method | GPU Required | Speed | Key Features |
|------|--------|:---:|-------|-------------|
| `unet` | 2-channel U-Net | Yes | ~30 Hz | CT+bone → simulated US, trained on Cavalcanti spine data |
| `pix2pix` | Pix2Pix GAN | Yes | ~25 Hz | Conditional adversarial training for sharper textures |
| `conv` | Physics v3 | No | ~35 Hz | Rayleigh speckle, CT backscatter, carrier PSF, attenuation |
| `ray` | Ray-tracing | No | ~28ms/frame | Snell's law refraction, shadow ratio 0.49 |

### Model Checkpoints

| Synthesis Mode | Checkpoint Path |
|---|---|
| U-Net | `model/runs/exp1_2IP/exp1/best_model.pth` |
| Pix2Pix | `model/runs/exp_pix2pix/best_model.pth` |
| Conv / Ray | No checkpoint needed (physics-based) |

Specify a custom checkpoint:
```bash
python live_unet_demo.py --checkpoint model/runs/exp_pix2pix/best_model.pth --sim-mode pix2pix
```

### Quantitative Evaluation

Calculate SSIM, PSNR, and MAE over test subjects:
```bash
python live_unet_demo.py --checkpoint model/runs/exp1_2IP/exp1/best_model.pth --eval
```

---

## 7. Training Autonomous Scanning Agents

All RL agents are trained inside the `RoboticUltrasoundGymEnv` Gymnasium environment using Stable-Baselines3.

By default, training uses **Strategy 2 (mask-based)** mode (`skip_unet=True`), which bypasses U-Net inference entirely and uses the binary bone mask as the observation image. This achieves **~440 FPS** versus ~12 FPS with U-Net enabled. The learned policy transfers naturally to U-Net outputs at evaluation time.

### 7.1 A2C (Advantage Actor-Critic)

```bash
# Recommended (on server with dedicated CPU cores)
nohup taskset -c 0,1,2,3 python3 train_a2c.py --timesteps 150000 --save-freq 30000 > train_a2c.log 2>&1 &

# Local training
python train_a2c.py --timesteps 100000 --n-steps 20 --lr 7e-4 --save-freq 5000
```

### 7.2 SAC (Soft Actor-Critic) — **Best Performance**

```bash
nohup taskset -c 0,1,2,3 python3 train_sac.py --timesteps 100000 --save-freq 20000 > train_sac.log 2>&1 &
```

### 7.3 PPO (Proximal Policy Optimization)

```bash
nohup taskset -c 0,1,2,3 python3 train_ppo.py --timesteps 100000 --n-steps 2048 --batch-size 64 > train_ppo.log 2>&1 &
```

### 7.4 Monitor Training with TensorBoard

```bash
tensorboard --logdir ./a2c_tensorboard/
```

Navigate to `http://localhost:6006` to view episode reward, policy loss, and value loss curves.

---

## 8. Evaluating Trained Policies

Use `enjoy_policy.py` to visually evaluate any trained policy in the PyBullet GUI. The algorithm is auto-detected from the checkpoint filename:

```bash
# A2C
python enjoy_policy.py --checkpoint a2c_checkpoints/a2c_final_model.zip

# SAC (best performing)
python enjoy_policy.py --checkpoint sac_checkpoints/sac_final_model.zip

# PPO
python enjoy_policy.py --checkpoint ppo_checkpoints/ppo_final_model.zip

# Behavioral Cloning (requires --algo bc)
python enjoy_policy.py --checkpoint bc_checkpoints/bc_policy.zip --algo bc

# GAIL
python enjoy_policy.py --checkpoint gail_checkpoints/gail_policy.zip --algo gail

# Headless evaluation (no GUI)
python enjoy_policy.py --checkpoint sac_checkpoints/sac_final_model.zip --headless
```

---

## 9. Imitation Learning from Real Robot Data

The project supports learning from the [Cavalcanti Robotic Lumbar Spine Dataset](https://data.mendeley.com/) — real UR5 scanning poses from 7 volunteers (21 sweeps, 68k poses).

### 9.1 Collect Expert Demonstrations

Parses `RUS_pose.txt` files, converts real scanning trajectories to normalized delta-actions, and replays them through the environment:

```bash
# Dry-run: print statistics without creating the environment
python3 collect_demos.py --data-root data/Cavalcanti --dry-run

# Full collection: replay through env and save trajectory .npz files
python3 collect_demos.py --data-root data/Cavalcanti --output-dir demos/ --stride 15

# Process specific volunteers only
python3 collect_demos.py --data-root data/Cavalcanti --output-dir demos/ --volunteers URS01 URS02
```

### 9.2 Train Behavioral Cloning Policy

```bash
nohup python3 train_bc.py --demos-dir demos/ --epochs 50 > train_bc.log 2>&1 &

# With custom hyperparameters
python3 train_bc.py --demos-dir demos/ --epochs 100 --lr 1e-4 --batch-size 128
```

### 9.3 Train GAIL Policy

```bash
nohup python3 train_gail.py --demos-dir demos/ --timesteps 100000 > train_gail.log 2>&1 &
```

---

## 10. Gymnasium Environment Reference

### Action Space — 6-DOF continuous, clipped to `[-1, 1]`:

| Index | Meaning | Scale Applied |
|---|---|---|
| 0–2 | `dx, dy, dz` (position delta) | ×0.01 m (±1 cm per step) |
| 3–5 | `droll, dpitch, dyaw` (orientation delta) | ×0.05 rad (±2.9° per step) |

### Observation Space — Dictionary:

| Key | Shape | Type | Description |
|---|---|---|---|
| `"image"` | `(256, 256)` | `uint8` | Synthesized B-mode US image (or bone mask) |
| `"force"` | `(1,)` | `float32` | Estimated normal contact force in Newtons |
| `"pose"` | `(7,)` | `float32` | EE position (xyz) + orientation (quaternion xyzw) |

### Reward Function:

| Component | Value | Condition |
|---|---|---|
| `R_f` (force reward) | `+1.0` | Good contact: 2 N ≤ F ≤ 8 N |
| `R_f` | `+0.5 × F` | Light contact: 0 < F < 2 N |
| `R_f` | `−1.0 × (F − 8)` | Over-pressure: F > 8 N |
| `R_f` | `−2.0` | No contact: F = 0 |
| `R_b` (bone reward) | `+1.5` | Bone pixels visible in segmentation slice |
| `R_sweep` (sweep) | `+0.5 × |action_y|` | Longitudinal sweep while in 2–8N contact over bone |
| `R_jerk` (smoothness) | `−0.1 × ‖Δaction‖²` | Always (penalizes inter-step jerk) |

### Termination Conditions:

- `terminated = True` if contact force exceeds **12 N** (patient safety limit).
- `terminated = True` if no contact is maintained for **30 consecutive steps**.
- `truncated = True` after **200 steps** (episode horizon).

### Contact Physics:

- Spring constant: **k = 800 N/m**, standoff = 3 mm
- Ideal contact window: **2–8 N**
- Orientation clamped to ±0.15 rad (±8.6°) from perpendicular

### Verify the Environment

```bash
python test_gym_env.py           # Visual mode (PyBullet GUI)
python test_gym_env.py --headless  # Headless mode (faster)
```

---

## 11. Cloud Training on Google Colab

For users without a local GPU, see **[COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)** for a full step-by-step guide to:
1. Upload the `ct-us-colab.zip` project bundle to Google Drive.
2. Extract and install dependencies on a Colab T4 GPU instance.
3. Launch 500k-step A2C training (~1.9 hours at ~70 FPS).
4. Back up checkpoints and TensorBoard logs back to Google Drive.

---

## 12. Repository Structure

```
ct_us/
│
├── ── Core Simulation ──────────────────────────────────────────────────────
│
├── live_unet_demo.py          # Central hub: PyBullet GUI + all 4 US synthesis modes
├── robotic_us_env.py          # Gymnasium Env wrapper (wraps PyBullet + slicing + synthesis)
├── extract_slice.py           # Registration-aware 2D oblique slicer (trilinear interpolation)
├── registration.py            # Exact affine transforms: PyBullet world ↔ CT voxel
│
├── ── Data Pipeline ────────────────────────────────────────────────────────
│
├── download_totalseg.py       # Streams 5 TotalSegmentator subjects from Zenodo (3.24 GB)
├── generate_patient_meshes.py # CT → patient_skin.obj + bone_label.nii.gz + meta.json
├── gen_data.py                # 3D patient volumes → paired 2D training slices
│
├── ── RL / IL Training ─────────────────────────────────────────────────────
│
├── train_a2c.py               # A2C training (peak +264.1)
├── train_sac.py               # SAC training (peak +317.0, project best)
├── train_ppo.py               # PPO training (benchmark)
├── train_bc.py                # Behavioral Cloning from expert demos
├── train_gail.py              # GAIL adversarial imitation learning
├── collect_demos.py           # Parse Cavalcanti UR5 poses → env replay → demo collection
├── enjoy_policy.py            # Visual evaluation of any trained policy (A2C/SAC/PPO/BC/GAIL)
├── test_gym_env.py            # Environment diagnostic & FPS benchmark
│
├── ── Generative Models ────────────────────────────────────────────────────
│
├── model/
│   ├── model.py               # 5-level U-Net (InstanceNorm, Sigmoid output, 2-ch input)
│   ├── dataset.py             # CTSimUSDataset: HU clipping + normalization + slice stacking
│   ├── train.py               # Training loop: L1 loss, AMP, CosineAnnealingLR
│   ├── inference.py           # Batch inference: PSNR/SSIM metrics, side-by-side plots
│   ├── prepare_cavalcanti.py  # Cavalcanti dataset preprocessor: DICOM + ICP + oblique reslice
│   ├── train_ultrabones.py    # UltraBones100k ex-vivo dataset training
│   ├── ultrabones_dataset.py  # UltraBones100k dataset loader
│   ├── reference_ct_slice.npy # Reference histogram template for match_histograms
│   ├── pix2pix/
│   │   ├── model.py           # Pix2Pix U-Net variant (Tanh output)
│   │   ├── discriminator.py   # PatchGAN discriminator
│   │   └── train_pix2pix.py   # Pix2Pix GAN training loop
│   └── runs/
│       ├── exp1_2IP/exp1/best_model.pth  ← Pre-trained 2-channel U-Net
│       └── exp_pix2pix/best_model.pth    ← Pre-trained Pix2Pix GAN
│
├── ── Checkpoints (git-ignored) ────────────────────────────────────────────
│
├── a2c_checkpoints/           # A2C agent checkpoints
├── sac_checkpoints/           # SAC agent checkpoints (project-best +317.0)
├── bc_checkpoints/            # Behavioral Cloning policy
├── gail_checkpoints/          # GAIL policy
│
├── ── Patient Data (git-ignored) ───────────────────────────────────────────
│
├── totalseg_patients/
│   └── {s0011,s0058,s0223,s0250,s0310}/
│       ├── ct.nii.gz, segmentations/, patient_skin.obj,
│       ├── bone_label.nii.gz, registration_meta.json
│
├── ── Documentation ────────────────────────────────────────────────────────
│
├── README.md                  # This file: setup, usage, reference
├── DEVELOPER_GUIDE.md         # Quick-reference handover for new developers
├── PROJECT_DOCUMENTATION.md   # Deep-dive technical architecture reference
├── COLAB_INSTRUCTIONS.md      # Google Colab training guide
├── commands.md                # All CLI command reference
├── agent.md                   # AI agent session context tracker
└── task.md                    # Project roadmap and task checklist
```

---

## 13. Documentation Map

| Document | Purpose |
|----------|---------|
| **README.md** | Setup, installation, usage, and quick reference (this file) |
| **DEVELOPER_GUIDE.md** | Developer handover: file reference, architecture, gotchas, checkpoints |
| **PROJECT_DOCUMENTATION.md** | Deep-dive: coordinate math, module walkthroughs, RL design, bug history |
| **COLAB_INSTRUCTIONS.md** | Step-by-step Google Colab training guide |
| **commands.md** | Central CLI command reference for all scripts |
| **model/NOTES.md** | U-Net architecture decisions and hyperparameter rationale |
| **SONOGYM_ANALYSIS.md** | Analysis of the SonoGym paper for reference |
| **agent.md** | AI agent session-to-session context tracker |
| **task.md** | Project roadmap and task completion checklist |

---

## 14. Git Workflow

### Pulling Updates (on training device)

```bash
git pull origin main
```

### Pushing Updates (on development device)

```bash
git add .
git commit -m "Brief description of what changed"
git push origin main
```

> **Note:** The `.gitignore` excludes `totalseg_patients/`, checkpoint directories, `__pycache__/`, and large binary files (`*.nii.gz`, `*.obj`, `*.pth`, `*.zip`, `*.png`, `*.npy`). Only code, scripts, and lightweight metadata are tracked.
