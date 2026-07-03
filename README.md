# CT-to-Ultrasound Robotic Scanning Simulation

> A PyBullet-based simulation environment where a Franka Emika Panda robot arm equipped with an ultrasound probe autonomously scans a patient torso, synthesizing realistic B-mode ultrasound images from CT volumes in real-time using a deep-learning U-Net.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements & Dependencies](#2-system-requirements--dependencies)
3. [Setup & Installation](#3-setup--installation)
4. [Data Pipeline](#4-data-pipeline)
5. [Running the Interactive Simulation](#5-running-the-interactive-simulation)
6. [Training the Reinforcement Learning Agent](#6-training-the-reinforcement-learning-agent)
7. [Cloud Training on Google Colab](#7-cloud-training-on-google-colab)
8. [Repository Structure](#8-repository-structure)
9. [Git Workflow](#9-git-workflow)

---

## 1. Project Overview

This project builds an end-to-end pipeline for **autonomous robotic ultrasound scanning**:

1. **Patient Mesh Generation** — 3D CT volumes from the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) dataset are processed into watertight skin meshes (`.obj`) and bone segmentation labels (`.nii.gz`) using marching cubes.

2. **Generative U-Net** — A 2-channel U-Net (CT intensity slice + binary bone mask → simulated B-mode ultrasound image) is trained to synthesize realistic ultrasound frames in real-time during the simulation, without running a full wave equation solver.

3. **PyBullet Physics Simulation** — The Franka Panda arm, the ultrasound probe (modeled as a curved convex-array transducer), and the patient mesh are loaded into a PyBullet physics scene with gravity, contact forces, and a hospital bed environment.

4. **Coordinate Registration** — A mathematically exact affine transform pipeline maps the probe's contact position in PyBullet world coordinates to a precise voxel index inside the CT volume, accounting for NIfTI LPS coordinates, mesh centering offsets, body placement, and non-isotropic mesh scaling.

5. **OpenAI Gymnasium Environment** — The simulation is wrapped in a standard `gymnasium.Env` interface, making it directly compatible with Stable-Baselines3 RL algorithms.

6. **A2C Reinforcement Learning Agent** — An Advantage Actor-Critic (A2C) agent is trained to navigate the probe across the patient's skin surface to maintain good acoustic contact and locate bone structures.

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

Install all Python dependencies:

```bash
pip install torch torchvision pybullet gymnasium stable-baselines3[extra] \
            nibabel scikit-image scipy opencv-python trimesh tqdm
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

### 3.4 (Optional) Train the U-Net

If you have a GPU and want to retrain the CT→US translation model from scratch:

```bash
# Generate 2D paired training slices from the 3D TCGA patient volumes
python gen_data.py --subject TCGA-QQ-A8VG
python gen_data.py --subject TCGA-QQ-ASV2
python gen_data.py --subject TCGA-QQ-ASVC

# Train the 2-channel U-Net (CT + bone mask → simulated ultrasound)
python model/train.py \
    --data_root dataset \
    --output_dir model/runs/exp1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 2e-4
```

A pre-trained checkpoint is already included at `model/runs/exp1_2IP/exp1/best_model.pth` and is used by default.

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
# Use the default patient (s0058)
python live_unet_demo.py

# Use a different subject
python live_unet_demo.py --subject totalseg_patients/s0223

# Enable intensity histogram matching (reduces domain gap between CT and training data)
python live_unet_demo.py --match-histogram

# Skip U-Net and display the raw bone mask overlay
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

## 6. Training the Reinforcement Learning Agent

The RL agent uses **Advantage Actor-Critic (A2C)** from Stable-Baselines3, trained inside the `RoboticUltrasoundGymEnv` Gymnasium environment.

### 6.1 Run Training Locally

```bash
python train_a2c.py \
    --timesteps 100000 \
    --subject totalseg_patients/s0058 \
    --n-steps 20 \
    --lr 7e-4 \
    --save-freq 5000 \
    --tb-log ./a2c_tensorboard/ \
    --save-dir ./a2c_checkpoints/
```

By default, the environment runs in **Strategy 2 (mask-based)** mode (`skip_unet=True`), which bypasses U-Net inference entirely and uses the binary bone mask as the observation image. This achieves **~440 FPS** on a dedicated Colab instance versus ~12 FPS with U-Net enabled. The learned policy transfers naturally to U-Net outputs at evaluation time.

### 6.2 Monitor Training with TensorBoard

```bash
tensorboard --logdir ./a2c_tensorboard/
```

Navigate to `http://localhost:6006` to view episode reward, policy loss, and value loss curves.

### 6.3 RL Environment Details

**Action Space** — 6-DOF continuous, clipped to `[-1, 1]`:

| Index | Meaning | Scale Applied |
|---|---|---|
| 0–2 | `dx, dy, dz` (position delta) | ×0.01 m (±1 cm per step) |
| 3–5 | `droll, dpitch, dyaw` (orientation delta) | ×0.05 rad (±2.9° per step) |

**Observation Space** — Dictionary:

| Key | Shape | Type | Description |
|---|---|---|---|
| `"image"` | `(256, 256)` | `uint8` | Synthesized B-mode US image (or bone mask) |
| `"force"` | `(1,)` | `float32` | Estimated normal contact force in Newtons |
| `"pose"` | `(7,)` | `float32` | EE position (xyz) + orientation (quaternion xyzw) |

**Reward Function:**

| Component | Value | Condition |
|---|---|---|
| `R_f` (force reward) | `+1.0` | Good contact: 2 N ≤ F ≤ 8 N |
| `R_f` | `+0.5 × F` | Light contact: 0 < F < 2 N |
| `R_f` | `−1.0 × (F − 8)` | Over-pressure: F > 8 N |
| `R_f` | `−2.0` | No contact: F = 0 |
| `R_b` (bone reward) | `+1.5` | Bone pixels visible in segmentation slice |
| `R_a` (smoothness) | `−0.1 × ‖action‖²` | Always (penalizes jerk) |

**Termination Conditions:**

- `terminated = True` if contact force exceeds **12 N** (patient safety limit).
- `terminated = True` if no contact is maintained for **30 consecutive steps**.
- `truncated = True` after **200 steps** (episode horizon).

### 6.4 Verify the Environment

Before training, run the diagnostic script to confirm the environment initializes correctly and measure performance:

```bash
# Visual mode (opens PyBullet GUI and OpenCV window)
python test_gym_env.py

# Headless mode (faster, no display)
python test_gym_env.py --headless
```

This script runs 200 random-action steps, prints per-step force/reward, verifies observation shapes, and reports FPS.

---

## 7. Cloud Training on Google Colab

For users without a local GPU, see **[COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)** for a full step-by-step guide to:
1. Upload the `ct-us-colab.zip` project bundle to Google Drive.
2. Extract and install dependencies on a Colab T4 GPU instance.
3. Launch 500k-step A2C training (~1.9 hours at ~70 FPS).
4. Back up checkpoints and TensorBoard logs back to Google Drive.

---

## 8. Repository Structure

```
ct_us/
├── live_unet_demo.py          # Interactive PyBullet GUI + real-time U-Net inference
├── robotic_us_env.py          # Gymnasium environment (wraps PyBullet + slicing + U-Net)
├── train_a2c.py               # A2C RL training script (Stable-Baselines3)
├── test_gym_env.py            # Environment diagnostic & speed benchmark
├── extract_slice.py           # 2D oblique slicer: registration-aware trilinear interpolation
├── registration.py            # Affine coord transforms (PyBullet world ↔ CT voxel)
├── generate_patient_meshes.py # CT volume → skin mesh + bone label + registration metadata
├── download_totalseg.py       # Stream & cache TotalSegmentator subjects from Zenodo
├── gen_data.py                # Generate 2D paired CT/SimUS training slices from 3D volumes
├── prepare_dataset.py         # Dataset preparation utilities
├── example_integration.py     # Standalone usage examples for the Gym environment
│
├── model/                     # U-Net generative model
│   ├── model.py               # 5-level U-Net architecture (InstanceNorm, Sigmoid output)
│   ├── dataset.py             # CTSimUSDataset: paired loading + HU clipping + normalization
│   ├── train.py               # Training loop (AMP, cosine LR, checkpointing, validation plots)
│   ├── inference.py           # Batch inference + PSNR/SSIM evaluation
│   ├── train_ultrabones.py    # Training on the UltraBones100k ex-vivo dataset
│   ├── ultrabones_dataset.py  # Dataset loader for UltraBones100k format
│   ├── reference_ct_slice.npy # Reference histogram template for intensity matching
│   ├── NOTES.md               # Architecture decisions, hyperparameters, and rationale
│   ├── requirements.txt       # Model-specific Python dependencies
│   └── runs/
│       └── exp1_2IP/exp1/
│           └── best_model.pth # Pre-trained 2-channel U-Net checkpoint
│
├── totalseg_patients/         # Patient volume data (git-ignored)
│   └── s0058/                 # Active default subject
│       ├── ct.nii.gz
│       ├── bone_label.nii.gz
│       ├── patient_skin.obj
│       └── registration_meta.json
│
├── a2c_checkpoints/           # Saved RL agent checkpoints (git-ignored)
├── a2c_tensorboard/           # TensorBoard training logs (git-ignored)
├── research_registration/     # Archive of experimental registration research code
│
├── README.md                  # This file
├── PROJECT_DOCUMENTATION.md   # Detailed architecture & algorithm documentation
├── COLAB_INSTRUCTIONS.md      # Step-by-step guide for Google Colab training
├── agent.md                   # Session-to-session context tracker for AI agents
└── task.md                    # Task checklist and project roadmap
```

---

## 9. Git Workflow

### Pulling Updates (on training device)

Before starting a new session, sync any changes pushed from another device:

```bash
git pull origin main
```

### Pushing Updates (on development device)

After modifying code or configuration:

```bash
# 1. Stage all changes (large data files & caches are auto-excluded by .gitignore)
git add .

# 2. Commit with a descriptive message
git commit -m "Brief description of what changed"

# 3. Push to GitHub
git push origin main
```

> **Note:** The `.gitignore` excludes `totalseg_patients/`, `a2c_checkpoints/`, `a2c_tensorboard/`, `__pycache__/`, and `*.nii.gz` / `*.obj` files. Only code, scripts, and lightweight metadata are tracked.
