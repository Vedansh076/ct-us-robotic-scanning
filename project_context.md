# CT → Ultrasound Robot Scanning Project

## Project Goal

Create a robotic ultrasound simulation where:

Robot
→ Probe
→ Raycast into patient
→ Determine CT slice location
→ Extract CT slice
→ Run trained CT→US model
→ Display predicted ultrasound in real time

The goal is not physics-based ultrasound simulation.

The goal is neural CT→Ultrasound synthesis.

---

# Current Status

## Dataset Generation

Files:

* gen_data.py
* extract_slice.py

Dataset generation:

1. Load CT.nii
2. Load SimUS.nii
3. Generate random center
4. Generate random quaternion
5. Extract CT slice
6. Extract SimUS slice
7. Save paired samples

Output:

dataset/
├── ct/
├── simus/
└── poses/

---

## Slice Extraction

File:

extract_slice.py

Function:

extract_slice(
volume,
center,
quaternion,
spacing,
size,
pixel_spacing
)

Uses:

* scipy Rotation
* map_coordinates

Quaternion determines slice orientation.

Center determines slice location.

---

## Trained Models

### U-Net

Status:

Working.

Used for live simulation.

### Pix2Pix

Status:

Training completed.

Need integration later.

Model structure:

model/pix2pix/

contains:

* model.py
* discriminator.py
* train_pix2pix.py
* inference_pix2pix.py

---

# Live Simulation

Main file:

live_unet_demo.py

Current pipeline:

Probe Pose
→ Raycast
→ Hit Point
→ CT Center
→ extract_slice()
→ U-Net
→ Predicted Ultrasound

---

# PyBullet Scene

Current scene contains:

* Franka Panda robot
* Hospital bed
* Human mesh
* Ultrasound probe
* Raycasting beam

Human mesh:

mosh_cmu_0511_f_lbs_10_207_0_v1.0.2.obj

---

# Raycasting

Current implementation:

Probe
→ cast ray
→ hit body mesh
→ compute hit point

Hit point drives CT slice center.

Current mapping is approximate.

Example:

relative_position_on_mesh
→ CT voxel center

Body mesh is NOT registered to CT.

---

# Important Constraint

Do NOT redesign architecture.

Keep:

Probe
→ Raycast
→ CT Slice
→ Neural Model

This architecture is required.

---

# Current Problems

## Problem 1

Generated ultrasound is worse than training examples.

Possible causes:

* distribution mismatch
* normalization mismatch
* limited probe orientation variation
* crude body-to-CT mapping

---

## Problem 2

Simulation realism.

Need:

* probe face contacting body
* realistic ultrasound scanning posture
* better probe geometry

---

## Problem 3

Performance.

Current FPS is low.

Need profiling.

---

# Future Work

Priority order:

1. Improve probe realism
2. Improve probe scanning trajectory
3. Add Pix2Pix inference option
4. Improve CT↔body mapping
5. Increase FPS

---

# Files That Matter Most

live_unet_demo.py
extract_slice.py
gen_data.py
model/model.py
model/inference.py

These files contain the core project logic.

Ignore old experimental folders unless needed.
