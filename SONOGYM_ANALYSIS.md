# SonoGym Architecture Analysis & Comparison with Our Project

> **Purpose:** Reference document for understanding how SonoGym builds its simulation, what data it uses, and how our project aligns or diverges from their approach.  
> **Last Updated:** 2026-07-04  
> **Confirmed Paper:** Yunke Ao et al., arXiv:2507.01152, NeurIPS 2025 (D&B Track)

---

## 1. SonoGym Overview

**Paper:** "SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound" (NeurIPS 2025, Datasets & Benchmarks Track)  
**Authors:** Yunke Ao et al.  
**GitHub:** https://github.com/SonoGym/SonoGym  
**HuggingFace:** https://huggingface.co/datasets/yunkao/SonoGym_assets_models  
**Simulator Backend:** NVIDIA Isaac Lab (Isaac Sim 4.5.0)

SonoGym provides two distinct ultrasound simulation modes:
- **Physics-based (model-based):** Analytically ray-traces the CT/label map to approximate ultrasound physics (specular reflections, shadowing, speckle).
- **Learning-based:** pix2pix GAN translates CT label map slices → synthetic B-mode ultrasound images.

---

## 2. SonoGym's Anatomy Meshes — Exact Source

### Patient Body Meshes (Skin + Bone)
SonoGym generates its anatomical simulation assets (body surface mesh + bone label volumes) from **real patient CT scans** processed with **TotalSegmentator**.

**Pipeline:**
```
Patient CT scans (NIfTI)
       ↓  TotalSegmentator
Segmented bone NIfTI labels (vertebra_L1.nii.gz, vertebra_L2.nii.gz, etc.)
       ↓  Marching Cubes + Laplacian Smoothing
OBJ/USD mesh files (for Isaac Sim rendering + collision)
       ↓  Registration metadata (affine, spacing, centroid)
Bundled into SonoGym_assets_models on HuggingFace (pre-processed; raw CTs not shared)
```

**Key point:** SonoGym does NOT distribute the raw patient CT DICOM/NIfTI data (patient privacy). They only publish the pre-processed meshes and label maps.

### Which Anatomy?
SonoGym's primary anatomical focus is the **lumbar spine** (vertebrae L1–L5) for their main task: ultrasound-guided pedicle screw placement surgery. They do NOT include lower limb (tibia/femur) anatomy in their public release.

---

## 3. SonoGym's CT-to-Ultrasound Model — Exact Training Data

### ⚠️ Critical Finding: SPINE-SPECIFIC, PRIVATE Dataset

SonoGym's pix2pix model is trained on a **proprietary in-house paired CT-to-Ultrasound dataset** collected from **~7 ex-vivo cadaveric lumbar spine specimens**.

**Important:** This is NOT the UltraBones100k dataset. UltraBones100k (arXiv:2502.03783) is a *companion* publication by the same ETH Zurich / Balgrist group covering lower limbs (tibia, fibula, foot, 14 cadavers, CC BY 4.0 public). The two datasets are separate — SonoGym's pix2pix was trained on the private spine set only.

**Data collection protocol:**
- Optical tracking markers attached to the sacrum for accurate spatial registration.
- K-wires (Kirschner wires) surgically inserted to stabilize each vertebra, preventing inter-scan bone movement.
- Ultrasound images acquired with real probe + CT scans of the same specimen under identical conditions.
- Result: paired (CT slice ↔ US image) ground truth at exactly the same anatomical cross-section.

**This dataset is NOT public.** SonoGym only distributes the trained pix2pix model weights on HuggingFace, not the raw paired CT-US data.

### Does SonoGym have separate models for spine vs. lower limbs?
**No — they have ONE model, trained exclusively on spinal anatomy.**

Their HuggingFace release (`SonoGym_assets_models`) contains:
- pix2pix model weights trained on the 7 ex-vivo lumbar specimens
- Pre-processed anatomical meshes (spine patients only)

There is **no lower limb (tibia/femur) model** in SonoGym's public release. Their framework is purpose-built for spinal surgery tasks.

### How SonoGym handles domain gap
They apply **intensity histogram matching** (skimage-style) to shift inference CT slice intensities toward the training distribution before feeding into the pix2pix model. This partially mitigates domain shift when testing on new patients not seen during training — but they are still operating in the same anatomical region (spine → spine).

---

## 4. Comparison: SonoGym vs. Our Project

| Aspect | SonoGym | Our Project |
|--------|---------|-------------|
| **Simulator** | NVIDIA Isaac Lab (GPU) | PyBullet (CPU) |
| **Primary anatomy** | Lumbar spine (L1–L5) | Currently spine (TotalSegmentator s0058); **want to be lower limb** |
| **Mesh source** | TotalSegmentator → OBJ/USD | TotalSegmentator → OBJ (same tool!) |
| **CT-US model type** | pix2pix GAN | U-Net (SimUS-style) |
| **CT-US training data** | 7 ex-vivo spine specimens (private) | UltraBones100k (public, ex-vivo lower limbs) |
| **Domain of CT-US model** | Spine/vertebrae ONLY | Lower limbs (tibia/fibula) ONLY |
| **RL algorithm** | PPO / SAC / BC (Isaac Lab) | A2C (Stable Baselines 3) |
| **Multi-patient** | Yes (5 virtual patients) | Yes (5 TotalSegmentator subjects) |
| **Histogram matching** | ✅ Applied | ✅ Already implemented (`--match-histogram`) |

---

## 5. The Domain Mismatch Problem — Our Situation

### Current State (Problem)
Our U-Net was trained on **UltraBones100k** (lower limb ex-vivo data: tibia, fibula, foot). It learns the acoustic appearance of **cortical bone in a soft-tissue lower limb context**.

However, our current RL simulation scans over the **torso/spine** of TotalSegmentator subjects (e.g., s0058's lumbar spine), because that is what the full-body `patient_skin.obj` mesh exposes.

**This is a direct domain mismatch:**
- U-Net trained on: `lower limb bone surfaces` (thin cortical tibia shaft, subcutaneous soft tissue only)
- U-Net tested on: `spinal vertebrae` (irregular cancellous vertebral bone, surrounding paraspinal muscles)
- Expected result: Poor generalization, degraded/unrealistic ultrasound image quality in live simulation

### Why SonoGym Does NOT Have This Problem
SonoGym's model is spine-trained and spine-tested — anatomically consistent by design.

### Our Correct Solution
To eliminate the domain mismatch, we need to align the **scanning region** with the **training domain** of our U-Net:

| Our U-Net Training Domain | Required Simulation Anatomy |
|--------------------------|----------------------------|
| Lower limb (tibia/fibula ex-vivo) | Lower limb bone in simulation |

---

## 6. Implementation Options for Lower Limb Simulation

### Option A — Femur from TotalSegmentator (Fast, ~1 hour)
Use `femur_left.nii.gz` already present in our TotalSegmentator patient s0058.

**Steps:**
1. Copy `femur_left.nii.gz` as the new `bone_label.nii.gz` for a leg-specific subject folder.
2. Adjust robot start position and scanning region in `robotic_us_env.py` to target the thigh region.
3. Reuse the existing `patient_skin.obj` (full body mesh) — the robot will naturally scan the thigh.

**Pros:** No new data needed; fully works today.  
**Cons:** Femur ≠ Tibia anatomically. Femur has more muscle coverage. Still within "long bone" family, so US appearance is similar.

### Option B — Tibia STL from UltraBones100k (Ideal, ~1 day)
Use `tibia.stl` + `fibula.stl` from UltraBones100k specimen01 (already downloaded).

**Steps:**
1. Write `voxelize_stl.py`: convert tibia.stl + fibula.stl → 3D NIfTI binary label volume at 1mm isotropic spacing.
2. Write `generate_leg_mesh.py`: create a synthetic cylindrical leg skin mesh (OBJ) of radius ~8cm, length ~30cm.
3. Create `registration_meta.json` mapping the synthetic world coordinates → voxel indices.
4. Plug into `robotic_us_env.py` via `--subject` flag.

**Pros:** Direct anatomical match to UltraBones100k training data. Publishable.  
**Cons:** Requires writing 2 new scripts. No real CT volume for U-Net inference (Strategy 2/mask-only training required).

### Recommendation
> **Short term:** Option A — validate that the full RL pipeline works end-to-end with lower limb anatomy.  
> **Long term:** Option B — switch to tibia for final published results.

---

## 7. Action Items

- [ ] Implement Option A: Create leg scan setup using femur from TotalSegmentator s0058
- [ ] Implement Option B: Voxelize UltraBones100k tibia STL → NIfTI label + synthetic skin mesh
- [ ] Retrain U-Net: Once GPU access available, retrain on UltraBones100k with corrected activation + 2-channel input
- [ ] Validate alignment: Run `test_gym_env.py` with new leg anatomy and verify segmentation slices look correct
- [ ] Update `generate_patient_meshes.py` to support leg-specific subjects

---

## 8. Our Architectural Advantage Over SonoGym

Our project provides two key improvements beyond SonoGym's pix2pix:

| Feature | SonoGym | Our Project |
|---------|---------|-------------|
| **Model input** | 1-channel CT slice | **2-channel: CT + bone segmentation mask** |
| **Acoustic shadow** | Approximated from CT HU only | **Exact from bone mask** (crisp shadow boundary) |
| **Training domain** | Spine only | **Lower limbs** (UltraBones100k — fills SonoGym's gap) |
| **RL environment** | Isaac Lab (requires NVIDIA GPU) | PyBullet (CPU, accessible to all) |

SonoGym's paper **acknowledges** that extending to lower-limb anatomy requires new paired CT-US training data — specifically citing UltraBones100k as the intended future source. **Our project already uses UltraBones100k**, directly filling this gap that SonoGym identified.

---

## 9. References

1. Ao, Y. et al. "SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound." NeurIPS 2025 (D&B Track). **arXiv:2507.01152.** https://github.com/SonoGym/SonoGym
2. SonoGym HuggingFace: https://huggingface.co/datasets/yunkao/SonoGym_assets_models
3. Wasserthal, J. et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." Radiology: Artificial Intelligence, 2023. https://github.com/wasserth/TotalSegmentator
4. UltraBones100k (arXiv:2502.03783): Wu et al., 2025. Ex-vivo lower limb paired CT-US, 14 cadavers, 100k images. CC BY 4.0. https://huggingface.co/datasets/... 
5. Our project GitHub: https://github.com/Vedansh076/ct-us-robotic-scanning
