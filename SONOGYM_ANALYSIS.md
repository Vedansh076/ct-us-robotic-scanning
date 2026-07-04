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

### What the paper actually says (verbatim from arXiv:2507.01152, Section 4)

> *"For learning-based (LB) simulation, we train a generative model using the pix2pix framework to translate **CT slices** into ultrasound images, leveraging a large in-house CT-to-ultrasound paired dataset collected from **7 ex-vivo spine specimens** [wu2025ultrabones100k]."*

> *"We apply intensity histogram matching between the input CT slices and a subset of training CT images to mitigate the domain gap in learning-based simulation."*

**Annotation on the [wu2025ultrabones100k] citation:**  
This citation (reference [53] in the paper) points to the UltraBones100k paper. However, the sentence says **"our in-house... paired dataset"** — meaning it was collected by the same research group, following the same data collection methodology described in UltraBones100k. It is **NOT** the public UltraBones100k download. It is a **separate, private 7-specimen spine dataset** collected internally.

### Exact model input: 1-channel CT slice only

The body text of the paper specifies that the learning-based model translates only the CT slice:
- Verbatim from Section 4: *"...translate CT slices into ultrasound images..."*
- Verbatim from Section 4: *"...matching between the input CT slices and a subset of training CT images..."*
- Verbatim from Section 5.1: *"...despite the domain gap between the input CT slices and the training CT data."*

**Summary of inputs per mode based on the text:**
| Mode | Input to US simulator |
|------|-----------------------|
| Physics-based (MB) | **Label slice** (primary) + CT slice (to refine reflections) |
| Learning-based (LB, pix2pix) | **CT slice only** (1 channel) |

SonoGym's pix2pix takes **1-input (1-channel CT)**. The segmentation label map is not fed into their neural network.

### Network training details (verbatim from Appendix)

> *"Our pix2pix network adopts the deep U-Net architecture illustrated in Fig. 11. The model is implemented based on MONAI [5] and trained using a combination of L1 loss and GAN loss, with respective weights of 1 and 0.01. In total, we train **five separate networks** for 15–25 epochs on our training dataset. To improve generalization to unseen CT resolutions (such as those encountered in our simulation data), we apply data augmentation via random downsampling and upsampling."*

They train 5 networks (different random seeds) so that at RL training time, they can randomly sample among them to improve domain randomization.

### Data collection details (verbatim from Appendix)

> *"We follow the setup described in [53] to collect a dataset of paired CT-US images from **seven ex-vivo spine specimens**. Optical markers are attached to the sacrum of each specimen, and additional K-wires (2.5 mm in diameter, 150 mm in length; DePuy Synthes, USA) are used to stabilize each vertebra, avoiding bone movement during data acquisition. CT scans were acquired for each specimen with an image resolution of 512 × 512 pixels, an in-plane pixel spacing of 0.839 mm × 0.839 mm, and a slice thickness of 0.6 mm (NAEOTOM Alpha, Siemens, Germany). For ultrasound imaging, we used the Aixplorer Ultimate system equipped with an SL18-5 linear probe."*

### Does SonoGym have separate models for spine vs. lower limbs?
**No.** The entire SonoGym public release is spine-centric. There is no lower-limb model in the repository. The paper does not demonstrate lower-limb US simulation.

### How SonoGym handles domain gap

> *"Our LB approach maintains high visual quality despite the domain gap between the input CT slices and the training CT data."* (Section 5, Q1)

> *"...generalization across different patients is still challenging, especially with a limited diversity of patient models."* (Section 5, Q4)

**They partially address domain gap via histogram matching and multi-seed training, but acknowledge it remains a limitation.**


---

## 4. Comparison: SonoGym vs. Our Project

| Aspect | SonoGym | Our Project |
|--------|---------|-------------|
| **Simulator** | NVIDIA Isaac Lab (GPU) | PyBullet (CPU) |
| **Primary anatomy** | Lumbar spine (L1–L5) | Currently spine (TotalSegmentator s0058); **want to be lower limb** |
| **Mesh source** | TotalSegmentator → OBJ/USD | TotalSegmentator → OBJ (same tool!) |
| **CT-US model input** | **1-channel (CT only)** | **2-channel (CT + Label)** |
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

## 8. Comparison of Model Input — Setting the Record Straight

The body text of the paper specifies that the learning-based model translates only the CT slice:
- Verbatim from Section 4: *"...translate CT slices into ultrasound images..."*

Therefore:
- **SonoGym's pix2pix is 1-channel CT -> ultrasound**.
- **Our project uses 2-channel (CT + bone segmentation mask) -> ultrasound**.

This is a key architectural difference:
- SonoGym relies on the pix2pix model to implicitly infer bone boundaries and shadow locations from the CT slice alone.
- Our U-Net receives the bone mask explicitly as channel 2, giving the network a direct, noise-free geometric cue of the bone surface. This enables sharper reflection rendering and exact acoustic shadowing.

### What we can accurately claim:
1. **Anatomy Domain:** SonoGym covers spine only. Our project targets lower limbs (using UltraBones100k). These are anatomically distinct domains.
2. **Model Input:** SonoGym uses 1-channel CT input for its learning-based simulator. Our project uses 2-channel input (CT + bone mask) to explicitly guide bone surface and shadow rendering.
3. **Training Data Origin:**
   * **SonoGym:** Verbatim from Section 4: *"...leveraging a large in-house CT-to-ultrasound paired dataset collected from 7 ex-vivo spine specimens [56]."* (Private spine dataset).
   * **Our Project:** Public **UltraBones100k** dataset (Lower limbs: tibia, fibula, foot).
4. **Platform Accessibility:** SonoGym requires NVIDIA Isaac Lab + GPU. Our system runs on PyBullet + CPU.

---

## 9. References

1. Ao, Y. et al. "SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound." NeurIPS 2025 (D&B Track). **arXiv:2507.01152.** https://github.com/SonoGym/SonoGym
2. SonoGym HuggingFace: https://huggingface.co/datasets/yunkao/SonoGym_assets_models
3. Wasserthal, J. et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." Radiology: Artificial Intelligence, 2023. https://github.com/wasserth/TotalSegmentator
4. UltraBones100k (arXiv:2502.03783): Wu et al., 2025. Ex-vivo lower limb paired CT-US, 14 cadavers, 100k images. CC BY 4.0.
5. Our project GitHub: https://github.com/Vedansh076/ct-us-robotic-scanning

