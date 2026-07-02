# Project State & Context Tracker

This file preserves the active state, findings, and context of the CT-to-Ultrasound Robotic Scanning project to ensure seamless transitions between assistant sessions.

---

* **Current Focus:** 2-channel semantic-guided (CT + Seg -> US) pipeline successfully integrated and verified with new `exp1_2IP` checkpoint. Ready to construct the local **Stage 3 OpenAI Gym Environment wrapper** in PyBullet.

---

## 2. Key Findings & Technical Bugs

### Bug 1: U-Net Activation Mismatch
* **Code Location:** `model/train.py` imports `UNet` from `pix2pix.model` (which uses `Tanh` activation, outputting to $[-1, 1]$), but trains it against target labels in $[0, 1]$ from `dataset.py`.
* **Inference Location:** `live_unet_demo.py` loads this checkpoint as `UNetOriginal` (which uses `Sigmoid` activation, outputting to $[0, 1]$).
* **Consequence:** Background voxels (trained as $\approx 0.0$ logits for `Tanh` output of $0.0$) map to $\operatorname{sigmoid}(0.0) = 0.5$ (grey) in the live demo. This produces low-contrast, near-black, or grey-filled ultrasound predictions.

### Bug 2: Normalization & Clinical Windowing Mismatches
* **Training:** `dataset.py` clips CT values to $[-150, 1250]$ HU and scales them to $[-1, 1]$.
* **Live Demo:** `live_unet_demo.py` clips CT values to $[-200, 300]$ HU and scales them to $[0, 1]$ (for U-Net) or $[-1, 1]$ (for Pix2Pix).
* **Consequence:** Out-of-distribution inputs lead to prediction degradation during simulation.

### Bug 3: Unintegrated Registration Math [RESOLVED]
* **Resolution:** Exact registration coordinates and registration-aware slice extraction functions from `research_registration/` have been promoted to the root files `live_unet_demo.py`, `registration.py`, and `extract_slice.py`. The simulation now loads patient-specific skin meshes (`patient_skin.obj` and `registration_meta.json`) by default.

### Bug 4: Raycast Occlusion & Hollow Torso Meshes [RESOLVED]
* **Raycast Occlusion:** The vertical snap raycasts were blocked by the robot arm/hand geometry, causing the probe to fail to snap and get buried inside the torso mesh, resulting in constant "hit: miss" status. Resolved by implementing a multi-step `raycast_skin_surface` function that ignores robot links, and offsetting `raycast_probe` 5 cm above the tip to handle penetration/compression.
* **Hollow Torso Meshes:** Torso meshes generated from CT volumes were open shells at the borders and had inverted normals that made them transparent under PyBullet back-face culling. Resolved by:
  1. Boundary-sealing and zero-padding the binary body volume before marching cubes in `generate_patient_meshes.py` to produce mathematically closed, solid manifold torso meshes.
  2. Inverting the triangle winding order (`faces[:, [0, 2, 1]]`) to orient all normals outwards, making the outer skin and flat caps fully opaque and visible.

### Decision 1: 2-Channel Semantic-Guided Model (CT + Seg -> US)
* **Architecture:** Transitioning the input pipeline from 1-channel raw CT to 2-channels (Channel 1: CT, Channel 2: binary bone segmentation mask). This ensures perfect, sharp acoustic shadowing and specular bone boundary reflections.
* **Generative Training:** We will train the model on the **UltraBones100k** dataset (ex-vivo rigid registration) to achieve maximum B-mode realism.
* **Simulator Anatomy:** We will use **CT-only data** (e.g. TotalSegmentator) to generate patient meshes inside PyBullet, utilizing our registration-aware slicing and histogram matching.

### Decision 2: Curved Clinical Probe & Flange Mount (Visual Design)
* **Visuals:** Redesigned the probe shape to look like a real curved/convex abdominal array. Replaced the boxy lower wedge with a horizontally oriented cylinder along the X-axis (radius `0.015` m, length `0.056` m). This provides a smooth, rounded scanning footprint that tapers cleanly into the cylinder handle with zero sharp boxy corners.
* **Mounting:** Permanently hid the robot's gripper fingers (links 9 and 10) but kept the hand base (link 8) visible. Extended the probe height profile to `Z = 0.240` m (shifting offsets upwards) to insert the probe's top mount directly into the hand base, replicating the fingerless end-effector layout of SonoGym with no visual gaps.

---

## 3. Instructions for Keeping Context Files Up-to-Date
To maintain consistency across sessions, every active agent must adhere to the following:
1. **Read on Startup:** Always check `agent.md` and `task.md` in the workspace root at the beginning of a session to understand the current progress.
2. **Update on Step Completion:** As items in `task.md` are started, marked in progress, or completed, edit `task.md` immediately to reflect the change.
3. **Record Findings:** If a new bug is found, or if a critical design decision is made, update Section 2 of `agent.md` to document the context.
4. **Transition Handover:** Before concluding a turn, update the "Current Focus" in `agent.md` with instructions for the next agent.
5. **Git Synchronization:** Commit and push verified code changes to the GitHub remote (`git push origin main`) to ensure changes are synced across devices.
