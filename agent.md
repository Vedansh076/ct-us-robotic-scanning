# Project State & Context Tracker

This file preserves the active state, findings, and context of the CT-to-Ultrasound Robotic Scanning project to ensure seamless transitions between assistant sessions.

---

* **Current Focus:** **Project Handover / Archival.** All stages (0–4) are complete. The project has **6 trained autonomous scanning algorithms** (A2C +264.1, SAC +317.0, PPO -237.0, BC loss 2.0, GAIL, ACT +261.01), **4 ultrasound synthesis modes** (unet, pix2pix, conv, ray), and comprehensive documentation (README.md, DEVELOPER_GUIDE.md, PROJECT_DOCUMENTATION.md, commands.md). The original developer is no longer actively working on the project. Future agents should read all documentation files before making changes. See `.agents/AGENTS.md` for critical invariants and testing requirements.

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

### Bug 5: Attenuation Unit Mismatch [RESOLVED]
* **Code Location:** `ModelBasedUSSimulator._compute_attenuation()` inside `live_unet_demo.py`.
* **Consequence:** The pixel size parameter `e` was passed in meters (`1.5e-4`), but `alpha` was specified in `dB/cm/MHz`. This unit mismatch (meters vs. centimeters) under-calculated the actual physical attenuation by 100×. Additionally, the decay exponent lacked a conversion from Decibels (dB) to Nepers (multiplication by `0.1151`). As a result, the sound wave passed through bone with virtually zero attenuation, making the vertebrae look fully transparent/bright inside and creating zero acoustic shadowing below the bone, rendering the ultrasound identical to the raw CT slice.
* **Resolution:** Converted `e` to centimeters (`e * 100.0`) and applied the `0.1151` Neper multiplier inside the exponential term. The bone boundary now registers as a highly reflective hyperechoic curve, and the region underneath it is realistically shadowed (attenuation shadow ratio dropped from 0.53 to 0.17).

### Bug 6: Salt-and-Pepper Speckle Texture [RESOLVED]
* **Code Location:** `ModelBasedUSSimulator.simulate()` step 7 (backscatter speckle) inside `live_unet_demo.py`.
* **Consequence:** The original speckle model used thresholded Gaussian noise (`S_map[T1 > mu1] = 0.0` with `mu1=0.20`), which zeroed out ~42% of pixels, creating a coarse salt-and-pepper dot pattern instead of smooth clinical-grade B-mode texture. Additionally, PSF_B was too small (2×2 sigma, 11×11 kernel) to produce realistic speckle grain size, the dynamic range was too harsh (50 dB), and there was no lateral coherence blur.
* **Resolution:** Replaced with Rayleigh-distributed speckle `sqrt(re² + im²)` where `re,im ~ N(0,1)` — the correct physics model for ultrasound backscatter envelopes. This produces always-positive values (no zero-holes) with smooth granular texture. Also: enlarged PSF_B to 3.5×3.5 sigma on 15×15 kernel, reduced DR to 38 dB (clinical MSK setting), added lateral coherence blur (σ=1.8 px, horizontal-only) to mimic beam-width limited resolution, and calibrated tissue mu0 (0.055 soft / 0.035 fat / 0.070 skin) for proper bone-tissue-shadow brightness separation.

### Bug 8: Soft Contact Spring vs Penetration Penalty Mismatch [RESOLVED]
* **Code Location:** `robotic_us_env.py` (`_compute_reward` and force calculation).
* **Consequence:** The initial force model used a soft virtual spring (`k=200 N/m`, standoff=8mm). Reaching the ideal 2–6N force window required compressing 2–22mm into the torso mesh. However, the reward function penalized any depth below 5mm, creating a lose-lose contradiction that caused the agent to embed deep into the body to harvest bone rewards.
* **Resolution:** Increased spring constant to `k=800 N/m` (3mm standoff) so ideal 2–6N contact is achieved at surface level (0–5mm penetration). Tightened orientation clamps to ±0.15 rad (±8.6°) to keep the probe strictly upright/perpendicular to the skin.

### Bug 9: Sparse Span Reward vs Dense Sweep Motion Reward [RESOLVED]
* **Code Location:** `robotic_us_env.py` (`_compute_reward`).
* **Consequence:** A one-time reward spike for expanding min/max visited Y span broke A2C value estimation (`explained_variance` dropped to ~0) and caused low-reward oscillations.
* **Resolution:** Replaced sparse span rewards with a dense longitudinal sweep reward (`R_sweep = 0.5 * |action_y|` when in 2–8N contact over bone), added inter-step action jerk penalty `R_jerk = -0.1 * ||action - last_action||²`, and randomized initial Y start positions (`±8 cm`) with a raycast fallback to ensure 100% valid skin contact at episode reset.

### Bug 8: PCA Rotation Candidate Centering Offset [RESOLVED]
* **Code Location:** `model/prepare_cavalcanti.py` in 90-degree PCA candidate pre-alignment.
* **Consequence:** The 90-degree rotated PCA candidate was rotated around the local translation origin `T_init[:3, 3]` instead of the actual centroid of the subsampled STL vertices `tc`. This shifted the rotated candidate's search center by up to 2.5 meters away from the patient torso, causing ICP to fail to align the rotated sweeps.
* **Resolution:** Corrected the centering shift in `prepare_cavalcanti.py` by applying rotation relative to the target centroid `tc`: `T_rot[:3, 3] = R90_3x3 @ (T_init[:3, 3] - tc) + tc`.

### Bug 9: Sagittal Sweep Sideways Orientation Penalty [RESOLVED]
* **Code Location:** `model/prepare_cavalcanti.py` candidate orientation safeguard.
* **Consequence:** The previous transverse constraint `abs(probe_x[0]) < 0.4` was penalizing candidates where the probe's width was aligned longitudinally. However, in the Cavalcanti dataset, sweeps are captured in different directions—R1 is a sagittal sweep (meaning the probe width is aligned longitudinally, so `abs(probe_x[0])` is close to 0). This caused the correct candidates to be penalized, forcing ICP to select wrong or out-of-bounds candidates for URS02-07, resulting in black/rotated slices.
* **Resolution:** Removed the strict `abs(probe_x[0]) < 0.4` sideways check since the fixed rotation center prevents sideways alignment drift naturally. Kept only the universal ceiling-pointing check (`probe_z[1] > 0`).


### Decision 1: 2-Channel Semantic-Guided Model (CT + Seg -> US)
* **Architecture:** Transitioning the input pipeline from 1-channel raw CT to 2-channels (Channel 1: CT, Channel 2: binary bone segmentation mask). This ensures perfect, sharp acoustic shadowing and specular bone boundary reflections.
* **Generative Training:** We will train/fine-tune the model on the **Cavalcanti et al. Robotic Lumbar Spine dataset** (and/or simulated lumbar spine pairs) to align with our clinical focus on lumbar spine navigation, using UltraBones100k only as a pre-training/secondary baseline.
* **Simulator Anatomy:** We will use **CT spine data** (specifically the TotalSegmentator patient lumbar vertebrae) to generate patient meshes inside PyBullet, utilizing our registration-aware slicing and histogram matching.

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
