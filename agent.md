# Project State & Context Tracker

This file preserves the active state, findings, and context of the CT-to-Ultrasound Robotic Scanning project to ensure seamless transitions between assistant sessions.

---

* **Current Focus:** **Stage 2 Completed (Histogram Matching)**. Ready for Stage 3 (OpenAI Gym Environment wrapping). Retraining is currently delayed due to lack of GPU access.

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

---

## 3. Instructions for Keeping Context Files Up-to-Date
To maintain consistency across sessions, every active agent must adhere to the following:
1. **Read on Startup:** Always check `agent.md` and `task.md` in the workspace root at the beginning of a session to understand the current progress.
2. **Update on Step Completion:** As items in `task.md` are started, marked in progress, or completed, edit `task.md` immediately to reflect the change.
3. **Record Findings:** If a new bug is found, or if a critical design decision is made, update Section 2 of `agent.md` to document the context.
4. **Transition Handover:** Before concluding a turn, update the "Current Focus" in `agent.md` with instructions for the next agent.
5. **Git Synchronization:** Commit and push verified code changes to the GitHub remote (`git push origin main`) to ensure changes are synced across devices.
