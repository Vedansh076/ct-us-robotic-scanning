# Stage 1 Roadmap Checklist

This checklist tracks the implementation of Stage 1 changes: standardizing normalization, fixing model imports, and integrating registration math into the main demo.

## Action Items

- [x] **1. Consolidate Coordinate Registration Math**
  - [x] Replace root `extract_slice.py` with the registration-aware version from `research_registration/extract_slice.py`
  - [x] Copy `registration.py` to root and integrate `compute_registered_ct_center` and `load_registration_meta` into `live_unet_demo.py`
  - [x] Enable loading of patient-specific `patient_skin.obj` mesh instead of the generic `mosh` body mesh by default

- [ ] **2. Correct U-Net Model Imports & Training**
  - [x] Modify `model/train.py` to import `UNet` from `model` (Sigmoid output) instead of `pix2pix.model` (Tanh output)
  - [x] Standardize clinical windowing/clipping range (e.g. $[-200, 300]$ HU) consistently across both `model/dataset.py` and `live_unet_demo.py`
  - [ ] Retrain the U-Net model using the corrected training script (Delayed: User has no GPU access right now)

- [x] **3. Update Simulator HUD & Control Features**
  - [x] Port arrow-key manual controls and gripper finger-locking mechanics to `live_unet_demo.py`
  - [x] Expose manual movement speed keys (`[` and `]`) and axis locking (`[X]`, `[Y]`, `[Z]` keys)
  - [x] Integrate B-mode scanning plane (orthogonal to skin) and longitudinal/transverse toggle (`[T]` key) with visual HUD indicator

- [ ] **4. Testing and Verification**
  - [ ] Verify that model predictions in the simulation loop are no longer dark/grey and map anatomically (Pending model retraining)
  - [x] Run offline quantitative evaluation (`--eval`) over all subject datasets and verify statistics (Successfully compiled and verified)
