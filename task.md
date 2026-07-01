# Project Roadmap Checklist

This checklist tracks the implementation of Stage 1 (Alignment & Registration) and Stage 2 (Generative Realism) changes.

## Action Items

- [x] **1. Consolidate Coordinate Registration Math**
  - [x] Replace root `extract_slice.py` with the registration-aware version from `research_registration/extract_slice.py`
  - [x] Copy `registration.py` to root and integrate `compute_registered_ct_center` and `load_registration_meta` into `live_unet_demo.py`
  - [x] Enable loading of patient-specific `patient_skin.obj` mesh instead of the generic `mosh` body mesh by default

- [ ] **2. Correct U-Net Model & Transition to 2-Channel Input**
  - [x] Modify `model/train.py` to import `UNet` from `model` (Sigmoid output) instead of `pix2pix.model` (Tanh output)
  - [x] Standardize clinical windowing/clipping range (e.g. $[-200, 300]$ HU) consistently across both `model/dataset.py` and `live_unet_demo.py`
  - [ ] Adapt U-Net/Pix2Pix architectures in `model/model.py` and `model/pix2pix/model.py` to take 2-channel input (CT + Seg)
  - [ ] Update `model/dataset.py` to load, normalize, and stack both CT and label slices
  - [ ] Update `live_unet_demo.py` and `extract_slice.py` to slice and stack label volumes during simulation
  - [ ] Retrain the U-Net model using the corrected training script (Delayed: User has no GPU access right now)

- [x] **3. Update Simulator HUD & Control Features**
  - [x] Port arrow-key manual controls and gripper finger-locking mechanics to `live_unet_demo.py`
  - [x] Expose manual movement speed keys (`[` and `]`) and axis locking (`[X]`, `[Y]`, `[Z]` keys)
  - [x] Integrate B-mode scanning plane (orthogonal to skin) and longitudinal/transverse toggle (`[T]` key) with visual HUD indicator

- [x] **4. Testing and Verification**
  - [x] Verify that model predictions in the simulation loop load and execute with the new 2-channel checkpoint (Successfully loaded and verified)
  - [x] Run offline quantitative evaluation (`--eval`) over all subject datasets and verify statistics (Successfully compiled and verified)

- [x] **5. Replicate Curved Probe & Fingerless Mount (Visual Alignment)**
  - [x] Hide gripper fingers permanently at startup (keeping hand base visible)
  - [x] Redesign probe components using a horizontal cylinder for a curved wedge footprint (no cube at end)
  - [x] Extend probe height Z range to 0.240 m to align and mount it directly into the hand base without gaps
  - [x] Change patient mesh visual color to a smooth pinkish/flesh skin tone

## Stage 2: Generative Realism (Histogram Matching)

- [x] **1. Implement Intensity Histogram Matching**
  - [x] Extract and save a representative reference CT slice to `model/reference_ct_slice.npy`
  - [x] Expose `--match-histogram` command-line argument in `live_unet_demo.py`
  - [x] Integrate `skimage.exposure.match_histograms` in the inference path to align incoming CT slices with the training distribution
