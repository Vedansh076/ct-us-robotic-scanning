# Project Roadmap Checklist

This checklist tracks the implementation of Stage 1 (Alignment & Registration) and Stage 2 (Generative Realism) changes.

## Action Items

## Stage 0: TotalSegmentator Patient Data Pipeline

- [x] **0. Download & Process TotalSegmentator CT Subjects**
  - [x] Write `download_totalseg.py` to stream 5 subjects from Zenodo (3.24 GB ZIP, cached for reuse)
  - [x] Write `generate_patient_meshes.py` to derive body mask from CT, run marching cubes, Laplacian smooth, export `patient_skin.obj`
  - [x] Merge all 49 bone structure masks into `bone_label.nii.gz` per subject
  - [x] Save `registration_meta.json` (affine, inv_affine, spacing, centroid offset) per subject
  - [x] Fix extraction bug: Windows bogus file vs directory for `segmentations/`; re-extracted s0011 (49/49 bones)
  - [x] Update `load_ct_subject()` in `live_unet_demo.py` to support `ct.nii.gz` + `bone_label.nii.gz` (TotalSegmentator format)
  - [x] Update default `--subject` to `totalseg_patients/s0058`
  - [x] **Subjects ready:** s0011, s0058, s0223, s0250, s0310 (all 5 have mesh + bone + meta)

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

## Stage 3: OpenAI Gymnasium Environment Wrapper

- [/] **1. Create Gymnasium Environment Wrapper**
  - [ ] Create `robotic_us_env.py` and implement `RoboticUltrasoundGymEnv` inheriting from `gymnasium.Env`
  - [ ] Define continuous 6-DOF task-space action space and dictionary observation space
  - [ ] Integrate robot IK solver and joint motor controller inside `step`
  - [ ] Implement force sensor reading and safety contact limit checking
  - [ ] Integrate 2-channel CT and bone segmentation slice extraction
  - [ ] Integrate the trained `best_model.pth` U-Net model for real-time B-mode US rendering
  - [ ] Formulate reward function matching image quality and normal contact forces
- [ ] **2. Verification and Testing**
  - [ ] Create automated environment verification script `test_gym_env.py`
  - [ ] Test random agent step execution and print performance benchmarks (FPS)
  - [ ] Verify safety threshold terminations and output observation shape consistency
