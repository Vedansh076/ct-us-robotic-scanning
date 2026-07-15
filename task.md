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
  - [x] **Bug fix:** Pad binary body volume with 0s before marching cubes to guarantee completely closed skin meshes (no hollow open shells)
  - [x] **Bug fix:** Invert mesh faces winding order to resolve back-face culling transparency (hollow look)
  - [x] **Bug fix:** Implement multi-step `raycast_skin_surface` to bypass robot geometry and avoid probe occlusion/misses
  - [x] **Bug fix:** Offset `raycast_probe` 5 cm above probe tip to handle skin penetration/compression cleanly
  - [x] **Subjects ready:** s0011, s0058, s0223, s0250, s0310 (all 5 have closed mesh + bone + meta)


- [x] **1. Consolidate Coordinate Registration Math**
  - [x] Replace root `extract_slice.py` with the registration-aware version from `research_registration/extract_slice.py`
  - [x] Copy `registration.py` to root and integrate `compute_registered_ct_center` and `load_registration_meta` into `live_unet_demo.py`
  - [x] Enable loading of patient-specific `patient_skin.obj` mesh instead of the generic `mosh` body mesh by default

- [x] **2. Correct U-Net Model & Transition to 2-Channel Input**
  - [x] Modify `model/train.py` to import `UNet` from `model` (Sigmoid output) instead of `pix2pix.model` (Tanh output)
  - [x] Standardize clinical windowing/clipping range (e.g. $[-200, 300]$ HU) consistently across both `model/dataset.py` and `live_unet_demo.py`
  - [x] Adapt U-Net/Pix2Pix architectures in `model/model.py` and `model/pix2pix/model.py` to take 2-channel input (CT + Seg)
  - [x] Update `model/dataset.py` to load, normalize, and stack both CT and label slices
  - [x] Update `live_unet_demo.py` and `extract_slice.py` to slice and stack label volumes during simulation
  - [x] Retrain the U-Net/Pix2Pix model on the Cavalcanti Spine Dataset using the remote GPU machine
    - [x] Set up Conda environment and Git repository on remote GPU machine
    - [x] Resolve file permissions and free up 71 GB disk space on GPU machine
    - [x] Download Cavalcanti dataset (31.3 GB, completed)
    - [x] Extract dataset (unzip complete, all 63 volunteers)
    - [x] Write `prepare_cavalcanti.py` preprocessing script (full 3D oblique reslicing)
    - [x] Update `train.py` and `train_pix2pix.py` with `--train_subjects auto` CLI support
    - [x] Run `prepare_cavalcanti.py --discover` to verify dataset structure on GPU machine
    - [x] Run `prepare_cavalcanti.py` full preprocessing (ICP registration + oblique reslicing)
    - [x] Train 2-channel U-Net on Cavalcanti processed data
    - [x] Train Pix2Pix on Cavalcanti processed data
    - [x] Copy trained checkpoint back to local workspace

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

## Stage 4: Model-Based (Physics/Convolution) Ultrasound Simulation

- [x] **1. Implement Model-Based US Simulator**
  - [x] Add `ModelBasedUSSimulator` class to `live_unet_demo.py` (NumPy physics-based, no neural network)
  - [x] Add `--sim-mode {unet,pix2pix,conv}` CLI argument
  - [x] Update `main()` to skip model loading when `--sim-mode conv`
  - [x] Dispatch inference to `ModelBasedUSSimulator.simulate()` in the sim loop
  - [x] Add `make_label_map(ct_slice, seg_slice)` helper for 3-class label map
  - [x] Verify bone hyperechoic reflectors + acoustic shadowing appear correctly (echo_bone=0.46, echo_shadow=0.13)

- [x] **2. Realism & Physics Overhaul (v3)**
  - [x] Implement Rayleigh-distributed speckle (envelope of complex Gaussian) to eliminate black-hole artifacts and produce smooth, granular B-mode texture.
  - [x] Add CT-modulated backscatter to generate tissue density heterogeneity (vessels, fascia, muscle striations).
  - [x] Add large-scale speckle modulation (SonoGym-style Al/fl parameters) for macro-scale tissue patterns.
  - [x] Add carrier-modulated specular reflection PSF (`cos(2π·f·x)`) to simulate transducer lateral sidelobes on bone echoes.
  - [x] Implement attenuation diffraction (horizontal Gaussian blur of attenuation map) for soft, realistic shadow boundaries.
  - [x] Add electronic noise floor and depth-independent noise mixing to prevent pure black shadows.

- [x] **3. Ray-Tracing Physics Simulator (v4)**
  - [x] Add speed-of-sound `c` to per-tissue acoustic parameters (`_US_TISSUE_PARAMS`)
  - [x] Implement `_raytrace_2d_vectorized()` — pure NumPy vectorized 2D ray-tracer (no Numba dependency)
  - [x] Add `simulate_raytrace()` method reusing existing speckle/noise/log-compression chain
  - [x] Add Snell's law refraction at tissue speed-of-sound boundaries
  - [x] Add surface-normal tilt estimation for initial ray deflection at curved bone
  - [x] Fix coupling-gel boundary: skip reflection at background→skin interface (R ≈ 0.9995 otherwise)
  - [x] Add `--sim-mode ray` CLI option and wire through all dispatch points
  - [x] Verify: shadow ratio 0.49 (vs conv 0.61), 28ms/frame on CPU, bone echo 0.93


## Stage 3: OpenAI Gymnasium Environment Wrapper

- [x] **1. Create Gymnasium Environment Wrapper**
  - [x] Create `robotic_us_env.py` and implement `RoboticUltrasoundGymEnv` inheriting from `gymnasium.Env`
  - [x] Define continuous 6-DOF task-space action space and dictionary observation space
  - [x] Integrate robot IK solver and joint motor controller inside `step`
  - [x] Implement force sensor reading and safety contact limit checking
  - [x] Integrate 2-channel CT and bone segmentation slice extraction
  - [x] Integrate the trained `best_model.pth` U-Net model for real-time B-mode US rendering
  - [x] Formulate reward function matching image quality and normal contact forces
- [x] **2. Verification and Testing**
  - [x] Create automated environment verification script `test_gym_env.py`
  - [x] Test random agent step execution and print performance benchmarks (FPS)
  - [x] Verify safety threshold terminations and output observation shape consistency
- [x] **3. Reinforcement Learning (RL) Scanning Agent Benchmarks**
  - [x] Train A2C agent for 150,000 timesteps (optimized to 160+ FPS): achieved peak reward **+264.1**
  - [x] Train SAC agent for 100,000 timesteps (Soft Actor-Critic off-policy max-entropy): achieved project-record reward **+317.0** with smooth continuous spine tracking
  - [x] Train PPO benchmark agent for comparison (-237 reward)
  - [x] Copy trained checkpoint `.zip` files back to the local workspace
  - [x] Create enjoy/verification script `enjoy_rl.py` supporting A2C, PPO, and SAC auto-detection
  - [x] Evaluate agents visually in PyBullet GUI: verified stable 3.3–3.98 N contact force, perpendicular orientation constraints (±8.6°), and full longitudinal spine coverage.
- [/] **4. Imitation Learning from Real Robotic Scanning Poses (Cavalcanti Dataset)**
  - [x] Create `collect_demos.py` — parse Cavalcanti RUS_pose.txt, auto-detect scanning axis, compute normalised delta-actions, replay through env
  - [x] Create `train_bc.py` — Behavioral Cloning training with `imitation` library + SB3 MultiInputPolicy
  - [x] Update `enjoy_rl.py` with `--algo bc` support for evaluating BC/DAgger policies
  - [x] Update `commands.md` with Section 6: Imitation Learning commands
  - [x] Install `imitation` library on remote server (`pip install imitation`)
  - [x] Run `collect_demos.py --dry-run` on server to validate Cavalcanti pose parsing: 21 sweeps, 67,525 poses → 3,996 actions
  - [x] Run full demo collection on server: 21 trajectories saved in 18s
  - [x] Train BC policy on server: loss 5.85 → 2.0 in 50 epochs (33s on CUDA), `prob_true_act` 0.3% → 20.3%
  - [x] Download BC checkpoint and evaluate locally: verified smooth visual sweeping and ideal contact force (3.89 N) in local PyBullet GUI.
