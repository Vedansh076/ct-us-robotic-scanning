# Project-Scoped Rules: CT-to-Ultrasound Robotic Scanning

This workspace contains three tracking files to preserve context, tasks, and execution commands across sessions:
1. [agent.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/agent.md) tracks the overall project state, findings, and context.
2. [task.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/task.md) tracks the checklist of pending and completed tasks.
3. [commands.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/commands.md) tracks all execution, training, and evaluation commands across all models.

## Rules for Assistant Agents:
1. **Context Check on Startup:** On your very first turn of any conversation or task in this workspace, you MUST read `agent.md`, `task.md`, and `commands.md` using `view_file` to establish context.
2. **Synchronize Tasks:** When you start or complete any task listed in `task.md`, update its checklist status (e.g. `[x]`, `[/]`, `[ ]`) immediately.
3. **Log Technical Findings:** If you discover new bugs, architectural details, or normalisation/activation mismatches, document them in Section 2 of `agent.md`.
4. **Maintain Command Reference:** Whenever adding or modifying executable scripts, model architectures, CLI flags, or execution workflows, you MUST update and maintain [commands.md](file:///e:/DELL/internship/Data/HumanSubjects/HumanSubjects/ct_us/commands.md).
5. **Active Workspace:** Make all changes in the `ct_us` main project folder. Do not work in `ct_us_standalone` directly unless explicitly requested to update the standalone package.
6. **Git Synchronization:** Upon completing a task or stage, verify compilation. Stage the modified code files, commit them, and push the updates to GitHub using `git push origin main` to keep the remote repository current.

## Architecture Awareness:

The codebase has a strict architecture. Before modifying any file, understand these relationships:

### Module Dependency Graph
```
live_unet_demo.py  ←  robotic_us_env.py  ←  train_*.py / enjoy_policy.py
       ↑                      ↑
  registration.py         extract_slice.py
       ↑
generate_patient_meshes.py
```

- `live_unet_demo.py` is the **central hub** (2667 lines). It exports functions used by `robotic_us_env.py`.
- `robotic_us_env.py` imports ~20 functions from `live_unet_demo.py`. Renaming or removing exported symbols breaks the Gym env.
- All RL training scripts (`train_a2c.py`, `train_sac.py`, `train_ppo.py`) and `enjoy_policy.py` import from `robotic_us_env.py`.
- `extract_slice.py` and `registration.py` are standalone utility modules.

### Critical Invariants — DO NOT BREAK:

1. **U-Net Activation**: `model/model.py` UNet uses `Sigmoid` output (range [0,1]). `model/pix2pix/model.py` UNet uses `Tanh` output (range [-1,1]). NEVER swap these imports in training or inference code.

2. **HU Windowing Consistency**: Training (`model/dataset.py`) clips CT to [-150, 1250] HU and scales to [-1, 1]. Inference in `live_unet_demo.py` MUST use the identical range. Mismatched ranges cause prediction degradation.

3. **Mesh Winding Order**: `generate_patient_meshes.py` inverts face winding with `faces[:, [0, 2, 1]]`. This MUST be preserved — without it, PyBullet renders meshes as transparent/hollow due to back-face culling.

4. **Zero-Padding Before Marching Cubes**: The binary body volume MUST be zero-padded on all 6 faces before `marching_cubes()`. This ensures the isosurface is a closed manifold, not an open shell.

5. **Raycast Robot Exclusion**: Always use `raycast_skin_surface()` (not raw PyBullet raycasts) to find skin surface Z. Raw raycasts get blocked by robot arm geometry.

6. **DummyVecEnv Only**: Training MUST use `DummyVecEnv`, NOT `SubprocVecEnv`. PyBullet physics clients and CUDA contexts cannot be safely forked across processes.

7. **Registration Chain**: The forward transform (voxel→world) and inverse transform (world→voxel) in `registration.py` are mathematically exact. Any modification must preserve:
   - NIfTI affine (LPS coords) → physical mm → meters → mesh centering → body rotation → PyBullet world
   - And the exact inverse of each step

8. **Action/Observation Space**: The Gym env's action space is `Box(-1, 1, shape=(6,))` with scaling [0.01m, 0.01m, 0.01m, 0.05rad, 0.05rad, 0.05rad]. The observation space is `Dict(image=(256,256) uint8, force=(1,) float32, pose=(7,) float32)`. Changing these shapes breaks all trained checkpoints.

9. **Spring Model Parameters**: k=800 N/m, standoff=3mm. Target force window: 2-8N. Safety termination: >12N. These values are calibrated together — changing one requires recalibrating the others.

10. **Strategy 2 Training**: By default, RL training uses `skip_unet=True` (binary bone mask as observation). This runs at ~440 FPS vs ~12 FPS with U-Net. The policy transfers to U-Net observations because the dominant visual feature (bone reflection) is present in both.

### Trained Checkpoints (DO NOT DELETE):
- `runs/cavalcanti_unet/best_model.pth` — Latest 2-channel U-Net (trained on Cavalcanti)
- `runs/cavalcanti_unet_cpu/best_model.pth` — Lightweight 2-channel U-Net (optimised for CPU)
- `model/runs/exp1_2IP/exp1/best_model.pth` — Default 2-channel U-Net
- `model/runs/exp_pix2pix/best_model.pth` — Latest Pix2Pix GAN
- `runs/cavalcanti_pix2pix/best_model.pth` — Alternate Pix2Pix GAN
- `a2c_checkpoints/a2c_final_model.zip` — A2C (reward +264.1)
- `sac_checkpoints/sac_final_model.zip` — SAC (reward +317.0, BEST)
- `bc_checkpoints/bc_policy.zip` — Behavioral Cloning (loss 2.0)
- `gail_checkpoints/gail_policy.zip` — GAIL

### Patient Data (5 subjects):
- `totalseg_patients/{s0011, s0058 (default), s0223, s0250, s0310}/`
- Each needs: `ct.nii.gz`, `segmentations/`, `patient_skin.obj`, `bone_label.nii.gz`, `registration_meta.json`

### US Synthesis Modes:
- `--sim-mode unet` — Neural net (needs GPU)
- `--sim-mode pix2pix` — GAN (needs GPU)
- `--sim-mode conv` — Physics convolution (CPU only)
- `--sim-mode ray` — Ray-tracing with Snell's law (CPU only)

## Testing Requirements:
Before committing changes to any core file, verify:
1. `python test_gym_env.py --headless` runs without errors (env init + 200 random steps)
2. If modifying `live_unet_demo.py`: test all 4 sim modes manually
3. If modifying `model/model.py` or `model/dataset.py`: verify `python live_unet_demo.py --eval` runs
4. If modifying `robotic_us_env.py`: verify a short training run `python train_a2c.py --timesteps 100`
5. If modifying `registration.py` or `extract_slice.py`: verify `python test_slice.py`
