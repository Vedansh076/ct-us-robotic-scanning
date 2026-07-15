# Execution & Model Command Reference

> **NOTICE FOR ALL ASSISTANT AGENTS:**
> This file is a central reference of all execution, training, and evaluation commands for the CT-to-Ultrasound Robotic Scanning project. **EVERY assistant agent working in this workspace MUST maintain and update this file whenever new scripts, trained models, CLI arguments, or execution workflows are added or modified.**

---

## 1. Autonomous Reinforcement Learning (RL) Scanning Agent

### A. Evaluate / Visualize Trained Agent in GUI Simulator (`enjoy_policy.py`)
Runs a trained Stable-Baselines3 model (.zip) in the live interactive PyBullet GUI simulator. Supports auto-detection for A2C, PPO, and SAC checkpoints.

```powershell
# Run final trained A2C scanning agent (Default/Recommended)
python enjoy_policy.py --checkpoint a2c_checkpoints/a2c_final_model.zip

# Run specific step checkpoint (e.g. 30k steps)
python enjoy_policy.py --checkpoint a2c_checkpoints/a2c_model_30000_steps.zip

# Run trained PPO or SAC model (auto-detected from filename)
python enjoy_policy.py --checkpoint ppo_checkpoints/ppo_final_model.zip
python enjoy_policy.py --checkpoint sac_checkpoints/sac_final_model.zip
```

### B. Train A2C Agent (`train_a2c.py`)
Trains an Advantage Actor-Critic (A2C) agent with 800 N/m contact physics, ±8.6° perpendicularity constraints, and dense longitudinal sweep rewards.

```bash
# Recommended server training command (pinned to dedicated CPU cores)
cd ~/workspace/lakshya/ct-us-robotic-scanning
git pull origin main
nohup taskset -c 0,1,2,3 python3 train_a2c.py --timesteps 150000 --save-freq 30000 > train_a2c_v8.log 2>&1 &
```

### C. Train PPO Agent (`train_ppo.py`)
Trains a Proximal Policy Optimization (PPO) continuous control benchmark agent.

```bash
cd ~/workspace/lakshya/ct-us-robotic-scanning
nohup taskset -c 0,1,2,3 python3 train_ppo.py --timesteps 100000 --n-steps 2048 --batch-size 64 > train_ppo.log 2>&1 &
```

### D. Train SAC Agent (`train_sac.py`)
Trains a Soft Actor-Critic (SAC) off-policy maximum entropy continuous control benchmark agent.

```bash
cd ~/workspace/lakshya/ct-us-robotic-scanning
nohup taskset -c 0,1,2,3 python3 train_sac.py --timesteps 100000 --save-freq 20000 > train_sac.log 2>&1 &
```

---

## 6. Imitation Learning from Cavalcanti Robotic Scanning Poses

### A. Collect Expert Demonstrations (`collect_demos.py`)
Parses `RUS_pose.txt` files from the Cavalcanti dataset, converts real robotic trajectories into normalised delta-actions, and replays them through the PyBullet env to collect `(observation, action)` pairs.

```bash
# Dry-run: print statistics without creating the environment
cd ~/workspace/lakshya/ct-us-robotic-scanning
python3 collect_demos.py --data-root data/Cavalcanti --dry-run

# Full collection: replay through env and save trajectory .npz files
nohup python3 collect_demos.py --data-root data/Cavalcanti --output-dir demos/ --stride 15 > collect_demos.log 2>&1 &

# Process specific volunteers only
python3 collect_demos.py --data-root data/Cavalcanti --output-dir demos/ --stride 15 --volunteers URS01 URS02
```

### B. Train Behavioral Cloning Policy (`train_bc.py`)
Trains a supervised BC policy from pre-collected expert demonstrations using the `imitation` library + SB3 MultiInputPolicy.

```bash
cd ~/workspace/lakshya/ct-us-robotic-scanning
nohup python3 train_bc.py --demos-dir demos/ --epochs 50 > train_bc.log 2>&1 &

# With custom hyperparameters
python3 train_bc.py --demos-dir demos/ --epochs 100 --lr 1e-4 --batch-size 128
```

### C. Evaluate BC Policy (`enjoy_policy.py --algo bc`)
Runs the trained BC/IL policy in the PyBullet GUI simulator.

```powershell
python enjoy_policy.py --checkpoint bc_checkpoints/bc_policy.zip --algo bc
```

### D. Train GAIL Policy (`train_gail.py`)
Trains a Generative Adversarial Imitation Learning (GAIL) policy. Requires `--bc-checkpoint` for BC pre-initialization to prevent conservative action collapse.

```bash
cd ~/workspace/lakshya/ct-us-robotic-scanning
# Recommended: BC-initialized GAIL (prevents conservative action collapse)
nohup taskset -c 0,1,2,3 python3 train_gail.py --demos-dir demos/ --timesteps 50000 --bc-checkpoint bc_checkpoints/bc_policy.zip > train_gail.log 2>&1 &

# Without BC init (may converge to tiny actions)
python3 train_gail.py --demos-dir demos/ --timesteps 50000
```

### E. Evaluate GAIL Policy (`enjoy_policy.py --algo gail`)
Runs the trained GAIL generator policy in the PyBullet GUI simulator.

```powershell
python enjoy_policy.py --checkpoint gail_checkpoints/gail_policy.zip --algo gail --skip-unet
```

---

## 2. Deep Generative Models & Live Simulation (`live_unet_demo.py`)

### A. 2-Channel U-Net Deep Generative Model
Predicts synthetic B-mode ultrasound images in real-time from stacked CT + bone label slices using the trained 2-channel U-Net.

```powershell
python live_unet_demo.py --checkpoint model/runs/exp1_2IP/exp1/best_model.pth --sim-mode unet
```

### B. 2-Channel Pix2Pix GAN Generative Model
Predicts textured ultrasound B-mode scans using the conditional Generative Adversarial Network.

```powershell
python live_unet_demo.py --checkpoint model/runs/exp_pix2pix/best_model.pth --sim-mode pix2pix
```

### C. Quantitative Model Evaluation (`--eval`)
Calculates quantitative metrics (SSIM, PSNR, MAE) over patient test subjects.

```powershell
python live_unet_demo.py --checkpoint model/runs/exp1_2IP/exp1/best_model.pth --eval
```

---

## 3. Physics-Based Ultrasound Simulators (No Neural Network Required)

### A. Model-Based Convolution Simulator (`--sim-mode conv`)
Runs physics v3 simulator (Rayleigh speckle, CT backscatter, carrier PSF, lateral coherence).

```powershell
python live_unet_demo.py --sim-mode conv
```

### B. Model-Based Vectorized Ray-Tracing Simulator (`--sim-mode ray`)
Runs pure-NumPy 2D ray-tracer with Snell's law refraction at tissue speed-of-sound boundaries.

```powershell
python live_unet_demo.py --sim-mode ray
```

---

## 4. Model Training & Data Preprocessing Scripts

### A. Train 2-Channel U-Net Generator (`model/train.py`)
Trains the 2-channel U-Net on CT and label slice stacks.

```bash
python model/train.py --data_dir Cavalcanti_processed/ --epochs 50 --batch_size 16 --train_subjects auto
```

### B. Train Pix2Pix GAN Generator (`model/train_pix2pix.py`)
Trains the Pix2Pix generator and PatchGAN discriminator.

```bash
python model/train_pix2pix.py --data_dir Cavalcanti_processed/ --epochs 50 --batch_size 16 --train_subjects auto
```

### C. Cavalcanti Spine Dataset ICP Registration & 3D Reslicing (`model/prepare_cavalcanti.py`)
Performs patient mesh registration, PCA bounding box centering, and 3D oblique slice stack generation.

```bash
# Verify raw dataset discovery
python model/prepare_cavalcanti.py --data_dir Cavalcanti_dataset/ --discover

# Full preprocessing run
python model/prepare_cavalcanti.py --data_dir Cavalcanti_dataset/ --output_dir Cavalcanti_processed/
```

---

## 5. Environment Verification & Diagnostics

### A. Gymnasium Environment Diagnostics (`test_gym_env.py`)
Tests random action step execution, step rate FPS benchmarking, and observation dictionary shapes.

```powershell
python test_gym_env.py
```

### B. Oblique CT Slice Extraction Test (`test_slice.py`)
Verifies registration-aware 3D reslicing accuracy and PyTorch `grid_sample` execution speed.

```powershell
python test_slice.py
```

---

## 7. Action Chunking with Transformers (ACT) Benchmark

### A. Train ACT Model (`train_act.py`)
Trains the Action Chunking policy. Recommended options:
* `--mlp`: (Highly Recommended) Trains the simplified MLP Action Chunking Policy (ACT-MLP). It has the fastest convergence and is the most stable on small datasets.
* `--no-cvae`: Trains the CVAE-less Transformer (ACT-Lite).
* (Default): Trains standard ACT with CVAE.

```bash
cd ~/workspace/lakshya/ct-us-robotic-scanning
# Recommended: MLP-ACT Policy (ACT-MLP)
nohup taskset -c 0,1,2,3 python3 train_act.py --demos-dir demos/ --epochs 100 --chunk-size 20 --mlp --lr 1e-3 > train_act.log 2>&1 &

# ACT-Lite (no CVAE Transformer)
nohup taskset -c 0,1,2,3 python3 train_act.py --demos-dir demos/ --epochs 300 --chunk-size 20 --no-cvae --lr 1e-3 > train_act.log 2>&1 &

# Standard ACT (with CVAE)
python3 train_act.py --demos-dir demos/ --epochs 100 --chunk-size 20
```

### B. Evaluate ACT Policy (`enjoy_act.py`)
Runs the trained ACT policy in the PyBullet GUI simulator. If using Transformer, set `--temperature 0.1` to prevent dilution from decayed future steps.

```powershell
python enjoy_act.py --checkpoint act_checkpoints/act_policy.zip --skip-unet
```
