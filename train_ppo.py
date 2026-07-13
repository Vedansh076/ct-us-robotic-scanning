"""
train_ppo.py — Proximal Policy Optimization (PPO) training for autonomous robotic ultrasound.
==============================================================================================

Trains a PPO policy (via Stable-Baselines3) inside the ``RoboticUltrasoundGymEnv``
Gymnasium environment. PPO is generally more stable and sample-efficient than A2C
for continuous control tasks due to its clipped surrogate objective and multiple
mini-batch gradient updates per rollout.

By default, U-Net inference is **bypassed** (``skip_unet=True`` inside the
environment) so the binary bone segmentation mask is used directly as the
``"image"`` observation.

Usage
-----
Basic training run (100 k steps, default settings)::

    python train_ppo.py

Extended training with custom options::

    python train_ppo.py \\
        --timesteps 200000 \\
        --subject totalseg_patients/s0058 \\
        --n-steps 256 \\
        --batch-size 64 \\
        --n-epochs 10 \\
        --lr 3e-4 \\
        --save-freq 10000 \\
        --tb-log ./ppo_tensorboard/ \\
        --save-dir ./ppo_checkpoints/

Monitor training progress with TensorBoard::

    tensorboard --logdir ./ppo_tensorboard/

Saved checkpoints are stored as ``ppo_model_<N>_steps.zip`` inside ``--save-dir``.
The final model is saved as ``ppo_final_model.zip``. If interrupted, the current
weights are saved as ``ppo_interrupted_model.zip``.

Notes
-----
- **PPO vs A2C**: PPO collects larger rollout buffers (n_steps=256 vs 20) and
  performs multiple gradient passes (n_epochs=10) on each buffer with a clipped
  loss. This makes PPO more sample-efficient and stable, especially for
  continuous control tasks with noisy reward signals.

- **Why DummyVecEnv?** The environment holds a live PyBullet physics client and
  (optionally) a GPU-resident U-Net. Spawning these inside child processes via
  ``SubprocVecEnv`` causes CUDA context conflicts and PyBullet handle errors.
  ``DummyVecEnv`` runs the single environment synchronously in the same process,
  which is safe and correct.

- **PPO hyperparameters**: The default settings (n_steps=256, batch_size=64,
  n_epochs=10, lr=3e-4, clip_range=0.2) follow SB3 recommended values for
  continuous control tasks.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import gymnasium as gym

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser(description="Train a PPO Agent for Autonomous Robotic Ultrasound Scanning")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train (default: 100k)")
    parser.add_argument("--tb-log", type=str, default="./ppo_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./ppo_checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save a checkpoint every N steps")
    parser.add_argument("--n-steps", type=int, default=256, help="Steps collected before each PPO update (default: 256)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for PPO gradient updates (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of gradient epochs per rollout (default: 10)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 60)
    print("  Training PPO Agent for Robotic Ultrasound Scanning")
    print("=" * 60)
    print(f"  Patient Subject:   {args.subject}")
    print(f"  Total Timesteps:   {args.timesteps}")
    print(f"  N-Steps per update:{args.n_steps}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  N-Epochs:          {args.n_epochs}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Clip Range:        {args.clip_range}")
    print(f"  Log Directory:     {args.tb_log}")
    print(f"  Save Directory:    {args.save_dir}")

    # 1. Instantiate the Gymnasium environment in headless (rgb_array) mode
    def make_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            checkpoint_path="model/runs/exp1_2IP/exp1/best_model.pth",
            device="auto",
            render_mode="rgb_array",  # Direct headless mode for training
            max_episode_steps=200,
            size=128                  # Downsample image observation to 128x128
        )
        return Monitor(env)

    # 2. Vectorize the environment
    train_env = DummyVecEnv([make_env])

    # 3. Setup periodic checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix="ppo_model",
        verbose=1
    )

    callbacks = [checkpoint_callback]

    # 4. Initialize PPO Model
    # Since our observation space is a Dictionary (Image + Force + Pose),
    # Stable-Baselines3 automatically uses a MultiInputPolicy to combine the inputs
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,       # Steps per env per update (256 for PPO)
        batch_size=args.batch_size,  # Mini-batch size for gradient updates
        n_epochs=args.n_epochs,     # Gradient epochs per rollout buffer
        gamma=0.99,
        gae_lambda=0.95,            # GAE lambda (0.95 is standard for PPO)
        clip_range=args.clip_range, # PPO clipping parameter
        ent_coef=0.01,              # Entropy bonus to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.tb_log,
        verbose=1,
        device="auto"               # Automatically selects CUDA if available, else CPU
    )

    print("\n  Starting training loop ...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        print("\n  [OK] Training completed successfully!")
        
        # Save final model
        final_path = os.path.join(args.save_dir, "ppo_final_model.zip")
        model.save(final_path)
        print(f"  Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving current checkpoint...")
        interrupted_path = os.path.join(args.save_dir, "ppo_interrupted_model.zip")
        model.save(interrupted_path)
        print(f"  Interrupted model saved to: {interrupted_path}")
        
    finally:
        train_env.close()
        print("  Environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
