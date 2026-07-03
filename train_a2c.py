"""
train_a2c.py — Advantage Actor-Critic (A2C) training for autonomous robotic ultrasound.
=========================================================================================

Trains an A2C policy (via Stable-Baselines3) inside the ``RoboticUltrasoundGymEnv``
Gymnasium environment. By default, U-Net inference is **bypassed** (``skip_unet=True``
inside the environment) so the binary bone segmentation mask is used directly as the
``"image"`` observation. This ``Strategy 2`` mode runs at ~440 FPS on a Colab T4 GPU
and completes 100 k steps in roughly 4 minutes.

The trained policy transfers to real ultrasound observations because the brightest
feature in a B-mode image (specular bone reflection) co-localises with the binary mask.

Usage
-----
Basic training run (100 k steps, default settings)::

    python train_a2c.py

Extended training with custom options::

    python train_a2c.py \\
        --timesteps 500000 \\
        --subject totalseg_patients/s0058 \\
        --n-steps 20 \\
        --lr 7e-4 \\
        --save-freq 50000 \\
        --tb-log ./a2c_tensorboard/ \\
        --save-dir ./a2c_checkpoints/

Monitor training progress with TensorBoard::

    tensorboard --logdir ./a2c_tensorboard/

Saved checkpoints are stored as ``a2c_model_<N>_steps.zip`` inside ``--save-dir``.
The final model is saved as ``a2c_final_model.zip``. If interrupted, the current
weights are saved as ``a2c_interrupted_model.zip``.

Notes
-----
- **Why DummyVecEnv?** The environment holds a live PyBullet physics client and
  (optionally) a GPU-resident U-Net. Spawning these inside child processes via
  ``SubprocVecEnv`` causes CUDA context conflicts and PyBullet handle errors.
  ``DummyVecEnv`` runs the single environment synchronously in the same process,
  which is safe and correct.

- **Why no EvalCallback?** Spawning a second simultaneous PyBullet + U-Net instance
  for evaluation would double memory usage and risk instability on most laptops.
  Use TensorBoard episode-reward curves to monitor learning instead.

- **A2C hyperparameters**: The default settings (n_steps=20, lr=7e-4, gamma=0.99,
  ent_coef=0.01) follow SB3 recommended values for continuous control tasks with
  short episode horizons (200 steps).
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import gymnasium as gym

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser(description="Train an A2C Agent for Autonomous Robotic Ultrasound Scanning")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train (default: 100k)")
    parser.add_argument("--tb-log", type=str, default="./a2c_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./a2c_checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--save-freq", type=int, default=5000, help="Save a checkpoint every N steps")
    parser.add_argument("--n-steps", type=int, default=20, help="Steps collected before each A2C update (default: 20)")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate (default: 7e-4)")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 60)
    print("  Training A2C Agent for Robotic Ultrasound Scanning")
    print("=" * 60)
    print(f"  Patient Subject: {args.subject}")
    print(f"  Total Timesteps: {args.timesteps}")
    print(f"  N-Steps per update: {args.n_steps}")
    print(f"  Learning Rate:   {args.lr}")
    print(f"  Log Directory:   {args.tb_log}")
    print(f"  Save Directory:  {args.save_dir}")

    # 1. Instantiate the Gymnasium environment in headless (rgb_array) mode
    def make_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            checkpoint_path="model/runs/exp1_2IP/exp1/best_model.pth",
            device="auto",
            render_mode="rgb_array",  # Direct headless mode for training
            max_episode_steps=200
        )
        return Monitor(env)

    # 2. Vectorize the environment
    # A2C runs synchronously; DummyVecEnv wraps a single env on CPU
    train_env = DummyVecEnv([make_env])

    # 3. Setup periodic checkpoint callback
    # NOTE: We skip EvalCallback to avoid spawning a second heavy PyBullet+UNet
    # instance simultaneously on CPU (memory and stability reasons).
    # Use TensorBoard to monitor training progress instead.
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix="a2c_model",
        verbose=1
    )

    callbacks = [checkpoint_callback]

    # 5. Initialize A2C Model
    # Since our observation space is a Dictionary (Image + Force + Pose),
    # Stable-Baselines3 automatically uses a MultiInputPolicy to combine the inputs
    model = A2C(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,   # Steps per env per update (20 is a good balance)
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,          # Entropy bonus to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        tensorboard_log=args.tb_log,
        verbose=1,
        device="auto"           # Automatically selects CUDA if available, else CPU
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
        final_path = os.path.join(args.save_dir, "a2c_final_model.zip")
        model.save(final_path)
        print(f"  Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving current checkpoint...")
        interrupted_path = os.path.join(args.save_dir, "a2c_interrupted_model.zip")
        model.save(interrupted_path)
        print(f"  Interrupted model saved to: {interrupted_path}")
        
    finally:
        train_env.close()
        print("  Environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
