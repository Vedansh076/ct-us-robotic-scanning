"""
train_sac.py — Soft Actor-Critic (SAC) training for autonomous robotic ultrasound.
===================================================================================

Trains a Soft Actor-Critic (SAC) policy (via Stable-Baselines3) inside the
``RoboticUltrasoundGymEnv`` Gymnasium environment.

SAC is an off-policy maximum entropy actor-critic algorithm specifically designed
for continuous robotic control. Key advantages for robotic ultrasound scanning:
  1. Replay Buffer (50k steps): Learns from all past physical transitions,
     offering 10x-50x higher sample efficiency than on-policy A2C/PPO.
  2. Maximum Entropy Objective: Optimizes for (Reward + Entropy), which naturally
     produces smooth, human-like probe trajectories without aggressive jitter.
  3. Automatic Temperature Tuning (ent_coef="auto"): Auto-tunes the entropy
     weight alpha to balance exploration vs. exploitation during training.

Usage
-----
Basic training run (50 k steps, default settings)::

    python train_sac.py

Extended training with custom options::

    python train_sac.py \\
        --timesteps 100000 \\
        --subject totalseg_patients/s0058 \\
        --buffer-size 50000 \\
        --batch-size 256 \\
        --lr 3e-4 \\
        --save-freq 10000 \\
        --tb-log ./sac_tensorboard/ \\
        --save-dir ./sac_checkpoints/

Monitor training progress with TensorBoard::

    tensorboard --logdir ./sac_tensorboard/

Saved checkpoints are stored as ``sac_model_<N>_steps.zip`` inside ``--save-dir``.
The final model is saved as ``sac_final_model.zip``.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import gymnasium as gym

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser(description="Train a SAC Agent for Autonomous Robotic Ultrasound Scanning")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train (default: 50k)")
    parser.add_argument("--tb-log", type=str, default="./sac_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./sac_checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save a checkpoint every N steps")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for gradient updates (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for actor/critic (default: 3e-4)")
    parser.add_argument("--learning-starts", type=int, default=1000, help="Steps before gradient updates begin (default: 1000)")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 60)
    print("  Training SAC Agent for Robotic Ultrasound Scanning")
    print("=" * 60)
    print(f"  Patient Subject:   {args.subject}")
    print(f"  Total Timesteps:   {args.timesteps}")
    print(f"  Buffer Size:       {args.buffer_size}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Learning Starts:   {args.learning_starts}")
    print(f"  Log Directory:     {args.tb_log}")
    print(f"  Save Directory:    {args.save_dir}")

    # 1. Instantiate the Gymnasium environment in headless (rgb_array) mode
    def make_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            checkpoint_path="runs/cavalcanti_unet/best_model.pth",
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
        name_prefix="sac_model",
        verbose=1
    )

    callbacks = [checkpoint_callback]

    # 4. Initialize SAC Model
    # MultiInputPolicy handles Dictionary observation (Image + Force + Pose)
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),     # Update Q-networks and actor every step
        gradient_steps=1,
        ent_coef="auto",            # Automatically tunes entropy coefficient alpha
        tensorboard_log=args.tb_log,
        verbose=1,
        device="auto"               # Automatically selects CUDA if available
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
        final_path = os.path.join(args.save_dir, "sac_final_model.zip")
        model.save(final_path)
        print(f"  Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving current checkpoint...")
        interrupted_path = os.path.join(args.save_dir, "sac_interrupted_model.zip")
        model.save(interrupted_path)
        print(f"  Interrupted model saved to: {interrupted_path}")
        
    finally:
        train_env.close()
        print("  Environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
