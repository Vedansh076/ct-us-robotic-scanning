#!/usr/bin/env python3
"""
train_gail.py — Train a GAIL (Generative Adversarial Imitation Learning) policy.
=============================================================================

Loads pre-collected expert trajectories and trains a GAIL discriminator alongside
an SB3 PPO generator policy to mimic the expert's spine-sweeping behavior.
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from robotic_us_env import RoboticUltrasoundGymEnv
from train_bc import load_demonstrations

# Register Gymnasium spaces and NumPy classes for safe unpickling in PyTorch 2.6+
try:
    import gymnasium
    th.serialization.add_safe_globals([
        gymnasium.spaces.dict.Dict,
        gymnasium.spaces.box.Box,
        gymnasium.spaces.discrete.Discrete,
        np.dtype,
        np.core.multiarray.scalar,
    ])
except Exception:
    pass


class MultiInputRewardNet(nn.Module):
    """Custom Multi-Input Reward Network (Discriminator) for GAIL.

    Processes Dict observation spaces containing images (CNN) and vectors (MLP),
    combined with the continuous action, to predict a scalar reward.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        # 1. CNN for the image observation (input: 1x128x128)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),   # 16x31x31
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),  # 32x12x12
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 4608

        # 2. MLP for force (1-dim) and pose (7-dim)
        self.vector_mlp = nn.Sequential(
            nn.Linear(1 + 7, 32),
            nn.ReLU(),
        )

        # 3. Action processor (6-dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

        # 4. Reward head to predict scalar reward
        total_features = cnn_out_dim + 32 + 32
        self.reward_head = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, next_state, done) -> th.Tensor:
        # Scale and format image
        img = state["image"]
        if img.ndim == 3:
            img = img.unsqueeze(1)
        img = img.float() / 255.0

        img_feats = self.cnn(img)

        # Process force and pose
        vec_in = th.cat([state["force"].float(), state["pose"].float()], dim=1)
        vec_feats = self.vector_mlp(vec_in)

        # Process action
        act_feats = self.action_mlp(action.float())

        # Fuse features and output scalar reward
        combined = th.cat([img_feats, vec_feats, act_feats], dim=1)
        reward = self.reward_head(combined)
        return reward.squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Train a GAIL policy from expert demonstrations")
    parser.add_argument("--demos-dir", type=str, required=True, help="Directory containing expert trajectories")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total GAIL training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Generator/Discriminator learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--save-dir", type=str, default="gail_checkpoints", help="Directory to save checkpoint")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  Generative Adversarial Imitation Learning (GAIL) Training")
    print("=" * 60)
    print(f"  Demos dir:   {args.demos_dir}")
    print(f"  Timesteps:   {args.timesteps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Save dir:    {args.save_dir}")
    print("=" * 60)

    # 1. Load expert trajectories
    print("\n[data] Loading expert demonstrations...")
    trajectories = load_demonstrations(args.demos_dir)

    # 2. Setup training environments (fast skip_unet mode)
    print("\n[env] Creating environments...")
    
    def make_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            device="cpu",
            render_mode="rgb_array",
            max_episode_steps=200,
            size=128,
        )
        return Monitor(env)

    # GAIL venv needs to be vectorized
    venv = DummyVecEnv([make_env])

    # 3. Setup custom Reward Network and PPO Generator
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import RewardNet
    from imitation.util.networks import RunningNorm

    # Wrap our custom MultiInputRewardNet to conform to the imitation base class
    class WrappedRewardNet(RewardNet):
        def __init__(self, observation_space, action_space):
            super().__init__(observation_space, action_space)
            self.net = MultiInputRewardNet(observation_space, action_space)

        def forward(self, state, action, next_state, done) -> th.Tensor:
            return self.net(state, action, next_state, done)

    print("\n[model] Initializing PPO Generator and Custom Discriminator...")
    generator = PPO(
        policy="MultiInputPolicy",
        env=venv,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=1,
    )

    # Use RunningNorm to normalize inputs to the reward net
    reward_net = WrappedRewardNet(venv.observation_space, venv.action_space)

    # 4. Instantiate GAIL Trainer
    print("\n[train] Initializing GAIL trainer...")
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=args.batch_size,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net,
    )

    # 5. Train GAIL
    print(f"\n[train] Training GAIL for {args.timesteps} timesteps...")
    t0 = time.time()
    gail_trainer.train(total_timesteps=args.timesteps)
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 6. Save the trained generator policy
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_path = save_dir / "gail_policy.zip"
    
    # Save the trained generator policy (standard SB3 compatible)
    generator.policy.save(str(policy_path))
    print(f"\n  [OK] GAIL policy saved to: {policy_path}")

    # Cleanup
    venv.close()
    print("\n" + "=" * 60)
    print("  GAIL training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
