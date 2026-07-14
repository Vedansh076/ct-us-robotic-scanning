#!/usr/bin/env python3
"""
train_gail.py — Train a GAIL policy using a custom flattened observation wrapper.
=============================================================================
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
from gail_wrapper import FlattenMultiInputWrapper, CustomFlatFeatureExtractor, FlatRewardNet

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


def flatten_trajectories(trajectories):
    """Convert expert trajectories containing Dict observations into flat Box format.
    """
    from imitation.data.types import Trajectory, maybe_unwrap_dictobs
    flat_trajectories = []
    
    for traj in trajectories:
        # Unwrap DictObs object to a standard dict of numpy arrays
        obs_dict = maybe_unwrap_dictobs(traj.obs)
        images = np.asarray(obs_dict["image"])  # (T+1, 128, 128)
        forces = np.asarray(obs_dict["force"])  # (T+1, 1)
        poses = np.asarray(obs_dict["pose"])    # (T+1, 7)
        
        # Flatten images to (T+1, 16384) and scale to [0, 1]
        images_flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
        
        # Concatenate along axis=1 to get (T+1, 16392)
        flat_obs = np.concatenate([images_flat, forces, poses], axis=1)
        
        flat_traj = Trajectory(
            obs=flat_obs,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal,
        )
        flat_trajectories.append(flat_traj)
        
    print(f"  [data] Flattened {len(flat_trajectories)} expert trajectories.")
    return flat_trajectories


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
    print("  Generative Adversarial Imitation Learning (GAIL) — Flat Wrapper")
    print("=" * 60)
    print(f"  Demos dir:   {args.demos_dir}")
    print(f"  Timesteps:   {args.timesteps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Save dir:    {args.save_dir}")
    print("=" * 60)

    # 1. Load and flatten expert trajectories
    print("\n[data] Loading expert demonstrations...")
    trajectories = load_demonstrations(args.demos_dir)
    flat_trajectories = flatten_trajectories(trajectories)

    # 2. Setup training environments with Flatten Wrapper
    print("\n[env] Creating environments...")
    
    def make_env():
        raw_env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            device="cpu",
            render_mode="rgb_array",
            max_episode_steps=200,
            size=128,
        )
        # Flatten Dict observations
        flat_env = FlattenMultiInputWrapper(raw_env)
        return Monitor(flat_env)

    venv = DummyVecEnv([make_env])

    # 3. Setup custom PPO policy with CustomFlatFeatureExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy

    print("\n[model] Initializing PPO Generator and Custom Discriminator...")
    policy_kwargs = dict(
        features_extractor_class=CustomFlatFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    generator = PPO(
        policy=ActorCriticPolicy,
        env=venv,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # Instantiate custom reward network
    reward_net = FlatRewardNet(venv.observation_space, venv.action_space)

    # 4. Instantiate GAIL Trainer from imitation library
    from imitation.algorithms.adversarial.gail import GAIL

    print("\n[train] Initializing GAIL trainer...")
    gail_trainer = GAIL(
        demonstrations=flat_trajectories,
        demo_batch_size=args.batch_size,
        venv=venv,
        gen_algo=generator,
        reward_net=reward_net,
        allow_variable_horizon=True,
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
