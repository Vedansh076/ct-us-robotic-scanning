#!/usr/bin/env python3
"""
train_bc.py — Train a Behavioral Cloning policy from expert demonstrations.
=============================================================================

Loads pre-collected expert trajectories (state-action pairs from Cavalcanti
robotic scanning poses replayed through RoboticUltrasoundGymEnv) and trains
a supervised policy to imitate the expert's scanning behavior.

Uses the `imitation` library's BC implementation with SB3's MultiInputPolicy
(same CNN architecture used by A2C/SAC/PPO agents).

Usage
-----
    # Train BC for 50 epochs on collected demos
    python train_bc.py --demos-dir demos/ --epochs 50

    # Train with custom learning rate and batch size
    python train_bc.py --demos-dir demos/ --epochs 100 --lr 1e-4 --batch-size 128
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))


def load_demonstrations(demos_dir):
    """Load expert trajectories from the demos directory.

    Expects .npz files saved by collect_demos.py, each containing:
      - obs_image, obs_force, obs_pose: observation arrays
      - actions: expert action array
      - terminal: boolean

    Returns
    -------
    trajectories : list of imitation.data.types.Trajectory
    """
    from imitation.data.types import Trajectory

    demos_path = Path(demos_dir)
    traj_files = sorted(demos_path.glob("trajectory_*.npz"))

    if not traj_files:
        raise FileNotFoundError(f"No trajectory_*.npz files found in {demos_dir}")

    trajectories = []
    total_steps = 0

    for tf in traj_files:
        data = np.load(tf, allow_pickle=True)

        # Reconstruct Dict observations
        obs_images = data["obs_image"]   # (T+1, H, W)
        obs_forces = data["obs_force"]   # (T+1, 1)
        obs_poses  = data["obs_pose"]    # (T+1, 7)
        actions    = data["actions"]      # (T, 6)
        terminal   = bool(data["terminal"])

        # Build list of dict observations
        n_obs = len(obs_images)
        obs_list = []
        for i in range(n_obs):
            obs_list.append({
                "image": obs_images[i],
                "force": obs_forces[i],
                "pose":  obs_poses[i],
            })

        traj = Trajectory(
            obs=obs_list,
            acts=actions,
            infos=None,
            terminal=terminal,
        )
        trajectories.append(traj)
        total_steps += len(actions)

    print(f"  Loaded {len(trajectories)} trajectories ({total_steps} total steps)")
    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Train Behavioral Cloning from expert demonstrations"
    )
    parser.add_argument(
        "--demos-dir", type=str, required=True,
        help="Directory containing trajectory_*.npz files from collect_demos.py",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of BC training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--save-dir", type=str, default="bc_checkpoints",
        help="Directory to save the trained BC policy",
    )
    parser.add_argument(
        "--subject", type=str, default="totalseg_patients/s0058",
        help="Subject directory (needed to create env for observation/action spaces)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device for training: auto, cpu, or cuda",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Behavioral Cloning — Training from Expert Demonstrations")
    print("=" * 60)
    print(f"  Demos dir:   {args.demos_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Save dir:    {args.save_dir}")
    print("=" * 60)

    # ---- 1. Load expert demonstrations ----
    print("\n[data] Loading expert demonstrations...")
    trajectories = load_demonstrations(args.demos_dir)

    # ---- 2. Create a temporary env for observation/action space specs ----
    print("\n[env] Creating env for space definitions...")
    from robotic_us_env import RoboticUltrasoundGymEnv

    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",
        render_mode="rgb_array",
        max_episode_steps=200,
        size=128,
    )

    # ---- 3. Set up BC trainer ----
    print("\n[train] Initializing BC trainer...")
    from imitation.algorithms.bc import BC
    import tempfile

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  Training device: {device}")

    rng = np.random.default_rng(42)

    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=rng,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.lr},
        device=device,
    )

    # ---- 4. Train ----
    print(f"\n[train] Training BC for {args.epochs} epochs...")
    t0 = time.time()
    bc_trainer.train(n_epochs=args.epochs)
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- 5. Save the trained policy ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_path = save_dir / "bc_policy.zip"

    # Save as SB3-compatible policy
    bc_trainer.policy.save(str(policy_path))
    print(f"\n  [OK] BC policy saved to: {policy_path}")

    # Also save training metadata
    meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "training_time_s": elapsed,
        "n_trajectories": len(trajectories),
        "device": str(device),
    }
    meta_path = save_dir / "bc_training_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  [OK] Training metadata saved to: {meta_path}")

    # Cleanup
    env.close()
    print("\n" + "=" * 60)
    print("  Behavioral Cloning training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
