#!/usr/bin/env python3
"""
collect_demos.py — Collect expert IL demonstrations from Cavalcanti RUS poses.
===============================================================================

Parses RUS_pose.txt files from the Cavalcanti Robotic Lumbar Spine Dataset,
converts the real robotic scanning trajectories into delta-actions compatible
with RoboticUltrasoundGymEnv, and replays them through the environment to
collect (observation, action) pairs for Behavioral Cloning.

The Cavalcanti dataset contains 6-DOF end-effector poses (x, y, z, roll,
pitch, yaw) recorded from a UR5 robot performing lumbar spine US scans at
constant 5 N force.  Consecutive pose differences are normalized to the env's
action space [-1, 1] using:

    action[:3] = clip(delta_pos / 0.005, -1, 1)   # ±5 mm per step
    action[3:] = clip(delta_euler / 0.05, -1, 1)   # ±0.05 rad per step

The scanning axis in Cavalcanti (largest total displacement) is auto-detected
and mapped to the env's Y axis (longitudinal spine direction).

Usage
-----
    # Dry-run: print statistics without creating the environment
    python collect_demos.py --data-root data/Cavalcanti --dry-run

    # Full collection: replay through env and save trajectories
    python collect_demos.py --data-root data/Cavalcanti --output-dir demos/ --stride 15

    # Process specific volunteers only
    python collect_demos.py --data-root data/Cavalcanti --output-dir demos/ \\
        --volunteers URS01 URS02 --stride 15
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add current directory to path for env imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# ---------------------------------------------------------------------------
# Constants — must match RoboticUltrasoundGymEnv action scaling
# ---------------------------------------------------------------------------
POS_SCALE = 0.005    # env multiplies action[:3] by 0.005 m
ORN_SCALE = 0.05     # env multiplies action[3:] by 0.05 rad


# ═══════════════════════════════════════════════════════════════════════════
# Pose parsing (adapted from model/prepare_cavalcanti.py)
# ═══════════════════════════════════════════════════════════════════════════

def parse_rus_poses(pose_file: str) -> np.ndarray:
    """Parse RUS_pose.txt → (N, 6) array of [x, y, z, roll, pitch, yaw].

    Coordinates are in **meters** and **radians**.
    Auto-detects degrees vs radians and converts if needed.
    """
    rows: list = []
    with open(pose_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("x"):
                continue
            parts = re.split(r"[\s,\t]+", line)
            parts = [p for p in parts if p]
            if len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts[:6]]
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        return np.empty((0, 6))

    poses = np.array(rows, dtype=np.float64)  # (N, 6)

    # Auto-detect degrees: if any Euler angle > 2π, assume degrees
    angles = poses[:, 3:6]
    if np.any(np.abs(angles) > 2.0 * np.pi):
        poses[:, 3:6] = np.deg2rad(poses[:, 3:6])

    return poses


# ═══════════════════════════════════════════════════════════════════════════
# Axis detection & mapping
# ═══════════════════════════════════════════════════════════════════════════

def detect_scanning_axis(poses: np.ndarray) -> Tuple[int, int, int]:
    """Detect the primary scanning axis from total positional displacement.

    Returns (scanning_axis, lateral_axis, vertical_axis) as indices into
    [x=0, y=1, z=2].  The scanning axis has the largest net displacement.
    """
    if len(poses) < 2:
        return 0, 1, 2

    total_disp = np.abs(poses[-1, :3] - poses[0, :3])
    order = np.argsort(total_disp)  # ascending
    return int(order[2]), int(order[0]), int(order[1])


def build_axis_map(scan_ax: int, lat_ax: int, vert_ax: int) -> np.ndarray:
    """Build a permutation array that maps Cavalcanti axes → env axes.

    Mapping convention:
      Cavalcanti scanning axis → env Y (index 1)
      Cavalcanti lateral axis  → env X (index 0)
      Cavalcanti vertical axis → env Z (index 2)

    Returns (3,) int array such that ``delta_env = delta_cav[axis_map]``.
    """
    axis_map = np.empty(3, dtype=int)
    # We want: env_pos[0] = cav_pos[lat_ax]
    #          env_pos[1] = cav_pos[scan_ax]
    #          env_pos[2] = cav_pos[vert_ax]
    # So we build a *selection* array: axis_map = [lat_ax, scan_ax, vert_ax]
    axis_map[0] = lat_ax
    axis_map[1] = scan_ax
    axis_map[2] = vert_ax
    return axis_map


# ═══════════════════════════════════════════════════════════════════════════
# Delta-action computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_expert_actions(
    poses: np.ndarray,
    stride: int = 15,
    axis_map: np.ndarray | None = None,
    max_steps: int = 200,
) -> np.ndarray:
    """Convert consecutive Cavalcanti poses into normalised delta-actions.

    Parameters
    ----------
    poses : (N, 6) — [x, y, z, roll, pitch, yaw] in m / rad
    stride : subsample factor (every *stride*-th frame)
    axis_map : (3,) permutation for mapping Cavalcanti pos → env pos
    max_steps : truncate to this many actions

    Returns
    -------
    actions : (M, 6) — normalised to [-1, 1]
    """
    actions: list = []

    for i in range(0, len(poses) - stride, stride):
        delta_pos = poses[i + stride, :3] - poses[i, :3]
        delta_euler = poses[i + stride, 3:] - poses[i, 3:]

        # Remap position axes (Cavalcanti → env)
        if axis_map is not None:
            delta_pos = delta_pos[axis_map]

        action = np.zeros(6, dtype=np.float32)
        action[:3] = np.clip(delta_pos / POS_SCALE, -1.0, 1.0)
        # Zero out orientation deltas — our env clamps to ±8.6° anyway,
        # and the Cavalcanti UR5 frame conventions differ from our Panda.
        # The position deltas alone capture the scanning trajectory pattern.
        action[3:] = 0.0

        actions.append(action)

        if len(actions) >= max_steps:
            break

    return np.array(actions, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# File discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_rus_pose_files(data_root: str) -> List[Tuple[str, Path]]:
    """Find all RUS_pose.txt files, returning (sweep_name, path) pairs."""
    us_root = Path(data_root) / "ultrasound"
    if not us_root.exists():
        print(f"ERROR: Ultrasound directory not found: {us_root}")
        return []

    results: list = []
    for sweep_dir in sorted(us_root.iterdir()):
        if not sweep_dir.is_dir():
            continue
        # Handle nested directories (URS01_R1/URS1_R1/RUS_pose.txt)
        for root, _dirs, files in os.walk(sweep_dir):
            for f in files:
                if f == "RUS_pose.txt":
                    results.append((sweep_dir.name, Path(root) / f))
    return sorted(results, key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════════
# Environment replay
# ═══════════════════════════════════════════════════════════════════════════

def replay_and_save(
    env,
    expert_actions: np.ndarray,
    sweep_name: str,
    output_dir: Path,
    traj_index: int,
) -> dict:
    """Replay expert actions through an existing env and save the trajectory.

    The env is NOT closed — it is reused across multiple calls with reset().

    Returns a stats dict with episode reward, steps, etc.
    """
    # Override max_episode_steps for this trajectory
    env.max_episode_steps = len(expert_actions) + 10

    obs, info = env.reset()

    # Storage for trajectory
    obs_images = [obs["image"]]
    obs_forces = [obs["force"]]
    obs_poses = [obs["pose"]]
    actions_taken = []
    total_reward = 0.0

    for step_i, action in enumerate(expert_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        obs_images.append(obs["image"])
        obs_forces.append(obs["force"])
        obs_poses.append(obs["pose"])
        actions_taken.append(action)
        total_reward += reward

        if terminated or truncated:
            break

    n_steps = len(actions_taken)

    # Save trajectory as .npz
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"trajectory_{traj_index:04d}_{sweep_name}.npz"

    np.savez_compressed(
        out_path,
        obs_image=np.array(obs_images, dtype=np.uint8),
        obs_force=np.array(obs_forces, dtype=np.float32),
        obs_pose=np.array(obs_poses, dtype=np.float32),
        actions=np.array(actions_taken, dtype=np.float32),
        terminal=True,
        sweep_name=sweep_name,
    )

    stats = {
        "sweep": sweep_name,
        "steps": n_steps,
        "reward": total_reward,
        "file": str(out_path),
        "mean_force": np.mean([f[0] for f in obs_forces[1:]]),
    }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Collect IL demonstrations from Cavalcanti robotic scanning poses"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Path to the Cavalcanti dataset root (contains ultrasound/ subdirectory)",
    )
    parser.add_argument(
        "--output-dir", default="demos",
        help="Output directory for trajectory .npz files (default: demos/)",
    )
    parser.add_argument(
        "--stride", type=int, default=15,
        help="Subsample every N-th Cavalcanti frame (default: 15). "
             "With ~3000 frames per sweep, stride=15 gives ~200 steps per episode.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=200,
        help="Maximum steps per episode trajectory (default: 200)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print statistics; do not create env or save trajectories.",
    )
    parser.add_argument(
        "--volunteers", nargs="+", default=None,
        help="Process only specific volunteers (e.g. URS01 URS02).",
    )
    parser.add_argument(
        "--subject", default="totalseg_patients/s0058",
        help="Patient subject directory for env replay (default: totalseg_patients/s0058)",
    )
    parser.add_argument(
        "--obs-size", type=int, default=128,
        help="Observation image size in pixels (default: 128, must match training).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Cavalcanti Expert Demo Collector for Imitation Learning")
    print("=" * 60)

    # ---- Discover pose files ----
    pose_files = find_rus_pose_files(args.data_root)
    if args.volunteers:
        pose_files = [
            (name, path) for name, path in pose_files
            if any(v in name for v in args.volunteers)
        ]

    if not pose_files:
        print("ERROR: No RUS_pose.txt files found.")
        sys.exit(1)

    print(f"\n  Found {len(pose_files)} robotic sweep trajectories")
    axis_names = ["X", "Y", "Z"]

    # ---- Process each sweep ----
    all_actions = []
    all_stats = []

    for sweep_name, pose_path in pose_files:
        poses = parse_rus_poses(str(pose_path))
        if len(poses) < 2:
            print(f"  [SKIP] {sweep_name}: too few poses ({len(poses)})")
            continue

        # Detect axes
        scan_ax, lat_ax, vert_ax = detect_scanning_axis(poses)
        axis_map = build_axis_map(scan_ax, lat_ax, vert_ax)

        # Compute expert actions
        actions = compute_expert_actions(
            poses, stride=args.stride, axis_map=axis_map,
            max_steps=args.max_episode_steps,
        )

        total_disp = np.abs(poses[-1, :3] - poses[0, :3])

        stats = {
            "sweep": sweep_name,
            "n_poses": len(poses),
            "n_actions": len(actions),
            "scanning_axis": axis_names[scan_ax],
            "total_disp_m": total_disp,
            "action_mean": actions.mean(axis=0) if len(actions) > 0 else np.zeros(6),
            "action_std": actions.std(axis=0) if len(actions) > 0 else np.zeros(6),
            "action_range": (
                actions.min() if len(actions) > 0 else 0,
                actions.max() if len(actions) > 0 else 0,
            ),
        }
        all_stats.append(stats)
        all_actions.append((sweep_name, actions))

        disp_str = ", ".join(f"{axis_names[i]}={total_disp[i]*1000:.1f}mm"
                            for i in range(3))
        print(f"\n  {sweep_name}:")
        print(f"    {len(poses)} poses → {len(actions)} actions (stride={args.stride})")
        print(f"    Scanning axis: {axis_names[scan_ax]}  |  Displacement: {disp_str}")
        print(f"    Action range: [{stats['action_range'][0]:.3f}, {stats['action_range'][1]:.3f}]")
        print(f"    Action mean (pos): [{actions[:,:3].mean(axis=0)}]")

    # ---- Summary ----
    total_actions = sum(s["n_actions"] for s in all_stats)
    total_poses = sum(s["n_poses"] for s in all_stats)
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {len(all_stats)} sweeps, {total_poses} poses → {total_actions} expert actions")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n  [DRY RUN] No environment replay performed.")
        return

    # ---- Full collection: replay through env ----
    print(f"\n[replay] Replaying {len(all_actions)} trajectories through env...")
    print(f"         Subject: {args.subject}")
    print(f"         Output:  {args.output_dir}")

    # Create the env ONCE and reuse across all trajectories
    from robotic_us_env import RoboticUltrasoundGymEnv
    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps + 10,
        size=args.obs_size,
    )
    print(f"  [env] Environment created successfully.")

    output_path = Path(args.output_dir)
    replay_stats = []

    for idx, (sweep_name, actions) in enumerate(all_actions):
        print(f"\n  [{idx+1}/{len(all_actions)}] Replaying {sweep_name} "
              f"({len(actions)} steps)...")
        t0 = time.time()
        stats = replay_and_save(
            env=env,
            expert_actions=actions,
            sweep_name=sweep_name,
            output_dir=output_path,
            traj_index=idx,
        )
        elapsed = time.time() - t0
        replay_stats.append(stats)
        print(f"    Reward: {stats['reward']:+.1f}  |  "
              f"Mean Force: {stats['mean_force']:.2f} N  |  "
              f"Time: {elapsed:.1f}s")

    # Clean up
    env.close()

    # ---- Final report ----
    print(f"\n{'=' * 60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Trajectories saved: {len(replay_stats)}")
    print(f"  Output directory:   {output_path}")
    total_steps = sum(s["steps"] for s in replay_stats)
    avg_reward = np.mean([s["reward"] for s in replay_stats])
    print(f"  Total steps:        {total_steps}")
    print(f"  Average reward:     {avg_reward:+.1f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
