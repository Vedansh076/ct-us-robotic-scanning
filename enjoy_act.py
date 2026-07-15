#!/usr/bin/env python3
"""
enjoy_act.py — Run a trained ACT agent visually in the PyBullet GUI simulator.
=============================================================================

Loads a trained ACT model (.zip) and executes it inside the RoboticUltrasoundGymEnv
using temporal ensembling to ensure smooth scanning trajectories.

Usage
-----
    python enjoy_act.py --checkpoint act_checkpoints/act_policy.zip --skip-unet
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from robotic_us_env import RoboticUltrasoundGymEnv
from train_act import ActionChunkingTransformer, ACTVisionEncoder, ACTProprioEncoder, ACTCvaeEncoder, MlpActionChunkingPolicy

# Alias classes in __main__ module to allow PyTorch to unpickle models saved from train_act.py
import __main__
__main__.MlpActionChunkingPolicy = MlpActionChunkingPolicy
__main__.ActionChunkingTransformer = ActionChunkingTransformer
__main__.ACTVisionEncoder = ACTVisionEncoder
__main__.ACTProprioEncoder = ACTProprioEncoder
__main__.ACTCvaeEncoder = ACTCvaeEncoder

# Monkey-patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
try:
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass


class ACTTemporalEnsemble:
    """Helper to manage overlapping action predictions and ensemble them.

    Implements: w_i = exp(-i / temperature)
    """
    def __init__(self, chunk_size=20, temperature=10.0):
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.all_predictions = [] # List of numpy arrays of shape (chunk_size, 6)

    def reset(self):
        self.all_predictions = []

    def update_and_get_action(self, new_chunk):
        """Append new chunk and calculate the ensembled action for the current step.

        Parameters
        ----------
        new_chunk : np.ndarray
            Predicted action sequence of shape (chunk_size, 6).

        Returns
        -------
        ensembled_action : np.ndarray
            Weighted average action of shape (6,).
        """
        # Add new chunk to list of predictions
        self.all_predictions.append(new_chunk)
        
        # Keep only the last chunk_size predictions
        if len(self.all_predictions) > self.chunk_size:
            self.all_predictions.pop(0)
            
        # Determine current age/index for each active prediction
        num_active = len(self.all_predictions)
        ensembled = np.zeros(6, dtype=np.float32)
        total_weight = 0.0
        
        for idx in range(num_active):
            # The older the prediction, the higher its index in the prediction list
            # prediction age i: 0 is the most recent prediction, 1 is from the previous step, etc.
            age = num_active - 1 - idx
            weight = np.exp(-age / self.temperature)
            
            # The prediction for the *current* step is at index 'age' in that chunk
            # e.g., for the chunk generated 'age' steps ago, its prediction for today is chunk[age]
            action_pred = self.all_predictions[idx][age]
            
            ensembled += action_pred * weight
            total_weight += weight
            
        return ensembled / total_weight


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ACT agent visually in PyBullet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained ACT model file (.zip)")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--skip-unet", action="store_true", help="Skip U-Net inference and return raw bone masks")
    parser.add_argument("--temperature", type=float, default=10.0, help="Temporal ensemble temperature (higher = smoother)")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Y-axis action multiplier")
    args = parser.parse_args()

    print("=" * 60)
    print("  Evaluating Trained ACT Agent")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Subject:      {args.subject}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Skip U-Net:   {args.skip_unet}")
    print("=" * 60)

    # 1. Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 2. Instantiate the Gymnasium environment
    render_mode = "rgb_array" if args.headless else "human"
    print(f"\n[env] Launching PyBullet simulator (mode: {render_mode})...")
    
    # Scale physics steps and joint speed limits proportionally to scale_y to ensure smooth tracking
    eval_substeps = int(np.clip(5 * (args.scale_y ** 0.5), 5, 20))
    eval_max_vel = float(np.clip(1.5 * args.scale_y, 1.5, 8.0))

    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",
        render_mode=render_mode,
        max_episode_steps=200,
        size=128,
        skip_unet=args.skip_unet,
        substeps=eval_substeps,
        max_velocity=eval_max_vel,
    )

    # 3. Load trained ACT policy
    print(f"\n[model] Loading trained ACT policy from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()
    
    # Load normalization stats if available
    import pickle
    stats_path = Path(args.checkpoint).parent / "norm_stats.pkl"
    norm_stats = None
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            norm_stats = pickle.load(f)
        print(f"[model] Loaded dataset normalization statistics from {stats_path}")
    else:
        print("[model] Warning: norm_stats.pkl not found. Evaluating with raw un-normalized values.")

    # Initialize temporal ensembler
    ensembler = ACTTemporalEnsemble(chunk_size=model.chunk_size, temperature=args.temperature)

    try:
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Starting Episode {ep} ---")
            obs, info = env.reset()
            ensembler.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                # Prepare observations for PyTorch
                image_np = obs["image"].astype(np.float32) / 255.0
                image_t = torch.tensor(image_np, device=device).unsqueeze(0).unsqueeze(0) # (1, 1, 128, 128)
                
                raw_force = obs["force"].astype(np.float32)
                raw_pose = obs["pose"].astype(np.float32)
                
                if norm_stats is not None:
                    force_norm = (raw_force - norm_stats["force_mean"][0]) / norm_stats["force_std"][0]
                    pose_norm = (raw_pose - norm_stats["pose_mean"][0]) / norm_stats["pose_std"][0]
                else:
                    force_norm = raw_force
                    pose_norm = raw_pose
                    
                force_t = torch.tensor(force_norm, device=device).unsqueeze(0) # (1, 1)
                pose_t = torch.tensor(pose_norm, device=device).unsqueeze(0)   # (1, 7)
                
                # Predict action chunk
                with torch.no_grad():
                    pred_chunk_t, _, _ = model(image_t, force_t, pose_t)
                    pred_chunk = pred_chunk_t.squeeze(0).cpu().numpy() # (chunk_size, 6)
                
                # Un-normalize predicted action chunk if stats exist
                if norm_stats is not None:
                    pred_chunk = pred_chunk * norm_stats["action_std"][0] + norm_stats["action_mean"][0]
                
                # Retrieve ensembled action
                action = ensembler.update_and_get_action(pred_chunk)
                
                # Apply optional scale-y
                action[1] = action[1] * args.scale_y
                
                # Step the simulator
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Print status
                import pybullet as p
                from live_unet_demo import PANDA_EE_LINK
                force = env.unwrapped.current_force
                ee_state = p.getLinkState(env.unwrapped.panda_id, PANDA_EE_LINK)
                pose = ee_state[4]
                print(f"  Step {step_count:3d} | Force: {force:4.2f} N | EE X: {pose[0]:.4f} | Y: {pose[1]:.4f} | Z: {pose[2]:.4f} | Action Y: {action[1]:.4f} Z: {action[2]:.4f}")
                
                # Small delay to make the simulation human-viewable
                time.sleep(args.delay)
                
            print(f"\n[Episode {ep} Finished] Steps: {step_count} | Total Reward: {episode_reward:.2f}")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[info] Demo interrupted by user.")
        
    finally:
        env.close()
        print("[info] Simulator environment closed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
