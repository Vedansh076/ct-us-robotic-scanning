"""
enjoy_policy.py — Run a trained RL or IL agent visually in the PyBullet GUI simulator.
=============================================================================

Loads a trained Stable-Baselines3 model checkpoint (.zip) — A2C, PPO, or SAC —
or an imitation learning policy (BC, GAIL) and executes it inside the
RoboticUltrasoundGymEnv environment in "human" rendering mode.

Usage
-----
    python enjoy_policy.py --checkpoint a2c_checkpoints/a2c_final_model.zip
    python enjoy_policy.py --checkpoint sac_checkpoints/sac_final_model.zip
    python enjoy_policy.py --checkpoint my_model.zip --algo sac
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from robotic_us_env import RoboticUltrasoundGymEnv

# Monkey-patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
try:
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained A2C agent visually in PyBullet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained SB3 model .zip file")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds for smoother viewing")
    parser.add_argument("--algo", type=str, default=None, choices=["a2c", "ppo", "sac", "bc", "gail"], help="Algorithm (auto-detected from filename if not set)")
    parser.add_argument("--headless", action="store_true", help="Run in headless (rgb_array) mode without opening PyBullet GUI")
    parser.add_argument("--skip-unet", action="store_true", help="Skip U-Net inference and return raw bone segmentation masks")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Multiplier for Y-axis action to speed up visual sweep (default: 1.0)")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="EMA smoothing factor for evaluation actions (default: 0.35)")
    parser.add_argument("--lock-x", action="store_true", help="Lock lateral X movement to keep probe centered on the spine ridge")
    args = parser.parse_args()

    print("=" * 60)
    print("  Evaluating Trained Agent")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Subject:    {args.subject}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Headless:   {args.headless}")
    print(f"  Skip U-Net: {args.skip_unet}")
    print("=" * 60)

    # 1. Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 2. Instantiate the Gymnasium environment
    render_mode = "rgb_array" if args.headless else "human"
    print(f"\n[env] Launching PyBullet simulator (mode: {render_mode})...")
    
    # Scale physics steps and joint speed limits proportionally to scale_y to ensure smooth tracking
    import numpy as np
    eval_substeps = int(np.clip(5 * (args.scale_y ** 0.5), 5, 20))
    eval_max_vel = float(np.clip(1.5 * args.scale_y, 1.5, 8.0))
    
    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",             # Run on CPU locally (no heavy GPU needed for demo)
        render_mode=render_mode,  # Set based on headless flag
        max_episode_steps=200,
        size=128,                 # Must match the 128x128 observation size used in training
        skip_unet=args.skip_unet,
        substeps=eval_substeps,
        max_velocity=eval_max_vel,
    )

    # 3. Auto-detect algorithm from checkpoint filename or --algo flag
    algo_name = args.algo
    if algo_name is None:
        cp_lower = os.path.basename(args.checkpoint).lower()
        if "sac" in cp_lower:
            algo_name = "sac"
        elif "ppo" in cp_lower:
            algo_name = "ppo"
        elif "gail" in cp_lower:
            algo_name = "gail"
        elif "bc" in cp_lower or "dagger" in cp_lower:
            algo_name = "bc"
        else:
            algo_name = "a2c"  # Default to A2C
    
    if algo_name == "gail":
        # Wrap environment to output flat observations
        from gail_wrapper import FlattenMultiInputWrapper
        env = FlattenMultiInputWrapper(env)
        
        print(f"[model] Loading trained GAIL generator policy...")
        from stable_baselines3.common.policies import ActorCriticPolicy
        policy = ActorCriticPolicy.load(args.checkpoint, device="cpu")
        model = None
    elif algo_name == "bc":
        # Load BC/IL policy (saved as raw SB3 policy, not wrapped in an algorithm)
        print(f"[model] Loading trained BC/IL policy...")
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        policy = MultiInputActorCriticPolicy.load(args.checkpoint, device="cpu")
        model = None  # We'll call policy.predict() directly
    else:
        if algo_name == "sac":
            AlgoClass = SAC
        elif algo_name == "ppo":
            AlgoClass = PPO
        else:
            AlgoClass = A2C
        print(f"[model] Loading trained {algo_name.upper()} policy...")
        model = AlgoClass.load(args.checkpoint, env=env, device="cpu")
        policy = None

    try:
        # Training dataset reference Z height (s0058 baseline = 1.238m)
        TRAIN_BASELINE_Z = 1.238
        
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Starting Episode {ep} ---")
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            # Ensure probe is settled in skin contact (>=1.5N force) at episode start
            start_force = float(obs["force"].reshape(-1)[0]) if isinstance(obs, dict) else float(obs.reshape(-1)[16384])
            settle_count = 0
            while start_force < 1.5 and settle_count < 5:
                down_vec = np.array([0.0, 0.0, -0.35, 0.0, 0.0, 0.0], dtype=np.float32)
                exec_down = down_vec.reshape(1, -1) if isinstance(env, DummyVecEnv) else down_vec
                obs, _, _, _, _ = env.step(exec_down)
                start_force = float(obs["force"].reshape(-1)[0]) if isinstance(obs, dict) else float(obs.reshape(-1)[16384])
                settle_count += 1
            
            # Compute baseline Z-height alignment offset relative to training reference (1.238m)
            inner_env = env.envs[0].unwrapped if hasattr(env, "envs") else (env.unwrapped if hasattr(env, "unwrapped") else env)
            z_offset = float(TRAIN_BASELINE_Z - inner_env.home_pos[2])
            
            while not done:
                # Prepare aligned observation with exact un-batched shapes matching Gymnasium specs
                if isinstance(obs, dict):
                    aligned_obs = {
                        "image": obs["image"].copy(),
                        "force": obs["force"].copy(),
                        "pose":  obs["pose"].copy(),
                    }
                    aligned_obs["pose"][2] += np.float32(z_offset)
                else:
                    aligned_obs = obs.copy()
                    if aligned_obs.ndim == 2:
                        aligned_obs[0, 16387] += z_offset
                    else:
                        aligned_obs[16387] += z_offset
                
                # Predict the next action deterministically
                if model is not None:
                    action, _states = model.predict(aligned_obs, deterministic=True)
                else:
                    action, _states = policy.predict(aligned_obs, deterministic=True)
                
                # Extract 1D action vector
                act_vec = action.reshape(-1)
                
                # Clamp lateral X drift to keep probe on the spine centerline
                if getattr(args, 'lock_x', False):
                    act_vec[0] = 0.0
                
                # Contact compliance: if probe loses skin contact (force == 0.0N), press downward to maintain scan
                curr_f = inner_env.current_force
                if curr_f == 0.0:
                    act_vec[2] = -0.25
                
                # Apply Y-axis speed multiplier for visual evaluation
                act_vec[1] = act_vec[1] * args.scale_y
                
                # Format action for VecEnv (1, 6) vs Raw Gym Env (6,)
                if isinstance(env, DummyVecEnv):
                    exec_action = act_vec.reshape(1, -1)
                else:
                    exec_action = act_vec
                
                # Step the simulator
                obs, reward, terminated, truncated, info = env.step(exec_action)
                
                # Unwrap scalar values if vector environment
                if isinstance(reward, np.ndarray):
                    reward = float(reward[0])
                if isinstance(terminated, np.ndarray):
                    terminated = bool(terminated[0])
                if isinstance(truncated, np.ndarray):
                    truncated = bool(truncated[0])
                
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Print status
                import pybullet as p
                from live_unet_demo import PANDA_EE_LINK
                inner_env = env.envs[0].unwrapped if hasattr(env, "envs") else (env.unwrapped if hasattr(env, "unwrapped") else env)
                force = inner_env.current_force
                ee_state = p.getLinkState(inner_env.panda_id, PANDA_EE_LINK)
                pose = ee_state[4]
                print(f"  Step {step_count:3d} | Force: {force:4.2f} N | EE X: {pose[0]:.4f} | Y: {pose[1]:.4f} | Z: {pose[2]:.4f} | Action Y: {act_vec[1]:.4f} Z: {act_vec[2]:.4f}")
                
                # Small delay to make the simulation human-viewable
                time.sleep(args.delay)
                
            print(f"\n[Episode {ep} Finished] Steps: {step_count} | Total Reward: {episode_reward:.2f}")
            time.sleep(1.0) # Pause between episodes
            
    except KeyboardInterrupt:
        print("\n[info] Demo interrupted by user.")
        
    finally:
        env.close()
        print("[info] Simulator environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
