"""
enjoy_rl.py — Run a trained RL agent visually in the PyBullet GUI simulator.
=============================================================================

Loads a trained Stable-Baselines3 model checkpoint (.zip) — A2C, PPO, or SAC —
and executes it inside the RoboticUltrasoundGymEnv environment in "human"
rendering mode. The algorithm is auto-detected from the checkpoint filename
(e.g. 'a2c_final_model.zip' → A2C, 'sac_final_model.zip' → SAC).

Usage
-----
    python enjoy_rl.py --checkpoint a2c_checkpoints/a2c_final_model.zip
    python enjoy_rl.py --checkpoint sac_checkpoints/sac_final_model.zip
    python enjoy_rl.py --checkpoint my_model.zip --algo sac
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
    args = parser.parse_args()

    print("=" * 60)
    print("  Evaluating Trained Agent")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Subject:    {args.subject}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Headless:   {args.headless}")
    print("=" * 60)

    # 1. Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 2. Instantiate the Gymnasium environment
    render_mode = "rgb_array" if args.headless else "human"
    print(f"\n[env] Launching PyBullet simulator (mode: {render_mode})...")
    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",             # Run on CPU locally (no heavy GPU needed for demo)
        render_mode=render_mode,  # Set based on headless flag
        max_episode_steps=200,
        size=128                  # Must match the 128x128 observation size used in training
    )

    # 3. Auto-detect algorithm from checkpoint filename or --algo flag
    algo_name = args.algo
    if algo_name is None:
        cp_lower = os.path.basename(args.checkpoint).lower()
        if "sac" in cp_lower:
            algo_name = "sac"
        elif "ppo" in cp_lower:
            algo_name = "ppo"
        elif "bc" in cp_lower or "dagger" in cp_lower or "gail" in cp_lower:
            algo_name = "bc"
        else:
            algo_name = "a2c"  # Default to A2C
    
    if algo_name == "bc":
        # Load BC/DAgger policy (saved as SB3 ActorCriticPolicy by imitation library)
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
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Starting Episode {ep} ---")
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                # Predict the next action deterministically
                if model is not None:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
                
                # Step the simulator
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Print status
                force = obs["force"][0]
                print(f"  Step {step_count:3d} | Force: {force:4.2f} N | Reward: {reward:6.2f} | Acc Reward: {episode_reward:7.2f}", end="\r")
                
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
