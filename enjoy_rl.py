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
from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained A2C agent visually in PyBullet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained SB3 model .zip file")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds for smoother viewing")
    parser.add_argument("--algo", type=str, default=None, choices=["a2c", "ppo", "sac"], help="Algorithm (auto-detected from filename if not set)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Evaluating Trained A2C Agent Visually in PyBullet")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Subject:    {args.subject}")
    print(f"  Episodes:   {args.episodes}")
    print("=" * 60)

    # 1. Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 2. Instantiate the Gymnasium environment in visual (human) mode
    print("\n[env] Launching PyBullet GUI simulator...")
    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        device="cpu",             # Run on CPU locally (no heavy GPU needed for demo)
        render_mode="human",      # Render the visual PyBullet window!
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
        else:
            algo_name = "a2c"  # Default to A2C
    
    if algo_name == "sac":
        AlgoClass = SAC
    elif algo_name == "ppo":
        AlgoClass = PPO
    else:
        AlgoClass = A2C

    print(f"[model] Loading trained {algo_name.upper()} policy...")
    model = AlgoClass.load(args.checkpoint, env=env, device="cpu")

    try:
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Starting Episode {ep} ---")
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                # Predict the next action deterministically
                action, _states = model.predict(obs, deterministic=True)
                
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
