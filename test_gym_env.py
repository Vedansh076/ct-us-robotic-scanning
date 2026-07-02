import os
import sys
import time
import numpy as np
from pathlib import Path

# Add current folder to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    print("=" * 60)
    print("  RoboticUltrasoundGymEnv Verification Script")
    print("=" * 60)

    # 1. Initialize environment in DIRECT (headless) mode
    t0 = time.time()
    print("  Initializing environment ...")
    env = RoboticUltrasoundGymEnv(
        subject_dir="totalseg_patients/s0058",
        checkpoint_path="model/runs/exp1_2IP/exp1/best_model.pth",
        device="auto",
        render_mode="rgb_array", # Direct headless mode
        max_episode_steps=100
    )
    print(f"  Initialized successfully in {time.time() - t0:.2f}s")
    
    # 2. Reset environment
    print("\n  Resetting environment ...")
    obs, info = env.reset()
    
    print("\n  Observation Space Verification:")
    for k, v in obs.items():
        print(f"    Key: {k:<10} | Shape: {str(v.shape):<14} | Type: {str(v.dtype):<10} | Range: [{v.min()}, {v.max()}]")
        
    # Check shape correctness
    assert obs['image'].shape == (256, 256), "Image shape mismatch!"
    assert obs['force'].shape == (1,), "Force shape mismatch!"
    assert obs['pose'].shape == (7,), "Pose shape mismatch!"
    print("  [OK] Observation space structure matches requirements.")

    # 3. Step execution with random actions
    print("\n  Running 100 steps with random actions ...")
    steps = 100
    total_reward = 0.0
    start_time = time.perf_counter()
    
    for i in range(steps):
        # Generate random continuous action in [-1.0, 1.0]
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i + 1) % 20 == 0 or terminated or truncated:
            print(f"    Step {i+1:3d} | Force: {float(obs['force'][0]):.3f} N | Reward: {reward:+.3f} | Terminated: {terminated} | Truncated: {truncated}")
            
        if terminated or truncated:
            print(f"  Episode ended at step {i+1}. Resetting ...")
            obs, info = env.reset()
            
    elapsed = time.perf_counter() - start_time
    fps = steps / elapsed
    print(f"\n  Performance Benchmark:")
    print(f"    Total Steps: {steps}")
    print(f"    Total Time:  {elapsed:.2f} s")
    print(f"    Frame Rate:  {fps:.1f} FPS")
    print(f"    Total Accumulated Reward: {total_reward:+.3f}")
    
    env.close()
    print("\n  [OK] Gym Environment verification passed completely!")
    print("=" * 60)

if __name__ == "__main__":
    main()
