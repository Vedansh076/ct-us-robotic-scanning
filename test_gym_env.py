import os
import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add current folder to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI visualizer")
    args = parser.parse_args()

    render_mode = "rgb_array" if args.headless else "human"

    print("=" * 60)
    print("  RoboticUltrasoundGymEnv Visual Verification Script")
    print("=" * 60)
    print(f"  Mode: {'Headless (DIRECT)' if args.headless else 'Visual (GUI)'}")

    # 1. Initialize environment
    t0 = time.time()
    print("  Initializing environment ...")
    env = RoboticUltrasoundGymEnv(
        subject_dir="totalseg_patients/s0058",
        checkpoint_path="runs/cavalcanti_unet/best_model.pth",
        device="auto",
        render_mode=render_mode,
        max_episode_steps=200,
        skip_unet=False
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

    if not args.headless:
        print("\n  [OpenCV] Opening side-by-side ultrasound prediction window...")
        cv2.namedWindow("Gym Environment US prediction", cv2.WINDOW_NORMAL)

    # 3. Step execution with random actions
    print("\n  Running 200 steps with random actions ...")
    steps = 200
    total_reward = 0.0
    start_time = time.perf_counter()
    
    for i in range(steps):
        # Generate random continuous action in [-1.0, 1.0]
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display B-mode image in GUI mode
        if not args.headless:
            cv2.imshow("Gym Environment US prediction", obs['image'])
            cv2.waitKey(1)
            time.sleep(0.02) # Cap the loop speed so it is pleasant to watch
            
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
    if not args.headless:
        cv2.destroyAllWindows()
    print("\n  [OK] Gym Environment verification passed completely!")
    print("=" * 60)

if __name__ == "__main__":
    main()
