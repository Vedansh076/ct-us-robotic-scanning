import time
import numpy as np
import pybullet as p
from robotic_us_env import RoboticUltrasoundGymEnv

def profile():
    print("Initializing environment...")
    env = RoboticUltrasoundGymEnv(
        subject_dir="totalseg_patients/s0058",
        device="auto",
        render_mode="rgb_array",
        max_episode_steps=200,
        size=128,
        skip_unet=True
    )
    
    env.reset()
    
    print("-" * 50)
    print("VOLUME PROPERTIES:")
    print(f"  CT shape:      {env.ct_volume.shape}")
    print(f"  CT dtype:      {env.ct_volume.dtype}")
    print(f"  CT strides:    {env.ct_volume.strides}")
    print(f"  CT C_CONTIG:   {env.ct_volume.flags['C_CONTIGUOUS']}")
    print(f"  Label shape:   {env.label_volume.shape}")
    print(f"  Label dtype:   {env.label_volume.dtype}")
    print(f"  Label strides: {env.label_volume.strides}")
    print(f"  Label C_CONTIG:{env.label_volume.flags['C_CONTIGUOUS']}")
    print("-" * 50)
    
    # Cast to float64 once to test casting overhead theory
    env.ct_volume = env.ct_volume.astype(np.float64)
    env.label_volume = env.label_volume.astype(np.float64)
    
    times = {
        "action_decode": [],
        "bullet_step": [],
        "link_state": [],
        "raycast": [],
        "slice_extract": [],
        "obs_dict": [],
        "reward_calc": [],
        "total_step": []
    }
    
    print("Running 100 steps...")
    for _ in range(100):
        action = env.action_space.sample()
        
        t_start = time.perf_counter()
        
        t0 = time.perf_counter()
        delta_pos = action[:3] * 0.01
        delta_euler = action[3:] * 0.05
        
        from live_unet_demo import PANDA_EE_LINK
        ee_state = p.getLinkState(env.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        curr_pos = np.array(ee_state[4])
        curr_orn = np.array(ee_state[5])
        
        curr_euler = np.array(p.getEulerFromQuaternion(curr_orn.tolist()))
        target_euler = curr_euler + delta_euler
        target_euler[0] = np.clip(target_euler[0], np.pi - 0.4, np.pi + 0.4)
        target_euler[1] = np.clip(target_euler[1], -0.4, 0.4)
        target_euler[2] = np.clip(target_euler[2], -0.6, 0.6)
        
        target_pos = curr_pos + delta_pos
        target_pos[0] = np.clip(target_pos[0], -0.45, 0.45)
        target_pos[1] = np.clip(target_pos[1], -0.80, 0.80)
        target_pos[2] = np.clip(target_pos[2], env.bed_top_z + 0.02, env.bed_top_z + 0.50)
        
        target_orn = np.array(p.getQuaternionFromEuler(target_euler.tolist()))
        
        from live_unet_demo import drive_panda_to_pose
        drive_panda_to_pose(env.panda_id, target_pos, target_orn)
        times["action_decode"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        for _ in range(5):
            p.stepSimulation()
        times["bullet_step"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        ee_state = p.getLinkState(env.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4], dtype=np.float32)
        ee_orn = np.array(ee_state[5], dtype=np.float32)
        times["link_state"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        from live_unet_demo import raycast_skin_surface, get_probe_pose_from_ee, update_probe_model, compute_probe_contact_force
        probe_pos_vis, quaternion_xyzw, probe_contact_position = get_probe_pose_from_ee(ee_pos, ee_orn)
        update_probe_model(env.probe_body_id, probe_contact_position, quaternion_xyzw)
        
        found_body, surface_z = raycast_skin_surface(ee_pos[0], ee_pos[1], env.body_id)
        hit_distance = None
        probe_tip_pos = ee_pos.copy()
        
        if found_body:
            from live_unet_demo import PROBE_TIP_FROM_EE
            rm = np.array(p.getMatrixFromQuaternion(ee_orn.tolist())).reshape(3, 3)
            probe_tip_pos = ee_pos + rm @ PROBE_TIP_FROM_EE
            ray_start = probe_tip_pos - rm[:, 2] * 0.05
            ray_end = probe_tip_pos + rm[:, 2] * 0.05
            res = p.rayTest(ray_start.tolist(), ray_end.tolist())
            if len(res) > 0 and res[0][0] == env.body_id:
                frac = res[0][2]
                hit_distance = frac * 0.10 - 0.05
            else:
                hit_distance = probe_tip_pos[2] - surface_z
        
        force_val = compute_probe_contact_force(hit_distance, desired_standoff=0.008, max_force=10.0)
        env.current_force = force_val
        times["raycast"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        if found_body and hit_distance is not None and hit_distance <= 0.05:
            from live_unet_demo import compute_registered_ct_center
            ct_center = compute_registered_ct_center(
                probe_tip_pos, env.reg_body_position, env.reg_body_orientation_matrix,
                env.reg_meta['inv_affine'], env.reg_meta['mesh_centering_offset'],
                env.ct_volume.shape, mesh_scale=env.mesh_scale
            )
            from extract_slice import extract_slice
            seg_slice = extract_slice(
                env.label_volume, center=ct_center, quaternion=ee_orn,
                spacing=env.spacing, size=env.slice_size, pixel_spacing=env.pixel_spacing,
                inv_affine=env.reg_meta['inv_affine'], mesh_scale=env.mesh_scale,
                body_orientation_matrix=env.reg_body_orientation_matrix,
                order=0
            )
            seg_slice = (seg_slice > 0.5).astype(np.float32)
            us_image_uint8 = (seg_slice * 255).astype(np.uint8)
            env.last_seg_slice = seg_slice
        else:
            us_image_uint8 = np.zeros((env.slice_size, env.slice_size), dtype=np.uint8)
            env.last_seg_slice = np.zeros((env.slice_size, env.slice_size), dtype=np.float32)
        times["slice_extract"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        obs = {
            "image": us_image_uint8,
            "force": np.array([force_val], dtype=np.float32),
            "pose": np.concatenate([ee_pos, ee_orn], dtype=np.float32)
        }
        times["obs_dict"].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        reward = env._compute_reward(action)
        times["reward_calc"].append(time.perf_counter() - t0)
        
        times["total_step"].append(time.perf_counter() - t_start)
        
    print("=" * 40)
    print("PROFILING BREAKDOWN (mean per step):")
    print("=" * 40)
    for k, v in times.items():
        print(f"  {k:<15}: {np.mean(v)*1000:6.2f} ms")
    print("=" * 40)
    
    env.close()

if __name__ == "__main__":
    profile()
