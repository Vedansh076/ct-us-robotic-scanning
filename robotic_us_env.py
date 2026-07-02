import os
import sys
import time
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import torch
import gymnasium as gym
from gymnasium import spaces

# Add current folder to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from live_unet_demo import (
    create_panda_robot,
    create_probe_model,
    create_registered_body_mesh,
    load_registration_meta,
    compute_registered_ct_center,
    load_unet,
    predict_ultrasound,
    raycast_skin_surface,
    compute_probe_contact_force,
    drive_panda_to_pose,
    PANDA_EE_LINK,
    PANDA_ARM_JOINTS,
    BED_CENTER,
    BED_MATTRESS_TOP_Z,
    PROBE_TIP_FROM_EE,
    PROBE_QUAT_FROM_EE,
    select_device,
    resolve_subject_dir,
    load_ct_subject
)
from extract_slice import extract_slice

class RoboticUltrasoundGymEnv(gym.Env):
    """
    Gymnasium Environment for Robotic Ultrasound Scanning using TotalSegmentator CT data.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self,
                 subject_dir="totalseg_patients/s0058",
                 checkpoint_path="model/runs/exp1_2IP/exp1/best_model.pth",
                 device="auto",
                 render_mode="human",
                 max_episode_steps=200,
                 mesh_scale=1.0,
                 size=256,
                 pixel_spacing=1.0,
                 base_features=64):
        super().__init__()
        
        self.subject_dir = Path(resolve_subject_dir(subject_dir))
        self.checkpoint_path = Path(checkpoint_path)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.mesh_scale_val = mesh_scale
        self.mesh_scale = np.array([mesh_scale]*3, dtype=np.float64)
        self.slice_size = size
        self.pixel_spacing = pixel_spacing
        self.base_features = base_features
        
        # Load device
        self.device = select_device(device)
        
        # Load CT Volume & Label Volume
        self.ct_volume, self.label_volume, self.spacing, self.volume_center = load_ct_subject(self.subject_dir)
        
        # Load U-Net Model (sigmoid)
        self.model = load_unet(self.checkpoint_path, self.device, base_features=self.base_features, dropout=0.0)
        
        # Action space: 6-DOF continuous relative controls in [-1, 1]
        # [dx, dy, dz, droll, dpitch, dyaw]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: Dict containing B-mode image, contact force, and pose
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.slice_size, self.slice_size), dtype=np.uint8),
            "force": spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
            "pose": spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32)
        })
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Initialize hospital scene
        from live_unet_demo import create_hospital_room
        self.bed_top_z = create_hospital_room()
        
        # Load patient mesh
        self.body_id, self.reg_body_position, self.reg_body_orientation_matrix, self.reg_meta = \
            create_registered_body_mesh(self.subject_dir, self.bed_top_z, self.mesh_scale)
            
        # Get patient centroid and extent
        from live_unet_demo import get_body_bounds
        self.mesh_bounds_min, self.mesh_bounds_max, self.body_center, self.body_extent = get_body_bounds(self.body_id)
        
        # Load Panda Robot
        self.panda_id = create_panda_robot()
        
        # Load Probe Model
        self.probe_body_id = create_probe_model()
        
        # Hide gripper fingers
        for link in [9, 10]:
            p.changeVisualShape(self.panda_id, link, rgbaColor=[0.0, 0.0, 0.0, 0.0])
            
        # Force/Episode variables
        self.step_counter = 0
        self.no_contact_counter = 0
        
        # Initial home orientation
        self.home_orn = np.array(p.getQuaternionFromEuler([np.pi, 0, 0]), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_counter = 0
        self.no_contact_counter = 0
        
        # Reset robot arm to home joint states
        home_joints = [0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.78]
        for jid, v in zip(PANDA_ARM_JOINTS, home_joints):
            p.resetJointState(self.panda_id, jid, v)
        for fj in (9, 10):
            p.resetJointState(self.panda_id, fj, 0.04)
            
        # Calculate dynamic home position snapped to the patient's skin surface (chest center)
        tx, ty = self.body_center[0], self.body_center[1]
        found_body, surface_z = raycast_skin_surface(tx, ty, self.body_id)
        if found_body:
            # Settle the probe exactly touching the skin surface with 8mm standoff
            self.home_pos = np.array([tx, ty, surface_z + 0.008 + 0.18], dtype=np.float32)
        else:
            self.home_pos = np.array([tx, ty, self.bed_top_z + 0.35], dtype=np.float32)
            
        # Drive robot to initial target pose and step a few times to settle
        drive_panda_to_pose(self.panda_id, self.home_pos, self.home_orn)
        for _ in range(60):
            p.stepSimulation()
            
        obs = self._get_obs()
        info = {}
        return obs, info
        
    def _get_obs(self):
        # 1. Get probe pose (PANDA_EE_LINK state)
        ee_state = p.getLinkState(self.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4], dtype=np.float32)
        ee_orn = np.array(ee_state[5], dtype=np.float32)
        
        # 2. Estimate contact force via raycasting
        found_body, surface_z = raycast_skin_surface(ee_pos[0], ee_pos[1], self.body_id)
        hit_distance = None
        probe_tip_pos = ee_pos.copy()
        
        if found_body:
            # Distance from probe tip to skin surface
            # Probe tip is 18cm below EE along Z-axis in local frame
            rm = np.array(p.getMatrixFromQuaternion(ee_orn.tolist())).reshape(3, 3)
            probe_tip_pos = ee_pos + rm @ PROBE_TIP_FROM_EE
            
            # Perform raycast from 5cm above the probe tip along probe direction
            ray_start = probe_tip_pos - rm[:, 2] * 0.05
            ray_end = probe_tip_pos + rm[:, 2] * 0.05
            res = p.rayTest(ray_start.tolist(), ray_end.tolist())
            if len(res) > 0 and res[0][0] == self.body_id:
                frac = res[0][2]
                hit_distance = frac * 0.10 - 0.05
            else:
                hit_distance = probe_tip_pos[2] - surface_z
        
        force_val = compute_probe_contact_force(hit_distance, desired_standoff=0.008, max_force=10.0)
        self.current_force = force_val
        
        # 3. Extract oblique slices and run U-Net inference
        if found_body and hit_distance is not None and hit_distance <= 0.05:
            # Map world hit position to CT voxel index
            ct_center = compute_registered_ct_center(
                probe_tip_pos, self.reg_body_position, self.reg_body_orientation_matrix,
                self.reg_meta['inv_affine'], self.reg_meta['mesh_centering_offset'],
                self.ct_volume.shape, mesh_scale=self.mesh_scale
            )
            
            ct_slice = extract_slice(
                self.ct_volume, center=ct_center, quaternion=ee_orn,
                spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                body_orientation_matrix=self.reg_body_orientation_matrix
            )
            
            seg_slice = extract_slice(
                self.label_volume, center=ct_center, quaternion=ee_orn,
                spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                body_orientation_matrix=self.reg_body_orientation_matrix
            )
            
            seg_slice = (seg_slice > 0.5).astype(np.float32)
            
            # Predict ultrasound image
            us_image = predict_ultrasound(
                self.model, ct_slice, seg_slice, self.device, is_pix2pix=False, enhance=True
            )
            
            # Save for reward calculation
            self.last_seg_slice = seg_slice
            self.last_ct_slice = ct_slice
        else:
            us_image = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
            self.last_seg_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
            self.last_ct_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
            
        us_image_uint8 = (us_image * 255).astype(np.uint8)
        
        obs = {
            "image": us_image_uint8,
            "force": np.array([force_val], dtype=np.float32),
            "pose": np.concatenate([ee_pos, ee_orn], dtype=np.float32)
        }
        return obs
        
    def step(self, action):
        self.step_counter += 1
        
        # Decode and scale action
        # Position displacement limits: dx, dy, dz in [-1 cm, 1 cm]
        delta_pos = action[:3] * 0.01
        # Orientation displacement limits: droll, dpitch, dyaw in [-3 deg, 3 deg]
        delta_euler = action[3:] * 0.05
        
        # Get current ee state
        ee_state = p.getLinkState(self.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        curr_pos = np.array(ee_state[4])
        curr_orn = np.array(ee_state[5])
        
        # Convert orientation to euler, add delta, convert back
        curr_euler = np.array(p.getEulerFromQuaternion(curr_orn.tolist()))
        target_euler = curr_euler + delta_euler
        # Clamp euler orientation to prevent flip-overs
        # Keep pitch within [-np.pi/4, np.pi/4] and roll within [np.pi - np.pi/4, np.pi + np.pi/4]
        target_euler[0] = np.clip(target_euler[0], np.pi - 0.4, np.pi + 0.4) # roll (facing down is pi)
        target_euler[1] = np.clip(target_euler[1], -0.4, 0.4)                # pitch
        target_euler[2] = np.clip(target_euler[2], -0.6, 0.6)                # yaw
        
        target_pos = curr_pos + delta_pos
        
        # Keep robot end effector within boundary bounds of mattress
        target_pos[0] = np.clip(target_pos[0], -0.45, 0.45) # X (bed width)
        target_pos[1] = np.clip(target_pos[1], -0.80, 0.80) # Y (bed length)
        target_pos[2] = np.clip(target_pos[2], self.bed_top_z + 0.02, self.bed_top_z + 0.50) # Z
        
        target_orn = np.array(p.getQuaternionFromEuler(target_euler.tolist()))
        
        # Drive robot using positional control
        drive_panda_to_pose(self.panda_id, target_pos, target_orn)
        
        # Step simulation to let robot move
        for _ in range(5):
            p.stepSimulation()
            
        # Get new observation
        obs = self._get_obs()
        
        # Calculate Reward
        reward = self._compute_reward(action)
        
        # Check Safety / Terminations
        terminated = False
        truncated = False
        
        # 1. Over-force safety limit (patient safety threshold)
        if self.current_force > 12.0:
            terminated = True
            
        # 2. No contact limit (30 consecutive steps of F = 0)
        if self.current_force == 0.0:
            self.no_contact_counter += 1
            if self.no_contact_counter >= 30:
                terminated = True
        else:
            self.no_contact_counter = 0
            
        # 3. Max step limit
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            
        info = {
            "force": float(self.current_force),
            "step": self.step_counter
        }
        
        return obs, reward, terminated, truncated, info
        
    def _compute_reward(self, action):
        F = self.current_force
        
        # 1. Force Reward: penalizes air gap (F=0) and high forces, rewards [2N, 8N] contact
        if F == 0.0:
            R_f = -2.0  # Big penalty for losing contact
        elif F > 8.0:
            R_f = -1.0 * (F - 8.0) # Penalty for pressing too hard
        elif F >= 2.0 and F <= 8.0:
            R_f = 1.0  # Reward for good contact
        else:
            # 0 < F < 2
            R_f = 0.5 * F  # Small reward for light contact
            
        # 2. Bone Tracking Reward: positive bonus if bone is visible
        R_b = 0.0
        if self.last_seg_slice is not None and np.sum(self.last_seg_slice) > 5.0:
            R_b = 1.5  # Encourage scanning near bones
            
        # 3. Action Smoothing Penalty
        R_a = -0.1 * np.sum(np.square(action))
        
        return R_f + R_b + R_a
        
    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
