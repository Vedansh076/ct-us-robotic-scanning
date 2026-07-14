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
    load_ct_subject,
    update_probe_model,
    get_probe_pose_from_ee
)
from extract_slice import extract_slice

class RoboticUltrasoundGymEnv(gym.Env):
    """Gymnasium environment for autonomous robotic ultrasound scanning.

    This environment wraps a PyBullet physics simulation of a Franka Emika Panda
    robot arm equipped with a curved ultrasound probe scanning over a 3D patient
    skin mesh. At each step, the probe's contact point is mapped to a voxel index
    inside a TotalSegmentator CT volume, an oblique 2D slice is extracted using
    trilinear interpolation, and (optionally) a U-Net synthesizes a B-mode
    ultrasound image from the CT + bone-mask input.

    The environment follows the standard OpenAI Gymnasium interface (``reset``,
    ``step``, ``close``) and is directly compatible with Stable-Baselines3.

    Action Space
    ------------
    ``spaces.Box(low=-1, high=1, shape=(6,), dtype=float32)``

    Each element is a normalized continuous control:

    =========  ===========  ==============================
    Index      Dimension    Physical Scale
    =========  ===========  ==============================
    0          dx           ×0.01 m  (±1 cm per step)
    1          dy           ×0.01 m  (±1 cm per step)
    2          dz           ×0.01 m  (±1 cm per step)
    3          d_roll       ×0.05 rad (±~2.9° per step)
    4          d_pitch      ×0.05 rad (±~2.9° per step)
    5          d_yaw        ×0.05 rad (±~2.9° per step)
    =========  ===========  ==============================

    Observation Space
    -----------------
    ``spaces.Dict`` with keys:

    - ``"image"``: ``(size, size)`` uint8 — synthesized B-mode ultrasound frame
      (or binary bone mask when ``skip_unet=True``).
    - ``"force"``: ``(1,)`` float32 — estimated normal contact force in Newtons,
      in range [0, 15].
    - ``"pose"``: ``(7,)`` float32 — end-effector pose as
      [x, y, z, qx, qy, qz, qw], in range [−5, 5].

    Reward
    ------
    Three components are summed every step:

    - **Force reward** (``R_f``): +1 for good contact (2–8 N), −2 for no contact,
      proportional penalty for over-pressure (>8 N).
    - **Bone reward** (``R_b``): +1.5 bonus when bone pixels are visible in the
      segmentation slice (>5 positive pixels).
    - **Smoothness penalty** (``R_a``): −0.1 × ‖action‖² to discourage jerk.

    Termination
    -----------
    - ``terminated=True`` if contact force exceeds **12 N** (patient safety limit).
    - ``terminated=True`` if no contact for **30 consecutive steps**.
    - ``truncated=True`` after ``max_episode_steps`` steps.

    Parameters
    ----------
    subject_dir : str
        Path to the TotalSegmentator subject folder. Must contain ``ct.nii.gz``,
        ``bone_label.nii.gz``, ``patient_skin.obj``, and
        ``registration_meta.json``.
    checkpoint_path : str
        Path to the pre-trained U-Net checkpoint (``best_model.pth``). Ignored
        when ``skip_unet=True``.
    device : str
        PyTorch device for U-Net inference. ``"auto"`` selects CUDA if available.
    render_mode : str
        ``"human"`` opens the PyBullet GUI; ``"rgb_array"`` runs headless (faster,
        suitable for RL training).
    max_episode_steps : int
        Maximum number of steps before the episode is truncated. Default: 200.
    mesh_scale : float
        Uniform scale factor applied to the patient skin mesh in PyBullet.
        The registration pipeline automatically compensates for this. Default: 1.0.
    size : int
        Height and width (pixels) of the output ultrasound image. Default: 256.
    pixel_spacing : float
        Physical resolution of each image pixel in mm. Default: 1.0 mm/pixel.
    base_features : int
        U-Net channel width at the first encoder level (doubles each level).
        Must match the checkpoint. Default: 64.
    skip_unet : bool
        If ``True``, bypasses U-Net inference entirely and uses the binary bone
        segmentation mask directly as the ``"image"`` observation. This mode runs
        at ~440 FPS and is used for fast RL training (Strategy 2). Default: True.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 subject_dir="totalseg_patients/s0058",
                 checkpoint_path="model/runs/exp1_2IP/exp1/best_model.pth",
                 device="auto",
                 render_mode="rgb_array",
                 max_episode_steps=200,
                 mesh_scale=1.0,
                 size=256,
                 pixel_spacing=1.0,
                 base_features=64,
                 skip_unet=True):
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
        self.skip_unet = skip_unet
        
        # Load device
        self.device = select_device(device)
        
        # Load CT Volume & Label Volume
        self.ct_volume, self.label_volume, self.spacing, self.volume_center = load_ct_subject(self.subject_dir)
        self.ct_volume = np.ascontiguousarray(self.ct_volume, dtype=np.float32)
        self.label_volume = np.ascontiguousarray(self.label_volume, dtype=np.float32)
        
        # Convert to PyTorch Tensor on target device (GPU or CPU) for fast grid_sample
        self.ct_volume = torch.from_numpy(self.ct_volume).to(self.device)
        self.label_volume = torch.from_numpy(self.label_volume).to(self.device)
        
        # Load U-Net Model (sigmoid)
        if not self.skip_unet:
            self.model = load_unet(self.checkpoint_path, self.device, base_features=self.base_features, dropout=0.0)
        else:
            self.model = None
        
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

        # Disable physical collisions between all links of the robot and the patient body mesh.
        # This prevents kinematic locking and allows smooth, force-guided trajectories.
        for j in range(-1, p.getNumJoints(self.panda_id)):
            p.setCollisionFilterPair(self.panda_id, self.body_id, j, -1, enableCollision=0)
            
        # Force/Episode variables
        self.step_counter = 0
        self.no_contact_counter = 0
        
        # Initial home orientation
        self.home_orn = np.array(p.getQuaternionFromEuler([np.pi, 0, 0]), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a canonical start state.

        Dynamically finds the skin surface Z at the patient mesh centroid via
        raycasting, drives the robot to a high approach pose using IK, then
        lowers it to 8 mm above the surface. Physics is stepped 60 times to
        allow the arm to settle before the first observation is collected.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility (passed to the Gymnasium base class).
        options : dict, optional
            Unused. Reserved for future configuration overrides.

        Returns
        -------
        obs : dict
            Initial observation (image, force, pose).
        info : dict
            Empty info dict (required by Gymnasium API).
        """
        super().reset(seed=seed)
        
        self.step_counter = 0
        self.no_contact_counter = 0
        
        # Calculate dynamic home position snapped to the patient's skin surface
        # Randomize Y position (±8 cm) along the spine with fallback to ensure 100% valid skin contact
        tx = self.body_center[0]
        rand_y = float(self.np_random.uniform(-0.08, 0.08))
        found_body, surface_z = raycast_skin_surface(tx, self.body_center[1] + rand_y, self.body_id)
        if found_body:
            ty = self.body_center[1] + rand_y
        else:
            ty = self.body_center[1]
            found_body, surface_z = raycast_skin_surface(tx, ty, self.body_id)
            
        if found_body:
            # Settle the probe exactly touching the skin surface with 3mm standoff
            self.home_pos = np.array([tx, ty, surface_z + 0.003 + 0.18], dtype=np.float32)
            # High approach target to prevent side collision
            approach_pos = np.array([tx, ty, surface_z + 0.10 + 0.18], dtype=np.float32)
        else:
            self.home_pos = np.array([tx, ty, self.bed_top_z + 0.35], dtype=np.float32)
            approach_pos = self.home_pos.copy()
            
        # Reset robot arm joints directly to the target home pose
        initial_joints = p.calculateInverseKinematics(
            self.panda_id, PANDA_EE_LINK,
            targetPosition=self.home_pos.tolist(),
            targetOrientation=self.home_orn.tolist()
        )
        for jid, v in zip(PANDA_ARM_JOINTS, initial_joints):
            p.resetJointState(self.panda_id, jid, v)
        for fj in (9, 10):
            p.resetJointState(self.panda_id, fj, 0.04)  # Ensure fingers are initialized properly
            
        # Settle the robot in place at home_pos (no large movement)
        drive_panda_to_pose(self.panda_id, self.home_pos, self.home_orn)
        for _ in range(60):
            p.stepSimulation()
            
        obs = self._get_obs()
        
        # Randomize designated sweep direction for this episode: +1.0 (headward) or -1.0 (footward)
        self.sweep_dir = float(self.np_random.choice([-1.0, 1.0]))
        self.last_action = np.zeros(6, dtype=np.float32)
        
        info = {}
        return obs, info
        
    def _get_obs(self):
        """Construct the current observation dictionary.

        Performs three operations in sequence:

        1. **Pose**: Reads the EE link world position and orientation quaternion.
        2. **Force**: Raycasts from the probe tip to estimate skin contact force.
           A linear spring model converts penetration depth to force in Newtons.
        3. **Image**: If the probe is in contact (hit_distance ≤ 5 cm), maps the
           probe tip world position to a CT voxel using
           ``compute_registered_ct_center()``, extracts an oblique CT slice and a
           bone segmentation slice via ``extract_slice()``, and either:
           - Runs U-Net inference to produce a synthetic B-mode ultrasound image
             (``skip_unet=False``), or
           - Binarizes the bone segmentation mask directly (``skip_unet=True``).
           If the probe is not in contact, a black image is returned.

        Returns
        -------
        obs : dict with keys:
            - ``"image"``: (size, size) uint8 — synthetic US image or bone mask.
            - ``"force"``: (1,) float32 — estimated contact force in Newtons.
            - ``"pose"``: (7,) float32 — [x, y, z, qx, qy, qz, qw].
        """
        # 1. Get probe pose (PANDA_EE_LINK state)
        ee_state = p.getLinkState(self.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4], dtype=np.float32)
        ee_orn = np.array(ee_state[5], dtype=np.float32)
        
        # Update probe visual model position
        probe_pos_vis, quaternion_xyzw, probe_contact_position = get_probe_pose_from_ee(
            ee_pos, ee_orn)
        update_probe_model(self.probe_body_id, probe_contact_position, quaternion_xyzw)
        
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
        
        self._last_hit_distance = hit_distance
        # Stiffer spring (k=800 N/m) with 3mm standoff so ideal force (2-6N)
        # is reached with only 0-5mm of surface compression instead of 20mm+.
        force_val = compute_probe_contact_force(hit_distance, desired_standoff=0.003, max_force=10.0)
        # Override with stiffer spring: recompute using k=800 instead of default 200
        if hit_distance is not None:
            compression = 0.003 - hit_distance
            if compression > 0:
                force_val = min(compression * 800.0, 10.0)
            else:
                force_val = 0.0  # Not in contact
        else:
            force_val = 0.0
        self.current_force = force_val
        
        # 3. Extract oblique slices and run U-Net inference
        if found_body and hit_distance is not None and hit_distance <= 0.05:
            # Map world hit position to CT voxel index
            ct_center = compute_registered_ct_center(
                probe_tip_pos, self.reg_body_position, self.reg_body_orientation_matrix,
                self.reg_meta['inv_affine'], self.reg_meta['mesh_centering_offset'],
                self.ct_volume.shape, mesh_scale=self.mesh_scale
            )
            
            # Only extract CT slice if U-Net is needed; skip for Strategy 2 (mask-based)
            if not self.skip_unet:
                ct_slice = extract_slice(
                    self.ct_volume, center=ct_center, quaternion=ee_orn,
                    spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                    inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                    body_orientation_matrix=self.reg_body_orientation_matrix
                )
            else:
                ct_slice = None
            
            seg_slice = extract_slice(
                self.label_volume, center=ct_center, quaternion=ee_orn,
                spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                body_orientation_matrix=self.reg_body_orientation_matrix,
                order=0  # Nearest neighbor for fast label extraction
            )
            
            seg_slice = (seg_slice > 0.5).astype(np.float32)
            
            # Predict ultrasound or use raw bone mask as observation
            if self.skip_unet:
                us_image_uint8 = (seg_slice * 255).astype(np.uint8)
            else:
                us_image = predict_ultrasound(
                    self.model, ct_slice, seg_slice, self.device, is_pix2pix=False, enhance=True
                )
                us_image_uint8 = (us_image * 255).astype(np.uint8)
            
            # Save for reward calculation
            self.last_seg_slice = seg_slice
            self.last_ct_slice = ct_slice
        else:
            us_image_uint8 = np.zeros((self.slice_size, self.slice_size), dtype=np.uint8)
            self.last_seg_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
            self.last_ct_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
        
        obs = {
            "image": us_image_uint8,
            "force": np.array([force_val], dtype=np.float32),
            "pose": np.concatenate([ee_pos, ee_orn], dtype=np.float32)
        }
        return obs
        
    def step(self, action):
        """Apply a 6-DOF action and advance the simulation by one step.

        Action decoding:
          - ``action[:3]`` → position delta, scaled ×0.01 m (±1 cm per axis).
          - ``action[3:]`` → orientation delta in Euler angles, scaled ×0.05 rad.

        The target EE position is clamped to the mattress bounding box (±45 cm
        in X, ±80 cm in Y). Euler angles are clamped to prevent gimbal flips
        (roll kept near π for a downward-facing probe).

        Physics is stepped 5 times after applying joint motor commands to allow
        the robot to converge before the next observation is collected.

        Parameters
        ----------
        action : (6,) float32
            Normalized action vector in [-1, 1]. Elements are:
            [dx, dy, dz, d_roll, d_pitch, d_yaw].

        Returns
        -------
        obs : dict
            New observation (image, force, pose).
        reward : float
            Scalar reward signal from ``_compute_reward()``.
        terminated : bool
            True if a safety limit was exceeded or prolonged no-contact.
        truncated : bool
            True if the episode horizon was reached.
        info : dict
            Diagnostic dict with ``"force"`` (float) and ``"step"`` (int).
        """
        self.step_counter += 1
        
        # Decode and scale action
        # Position displacement limits: dx, dy, dz in [-5 mm, 5 mm] (fine control)
        delta_pos = action[:3] * 0.005
        # Orientation displacement limits: droll, dpitch, dyaw in [-3 deg, 3 deg]
        delta_euler = action[3:] * 0.05
        
        # Get current ee state
        ee_state = p.getLinkState(self.panda_id, PANDA_EE_LINK, computeForwardKinematics=True)
        curr_pos = np.array(ee_state[4])
        curr_orn = np.array(ee_state[5])
        
        # Convert orientation to euler, add delta, convert back
        curr_euler = np.array(p.getEulerFromQuaternion(curr_orn.tolist()))
        target_euler = curr_euler + delta_euler
        # Clamp euler orientation to keep probe nearly perpendicular to skin
        # Tight clamps (±0.15 rad ≈ ±8.6°) ensure probe stays upright
        target_euler[0] = np.clip(target_euler[0], np.pi - 0.15, np.pi + 0.15) # roll (facing down is pi)
        target_euler[1] = np.clip(target_euler[1], -0.15, 0.15)                # pitch
        target_euler[2] = np.clip(target_euler[2], -0.15, 0.15)                # yaw
        
        target_pos = curr_pos + delta_pos
        
        # Keep robot end effector within boundary bounds of mattress
        target_pos[0] = np.clip(target_pos[0], -0.45, 0.45) # X (bed width)
        target_pos[1] = np.clip(target_pos[1], -0.80, 0.80) # Y (bed length)
        target_pos[2] = np.clip(target_pos[2], self.bed_top_z + 0.02, self.bed_top_z + 0.60) # Z
        
        print(f"DEBUG STEP {self.step_counter} | action: {action[:3]} | curr: {curr_pos[1]:.4f} | delta: {delta_pos[1]:.4f} | target: {target_pos[1]:.4f}")
        
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
        if self.current_force > 10.0:
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
        """Compute the scalar reward for the current step.

        The reward has five additive components:

        **Force reward** (``R_f``) — encourages clinically appropriate contact:
          - F = 0 N : ``−1.0`` (penalty, probe lost contact)
          - 0 < F < 2 N: ``+0.5 × F`` (light contact, partial reward)
          - 2 ≤ F ≤ 6 N: ``+2.0`` (ideal clinical contact window)
          - 6 < F ≤ 8 N: ``+0.5`` (acceptable but not ideal)
          - F > 8 N: ``−1.0 × (F − 8)`` (linear penalty, patient safety)

        **Bone reward** (``R_b``) — encourages scanning over anatomical targets:
          - If the segmentation slice contains >5 bone pixels: ``+1.0``.
          - Otherwise: ``0.0``.

        **Smoothness penalty** (``R_a``) — discourages large actions:
          - ``−0.05 × ‖action‖²``  (always applied).

        **Action Jerk penalty** (``R_jerk``) — discourages high-frequency oscillation/flipping:
          - ``−0.1 × ‖action - last_action‖²``

        **Directional Sweep reward** (``R_sweep``) — encourages unidirectional movement along the spine:
          - ``+0.5 × action_y × sweep_dir`` active ONLY when probe is in valid contact (2–8 N) AND over bone.

        Parameters
        ----------
        action : (6,) float32
            The action taken this step.

        Returns
        -------
        reward : float
            Total scalar reward: ``R_f + R_b + R_a + R_jerk + R_sweep``.
        """
        F = self.current_force
        
        # 1. Force Reward: penalizes air gap (F=0) and high forces, rewards [2N, 6N] contact
        if F == 0.0:
            R_f = -1.0  # Penalty for losing contact
        elif F > 8.0:
            R_f = -1.0 * (F - 8.0)  # Linear penalty for pressing too hard
        elif F > 6.0:
            R_f = 0.5   # Acceptable but not ideal
        elif F >= 2.0:
            R_f = 2.0   # Strong reward for ideal clinical contact window
        else:
            # 0 < F < 2
            R_f = 0.5 * F  # Small reward for light contact
            
        # 2. Bone Tracking Reward: positive bonus if bone is visible
        R_b = 0.0
        bone_visible = False
        if self.last_seg_slice is not None and np.sum(self.last_seg_slice) > 5.0:
            R_b = 1.0  # Encourage scanning near bones
            bone_visible = True
            
        # 3. Action Magnitude Penalty
        R_a = -0.05 * np.sum(np.square(action))
        
        # 4. Action Jerk Penalty (prevents high-frequency direction flipping/vibration)
        if hasattr(self, "last_action") and self.last_action is not None:
            R_jerk = -0.1 * np.sum(np.square(action - self.last_action))
        else:
            R_jerk = 0.0
        self.last_action = action.copy()
        
        # 5. Directional Sweep Motion Reward along spine (action[1] is dy)
        R_sweep = 0.0
        if F >= 2.0 and F <= 8.0 and bone_visible:
            sweep_dir = getattr(self, "sweep_dir", 1.0)
            # Reward moving in designated direction, punish reversing
            R_sweep = 0.5 * (float(action[1]) * sweep_dir)
        
        return R_f + R_b + R_a + R_jerk + R_sweep
        
    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
