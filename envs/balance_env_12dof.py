"""
Cyrus Humanoid - 12-DoF Lower-Body Dynamic Balance Environment
Engineered to replicate the Unitree G1 SOTA 3-minute convergence benchmark:
1. 12 Driven Joints (Lower Body: Hip Yaw/Roll/Pitch, Knee, Ankle Pitch/Roll x 2)
2. Upper body locked in athletic defensive guard carriage
3. Exact Unitree G1 Bounded Reward Suite (height, upright, alive, stay-put, smooth regularizers)
4. Automated Push-Perturbation Curriculum (0.0 -> 0.1 -> 0.2 -> 0.3+ m/s)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

PROJECT_DIR = r"C:\Users\prajval\robotics"
XML_PATH = os.path.join(PROJECT_DIR, "cyrus", "models", "cyrus.xml")

class CyrusBalance12DoFEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, frame_skip=10, max_episode_steps=500):
        super().__init__()
        self.frame_skip = frame_skip  # 10 substeps * 0.002s = 0.02s per step (50Hz policy)
        self.max_episode_steps = max_episode_steps  # 500 steps * 0.02s = 10.0 seconds

        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"MuJoCo model not found at {XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # 12 Lower-Body Actuators: indices 15 to 26 in cyrus.xml
        # right_hip_yaw(15), right_hip_roll(16), right_hip_pitch(17), right_knee(18), right_ankle_pitch(19), right_ankle_roll(20)
        # left_hip_yaw(21), left_hip_roll(22), left_hip_pitch(23), left_knee(24), left_ankle_pitch(25), left_ankle_roll(26)
        self.leg_act_indices = list(range(15, 27))
        self.num_actions = len(self.leg_act_indices)  # Exactly 12

        # Actuator limits
        self.act_min = self.model.actuator_ctrlrange[self.leg_act_indices, 0]
        self.act_max = self.model.actuator_ctrlrange[self.leg_act_indices, 1]
        self.act_mid = 0.5 * (self.act_max + self.act_min)
        self.act_range = 0.5 * (self.act_max - self.act_min)

        # Action Space: 12-dimensional delta normalized between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

        # Observation Space: 47 dimensions
        # 3 (torso projected gravity) + 3 (torso omega) + 3 (torso vel) +
        # 12 (leg qpos error) + 12 (leg qvel) + 12 (prev action) + 2 (foot contacts) = 47
        self.obs_dim = 47
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Body & Site IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Map leg joint qpos / qvel addresses
        leg_joint_names = [
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]
        self.leg_qpos_adr = []
        self.leg_qvel_adr = []
        for name in leg_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.leg_qpos_adr.append(self.model.jnt_qposadr[jid])
            self.leg_qvel_adr.append(self.model.jnt_dofadr[jid])

        # Nominal Standing Pose (Upper body in athletic guard, legs upright)
        self.nominal_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self.nominal_qpos[2] = 0.84  # Pelvis height
        self.nominal_qpos[3] = 1.0   # Quaternion w

        # Athletic Guard Carriage for Upper Body
        # Slight elbow flexion, shoulders forward, fingers curled
        self.full_ctrl_default = np.zeros(self.model.nu, dtype=np.float32)
        # Shoulders & elbows
        r_sh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_act")
        l_sh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_act")
        r_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_act")
        l_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_act")
        if r_sh != -1: self.full_ctrl_default[r_sh] = 0.15
        if l_sh != -1: self.full_ctrl_default[l_sh] = 0.15
        if r_el != -1: self.full_ctrl_default[r_el] = -0.50
        if l_el != -1: self.full_ctrl_default[l_el] = -0.50

        # Fingers slightly curled
        for f_name in ["right_finger_act", "right_thumb_act", "left_finger_act", "left_thumb_act"]:
            f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f_name)
            if f_id != -1:
                self.full_ctrl_default[f_id] = 0.35

        # Nominal Leg Angles
        self.nominal_leg_angles = np.zeros(12, dtype=np.float32)

        # Perturbation curriculum
        self.push_curriculum_vel = 0.0  # m/s (auto-ramps from 0.0 to 0.35 m/s)
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0

    def set_push_level(self, vel):
        """Sets the perturbation shove magnitude (m/s)."""
        self.push_curriculum_vel = float(vel)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set initial standing qpos with subtle exploration noise
        self.data.qpos[:] = self.nominal_qpos.copy()
        # Small jitter on standing height and leg joints to prevent brittle memorization
        self.data.qpos[2] = 0.84 + np.random.uniform(-0.015, 0.015)
        for adr in self.leg_qpos_adr:
            self.data.qpos[adr] += np.random.uniform(-0.03, 0.03)

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.full_ctrl_default.copy()
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Projected Gravity Vector in Torso Frame (3)
        torso_rot = self.data.xmat[self.torso_id].reshape(3, 3)
        proj_gravity = torso_rot.T @ np.array([0.0, 0.0, -1.0])

        # 2. Torso Angular Velocity in Body Frame (3)
        # MuJoCo cvel has [rot_vel(3), lin_vel(3)]
        torso_cvel = self.data.cvel[self.torso_id]
        torso_ang_vel = torso_rot.T @ torso_cvel[:3]

        # 3. Torso Linear Velocity in Body Frame (3)
        torso_lin_vel = torso_rot.T @ torso_cvel[3:]

        # 4. Leg Joint Positions Error (12)
        leg_qpos = np.array([self.data.qpos[adr] for adr in self.leg_qpos_adr], dtype=np.float32)
        leg_qpos_err = leg_qpos - self.nominal_leg_angles

        # 5. Leg Joint Velocities (12)
        leg_qvel = np.array([self.data.qvel[adr] for adr in self.leg_qvel_adr], dtype=np.float32)

        # 6. Previous Action (12)
        prev_act = self.prev_action.copy()

        # 7. Foot Contacts (2)
        r_sole_z = float(self.data.site_xpos[self.r_sole_id, 2])
        l_sole_z = float(self.data.site_xpos[self.l_sole_id, 2])
        r_contact = 1.0 if r_sole_z < 0.045 else 0.0
        l_contact = 1.0 if l_sole_z < 0.045 else 0.0

        obs = np.concatenate([
            proj_gravity,       # 3
            torso_ang_vel,      # 3
            torso_lin_vel,      # 3
            leg_qpos_err,       # 12
            leg_qvel * 0.1,     # 12 (scaled)
            prev_act,           # 12
            [r_contact, l_contact] # 2
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Map 12 actions to lower body actuator targets (relative PD delta)
        target_ctrl = self.full_ctrl_default.copy()
        # Scale action by actuator range (0.35 modulation scale for compliance)
        leg_targets = self.act_mid + action * (0.35 * self.act_range)
        target_ctrl[self.leg_act_indices] = leg_targets
        self.data.ctrl[:] = target_ctrl

        # Push Perturbation Injection (Push Curriculum)
        # Apply occasional impulsive shove to torso to teach reactive balance recovery
        if self.push_curriculum_vel > 0.001 and (self._step_count == 100 or self._step_count == 250):
            shove_theta = np.random.uniform(0, 2 * np.pi)
            self.data.qvel[0] += self.push_curriculum_vel * np.cos(shove_theta)
            self.data.qvel[1] += self.push_curriculum_vel * np.sin(shove_theta)
            self.data.qvel[4] += np.random.uniform(-0.5, 0.5) * self.push_curriculum_vel

        # Step Physics (10 substeps * 0.002s = 0.02s per 50Hz RL step)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Telemetry & Kinematics
        pelvis_z = float(self.data.qpos[2])
        torso_up = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        lin_vel_xy = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
        ang_vel_rp = float(np.hypot(self.data.qvel[3], self.data.qvel[4]))
        r_sole_z = float(self.data.site_xpos[self.r_sole_id, 2])
        l_sole_z = float(self.data.site_xpos[self.l_sole_id, 2])

        # -------------------------------------------------------------
        # EXACT UNITREE G1 BOUNDED REWARD FUNCTION SUITE
        # -------------------------------------------------------------
        # 1. Height Reward: Gaussian centered at target standing height (0.84m)
        r_height = float(np.exp(-((pelvis_z - 0.84) ** 2) / 0.04))

        # 2. Upright Trunk: Projection of torso z-axis on global vertical
        r_upright = float(np.clip(torso_up, 0.0, 1.0))

        # 3. Stay Put: Suppress translational drift
        r_stay_put = float(np.exp(-(lin_vel_xy ** 2) / 0.25))

        # 4. Alive Bonus (constant survival incentive)
        r_alive = 1.0

        # Regularization & Smoothness Penalties (Small, non-disruptive)
        p_roll_pitch = 0.02 * (ang_vel_rp ** 2)
        p_torque = 0.0001 * float(np.sum(np.square(self.data.qfrc_actuator[self.leg_act_indices])))
        p_action_rate = 0.01 * float(np.sum(np.square(action - self.prev_action)))
        p_vert_vel = 0.02 * float(self.data.qvel[2] ** 2)
        leg_qpos = np.array([self.data.qpos[adr] for adr in self.leg_qpos_adr], dtype=np.float32)
        p_pose = 0.01 * float(np.sum(np.square(leg_qpos - self.nominal_leg_angles)))

        # Scaled per-step reward (sums to ~25.0 over 500 steps / 10 seconds)
        reward = (
            0.018 * r_height +
            0.018 * r_upright +
            0.009 * r_alive +
            0.008 * r_stay_put -
            p_roll_pitch - p_torque - p_action_rate - p_vert_vel - p_pose
        )

        # Fall Termination
        # Fall occurs if pelvis drops below 0.50m or torso tilts beyond 55 deg
        is_fall = (pelvis_z < 0.50) or (torso_up < 0.55) or (max(r_sole_z, l_sole_z) > 0.35)
        terminated = bool(is_fall)
        truncated = bool(self._step_count >= self.max_episode_steps)

        if terminated:
            reward -= 1.07  # Exact Unitree single terminal fall penalty

        self.prev_action = action.copy()
        obs = self._get_obs()

        info = {
            "height": pelvis_z,
            "upright": torso_up,
            "is_fall": is_fall,
            "steps": self._step_count,
            "duration_s": self._step_count * 0.02,
            "push_vel": self.push_curriculum_vel
        }
        return obs, float(reward), terminated, truncated, info
