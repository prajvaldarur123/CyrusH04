"""
Cyrus Humanoid - Unified Skill-Unlocking Environment (SOTA Unitree G1 Benchmark)
Supports progressive skill acquisition:
  Skill 1: Quiet Stand & Postural Homeostasis (0.0 m/s push)
  Skill 2: Ankle Push Recovery (0.10 m/s omnidirectional shoves)
  Skill 3: Hip Push Recovery & CoM Lowering (0.25 m/s shoves)
  Skill 4: Dynamic Squat & Height Modulation (tracking variable height commands)
  Skill 5: Heavy Push & Impact Dissipation (0.40 m/s shoves)
Uses the exact Unitree G1 bounded Gaussian reward suite (zero cliff penalties).
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

PROJECT_DIR = r"C:\Users\prajval\robotics"
XML_PATH = os.path.join(PROJECT_DIR, "cyrus", "models", "cyrus.xml")

SKILL_NAMES = {
    1: "Quiet Stand & Postural Homeostasis (0.0 m/s push)",
    2: "Ankle Strategy Push Recovery (0.10 m/s shoves)",
    3: "Hip Strategy Push Recovery & CoM Drop (0.25 m/s shoves)",
    4: "Dynamic Squat & Pelvis Height Modulation (0.60m - 0.84m)",
    5: "Heavy Push & High-Impact Dissipation (0.40 m/s shoves)"
}

class CyrusHumanoidSkillEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, frame_skip=10, max_episode_steps=500):
        super().__init__()
        self.frame_skip = frame_skip  # 10 substeps * 0.002s = 0.02s (50 Hz control policy)
        self.max_episode_steps = max_episode_steps  # 500 steps = 10.0 seconds per episode

        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"MuJoCo model not found at {XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # 12 Lower-Body Driven Joints (SOTA Unitree G1 formulation)
        # 15..20 (Right Leg), 21..26 (Left Leg)
        self.leg_act_indices = list(range(15, 27))
        self.num_actions = 12

        self.act_min = self.model.actuator_ctrlrange[self.leg_act_indices, 0]
        self.act_max = self.model.actuator_ctrlrange[self.leg_act_indices, 1]
        self.act_mid = 0.5 * (self.act_max + self.act_min)
        self.act_range = 0.5 * (self.act_max - self.act_min)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation Space (48 dims):
        # 3 (projected gravity) + 3 (torso ang vel) + 3 (torso lin vel) +
        # 12 (leg qpos err) + 12 (leg qvel) + 12 (prev act) + 2 (contacts) + 1 (height command) = 48
        self.obs_dim = 48
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Body & Site IDs
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        leg_names = [
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]
        self.leg_qpos_adr = []
        self.leg_qvel_adr = []
        for name in leg_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.leg_qpos_adr.append(self.model.jnt_qposadr[jid])
            self.leg_qvel_adr.append(self.model.jnt_dofadr[jid])

        # Nominal Standing Posture (Pelvis Z = 0.84m)
        self.nominal_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self.nominal_qpos[2] = 0.84
        self.nominal_qpos[3] = 1.0

        # Upper Body Athletic Carriage (Shoulders forward, elbows in guard, fingers curled)
        self.upper_ctrl_default = np.zeros(self.model.nu, dtype=np.float32)
        r_sh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_act")
        l_sh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_act")
        r_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_act")
        l_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_act")
        if r_sh != -1: self.upper_ctrl_default[r_sh] = 0.15
        if l_sh != -1: self.upper_ctrl_default[l_sh] = 0.15
        if r_el != -1: self.upper_ctrl_default[r_el] = -0.50
        if l_el != -1: self.upper_ctrl_default[l_el] = -0.50

        for f_name in ["right_finger_act", "right_thumb_act", "left_finger_act", "left_thumb_act"]:
            f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f_name)
            if f_id != -1:
                self.upper_ctrl_default[f_id] = 0.35

        self.nominal_leg_angles = np.zeros(12, dtype=np.float32)

        # Active Skill State
        self.active_skill = 1
        self.push_mag = 0.0
        self.target_height = 0.84
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0

    def set_skill(self, skill_id):
        self.active_skill = int(skill_id)
        if self.active_skill == 1:
            self.push_mag = 0.0
            self.target_height = 0.84
        elif self.active_skill == 2:
            self.push_mag = 0.10
            self.target_height = 0.84
        elif self.active_skill == 3:
            self.push_mag = 0.25
            self.target_height = 0.84
        elif self.active_skill == 4:
            self.push_mag = 0.10
            # Height will modulate dynamically during episode
        elif self.active_skill == 5:
            self.push_mag = 0.40
            self.target_height = 0.84

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Subtle stance exploration jitter (+/- 1.5cm height, +/- 0.03 rad leg angles)
        self.data.qpos[:] = self.nominal_qpos.copy()
        self.data.qpos[2] = 0.84 + np.random.uniform(-0.015, 0.015)
        for adr in self.leg_qpos_adr:
            self.data.qpos[adr] += np.random.uniform(-0.03, 0.03)

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.upper_ctrl_default.copy()
        mujoco.mj_forward(self.model, self.data)

        self.target_height = 0.84
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        d = self.data
        torso_rot = d.xmat[self.torso_id].reshape(3, 3)
        proj_gravity = torso_rot.T @ np.array([0.0, 0.0, -1.0])
        torso_cvel = d.cvel[self.torso_id]
        torso_ang_vel = torso_rot.T @ torso_cvel[:3]
        torso_lin_vel = torso_rot.T @ torso_cvel[3:]

        leg_qpos = np.array([d.qpos[adr] for adr in self.leg_qpos_adr], dtype=np.float32)
        leg_qpos_err = leg_qpos - self.nominal_leg_angles
        leg_qvel = np.array([d.qvel[adr] for adr in self.leg_qvel_adr], dtype=np.float32)

        prev_act = self.prev_action.copy()
        r_sole_z = float(d.site_xpos[self.r_sole_id, 2])
        l_sole_z = float(d.site_xpos[self.l_sole_id, 2])
        r_contact = 1.0 if r_sole_z < 0.045 else 0.0
        l_contact = 1.0 if l_sole_z < 0.045 else 0.0

        obs = np.concatenate([
            proj_gravity,               # 3
            torso_ang_vel,              # 3
            torso_lin_vel,              # 3
            leg_qpos_err,               # 12
            leg_qvel * 0.1,             # 12
            prev_act,                   # 12
            [r_contact, l_contact],     # 2
            [self.target_height - 0.84] # 1 (height command relative to nominal)
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Dynamic Height Modulation for Skill 4 (Squat modulation)
        if self.active_skill == 4:
            # Smooth sine wave height target between 0.65m and 0.84m
            t_s = self._step_count * 0.02
            self.target_height = 0.745 + 0.095 * np.cos(2.0 * np.pi * t_s / 4.0)
        else:
            self.target_height = 0.84

        # Action Mapping to 12 Leg Actuators (PD target offsets)
        target_ctrl = self.upper_ctrl_default.copy()
        leg_targets = self.act_mid + action * (0.35 * self.act_range)
        target_ctrl[self.leg_act_indices] = leg_targets
        self.data.ctrl[:] = target_ctrl

        # Omnidirectional Perturbation Push (Skills 2, 3, 5)
        # Injects random 360-degree impulse at steps 100, 250, and 380
        if self.push_mag > 0.001 and (self._step_count in [100, 250, 380]):
            shove_theta = np.random.uniform(0, 2 * np.pi)
            self.data.qvel[0] += self.push_mag * np.cos(shove_theta)
            self.data.qvel[1] += self.push_mag * np.sin(shove_theta)
            self.data.qvel[4] += np.random.uniform(-0.4, 0.4) * self.push_mag

        # Step Physics (10 substeps * 0.002s = 0.02s per step)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # State Telemetry
        pelvis_z = float(self.data.qpos[2])
        torso_up = float(self.data.xmat[self.torso_id].reshape(3, 3)[2, 2])
        lin_vel_xy = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
        ang_vel_rp = float(np.hypot(self.data.qvel[3], self.data.qvel[4]))
        r_sole_z = float(self.data.site_xpos[self.r_sole_id, 2])
        l_sole_z = float(self.data.site_xpos[self.l_sole_id, 2])

        # -----------------------------------------------------------------
        # UNITREE G1 BOUNDED REWARD SUITE (ZERO CLIFF PENALTIES)
        # -----------------------------------------------------------------
        # 1. Height Tracking Reward (Gaussian centered on active target height)
        r_height = float(np.exp(-((pelvis_z - self.target_height) ** 2) / 0.04))

        # 2. Upright Trunk: Projection on vertical
        r_upright = float(np.clip(torso_up, 0.0, 1.0))

        # 3. Stay Put: Horizontal drift suppression
        r_stay_put = float(np.exp(-(lin_vel_xy ** 2) / 0.25))

        # 4. Alive Survival Incentive
        r_alive = 1.0

        # Regularization & Smoothness Penalties (Small, quadratic)
        p_roll_pitch = 0.02 * (ang_vel_rp ** 2)
        p_torque = 0.0001 * float(np.sum(np.square(self.data.qfrc_actuator[self.leg_act_indices])))
        p_action_rate = 0.01 * float(np.sum(np.square(action - self.prev_action)))
        p_vert_vel = 0.02 * float(self.data.qvel[2] ** 2)
        leg_qpos = np.array([self.data.qpos[adr] for adr in self.leg_qpos_adr], dtype=np.float32)
        p_pose = 0.01 * float(np.sum(np.square(leg_qpos - self.nominal_leg_angles)))

        reward = (
            0.018 * r_height +
            0.018 * r_upright +
            0.009 * r_alive +
            0.008 * r_stay_put -
            p_roll_pitch - p_torque - p_action_rate - p_vert_vel - p_pose
        )

        # Fall Termination
        min_allowed_z = 0.45 if self.active_skill == 4 else 0.50
        is_fall = (pelvis_z < min_allowed_z) or (torso_up < 0.55) or (max(r_sole_z, l_sole_z) > 0.35)
        terminated = bool(is_fall)
        truncated = bool(self._step_count >= self.max_episode_steps)

        if terminated:
            reward -= 1.07  # Unitree single terminal fall penalty

        self.prev_action = action.copy()
        obs = self._get_obs()

        info = {
            "skill": self.active_skill,
            "skill_name": SKILL_NAMES.get(self.active_skill, "Custom"),
            "height": pelvis_z,
            "target_height": self.target_height,
            "upright": torso_up,
            "is_fall": is_fall,
            "duration_s": self._step_count * 0.02,
            "push_mag": self.push_mag
        }
        return obs, float(reward), terminated, truncated, info
