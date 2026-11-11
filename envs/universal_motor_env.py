import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CyrusUniversalMotorEnv(gym.Env):
    """
    Cyrus Universal Whole-Body Motor Intelligence Environment.
    Learns a single unified policy conditioned on an active Target Command Vector:
      [v_x_target, v_y_target, omega_yaw_target, z_pelvis_target, mode]
    Capable of executing:
      - Mode 0: STAND (Zero-drift inverted-pendulum balance)
      - Mode 1: WALK  (Omnidirectional walking 0.5 - 1.5 m/s)
      - Mode 2: RUN   (High-speed sprinting 2.0 - 3.8 m/s)
      - Mode 3: JUMP  (Crouch, explode, and absorb to Z=1.15m)
      - Mode 4: RECOVER (Autonomous self-righting from any arbitrary fall)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, frame_skip=10, render_mode=None):
        super().__init__()

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "cyrus.xml")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        # Store baseline physics parameters for domain randomization
        self.base_geom_friction = self.model.geom_friction.copy()
        self.base_body_mass = self.model.body_mass.copy()

        # Cache anatomical IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_upper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.right_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self.left_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.left_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Joints
        self.r_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        self.l_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.r_knee_qpos_idx = self.model.jnt_qposadr[self.r_knee_id]
        self.l_knee_qpos_idx = self.model.jnt_qposadr[self.l_knee_id]

        self.r_hip_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")
        self.l_hip_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")
        self.r_ank_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_pitch")
        self.l_ank_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch")
        self.r_hip_qpos_idx = self.model.jnt_qposadr[self.r_hip_p_id]
        self.l_hip_qpos_idx = self.model.jnt_qposadr[self.l_hip_p_id]
        self.r_ank_qpos_idx = self.model.jnt_qposadr[self.r_ank_p_id]
        self.l_ank_qpos_idx = self.model.jnt_qposadr[self.l_ank_p_id]

        # Action Space: 27 continuous joint delta targets
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        # Mapped 27 base body joint addresses (excludes unactuated fingers)
        self.jnt_ids = [self.model.actuator_trnid[i, 0] for i in range(27)]
        self.qpos_adr = [self.model.jnt_qposadr[jid] for jid in self.jnt_ids]
        self.dof_adr = [self.model.jnt_dofadr[jid] for jid in self.jnt_ids]

        # Observation Space: 72 base proprioceptive sensors + 5 active command features = 77
        obs_dim = 77
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.max_episode_steps = 450
        self._step_count = 0
        self.current_ctrl = np.zeros(27, dtype=np.float32)
        self.prev_action = np.zeros(27, dtype=np.float32)

        # Neutral standing control reference
        self.neutral_standing_ctrl = np.zeros(27, dtype=np.float32)
        r_sh_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_act")
        l_sh_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_act")
        r_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_act")
        l_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_act")
        if r_sh_p != -1: self.neutral_standing_ctrl[r_sh_p] = 0.05
        if l_sh_p != -1: self.neutral_standing_ctrl[l_sh_p] = 0.05
        if r_el != -1: self.neutral_standing_ctrl[r_el] = -0.15
        if l_el != -1: self.neutral_standing_ctrl[l_el] = -0.15

        # Active Command Vector: [vx_cmd, vy_cmd, omega_cmd, z_cmd, mode]
        self.command = np.array([0.0, 0.0, 0.0, 0.84, 0.0], dtype=np.float32)
        self.command_timer = 0
        self.consecutive_standing_steps = 0
        self.target_stand_steps = 250  # 5.0 seconds at 50Hz
        self.max_episode_steps = 600   # 12.0s: 3.0s initial stand + knockdown + 5.0s hold

        from cyrus.controller.kinematic_observer import CyrusKinematicObserver
        self.analyzer = CyrusKinematicObserver(self.model)

        self.prev_z = 0.84
        self.max_z_in_ep = 0.84
        self.has_succeeded = False

    def _sample_new_command(self):
        """Samples a new behavior command to teach flexible multi-task coordination."""
        mode_choice = float(self.np_random.random())
        if mode_choice < 0.35:
            # Mode 0: STAND (Hold balance)
            self.command = np.array([0.0, 0.0, 0.0, 0.84, 0.0], dtype=np.float32)
        elif mode_choice < 0.65:
            # Mode 1: WALK (0.5 to 1.5 m/s)
            vx = float(self.np_random.uniform(0.5, 1.5))
            vy = float(self.np_random.uniform(-0.25, 0.25))
            omega = float(self.np_random.uniform(-0.4, 0.4))
            self.command = np.array([vx, vy, omega, 0.82, 1.0], dtype=np.float32)
        elif mode_choice < 0.85:
            # Mode 2: RUN (2.0 to 3.5 m/s sprint)
            vx = float(self.np_random.uniform(2.0, 3.5))
            self.command = np.array([vx, 0.0, 0.0, 0.76, 2.0], dtype=np.float32)
        else:
            # Mode 3: JUMP (Vertical hop to 1.15m)
            self.command = np.array([0.0, 0.0, 0.0, 1.15, 3.0], dtype=np.float32)
        self.command_timer = self.np_random.integers(100, 200)

    def _apply_domain_randomization(self):
        """Pillar 3: Randomizes ground friction and link masses for unshakeable zero-shot robustness."""
        # 1. Floor friction randomization: 0.4 (ice) to 1.3 (rubber)
        new_friction = float(self.np_random.uniform(0.4, 1.3))
        self.model.geom_friction[:, 0] = self.base_geom_friction[:, 0] * new_friction

        # 2. Body mass randomization: +/- 10%
        mass_scale = self.np_random.uniform(0.90, 1.10, size=self.model.nbody)
        self.model.body_mass[:] = self.base_body_mass * mass_scale

    def _get_obs(self):
        # 1. Pelvis pos (3) & quaternion (4)
        pelvis_pos = self.data.qpos[:3].copy()
        pelvis_quat = self.data.qpos[3:7].copy()

        # 2. Pelvis linear & angular velocities
        pelvis_lin_vel = self.data.qvel[:3].copy()
        pelvis_ang_vel = self.data.qvel[3:6].copy()

        # 3. Joint angles (27) & joint velocities (27)
        joint_pos = self.data.qpos[self.qpos_adr].copy()
        joint_vel = self.data.qvel[self.dof_adr].copy() * 0.1

        # 4. Critical limb contact heights (4)
        r_palm_z = float(self.data.site_xpos[self.right_palm_id][2])
        l_palm_z = float(self.data.site_xpos[self.left_palm_id][2])
        r_sole_z = float(self.data.site_xpos[self.right_sole_id][2])
        l_sole_z = float(self.data.site_xpos[self.left_sole_id][2])
        limb_heights = np.array([r_palm_z, l_palm_z, r_sole_z, l_sole_z], dtype=np.float32)

        # 5. Uprightness indicator
        torso_z_world = self.data.xmat[self.torso_upper_id].reshape(3, 3)[:, 2]
        uprightness = np.array([torso_z_world[2]], dtype=np.float32)

        # 6. Active Command Vector (5)
        # If robot is on the floor (z < 0.50m), command automatically defaults to recovery mode 4
        z = float(self.data.qpos[2])
        active_command = self.command.copy()
        if z < 0.50:
            active_command = np.array([0.0, 0.0, 0.0, 0.84, 4.0], dtype=np.float32)

        return np.concatenate([
            pelvis_pos,
            pelvis_quat,
            pelvis_lin_vel,
            pelvis_ang_vel,
            joint_pos,
            joint_vel,
            limb_heights,
            uprightness,
            active_command
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # Apply Domain Randomization
        self._apply_domain_randomization()

        # 100% Physical Standing Knockdown & Floor Recovery Cycle:
        # Cyrus starts erect on two feet (Z = 0.84m), absorbs a random 3D physical shove,
        # tumbles naturally and impacts the floor under pure physics,
        # 100% Physical Standing -> Knockdown -> Recovery Cycle:
        # 1. Cyrus spawns erect on two feet (Z = 0.84m) in neutral balance and holds standing for 3.0s (Mode 0: STAND).
        # 2. At t = 3.0s, a dynamic 3D shove strikes him and switches command to Mode 4: RECOVER.
        # 3. He tumbles naturally, impacts the floor under pure physics, and must self-right, push up,
        #    and lock continuous 2-foot standing balance for 5.0 full seconds (250 steps)!
        self.data.qpos[:3] = [0.0, 0.0, 0.84]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.neutral_standing_ctrl.copy()
        self.data.ctrl[:27] = self.neutral_standing_ctrl.copy()
        if self.model.nu > 27:
            self.data.ctrl[27:] = 0.35
        self.data.qvel[:] = 0.0
        self.prev_action[:] = 0.0
        self.current_ctrl = self.neutral_standing_ctrl.copy()
        self.consecutive_standing_steps = 0
        self.knockdown_done = False
        self.initial_stand_target = 150
        self.fall_type_str = "INITIAL STAND (3.0s)"

        self.command = np.array([0.0, 0.0, 0.0, 0.84, 0.0], dtype=np.float32)  # Mode 0: STAND initially!

        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.prev_z = float(self.data.qpos[2])
        self.max_z_in_ep = self.prev_z

        self.command_timer = self.np_random.integers(100, 200)

        obs = self._get_obs()
        return obs, {
            "fall_type": self.fall_type_str,
            "height": self.prev_z,
            "command_mode": int(self.command[4])
        }

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Dynamic Command Resampling (when standing)
        z = float(self.data.qpos[2])
        if z >= 0.75:
            self.command_timer -= 1
            if self.command_timer <= 0:
                self._sample_new_command()

        ctrl_min = self.model.actuator_ctrlrange[:27, 0]
        ctrl_max = self.model.actuator_ctrlrange[:27, 1]
        target_ctrl = self.neutral_standing_ctrl + action * 0.55 * (ctrl_max - ctrl_min)
        target_ctrl = np.clip(target_ctrl, ctrl_min, ctrl_max)

        self.current_ctrl = 0.65 * self.current_ctrl + 0.35 * target_ctrl
        self.data.ctrl[:27] = self.current_ctrl
        if self.model.nu > 27:
            self.data.ctrl[27:] = 0.35

        # Random Push Disturbance (Pillar 3: Active Push-Recovery)
        if self._step_count % 45 == 0 and z >= 0.78:
            push_angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            push_force = float(self.np_random.uniform(20.0, 50.0) / 61.32)
            self.data.qvel[0] += float(np.cos(push_angle) * push_force)
            self.data.qvel[1] += float(np.sin(push_angle) * push_force)

        # Step 100% authentic physics
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z = float(self.data.qpos[2])
        uprightness = float(obs[-6])
        r_sole_z = float(self.data.site_xpos[self.right_sole_id][2])
        l_sole_z = float(self.data.site_xpos[self.left_sole_id][2])
        r_palm_z = float(self.data.site_xpos[self.right_palm_id][2])
        l_palm_z = float(self.data.site_xpos[self.left_palm_id][2])

        # Center of mass and foot distances
        r_sole_pos = self.data.site_xpos[self.right_sole_id]
        l_sole_pos = self.data.site_xpos[self.left_sole_id]
        pelvis_pos = self.data.qpos[:3]
        mean_foot_dist = 0.5 * (np.hypot(r_sole_pos[0] - pelvis_pos[0], r_sole_pos[1] - pelvis_pos[1]) +
                                np.hypot(l_sole_pos[0] - pelvis_pos[0], l_sole_pos[1] - pelvis_pos[1]))

        # Target command parameters
        cmd_vx, cmd_vy, cmd_omega, cmd_z, mode = self.command

        # =============================================================
        # PILLAR 2: UNIVERSAL EQUILIBRIUM REWARD FUNCTION
        # =============================================================
        reward = 0.0

        # 1. Linear & Angular Velocity Tracking (Pillar 2 - Law 1)
        curr_vx = float(self.data.qvel[0])
        curr_vy = float(self.data.qvel[1])
        curr_omega = float(self.data.qvel[5])
        vel_err = (curr_vx - cmd_vx)**2 + (curr_vy - cmd_vy)**2
        r_vel = np.exp(-2.0 * vel_err)
        r_yaw = np.exp(-2.0 * (curr_omega - cmd_omega)**2)
        reward += 15.0 * r_vel + 5.0 * r_yaw

        # 2. Dynamic Center-of-Mass (CoM) Stability (Pillar 2 - Law 2)
        feet_mid_x = 0.5 * (r_sole_pos[0] + l_sole_pos[0])
        feet_mid_y = 0.5 * (r_sole_pos[1] + l_sole_pos[1])
        com_dist = np.hypot(pelvis_pos[0] - feet_mid_x, pelvis_pos[1] - feet_mid_y)
        r_balance = np.exp(-4.0 * com_dist**2)
        reward += 10.0 * r_balance

        # 3. Torso Verticality & Head Stabilization (Pillar 2 - Law 3)
        r_upright = max(0.0, uprightness)
        reward += 8.0 * r_upright

        # 4. Target Height Tracking (Standing vs Jump)
        height_err = (z - cmd_z)**2
        r_height = np.exp(-10.0 * height_err)
        reward += 12.0 * r_height

        # 5. Weight-Transfer & Ground Push Reflex (when recovering from floor z < 0.60m)
        if z < 0.60:
            torso_z = float(self.data.xpos[self.torso_upper_id][2])
            # Anti-Scorpion Rule: Heavily penalize kicking feet high into the air while lying down
            max_foot_z = max(r_sole_z, l_sole_z)
            if z < 0.45 and max_foot_z > 0.18:
                reward -= 20.0 + 35.0 * (max_foot_z - 0.18)

            # Push-up Chest Elevation Rule: Reward pushing upper torso off the floor using arms!
            if z < 0.45 and torso_z > 0.18:
                reward += 25.0 * min(1.0, (torso_z - 0.18) * 5.0)

            # Ground Palm Press Incentive: Reward planting palms firmly on mat to push torso
            if z < 0.45 and (r_palm_z < 0.12 or l_palm_z < 0.12):
                reward += 10.0

            # Arm push off floor
            delta_z = z - self.prev_z
            if (r_palm_z < 0.15 or l_palm_z < 0.15) and delta_z > 0.001:
                reward += 8.0 * min(1.0, delta_z * 60.0)

            # Knee flexion & active ground foot pull
            r_knee_angle = float(self.data.qpos[self.r_knee_qpos_idx])
            l_knee_angle = float(self.data.qpos[self.l_knee_qpos_idx])
            mean_knee_angle = 0.5 * (r_knee_angle + l_knee_angle)
            if mean_knee_angle > 0.25:
                reward += 8.0 * min(1.4, mean_knee_angle)

            # Weight transfer to foot soles (only if not bridging backwards on back!)
            torso_mat = self.data.xmat[self.torso_upper_id].reshape(3, 3)
            chest_facing_up = float(torso_mat[0, 2])
            if z >= 0.32 and chest_facing_up <= 0.20 and r_sole_z < 0.06 and l_sole_z < 0.06 and mean_foot_dist < 0.35:
                reward += 12.0

            # Hand release incentive
            if z >= 0.35:
                if r_palm_z > 0.25 and l_palm_z > 0.25:
                    reward += 10.0
                elif r_palm_z < 0.12 or l_palm_z < 0.12:
                    reward -= 5.0

        # WHOLE-BODY ANATOMICAL GUARDRAILS (ALL 27 JOINTS & 21 SEGMENTS)
        # Part 1: Head & Cranial Protection
        head_z = float(self.data.xpos[self.head_id][2])
        if z >= 0.30 and head_z < 0.20:
            reward -= 15.0 + 25.0 * (0.20 - head_z)

        # Part 2: Spine & Core Stability
        waist_yaw = float(self.data.qpos[7])
        waist_roll = float(self.data.qpos[8])
        if abs(waist_roll) > 0.25:
            reward -= 6.0 * (abs(waist_roll) - 0.25)
        if abs(waist_yaw) > 0.45:
            reward -= 4.0 * (abs(waist_yaw) - 0.45)

        # Part 3: Arm Symmetry & Bilateral Ground Push
        if z < 0.45:
            arm_asym = abs(r_palm_z - l_palm_z)
            if arm_asym > 0.20:
                reward -= 15.0 + 25.0 * (arm_asym - 0.20)
            if r_palm_z < 0.15 and l_palm_z < 0.15:
                reward += 15.0

        # Part 4: Anti-Splits & Stance Width
        foot_span = float(np.hypot(r_sole_pos[0] - l_sole_pos[0], r_sole_pos[1] - l_sole_pos[1]))
        if foot_span > 0.50:
            reward -= 15.0 * (foot_span - 0.50)
        elif z >= 0.75 and 0.18 <= foot_span <= 0.38:
            reward += 5.0

        # Part 5: Supine Anti-Bridge & Roll-Over Mechanics (Chest facing upward, back on floor)
        torso_mat = self.data.xmat[self.torso_upper_id].reshape(3, 3)
        chest_facing_up = float(torso_mat[0, 2])
        torso_z = float(self.data.xpos[self.torso_upper_id][2])
        if chest_facing_up > 0.15 and uprightness < 0.60:
            # A) Anti-Bridge Rule: Penalize raising pelvis higher than torso while cranium is pinned on mat
            if z > torso_z + 0.04 or (z > 0.28 and head_z < 0.18):
                reward -= 20.0 + 35.0 * max(0.0, z - torso_z)

            # B) Supine-to-Prone Roll Incentive: Reward rotating body onto flank or belly (reducing chest_facing_up)
            roll_progress = max(0.0, 1.0 - max(0.0, chest_facing_up))
            reward += 15.0 * roll_progress

            # C) Sit-Up / Chin Tuck Incentive: Reward curling chin and raising torso above hips
            if torso_z > z and head_z > 0.22:
                reward += 20.0 * min(1.0, (torso_z - z) * 5.0)

        # 6. Action Smoothing / Jerk Penalty (Pillar 2 - Law 4)
        action_diff = action - self.prev_action
        reward -= 0.04 * float(np.sum(np.square(action_diff)))
        reward -= 0.005 * float(np.sum(np.square(action)))
        self.prev_action = action.copy()

        # 7. 5-Second Two-Foot Balance Lock
        is_standing_on_two_feet = bool(
            z >= 0.78 and
            uprightness >= 0.85 and
            r_sole_z < 0.055 and
            l_sole_z < 0.055 and
            r_palm_z > 0.30 and
            l_palm_z > 0.30 and
            mean_foot_dist < 0.35
        )

        # -------------------------------------------------------------
        # PHASE CONTROLLER: Initial 3.0s Stand -> Knockdown Shove -> Recovery 5.0s Hold
        # -------------------------------------------------------------
        if not self.knockdown_done:
            if is_standing_on_two_feet:
                self.consecutive_standing_steps += 1
                reward += 30.0  # Reward maintaining erect balance during initial 3 seconds!
            else:
                self.consecutive_standing_steps = 0

            # At t = 3.0s (150 steps), deliver dynamic physical knockdown shove!
            if self._step_count >= self.initial_stand_target:
                theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
                push_speed = float(self.np_random.uniform(3.0, 4.5))
                self.data.qvel[0] += float(np.cos(theta) * push_speed)
                self.data.qvel[1] += float(np.sin(theta) * push_speed)
                self.data.qvel[3:6] += self.np_random.uniform(-2.5, 2.5, size=3)

                cos_t, sin_t = np.cos(theta), np.sin(theta)
                if abs(cos_t) > abs(sin_t):
                    dir_str = "FORWARD" if cos_t > 0 else "BACKWARD"
                else:
                    dir_str = "RIGHT" if sin_t > 0 else "LEFT"
                self.fall_type_str = f"PUSH {dir_str} (Dynamic Shove)"
                self.knockdown_done = True
                self.command[4] = 4.0  # Switch to Mode 4: RECOVER!
                self.consecutive_standing_steps = 0  # Reset counter for 5.0s recovery balance hold!
        else:
            # Phase 3: Floor Recovery & 5-Second Continuous Standing Balance Lock
            if is_standing_on_two_feet:
                self.consecutive_standing_steps += 1
                reward += 30.0 + 0.2 * min(250, self.consecutive_standing_steps)
                if self.consecutive_standing_steps >= self.target_stand_steps:
                    self.has_succeeded = True
                    reward += 300.0  # Full 5.0-second mastery bonus!
            else:
                if z < 0.72 or uprightness < 0.70 or r_sole_z > 0.08 or l_sole_z > 0.08:
                    self.consecutive_standing_steps = 0

        is_success = bool(self.knockdown_done and self.consecutive_standing_steps >= self.target_stand_steps)

        if z > self.max_z_in_ep:
            self.max_z_in_ep = z
        self.prev_z = z

        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        kinematic_state = self.analyzer.analyze_frame(self.data)

        info = {
            "height": z,
            "max_height": self.max_z_in_ep,
            "uprightness": uprightness,
            "stand_hold_sec": round(self.consecutive_standing_steps * 0.02, 2),
            "initial_stand_sec": 3.0 if self.knockdown_done else round(min(150, self._step_count) * 0.02, 2),
            "command_mode": int(mode),
            "vel_x": curr_vx,
            "cmd_vx": cmd_vx,
            "fall_type": self.fall_type_str,
            "stage": kinematic_state['stage_name'],
            "faults": kinematic_state['faults'],
            "has_faults": kinematic_state['has_faults'],
            "is_success": is_success
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
