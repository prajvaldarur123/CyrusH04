import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CyrusWalkBalanceEnv(gym.Env):
    """
    Cyrus Whole-Body Humanoid Balance & Walking Curriculum Environment.
    Implements the complete A-to-Z humanoid locomotion blueprint:
      Phase 1: Body State & Joint Limit Awareness
      Phase 2: Stable Stance (30s-60s hold without falling)
      Phase 3: Weight Shifting (Lateral pelvis sway & swing foot unloading)
      Phase 4: Single-Leg Balancing (2-5s hold per leg with arm balance)
      Phase 5: Foot Lifting & Clearance (>=0.05m clearance, anti-drag)
      Phase 6: Capture Point Foot Placement (LIPM dynamics)
      Phase 7: Walking Cycle (Alternating stance/swing, 10-step milestone)
      Phase 8: Velocity & Directional Tracking (vx, vy, yaw rate)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Curriculum Phase Constants
    PHASE_STANCE = 1
    PHASE_WEIGHT_SHIFT = 2
    PHASE_SINGLE_LEG = 3
    PHASE_SWING_CLEARANCE = 4
    PHASE_WALKING_CYCLE = 5
    PHASE_COMMAND_TRACK = 6

    PHASE_NAMES = {
        1: "STANCE (30s-60s)",
        2: "WEIGHT SHIFTING",
        3: "SINGLE-LEG BALANCE",
        4: "SWING & CLEARANCE",
        5: "WALKING CYCLE (10-STEP)",
        6: "COMMAND TRACKING"
    }

    def __init__(self, model_path=None, frame_skip=10, render_mode=None, curriculum_phase=1):
        super().__init__()

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "cyrus.xml")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        # Anatomical Body & Site IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_upper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.r_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.l_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")

        self.right_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self.left_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.left_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Key Joints
        self.r_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        self.l_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.r_knee_qpos_idx = self.model.jnt_qposadr[self.r_knee_id]
        self.l_knee_qpos_idx = self.model.jnt_qposadr[self.l_knee_id]

        self.r_ank_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_pitch")
        self.l_ank_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch")
        self.r_ank_qpos_idx = self.model.jnt_qposadr[self.r_ank_p_id]
        self.l_ank_qpos_idx = self.model.jnt_qposadr[self.l_ank_p_id]

        # Key Shoulder and Hip Pitch joints for Reciprocal Arm Counter-Swing (Principle 7)
        self.r_hip_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")
        self.l_hip_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch")
        self.r_sh_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch")
        self.l_sh_p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch")
        self.r_hip_qpos_idx = self.model.jnt_qposadr[self.r_hip_pitch_id] if self.r_hip_pitch_id != -1 else None
        self.l_hip_qpos_idx = self.model.jnt_qposadr[self.l_hip_pitch_id] if self.l_hip_pitch_id != -1 else None
        self.r_sh_qpos_idx = self.model.jnt_qposadr[self.r_sh_p_id] if self.r_sh_p_id != -1 else None
        self.l_sh_qpos_idx = self.model.jnt_qposadr[self.l_sh_p_id] if self.l_sh_p_id != -1 else None

        # Action Space: 27 continuous joint delta targets [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        # Observation Space: 72 proprioceptive base + 5 curriculum & command features = 77
        obs_dim = 77
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Control timing (MuJoCo dt = 0.002, frame_skip = 10 -> control dt = 0.02s / 50Hz)
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self.max_episode_steps = 1500  # 30.0s balance window
        self._step_count = 0

        # Mapped 27 base body joint addresses (excludes unactuated fingers)
        self.jnt_ids = [self.model.actuator_trnid[i, 0] for i in range(27)]
        self.qpos_adr = [self.model.jnt_qposadr[jid] for jid in self.jnt_ids]
        self.dof_adr = [self.model.jnt_dofadr[jid] for jid in self.jnt_ids]

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

        self.act_r_sh_p = r_sh_p
        self.act_l_sh_p = l_sh_p
        self.act_r_el = r_el
        self.act_l_el = l_el
        self.act_waist_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_yaw_act")
        self.act_waist_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch_act")
        self.act_r_hip_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_roll_act")
        self.act_l_hip_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_roll_act")
        self.act_r_hip_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_act")
        self.act_l_hip_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_act")
        self.act_r_knee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        self.act_l_knee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.act_r_ank_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_ankle_pitch_act")
        self.act_l_ank_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_ankle_pitch_act")
        self.act_r_ank_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_ankle_roll_act")
        self.act_l_ank_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_ankle_roll_act")
        from cyrus.controller.balance import CyrusBalanceController
        self.balance = CyrusBalanceController(self.model)

        self.current_ctrl = self.neutral_standing_ctrl.copy()
        self.prev_action = np.zeros(27, dtype=np.float32)

        # Curriculum state
        self.curriculum_phase = curriculum_phase
        self.consecutive_standing_steps = 0
        self.consecutive_walking_steps = 0
        self.consecutive_double_support = 0
        self.stance_target_steps = 1500  # 30s target

        # Gait State Machine for Phases 3-7
        # 0: Double Stance, 1: Left Stance (Right Swing), 2: Right Stance (Left Swing)
        self.gait_state = 0
        self.gait_phase_timer = 0
        self.weight_shift_target_y = 0.0
        self.active_support_leg = "both"  # "left", "right", "both"
        self.step_counter = 0

        # LIPM constant: omega0 = sqrt(g / z0)
        self.omega_0 = float(np.sqrt(9.81 / 0.84))

        # Command Vector: [vx_cmd, vy_cmd, omega_cmd, lateral_com_target, phase_indicator]
        self.command = np.array([0.0, 0.0, 0.0, 0.0, float(self.curriculum_phase)], dtype=np.float32)

    def _get_obs(self):
        # 1. Base Pelvis State
        pelvis_pos = self.data.qpos[:3].copy()
        pelvis_quat = self.data.qpos[3:7].copy()
        pelvis_lin_vel = self.data.qvel[:3].copy()
        pelvis_ang_vel = self.data.qvel[3:6].copy()

        # 2. Joint State (27 angles, 27 velocities)
        joint_pos = self.data.qpos[self.qpos_adr].copy()
        joint_vel = self.data.qvel[self.dof_adr].copy() * 0.1

        # 3. Limb Heights
        r_palm_z = self.data.site_xpos[self.right_palm_id][2]
        l_palm_z = self.data.site_xpos[self.left_palm_id][2]
        r_sole_z = self.data.site_xpos[self.right_sole_id][2]
        l_sole_z = self.data.site_xpos[self.left_sole_id][2]
        limb_heights = np.array([r_palm_z, l_palm_z, r_sole_z, l_sole_z], dtype=np.float32)

        # 4. Uprightness (Torso Z alignment)
        torso_z_world = self.data.xmat[self.torso_upper_id].reshape(3, 3)[:, 2]
        uprightness = np.array([torso_z_world[2]], dtype=np.float32)

        base_72 = np.concatenate([
            pelvis_pos,
            pelvis_quat,
            pelvis_lin_vel,
            pelvis_ang_vel,
            joint_pos,
            joint_vel,
            limb_heights,
            uprightness
        ])

        # 5. Curriculum & Command features (5)
        return np.concatenate([base_72, self.command]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # Spawn upright in neutral standing stance (Z = 0.84m)
        self.data.qpos[:3] = [0.0, 0.0, 0.84]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.neutral_standing_ctrl.copy()
        self.data.ctrl[:27] = self.neutral_standing_ctrl.copy()
        if self.model.nu > 27:
            self.data.ctrl[27:] = 0.35  # relaxed curved fingers
        self.data.qvel[:] = 0.0

        # Small posture randomization for robustness
        self.data.qpos[self.qpos_adr] += self.np_random.uniform(-0.02, 0.02, size=27)
        mujoco.mj_forward(self.model, self.data)

        self.current_ctrl = self.neutral_standing_ctrl.copy()
        self.prev_action = np.zeros(27, dtype=np.float32)
        self.consecutive_standing_steps = 0
        self.consecutive_walking_steps = 0
        self.consecutive_double_support = 0
        self.step_counter = 0
        self.last_swing_foot = None
        self.r_lifted = False
        self.l_lifted = False
        self.stagnation_steps = 0
        self.max_x_in_ep = float(self.data.qpos[0])
        self.max_foot_x_in_ep = float(min(self.data.site_xpos[self.right_sole_id][0], self.data.site_xpos[self.left_sole_id][0]))
        self.max_lead_foot_x_in_ep = float(max(self.data.site_xpos[self.right_sole_id][0], self.data.site_xpos[self.left_sole_id][0]))
        self.gait_state = 0
        self.gait_phase_timer = 0
        self.weight_shift_target_y = 0.0

        # Setup command based on curriculum phase
        self._update_phase_command()

        obs = self._get_obs()
        return obs, {
            "curriculum_phase": self.curriculum_phase,
            "phase_name": self.PHASE_NAMES.get(self.curriculum_phase, "UNKNOWN"),
            "height": float(self.data.qpos[2]),
            "standing_seconds": 0.0,
            "steps_completed": 0
        }

    def _update_phase_command(self):
        if self.curriculum_phase == self.PHASE_STANCE:
            # Mode 1: Pure Balance Stance (vx=0, vy=0, omega=0, sway=0)
            self.command = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.active_support_leg = "both"
        elif self.curriculum_phase == self.PHASE_WEIGHT_SHIFT:
            # Mode 2: Lateral Weight Shifting sway (+/- 0.08m)
            self.command = np.array([0.0, 0.0, 0.0, self.weight_shift_target_y, 2.0], dtype=np.float32)
        elif self.curriculum_phase == self.PHASE_SINGLE_LEG:
            # Mode 3: Single-leg balance target
            self.command = np.array([0.0, 0.0, 0.0, self.weight_shift_target_y, 3.0], dtype=np.float32)
        elif self.curriculum_phase in (self.PHASE_SWING_CLEARANCE, self.PHASE_WALKING_CYCLE):
            # Mode 4/5: Forward walking gait (vx=0.5m/s)
            self.command = np.array([0.5, 0.0, 0.0, self.weight_shift_target_y, float(self.curriculum_phase)], dtype=np.float32)
        else:
            # Mode 6: Dynamic Command Tracking
            vx = float(self.np_random.uniform(0.4, 1.0))
            vy = float(self.np_random.uniform(-0.15, 0.15))
            omega = float(self.np_random.uniform(-0.25, 0.25))
            self.command = np.array([vx, vy, omega, 0.0, 6.0], dtype=np.float32)

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Actuation limits + Biomechanical Reciprocal Gait Reference (27 base body actuators)
        ctrl_min = self.model.actuator_ctrlrange[:27, 0]
        ctrl_max = self.model.actuator_ctrlrange[:27, 1]
        base_ctrl = self.neutral_standing_ctrl.copy()
        if self.curriculum_phase in (self.PHASE_SWING_CLEARANCE, self.PHASE_WALKING_CYCLE, self.PHASE_COMMAND_TRACK):
            phase_in_cycle = (self.gait_phase_timer + 1) % 35
            if phase_in_cycle < 18:
                s = float(phase_in_cycle) / 18.0
                knee_wave = float(np.sin(np.pi * min(1.0, s / 0.75)))
                hip_fwd = float(np.sin(0.5 * np.pi * min(1.0, s / 0.70)))
                base_ctrl[self.act_waist_p] = 0.06
                base_ctrl[self.act_waist_y] = -0.08 * hip_fwd
                base_ctrl[self.act_r_hip_p] = -0.34 * hip_fwd
                base_ctrl[self.act_r_knee] = 0.03 + 0.82 * knee_wave
                base_ctrl[self.act_r_ank_p] = -0.38 * knee_wave - 0.04 * hip_fwd
                base_ctrl[self.act_l_hip_p] = 0.06 * hip_fwd
                base_ctrl[self.act_l_knee] = 0.03 + 0.08 * hip_fwd
                base_ctrl[self.act_l_ank_p] = -0.16 * hip_fwd
                base_ctrl[self.act_l_hip_r] = -0.09 * knee_wave
                base_ctrl[self.act_r_hip_r] = -0.09 * knee_wave
                base_ctrl[self.act_l_ank_r] = 0.09 * knee_wave
                # Natural contralateral arm counter-swing & active elbow flexion
                base_ctrl[self.act_l_sh_p] = 0.05 + 0.32 * hip_fwd
                base_ctrl[self.act_r_sh_p] = 0.05 - 0.32 * hip_fwd
                base_ctrl[self.act_l_el] = -0.15 - 0.40 * knee_wave
                base_ctrl[self.act_r_el] = -0.15 + 0.10 * knee_wave
            else:
                s = float(phase_in_cycle - 18) / 17.0
                knee_wave = float(np.sin(np.pi * min(1.0, s / 0.75)))
                hip_fwd = float(np.sin(0.5 * np.pi * min(1.0, s / 0.70)))
                base_ctrl[self.act_waist_p] = 0.06
                base_ctrl[self.act_waist_y] = 0.08 * hip_fwd
                base_ctrl[self.act_l_hip_p] = -0.34 * hip_fwd
                base_ctrl[self.act_l_knee] = 0.03 + 0.82 * knee_wave
                base_ctrl[self.act_l_ank_p] = -0.38 * knee_wave - 0.04 * hip_fwd
                base_ctrl[self.act_r_hip_p] = 0.06 * hip_fwd
                base_ctrl[self.act_r_knee] = 0.03 + 0.08 * hip_fwd
                base_ctrl[self.act_r_ank_p] = -0.16 * hip_fwd
                base_ctrl[self.act_r_hip_r] = 0.09 * knee_wave
                base_ctrl[self.act_l_hip_r] = 0.09 * knee_wave
                base_ctrl[self.act_r_ank_r] = -0.09 * knee_wave
                # Natural contralateral arm counter-swing & active elbow flexion
                base_ctrl[self.act_r_sh_p] = 0.05 + 0.32 * hip_fwd
                base_ctrl[self.act_l_sh_p] = 0.05 - 0.32 * hip_fwd
                base_ctrl[self.act_r_el] = -0.15 - 0.40 * knee_wave
                base_ctrl[self.act_l_el] = -0.15 + 0.10 * knee_wave
            target_ctrl = base_ctrl + action * 0.08 * (ctrl_max - ctrl_min)
        else:
            target_ctrl = base_ctrl + action * 0.35 * (ctrl_max - ctrl_min)
        target_ctrl = np.clip(target_ctrl, ctrl_min, ctrl_max)

        # First-order motor filter (responsive enough for 0.70s gait cycles)
        self.current_ctrl = 0.60 * self.current_ctrl + 0.40 * target_ctrl
        self.data.ctrl[:27] = self.current_ctrl
        if self.model.nu > 27:
            self.data.ctrl[27:] = 0.35

        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        # -------------------------------------------------------------
        # PHASE 1: BODY AWARENESS & KINEMATIC STATE ESTIMATION
        # -------------------------------------------------------------
        pelvis_pos = self.data.qpos[:3].copy()
        z = float(pelvis_pos[2])
        torso_mat = self.data.xmat[self.torso_upper_id].reshape(3, 3)
        uprightness = float(torso_mat[2, 2])
        
        # Exact Pitch and Roll angles (radians) for BOTH upper torso and pelvis
        pitch = float(np.arctan2(-torso_mat[2, 0], np.sqrt(torso_mat[2, 1]**2 + torso_mat[2, 2]**2)))
        roll = float(np.arctan2(torso_mat[2, 1], torso_mat[2, 2]))
        pitch_deg = abs(np.degrees(pitch))
        roll_deg = abs(np.degrees(roll))

        pelvis_mat = self.data.xmat[self.pelvis_id].reshape(3, 3)
        pelvis_pitch = float(np.arctan2(pelvis_mat[0, 2], pelvis_mat[2, 2]))
        pelvis_roll = float(np.arctan2(pelvis_mat[1, 2], pelvis_mat[2, 2]))
        pelvis_pitch_deg = abs(np.degrees(pelvis_pitch))
        pelvis_roll_deg = abs(np.degrees(pelvis_roll))

        r_sole_pos = self.data.site_xpos[self.right_sole_id].copy()
        l_sole_pos = self.data.site_xpos[self.left_sole_id].copy()
        r_sole_z = float(r_sole_pos[2])
        l_sole_z = float(l_sole_pos[2])

        r_grounded = bool(r_sole_z < 0.035)
        l_grounded = bool(l_sole_z < 0.035)

        # Center of Mass (CoM) & Velocities
        com = self.data.subtree_com[self.pelvis_id].copy()
        com_vel = self.data.qvel[:3].copy()
        actual_vx = float(com_vel[0])
        actual_vy = float(com_vel[1])
        pelvis_ang_vel = self.data.qvel[3:6].copy()
        actual_wz = float(pelvis_ang_vel[2])

        # Capture Point (LIPM) & CoM offset relative to leading foot and foot midpoint
        cp_x = float(com[0] + com_vel[0] / self.omega_0)
        cp_y = float(com[1] + com_vel[1] / self.omega_0)
        lead_foot_x = max(float(r_sole_pos[0]), float(l_sole_pos[0]))
        mean_foot_x = 0.5 * (float(r_sole_pos[0]) + float(l_sole_pos[0]))
        com_ahead = float(com[0]) - lead_foot_x
        pelvis_vs_feet_x = float(com[0]) - mean_foot_x
        pelvis_lead_err = pelvis_vs_feet_x - 0.03
        r_hip_val = float(self.data.qpos[self.r_hip_qpos_idx]) if self.r_hip_qpos_idx is not None else 0.0
        l_hip_val = float(self.data.qpos[self.l_hip_qpos_idx]) if self.l_hip_qpos_idx is not None else 0.0
        parallel_leg_lean = bool(r_hip_val > 0.08 and l_hip_val > 0.08)
        body_balanced = bool(
            z >= 0.80 and pitch_deg <= 8.0 and roll_deg <= 8.0
            and pelvis_pitch_deg <= 8.0 and pelvis_roll_deg <= 8.0
            and com_ahead <= 0.10 and abs(pelvis_vs_feet_x) <= 0.14
            and not parallel_leg_lean and (r_grounded or l_grounded)
        )

        # -------------------------------------------------------------
        # ANTI-SPLITS CONSTRAINT: NATURAL HUMAN FOOT WIDTH (0.20m - 0.32m)
        # -------------------------------------------------------------
        reward = 0.0
        foot_width_y = abs(float(l_sole_pos[1] - r_sole_pos[1]))
        if foot_width_y > 0.35:
            # Heavy penalty for "splits" / wide-stance sumo cheating
            reward -= 50.0 * float((foot_width_y - 0.35) ** 2)
        elif foot_width_y < 0.15:
            # Penalty for feet colliding
            reward -= 20.0 * float((0.15 - foot_width_y) ** 2)
        else:
            reward += 6.0 * float(np.exp(-((foot_width_y - 0.24) ** 2) / 0.005))

        # -------------------------------------------------------------
        # VAN MARUM ET AL. EXPONENTIAL REGULARIZED REWARD FOUNDATION
        # -------------------------------------------------------------
        # 1. Height maintenance (Nominal 0.84m)
        r_height = float(np.exp(-((z - 0.84) ** 2) / 0.015))

        # 2. Torso & Pelvis uprightness (stops waist counter-arching cheat)
        r_upright = float(
            np.exp(-(pitch ** 2) / 0.03) * np.exp(-(roll ** 2) / 0.03)
            * np.exp(-(pelvis_pitch ** 2) / 0.03) * np.exp(-(pelvis_roll ** 2) / 0.03)
        )

        # GATING STANDING REWARD: In Walking Phases, DO NOT reward standing still or rocking backward!
        if self.curriculum_phase in (self.PHASE_SWING_CLEARANCE, self.PHASE_WALKING_CYCLE, self.PHASE_COMMAND_TRACK):
            cur_trail_foot_x = float(min(r_sole_pos[0], l_sole_pos[0]))
            made_fwd_progress = bool(
                (float(pelvis_pos[0]) > self.max_x_in_ep + 0.001)
                or (cur_trail_foot_x > self.max_foot_x_in_ep + 0.001)
                or (lead_foot_x > self.max_lead_foot_x_in_ep + 0.001)
            )
            if float(pelvis_pos[0]) > self.max_x_in_ep:
                if body_balanced:
                    reward += 300.0 * (float(pelvis_pos[0]) - self.max_x_in_ep)
                self.max_x_in_ep = float(pelvis_pos[0])
            if cur_trail_foot_x > self.max_foot_x_in_ep:
                if body_balanced:
                    reward += 300.0 * (cur_trail_foot_x - self.max_foot_x_in_ep)
                self.max_foot_x_in_ep = cur_trail_foot_x
            if lead_foot_x > self.max_lead_foot_x_in_ep:
                if body_balanced:
                    reward += 150.0 * (lead_foot_x - self.max_lead_foot_x_in_ep)
                self.max_lead_foot_x_in_ep = lead_foot_x

            # Backward movement penalties
            if actual_vx < 0.0:
                reward -= 40.0 * abs(actual_vx)
            if float(pelvis_pos[0]) < 0.0:
                reward -= 50.0 * abs(float(pelvis_pos[0]))
            if float(pelvis_pos[0]) < self.max_x_in_ep - 0.02:
                reward -= 30.0 * (self.max_x_in_ep - float(pelvis_pos[0]))

            # Anti-freezing double support cadence penalty (forces cyclic stepping)
            if r_grounded and l_grounded:
                self.consecutive_double_support += 1
                if self.consecutive_double_support > 25:
                    reward -= 0.60 * float(self.consecutive_double_support - 25)
            else:
                self.consecutive_double_support = 0

            # Walking Curriculum Gate: In Phase 5, standing still without advancing forward attenuates standing rewards!
            if actual_vx < 0.05 and cur_trail_foot_x < 0.10:
                self.stagnation_steps += 1
                r_height *= 0.20
                r_upright *= 0.20
                reward -= 8.0
            else:
                self.stagnation_steps = 0
                if body_balanced:
                    target_speed = float(self.command[0])
                    # 1. Continuous Forward Velocity Drive
                    reward += 50.0 * min(max(0.0, actual_vx), target_speed + 0.3)
                    # 2. Continuous Monotonic Forward Displacement Gradient toward +1.50m
                    reward += 70.0 * min(1.50, max(0.0, float(pelvis_pos[0])))
                    reward += 55.0 * min(1.50, max(0.0, cur_trail_foot_x))
                    reward += 35.0 * min(1.50, max(0.0, lead_foot_x))
                    # 3. Coupled Pelvis Follow-Through
                    foot_advance_scale = min(1.5, max(0.0, lead_foot_x / 0.20))
                    reward += 28.0 * foot_advance_scale * float(np.exp(-25.0 * (pelvis_vs_feet_x ** 2)))
                else:
                    reward -= 15.0 * max(0.0, max(pitch_deg, pelvis_pitch_deg) - 8.0)
                    if com_ahead > 0.10:
                        reward -= 45.0 * (com_ahead - 0.10)
                    if abs(pelvis_vs_feet_x) > 0.13:
                        reward -= 50.0 * (abs(pelvis_vs_feet_x) - 0.13)
                    if parallel_leg_lean:
                        reward -= 50.0 * (r_hip_val + l_hip_val)
                    if not r_grounded and not l_grounded:
                        reward -= 25.0

        reward += 12.0 * r_height
        reward += 15.0 * r_upright

        # 3. Head stability
        head_z = float(self.data.xpos[self.head_id][2])
        if head_z > 1.25:
            reward += 3.0

        # 4. Stand timer tracking: ONLY in pure stance phase!
        is_upright = (z >= 0.78 and uprightness >= 0.88 and pitch_deg < 15.0 and roll_deg < 15.0)
        if is_upright:
            self.consecutive_standing_steps += 1
            if self.curriculum_phase == self.PHASE_STANCE:
                reward += 3.0
        else:
            self.consecutive_standing_steps = 0

        # -------------------------------------------------------------
        # PHASE 3: WEIGHT SHIFTING CURRICULUM
        # -------------------------------------------------------------
        if self.curriculum_phase == self.PHASE_WEIGHT_SHIFT:
            self.gait_phase_timer += 1
            if self.gait_phase_timer >= 100:
                self.gait_phase_timer = 0
                self.weight_shift_target_y = 0.08 if self.weight_shift_target_y <= 0.0 else -0.08
                self.active_support_leg = "left" if self.weight_shift_target_y > 0 else "right"
                self.command[3] = self.weight_shift_target_y

            target_y_pos = (l_sole_pos[1] if self.weight_shift_target_y > 0 else r_sole_pos[1])
            lateral_err = abs(com[1] - target_y_pos)
            reward += 15.0 * np.exp(-25.0 * (lateral_err ** 2))

            if self.active_support_leg == "left":
                if r_sole_z > 0.02 or not r_grounded:
                    reward += 10.0
            else:
                if l_sole_z > 0.02 or not l_grounded:
                    reward += 10.0

        # -------------------------------------------------------------
        # PHASE 4: SINGLE-LEG BALANCING CURRICULUM
        # -------------------------------------------------------------
        elif self.curriculum_phase == self.PHASE_SINGLE_LEG:
            self.gait_phase_timer += 1
            if self.gait_phase_timer >= 150:
                self.gait_phase_timer = 0
                self.active_support_leg = "right" if self.active_support_leg == "left" else "left"

            if self.active_support_leg == "left":
                if l_grounded and r_sole_z >= 0.035 and is_upright:
                    reward += 20.0
                elif not l_grounded:
                    reward -= 10.0
            else:
                if r_grounded and l_sole_z >= 0.035 and is_upright:
                    reward += 20.0
                elif not r_grounded:
                    reward -= 10.0

        # -------------------------------------------------------------
        # PHASE 5, 6 & 7: SWING, FOOT CLEARANCE, CAPTURE POINT & WALKING
        # -------------------------------------------------------------
        elif self.curriculum_phase in (self.PHASE_SWING_CLEARANCE, self.PHASE_WALKING_CYCLE, self.PHASE_COMMAND_TRACK):
            self.gait_phase_timer += 1
            
            gait_period_steps = 35
            phase_in_cycle = self.gait_phase_timer % gait_period_steps

            target_vx = self.command[0]

            # 1. Exponential Forward Velocity Tracking (ONLY when body_balanced is True)
            if body_balanced:
                r_vel_x = float(np.exp(-((actual_vx - target_vx) ** 2) / 0.04))
                reward += 18.0 * r_vel_x

            # Swing-foot forward lead incentive: reward advancing swing foot ahead of stance foot
            if phase_in_cycle < 18 and l_grounded and z >= 0.80 and pelvis_pitch_deg <= 10.0:
                swing_lead = float(r_sole_pos[0] - l_sole_pos[0])
                if swing_lead > 0.0:
                    reward += 35.0 * min(0.22, swing_lead)
            elif phase_in_cycle >= 18 and r_grounded and z >= 0.80 and pelvis_pitch_deg <= 10.0:
                swing_lead = float(l_sole_pos[0] - r_sole_pos[0])
                if swing_lead > 0.0:
                    reward += 35.0 * min(0.22, swing_lead)

            # 2. Lateral & Yaw Drift Suppression
            reward -= 5.0 * (actual_vy ** 2)
            reward -= 3.0 * (actual_wz ** 2)

            # Principle 7: Whole-Body Contralateral Arm Counter-Swing & Step Symmetry Coordination
            if self.r_hip_qpos_idx is not None and self.r_sh_qpos_idx is not None:
                r_hip_val = float(self.data.qpos[self.r_hip_qpos_idx])
                l_hip_val = float(self.data.qpos[self.l_hip_qpos_idx])
                r_sh_val = float(self.data.qpos[self.r_sh_qpos_idx])
                l_sh_val = float(self.data.qpos[self.l_sh_qpos_idx])
                # True contralateral swing: Left arm swings forward with Right leg, Right arm with Left leg
                target_r_sh = 0.05 + 0.45 * (r_hip_val - l_hip_val)
                target_l_sh = 0.05 + 0.45 * (l_hip_val - r_hip_val)
                arm_err = (r_sh_val - target_r_sh) ** 2 + (l_sh_val - target_l_sh) ** 2
                reward += 10.0 * float(np.exp(-arm_err / 0.06))

            # 3. Bilateral Symmetrical Stance / Swing Alternation & Principle 5 Trajectory Shaping
            r_knee_val = float(self.data.qpos[self.r_knee_qpos_idx])
            l_knee_val = float(self.data.qpos[self.l_knee_qpos_idx])
            if parallel_leg_lean:
                reward -= 40.0 * (r_hip_val + l_hip_val)
            if phase_in_cycle < 18:
                self.gait_state = 1
                if l_grounded:
                    reward += 6.0
                else:
                    reward -= 4.0
                
                # Principle 5: Smooth Cycloidal Clearance (Target apex: 55mm) + Reciprocal Hip/Knee Swing
                s = float(phase_in_cycle) / 18.0
                z_target = 0.055 * 4.0 * s * (1.0 - s) + 0.010
                clearance_err = abs(r_sole_z - z_target)
                reward += 10.0 * float(np.exp(-(clearance_err ** 2) / 0.002))

                knee_wave = float(np.sin(np.pi * min(1.0, s / 0.75)))
                hip_fwd = float(np.sin(0.5 * np.pi * min(1.0, s / 0.70)))
                r_hip_target = -0.24 * hip_fwd
                r_knee_target = 0.03 + 0.52 * knee_wave
                l_hip_target = 0.18 * hip_fwd
                gait_err = ((r_hip_val - r_hip_target) ** 2 + (r_knee_val - r_knee_target) ** 2 + (l_hip_val - l_hip_target) ** 2)
                reward += 20.0 * float(np.exp(-gait_err / 0.08))
                if s >= 0.5 and r_sole_pos[0] > l_sole_pos[0]:
                    reward += 15.0 * min(0.18, float(r_sole_pos[0] - l_sole_pos[0]))
                
                if r_sole_z < 0.015:
                    reward -= 6.0  # Foot dragging penalty
                elif r_sole_z > 0.095:
                    reward -= 15.0 * float((r_sole_z - 0.095) ** 2)  # Excessive foot lifting penalty

                # Principle 4: Capture Point Landing Alignment
                if phase_in_cycle >= 13:
                    dist_to_cp = float(np.hypot(r_sole_pos[0] - cp_x, r_sole_pos[1] - cp_y))
                    reward += 8.0 * float(np.exp(-12.0 * dist_to_cp))
            else:
                self.gait_state = 2
                if r_grounded:
                    reward += 6.0
                else:
                    reward -= 4.0

                # Principle 5: Smooth Cycloidal Clearance (Target apex: 55mm) + Reciprocal Hip/Knee Swing
                s = float(phase_in_cycle - 18) / 17.0
                z_target = 0.055 * 4.0 * s * (1.0 - s) + 0.010
                clearance_err = abs(l_sole_z - z_target)
                reward += 10.0 * float(np.exp(-(clearance_err ** 2) / 0.002))

                knee_wave = float(np.sin(np.pi * min(1.0, s / 0.75)))
                hip_fwd = float(np.sin(0.5 * np.pi * min(1.0, s / 0.70)))
                l_hip_target = -0.24 * hip_fwd
                l_knee_target = 0.03 + 0.52 * knee_wave
                r_hip_target = 0.18 * hip_fwd
                gait_err = ((l_hip_val - l_hip_target) ** 2 + (l_knee_val - l_knee_target) ** 2 + (r_hip_val - r_hip_target) ** 2)
                reward += 20.0 * float(np.exp(-gait_err / 0.08))
                if s >= 0.5 and l_sole_pos[0] > r_sole_pos[0]:
                    reward += 15.0 * min(0.18, float(l_sole_pos[0] - r_sole_pos[0]))

                if l_sole_z < 0.015:
                    reward -= 6.0  # Foot dragging penalty
                elif l_sole_z > 0.095:
                    reward -= 15.0 * float((l_sole_z - 0.095) ** 2)  # Excessive foot lifting penalty

                # Principle 4: Capture Point Landing Alignment
                if phase_in_cycle >= 30:
                    dist_to_cp = float(np.hypot(l_sole_pos[0] - cp_x, l_sole_pos[1] - cp_y))
                    reward += 8.0 * float(np.exp(-12.0 * dist_to_cp))

            # 4. REAL PHYSICAL STEP DETECTION (NOT clock-based) + HUMANOID-GYM AIR-TIME & STEP SYMMETRY
            if r_sole_z > 0.032:
                self.r_lifted = True
                self.r_air_steps = getattr(self, "r_air_steps", 0) + 1
            if l_sole_z > 0.032:
                self.l_lifted = True
                self.l_air_steps = getattr(self, "l_air_steps", 0) + 1

            # Detect genuine foot strikes in front of support foot (MUST BE UPRIGHT, NOT FALLING)
            if self.r_lifted and r_grounded and l_grounded:
                air_t = min(0.35, getattr(self, "r_air_steps", 0) * 0.02)
                if (r_sole_pos[0] > l_sole_pos[0] + 0.065 and z >= 0.78
                        and pitch_deg <= 10.0 and pelvis_pitch_deg <= 10.0
                        and com_ahead <= 0.15 and self.last_swing_foot != "right"):
                    self.step_counter += 1
                    self.last_swing_foot = "right"
                    self.r_lifted = False
                    self.r_air_steps = 0
                    reward += 100.0 + 120.0 * air_t  # Humanoid-Gym air-time + true forward step bonus!
                elif r_grounded and r_sole_z < 0.025:
                    self.r_lifted = False
                    self.r_air_steps = 0

            if self.l_lifted and l_grounded and r_grounded:
                air_t = min(0.35, getattr(self, "l_air_steps", 0) * 0.02)
                if (l_sole_pos[0] > r_sole_pos[0] + 0.065 and z >= 0.78
                        and pitch_deg <= 10.0 and pelvis_pitch_deg <= 10.0
                        and com_ahead <= 0.15 and self.last_swing_foot != "left"):
                    self.step_counter += 1
                    self.last_swing_foot = "left"
                    self.l_lifted = False
                    self.l_air_steps = 0
                    reward += 100.0 + 120.0 * air_t  # Humanoid-Gym air-time + true forward step bonus!
                elif l_grounded and l_sole_z < 0.025:
                    self.l_lifted = False
                    self.l_air_steps = 0

            if self.step_counter >= 10:
                reward += 200.0

        # -------------------------------------------------------------
        # REGULARIZATION & ACTUATOR EFFORT (van Marum et al.)
        # -------------------------------------------------------------
        # Action rate-of-change penalty (suppress joint jerk)
        action_diff = action - self.prev_action
        reward -= 0.10 * float(np.sum(np.square(action_diff)))
        reward -= 0.005 * float(np.sum(np.square(action)))
        self.prev_action = action.copy()

        # Mechanical Effort penalty: sum(tau^2)
        torques = self.data.actuator_force[:27]
        reward -= 0.0003 * float(np.sum(np.square(torques)))

        # Joint limit soft barrier penalty (prevent ankle/knee limit saturation)
        limit_violations = 0.0
        for j in range(27):
            q_val = self.data.qpos[7 + j]
            q_min = self.model.jnt_range[j + 1, 0] + 0.04
            q_max = self.model.jnt_range[j + 1, 1] - 0.04
            if q_val < q_min:
                limit_violations += (q_min - q_val) ** 2
            elif q_val > q_max:
                limit_violations += (q_val - q_max) ** 2
        reward -= 5.0 * float(limit_violations)

        # Principle 10: Knee compliance (prevent rigid hyperextension lock)
        r_knee_val = float(self.data.qpos[self.r_knee_qpos_idx])
        l_knee_val = float(self.data.qpos[self.l_knee_qpos_idx])
        if r_knee_val < 0.01:
            reward -= 5.0 * float((0.01 - r_knee_val) ** 2)
        if l_knee_val < 0.01:
            reward -= 5.0 * float((0.01 - l_knee_val) ** 2)

        # -------------------------------------------------------------
        # STRICT TERMINATION & ANTI-STAGNATION
        # -------------------------------------------------------------
        terminated = bool(
            z < 0.65 or pitch_deg > 18.0 or roll_deg > 18.0
            or pelvis_pitch_deg > 16.0 or pelvis_roll_deg > 16.0
            or com_ahead > 0.28 or uprightness < 0.85
        )
        
        # Kill shivering stagnation: if robot refuses to walk forward in Phase 5
        if self.curriculum_phase in (self.PHASE_SWING_CLEARANCE, self.PHASE_WALKING_CYCLE, self.PHASE_COMMAND_TRACK):
            if self.stagnation_steps >= 175:  # 3.5 seconds of absolute freezing in place
                terminated = True
                reward -= 50.0

        if terminated:
            reward -= 250.0  # Heavy crash penalty completely destroys dive-bomb exploitation!

        truncated = bool(self._step_count >= self.max_episode_steps)

        obs = self._get_obs()
        info = {
            "height": z,
            "uprightness": uprightness,
            "standing_seconds": round(self.consecutive_standing_steps * self.control_dt, 2),
            "steps_completed": self.step_counter,
            "curriculum_phase": self.curriculum_phase,
            "phase_name": self.PHASE_NAMES.get(self.curriculum_phase, "UNKNOWN"),
            "forward_vel": float(com_vel[0]),
            "episode_duration_s": round(self._step_count * self.control_dt, 2),
            "max_x_in_ep": float(self.max_x_in_ep),
            "max_foot_x_in_ep": float(self.max_foot_x_in_ep),
            "r_sole_z": r_sole_z,
            "l_sole_z": l_sole_z,
            "capture_point_x": cp_x,
            "capture_point_y": cp_y
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def set_curriculum_phase(self, phase):
        self.curriculum_phase = phase
        self._update_phase_command()

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
