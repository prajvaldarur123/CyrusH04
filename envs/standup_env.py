import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class CyrusStandupEnv(gym.Env):
    """
    Gymnasium reinforcement learning environment for Cyrus Humanoid Autonomous Standup.
    Goal: Learn to rise from arbitrary fallen states (prone, supine, side) into upright standing.
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

        # Cache IDs
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_upper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.right_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self.left_palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.left_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Knee joint indices for foot retraction & tuck tracking
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

        # 27 Actuators: continuous delta targets between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32)

        # 150-dimensional Whole-Body Proprioceptive Schema
        obs_dim = 150
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._step_count = 0
        self.max_episode_steps = 750  # 15.0s episodes: 3.0s initial stand + tumble/4-stage recovery + 5.0s standing hold
        self.consecutive_standing_steps = 0
        self.target_stand_steps = 250  # 250 steps at 50Hz = 5.0 seconds
        self.initial_stand_target = 150  # 150 steps at 50Hz = 3.0 seconds initial standing
        self.knockdown_done = False

        # Universal Neutral Reference Pose (erect standing posture)
        self.neutral_standing_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        r_sh_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_act")
        l_sh_p = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_act")
        r_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_act")
        l_el = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_act")
        if r_sh_p != -1: self.neutral_standing_ctrl[r_sh_p] = 0.05
        if l_sh_p != -1: self.neutral_standing_ctrl[l_sh_p] = 0.05
        if r_el != -1: self.neutral_standing_ctrl[r_el] = -0.15
        if l_el != -1: self.neutral_standing_ctrl[l_el] = -0.15

        # Base 27 body joint indices for 72-dim observation compatibility
        base_joint_names = [
            'waist_yaw', 'waist_roll', 'waist_pitch',
            'neck_yaw', 'neck_pitch',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]
        self.base_qpos_adr = []
        self.base_qvel_adr = []
        for name in base_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                self.base_qpos_adr.append(self.model.jnt_qposadr[jid])
                self.base_qvel_adr.append(self.model.jnt_dofadr[jid])

        from cyrus.controller.standup_reflex import CyrusStandupReflex
        from cyrus.controller.kinematic_observer import CyrusKinematicObserver
        from cyrus.controller.body_schema_observer import CyrusBodySchemaObserver
        self.reflex = CyrusStandupReflex(self.model)
        self.analyzer = CyrusKinematicObserver(self.model)
        self.body_observer = CyrusBodySchemaObserver(self.model)

        self.has_succeeded = False
        self.standing_steps = 0
        self.fall_type_str = "STANDUP"
        self.is_curriculum = False
        self.prev_z = 0.84
        self.max_z_in_ep = 0.84
        self.anchor_x = 0.0
        self.anchor_y = 0.0
        self.prev_action = np.zeros(27, dtype=np.float32)

    def _get_obs(self):
        return self.body_observer.extract_schema(
            self.data,
            anchor_x=getattr(self, 'anchor_x', 0.0),
            anchor_y=getattr(self, 'anchor_y', 0.0)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)
        self.body_observer.reset()

        # -------------------------------------------------------------
        # SYSTEMATIC 3-PHASE STANDUP CURRICULUM:
        # Phase 1: Canonical Prone Get-up (Belly on mat, hands posted cleanly in pushup position -> Bear-Pike -> Squat -> Stand)
        # Phase 2: Canonical Supine Roll-and-Rise (Back on mat -> Active Flank Roll to Prone -> Pushup -> Stand)
        # Phase 3: Omnidirectional Shoves & Tumbling Rejection (Full 3D shoves once get-up is 100% mastered)
        # -------------------------------------------------------------
        self.curriculum_phase = int(os.environ.get("STANDUP_PHASE", 1))

        if self.curriculum_phase == 1:
            # PHASE 1: STANDING -> FORWARD FALL ON FACE (PRONE) -> REFLEX RECOVERY
            self.knockdown_done = False
            self.initial_stand_target = 75  # 1.5s initial erect standing balance hold
            self.latched_prone = True
            self.fall_type_str = "INITIAL STAND (1.5s)"

            # Start standing in rock-solid dynamic equilibrium:
            self.data.qpos[:3] = [0.0, 0.0, 0.84]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[7:7+len(self.neutral_standing_ctrl)] = self.neutral_standing_ctrl.copy()
            self.data.ctrl[:] = self.neutral_standing_ctrl.copy()
            self.current_ctrl = self.neutral_standing_ctrl.copy()
            self.data.qvel[:] = 0.0
            self.prev_action[:] = 0.0
            self.consecutive_standing_steps = 0
            self.initial_stand_steps = 0
            self.anchor_x = float(self.data.qpos[0])
            self.anchor_y = float(self.data.qpos[1])
            self.reflex.reset(anchor_x=self.anchor_x, anchor_y=self.anchor_y)
            self.reflex.recovery_time = 0.0

        elif self.curriculum_phase == 2:
            # PHASE 2: CANONICAL SUPINE ROLL-AND-RISE
            self.knockdown_done = True
            self.initial_stand_target = 0
            self.latched_prone = False
            self.fall_type_str = "CANONICAL SUPINE ROLL-AND-RISE"

            # Torso flat on back, pitched backward 90 deg:
            self.data.qpos[:3] = [0.0, 0.0, 0.12]
            self.data.qpos[3:7] = [0.7071068, 0.0, -0.7071068, 0.0]
            self.data.qpos[7:self.model.nq] = 0.0
            self.data.ctrl[:] = 0.0
            self.current_ctrl = np.zeros(27, dtype=np.float32)
            self.data.qvel[:] = 0.0
            self.prev_action[:] = 0.0
            self.consecutive_standing_steps = 0
            self.initial_stand_steps = 0
            self.anchor_x = 0.0
            self.anchor_y = 0.0
            self.reflex.reset(anchor_x=0.0, anchor_y=0.0)
            self.reflex.recovery_time = 0.0

        else:
            # PHASE 3: OMNIDIRECTIONAL SHOVES & TUMBLING REJECTION
            self.is_curriculum = False
            self.data.qpos[:3] = [0.0, 0.0, 0.84]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[7:7+len(self.neutral_standing_ctrl)] = self.neutral_standing_ctrl.copy()
            self.data.ctrl[:] = self.neutral_standing_ctrl.copy()
            self.data.qvel[:] = 0.0
            self.prev_action[:] = 0.0
            self.current_ctrl = self.neutral_standing_ctrl.copy()
            self.consecutive_standing_steps = 0
            self.initial_stand_steps = 0
            self.knockdown_done = False
            self.fall_type_str = "INITIAL STAND (3.0s)"
            self.anchor_x = float(self.data.qpos[0])
            self.anchor_y = float(self.data.qpos[1])
            self.reflex.reset(anchor_x=self.anchor_x, anchor_y=self.anchor_y)

        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.prev_z = float(self.data.qpos[2])
        self.max_z_in_ep = self.prev_z
        self.has_succeeded = False
        self.standing_steps = 0
        obs = self._get_obs()
        return obs, {
            "is_success": False,
            "height": self.prev_z,
            "max_height": self.max_z_in_ep,
            "fall_type": self.fall_type_str,
            "is_curriculum": self.is_curriculum
        }

    def step(self, action):
        self._step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.body_observer.update_action(action)
        if len(action) < self.model.nu:
            act_full = np.zeros(self.model.nu, dtype=np.float32)
            act_full[:len(action)] = action
            action_full = act_full
        else:
            action_full = action

        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]

        # Universal Neutral + Biomechanical Reflex Residual Motor Learning:
        cur_z = float(self.data.qpos[2])
        cur_up = float(self.data.xmat[self.torso_upper_id].reshape(3, 3)[2, 2])
        cur_r_sole = float(self.data.site_xpos[self.right_sole_id][2])
        cur_l_sole = float(self.data.site_xpos[self.left_sole_id][2])
        guidance_ctrl = self.neutral_standing_ctrl

        if not self.knockdown_done:
            self.reflex.balance.nominal_pose["right_knee_act"] = 0.0
            self.reflex.balance.nominal_pose["left_knee_act"] = 0.0
            self.reflex.balance.nominal_pose["right_hip_pitch_act"] = 0.0
            self.reflex.balance.nominal_pose["left_hip_pitch_act"] = 0.0
            self.reflex.balance.nominal_pose["right_ankle_pitch_act"] = 0.0
            self.reflex.balance.nominal_pose["left_ankle_pitch_act"] = 0.0
            self.reflex.balance.nominal_pose["waist_pitch_act"] = 0.0
            base_ctrl = self.reflex.balance.compute_control(self.data)
            target_ctrl = base_ctrl + action_full * 0.008 * (ctrl_max - ctrl_min)
        else:
            steps_since_shove = max(0, self._step_count - self.initial_stand_target)
            if steps_since_shove == 25:
                self.anchor_x = float(self.data.qpos[0])
                self.anchor_y = float(self.data.qpos[1])
            torso_mat_cur = self.data.xmat[self.torso_upper_id].reshape(3, 3)
            if self.curriculum_phase == 1:
                is_prone = True
                # Progress reflex stages once Cyrus touches down onto mat or 18 steps post-shove
                if cur_z < 0.40 or steps_since_shove > 18:
                    self.reflex.recovery_time += self.frame_skip * self.model.opt.timestep
            elif self.curriculum_phase == 2:
                if steps_since_shove <= 15:
                    is_prone = False
                else:
                    is_prone = bool(torso_mat_cur[2, 0] < 0.0)
                self.reflex.recovery_time += self.frame_skip * self.model.opt.timestep
            else:
                if steps_since_shove <= 25:
                    self.latched_prone = bool(torso_mat_cur[2, 0] < 0.0)
                is_prone = getattr(self, "latched_prone", bool(torso_mat_cur[2, 0] < 0.0))
                if steps_since_shove > 25:
                    self.reflex.recovery_time += self.frame_skip * self.model.opt.timestep
            guidance_ctrl, stage_idx, _ = self.reflex.get_guidance_targets(self.reflex.recovery_time, prone=is_prone)
            if cur_z >= 0.74 and cur_up >= 0.82 and cur_r_sole < 0.065 and cur_l_sole < 0.065:
                self.reflex.balance.nominal_pose["right_knee_act"] = float(guidance_ctrl[self.reflex.act_r_knee])
                self.reflex.balance.nominal_pose["left_knee_act"] = float(guidance_ctrl[self.reflex.act_l_knee])
                self.reflex.balance.nominal_pose["right_hip_pitch_act"] = float(guidance_ctrl[self.reflex.act_r_hip_p])
                self.reflex.balance.nominal_pose["left_hip_pitch_act"] = float(guidance_ctrl[self.reflex.act_l_hip_p])
                self.reflex.balance.nominal_pose["right_ankle_pitch_act"] = float(guidance_ctrl[self.reflex.act_r_ank_p])
                self.reflex.balance.nominal_pose["left_ankle_pitch_act"] = float(guidance_ctrl[self.reflex.act_l_ank_p])
                self.reflex.balance.nominal_pose["waist_pitch_act"] = float(guidance_ctrl[self.reflex.act_waist_p])
                base_ctrl = self.reflex.balance.compute_control(self.data)
                target_ctrl = base_ctrl + action_full * 0.05 * (ctrl_max - ctrl_min)
            else:
                target_ctrl = guidance_ctrl + action_full * 0.08 * (ctrl_max - ctrl_min)

        target_ctrl = np.clip(target_ctrl, ctrl_min, ctrl_max)

        # Smooth physical actuation
        self.current_ctrl = 0.65 * self.current_ctrl + 0.35 * target_ctrl
        self.data.ctrl[:] = self.current_ctrl

        # 100% Authentic physics: zero external forces or skyhooks
        for _ in range(self.frame_skip):
            self.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        z = float(self.data.qpos[2])
        uprightness = float(self.data.xmat[self.torso_upper_id].reshape(3, 3)[2, 2])
        r_sole_z = float(self.data.site_xpos[self.right_sole_id][2])
        l_sole_z = float(self.data.site_xpos[self.left_sole_id][2])
        r_palm_z = float(self.data.site_xpos[self.right_palm_id][2])
        l_palm_z = float(self.data.site_xpos[self.left_palm_id][2])
        drift = float(np.hypot(self.data.qpos[0] - self.anchor_x, self.data.qpos[1] - self.anchor_y))

        # Foot-to-Pelvis horizontal distance and knee flexion
        r_sole_pos = self.data.site_xpos[self.right_sole_id]
        l_sole_pos = self.data.site_xpos[self.left_sole_id]
        pelvis_pos = self.data.qpos[:3]
        r_foot_dist = float(np.hypot(r_sole_pos[0] - pelvis_pos[0], r_sole_pos[1] - pelvis_pos[1]))
        l_foot_dist = float(np.hypot(l_sole_pos[0] - pelvis_pos[0], l_sole_pos[1] - pelvis_pos[1]))
        mean_foot_dist = 0.5 * (r_foot_dist + l_foot_dist)

        r_knee_angle = float(self.data.qpos[self.r_knee_qpos_idx])
        l_knee_angle = float(self.data.qpos[self.l_knee_qpos_idx])
        mean_knee_angle = 0.5 * (r_knee_angle + l_knee_angle)

        reward = 0.0
        if self.knockdown_done:
            ref_err = float(np.mean(np.square(self.data.qpos[7:7+len(guidance_ctrl)] - guidance_ctrl)))
            reward += 25.0 * float(np.exp(-ref_err / 0.20))

        # -------------------------------------------------------------
        # BIOMECHANICAL ADJUSTMENT 1: Push-Up Chest Elevation & Ground Foot Tuck
        # Prevents the "Scorpion Cheat" (flailing feet in the air while face is pinned to floor).
        # Enforces human recovery: Push chest off mat with arms, keep feet on floor, tuck knees!
        # -------------------------------------------------------------
        torso_z = float(self.data.xpos[self.torso_upper_id][2])

        if z < 0.45:
            # 1. Anti-Scorpion & HoST (RSS 2025) Grounded-Sole Rule:
            # Heavily penalize lifting feet off the floor while lying down; reward keeping both soles grounded!
            max_foot_z = max(r_sole_z, l_sole_z)
            if max_foot_z > 0.08:
                reward -= 25.0 + 45.0 * (max_foot_z - 0.08)
            if r_sole_z < 0.055 and l_sole_z < 0.055:
                reward += 25.0
            elif r_sole_z < 0.055 or l_sole_z < 0.055:
                reward += 12.0
            elif self.knockdown_done:
                reward -= 25.0

            # 2. Push-up Chest Elevation Rule: Reward pushing upper torso off the floor using arms!
            if torso_z > 0.18:
                reward += 25.0 * min(1.0, (torso_z - 0.18) * 5.0)

            # 3. Ground Palm Press & Arm-Push Clearance: Reward pressing palms and extending elbows to lift chest
            min_palm_z = min(r_palm_z, l_palm_z)
            if min_palm_z < 0.12:
                reward += 10.0
                chest_push_clearance = max(0.0, torso_z - min_palm_z)
                reward += 20.0 * min(1.0, chest_push_clearance * 3.5)

            # 4. Knee Flexion & Active Ground Foot Pull (ONLY when soles are grounded and knee < 1.55 rad to prevent 1.99 rad Scorpion saturation!)
            if 0.25 < mean_knee_angle < 1.55 and r_sole_z < 0.12 and l_sole_z < 0.12:
                reward += 8.0 * min(1.3, mean_knee_angle)
            elif mean_knee_angle >= 1.65:
                reward -= 20.0 * (mean_knee_angle - 1.65)

            # Ground Foot Retraction + HoST (RSS 2025) CoM-Over-Feet Alignment Reward
            if r_sole_z < 0.12 and l_sole_z < 0.12:
                retraction_progress = max(0.0, 0.85 - mean_foot_dist)
                reward += 10.0 * retraction_progress
                reward += 18.0 * float(np.exp(-8.0 * (mean_foot_dist ** 2)))

                if mean_foot_dist < 0.32:
                    reward += 10.0

        # -------------------------------------------------------------
        # WHOLE-BODY ANATOMICAL GUARDRAILS (ALL 27 JOINTS & 21 SEGMENTS)
        # -------------------------------------------------------------
        # Part 1: Head & Cranial Protection (Protect face/neck from pinning & bridging)
        head_z = float(self.data.xpos[self.head_id][2])

        # Detect head floor contact:
        head_on_floor = False
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 == 0 and b2 == self.head_id) or (b2 == 0 and b1 == self.head_id):
                head_on_floor = True
                break

        # Zero-tolerance for neck bridging / head pinning:
        # A human must NEVER lift their hips while leaving the forehead/cranium dragging on the ground!
        # Zero-tolerance for neck bridging / head pinning:
        # Catastrophic penalty: A human must NEVER lift hips while cranium is dragging on the ground!
        if z > 0.22:
            if head_on_floor or head_z < 0.18:
                reward -= 100.0 + 50.0 * max(0.0, z - 0.22)
            if z > torso_z + 0.04 and head_z < 0.22:
                reward -= 35.0 + 35.0 * (z - torso_z)

        # Incentive: Active chest & cranium elevation during prone pushup:
        if z >= 0.20 and not head_on_floor:
            if torso_z > 0.20 and head_z > 0.22:
                reward += 15.0 * min(1.0, (head_z - 0.20) * 5.0)

        # Neck angle posture penalty (prevent extreme neck twist)
        neck_yaw = float(self.data.qpos[10])
        neck_pitch = float(self.data.qpos[11])
        if abs(neck_yaw) > 0.8:
            reward -= 3.0 * (abs(neck_yaw) - 0.8)
        if abs(neck_pitch) > 0.5:
            reward -= 3.0 * (abs(neck_pitch) - 0.5)

        # Part 2: Spine & Core Stability (Prevent unnatural waist contortions)
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
            # Bilateral Palm Press: Reward planting BOTH hands on the mat to execute a true push-up
            if r_palm_z < 0.15 and l_palm_z < 0.15:
                reward += 15.0

        # Part 4: Anti-Splits & Stance Width (Keep feet in natural human envelope)
        foot_span = float(np.hypot(r_sole_pos[0] - l_sole_pos[0], r_sole_pos[1] - l_sole_pos[1]))
        if foot_span > 0.50:
            reward -= 15.0 * (foot_span - 0.50)
        elif z >= 0.75 and 0.18 <= foot_span <= 0.38:
            reward += 5.0  # Ideal shoulder-width standing base bonus

        # Part 5: Supine Anti-Bridge & Roll-Over Mechanics (Chest facing upward, back on floor)
        torso_mat = self.data.xmat[self.torso_upper_id].reshape(3, 3)
        chest_facing_up = float(torso_mat[2, 0])
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

        # -------------------------------------------------------------
        # BIOMECHANICAL ADJUSTMENT 2: Progressive Elevation Milestones
        # Sequential milestones to climb the ladder to full standing.
        # -------------------------------------------------------------
        # Milestone 1: Dynamic Weight-Transfer Bridge (z in [0.32m, 0.65m])
        # A) Foot Sole Load: Reward planting both feet flat under pelvis (ONLY if NOT bridging on back!)
        if z >= 0.32 and chest_facing_up <= 0.20 and r_sole_z < 0.06 and l_sole_z < 0.06 and mean_foot_dist < 0.35:
            reward += 12.0  # Weight transferred into foot soles!

        # B) Hand Release Incentive: Once pelvis is rising to squat (z >= 0.52m), reward lifting hands off the floor!
        if z >= 0.52:
            if r_palm_z > 0.25 and l_palm_z > 0.25:
                reward += 10.0  # Hands detached cleanly from floor!
            elif r_palm_z < 0.12 or l_palm_z < 0.12:
                # Disincentivize camping on palms when pelvis is high
                reward -= 5.0

        # Milestone 2: Squat plant with feet flat under pelvis (z >= 0.55m)
        if z >= 0.55 and r_sole_z < 0.08 and l_sole_z < 0.08 and mean_foot_dist < 0.30:
            reward += 16.0

        # Milestone 3: Full Standing on Two Feet & 5-Second Continuous Balance Lock
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
                self.initial_stand_steps = getattr(self, 'initial_stand_steps', 0) + 1
                reward += 30.0  # Reward maintaining erect balance during initial 3 seconds!
            else:
                self.consecutive_standing_steps = 0

            # If robot collapses on its own before the 3.0s shove, do not fake 3.0s InitStand!
            if z < 0.65:
                reward -= 120.0
                self.fall_type_str = "SELF-TUMBLE (Failed InitStand)"
                self.knockdown_done = True
                self.consecutive_standing_steps = 0
            # At t >= self.initial_stand_target, deliver physical knockdown shove!
            elif self._step_count >= self.initial_stand_target:
                if self.curriculum_phase == 1:
                    # CANONICAL FORWARD FALL ON FACE (PRONE)
                    push_speed = float(self.np_random.uniform(3.4, 4.2))
                    self.data.qvel[0] += push_speed      # Linear forward drive
                    self.data.qvel[4] += -2.5            # Pitch forward around Y-axis
                    self.fall_type_str = "FORWARD KNOCKDOWN (FALL ON FACE / PRONE)"
                    self.latched_prone = True
                else:
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
                self.consecutive_standing_steps = 0  # Reset counter for 5.0s recovery balance hold!
        else:
            # Phase 3: Floor Recovery & 5-Second Continuous Standing Balance Lock
            if is_standing_on_two_feet:
                self.consecutive_standing_steps += 1
                self.standing_steps += 1
                # Sustained standing reward: +50 per step + progressive duration bonus
                reward += 50.0 + 0.2 * min(250, self.consecutive_standing_steps)

                # Mastered full 5 seconds (250 steps at 50Hz = 5.0s)?
                if self.consecutive_standing_steps >= self.target_stand_steps:
                    self.has_succeeded = True
                    reward += 300.0  # Massive 5-Second Continuous Two-Foot Mastery Bonus!
            else:
                # If he falls or loses foot contact before 5 seconds, reset continuous timer!
                if z < 0.72 or uprightness < 0.70 or r_sole_z > 0.08 or l_sole_z > 0.08:
                    self.consecutive_standing_steps = 0

        is_success = bool(self.knockdown_done and self.consecutive_standing_steps >= self.target_stand_steps)

        # Core climbing dynamics:
        # 1. Height progress reward (rewarding climbing upwards towards 0.84m)
        delta_z = z - self.prev_z
        if z < 0.84:
            if not head_on_floor and not (z > torso_z + 0.04 and head_z < 0.22):
                if delta_z > 0:
                    reward += delta_z * 60.0
                else:
                    reward += delta_z * 15.0
        elif z > 0.86:
            # ACCELERATION LEVER 3: Over-jump ceiling penalty (penalize rocket launching into air)
            reward -= (z - 0.84) * 60.0

        # ACCELERATION LEVER 2: Apex braking & velocity damping near 0.84m standing height
        pelvis_vz = float(self.data.qvel[2])
        if z > 0.75 and pelvis_vz > 0.25:
            # Penalize excessive launch velocity so legs decelerate smoothly to stick standing
            reward -= (pelvis_vz - 0.25) * 20.0

        # 2. Sustained height elevation reward + HoST (RSS 2025) Head/Torso Uprightness-Coupled Elevation
        reward += 12.0 * min(0.84, z)
        up_gate = max(0.0, 0.5 * (1.0 + float(np.clip(uprightness, -1.0, 1.0))))
        reward += 18.0 * min(1.45, head_z) * up_gate + 14.0 * min(1.15, torso_z) * up_gate
        if head_z > z + 0.10 and torso_z > z + 0.04:
            reward += 15.0 * min(1.0, (head_z - z) * 2.0)

        # 3. Upright torso orientation guidance (scaled by elevation to avoid floor cobra cheat)
        upright_factor = min(1.0, max(0.15, z / 0.40))
        reward += 6.0 * np.clip(uprightness, -1.0, 1.0) * upright_factor

        # 4. In-place ground stability (penalizing horizontal drift from fall spot)
        reward -= 3.0 * min(1.0, drift)

        # 5. ACCELERATION LEVER 1: Ground magnet (feet must stay glued to the floor)
        if r_sole_z < 0.05 and l_sole_z < 0.05:
            reward += 8.0
        elif z > 0.50:
            # Strong penalty when feet are airborne while the robot is high
            airborne_dist = max(0.0, r_sole_z - 0.05) + max(0.0, l_sole_z - 0.05)
            reward -= airborne_dist * 40.0

        # SKILL 2: Coordinated Arm Ground-Push (Palm Lever)
        # When low on the floor, actively reward pressing palms into the ground to lift torso
        if z < 0.50 and (r_palm_z < 0.15 or l_palm_z < 0.15) and delta_z > 0.0005:
            reward += 12.0 * min(1.0, delta_z * 100.0)

        # SKILL 1: Active Ankle Inverted-Pendulum Balancing
        # Eliminates horizontal drift once upright by micro-adjusting ankles
        if z >= 0.78:
            pelvis_vxy = float(np.hypot(self.data.qvel[0], self.data.qvel[1]))
            pelvis_ang_xy = float(np.hypot(self.data.qvel[3], self.data.qvel[4]))
            if pelvis_vxy < 0.12 and pelvis_ang_xy < 0.20:
                reward += 10.0  # Rock-solid still balance bonus
            else:
                reward -= min(2.0, pelvis_vxy) * 12.0
                reward -= min(2.0, pelvis_ang_xy) * 4.0

        # SKILL 4: Natural Arm Settling (Human "Ready" Stance)
        # Relaxes arms naturally by his sides once standing instead of stiff flailing
        if z >= 0.80:
            arm_diff = self.data.ctrl[5:15] - self.neutral_standing_ctrl[5:15]
            arm_error = float(np.mean(np.square(arm_diff)))
            reward += 4.0 * max(0.0, 1.0 - 5.0 * arm_error)

        # SKILL 3: Knockdown & Disturbance Rejection (Train active push-recovery reflexes)
        # Apply occasional gentle nudges once stably standing so Cyrus learns to brace and hold
        if self.standing_steps >= 25 and (self.standing_steps % 35 == 0):
            push_angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            push_impulse = float(self.np_random.uniform(15.0, 30.0) / 61.32)
            self.data.qvel[0] += float(np.cos(push_angle) * push_impulse)
            self.data.qvel[1] += float(np.sin(push_angle) * push_impulse)

        # 6. Action Smoothing & Energy Conservation (Anti-Jitter)
        # Penalizes high-frequency joint jerk and actuator saturation
        action_diff = action - self.prev_action
        reward -= 0.04 * float(np.sum(np.square(action_diff)))
        self.prev_action = action.copy()
        self.prev_z = z

        reward -= 0.005 * float(np.sum(np.square(action)))

        if z > self.max_z_in_ep:
            self.max_z_in_ep = z

        # Fast Trial Recycling Engine (Accelerates trials/sec by 3x - 5x):
        terminated = False
        if is_success:
            terminated = True
        # 2. Catastrophic drift out of training envelope
        elif drift > 2.2:
            terminated = True
        # 3. Pinned flat on floor with no upward progress after knockdown
        elif self.knockdown_done and (self._step_count > 200 if self.curriculum_phase in (1, 2) else True) and z < 0.25 and torso_z < 0.28 and uprightness < 0.50:
            self.stuck_steps = getattr(self, 'stuck_steps', 0) + 1
            if self.stuck_steps >= (150 if self.curriculum_phase in (1, 2) else 220):
                terminated = True
        else:
            self.stuck_steps = 0

        truncated = self._step_count >= self.max_episode_steps

        kinematic_state = self.analyzer.analyze_frame(self.data, anchor_pos=(self.anchor_x, self.anchor_y))

        info = {
            "height": z,
            "max_height": self.max_z_in_ep,
            "uprightness": uprightness,
            "torso_z": kinematic_state['torso_z'],
            "drift_mm": drift * 1000.0,
            "fall_type": self.fall_type_str,
            "mean_foot_dist": mean_foot_dist,
            "mean_knee_angle": mean_knee_angle,
            "stand_hold_sec": round(self.consecutive_standing_steps * 0.02, 2),
            "initial_stand_sec": round(getattr(self, 'initial_stand_steps', 0) * 0.02, 2),
            "knockdown_done": self.knockdown_done,
            "standing_steps": self.consecutive_standing_steps,
            "is_curriculum": self.is_curriculum,
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
