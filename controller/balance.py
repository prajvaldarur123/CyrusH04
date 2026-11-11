import numpy as np
import mujoco


class CyrusBalanceController:
    """
    Human-Inspired 10-Principle Dynamic Balance & Push Recovery Controller for Cyrus.
    Implements:
    1. Center of Mass (CoM) & ZMP Control (LIPM inverted pendulum dynamics)
    2. Ankle Strategy for small disturbances (corrective ankle torque shifting CoP)
    3. Hip Strategy for medium disturbances (coordinated hip/waist counter-rotation)
    4. Stepping Strategy for large disturbances (Capture Point foot placement)
    5. Arm Counterbalancing (angular momentum compensation against body rotation)
    6. Knee Flexion & Shock Absorption (compliant kinetic energy dissipation)
    7. Torso Stabilization (upright cervical/lumbar alignment)
    8. Foot Contact & Friction Management (flat sole contact, no slipping or tiptoeing)
    9. Recovery & Stabilization (smooth return to stationary standing)
    10. Adaptive Strategy Selector (Ankle -> Hip+Arms -> Stepping)
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model

        # Cache body, geom, and site IDs
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")

        self.rf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        self.lf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        self.right_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.left_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Build actuator index mapping
        self.act_map = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.act_map[name] = i

        # Default natural standing posture
        self.nominal_pose = {
            # Spine
            "waist_yaw_act": 0.0,
            "waist_roll_act": 0.0,
            "waist_pitch_act": 0.0,
            # Head
            "neck_yaw_act": 0.0,
            "neck_pitch_act": 0.0,
            # Arms (naturally relaxed at sides with athletic carriage)
            "right_shoulder_pitch_act": 0.05,
            "right_shoulder_roll_act": -0.08,
            "right_shoulder_yaw_act": 0.0,
            "right_elbow_act": -0.25,
            "right_wrist_act": 0.0,
            "left_shoulder_pitch_act": 0.05,
            "left_shoulder_roll_act": 0.08,
            "left_shoulder_yaw_act": 0.0,
            "left_elbow_act": -0.25,
            "left_wrist_act": 0.0,
            # Legs (erect, feet parallel and flat)
            "right_hip_yaw_act": 0.0,
            "right_hip_roll_act": 0.0,
            "right_hip_pitch_act": 0.0,
            "right_knee_act": 0.0,
            "right_ankle_pitch_act": 0.0,
            "right_ankle_roll_act": 0.0,
            "left_hip_yaw_act": 0.0,
            "left_hip_roll_act": 0.0,
            "left_hip_pitch_act": 0.0,
            "left_knee_act": 0.0,
            "left_ankle_pitch_act": 0.0,
            "left_ankle_roll_act": 0.0,
        }

        # Filter state for Center of Mass velocity
        self.prev_com = None
        self.com_vel = np.zeros(3)

        # Stepping Strategy State Machine
        self.step_state = "IDLE"  # "IDLE", "SWING", "LAND"
        self.swing_leg = None
        self.step_timer = 0.0
        self.step_duration = 0.32
        self.step_target_pitch = 0.0
        self.step_target_roll = 0.0

        # Active strategy indicator
        self.active_strategy = "ANKLE"

        # Vestibular feedback gains
        self.kp_pitch = 3.6
        self.kd_pitch = 0.45
        self.kp_roll = 2.8
        self.kd_roll = 0.35

    def reset(self):
        """Resets dynamic filters and stepping state to initial standing equilibrium."""
        self.step_state = "IDLE"
        self.swing_leg = None
        self.step_timer = 0.0
        self.prev_com = None
        self.com_vel = np.zeros(3)
        self.active_strategy = "ANKLE"

    def update_state(self, data: mujoco.MjData, dt: float = 0.002) -> dict:
        """
        Calculates Center of Mass, filtered velocity, and Instantaneous Capture Point (ICP).
        """
        com = data.subtree_com[self.pelvis_id].copy()
        if self.prev_com is None:
            self.prev_com = com.copy()
            self.com_vel = np.zeros(3)
        else:
            raw_vel = (com - self.prev_com) / dt
            self.com_vel = 0.65 * self.com_vel + 0.35 * raw_vel
            self.prev_com = com.copy()

        # Inverted pendulum natural frequency
        z0 = max(0.40, com[2])
        omega_0 = np.sqrt(9.81 / z0)

        # Instantaneous Capture Point: xi = x + x_dot / omega_0
        icp_x = com[0] + self.com_vel[0] / omega_0
        icp_y = com[1] + self.com_vel[1] / omega_0
        capture_point = np.array([icp_x, icp_y])

        # Dual-foot support center from foot geoms
        rf_pos = data.geom_xpos[self.rf_geom_id][:2]
        lf_pos = data.geom_xpos[self.lf_geom_id][:2]
        support_center = 0.5 * (rf_pos + lf_pos)
        cp_offset = capture_point - support_center

        # Foot boundaries (heel -0.08, toe +0.14, lateral +-0.06)
        x_min = min(rf_pos[0], lf_pos[0]) - 0.08
        x_max = max(rf_pos[0], lf_pos[0]) + 0.14
        y_min = min(rf_pos[1], lf_pos[1]) - 0.06
        y_max = max(rf_pos[1], lf_pos[1]) + 0.06

        mos_sagittal = min(x_max - icp_x, icp_x - x_min)
        mos_coronal = min(y_max - icp_y, icp_y - y_min)

        return {
            "com": com,
            "com_vel": self.com_vel,
            "omega_0": omega_0,
            "capture_point": capture_point,
            "support_center": support_center,
            "cp_offset": cp_offset,
            "mos_sagittal": mos_sagittal,
            "mos_coronal": mos_coronal,
        }

    def compute_control(self, data: mujoco.MjData, gesture_offsets: dict = None) -> np.ndarray:
        """
        Calculates whole-body joint targets across all 10 biomechanical principles.
        """
        dt = self.model.opt.timestep
        st = self.update_state(data, dt=dt)
        cp_offset = st["cp_offset"]
        cp_dist = np.linalg.norm(cp_offset)

        pelvis_rot = data.xmat[self.pelvis_id].reshape(3, 3)
        # Exact kinematic pitch & roll from rotation matrix (replaces raw quaternion component bug)
        pelvis_pitch = float(np.arctan2(pelvis_rot[0, 2], pelvis_rot[2, 2]))
        pelvis_roll = float(np.arctan2(pelvis_rot[1, 2], pelvis_rot[2, 2]))
        pitch_vel = float(data.qvel[4])
        roll_vel = float(data.qvel[3])

        # -------------------------------------------------------------
        # 1. ADAPTIVE STRATEGY SELECTOR
        # -------------------------------------------------------------
        if self.step_state == "IDLE":
            # Stepping triggers when Capture Point is well beyond the foot boundary
            if cp_offset[0] < -0.13 or cp_offset[0] > +0.18 or abs(cp_offset[1]) > 0.16 or abs(pelvis_pitch) > 0.32:
                self.step_state = "SWING"
                self.step_timer = 0.0
                self.active_strategy = "STEPPING"

                # Choose swing leg:
                if cp_offset[1] > 0.02:
                    self.swing_leg = "left"   # Pushed left -> step left
                elif cp_offset[1] < -0.02:
                    self.swing_leg = "right"  # Pushed right -> step right
                else:
                    self.swing_leg = "left" if roll_vel >= 0 else "right"

                self.step_target_pitch = np.clip(cp_offset[0] * 3.5, -0.55, 0.55)
                self.step_target_roll = np.clip(cp_offset[1] * 2.5, -0.25, 0.25)

            elif cp_dist > 0.035 or abs(pitch_vel) > 0.6 or abs(roll_vel) > 0.5 or abs(pelvis_pitch) > 0.08:
                self.active_strategy = "HIP_ARM"
            else:
                self.active_strategy = "ANKLE"

        # Initialize targets from nominal standing posture
        targets = np.zeros(self.model.nu, dtype=np.float32)
        for act_name, val in self.nominal_pose.items():
            if act_name in self.act_map:
                targets[self.act_map[act_name]] = val

        # -------------------------------------------------------------
        # 2. ANKLE STRATEGY (Corrective ankle torque shifting CoP)
        # -------------------------------------------------------------
        cmd_pitch = np.clip(self.kp_pitch * pelvis_pitch + self.kd_pitch * pitch_vel, -0.75, 0.75)
        cmd_roll = np.clip(-self.kp_roll * pelvis_roll - self.kd_roll * roll_vel, -0.35, 0.35)

        targets[self.act_map["right_ankle_pitch_act"]] += cmd_pitch
        targets[self.act_map["left_ankle_pitch_act"]] += cmd_pitch
        targets[self.act_map["right_ankle_roll_act"]] += cmd_roll
        targets[self.act_map["left_ankle_roll_act"]] += cmd_roll

        # -------------------------------------------------------------
        # 3. HIP STRATEGY & TORSO STABILIZATION
        # -------------------------------------------------------------
        # When falling back (pitch < 0): hips extend (hip_p < 0), waist leans forward (waist_p > 0)
        # When falling forward (pitch > 0): hips flex (hip_p > 0), waist leans back (waist_p < 0)
        hip_p = np.clip(2.0 * pelvis_pitch + 0.18 * pitch_vel, -0.55, 0.55)
        waist_p = np.clip(-1.6 * pelvis_pitch - 0.12 * pitch_vel, -0.40, 0.40)
        hip_r = np.clip(-1.3 * pelvis_roll - 0.14 * roll_vel, -0.30, 0.30)
        waist_r = np.clip(1.1 * pelvis_roll + 0.10 * roll_vel, -0.25, 0.25)

        targets[self.act_map["right_hip_pitch_act"]] += hip_p
        targets[self.act_map["left_hip_pitch_act"]] += hip_p
        targets[self.act_map["waist_pitch_act"]] += waist_p
        targets[self.act_map["right_hip_roll_act"]] += hip_r
        targets[self.act_map["left_hip_roll_act"]] += hip_r
        targets[self.act_map["waist_roll_act"]] += waist_r

        # -------------------------------------------------------------
        # 4. ARM COUNTERBALANCING (Dynamic Angular Momentum Injection)
        # -------------------------------------------------------------
        # Swing arms opposite to torso pitch angular velocity to inject counter-torque
        arm_p = np.clip(-0.65 * pitch_vel - 1.8 * pelvis_pitch, -1.0, 1.0)
        targets[self.act_map["right_shoulder_pitch_act"]] += arm_p + 0.05
        targets[self.act_map["left_shoulder_pitch_act"]] += arm_p + 0.05

        # Lateral roll arm reaction
        targets[self.act_map["right_shoulder_roll_act"]] += np.clip(-0.08 + 0.40 * roll_vel, -0.50, 0.50)
        targets[self.act_map["left_shoulder_roll_act"]] += np.clip(0.08 + 0.40 * roll_vel, -0.50, 0.50)

        # Athletic elbow shock compliance
        targets[self.act_map["right_elbow_act"]] -= 0.15 * min(1.0, abs(pitch_vel))
        targets[self.act_map["left_elbow_act"]] -= 0.15 * min(1.0, abs(pitch_vel))

        # -------------------------------------------------------------
        # 5. KNEE FLEXION & SHOCK ABSORPTION
        # -------------------------------------------------------------
        impact = abs(pitch_vel) + abs(roll_vel) + 2.0 * abs(pelvis_pitch)
        if impact > 0.40 and self.step_state == "IDLE":
            knee_bend = np.clip(0.12 * impact, 0.0, 0.25)
            targets[self.act_map["right_knee_act"]] += knee_bend
            targets[self.act_map["left_knee_act"]] += knee_bend
            targets[self.act_map["right_hip_pitch_act"]] -= knee_bend / 2.0
            targets[self.act_map["left_hip_pitch_act"]] -= knee_bend / 2.0
            targets[self.act_map["right_ankle_pitch_act"]] -= knee_bend / 2.0
            targets[self.act_map["left_ankle_pitch_act"]] -= knee_bend / 2.0

        # -------------------------------------------------------------
        # 6. STEPPING STRATEGY EXECUTION
        # -------------------------------------------------------------
        if self.step_state == "SWING":
            self.step_timer += dt
            s = min(1.0, max(0.0, self.step_timer / self.step_duration))
            lift = 0.05 * 4.0 * s * (1.0 - s)
            advance = 0.5 * (1.0 - np.cos(np.pi * s))

            sw = self.swing_leg
            st = "left" if sw == "right" else "right"

            sw_hp = self.step_target_pitch * advance
            sw_hr = self.step_target_roll * advance
            sw_k = lift / 0.12

            targets[self.act_map[f"{sw}_hip_pitch_act"]] = sw_hp
            targets[self.act_map[f"{sw}_hip_roll_act"]] = sw_hr
            targets[self.act_map[f"{sw}_knee_act"]] = sw_k
            targets[self.act_map[f"{sw}_ankle_pitch_act"]] = -sw_hp / 2.0 - sw_k / 2.0

            # Stance leg stabilizes weight and counters roll
            targets[self.act_map[f"{st}_knee_act"]] += 0.08
            targets[self.act_map[f"{st}_hip_roll_act"]] += (0.05 if st == "right" else -0.05)

            if s >= 1.0:
                self.step_state = "LAND"
                self.step_timer = 0.0

        elif self.step_state == "LAND":
            self.step_timer += dt
            if self.step_timer >= 0.25:
                self.step_state = "IDLE"
                self.swing_leg = None

        # -------------------------------------------------------------
        # 7. ADD HIGH-LEVEL GESTURE OFFSETS
        # -------------------------------------------------------------
        if gesture_offsets:
            for act_name, offset in gesture_offsets.items():
                if act_name in self.act_map:
                    targets[self.act_map[act_name]] += offset

        # Clamp all commands within physical actuator limits
        for i in range(self.model.nu):
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[i]
            targets[i] = np.clip(targets[i], ctrl_min, ctrl_max)

        return targets

    def get_telemetry(self, data: mujoco.MjData) -> dict:
        """
        Returns real-time telemetry dictionary of CoM, Capture Point, Margin of Stability, and Strategy.
        """
        st = self.update_state(data)
        return {
            "strategy": self.active_strategy,
            "step_state": self.step_state,
            "swing_leg": self.swing_leg,
            "com": st["com"],
            "com_vel": st["com_vel"],
            "capture_point": st["capture_point"],
            "mos_sagittal": st["mos_sagittal"],
            "mos_coronal": st["mos_coronal"],
        }


# Alias for explicit naming
CyrusDynamicBalanceEngine = CyrusBalanceController
