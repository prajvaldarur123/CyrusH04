import numpy as np
import mujoco


class CyrusKinematicObserver:
    """
    Frame-by-Frame 3D Anatomical Coordinate & Kinematic Diagnostic Engine.
    Continuously inspects all 3D coordinates, detects biomechanical faults,
    classifies recovery stages, and enforces healthy human motor control.
    """

    # Recovery Stages
    STAGE_FALLEN = 0
    STAGE_PUSH_UP = 1
    STAGE_QUADRUPED = 2
    STAGE_WEIGHT_TRANSFER = 3
    STAGE_BIPEDAL_BALANCE = 4

    STAGE_NAMES = {
        0: "FALLEN / FLOOR",
        1: "CHEST PUSH-UP",
        2: "QUADRUPED TUCK",
        3: "WEIGHT TRANSFER",
        4: "BIPEDAL BALANCE (5s HOLD)"
    }

    def __init__(self, model):
        self.model = model

        # Segment Body IDs
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_upper')
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'head')
        self.r_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
        self.l_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_hand')
        self.r_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_shin')
        self.l_shin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_shin')
        self.r_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_foot')
        self.l_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_foot')

        # Sites
        self.r_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_palm')
        self.l_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_palm')
        self.r_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_sole')
        self.l_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_sole')

        # Geoms
        self.rf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'right_foot_geom')
        self.lf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot_geom')
        self.foot_half_height = model.geom_size[self.rf_geom_id][2] if self.rf_geom_id != -1 else 0.02

    def analyze_frame(self, data, anchor_pos=None):
        """
        Inspects live frame physics and returns a complete anatomical coordinate & fault diagnosis.
        """
        head_pos = data.xpos[self.head_id].copy()
        pelvis_pos = data.xpos[self.pelvis_id].copy()
        torso_pos = data.xpos[self.torso_id].copy()
        r_foot_pos = data.xpos[self.r_foot_id].copy()
        l_foot_pos = data.xpos[self.l_foot_id].copy()
        r_hand_pos = data.xpos[self.r_hand_id].copy()
        l_hand_pos = data.xpos[self.l_hand_id].copy()
        r_knee_pos = data.xpos[self.r_shin_id].copy()
        l_knee_pos = data.xpos[self.l_shin_id].copy()

        # Palm site heights
        r_palm_z = float(data.site_xpos[self.r_palm_id][2]) if self.r_palm_id != -1 else r_hand_pos[2]
        l_palm_z = float(data.site_xpos[self.l_palm_id][2]) if self.l_palm_id != -1 else l_hand_pos[2]

        # Sole site heights
        r_sole_z = float(data.site_xpos[self.r_sole_id][2]) if self.r_sole_id != -1 else r_foot_pos[2]
        l_sole_z = float(data.site_xpos[self.l_sole_id][2]) if self.l_sole_id != -1 else l_foot_pos[2]
        mean_sole_z = 0.5 * (r_sole_z + l_sole_z)

        # Torso Orientation & Uprightness
        torso_mat = data.xmat[self.torso_id].reshape(3, 3)
        uprightness = float(torso_mat[2, 2])
        pitch_forward = float(torso_mat[0, 2])

        # Center of Mass (CoM)
        com = data.subtree_com[self.pelvis_id].copy()
        feet_mid_x = 0.5 * (r_foot_pos[0] + l_foot_pos[0])
        feet_mid_y = 0.5 * (r_foot_pos[1] + l_foot_pos[1])
        com_dist_to_feet = float(np.hypot(com[0] - feet_mid_x, com[1] - feet_mid_y))

        # Horizontal foot distance from pelvis
        mean_foot_dist = float(0.5 * (np.hypot(r_foot_pos[0] - pelvis_pos[0], r_foot_pos[1] - pelvis_pos[1]) +
                                     np.hypot(l_foot_pos[0] - pelvis_pos[0], l_foot_pos[1] - pelvis_pos[1])))

        # Velocities
        pelvis_vz = float(data.qvel[2])

        # Contact flags
        palms_planted = bool(r_palm_z < 0.15 or l_palm_z < 0.15)
        palms_airborne = bool(r_palm_z > 0.25 and l_palm_z > 0.25)
        soles_grounded = bool(r_sole_z < 0.06 and l_sole_z < 0.06)

        # -------------------------------------------------------------
        # 1. BIOMECHANICAL STAGE CLASSIFICATION
        # -------------------------------------------------------------
        z = pelvis_pos[2]
        torso_z = torso_pos[2]

        if z >= 0.78 and uprightness >= 0.85 and soles_grounded and palms_airborne and mean_foot_dist < 0.35:
            current_stage = self.STAGE_BIPEDAL_BALANCE
        elif z >= 0.50 and soles_grounded and palms_airborne:
            current_stage = self.STAGE_WEIGHT_TRANSFER
        elif z >= 0.32 and soles_grounded and mean_foot_dist < 0.38 and torso_z >= 0.25:
            current_stage = self.STAGE_QUADRUPED
        elif torso_z >= 0.22 and palms_planted and r_sole_z < 0.15 and l_sole_z < 0.15:
            current_stage = self.STAGE_PUSH_UP
        else:
            current_stage = self.STAGE_FALLEN

        # -------------------------------------------------------------
        # 2. BIOMECHANICAL FAULT DETECTION & ROOT CAUSE ANALYSIS
        # -------------------------------------------------------------
        faults = []

        # Fault A: Scorpion Flail (Airborne feet while torso on floor)
        max_sole_z = max(r_sole_z, l_sole_z)
        if z < 0.45 and max_sole_z > 0.18:
            faults.append(f"SCORPION_FLAIL (Feet at {max_sole_z:.2f}m in air while chest on mat)")

        # Fault B: Chest Pinned to Ground (No push-up effort)
        if z < 0.35 and torso_z < 0.16:
            faults.append(f"CHEST_PINNED (Upper chest at {torso_z:.2f}m not lifting off floor)")

        # Fault C: Catapult / Launch Exploit
        if z > 0.90 or pelvis_vz > 1.1:
            faults.append(f"CATAPULT_LAUNCH (Excessive height {z:.2f}m / vertical speed {pelvis_vz:+.2f}m/s)")

        # Fault D: Uneven Foot Grounding during standing
        if z >= 0.65 and (r_sole_z > 0.08 or l_sole_z > 0.08):
            faults.append(f"FEET_AIRBORNE (Right: {r_sole_z:.2f}m, Left: {l_sole_z:.2f}m off floor)")

        # Fault E: Hand Camping
        if z >= 0.45 and palms_planted:
            faults.append(f"PALM_CAMPING (Palms at {min(r_palm_z, l_palm_z):.2f}m glued to ground)")

        # Fault F: Head Pinned (Cranium dragging on mat)
        if z >= 0.30 and head_pos[2] < 0.20:
            faults.append(f"HEAD_PINNED (Head at {head_pos[2]:.2f}m dragging while pelvis is rising)")

        # Fault G: Spine Contortion (Waist twist/roll excess)
        waist_yaw = float(data.qpos[7])
        waist_roll = float(data.qpos[8])
        if abs(waist_roll) > 0.35 or abs(waist_yaw) > 0.65:
            faults.append(f"SPINE_CONTORTION (Waist roll {waist_roll:+.2f} rad, yaw {waist_yaw:+.2f} rad)")

        # Fault H: Excessive Splits (Over-extended lateral foot stance)
        foot_span = float(np.hypot(r_foot_pos[0] - l_foot_pos[0], r_foot_pos[1] - l_foot_pos[1]))
        if foot_span > 0.55:
            faults.append(f"EXCESSIVE_SPLITS (Foot span {foot_span:.2f}m exceeds human norm 0.35m)")

        # Fault I: Arm Asymmetry (Uncoordinated one-arm push)
        arm_asymmetry = float(abs(r_palm_z - l_palm_z))
        if z < 0.45 and arm_asymmetry > 0.30:
            faults.append(f"ARM_ASYMMETRY (Palms height delta {arm_asymmetry:.2f}m on floor)")

        # Horizontal drift from anchor
        drift_mm = float(np.hypot(pelvis_pos[0] - anchor_pos[0], pelvis_pos[1] - anchor_pos[1]) * 1000.0) if anchor_pos is not None else 0.0

        return {
            'stage': current_stage,
            'stage_name': self.STAGE_NAMES[current_stage],
            'faults': faults,
            'has_faults': len(faults) > 0,
            'pelvis_pos': pelvis_pos,
            'pelvis_z': z,
            'pelvis_vz': pelvis_vz,
            'torso_pos': torso_pos,
            'torso_z': torso_z,
            'head_pos': head_pos,
            'r_palm_z': r_palm_z,
            'l_palm_z': l_palm_z,
            'r_sole_z': r_sole_z,
            'l_sole_z': l_sole_z,
            'r_knee_pos': r_knee_pos,
            'l_knee_pos': l_knee_pos,
            'mean_foot_dist': mean_foot_dist,
            'com': com,
            'com_dist_to_feet': com_dist_to_feet,
            'uprightness': uprightness,
            'pitch_forward': pitch_forward,
            'palms_planted': palms_planted,
            'palms_airborne': palms_airborne,
            'soles_grounded': soles_grounded,
            'drift_mm': drift_mm
        }

    def format_frame_summary(self, frame_idx, state):
        """Formats a clean, detailed single-line coordinate & fault telemetry report."""
        fault_str = " | FAULTS: " + ", ".join(state['faults']) if state['has_faults'] else " | [+] HEALTHY"
        return (
            f"[F{frame_idx:03d} | {state['stage_name']:17s}] "
            f"Pelvis: {state['pelvis_z']:.2f}m | Torso: {state['torso_z']:.2f}m | "
            f"Palms: [{state['r_palm_z']:.2f}m, {state['l_palm_z']:.2f}m] | "
            f"Soles: [{state['r_sole_z']:.2f}m, {state['l_sole_z']:.2f}m] | "
            f"Upright: {state['uprightness']:.2f}{fault_str}"
        )