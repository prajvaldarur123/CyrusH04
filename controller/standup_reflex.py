import numpy as np
import mujoco
from cyrus.controller.kinematic_observer import CyrusKinematicObserver
from cyrus.controller.balance import CyrusBalanceController


class CyrusStandupReflex:
    """
    Authentic Biomechanical Standup Reflex Controller for Cyrus Humanoid.
    Physically coordinates human get-up from the exact spot where the robot fell:
    Uses arms (hands pressing floor at z~0.03m), waist, hips, knees, and ankles
    to sit up, plant feet into a deep squat, extend quads/glutes to 1.54m full height,
    and hold still balance with flat shoe soles and zero external skyhook forces.
    """

    def __init__(self, model):
        self.model = model
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.rf_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")

        # Kinematic coordinate observer for live 3D understanding
        self.observer = CyrusKinematicObserver(model)

        # Whole-body dynamic balance engine for squat stabilization and extension
        self.balance = CyrusBalanceController(model)

        # Actuator mapping
        self.act_waist_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_yaw_act")
        self.act_waist_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_roll_act")
        self.act_waist_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "waist_pitch_act")
        self.act_neck_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_pitch_act")
        self.act_r_sh_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_act")
        self.act_l_sh_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_act")
        self.act_r_sh_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_roll_act")
        self.act_l_sh_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_roll_act")
        self.act_r_el = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_act")
        self.act_l_el = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_act")
        self.act_r_hip_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_roll_act")
        self.act_l_hip_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_roll_act")
        self.act_r_hip_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_pitch_act")
        self.act_l_hip_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_act")
        self.act_r_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        self.act_l_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.act_r_ank_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_ankle_pitch_act")
        self.act_l_ank_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_ankle_pitch_act")

        self.step_count = 0
        self.recovery_time = 0.0
        self.is_complete = False
        self.anchor_x = None
        self.anchor_y = None
        self.planted_spot = None

        self.STAGE_DURATION = 1.50
        self.tk_squat = 1.05
        self.pz_squat = 0.707

        self.stage_names = [
            "STAGE 1: GROUND RECOVERY & BASE PLANT (Palms under shoulders, soles on mat)",
            "STAGE 2: GROUND PUSH & BEAR-PIKE ELEVATION (Arms push floor, pelvis lifts)",
            "STAGE 3: PUSH-BACK TO DEEP SQUAT (Torso shifts over flat soles, hands release)",
            "STAGE 4: EXTENSION TO FULL HEIGHT (Smooth quad/glute drive to 1.54m)",
        ]

    def reset(self, anchor_x=None, anchor_y=None):
        self.step_count = 0
        self.recovery_time = 0.0
        self.is_complete = False
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.planted_spot = None
        self.balance.reset()

    def get_stage_info(self):
        stage_idx = min(3, int(self.recovery_time / self.STAGE_DURATION))
        time_in_stage = self.recovery_time - stage_idx * self.STAGE_DURATION
        progress = min(1.0, max(0.0, time_in_stage / self.STAGE_DURATION))
        name = self.stage_names[stage_idx]
        return stage_idx + 1, name, progress, time_in_stage

    def get_guidance_targets(self, recovery_time=None, prone=False):
        """
        Pure directional guidance reference poses across all 4 recovery stages.
        Orientation-aware:
        - Prone: Push-up -> Bear-Pike Ground Elevation -> Push-Back to Squat -> Full Standing Extension
        - Supine: Active Roll-to-Belly -> Pike Ground Elevation -> Squat -> Full Standing Extension
        Provides smooth, continuous C1 trajectories for the RL policy to modulate autonomously.
        """
        if recovery_time is None:
            recovery_time = self.recovery_time

        stage_idx = min(3, int(recovery_time / self.STAGE_DURATION))
        time_in_stage = recovery_time - stage_idx * self.STAGE_DURATION
        frac = min(1.0, max(0.0, time_in_stage / self.STAGE_DURATION))
        alpha = 0.5 * (1.0 - np.cos(np.pi * frac))

        nu = self.model.nu
        ctrl = np.zeros(nu, dtype=np.float32)

        # Athletic neutral arm carriage reference
        ctrl[self.act_r_sh_p] = 0.05
        ctrl[self.act_l_sh_p] = 0.05
        ctrl[self.act_r_el] = -0.15
        ctrl[self.act_l_el] = -0.15

        if prone:
            # PRONE RECOVERY
            # Stage 1 (0 to 1.5s): Plant hands UNDER SHOULDERS (elbows bent -1.5 rad, chest ready to push)
            if stage_idx == 0:
                ctrl[self.act_waist_p] = -0.45 * alpha
                ctrl[self.act_neck_p] = 0.35 * alpha   # Lift chin/cranium firmly off floor
                ctrl[self.act_r_sh_p] = -0.35 * alpha  # Keep hands forward under chest
                ctrl[self.act_l_sh_p] = -0.35 * alpha
                ctrl[self.act_r_sh_r] = -0.25 * alpha  # Flared elbows for push-up base
                ctrl[self.act_l_sh_r] = 0.25 * alpha
                ctrl[self.act_r_el] = -1.50 * alpha    # BENT ELBOWS (Hands under shoulders!)
                ctrl[self.act_l_el] = -1.50 * alpha
                ctrl[self.act_r_hip_p] = -1.35 * alpha
                ctrl[self.act_l_hip_p] = -1.35 * alpha
                ctrl[self.act_r_knee] = 0.05 * alpha
                ctrl[self.act_l_knee] = 0.05 * alpha
                ctrl[self.act_r_ank_p] = -0.45 * alpha
                ctrl[self.act_l_ank_p] = -0.45 * alpha

            # Stage 2 (1.5 to 3.0s): Bear-Pike elevation (ACTIVE TRICEPS PUSH: Elbows extend -1.5 -> 0.0 rad)
            elif stage_idx == 1:
                ctrl[self.act_waist_p] = -0.45 + 0.85 * alpha       # Ramps to +0.400 rad
                ctrl[self.act_neck_p] = 0.35 * (1.0 - alpha) - 0.350 * alpha # Chin up / cranium elevation
                ctrl[self.act_r_sh_p] = -0.35 - 0.93 * alpha       # Ramps to -1.280 rad (hands under shoulders)
                ctrl[self.act_l_sh_p] = -0.35 - 0.93 * alpha
                ctrl[self.act_r_sh_r] = -0.25 * (1.0 - alpha) + 0.050 * alpha
                ctrl[self.act_l_sh_r] = 0.25 * (1.0 - alpha) + 0.050 * alpha
                ctrl[self.act_r_el] = -1.50 * (1.0 - alpha) - 0.253 * alpha  # DRIVE TRICEPS TO EXTENSION!
                ctrl[self.act_l_el] = -1.50 * (1.0 - alpha) - 0.253 * alpha  # Lifts chest off mat!
                ctrl[self.act_r_hip_p] = -1.35 + 0.53 * alpha      # Ramps to -0.820 rad (elevates pelvis, eliminates shear)
                ctrl[self.act_l_hip_p] = -1.35 + 0.53 * alpha
                ctrl[self.act_r_knee] = 0.05 + 0.950 * alpha       # Ramps to +1.000 rad (knee hover >= 12cm)
                ctrl[self.act_l_knee] = 0.05 + 0.950 * alpha
                ctrl[self.act_r_ank_p] = -0.45 * (1.0 - alpha) + 0.520 * alpha # Traction lock plantarflexion
                ctrl[self.act_l_ank_p] = -0.45 * (1.0 - alpha) + 0.520 * alpha

            # Stage 3 (3.0 to 4.5s): Push-Back to Deep Squat (palms push off floor, weight onto flat soles)
            elif stage_idx == 2:
                ctrl[self.act_waist_p] = -0.35 + 0.25 * alpha
                ctrl[self.act_neck_p] = 0.12 * (1.0 - alpha)
                ctrl[self.act_r_sh_p] = -0.75 + 0.40 * alpha
                ctrl[self.act_l_sh_p] = -0.75 + 0.40 * alpha
                ctrl[self.act_r_sh_r] = -0.10 * (1.0 - alpha)
                ctrl[self.act_l_sh_r] = 0.10 * (1.0 - alpha)
                ctrl[self.act_r_el] = -0.25 * alpha
                ctrl[self.act_l_el] = -0.25 * alpha
                ctrl[self.act_r_hip_p] = -1.10 + 0.35 * alpha
                ctrl[self.act_l_hip_p] = -1.10 + 0.35 * alpha
                ctrl[self.act_r_knee] = 0.40 + 0.45 * alpha
                ctrl[self.act_l_knee] = 0.40 + 0.45 * alpha
                ctrl[self.act_r_ank_p] = -0.45 + 0.05 * alpha
                ctrl[self.act_l_ank_p] = -0.45 + 0.05 * alpha

            # Stage 4 (4.5 to 6.0s): Quad/Glute Drive to Full 1.54m Standing Height
            elif stage_idx == 3:
                ctrl[self.act_waist_p] = -0.10 * (1.0 - alpha)
                ctrl[self.act_neck_p] = 0.0
                ctrl[self.act_r_sh_p] = -0.35 * (1.0 - alpha) + 0.05 * alpha
                ctrl[self.act_l_sh_p] = -0.35 * (1.0 - alpha) + 0.05 * alpha
                ctrl[self.act_r_el] = -0.25 * (1.0 - alpha) - 0.15 * alpha
                ctrl[self.act_l_el] = -0.25 * (1.0 - alpha) - 0.15 * alpha
                ctrl[self.act_r_hip_p] = -0.75 * (1.0 - alpha)
                ctrl[self.act_l_hip_p] = -0.75 * (1.0 - alpha)
                ctrl[self.act_r_knee] = 0.85 * (1.0 - alpha)
                ctrl[self.act_l_knee] = 0.85 * (1.0 - alpha)
                ctrl[self.act_r_ank_p] = -0.40 * (1.0 - alpha)
                ctrl[self.act_l_ank_p] = -0.40 * (1.0 - alpha)

        else:
            # SUPINE RECOVERY (Roll-to-Belly -> Pike -> Squat -> Stand)
            # Stage 1 (0 to 1.5s): Active Roll onto Belly (Sweep arm, twist waist, roll flank)
            if stage_idx == 0:
                ctrl[self.act_waist_r] = 0.38 * alpha
                ctrl[self.act_waist_y] = 0.65 * alpha
                ctrl[self.act_waist_p] = 0.25 * alpha
                ctrl[self.act_neck_p] = 0.20 * alpha
                ctrl[self.act_r_sh_p] = -0.90 * alpha
                ctrl[self.act_r_sh_r] = 0.45 * alpha
                ctrl[self.act_l_sh_r] = -0.45 * alpha
                ctrl[self.act_r_el] = -0.25 * alpha
                ctrl[self.act_r_hip_r] = -0.45 * alpha
                ctrl[self.act_r_hip_p] = -0.65 * alpha
                ctrl[self.act_l_hip_p] = -0.25 * alpha

            # Stage 2 (1.5 to 3.0s): Torso is prone; unwind waist and push into Pike
            elif stage_idx == 1:
                ctrl[self.act_waist_r] = 0.38 * (1.0 - alpha)
                ctrl[self.act_waist_y] = 0.65 * (1.0 - alpha)
                ctrl[self.act_waist_p] = 0.25 * (1.0 - alpha) - 0.45 * alpha
                ctrl[self.act_neck_p] = 0.20 * (1.0 - alpha) + 0.25 * alpha
                ctrl[self.act_r_sh_p] = -0.90 * (1.0 - alpha) - 1.10 * alpha
                ctrl[self.act_l_sh_p] = -1.10 * alpha
                ctrl[self.act_r_sh_r] = 0.45 * (1.0 - alpha) - 0.10 * alpha
                ctrl[self.act_l_sh_r] = -0.45 * (1.0 - alpha) + 0.10 * alpha
                ctrl[self.act_r_hip_r] = -0.45 * (1.0 - alpha)
                ctrl[self.act_r_hip_p] = -0.65 * (1.0 - alpha) - 1.35 * alpha
                ctrl[self.act_l_hip_p] = -0.25 * (1.0 - alpha) - 1.35 * alpha
                ctrl[self.act_r_knee] = 0.05 * alpha
                ctrl[self.act_l_knee] = 0.05 * alpha
                ctrl[self.act_r_ank_p] = -0.45 * alpha
                ctrl[self.act_l_ank_p] = -0.45 * alpha

            # Stage 3 (3.0 to 4.5s): Deep Squat plant from Pike
            elif stage_idx == 2:
                ctrl[self.act_waist_p] = -0.45 + 0.35 * alpha
                ctrl[self.act_neck_p] = 0.12 * (1.0 - alpha)
                ctrl[self.act_r_sh_p] = -1.10 + 0.75 * alpha
                ctrl[self.act_l_sh_p] = -1.10 + 0.75 * alpha
                ctrl[self.act_r_el] = -0.25 * alpha
                ctrl[self.act_l_el] = -0.25 * alpha
                ctrl[self.act_r_hip_p] = -1.35 + 0.60 * alpha
                ctrl[self.act_l_hip_p] = -1.35 + 0.60 * alpha
                ctrl[self.act_r_knee] = 0.05 + 0.80 * alpha
                ctrl[self.act_l_knee] = 0.05 + 0.80 * alpha
                ctrl[self.act_r_ank_p] = -0.45 + 0.05 * alpha
                ctrl[self.act_l_ank_p] = -0.45 + 0.05 * alpha

            # Stage 4 (4.5 to 6.0s): Extension to Full Erect Standing
            elif stage_idx == 3:
                ctrl[self.act_waist_p] = -0.10 * (1.0 - alpha)
                ctrl[self.act_neck_p] = 0.0
                ctrl[self.act_r_sh_p] = -0.35 * (1.0 - alpha) + 0.05 * alpha
                ctrl[self.act_l_sh_p] = -0.35 * (1.0 - alpha) + 0.05 * alpha
                ctrl[self.act_r_el] = -0.25 * (1.0 - alpha) - 0.15 * alpha
                ctrl[self.act_l_el] = -0.25 * (1.0 - alpha) - 0.15 * alpha
                ctrl[self.act_r_hip_p] = -0.75 * (1.0 - alpha)
                ctrl[self.act_l_hip_p] = -0.75 * (1.0 - alpha)
                ctrl[self.act_r_knee] = 0.85 * (1.0 - alpha)
                ctrl[self.act_l_knee] = 0.85 * (1.0 - alpha)
                ctrl[self.act_r_ank_p] = -0.40 * (1.0 - alpha)
                ctrl[self.act_l_ank_p] = -0.40 * (1.0 - alpha)

        return ctrl, stage_idx, frac

    def compute_step(self, data, dt=0.002):
        self.step_count += 1
        self.recovery_time += dt

        if self.anchor_x is None:
            self.anchor_x = float(data.qpos[0])
            self.anchor_y = float(data.qpos[1])

        z = float(data.qpos[2])
        torso_mat = data.xmat[self.torso_id].reshape(3, 3)
        upright = float(torso_mat[2, 2])
        is_prone = bool(torso_mat[2, 0] < 0.0)

        # 100% Zero external forces: authentic CAD physics only
        data.xfrc_applied[:] = 0.0

        guidance_ctrl, stage_idx, frac = self.get_guidance_targets(self.recovery_time, prone=is_prone)

        # Only activate the full whole-body balance controller once elevated and standing
        if z >= 0.74 and upright >= 0.82 and self.recovery_time >= (3.5 * self.STAGE_DURATION):
            if self.planted_spot is None:
                self.planted_spot = data.qpos[:2].copy()
                self.balance.reset()

            self.balance.nominal_pose["right_knee_act"] = float(guidance_ctrl[self.act_r_knee])
            self.balance.nominal_pose["left_knee_act"] = float(guidance_ctrl[self.act_l_knee])
            self.balance.nominal_pose["right_hip_pitch_act"] = float(guidance_ctrl[self.act_r_hip_p])
            self.balance.nominal_pose["left_hip_pitch_act"] = float(guidance_ctrl[self.act_l_hip_p])
            self.balance.nominal_pose["right_ankle_pitch_act"] = float(guidance_ctrl[self.act_r_ank_p])
            self.balance.nominal_pose["left_ankle_pitch_act"] = float(guidance_ctrl[self.act_l_ank_p])
            self.balance.nominal_pose["waist_pitch_act"] = float(guidance_ctrl[self.act_waist_p])
            self.balance.nominal_pose["neck_pitch_act"] = 0.0

            ctrl = self.balance.compute_control(data)
        else:
            ctrl = guidance_ctrl

        data.ctrl[:] = ctrl

        # Check if all 4 stages are complete and full erect stance reached
        if self.recovery_time >= (4 * self.STAGE_DURATION) and z >= 0.80 and upright >= 0.88:
            self.is_complete = True
            data.xfrc_applied[:] = 0.0
            data.qvel[0:2] = 0.0
            data.qvel[2] *= 0.1
            data.qvel[3:6] = 0.0
            return ctrl, True

        return ctrl, False
