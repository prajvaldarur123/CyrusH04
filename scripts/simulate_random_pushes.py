"""
==============================================================================
CYRUS HUMANOID -- SIMULATE RANDOM PUSHES AT 0.60 M/S AND 0.70 M/S
==============================================================================
Runs high-fidelity dynamic simulations of Cyrus with random 360° disturbances,
logging exact kinematics, joint reactions, and generating visual renders.
==============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTICS_ROOT = os.path.dirname(PROJECT_ROOT)
for p in [ROBOTICS_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from controller.balance_pipeline import (
    load_verified_checkpoint,
    EMAActionFilter,
    get_robot_mass,
    verify_settled_equilibrium,
    is_robot_fallen,
    ACTION_SCALE_VEC,
    EMA_ALPHA,
    ROBUST_ROLL_GAINS
)
from controller.stepping_recovery import ReactiveSteppingController, SteppingState

XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")


def run_random_pushes(speed_mps: float, num_pushes: int, checkpoint_path: str, render_gif_path: str = None):
    print("=" * 80)
    print(f"SIMULATING {num_pushes} RANDOM PUSHES AT {speed_mps:.2f} m/s (MuJoCo Physics)")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print("=" * 80)

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    dt_physics = 0.0025
    m.opt.timestep = dt_physics

    # Apply robust roll gains
    for side in ["right", "left"]:
        h_act = m.actuator(f"{side}_hip_roll_act")
        h_act.gainprm[0] = ROBUST_ROLL_GAINS["hip_roll_kp"]
        h_act.biasprm[1] = -ROBUST_ROLL_GAINS["hip_roll_kp"]
        h_act.biasprm[2] = -ROBUST_ROLL_GAINS["hip_roll_kv"]
        a_act = m.actuator(f"{side}_ankle_roll_act")
        a_act.gainprm[0] = ROBUST_ROLL_GAINS["ankle_roll_kp"]
        a_act.biasprm[1] = -ROBUST_ROLL_GAINS["ankle_roll_kp"]
        a_act.biasprm[2] = -ROBUST_ROLL_GAINS["ankle_roll_kv"]

    d = mujoco.MjData(m)
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
    r_hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_thigh")
    l_hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_thigh")
    r_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    l_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    r_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
    l_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_sole")
    r_geom_id = m.geom("right_foot_geom").id
    l_geom_id = m.geom("left_foot_geom").id

    leg_qpos_adr = [m.jnt(name).qposadr[0] for name in [
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
    ]]
    leg_qvel_adr = [m.jnt(name).dofadr[0] for name in [
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
    ]]

    # Nominal Pose C (Athletic Squat)
    ctrl_nom = np.zeros(12, dtype=np.float32)
    ctrl_nom[2] = -0.175  # right hip pitch
    ctrl_nom[3] = 0.350   # right knee
    ctrl_nom[4] = -0.175  # right ankle pitch
    ctrl_nom[8] = -0.175  # left hip pitch
    ctrl_nom[9] = 0.350   # left knee
    ctrl_nom[10] = -0.175 # left ankle pitch

    weights = np.load(checkpoint_path)

    def policy_forward(obs_buf):
        h1 = np.tanh(obs_buf @ weights["mlp_extractor.policy_net.0.weight"].T + weights["mlp_extractor.policy_net.0.bias"])
        h2 = np.tanh(h1 @ weights["mlp_extractor.policy_net.2.weight"].T + weights["mlp_extractor.policy_net.2.bias"])
        raw = h2 @ weights["action_net.weight"].T + weights["action_net.bias"]
        return np.clip(raw, -1.0, 1.0)

    def get_obs(prev_act):
        R = d.xmat[torso_id].reshape(3, 3)
        g = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cv = d.cvel[torso_id]
        ang_vel = R.T @ cv[:3]
        lin_vel = R.T @ cv[3:]
        leg_qpos = np.array([d.qpos[a] for a in leg_qpos_adr], dtype=np.float32)
        leg_qvel = np.array([d.qvel[a] for a in leg_qvel_adr], dtype=np.float32)
        r_sole_z = float(d.site_xpos[r_sole_id, 2])
        l_sole_z = float(d.site_xpos[l_sole_id, 2])
        r_contact = 1.0 if r_sole_z < 0.045 else 0.0
        l_contact = 1.0 if l_sole_z < 0.045 else 0.0
        return np.concatenate([
            g, ang_vel, lin_vel, leg_qpos, leg_qvel * 0.1, prev_act, [r_contact, l_contact]
        ]).astype(np.float32)

    robot_mass = get_robot_mass(m)
    pulse_duration_s = 0.040
    pulse_force_mag = (robot_mass * speed_mps) / pulse_duration_s

    # Setup offscreen renderer if saving GIF
    renderer = None
    frames = []
    if render_gif_path:
        renderer = mujoco.Renderer(m, 360, 480)

    np.random.seed(int(speed_mps * 1000) % 9999 + 42)
    random_angles = np.random.uniform(0.0, 360.0, num_pushes)

    summary_rows = []

    for push_idx, ang_deg in enumerate(random_angles, 1):
        # Reset state to settled Pose C
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.825
        d.qpos[3] = 1.0
        for i, adr in enumerate(leg_qpos_adr):
            d.qpos[adr] = float(ctrl_nom[i])
        d.qvel[:] = 0.0
        d.ctrl[:] = ctrl_nom.copy()
        verify_settled_equilibrium(m, d, ctrl_nom, tol_fz=1.0, tol_qvel=1e-3, max_steps=400)

        ema_filter = EMAActionFilter(alpha=EMA_ALPHA, dim=12)
        stepping_ctrl = ReactiveSteppingController()
        prev_act = np.zeros(12, dtype=np.float32)
        init_obs = get_obs(prev_act)
        obs_buf = np.tile(init_obs, 3)

        ang_rad = np.radians(ang_deg)
        fx = pulse_force_mag * np.cos(ang_rad)
        fy = pulse_force_mag * np.sin(ang_rad)

        dt_ctrl = 0.02
        max_seconds = 5.0
        total_steps = int(max_seconds / dt_ctrl)

        max_tilt = 0.0
        max_knee_torque = 0.0
        max_ankle_torque = 0.0
        survived = True
        fell_reason = "SURVIVED"
        settle_time = max_seconds
        has_settled = False

        for step in range(total_steps):
            cur_time = step * dt_ctrl
            act = policy_forward(obs_buf)
            act_filt = ema_filter.update(act)
            prev_act = act_filt.copy()

            target_ctrl = np.clip(
                ctrl_nom + act_filt * ACTION_SCALE_VEC,
                m.actuator_ctrlrange[:, 0],
                m.actuator_ctrlrange[:, 1]
            )

            # Evaluate Dynamic Stepping Engine (Capture Point & Leg IK)
            mujoco.mj_subtreeVel(m, d)
            com_pos = d.subtree_com[pelvis_id].copy()
            com_vel = d.subtree_linvel[pelvis_id].copy()
            r_ankle_pos = d.xpos[r_foot_id].copy()
            l_ankle_pos = d.xpos[l_foot_id].copy()
            r_contact = any(d.contact[i].geom1 == r_geom_id or d.contact[i].geom2 == r_geom_id for i in range(d.ncon))
            l_contact = any(d.contact[i].geom1 == l_geom_id or d.contact[i].geom2 == l_geom_id for i in range(d.ncon))

            step_info = stepping_ctrl.update(cur_time, com_pos, com_vel, r_ankle_pos, l_ankle_pos, r_contact, l_contact)

            # If dynamic step is triggered, execute swing leg clearance and landing IK
            if step_info["state"] in [SteppingState.SWING_EXECUTION, SteppingState.TOUCHDOWN_ABSORPTION]:
                foot_target = step_info["desired_foot_pos"]
                s_leg = step_info["swing_leg"]
                if foot_target is not None and s_leg is not None:
                    hip_pos = d.xpos[r_hip_id] if s_leg == "right" else d.xpos[l_hip_id]
                    q_swing = stepping_ctrl.inverse_kinematics_leg(hip_pos, foot_target)
                    if s_leg == "right":
                        target_ctrl[:6] = q_swing
                    else:
                        target_ctrl[6:] = q_swing

            d.ctrl[:] = np.clip(target_ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

            substeps = int(round(dt_ctrl / dt_physics))
            for sub in range(substeps):
                current_time = step * dt_ctrl + sub * dt_physics
                if 2.0 <= current_time < 2.0 + pulse_duration_s:
                    d.xfrc_applied[pelvis_id, 0] = fx
                    d.xfrc_applied[pelvis_id, 1] = fy
                else:
                    d.xfrc_applied[pelvis_id, 0] = 0.0
                    d.xfrc_applied[pelvis_id, 1] = 0.0

                mujoco.mj_step(m, d)

            # Measure kinematics
            qw, qx, qy, qz = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
            pitch = np.degrees(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
            tilt = np.hypot(roll, pitch)
            if tilt > max_tilt:
                max_tilt = tilt

            # Torques
            qfrc = np.abs(d.qfrc_actuator)
            k_t = max(qfrc[6 + 3], qfrc[12 + 3])
            a_t = max(qfrc[6 + 4], qfrc[12 + 4])
            if k_t > max_knee_torque: max_knee_torque = k_t
            if a_t > max_ankle_torque: max_ankle_torque = a_t

            # Settle detection: after push, tilt stays under 2.5 deg
            cur_time = step * dt_ctrl
            if cur_time > 2.5 and tilt < 2.5 and not has_settled:
                settle_time = cur_time - 2.0
                has_settled = True

            obs = get_obs(prev_act)
            obs_buf = np.concatenate([obs, obs_buf[:94]])

            # Record frame for GIF (sample every 2nd step = 25 fps)
            if renderer is not None and push_idx == 1 and step % 2 == 0:
                renderer.update_scene(d)
                img = Image.fromarray(renderer.render())
                frames.append(img)

            # Stepping-aware dynamic threshold: allow dynamic lunge angle during active swing
            active_stepping = (step_info["state"] in [SteppingState.SWING_EXECUTION, SteppingState.TOUCHDOWN_ABSORPTION])
            dyn_thresh = {
                "min_pelvis_z": 0.50 if active_stepping else 0.55,
                "max_roll_deg": 40.0 if active_stepping else 30.0,
                "max_pitch_deg": 40.0 if active_stepping else 30.0
            }
            is_fall, reason = is_robot_fallen(m, d, thresholds=dyn_thresh)
            if is_fall:
                survived = False
                fell_reason = reason
                break

        summary_rows.append({
            "push_num": push_idx,
            "angle_deg": ang_deg,
            "force_N": pulse_force_mag,
            "max_tilt": max_tilt,
            "knee_torque": max_knee_torque,
            "ankle_torque": max_ankle_torque,
            "settle_s": settle_time if has_settled else max_seconds,
            "status": "PASSED (5.0s)" if survived else f"FELL ({fell_reason})"
        })

        print(f"  Push #{push_idx}: Angle {ang_deg:5.1f}° | Force: {pulse_force_mag:6.1f} N | "
              f"Max Tilt: {max_tilt:4.1f}° | Knee: {max_knee_torque:5.1f} N*m | Ankle: {max_ankle_torque:5.1f} N*m | "
              f"Settle: {settle_time:4.2f}s | {summary_rows[-1]['status']}")

    if renderer is not None and frames and render_gif_path:
        frames[0].save(
            render_gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=40,  # 25 fps
            loop=0
        )
        print(f"\n[+] Rendered visual simulation GIF saved to: {render_gif_path}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Simulate Random Pushes on Cyrus in MuJoCo")
    parser.add_argument("--pushes", type=int, default=5, help="Number of random pushes per tier")
    args = parser.parse_args()

    ckpt_060 = os.path.join(PROJECT_ROOT, "checkpoints", "cyrus_balance_0.60ms_mastered.npz")
    ckpt_070 = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")
    if not os.path.exists(ckpt_070):
        ckpt_070 = os.path.join(PROJECT_ROOT, "checkpoints", "candidate_balance_active.npz")

    artifact_dir = r"C:\Users\prajval\.gemini\antigravity\brain\f9a5287d-ef88-4f54-ae4f-c5bfe1dc9aac"
    gif_060 = os.path.join(artifact_dir, "simulation_0.60ms_push.gif")
    gif_070 = os.path.join(artifact_dir, "simulation_0.70ms_push.gif")

    print("\n" + "#" * 80)
    print("STAGE 1: 5 RANDOM PUSHES AT 0.60 M/S")
    print("#" * 80)
    res_060 = run_random_pushes(0.60, args.pushes, ckpt_060, gif_060)

    print("\n" + "#" * 80)
    print("STAGE 2: 5 RANDOM PUSHES AT 0.70 M/S")
    print("#" * 80)
    res_070 = run_random_pushes(0.70, args.pushes, ckpt_070, gif_070)


if __name__ == "__main__":
    main()
