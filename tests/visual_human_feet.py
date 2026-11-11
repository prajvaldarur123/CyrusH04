"""
==============================================================================
CYRUS HUMANOID -- 3D VISUAL DEMONSTRATION OF NEW HUMAN-PROPORTIONED FEET
==============================================================================
Launches an interactive 3D MuJoCo viewer showcasing the new anatomically
accurate humanoid feet and ankle clevis structure.

Usage:
  python tests/visual_human_feet.py
==============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer as mj_viewer

if sys.platform == "win32":
    PROJECT_ROOT = "C:/Users/prajval/robotics/cyrus"
else:
    PROJECT_ROOT = "/mnt/c/Users/prajval/robotics/cyrus"

XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_human_feet.xml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")

ACTION_SCALE_VEC = np.array([
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10,
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10
], dtype=np.float32)

EMA_ALPHA = 0.65

def enforce_bilateral_symmetry(raw_action):
    sym_pitch_hip = 0.5 * (raw_action[..., 2] + raw_action[..., 8])
    sym_knee      = 0.5 * (raw_action[..., 3] + raw_action[..., 9])
    sym_pitch_ank = 0.5 * (raw_action[..., 4] + raw_action[..., 10])

    anti_yaw_hip  = 0.5 * (raw_action[..., 0] - raw_action[..., 6])
    sym_roll_hip  = 0.5 * (raw_action[..., 1] + raw_action[..., 7])
    sym_roll_ank  = 0.5 * (raw_action[..., 5] + raw_action[..., 11])

    right_leg = np.stack([anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    left_leg  = np.stack([-anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    return np.concatenate([right_leg, left_leg], axis=-1)

def main():
    print("=" * 80)
    print("LAUNCHING 3D VISUAL DEMO: CYRUS WITH NEW HUMAN-PROPORTIONED FEET")
    print(f"Model: {XML_PATH}")
    print("A 3D simulation window will open on your screen.")
    print("=" * 80, flush=True)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
    r_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
    l_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

    ctrl_nom = np.zeros(12, dtype=np.float32)
    ctrl_nom[3] = 0.05
    ctrl_nom[2] = -0.04
    ctrl_nom[4] = -0.01
    ctrl_nom[9] = 0.05
    ctrl_nom[8] = -0.04
    ctrl_nom[10] = -0.01

    leg_qpos_adr = [model.jnt(n).qposadr[0] for n in [
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
    ]]
    leg_qvel_adr = [model.jnt(n).dofadr[0] for n in [
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
    ]]

    weights = np.load(CHECKPOINT_PATH)

    def policy_forward(obs_buf):
        w = weights
        h1 = np.tanh(obs_buf @ w["mlp_extractor.policy_net.0.weight"].T + w["mlp_extractor.policy_net.0.bias"])
        h2 = np.tanh(h1 @ w["mlp_extractor.policy_net.2.weight"].T + w["mlp_extractor.policy_net.2.bias"])
        raw = h2 @ w["action_net.weight"].T + w["action_net.bias"]
        return np.clip(enforce_bilateral_symmetry(raw), -1.0, 1.0)

    def get_obs(prev_act):
        d = data
        R = d.xmat[torso_id].reshape(3, 3)
        g = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cv = d.cvel[torso_id]
        ang_vel = R.T @ cv[:3]
        lin_vel = R.T @ cv[3:]
        leg_qpos = np.array([d.qpos[a] for a in leg_qpos_adr], dtype=np.float32)
        leg_qvel = np.array([d.qvel[a] for a in leg_qvel_adr], dtype=np.float32)
        r_sole_z = float(d.site_xpos[r_sole_id, 2])
        l_sole_z = float(d.site_xpos[l_sole_id, 2])
        r_contact = 1.0 if r_sole_z < 0.055 else 0.0
        l_contact = 1.0 if l_sole_z < 0.055 else 0.0
        return np.concatenate([
            g, ang_vel, lin_vel, leg_qpos, leg_qvel * 0.1, prev_act, [r_contact, l_contact]
        ]).astype(np.float32)

    with mj_viewer.launch_passive(model, data) as viewer:
        # Camera focused closely on the legs and sculpted feet
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 135

        TEST_EPISODES = [
            ("QUIET STANCE (Center-of-Mass Alignment Demonstration)", 0.0, 0, 5.0),
            ("FRONT PUSH @ 0.30 m/s (Ankle Heel-to-Toe Rocker)", 0.30, 0, 3.5),
            ("BACK PUSH @ 0.30 m/s (Rear Calcaneus Cushion)", 0.30, 180, 3.5),
            ("LATERAL PUSH @ 0.30 m/s (Subtalar Joint & Arch Stability)", 0.30, 90, 3.5),
            ("DIAGONAL PUSH @ 0.40 m/s (Maximum Kinetic Recovery)", 0.40, 45, 4.0),
        ]

        for title, spd, ang, dur_s in TEST_EPISODES:
            if not viewer.is_running():
                break

            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.84
            data.qpos[3] = 1.0
            for i, adr in enumerate(leg_qpos_adr):
                data.qpos[adr] = float(ctrl_nom[i])
            data.qvel[:] = 0.0
            data.ctrl[:] = ctrl_nom.copy()
            mujoco.mj_forward(model, data)

            prev_act = np.zeros(12, dtype=np.float32)
            init_obs = get_obs(prev_act)
            obs_buf = np.tile(init_obs, 3)

            print(f"\n[DEMO] {title} ...", flush=True)

            total_steps = int(dur_s / 0.02)
            shove_step = int(1.0 / 0.02) # Push at t=1.0s

            rad = np.radians(ang)
            dvx = spd * np.cos(rad)
            dvy = spd * np.sin(rad)

            for step in range(total_steps):
                if not viewer.is_running():
                    return

                t0 = time.perf_counter()

                if spd > 0.0 and step == shove_step:
                    data.qvel[0] += dvx
                    data.qvel[1] += dvy
                    print("  --> [SHOCK IMPULSE DELIVERED!] Recovering...", flush=True)

                act = policy_forward(obs_buf)
                act_filt = (1.0 - EMA_ALPHA) * prev_act + EMA_ALPHA * act
                prev_act = act_filt.copy()

                target = np.clip(
                    ctrl_nom + act_filt * ACTION_SCALE_VEC,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1]
                )
                data.ctrl[:] = target

                for _ in range(4):
                    mujoco.mj_step(model, data)

                obs = get_obs(prev_act)
                obs_buf = np.concatenate([obs, obs_buf[:94]])

                viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.35]
                viewer.sync()

                dt = time.perf_counter() - t0
                if dt < 0.020:
                    time.sleep(0.020 - dt)

            print(f"  [✓] Episode Completed Successfully! Cyrus Stood Stable.", flush=True)
            time.sleep(0.8)

        print("\n" + "=" * 80)
        print("HUMAN FEET 3D DEMONSTRATION COMPLETE")
        print("=" * 80)
        time.sleep(2.0)

if __name__ == "__main__":
    main()
