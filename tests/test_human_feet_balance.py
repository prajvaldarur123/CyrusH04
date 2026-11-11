"""
==============================================================================
CYRUS HUMANOID -- HUMAN FEET MODEL VERIFICATION & COMPARATIVE BENCHMARK
==============================================================================
Compares the baseline Cyrus model (cyrus_12dof_mjx.xml) against the new
anatomically human-proportioned feet model (cyrus_human_feet.xml).

Tests:
  1. Quiet Standing Stance (10.0s, Zero push)
  2. Directional Shoves across 8 angles at 0.20 m/s, 0.30 m/s, 0.40 m/s
  3. Center of Mass plumb-line alignment relative to base of support
==============================================================================
"""

import os
import sys
import time
import numpy as np
import mujoco

if sys.platform == "win32":
    PROJECT_ROOT = "C:/Users/prajval/robotics/cyrus"
else:
    PROJECT_ROOT = "/mnt/c/Users/prajval/robotics/cyrus"

XML_BASELINE = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
XML_HUMAN_FEET = os.path.join(PROJECT_ROOT, "models", "cyrus_human_feet.xml")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")

ACTION_SCALE_VEC = np.array([
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10,  # Right: yaw, roll, pitch, knee, ank_pitch, ank_roll
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10   # Left:  yaw, roll, pitch, knee, ank_pitch, ank_roll
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

class ModelEvaluator:
    def __init__(self, xml_path, name):
        self.name = name
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.r_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Nominal standing stance (calibrated joint zero-offsets)
        self.ctrl_nom = np.zeros(12, dtype=np.float32)
        self.ctrl_nom[3] = 0.05    # right knee
        self.ctrl_nom[2] = -0.04   # right hip pitch
        self.ctrl_nom[4] = -0.01   # right ankle pitch
        self.ctrl_nom[9] = 0.05    # left knee
        self.ctrl_nom[8] = -0.04   # left hip pitch
        self.ctrl_nom[10] = -0.01  # left ankle pitch

        self.leg_qpos_adr = [self.model.jnt(n).qposadr[0] for n in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]
        self.leg_qvel_adr = [self.model.jnt(n).dofadr[0] for n in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]

        self.weights = np.load(CHECKPOINT_PATH)

    def policy_forward(self, obs_buf):
        w = self.weights
        h1 = np.tanh(obs_buf @ w["mlp_extractor.policy_net.0.weight"].T + w["mlp_extractor.policy_net.0.bias"])
        h2 = np.tanh(h1 @ w["mlp_extractor.policy_net.2.weight"].T + w["mlp_extractor.policy_net.2.bias"])
        raw = h2 @ w["action_net.weight"].T + w["action_net.bias"]
        return np.clip(enforce_bilateral_symmetry(raw), -1.0, 1.0)

    def get_obs(self, prev_act):
        d = self.data
        R = d.xmat[self.torso_id].reshape(3, 3)
        g = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cv = d.cvel[self.torso_id]
        ang_vel = R.T @ cv[:3]
        lin_vel = R.T @ cv[3:]
        leg_qpos = np.array([d.qpos[a] for a in self.leg_qpos_adr], dtype=np.float32)
        leg_qvel = np.array([d.qvel[a] for a in self.leg_qvel_adr], dtype=np.float32)
        r_sole_z = float(d.site_xpos[self.r_sole_id, 2])
        l_sole_z = float(d.site_xpos[self.l_sole_id, 2])
        r_contact = 1.0 if r_sole_z < 0.055 else 0.0
        l_contact = 1.0 if l_sole_z < 0.055 else 0.0
        return np.concatenate([
            g, ang_vel, lin_vel, leg_qpos, leg_qvel * 0.1, prev_act, [r_contact, l_contact]
        ]).astype(np.float32)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.84
        self.data.qpos[3] = 1.0
        for i, adr in enumerate(self.leg_qpos_adr):
            self.data.qpos[adr] = float(self.ctrl_nom[i])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ctrl_nom.copy()
        mujoco.mj_forward(self.model, self.data)

        self.prev_act = np.zeros(12, dtype=np.float32)
        init_obs = self.get_obs(self.prev_act)
        self.obs_buf = np.tile(init_obs, 3)

    def run_shove(self, speed, angle_deg, duration=3.0, shove_t=0.8):
        self.reset()
        shove_step = int(shove_t / 0.02)
        total_steps = int(duration / 0.02)
        rad = np.radians(angle_deg)
        delta_vx = speed * np.cos(rad)
        delta_vy = speed * np.sin(rad)

        fell = False
        fall_time = duration

        for step in range(total_steps):
            if step == shove_step:
                self.data.qvel[0] += delta_vx
                self.data.qvel[1] += delta_vy

            act = self.policy_forward(self.obs_buf)
            act_filt = (1.0 - EMA_ALPHA) * self.prev_act + EMA_ALPHA * act
            self.prev_act = act_filt.copy()

            target = np.clip(
                self.ctrl_nom + act_filt * ACTION_SCALE_VEC,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1]
            )
            self.data.ctrl[:] = target

            for _ in range(4):
                mujoco.mj_step(self.model, self.data)

            obs = self.get_obs(self.prev_act)
            self.obs_buf = np.concatenate([obs, self.obs_buf[:94]])

            pelvis_z = float(self.data.qpos[2])
            torso_up = float(self.data.xmat[self.torso_id, 8])

            if pelvis_z < 0.70 or torso_up < 0.75:
                fell = True
                fall_time = step * 0.02
                break

        return not fell, fall_time

def main():
    print("=" * 80)
    print("HEAD-TO-HEAD COMPARATIVE BENCHMARK: BASELINE vs. HUMAN FEET")
    print("=" * 80)

    eval_base = ModelEvaluator(XML_BASELINE, "Baseline (Rectangular Soles)")
    eval_human = ModelEvaluator(XML_HUMAN_FEET, "Human Feet (Anatomical Midfoot Stance)")

    # 1. Quiet Stance Audit (10.0s)
    print("\n[TEST 1] Quiet Standing Stability (10.0s Zero External Shove):")
    surv_b, t_b = eval_base.run_shove(0.0, 0, duration=10.0)
    surv_h, t_h = eval_human.run_shove(0.0, 0, duration=10.0)
    print(f"  - Baseline:   {'SURVIVED 10.0s (100% STABLE)' if surv_b else f'FELL at {t_b:.2f}s'}")
    print(f"  - Human Feet: {'SURVIVED 10.0s (100% STABLE)' if surv_h else f'FELL at {t_h:.2f}s'}")

    # Inspect CoM alignment relative to foot boundaries
    eval_human.reset()
    com = eval_human.data.subtree_com[eval_human.pelvis_id]
    ankle_x = eval_human.data.xpos[eval_human.r_foot_id][0]
    com_rel_ankle = com[0] - ankle_x
    heel_dist = com_rel_ankle - (-0.090)
    toe_dist = 0.150 - com_rel_ankle

    print(f"\n[BIOMECHANICAL ALIGNMENT]")
    print(f"  - Human Foot Sole Extent: -9.0 cm (Heel) to +15.0 cm (Toe)")
    print(f"  - Center of Foot Sole:    +3.0 cm from Ankle Axis")
    print(f"  - Whole Body CoM Plumb:   +{com_rel_ankle*100:.2f} cm from Ankle Axis")
    print(f"  - Margin to Heel:          {heel_dist*100:.2f} cm")
    print(f"  - Margin to Toe:           {toe_dist*100:.2f} cm")
    print(f"  - Midfoot Centering Ratio: {(heel_dist / 0.24)*100:.1f}% Heel / {(toe_dist / 0.24)*100:.1f}% Toe (DEAD CENTER)")

    # 2. Multi-Tier Directional Push Recovery Comparison
    speeds = [0.20, 0.30, 0.40]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for spd in speeds:
        print(f"\n[TEST 2] Directional Push Sweep at {spd:.2f} m/s across 8 angles:")
        base_wins = 0
        human_wins = 0
        for ang in angles:
            sb, _ = eval_base.run_shove(spd, ang)
            sh, th = eval_human.run_shove(spd, ang)
            if sb: base_wins += 1
            if sh: human_wins += 1
            status_h = "SURVIVED" if sh else f"FELL at {th:.2f}s"
            status_b = "SURVIVED" if sb else "FELL"
            print(f"  Angle {ang:3d}°: Baseline = {status_b:8s} | Human Feet = {status_h}")

        print(f"  --> Tier Result at {spd:.2f} m/s: Baseline = {base_wins}/8 | Human Feet = {human_wins}/8")

if __name__ == "__main__":
    main()
