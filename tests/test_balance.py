"""
==============================================================================
CYRUS HUMANOID -- ABILITY TEST SUITE: DYNAMIC BALANCE & PUSH RECOVERY
==============================================================================
Evaluates the trained Balance Neural Network on the Cyrus Humanoid.
Can be executed in automated benchmark mode or interactive visual 3D mode.

Usage:
  python tests/test_balance.py             # Automated test suite (headless report)
  python tests/test_balance.py --visual    # Interactive 3D visual test in MuJoCo
==============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
try:
    import mujoco.viewer as mj_viewer
except ImportError:
    mj_viewer = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
CHECKPOINT_PROD = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")
CHECKPOINT_WEIGHTS = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_weights.npz")

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

class BalanceTester:
    def __init__(self, checkpoint_path=None):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        self.ctrl_nom = np.zeros(12, dtype=np.float32)
        self.ctrl_nom[3] = 0.05
        self.ctrl_nom[2] = -0.04
        self.ctrl_nom[4] = -0.01
        self.ctrl_nom[9] = 0.05
        self.ctrl_nom[8] = -0.04
        self.ctrl_nom[10] = -0.01

        self.leg_qpos_adr = [self.model.jnt(name).qposadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]
        self.leg_qvel_adr = [self.model.jnt(name).dofadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]

        self.weights = None
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, custom_path=None):
        if custom_path:
            p = Path(custom_path)
            if p.is_file():
                target = str(p)
            elif (Path(PROJECT_ROOT) / custom_path).is_file():
                target = str(Path(PROJECT_ROOT) / custom_path)
            elif (Path(PROJECT_ROOT) / "checkpoints" / custom_path).is_file():
                target = str(Path(PROJECT_ROOT) / "checkpoints" / custom_path)
            elif (Path(PROJECT_ROOT) / "checkpoints" / p.name).is_file():
                target = str(Path(PROJECT_ROOT) / "checkpoints" / p.name)
            else:
                target = os.path.join(PROJECT_ROOT, "checkpoints", custom_path)
        else:
            candidates = [CHECKPOINT_PROD, CHECKPOINT_WEIGHTS]
            existing = [c for c in candidates if os.path.exists(c)]
            if not existing:
                raise FileNotFoundError(f"No checkpoint found in {PROJECT_ROOT}/checkpoints")
            target = max(existing, key=os.path.getmtime)
        self.weights = np.load(target)
        self.loaded_file = os.path.basename(target)
        print(f"[+] Loaded Balance Neural Policy: {self.loaded_file}")

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
        r_contact = 1.0 if r_sole_z < 0.045 else 0.0
        l_contact = 1.0 if l_sole_z < 0.045 else 0.0
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

    def run_episode(self, push_mag=0.0, push_angle_deg=0.0, max_seconds=10.0, visual=False):
        self.reset()
        shove_rad = np.radians(push_angle_deg)
        step_push = (push_mag / 8.0)
        shove_vec = np.array([np.cos(shove_rad) * step_push, np.sin(shove_rad) * step_push, 0.0])

        total_steps = int(max_seconds / 0.02)
        fell = False
        fall_time = max_seconds

        viewer_handle = None
        if visual and mj_viewer is not None:
            viewer_handle = mj_viewer.launch_passive(self.model, self.data)

        for step in range(total_steps):
            t_start = time.perf_counter()
            # Apply shove at t = 2.0s (step 100 to 107) and t = 6.0s (step 300 to 307)
            if push_mag > 0.0:
                is_shove = ((100 <= step < 108) or (300 <= step < 308))
                if is_shove:
                    self.data.qvel[0] += shove_vec[0]
                    self.data.qvel[1] += shove_vec[1]

            # Policy forward pass
            act = self.policy_forward(self.obs_buf)
            act_filt = (1.0 - EMA_ALPHA) * self.prev_act + EMA_ALPHA * act
            self.prev_act = act_filt.copy()

            target_ctrl = np.clip(self.ctrl_nom + act_filt * ACTION_SCALE_VEC, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            self.data.ctrl[:] = target_ctrl

            for _ in range(4):
                mujoco.mj_step(self.model, self.data)

            obs = self.get_obs(self.prev_act)
            self.obs_buf = np.concatenate([obs, self.obs_buf[:94]])

            z = float(self.data.qpos[2])
            torso_up = float(self.data.xmat[self.torso_id, 8])

            if viewer_handle is not None:
                viewer_handle.cam.lookat[:] = self.data.qpos[:3]
                viewer_handle.sync()
                elapsed = time.perf_counter() - t_start
                if elapsed < 0.020:
                    time.sleep(0.020 - elapsed)

            if z < 0.70 or torso_up < 0.75:
                fell = True
                fall_time = step * 0.02
                break

        if viewer_handle is not None:
            viewer_handle.close()

        return not fell, fall_time

def main():
    parser = argparse.ArgumentParser(description="Cyrus Humanoid Ability Test: Dynamic Balance")
    parser.add_argument("--visual", action="store_true", help="Launch interactive 3D visual verification")
    parser.add_argument("--push", type=float, default=0.30, help="Push force magnitude to test (default: 0.30 m/s)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to test")
    args = parser.parse_args()

    print("=" * 80)
    print("CYRUS HUMANOID -- ABILITY TEST: DYNAMIC BALANCE & PUSH RECOVERY")
    print("=" * 80)

    tester = BalanceTester(checkpoint_path=args.checkpoint)

    # --- TEST 1: Quiet Stance (Zero Push) ---
    print("\n[TEST 1/2] Quiet Standing Stability (Zero External Push for 10.0s)...")
    survived, t = tester.run_episode(push_mag=0.0, max_seconds=10.0, visual=args.visual)
    status_str = f"PASSED (Stood 10.0s, Pelvis Z = {tester.data.qpos[2]:.3f}m)" if survived else f"FAILED (Fell at {t:.2f}s)"
    print(f"  --> {status_str}")

    # --- TEST 2: Directional Shoves across 8 Angles ---
    push_val = args.push
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    print(f"\n[TEST 2/2] Directional Impulse Shove Recovery ({push_val:.2f} m/s across 8 angles)...")
    results = []
    for ang in angles:
        surv, t_end = tester.run_episode(push_mag=push_val, push_angle_deg=ang, max_seconds=10.0, visual=args.visual)
        results.append(surv)
        tag = "SURVIVED (10.0s)" if surv else f"FELL at {t_end:.1f}s"
        print(f"  Angle {ang:3d}°: {tag}")

    passed_count = sum(results)
    total_count = len(results)
    success_rate = (passed_count / total_count) * 100.0

    print("\n" + "=" * 80)
    print("BALANCE ABILITY TEST REPORT")
    print("=" * 80)
    print(f"Checkpoint Tested:  {tester.loaded_file}")
    print(f"Quiet Stance:       {'100% STABLE (PASS)' if survived else 'UNSTABLE (FAIL)'}")
    print(f"Shove Force Tested: {push_val:.2f} m/s")
    print(f"Angles Survived:    {passed_count}/{total_count} ({success_rate:.1f}%)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
