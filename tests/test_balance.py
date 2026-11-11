"""
==============================================================================
CYRUS HUMANOID -- ABILITY TEST SUITE: DYNAMIC BALANCE & PUSH RECOVERY
==============================================================================
Evaluates the trained Balance Policy on the Cyrus Humanoid using the centralized
balance pipeline (cryptographic manifest verification, settled start, standardized metrics).

Usage:
  python tests/test_balance.py                      # Automated test suite (headless report)
  python tests/test_balance.py --visual             # Interactive 3D visual test in MuJoCo
  python tests/test_balance.py --pose pose_c        # Test with Pose C (Athletic Squat)
  python tests/test_balance.py --protocol pulse     # Calibrated force pulse (default)
  python tests/test_balance.py --protocol jump      # Velocity jump (legacy)
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
ROBOTICS_ROOT = os.path.dirname(PROJECT_ROOT)
for p in [ROBOTICS_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from controller.balance_pipeline import (
    load_verified_checkpoint,
    enforce_bilateral_symmetry,
    EMAActionFilter,
    get_robot_mass,
    get_robot_com,
    measure_ground_wrench,
    verify_settled_equilibrium,
    is_robot_fallen,
    POSE_A,
    POSE_C,
    ROBUST_ROLL_GAINS,
    ACTION_SCALE_VEC,
    EMA_ALPHA
)

XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "manifest.json")


class BalanceTester:
    def __init__(self, checkpoint_path=None, pose_name="pose_c", protocol="pulse", dt_physics=0.0025):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.dt_physics = float(dt_physics)
        self.model.opt.timestep = self.dt_physics
        self.data = mujoco.MjData(self.model)

        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.r_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        self.pose_name = pose_name.lower()
        self.protocol = protocol.lower()

        if self.pose_name == "pose_c":
            self.ctrl_nom = POSE_C.copy()
            self.init_z = 0.825
            # Configure robust roll gains
            self._apply_robust_gains()
        else:
            self.ctrl_nom = POSE_A.copy()
            self.init_z = 0.840

        self.leg_qpos_adr = [self.model.jnt(name).qposadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]
        self.leg_qvel_adr = [self.model.jnt(name).dofadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]

        self.weights = None
        self.loaded_file = None
        self.load_checkpoint(checkpoint_path)
        self.ema_filter = EMAActionFilter(alpha=EMA_ALPHA, dim=12)

    def _apply_robust_gains(self):
        for side in ["right", "left"]:
            h_act = self.model.actuator(f"{side}_hip_roll_act")
            h_act.gainprm[0] = ROBUST_ROLL_GAINS["hip_roll_kp"]
            h_act.biasprm[1] = -ROBUST_ROLL_GAINS["hip_roll_kp"]
            h_act.biasprm[2] = -ROBUST_ROLL_GAINS["hip_roll_kv"]

            a_act = self.model.actuator(f"{side}_ankle_roll_act")
            a_act.gainprm[0] = ROBUST_ROLL_GAINS["ankle_roll_kp"]
            a_act.biasprm[1] = -ROBUST_ROLL_GAINS["ankle_roll_kp"]
            a_act.biasprm[2] = -ROBUST_ROLL_GAINS["ankle_roll_kv"]

    def load_checkpoint(self, custom_path=None):
        if custom_path:
            p = Path(custom_path)
            target = str(p if p.is_file() else Path(PROJECT_ROOT) / custom_path)
            self.weights = np.load(target)
            self.loaded_file = target
            print(f"[+] Loaded Policy Checkpoint (Manual): {os.path.basename(target)}")
        else:
            data, entry = load_verified_checkpoint(MANIFEST_PATH, "active_production")
            self.weights = data
            self.loaded_file = entry["file"]
            print(f"[+] Loaded Policy Checkpoint (Verified Manifest): {entry['file']} (SHA-256: {entry['sha256'][:16]}...)")

    def policy_forward(self, obs_buf):
        w = self.weights
        h1 = np.tanh(obs_buf @ w["mlp_extractor.policy_net.0.weight"].T + w["mlp_extractor.policy_net.0.bias"])
        h2 = np.tanh(h1 @ w["mlp_extractor.policy_net.2.weight"].T + w["mlp_extractor.policy_net.2.bias"])
        raw = h2 @ w["action_net.weight"].T + w["action_net.bias"]
        return np.clip(raw, -1.0, 1.0)

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
        self.data.qpos[2] = self.init_z
        self.data.qpos[3] = 1.0
        for i, adr in enumerate(self.leg_qpos_adr):
            self.data.qpos[adr] = float(self.ctrl_nom[i])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ctrl_nom.copy()

        # Step until physical equilibrium is verified
        verify_settled_equilibrium(self.model, self.data, self.ctrl_nom, tol_fz=1.0, tol_qvel=1e-3, max_steps=400)

        self.ema_filter.reset()
        self.prev_act = np.zeros(12, dtype=np.float32)
        init_obs = self.get_obs(self.prev_act)
        self.obs_buf = np.tile(init_obs, 3)

    def run_episode(self, push_mag=0.0, push_angle_deg=0.0, max_seconds=10.0, visual=False):
        self.reset()
        shove_rad = np.radians(push_angle_deg)
        dt_ctrl = 0.02
        total_steps = int(max_seconds / dt_ctrl)
        fell = False
        fall_time = max_seconds
        fall_reason = "STABLE"

        # Protocol parameters
        robot_mass = get_robot_mass(self.model)
        pulse_duration_s = 0.040  # 40 ms
        pulse_force_mag = (robot_mass * push_mag) / pulse_duration_s if pulse_duration_s > 0 else 0.0
        pulse_fx = pulse_force_mag * np.cos(shove_rad)
        pulse_fy = pulse_force_mag * np.sin(shove_rad)

        viewer_handle = None
        if visual and mj_viewer is not None:
            viewer_handle = mj_viewer.launch_passive(self.model, self.data)

        # Disturbance timing: t = 2.0s
        push_step_start = int(2.0 / dt_ctrl)
        push_substeps_total = int(pulse_duration_s / 0.005)  # 8 substeps

        substep_counter = 0

        for step in range(total_steps):
            t_start = time.perf_counter()

            # Policy forward pass (at 50 Hz)
            act = self.policy_forward(self.obs_buf)
            act_filt = self.ema_filter.update(act)
            self.prev_act = act_filt.copy()

            target_ctrl = np.clip(
                self.ctrl_nom + act_filt * ACTION_SCALE_VEC,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1]
            )
            self.data.ctrl[:] = target_ctrl

            # Physics simulation substeps (e.g. 8 substeps of 0.0025s = 0.02s)
            substeps = int(round(dt_ctrl / self.dt_physics))
            for sub in range(substeps):
                current_time = step * dt_ctrl + sub * self.dt_physics

                # Apply disturbance
                if push_mag > 0.0 and (2.0 <= current_time < 2.0 + pulse_duration_s):
                    if self.protocol == "pulse":
                        self.data.xfrc_applied[self.pelvis_id, 0] = pulse_fx
                        self.data.xfrc_applied[self.pelvis_id, 1] = pulse_fy
                    elif self.protocol == "jump":
                        # Legacy velocity jump: inject on first sub-step
                        if current_time == 2.0:
                            self.data.qvel[0] += push_mag * np.cos(shove_rad)
                            self.data.qvel[1] += push_mag * np.sin(shove_rad)
                else:
                    self.data.xfrc_applied[self.pelvis_id, 0] = 0.0
                    self.data.xfrc_applied[self.pelvis_id, 1] = 0.0

                mujoco.mj_step(self.model, self.data)

            obs = self.get_obs(self.prev_act)
            self.obs_buf = np.concatenate([obs, self.obs_buf[:94]])

            if viewer_handle is not None:
                viewer_handle.sync()
                elapsed = time.perf_counter() - t_start
                if elapsed < dt_ctrl:
                    time.sleep(dt_ctrl - elapsed)

            # Standardized Fall Detection
            is_fall, reason = is_robot_fallen(self.model, self.data)
            if is_fall:
                fell = True
                fall_time = step * dt_ctrl
                fall_reason = reason
                break

        if viewer_handle is not None:
            viewer_handle.close()

        return not fell, fall_time, fall_reason


def main():
    parser = argparse.ArgumentParser(description="Cyrus Humanoid Ability Test: Dynamic Balance & Push Recovery")
    parser.add_argument("--visual", action="store_true", help="Launch interactive 3D visual verification")
    parser.add_argument("--push", type=float, default=0.30, help="Push speed target in m/s (default: 0.30 m/s)")
    parser.add_argument("--pose", type=str, default="pose_c", choices=["pose_a", "pose_c"], help="Nominal posture")
    parser.add_argument("--protocol", type=str, default="pulse", choices=["pulse", "jump"], help="Push injection protocol")
    parser.add_argument("--dt", type=float, default=0.0025, help="Physics simulation timestep (default: 0.0025s = 2.5ms)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to test")
    args = parser.parse_args()

    print("=" * 80)
    print("CYRUS HUMANOID -- ABILITY TEST: DYNAMIC BALANCE & PUSH RECOVERY")
    print("=" * 80)

    tester = BalanceTester(
        checkpoint_path=args.checkpoint,
        pose_name=args.pose,
        protocol=args.protocol,
        dt_physics=args.dt
    )

    # --- TEST 1: Quiet Stance (Zero Push) ---
    print(f"\n[TEST 1/2] Quiet Standing Stability (Zero External Push for 10.0s, Pose: {args.pose.upper()})...")
    survived, t, reason = tester.run_episode(push_mag=0.0, max_seconds=10.0, visual=args.visual)
    status_str = f"PASSED (Stood 10.0s, Pelvis Z = {tester.data.qpos[2]:.3f}m)" if survived else f"FAILED ({reason} at {t:.2f}s)"
    print(f"  --> {status_str}")

    # --- TEST 2: Directional Shoves across 8 Angles ---
    push_val = args.push
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    print(f"\n[TEST 2/2] Directional Push Recovery ({push_val:.2f} m/s, Protocol: {args.protocol.upper()} across 8 angles)...")
    results = []
    for ang in angles:
        surv, t_end, r_reason = tester.run_episode(push_mag=push_val, push_angle_deg=ang, max_seconds=10.0, visual=args.visual)
        results.append(surv)
        tag = "SURVIVED (10.0s)" if surv else f"FELL at {t_end:.2f}s ({r_reason})"
        print(f"  Angle {ang:3d}°: {tag}")

    passed_count = sum(results)
    total_count = len(results)
    success_rate = (passed_count / total_count) * 100.0

    print("\n" + "=" * 80)
    print("BALANCE ABILITY TEST REPORT")
    print("=" * 80)
    print(f"Checkpoint Tested:  {tester.loaded_file}")
    print(f"Posture Mode:       {args.pose.upper()}")
    print(f"Push Protocol:      {args.protocol.upper()}")
    print(f"Target Velocity:    {push_val:.2f} m/s")
    print(f"Angles Survived:    {passed_count}/{total_count} ({success_rate:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
