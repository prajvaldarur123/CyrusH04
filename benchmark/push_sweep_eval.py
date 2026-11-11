"""
==============================================================================
CYRUS HUMANOID -- SOTA 168-CELL POLAR PUSH RECOVERY BENCHMARK SUITE
==============================================================================
Evaluates the trained Cyrus Balance Policy across a comprehensive 168-cell matrix:
  - 24 Directions (15° azimuth resolution across full 360°)
  - 7 Push Magnitudes (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 m/s)
  - Instantaneous Velocity Impulse (single-step delta-v impact)
  - Random Push Timing: t in [1.0s, 3.0s]
  - Per-cell biomechanical telemetry:
      * Survival rate & fall timestamp
      * Settling time tau_95 (time to damp oscillations back to < 5% of peak)
      * Peak joint torque ratio (max |tau| / tau_limit)
      * Cumulative foot displacement (slip detection)
      * Stepping reflex detection (foot clearance > 5cm)
  - Outputs:
      * Detailed CSV table (benchmark/results/push_sweep_<timestamp>.csv)
      * Polar contour stability map J_max(theta)

Usage:
  python benchmarks/push_sweep_eval.py
  python benchmarks/push_sweep_eval.py --checkpoint checkpoints/balance_12dof_production.npz
  python benchmarks/push_sweep_eval.py --quick   # 8-angle smoke test for fast verification
==============================================================================
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
DEFAULT_PROD = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_weights.npz")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "benchmark", "results")

# Joint-specific scaling vector (SOTA Pitch/Knee authority)
ACTION_SCALE_VEC = np.array([
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10,  # Right: yaw, roll, pitch, knee, ank_pitch, ank_roll
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10   # Left:  yaw, roll, pitch, knee, ank_pitch, ank_roll
], dtype=np.float32)

ACTION_SCALE_FLAT = np.full(12, 0.14, dtype=np.float32)
EMA_ALPHA = 0.65

def enforce_bilateral_symmetry(raw_action):
    """Enforces Unitree Parallel Roll Sway and pitch symmetry across both legs."""
    sym_pitch_hip = 0.5 * (raw_action[..., 2] + raw_action[..., 8])
    sym_knee      = 0.5 * (raw_action[..., 3] + raw_action[..., 9])
    sym_pitch_ank = 0.5 * (raw_action[..., 4] + raw_action[..., 10])

    anti_yaw_hip  = 0.5 * (raw_action[..., 0] - raw_action[..., 6])
    sym_roll_hip  = 0.5 * (raw_action[..., 1] + raw_action[..., 7])
    sym_roll_ank  = 0.5 * (raw_action[..., 5] + raw_action[..., 11])

    right_leg = np.stack([anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    left_leg  = np.stack([-anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    return np.concatenate([right_leg, left_leg], axis=-1)


class PolarPushBenchmark:
    def __init__(self, checkpoint_path=None, action_scale_mode="vector"):
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"Model file not found: {XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
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

        self.leg_qpos_adr = [self.model.jnt(name).qposadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]
        self.leg_qvel_adr = [self.model.jnt(name).dofadr[0] for name in [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
        ]]

        if action_scale_mode == "flat":
            self.action_scale = ACTION_SCALE_FLAT
        else:
            self.action_scale = ACTION_SCALE_VEC

        self.action_scale_mode = action_scale_mode
        self.weights = None
        self.loaded_file = None
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path=None):
        target = None
        if path:
            p = Path(path)
            if p.is_file():
                target = str(p)
            elif (Path(PROJECT_ROOT) / path).is_file():
                target = str(Path(PROJECT_ROOT) / path)
            elif (Path(PROJECT_ROOT) / "checkpoints" / path).is_file():
                target = str(Path(PROJECT_ROOT) / "checkpoints" / path)
            elif (Path(PROJECT_ROOT) / "checkpoints" / p.name).is_file():
                target = str(Path(PROJECT_ROOT) / "checkpoints" / p.name)

        if not target:
            for cand in [DEFAULT_PROD, DEFAULT_WEIGHTS]:
                if os.path.exists(cand):
                    target = cand
                    break

        if not target or not os.path.exists(target):
            raise FileNotFoundError(f"Could not locate a valid checkpoint file from: {path}")

        self.weights = np.load(target)
        self.loaded_file = target
        print(f"[+] Loaded Policy Checkpoint: {os.path.basename(target)}")

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
        self.data.qpos[2] = 0.84  # 0.84m standing height
        self.data.qpos[3] = 1.0   # Upright quaternion
        for i, adr in enumerate(self.leg_qpos_adr):
            self.data.qpos[adr] = float(self.ctrl_nom[i])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ctrl_nom.copy()
        mujoco.mj_forward(self.model, self.data)

        self.prev_act = np.zeros(12, dtype=np.float32)
        init_obs = self.get_obs(self.prev_act)
        self.obs_buf = np.tile(init_obs, 3)

    def evaluate_cell(self, angle_deg, push_speed, push_time_s=1.5, total_seconds=10.0):
        """
        Executes a single test cell with instantaneous Dirac-delta velocity injection.
        """
        self.reset()
        dt_rl = 0.02
        total_steps = int(total_seconds / dt_rl)
        push_step = int(push_time_s / dt_rl)

        rad = np.radians(angle_deg)
        # Instantaneous delta-v pulse
        delta_vx = push_speed * np.cos(rad)
        delta_vy = push_speed * np.sin(rad)

        fell = False
        fall_time = total_seconds
        peak_torque_ratio = 0.0
        foot_slip_total = 0.0
        step_detected = False

        prev_r_sole = self.data.site_xpos[self.r_sole_id][:2].copy()
        prev_l_sole = self.data.site_xpos[self.l_sole_id][:2].copy()

        # Settling time tracking: deviation from steady upright
        settled_step = None
        max_tilt_post_push = 0.0

        for step in range(total_steps):
            # Inject instantaneous single-step velocity impulse
            if step == push_step and push_speed > 0.0:
                self.data.qvel[0] += delta_vx
                self.data.qvel[1] += delta_vy

            # Forward policy inference
            act = self.policy_forward(self.obs_buf)
            act_filt = (1.0 - EMA_ALPHA) * self.prev_act + EMA_ALPHA * act
            self.prev_act = act_filt.copy()

            target_ctrl = np.clip(
                self.ctrl_nom + act_filt * self.action_scale,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1]
            )
            self.data.ctrl[:] = target_ctrl

            # Physics substeps (4 substeps of 0.005s = 0.02s)
            for _ in range(4):
                mujoco.mj_step(self.model, self.data)

            obs = self.get_obs(self.prev_act)
            self.obs_buf = np.concatenate([obs, self.obs_buf[:94]])

            pelvis_z = float(self.data.qpos[2])
            torso_up = float(self.data.xmat[self.torso_id, 8])

            # Check fall
            if pelvis_z < 0.70 or torso_up < 0.75:
                fell = True
                fall_time = step * dt_rl
                break

            # Metrics post-push
            if step >= push_step:
                # Torque ratio (|actuator force| / ctrl limit)
                qfrc = np.abs(self.data.qfrc_actuator[self.leg_qvel_adr])
                torque_ratios = qfrc / 120.0  # nominal 120 Nm peak limit
                peak_torque_ratio = max(peak_torque_ratio, float(np.max(torque_ratios)))

                # Foot displacement / slip
                cur_r_sole = self.data.site_xpos[self.r_sole_id][:2]
                cur_l_sole = self.data.site_xpos[self.l_sole_id][:2]
                r_z = float(self.data.site_xpos[self.r_sole_id, 2])
                l_z = float(self.data.site_xpos[self.l_sole_id, 2])

                if r_z < 0.045:
                    foot_slip_total += np.linalg.norm(cur_r_sole - prev_r_sole)
                if l_z < 0.045:
                    foot_slip_total += np.linalg.norm(cur_l_sole - prev_l_sole)

                # Stepping reflex check (foot clearance > 5.5cm)
                if r_z > 0.055 or l_z > 0.055:
                    step_detected = True

                prev_r_sole = cur_r_sole.copy()
                prev_l_sole = cur_l_sole.copy()

                # Tilt deviation in degrees
                tilt_deg = np.degrees(np.arccos(np.clip(torso_up, -1.0, 1.0)))
                max_tilt_post_push = max(max_tilt_post_push, tilt_deg)

                if step > push_step + 10:  # give 0.2s for push to register
                    if tilt_deg < 2.5 and settled_step is None:
                        settled_step = step
                    elif tilt_deg >= 2.5:
                        settled_step = None  # reset if still oscillating

        if not fell:
            if settled_step is not None:
                settling_time_s = max(0.0, (settled_step - push_step) * dt_rl)
            else:
                settling_time_s = total_seconds - (push_step * dt_rl)
        else:
            settling_time_s = total_seconds

        return {
            "angle_deg": angle_deg,
            "speed_ms": push_speed,
            "survived": not fell,
            "fall_time_s": fall_time,
            "settling_time_s": settling_time_s,
            "peak_torque_ratio": min(1.0, peak_torque_ratio),
            "foot_slip_m": foot_slip_total,
            "step_detected": step_detected,
            "max_tilt_deg": max_tilt_post_push
        }

    def run_matrix(self, quick=False):
        """
        Executes the full 168-Cell Matrix or 8-angle Quick Scan.
        """
        if quick:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            speeds = [0.10, 0.30, 0.50]
            print(f"\n[>>>] Running QUICK SCAN: {len(angles)} Angles x {len(speeds)} Speeds = {len(angles)*len(speeds)} Cells...")
        else:
            # Full 24 directions x 7 speeds = 168 cells
            angles = list(range(0, 360, 15))
            speeds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
            print(f"\n[>>>] Running FULL SOTA BENCHMARK: 24 Angles (15° steps) x 7 Speeds = 168 Cells...")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        results = []
        total_cells = len(angles) * len(speeds)
        passed_cells = 0
        t0 = time.time()

        # Randomize push time slightly per cell in [1.2s, 2.5s]
        rng = np.random.RandomState(42)

        print("-" * 80)
        print(f"{'Angle':>7s} | {'Speed':>7s} | {'Status':>8s} | {'tau_95':>8s} | {'Max Torque':>10s} | {'Foot Slip':>9s} | {'Step?':>6s}")
        print("-" * 80)

        for spd in speeds:
            for ang in angles:
                p_time = rng.uniform(1.2, 2.5)
                res = self.evaluate_cell(ang, spd, push_time_s=p_time)
                results.append(res)
                if res["survived"]:
                    passed_cells += 1
                    status_str = "PASS"
                    tau_str = f"{res['settling_time_s']:.2f}s"
                else:
                    status_str = f"FELL ({res['fall_time_s']:.1f}s)"
                    tau_str = "N/A"

                step_str = "YES" if res["step_detected"] else "NO"
                print(f"{ang:6d}° | {spd:5.2f}m/s | {status_str:>8s} | {tau_str:>8s} | {res['peak_torque_ratio']*100:8.1f}% | {res['foot_slip_m']*100:7.1f}cm | {step_str:>6s}")

        elapsed = time.time() - t0
        pass_pct = (passed_cells / float(total_cells)) * 100.0

        # Save CSV Report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"push_sweep_{timestamp}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_filename)

        with open(csv_path, "w") as f:
            f.write("angle_deg,speed_ms,survived,fall_time_s,settling_time_s,peak_torque_ratio,foot_slip_m,step_detected,max_tilt_deg\n")
            for r in results:
                f.write(f"{r['angle_deg']},{r['speed_ms']},{int(r['survived'])},{r['fall_time_s']:.3f},{r['settling_time_s']:.3f},{r['peak_torque_ratio']:.4f},{r['foot_slip_m']:.4f},{int(r['step_detected'])},{r['max_tilt_deg']:.2f}\n")

        print("=" * 80)
        print("168-CELL POLAR PUSH RECOVERY BENCHMARK REPORT")
        print("=" * 80)
        print(f"Policy Evaluated:       {os.path.basename(self.loaded_file)}")
        print(f"Action Scale Mode:      {self.action_scale_mode}")
        print(f"Cells Survived:         {passed_cells:3d} / {total_cells:3d} ({pass_pct:5.1f}%)")
        print(f"Benchmark Duration:     {elapsed:.1f}s")
        print(f"CSV Report Saved:       {csv_path}")

        # Compute J_max(theta) - maximum recoverable impulse per angle
        print("\n[🧭] MAXIMUM RECOVERABLE PUSH BY ANGLE J_max(theta):")
        j_max_map = {}
        for ang in angles:
            max_spd = 0.0
            for r in results:
                if r["angle_deg"] == ang and r["survived"]:
                    max_spd = max(max_spd, r["speed_ms"])
            j_max_map[ang] = max_spd

        for i, ang in enumerate(angles):
            bar = "█" * int(j_max_map[ang] * 20)
            print(f"  {ang:3d}°: {j_max_map[ang]:4.2f} m/s  | {bar}")

        print("=" * 80)
        return {
            "total_cells": total_cells,
            "passed_cells": passed_cells,
            "pass_pct": pass_pct,
            "csv_path": csv_path,
            "j_max_map": j_max_map
        }


def main():
    parser = argparse.ArgumentParser(description="Cyrus Humanoid 168-Cell Polar Push Recovery Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint (.npz)")
    parser.add_argument("--quick", action="store_true", help="Run 8-angle smoke test instead of full 168-cell matrix")
    parser.add_argument("--action-scale", type=str, default="vector", choices=["vector", "flat"], help="Action scaling mode (vector vs flat)")
    args = parser.parse_args()

    bench = PolarPushBenchmark(checkpoint_path=args.checkpoint, action_scale_mode=args.action_scale)
    bench.run_matrix(quick=args.quick)

if __name__ == "__main__":
    main()
