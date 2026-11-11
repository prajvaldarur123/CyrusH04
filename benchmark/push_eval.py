import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cyrus.envs.standup_env import CyrusStandupEnv
from cyrus.controller.balance import CyrusBalanceController
from cyrus.benchmark.metrics import compute_capture_point, compute_mechanical_work, compute_mechanical_effort


def evaluate_push_recovery(
    policy=None,
    use_controller: bool = False,
    force_levels: list = None,
    push_duration_s: float = 0.15,  # 150 ms impulse duration
    is_walk_policy: bool = False
) -> dict:
    """
    Evaluates Push Recovery across an 8-direction disturbance grid:
      Forward (0 deg), Backward (180 deg), Left (90 deg), Right (270 deg),
      Front-Left (45 deg), Front-Right (315 deg), Back-Left (135 deg), Back-Right (225 deg).

    CRITICAL PHYSICAL CONSTRAINTS:
      - Force (F in N), Duration (dt in s), and Impulse (J in N*s) are recorded separately.
      - Velocity change (delta_v = J / m) is recorded separately and never confused with impulse.
      - Strategy classification: Ankle, Hip, Arm, Stepping.
    """
    if force_levels is None:
        force_levels = [40.0, 80.0, 120.0, 160.0, 200.0]

    DIRECTIONS = [
        {"name": "Forward",     "angle_deg": 0.0,   "dir": np.array([+1.0, 0.0])},
        {"name": "Backward",    "angle_deg": 180.0, "dir": np.array([-1.0, 0.0])},
        {"name": "Left",        "angle_deg": 90.0,  "dir": np.array([0.0, +1.0])},
        {"name": "Right",       "angle_deg": 270.0, "dir": np.array([0.0, -1.0])},
        {"name": "Front-Left",  "angle_deg": 45.0,  "dir": np.array([+0.7071, +0.7071])},
        {"name": "Front-Right", "angle_deg": 315.0, "dir": np.array([+0.7071, -0.7071])},
        {"name": "Back-Left",   "angle_deg": 135.0, "dir": np.array([-0.7071, +0.7071])},
        {"name": "Back-Right",  "angle_deg": 225.0, "dir": np.array([-0.7071, -0.7071])},
    ]

    env = CyrusStandupEnv()
    model = env.model
    data = env.data
    robot_mass = float(np.sum(model.body_mass))  # 61.32 kg
    sim_dt = model.opt.timestep                  # 0.002s
    frame_skip = env.frame_skip                  # 10
    control_dt = sim_dt * frame_skip             # 0.02s (50Hz)

    torso_id = env.torso_upper_id
    r_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch")
    r_sh_p_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch")

    # Mapped 27 base body joint addresses (excludes unactuated/tendon fingers)
    jnt_ids = [model.actuator_trnid[i, 0] for i in range(27)]
    qpos_adr = [model.jnt_qposadr[jid] for jid in jnt_ids]
    dof_adr = [model.jnt_dofadr[jid] for jid in jnt_ids]

    controller = CyrusBalanceController(model) if use_controller else None

    results = []

    for force_mag in force_levels:
        impulse_ns = float(force_mag * push_duration_s)
        theoretical_delta_v = float(impulse_ns / robot_mass)

        for d_info in DIRECTIONS:
            d_name = d_info["name"]
            fx = float(d_info["dir"][0] * force_mag)
            fy = float(d_info["dir"][1] * force_mag)

            # 1. Reset to clean erect standing posture
            obs, _ = env.reset(seed=42)
            env.data.qpos[:3] = [0.0, 0.0, 0.84]
            env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            env.data.qpos[qpos_adr] = 0.0
            env.data.qvel[:] = 0.0
            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(model, env.data)
            env.knockdown_done = True
            if controller: controller.reset()

            # 2. Settle period (0.4s = 20 control steps)
            for _ in range(20):
                if use_controller:
                    ctrl = controller.compute_control(env.data)
                    env.data.ctrl[:] = ctrl
                    for _ in range(frame_skip):
                        mujoco.mj_step(model, env.data)
                else:
                    if is_walk_policy:
                        walk_obs = np.concatenate([obs, [0.0, 0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
                        action, _ = policy.predict(walk_obs, deterministic=True)
                    else:
                        action, _ = policy.predict(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(action)

            # 3. Apply External Force Perturbation for push_duration_s
            push_steps = int(round(push_duration_s / control_dt))
            for _ in range(push_steps):
                env.data.xfrc_applied[torso_id, 0] = fx
                env.data.xfrc_applied[torso_id, 1] = fy
                if use_controller:
                    ctrl = controller.compute_control(env.data)
                    env.data.ctrl[:] = ctrl
                    for _ in range(frame_skip):
                        mujoco.mj_step(model, env.data)
                else:
                    if is_walk_policy:
                        walk_obs = np.concatenate([obs, [0.0, 0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
                        action, _ = policy.predict(walk_obs, deterministic=True)
                    else:
                        action, _ = policy.predict(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(action)

            # Clear force immediately after push duration expires
            env.data.xfrc_applied[torso_id, :] = 0.0

            # 4. Post-Perturbation Recovery Phase (3.0s = 150 control steps)
            recovery_steps = 150
            fell = False
            recovered = False
            recovery_time_s = None

            max_hip_torque = 0.0
            max_ankle_torque = 0.0
            max_arm_vel = 0.0
            stepped = False

            torques_list = []
            joint_vels_list = []

            for step_r in range(recovery_steps):
                t_rec = step_r * control_dt

                if use_controller:
                    ctrl = controller.compute_control(env.data)
                    env.data.ctrl[:] = ctrl
                    for _ in range(frame_skip):
                        mujoco.mj_step(model, env.data)
                else:
                    if is_walk_policy:
                        walk_obs = np.concatenate([obs, [0.0, 0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
                        action, _ = policy.predict(walk_obs, deterministic=True)
                    else:
                        action, _ = policy.predict(obs, deterministic=True)
                    obs, _, _, _, _ = env.step(action)

                z = float(env.data.qpos[2])
                torso_mat = env.data.xmat[torso_id].reshape(3, 3)
                uprightness = float(torso_mat[2, 2])
                v_com = np.linalg.norm(env.data.qvel[:2])

                torques_list.append(env.data.ctrl[:27].copy())
                joint_vels_list.append(env.data.qvel[dof_adr].copy())

                # Strategy identification parameters
                r_sole_z = float(env.data.site_xpos[env.right_sole_id][2])
                l_sole_z = float(env.data.site_xpos[env.left_sole_id][2])
                if r_sole_z > 0.06 or l_sole_z > 0.06:
                    stepped = True

                # Actuator activity
                hip_tq = float(np.abs(env.data.ctrl[18])) if len(env.data.ctrl) > 18 else 0.0
                ank_tq = float(np.abs(env.data.ctrl[20])) if len(env.data.ctrl) > 20 else 0.0
                arm_v = float(np.abs(env.data.qvel[10])) if len(env.data.qvel) > 10 else 0.0
                max_hip_torque = max(max_hip_torque, hip_tq)
                max_ankle_torque = max(max_ankle_torque, ank_tq)
                max_arm_vel = max(max_arm_vel, arm_v)

                if z < 0.45 or uprightness < 0.60:
                    fell = True
                    break

                # Recovery criteria: uprightness restored and horizontal velocity dissipated
                if not recovered and uprightness >= 0.88 and v_com < 0.08 and z >= 0.78:
                    recovered = True
                    recovery_time_s = t_rec

            is_success = bool(not fell and recovered)

            # Classify primary strategy
            if stepped:
                strategy = "Stepping Strategy"
            elif max_hip_torque > 0.35 and max_arm_vel > 0.8:
                strategy = "Hip & Arm Strategy"
            elif max_hip_torque > 0.35:
                strategy = "Hip Strategy"
            else:
                strategy = "Ankle Strategy"

            ep_work = compute_mechanical_work(np.array(torques_list), np.array(joint_vels_list), control_dt)

            results.append({
                "direction": d_name,
                "angle_deg": float(d_info["angle_deg"]),
                "applied_force_N": float(force_mag),
                "push_duration_s": float(push_duration_s),
                "total_impulse_Ns": impulse_ns,
                "induced_delta_v_ms": theoretical_delta_v,
                "success": is_success,
                "fell": fell,
                "recovery_time_s": recovery_time_s if is_success else None,
                "primary_strategy": strategy,
                "mechanical_work_J": ep_work
            })

    env.close()

    total_tests = len(results)
    success_count = sum(1 for r in results if r["success"])
    success_rate = (success_count / total_tests) * 100.0 if total_tests > 0 else 0.0

    # Find max recoverable disturbance per direction
    max_recoverable = {}
    for d_info in DIRECTIONS:
        d_name = d_info["name"]
        d_successes = [r["total_impulse_Ns"] for r in results if r["direction"] == d_name and r["success"]]
        max_recoverable[d_name] = float(max(d_successes)) if d_successes else 0.0

    summary = {
        "total_perturbations": total_tests,
        "overall_recovery_success_pct": success_rate,
        "max_recoverable_impulse_by_direction_Ns": max_recoverable,
        "maximum_overall_recoverable_impulse_Ns": float(max(max_recoverable.values())),
        "average_recovery_time_s": float(np.mean([r["recovery_time_s"] for r in results if r["recovery_time_s"] is not None])) if success_count > 0 else None,
        "tests": results
    }
    return summary
