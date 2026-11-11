import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cyrus.envs.standup_env import CyrusStandupEnv
from cyrus.benchmark.metrics import compute_mechanical_work, compute_mechanical_effort


def evaluate_self_righting(
    policy,
    trials_per_pose: int = 5,
    max_steps_per_trial: int = 400,  # 400 steps at 50Hz = 8.0s
    post_recovery_hold_target_s: float = 3.0
) -> dict:
    """
    Evaluates Ground Self-Righting from 4 Canonical Anatomical Initial Poses:
      1. Prone (Belly Down, Face Down)
      2. Supine (Back Down, Face Up)
      3. Lateral Left (Side-Lying on Left Flank)
      4. Lateral Right (Side-Lying on Right Flank)

    Measures:
      - Success rate per initial orientation and overall
      - Time to reach standing (z >= 0.78m, uprightness >= 0.85)
      - Time spent in unstable intermediate stages (push-up, quadruped, kneeling)
      - Whether the robot remains upright after recovery (sustained 3.0s hold)
      - Mechanical work and energy expenditure
    """
    POSES = [
        {"name": "Prone (Belly Down)",   "quat": [0.7071, 0.7071, 0.0, 0.0],  "z_init": 0.18},
        {"name": "Supine (Back Down)",   "quat": [0.7071, -0.7071, 0.0, 0.0], "z_init": 0.18},
        {"name": "Lateral Left",        "quat": [0.7071, 0.0, 0.7071, 0.0],  "z_init": 0.18},
        {"name": "Lateral Right",       "quat": [0.7071, 0.0, -0.7071, 0.0], "z_init": 0.18},
    ]

    env = CyrusStandupEnv()
    model = env.model
    dt = model.opt.timestep * env.frame_skip  # 0.02s

    # Mapped 27 base body joint addresses (excludes unactuated fingers)
    jnt_ids = [model.actuator_trnid[i, 0] for i in range(27)]
    qpos_adr = [model.jnt_qposadr[jid] for jid in jnt_ids]
    dof_adr = [model.jnt_dofadr[jid] for jid in jnt_ids]

    results = []

    for pose_info in POSES:
        p_name = pose_info["name"]
        p_quat = pose_info["quat"]
        p_z = pose_info["z_init"]

        for t_idx in range(trials_per_pose):
            seed = 42 + hash(p_name) % 1000 + t_idx * 53
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)

            # Enforce exact initial pose on the floor
            env.data.qpos[:3] = [0.0, 0.0, p_z]
            env.data.qpos[3:7] = p_quat
            env.data.qpos[qpos_adr] = np.random.uniform(-0.05, 0.05, size=27)  # small joint variance
            env.data.qvel[:] = 0.0
            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(model, env.data)
            env.knockdown_done = True  # already on floor, evaluating ground ascent

            stood_up = False
            time_to_standing = None
            intermediate_time_s = 0.0
            standing_hold_steps = 0
            required_hold_steps = int(round(post_recovery_hold_target_s / dt))

            peak_z = float(p_z)
            torques_list = []
            joint_vels_list = []

            for step in range(max_steps_per_trial):
                t_elapsed = step * dt
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                z = float(env.data.qpos[2])
                peak_z = max(peak_z, z)
                torso_mat = env.data.xmat[env.torso_upper_id].reshape(3, 3)
                uprightness = float(torso_mat[2, 2])
                r_sole_z = float(env.data.site_xpos[env.right_sole_id][2])
                l_sole_z = float(env.data.site_xpos[env.left_sole_id][2])

                torques_list.append(env.data.ctrl[:27].copy())
                joint_vels_list.append(env.data.qvel[dof_adr].copy())

                is_standing_now = bool(
                    z >= 0.78 and
                    uprightness >= 0.85 and
                    r_sole_z < 0.06 and
                    l_sole_z < 0.06
                )

                if is_standing_now:
                    if not stood_up:
                        stood_up = True
                        time_to_standing = t_elapsed
                    standing_hold_steps += 1
                else:
                    standing_hold_steps = 0
                    if 0.25 <= z < 0.78:
                        intermediate_time_s += dt

                # Mastered required hold duration post recovery?
                if standing_hold_steps >= required_hold_steps:
                    break

                if terminated or truncated:
                    break

            post_recovery_stable = bool(standing_hold_steps >= required_hold_steps)
            ep_work = compute_mechanical_work(np.array(torques_list), np.array(joint_vels_list), dt)
            ep_effort = compute_mechanical_effort(np.array(torques_list), dt)

            results.append({
                "pose": p_name,
                "trial": t_idx + 1,
                "seed": seed,
                "stood_up": stood_up,
                "post_recovery_stable": post_recovery_stable,
                "time_to_standing_s": time_to_standing,
                "intermediate_time_s": float(intermediate_time_s),
                "peak_height_m": float(peak_z),
                "mechanical_work_J": ep_work,
                "mechanical_effort_N2s": ep_effort
            })

    env.close()

    # Aggregate by pose
    pose_summaries = {}
    for pose_info in POSES:
        p_name = pose_info["name"]
        p_res = [r for r in results if r["pose"] == p_name]
        p_succ = sum(1 for r in p_res if r["post_recovery_stable"])
        times = [r["time_to_standing_s"] for r in p_res if r["time_to_standing_s"] is not None]
        pose_summaries[p_name] = {
            "success_rate_pct": float(p_succ / len(p_res) * 100.0) if p_res else 0.0,
            "mean_time_to_standing_s": float(np.mean(times)) if times else None,
            "mean_peak_height_m": float(np.mean([r["peak_height_m"] for r in p_res]))
        }

    overall_success = sum(1 for r in results if r["post_recovery_stable"])
    overall_rate = float(overall_success / len(results) * 100.0)

    summary = {
        "total_trials": len(results),
        "overall_self_righting_success_pct": overall_rate,
        "pose_summaries": pose_summaries,
        "mean_intermediate_time_s": float(np.mean([r["intermediate_time_s"] for r in results])),
        "mean_peak_height_overall_m": float(np.mean([r["peak_height_m"] for r in results])),
        "trials": results
    }
    return summary
