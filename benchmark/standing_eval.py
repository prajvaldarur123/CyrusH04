import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cyrus.envs.standup_env import CyrusStandupEnv
from cyrus.benchmark.metrics import (
    compute_mechanical_work,
    compute_mechanical_effort,
    compute_joint_limit_violations,
    rotation_matrix_to_euler
)


def evaluate_standing_stability(
    policy,
    num_episodes: int = 10,
    max_steps_per_episode: int = 500,  # 500 steps at 50Hz = 10.0s
    seeds: list = None,
    is_walk_policy: bool = False
) -> dict:
    """
    Evaluates Standing Stability under pure stance hold (no initial external shove).
    Measures:
      - Standing success %
      - Average and maximum uninterrupted standing duration (seconds)
      - Pelvis height stability (mean, std, min, max)
      - Torso roll and pitch deviations (mean, std, max, RMSE)
      - Fall percentage
      - Mechanical work and effort
      - Joint limit violations
    """
    if seeds is None:
        seeds = [42 + i * 101 for i in range(num_episodes)]

    env = CyrusStandupEnv()
    model = env.model
    dt = model.opt.timestep * env.frame_skip

    # Extract joint addresses and limits for all 27 actuated joints
    jnt_ids = [model.actuator_trnid[i, 0] for i in range(27)]
    qpos_adr = [model.jnt_qposadr[jid] for jid in jnt_ids]
    dof_adr = [model.jnt_dofadr[jid] for jid in jnt_ids]
    joint_limits = np.zeros((27, 2), dtype=np.float32)
    for i in range(27):
        joint_limits[i] = model.jnt_range[jnt_ids[i]]

    episode_records = []

    for ep_idx, seed in enumerate(seeds[:num_episodes]):
        np.random.seed(seed)
        obs, info = env.reset(seed=seed)
        
        # Override initial pose: start directly in nominal erect standing posture at z=0.84m
        env.data.qpos[:3] = [0.0, 0.0, 0.84]
        env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        env.data.qpos[qpos_adr] = 0.0
        env.data.qvel[:] = 0.0
        env.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, env.data)
        env.knockdown_done = True  # bypass initial 3.0s tumble shove to evaluate pure standing stability

        pelvis_z_list = []
        roll_list = []
        pitch_list = []
        torques_list = []
        joint_vels_list = []
        joint_pos_list = []

        uninterrupted_stand_steps = 0
        max_stand_steps_in_ep = 0
        fell = False
        standing_threshold_z = 0.75
        upright_threshold = 0.85

        for step in range(max_steps_per_episode):
            # Compute policy action
            if is_walk_policy:
                # 77-dim obs: append stance command [0, 0, 0, 0, 1.0]
                walk_obs = np.concatenate([obs, [0.0, 0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
                action, _ = policy.predict(walk_obs, deterministic=True)
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            # Record kinematics
            z = float(env.data.qpos[2])
            pelvis_z_list.append(z)

            torso_mat = env.data.xmat[env.torso_upper_id].reshape(3, 3)
            roll, pitch, _ = rotation_matrix_to_euler(torso_mat)
            roll_list.append(roll)
            pitch_list.append(pitch)
            uprightness = float(torso_mat[2, 2])

            # Actuation and energy
            torques_list.append(env.data.ctrl[:27].copy())
            joint_vels_list.append(env.data.qvel[dof_adr].copy())
            joint_pos_list.append(env.data.qpos[qpos_adr].copy())

            # Check standing criteria
            r_sole_z = float(env.data.site_xpos[env.right_sole_id][2])
            l_sole_z = float(env.data.site_xpos[env.left_sole_id][2])
            is_currently_standing = bool(
                z >= standing_threshold_z and
                uprightness >= upright_threshold and
                r_sole_z < 0.06 and
                l_sole_z < 0.06
            )

            if is_currently_standing:
                uninterrupted_stand_steps += 1
                if uninterrupted_stand_steps > max_stand_steps_in_ep:
                    max_stand_steps_in_ep = uninterrupted_stand_steps
            else:
                uninterrupted_stand_steps = 0

            # Fall check
            if z < 0.50 or uprightness < 0.60:
                fell = True
                break

            if terminated or truncated:
                break

        # Compute episode-level metrics
        pelvis_z_arr = np.array(pelvis_z_list)
        roll_arr = np.array(roll_list)
        pitch_arr = np.array(pitch_list)
        torques_arr = np.array(torques_list)
        joint_vels_arr = np.array(joint_vels_list)
        joint_pos_arr = np.array(joint_pos_list)

        ep_duration = len(pelvis_z_arr) * dt
        uninterrupted_stand_duration = max_stand_steps_in_ep * dt
        standing_success = bool(not fell and uninterrupted_stand_duration >= (ep_duration * 0.85))

        ep_work = compute_mechanical_work(torques_arr, joint_vels_arr, dt)
        ep_effort = compute_mechanical_effort(torques_arr, dt)
        jl_viol = compute_joint_limit_violations(joint_pos_arr, joint_limits)

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "success": standing_success,
            "fell": fell,
            "duration_s": ep_duration,
            "uninterrupted_stand_s": uninterrupted_stand_duration,
            "pelvis_z_mean": float(np.mean(pelvis_z_arr)),
            "pelvis_z_std": float(np.std(pelvis_z_arr)),
            "pelvis_z_min": float(np.min(pelvis_z_arr)),
            "pelvis_z_max": float(np.max(pelvis_z_arr)),
            "roll_rmse_deg": float(np.sqrt(np.mean(np.square(roll_arr))) * 180.0 / np.pi),
            "pitch_rmse_deg": float(np.sqrt(np.mean(np.square(pitch_arr))) * 180.0 / np.pi),
            "max_roll_dev_deg": float(np.max(np.abs(roll_arr)) * 180.0 / np.pi),
            "max_pitch_dev_deg": float(np.max(np.abs(pitch_arr)) * 180.0 / np.pi),
            "mechanical_work_J": ep_work,
            "mechanical_effort_N2s": ep_effort,
            "joint_limit_violations": jl_viol["violation_count"],
            "max_joint_limit_pen_rad": jl_viol["max_penetration_rad"]
        })

    env.close()

    # Aggregate over all episodes
    success_rate = float(np.mean([r["success"] for r in episode_records]) * 100.0)
    fall_rate = float(np.mean([r["fell"] for r in episode_records]) * 100.0)

    summary = {
        "num_episodes": len(episode_records),
        "standing_success_pct": success_rate,
        "fall_pct": fall_rate,
        "uninterrupted_standing_duration_mean_s": float(np.mean([r["uninterrupted_stand_s"] for r in episode_records])),
        "uninterrupted_standing_duration_std_s": float(np.std([r["uninterrupted_stand_s"] for r in episode_records])),
        "max_standing_duration_s": float(np.max([r["uninterrupted_stand_s"] for r in episode_records])),
        "pelvis_z_mean_m": float(np.mean([r["pelvis_z_mean"] for r in episode_records])),
        "pelvis_z_std_m": float(np.mean([r["pelvis_z_std"] for r in episode_records])),
        "roll_rmse_deg": float(np.mean([r["roll_rmse_deg"] for r in episode_records])),
        "pitch_rmse_deg": float(np.mean([r["pitch_rmse_deg"] for r in episode_records])),
        "mechanical_work_J_mean": float(np.mean([r["mechanical_work_J"] for r in episode_records])),
        "mechanical_effort_N2s_mean": float(np.mean([r["mechanical_effort_N2s"] for r in episode_records])),
        "joint_limit_violations_mean": float(np.mean([r["joint_limit_violations"] for r in episode_records])),
        "episodes": episode_records
    }
    return summary
