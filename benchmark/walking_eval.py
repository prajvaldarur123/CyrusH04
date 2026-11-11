import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cyrus.envs.walk_balance_env import CyrusWalkBalanceEnv
from cyrus.benchmark.metrics import (
    compute_mechanical_work,
    compute_mechanical_effort,
    compute_step_symmetry,
    compute_foot_slipping,
    compute_joint_jerk,
    compute_joint_limit_violations
)


def evaluate_walking_performance(
    policy,
    num_episodes: int = 10,
    commanded_velocity: float = 0.8,  # nominal walking command 0.8 m/s
    max_steps_per_episode: int = 750,  # 750 steps at 50Hz = 15.0s
    seeds: list = None
) -> dict:
    """
    Evaluates Walking Performance & Gait Quality under commanded forward velocity.
    Measures:
      - Sustained average forward velocity vs. peak forward velocity
      - Total distance travelled
      - Commanded vs. actual velocity tracking error
      - Walking episode duration
      - Number of steps completed
      - Fall percentage
      - Left-right step symmetry (SI %)
      - Stance and swing durations (Left and Right)
      - Swing foot clearance (peak and mean)
      - Foot slipping velocity during stance
      - Contact chatter / bounce frequency
      - Unnecessary joint jerk / oscillations
      - Mechanical work and effort
    """
    if seeds is None:
        seeds = [100 + i * 37 for i in range(num_episodes)]

    env = CyrusWalkBalanceEnv(curriculum_phase=5)  # Phase 5: walking cycle
    model = env.model
    dt = env.control_dt

    # Mapped 27 base body joint addresses (excludes unactuated fingers)
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

        # Set target command: commanded_velocity forward, zero lateral, zero yaw, 0 sway, phase 5
        env.target_fwd_vel = commanded_velocity
        env.target_lat_vel = 0.0
        env.target_yaw_rate = 0.0

        pos_x_list = []
        pos_y_list = []
        vel_x_list = []
        torques_list = []
        joint_vels_list = []
        joint_pos_list = []

        r_foot_forces = []
        l_foot_forces = []
        r_sole_vels = []
        l_sole_vels = []
        r_clearances = []
        l_clearances = []

        # Step detector tracking
        r_step_times = []
        l_step_times = []
        r_stance_times = []
        l_stance_times = []

        r_in_stance = True
        l_in_stance = True
        r_phase_start = 0.0
        l_phase_start = 0.0

        fell = False
        initial_x = float(env.data.qpos[0])
        initial_y = float(env.data.qpos[1])

        for step in range(max_steps_per_episode):
            t_curr = step * dt

            # Extract 77-dim obs for walking policy
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Kinematics
            px = float(env.data.qpos[0])
            py = float(env.data.qpos[1])
            pz = float(env.data.qpos[2])
            vx = float(env.data.qvel[0])

            pos_x_list.append(px)
            pos_y_list.append(py)
            vel_x_list.append(vx)

            # Actuation
            torques_list.append(env.data.ctrl[:27].copy())
            joint_vels_list.append(env.data.qvel[dof_adr].copy())
            joint_pos_list.append(env.data.qpos[qpos_adr].copy())

            # Foot contact and ground reaction forces
            r_sole_z = float(env.data.site_xpos[env.right_sole_id][2])
            l_sole_z = float(env.data.site_xpos[env.left_sole_id][2])

            # Extract vertical normal forces via sensor / body contact approximation
            r_fz = max(0.0, float(env.data.cfrc_ext[env.r_foot_id, 2]))
            l_fz = max(0.0, float(env.data.cfrc_ext[env.l_foot_id, 2]))
            r_foot_forces.append(r_fz)
            l_foot_forces.append(l_fz)

            # Foot linear velocities (approximated from body jacobian / site xvel)
            r_foot_vel = env.data.cvel[env.r_foot_id][:3].copy()
            l_foot_vel = env.data.cvel[env.l_foot_id][:3].copy()
            r_sole_vels.append(r_foot_vel)
            l_sole_vels.append(l_foot_vel)

            # Step detection & phase duration tracking
            r_now_stance = bool(r_sole_z < 0.055)
            l_now_stance = bool(l_sole_z < 0.055)

            if not r_now_stance:
                r_clearances.append(r_sole_z)
            if not l_now_stance:
                l_clearances.append(l_sole_z)

            # Right leg transition
            if r_now_stance != r_in_stance:
                dur = t_curr - r_phase_start
                if r_in_stance:
                    r_stance_times.append(dur)
                else:
                    r_step_times.append(dur)
                r_in_stance = r_now_stance
                r_phase_start = t_curr

            # Left leg transition
            if l_now_stance != l_in_stance:
                dur = t_curr - l_phase_start
                if l_in_stance:
                    l_stance_times.append(dur)
                else:
                    l_step_times.append(dur)
                l_in_stance = l_now_stance
                l_phase_start = t_curr

            # Fall check
            torso_mat = env.data.xmat[env.torso_upper_id].reshape(3, 3)
            uprightness = float(torso_mat[2, 2])
            if pz < 0.45 or uprightness < 0.60:
                fell = True
                break

            if terminated or truncated:
                break

        # Compute trajectory-level metrics
        final_x = pos_x_list[-1] if pos_x_list else initial_x
        final_y = pos_y_list[-1] if pos_y_list else initial_y
        total_dist = float(np.hypot(final_x - initial_x, final_y - initial_y))

        vel_x_arr = np.array(vel_x_list)
        torques_arr = np.array(torques_list)
        joint_vels_arr = np.array(joint_vels_list)
        joint_pos_arr = np.array(joint_pos_list)

        ep_duration = len(vel_x_arr) * dt
        sustained_fwd_vel = float(total_dist / max(0.001, ep_duration)) if (final_x - initial_x) >= 0 else -float(total_dist / max(0.001, ep_duration))
        peak_fwd_vel = float(np.max(vel_x_arr)) if len(vel_x_arr) > 0 else 0.0

        tracking_mae = float(np.mean(np.abs(vel_x_arr - commanded_velocity)))
        tracking_rmse = float(np.sqrt(np.mean(np.square(vel_x_arr - commanded_velocity))))

        # Gait Quality
        total_steps = len(r_step_times) + len(l_step_times)
        step_symmetry_si = compute_step_symmetry(l_step_times, r_step_times)

        r_slip = compute_foot_slipping(np.array(r_sole_vels), np.array(r_foot_forces))
        l_slip = compute_foot_slipping(np.array(l_sole_vels), np.array(l_foot_forces))
        mean_foot_slip = 0.5 * (r_slip["mean_slip_vel"] + l_slip["mean_slip_vel"])

        mean_swing_clearance = float(np.mean(r_clearances + l_clearances)) if (r_clearances or l_clearances) else 0.0
        peak_swing_clearance = float(np.max(r_clearances + l_clearances)) if (r_clearances or l_clearances) else 0.0

        ep_work = compute_mechanical_work(torques_arr, joint_vels_arr, dt)
        ep_effort = compute_mechanical_effort(torques_arr, dt)
        ep_jerk = compute_joint_jerk(joint_pos_arr, dt)
        jl_viol = compute_joint_limit_violations(joint_pos_arr, joint_limits)

        episode_records.append({
            "episode": ep_idx + 1,
            "seed": seed,
            "fell": fell,
            "duration_s": ep_duration,
            "total_distance_m": total_dist,
            "sustained_fwd_velocity_ms": sustained_fwd_vel,
            "peak_fwd_velocity_ms": peak_fwd_vel,
            "commanded_velocity_ms": commanded_velocity,
            "velocity_tracking_mae": tracking_mae,
            "velocity_tracking_rmse": tracking_rmse,
            "steps_completed": total_steps,
            "step_symmetry_si_pct": step_symmetry_si,
            "mean_stance_duration_s": float(np.mean(r_stance_times + l_stance_times)) if (r_stance_times or l_stance_times) else 0.0,
            "mean_swing_duration_s": float(np.mean(r_step_times + l_step_times)) if (r_step_times or l_step_times) else 0.0,
            "mean_swing_clearance_mm": mean_swing_clearance * 1000.0,
            "peak_swing_clearance_mm": peak_swing_clearance * 1000.0,
            "mean_foot_slip_ms": mean_foot_slip,
            "joint_jerk": ep_jerk,
            "mechanical_work_J": ep_work,
            "mechanical_effort_N2s": ep_effort,
            "joint_limit_violations": jl_viol["violation_count"]
        })

    env.close()

    fall_rate = float(np.mean([r["fell"] for r in episode_records]) * 100.0)

    summary = {
        "num_episodes": len(episode_records),
        "fall_pct": fall_rate,
        "sustained_fwd_velocity_mean_ms": float(np.mean([r["sustained_fwd_velocity_ms"] for r in episode_records])),
        "sustained_fwd_velocity_std_ms": float(np.std([r["sustained_fwd_velocity_ms"] for r in episode_records])),
        "peak_fwd_velocity_mean_ms": float(np.mean([r["peak_fwd_velocity_ms"] for r in episode_records])),
        "peak_fwd_velocity_max_ms": float(np.max([r["peak_fwd_velocity_ms"] for r in episode_records])),
        "total_distance_mean_m": float(np.mean([r["total_distance_m"] for r in episode_records])),
        "velocity_tracking_rmse_mean": float(np.mean([r["velocity_tracking_rmse"] for r in episode_records])),
        "steps_completed_mean": float(np.mean([r["steps_completed"] for r in episode_records])),
        "step_symmetry_si_mean_pct": float(np.mean([r["step_symmetry_si_pct"] for r in episode_records])),
        "mean_swing_clearance_mm": float(np.mean([r["mean_swing_clearance_mm"] for r in episode_records])),
        "peak_swing_clearance_mm": float(np.max([r["peak_swing_clearance_mm"] for r in episode_records])),
        "mean_foot_slip_ms": float(np.mean([r["mean_foot_slip_ms"] for r in episode_records])),
        "joint_jerk_mean": float(np.mean([r["joint_jerk"] for r in episode_records])),
        "mechanical_work_J_mean": float(np.mean([r["mechanical_work_J"] for r in episode_records])),
        "mechanical_effort_N2s_mean": float(np.mean([r["mechanical_effort_N2s"] for r in episode_records])),
        "joint_limit_violations_mean": float(np.mean([r["joint_limit_violations"] for r in episode_records])),
        "episodes": episode_records
    }
    return summary


if __name__ == "__main__":
    import argparse
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description="Evaluate Cyrus Humanoid Walking Performance")
    parser.add_argument("--model", type=str, default="cyrus/checkpoints/walk/best_model.zip", help="Path to walk policy")
    parser.add_argument("--episodes", type=int, default=6, help="Number of evaluation episodes")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model checkpoint {args.model} not found.")
        sys.exit(1)

    print(f"[+] Loading walking model: {args.model}")
    policy = PPO.load(args.model, device="cpu")
    summary = evaluate_walking_performance(policy, num_episodes=args.episodes)

    print("\n" + "=" * 80)
    print("      CYRUS HUMANOID WALKING PERFORMANCE & GAIT QUALITY REPORT")
    print("=" * 80)
    print(f" Sustained Forward Velocity (mean): {summary['sustained_fwd_velocity_mean_ms']:+.3f} +/- {summary['sustained_fwd_velocity_std_ms']:.3f} m/s")
    print(f" Peak Instantaneous Velocity (max): {summary['peak_fwd_velocity_max_ms']:+.3f} m/s")
    print(f" Net Translation Distance (mean):   {summary['total_distance_mean_m']:.2f} m")
    print(f" Velocity Tracking RMSE:            {summary['velocity_tracking_rmse_mean']:.3f} m/s")
    print(f" Steps Completed (mean):            {summary['steps_completed_mean']:.1f} steps")
    print(f" Step Symmetry Index (SI):          {summary['step_symmetry_si_mean_pct']:.1f}% (0% = perfect symmetry)")
    print(f" Swing Foot Clearance (mean):       {summary['mean_swing_clearance_mm']:.1f} mm")
    print(f" Foot Slipping Velocity:            {summary['mean_foot_slip_ms']:.3f} m/s")
    print(f" Joint Jerk:                        {summary['joint_jerk_mean']:.1f} rad/s^3")
    print(f" Mechanical Work:                   {summary['mechanical_work_J_mean']:.2f} J")
    print(f" Mechanical Effort:                 {summary['mechanical_effort_N2s_mean']:.2f} N^2*s")
    print(f" Joint Limit Violations:            {summary['joint_limit_violations_mean']:.1f} steps/ep")
    print(f" Fall Rate:                         {summary['fall_pct']:.1f}%")
    print("=" * 80 + "\n")
