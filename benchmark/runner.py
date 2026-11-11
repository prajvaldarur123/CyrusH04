import os
import sys
import time
import json
import csv
import hashlib
import argparse
import numpy as np
from datetime import datetime

# Ensure robotics directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stable_baselines3 import PPO

from cyrus.benchmark.standing_eval import evaluate_standing_stability
from cyrus.benchmark.walking_eval import evaluate_walking_performance
from cyrus.benchmark.push_eval import evaluate_push_recovery
from cyrus.benchmark.recovery_eval import evaluate_self_righting


def get_file_hash(filepath: str) -> str:
    """Computes MD5 hash of a file for exact checkpoint verification."""
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def run_full_benchmark(
    standup_model_path: str = "cyrus/checkpoints/best_model.zip",
    walk_model_path: str = "cyrus/checkpoints/walk/best_model.zip",
    num_episodes: int = 10,
    output_dir: str = "cyrus/benchmark/results"
) -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(standup_model_path):
        standup_model_path = os.path.join(base_dir, standup_model_path)
    if not os.path.isabs(walk_model_path):
        walk_model_path = os.path.join(base_dir, walk_model_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 90)
    print("      CYRUS HUMANOID ROBOTICS — COMPREHENSIVE QUANTITATIVE BENCHMARK SUITE")
    print("      Research Basis: van Marum et al., 'Robust Humanoid Standing and Walking'")
    print("=" * 90)

    # 1. Model Metadata & Hash Tracking
    print("\n[1/5] Verifying Policy Checkpoints & System Architecture...")
    standup_exists = os.path.exists(standup_model_path)
    walk_exists = os.path.exists(walk_model_path)

    metadata = {
        "benchmark_timestamp": timestamp_str,
        "robot_name": "Cyrus Humanoid",
        "total_mass_kg": 61.32,
        "total_actuators": 27,
        "total_bodies": 21,
        "standup_policy": {
            "path": standup_model_path,
            "exists": standup_exists,
            "md5_hash": get_file_hash(standup_model_path) if standup_exists else None,
            "obs_dim": 72,
            "act_dim": 27
        },
        "walk_policy": {
            "path": walk_model_path,
            "exists": walk_exists,
            "md5_hash": get_file_hash(walk_model_path) if walk_exists else None,
            "obs_dim": 77,
            "act_dim": 27
        },
        "config": {
            "episodes_per_suite": num_episodes,
            "timestep_s": 0.002,
            "frame_skip": 10,
            "control_dt_s": 0.02,
            "deterministic_inference": True
        }
    }

    print(f" [+] Standup Checkpoint: {standup_model_path} (MD5: {metadata['standup_policy']['md5_hash']})")
    print(f" [+] Walking Checkpoint: {walk_model_path} (MD5: {metadata['walk_policy']['md5_hash']})")

    # Load policies onto CPU for pure, deterministic evaluation
    policy_standup = PPO.load(standup_model_path, device="cpu") if standup_exists else None
    policy_walk = PPO.load(walk_model_path, device="cpu") if walk_exists else None

    results = {"metadata": metadata}

    # 2. Suite 1: Standing Stability
    print("\n[2/5] Running Suite 1: Standing Stability Evaluation...")
    if policy_walk:
        print("  -> Benchmarking Walking Policy Stance Hold...")
        standing_results = evaluate_standing_stability(
            policy=policy_walk,
            num_episodes=num_episodes,
            is_walk_policy=True
        )
    elif policy_standup:
        print("  -> Benchmarking Standup Policy Stance Hold...")
        standing_results = evaluate_standing_stability(
            policy=policy_standup,
            num_episodes=num_episodes,
            is_walk_policy=False
        )
    else:
        standing_results = {"error": "No policy checkpoint available"}
    results["standing_stability"] = standing_results

    if "standing_success_pct" in standing_results:
        print(f"  [+] Success Rate: {standing_results['standing_success_pct']:.1f}% | "
              f"Hold: {standing_results['uninterrupted_standing_duration_mean_s']:.2f}s +/- {standing_results['uninterrupted_standing_duration_std_s']:.2f}s | "
              f"Pelvis Z: {standing_results['pelvis_z_mean_m']:.3f}m | "
              f"Roll RMSE: {standing_results['roll_rmse_deg']:.2f} deg | Pitch RMSE: {standing_results['pitch_rmse_deg']:.2f} deg")

    # 3. Suite 2: Walking Performance & Gait Quality
    print("\n[3/5] Running Suite 2: Walking Performance & Gait Quality Evaluation...")
    if policy_walk:
        walking_results = evaluate_walking_performance(
            policy=policy_walk,
            num_episodes=num_episodes,
            commanded_velocity=0.8
        )
    else:
        walking_results = {"error": "Walking policy checkpoint not available"}
    results["walking_performance"] = walking_results

    if "sustained_fwd_velocity_mean_ms" in walking_results:
        print(f"  [+] Sustained Velocity: {walking_results['sustained_fwd_velocity_mean_ms']:+.3f} +/- {walking_results['sustained_fwd_velocity_std_ms']:.3f} m/s | "
              f"Peak Velocity: {walking_results['peak_fwd_velocity_max_ms']:+.3f} m/s | "
              f"Distance: {walking_results['total_distance_mean_m']:.2f}m | "
              f"Steps: {walking_results['steps_completed_mean']:.1f} | "
              f"Symmetry SI: {walking_results['step_symmetry_si_mean_pct']:.1f}% | "
              f"Clearance: {walking_results['mean_swing_clearance_mm']:.1f} mm | "
              f"Slip: {walking_results['mean_foot_slip_ms']:.3f} m/s")

    # 4. Suite 3: Multi-Directional Push Recovery Disturbance Suite
    print("\n[4/5] Running Suite 3: 8-Direction Push Recovery Disturbance Suite...")
    print("  -> Benchmarking Procedural Dynamic Balance Controller (LIPM + Ankle/Hip/Arm/Step)...")
    push_results = evaluate_push_recovery(
        use_controller=True,
        force_levels=[50.0, 100.0, 150.0, 200.0]
    )
    results["push_recovery"] = push_results

    if "overall_recovery_success_pct" in push_results:
        print(f"  [+] Overall Recovery Success Rate: {push_results['overall_recovery_success_pct']:.1f}% | "
              f"Max Recoverable Impulse: {push_results['maximum_overall_recoverable_impulse_Ns']:.1f} N*s | "
              f"Avg Recovery Time: {push_results['average_recovery_time_s'] or 0.0:.2f}s")
        for d_name, max_j in push_results["max_recoverable_impulse_by_direction_Ns"].items():
            print(f"      - {d_name:12s}: Max Recoverable Impulse = {max_j:5.1f} N*s (equiv delta_v = {max_j/61.32:.2f} m/s)")

    # 5. Suite 4: Multi-Pose Self-Righting Suite
    print("\n[5/5] Running Suite 4: Multi-Pose Self-Righting Suite (Prone, Supine, Lateral L/R)...")
    if policy_standup:
        recovery_results = evaluate_self_righting(
            policy=policy_standup,
            trials_per_pose=num_episodes // 2
        )
    else:
        recovery_results = {"error": "Standup policy checkpoint not available"}
    results["self_righting"] = recovery_results

    if "overall_self_righting_success_pct" in recovery_results:
        print(f"  [+] Overall Self-Righting Success: {recovery_results['overall_self_righting_success_pct']:.1f}% | "
              f"Avg Peak Elevation: {recovery_results['mean_peak_height_overall_m']:.2f}m")
        for p_name, p_sum in recovery_results["pose_summaries"].items():
            print(f"      - {p_name:22s}: Success = {p_sum['success_rate_pct']:5.1f}% | Peak Height = {p_sum['mean_peak_height_m']:.2f}m")

    # Serialize Full JSON Report
    json_path = os.path.join(output_dir, f"baseline_benchmark_{timestamp_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full Machine-Readable JSON Exported: {json_path}")

    # Serialize Summary CSV
    csv_path = os.path.join(output_dir, f"baseline_summary_{timestamp_str}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Metric", "Value", "Unit", "Interpretation"])
        if "standing_success_pct" in standing_results:
            writer.writerow(["Standing Stability", "Success Rate", f"{standing_results['standing_success_pct']:.1f}", "%", "Uninterrupted standing without fall"])
            writer.writerow(["Standing Stability", "Uninterrupted Duration Mean", f"{standing_results['uninterrupted_standing_duration_mean_s']:.2f}", "s", "Continuous stable hold"])
            writer.writerow(["Standing Stability", "Pelvis Height Mean", f"{standing_results['pelvis_z_mean_m']:.3f}", "m", "Nominal 0.84m"])
            writer.writerow(["Standing Stability", "Roll RMSE", f"{standing_results['roll_rmse_deg']:.2f}", "deg", "Lateral tilt deviation"])
            writer.writerow(["Standing Stability", "Pitch RMSE", f"{standing_results['pitch_rmse_deg']:.2f}", "deg", "Sagittal tilt deviation"])
            writer.writerow(["Standing Stability", "Fall Rate", f"{standing_results['fall_pct']:.1f}", "%", "Catastrophic collapse"])

        if "sustained_fwd_velocity_mean_ms" in walking_results:
            writer.writerow(["Walking Performance", "Sustained Velocity Mean", f"{walking_results['sustained_fwd_velocity_mean_ms']:+.3f}", "m/s", "Distance / Time (real walking)"])
            writer.writerow(["Walking Performance", "Peak Forward Velocity", f"{walking_results['peak_fwd_velocity_max_ms']:+.3f}", "m/s", "Single-frame apex velocity"])
            writer.writerow(["Walking Performance", "Distance Mean", f"{walking_results['total_distance_mean_m']:.2f}", "m", "Net translation"])
            writer.writerow(["Walking Performance", "Velocity Tracking RMSE", f"{walking_results['velocity_tracking_rmse_mean']:.3f}", "m/s", "Command tracking error"])
            writer.writerow(["Walking Performance", "Steps Completed Mean", f"{walking_results['steps_completed_mean']:.1f}", "steps", "Detected foot strikes"])
            writer.writerow(["Gait Quality", "Step Symmetry Index (SI)", f"{walking_results['step_symmetry_si_mean_pct']:.1f}", "%", "0% = perfect bilateral symmetry"])
            writer.writerow(["Gait Quality", "Swing Clearance Mean", f"{walking_results['mean_swing_clearance_mm']:.1f}", "mm", "Ground clearance during swing"])
            writer.writerow(["Gait Quality", "Swing Clearance Peak", f"{walking_results['peak_swing_clearance_mm']:.1f}", "mm", "Maximum foot apex clearance"])
            writer.writerow(["Gait Quality", "Foot Slip Velocity", f"{walking_results['mean_foot_slip_ms']:.3f}", "m/s", "Sole horizontal drift in stance"])

        if "overall_recovery_success_pct" in push_results:
            writer.writerow(["Push Recovery", "Overall Success Rate", f"{push_results['overall_recovery_success_pct']:.1f}", "%", "8-direction disturbance rejection"])
            writer.writerow(["Push Recovery", "Max Recoverable Impulse", f"{push_results['maximum_overall_recoverable_impulse_Ns']:.1f}", "N*s", "Force * duration threshold"])

        if "overall_self_righting_success_pct" in recovery_results:
            writer.writerow(["Self-Righting", "Overall Success Rate", f"{recovery_results['overall_self_righting_success_pct']:.1f}", "%", "Recovery across 4 poses"])
            writer.writerow(["Self-Righting", "Average Peak Elevation", f"{recovery_results['mean_peak_height_overall_m']:.2f}", "m", "Height reached during recovery"])

    print(f"[+] Summary CSV Exported: {csv_path}")
    print("=" * 90 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cyrus Humanoid Quantitative Benchmark Suite")
    parser.add_argument("--standup_model", type=str, default="cyrus/checkpoints/best_model.zip")
    parser.add_argument("--walk_model", type=str, default="cyrus/checkpoints/walk/best_model.zip")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="cyrus/benchmark/results")
    args = parser.parse_args()

    run_full_benchmark(
        standup_model_path=args.standup_model,
        walk_model_path=args.walk_model,
        num_episodes=args.episodes,
        output_dir=args.output_dir
    )
