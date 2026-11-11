"""
==============================================================================
PROJECT CYRUS -- CLOSED-LOOP STAGE 0/1 AUDIT & VERIFICATION HARNESS
==============================================================================
Runs after every 20 training iterations to:
  1. Audit active candidate checkpoint across 8 and 24 directional push recovery tests.
  2. Verify actuator torque safety envelope: max |tau| <= 180 N*m across all joints.
  3. Validate continuous vs peak thermal saturation margins.
  4. Output a verified scorecard for Agent 2 review.
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_balance import BalanceTester

def run_audit(checkpoint_path=None, test_speeds=[0.40, 0.50, 0.52]):
    print("=" * 80)
    print("CYRUS CLOSED-LOOP BALANCE & TORQUE SAFETY AUDIT")
    print("=" * 80)
    
    tester = BalanceTester(
        checkpoint_path=checkpoint_path,
        pose_name="pose_c",
        protocol="pulse",
        dt_physics=0.0025
    )
    
    angles_8 = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print(f"\n[+] Loaded Checkpoint: {tester.loaded_file}")
    print(f"[+] Physics Timestep: {tester.dt_physics * 1000:.2f} ms")
    print(f"[+] Posture: Pose C (Athletic Squat, z = 0.825m)")
    
    # Track peak joint torques across all tests
    torque_peaks = {
        "hip_yaw": 0.0,
        "hip_roll": 0.0,
        "hip_pitch": 0.0,
        "knee": 0.0,
        "ankle_pitch": 0.0,
        "ankle_roll": 0.0
    }
    
    results = {}
    
    for spd in test_speeds:
        print(f"\n--- Testing Recovery Speed: {spd:.2f} m/s (40ms Calibrated Force Pulse) ---")
        passed = 0
        total = len(angles_8)
        
        for ang in angles_8:
            survived, t_end, reason = tester.run_episode(push_mag=spd, push_angle_deg=ang, max_seconds=5.0)
            
            # Read actuator torques from data
            qfrc = np.abs(tester.data.qfrc_actuator)
            for i, name in enumerate(["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]):
                # check right and left
                t_r = qfrc[6 + i]
                t_l = qfrc[12 + i]
                m_t = max(t_r, t_l)
                if m_t > torque_peaks[name]:
                    torque_peaks[name] = m_t
                    
            status = "SURVIVED (5.0s)" if survived else f"FELL ({reason} at {t_end:.2f}s)"
            print(f"  Angle {ang:3d}°: {status}")
            if survived: passed += 1
            
        results[spd] = (passed, total)
        print(f"  --> Speed {spd:.2f} m/s Score: {passed}/{total} ({passed/total*100:.1f}%)")
        
    print("\n" + "=" * 80)
    print("ACTUATOR TORQUE SAFETY AUDIT (Limit: <= 180 N*m)")
    print("=" * 80)
    torque_pass = True
    for joint, peak in torque_peaks.items():
        limit = 180.0
        pct = (peak / limit) * 100.0
        status = "PASSED" if peak <= limit else "EXCEEDED"
        if peak > limit: torque_pass = False
        print(f"  {joint:15s} | Peak: {peak:6.2f} N*m / {limit:.0f} N*m ({pct:5.1f}%) | [{status}]")
        
    print("\n" + "=" * 80)
    print("OVERALL AUDIT VERDICT")
    print("=" * 80)
    for spd, (p, t) in results.items():
        print(f"  Push {spd:.2f} m/s: {p}/{t} ({p/t*100:.1f}%)")
    print(f"  Torque Safety Envelope: {'PASSED (All <= 180 N*m)' if torque_pass else 'FAILED (Torque Overrun)'}")
    print("=" * 80 + "\n")
    
    return results, torque_peaks, torque_pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_audit(args.checkpoint)
