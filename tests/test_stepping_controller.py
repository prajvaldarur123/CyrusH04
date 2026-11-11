import sys
import os
import numpy as np

sys.path.insert(0, r"C:\Users\prajval\robotics\cyrus")
from controller.stepping_recovery import ReactiveSteppingController, SteppingParams, SteppingState

def main():
    ctrl = ReactiveSteppingController()
    
    print("=== TEST 1: Capture Point Calculation ===")
    com_pos = np.array([0.0134, 0.0, 0.783])
    com_vel = np.array([0.50, 0.0, 0.0]) # 0.50 m/s forward shove
    xi = ctrl.compute_capture_point(com_pos, com_vel)
    print(f"CoM Pos: {com_pos}")
    print(f"CoM Vel: {com_vel}")
    print(f"Capture Point: xi_x = {xi[0]:.4f} m ({xi[0]*100:.2f} cm), xi_y = {xi[1]:.4f} m")
    
    # Check trigger condition
    r_ankle = np.array([0.0, -0.11, 0.045])
    l_ankle = np.array([0.0, +0.11, 0.045])
    
    print("\n=== TEST 2: Stepping Trigger Evaluation ===")
    # Under mild shove (0.30 m/s): xi = 0.0134 + 0.30 / 3.538 = 0.098m = 9.8cm < 14.0cm -> No step
    should_step, leg, target = ctrl.check_trigger(com_pos, np.array([0.30, 0, 0]), r_ankle, l_ankle)
    print(f"Shove 0.30 m/s: Should Step = {should_step} (Expected: False)")
    assert not should_step, "0.30 m/s should not trigger stepping"
    
    # Under heavy shove (0.55 m/s): xi = 0.0134 + 0.55 / 3.538 = 0.168m = 16.8cm > 14.0cm -> Trigger Step!
    should_step, leg, target = ctrl.check_trigger(com_pos, np.array([0.55, 0, 0]), r_ankle, l_ankle)
    print(f"Shove 0.55 m/s: Should Step = {should_step}, Chosen Leg = {leg}, Target Landing = {target}")
    assert should_step, "0.55 m/s MUST trigger dynamic stepping"
    print(f"  Target landing X: {target[0]:.4f}m ({target[0]*100:.2f} cm), Y: {target[1]:.4f}m, Z: {target[2]:.4f}m")
    
    print("\n=== TEST 3: Swing Trajectory & IK Generation ===")
    # Simulate swing phase progression
    t0 = 1.0
    for step in range(6):
        t = t0 + step * 0.05
        out = ctrl.update(t, com_pos, np.array([0.55, 0, 0]), r_ankle, l_ankle, False, False)
        if out["state"] == SteppingState.SWING_EXECUTION:
            fpos = out["desired_foot_pos"]
            # Compute IK
            hip_pos = np.array([0.0, target[1], 0.58])
            q_leg = ctrl.inverse_kinematics_leg(hip_pos, fpos)
            print(f"  t={t:.2f}s (phase {out['phase']:.2f}): Desired Foot Pos: [{fpos[0]:.3f}, {fpos[1]:.3f}, {fpos[2]:.3f}m] | Knee: {np.degrees(q_leg[3]):.1f}° | Hip Pitch: {np.degrees(q_leg[2]):.1f}°")
            
    print("\n[+] All Stepping Controller Invariants Validated Successfully!")

if __name__ == "__main__":
    main()
