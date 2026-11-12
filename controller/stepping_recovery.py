"""
==============================================================================
PROJECT CYRUS -- GATE 5: DYNAMIC STEPPING PUSH RECOVERY CONTROLLER
==============================================================================
Implements the Reactive Stepping Inverted Pendulum Model (LIPM) for Cyrus Humanoid:
When an external disturbance pushes the Capture Point beyond the physical
polygon of support (xi > x_toe - 0.02m), an automated recovery step is triggered.

Key Invariants:
  1. Capture Point: xi = x_com + v_com / omega_0, where omega_0 = sqrt(g / z_com)
  2. Reactive Trigger: xi_x > x_toe - 0.02m (Forward) or xi_x < x_heel + 0.02m (Backward)
  3. Foot Target: x_land = xi + k_margin * v_com (with k_margin = 0.05s)
  4. Parabolic Clearance: h_clear = 0.07m (7 cm peak) over T_swing = 0.25s
  5. Closed-Form 6-DOF Leg IK: Exact mapping to Cyrus thigh (0.20m) & shin (0.34m)
==============================================================================
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

class SteppingState(Enum):
    QUIET_STANCE = 0
    SWING_EXECUTION = 1
    TOUCHDOWN_ABSORPTION = 2
    DUAL_SUPPORT_SETTLE = 3

@dataclass
class SteppingParams:
    toe_limit: float = 0.140       # Toe edge from ankle: +14.0 cm
    heel_limit: float = -0.070     # Heel edge from ankle: -7.0 cm
    lateral_limit: float = 0.050   # Foot outer edge from ankle: +5.0 cm
    trigger_margin: float = 0.010  # Safety trigger margin: 1.0 cm
    k_margin: float = 0.050        # Velocity landing offset coefficient: 0.05 s
    h_clear: float = 0.080         # Parabolic foot clearance: 8.0 cm
    t_swing: float = 0.260         # Swing phase duration: 0.26 s
    t_absorb: float = 0.120        # Touchdown compliance absorption: 0.12 s
    nominal_z_com: float = 0.783   # Nominal center of mass height: 0.783 m
    gravity: float = 9.80665       # Gravitational acceleration
    thigh_len: float = 0.400       # Cyrus thigh link length (hip to knee): exactly 0.400m
    shin_len: float = 0.340        # Cyrus shin link length (knee to ankle): exactly 0.340m
    ankle_to_sole: float = 0.040   # Ankle center to foot sole

class ReactiveSteppingController:
    def __init__(self, params: Optional[SteppingParams] = None):
        self.p = params or SteppingParams()
        self.state = SteppingState.QUIET_STANCE
        self.swing_leg: Optional[str] = None # "right" or "left"
        self.t_state_start = 0.0
        
        # Trajectory anchors
        self.swing_start_pos = np.zeros(3)
        self.swing_target_pos = np.zeros(3)
        
        # Natural pendulum frequency
        self.omega_0 = np.sqrt(self.p.gravity / self.p.nominal_z_com)

    def compute_capture_point(self, com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
        """Computes 2D Capture Point: xi = r_com + v_com / omega_0"""
        z_c = max(com_pos[2], 0.40)
        omega = np.sqrt(self.p.gravity / z_c)
        xi = com_pos[:2] + com_vel[:2] / omega
        return xi

    def check_trigger(self, com_pos: np.ndarray, com_vel: np.ndarray,
                      r_ankle_pos: np.ndarray, l_ankle_pos: np.ndarray) -> Tuple[bool, Optional[str], np.ndarray]:
        """
        Evaluates if the capture point exceeds the support polygon margin.
        Returns: (should_step, swing_leg, target_landing_pos)
        """
        xi = self.compute_capture_point(com_pos, com_vel)
        
        # Support polygon center and bounds (dual-stance)
        mean_ankle = 0.5 * (r_ankle_pos[:2] + l_ankle_pos[:2])
        fwd_threshold = mean_ankle[0] + self.p.toe_limit - self.p.trigger_margin
        bwd_threshold = mean_ankle[0] + self.p.heel_limit + self.p.trigger_margin
        
        should_step = False
        chosen_leg = None
        
        # Forward push recovery
        if xi[0] > fwd_threshold:
            should_step = True
            # Choose swing leg matching lateral sway or default right
            chosen_leg = "left" if com_vel[1] > 0.0 else "right"
            
        # Backward push recovery
        elif xi[0] < bwd_threshold:
            should_step = True
            chosen_leg = "left" if com_vel[1] > 0.0 else "right"
            
        # Lateral push recovery
        elif xi[1] > (l_ankle_pos[1] + self.p.lateral_limit - self.p.trigger_margin):
            should_step = True
            chosen_leg = "left"  # Step out with left leg
        elif xi[1] < (r_ankle_pos[1] - self.p.lateral_limit + self.p.trigger_margin):
            should_step = True
            chosen_leg = "right" # Step out with right leg

        if not should_step:
            return False, None, np.zeros(3)

        # Target landing position: x_land = xi + k_margin * v_com
        x_target = xi[0] + self.p.k_margin * com_vel[0]
        y_target = xi[1] + self.p.k_margin * com_vel[1]
        
        # Enforce minimum stance separation to avoid foot self-collision
        min_half_width = 0.11 # 11 cm half-stance
        if chosen_leg == "right":
            y_target = min(y_target, l_ankle_pos[1] - min_half_width)
        else:
            y_target = max(y_target, r_ankle_pos[1] + min_half_width)
            
        z_target = self.p.ankle_to_sole # Ankle height at flat foot ground contact
        target_landing = np.array([x_target, y_target, z_target], dtype=np.float64)
        
        return True, chosen_leg, target_landing

    def update(self, current_time: float, com_pos: np.ndarray, com_vel: np.ndarray,
               r_ankle_pos: np.ndarray, l_ankle_pos: np.ndarray,
               r_contact: bool, l_contact: bool) -> Dict:
        """
        Advances the stepping state machine and outputs swing foot position trajectory.
        """
        dt_state = current_time - self.t_state_start
        
        if self.state == SteppingState.QUIET_STANCE:
            should_step, leg, target = self.check_trigger(com_pos, com_vel, r_ankle_pos, l_ankle_pos)
            if should_step:
                self.state = SteppingState.SWING_EXECUTION
                self.swing_leg = leg
                self.t_state_start = current_time
                self.swing_start_pos = r_ankle_pos.copy() if leg == "right" else l_ankle_pos.copy()
                self.swing_target_pos = target
                dt_state = 0.0

        if self.state == SteppingState.SWING_EXECUTION:
            # Normalized phase s in [0, 1]
            s = np.clip(dt_state / self.p.t_swing, 0.0, 1.0)
            
            # Smooth horizontal trajectory (cubic minimum-jerk profile)
            s_smooth = 3.0 * s**2 - 2.0 * s**3
            cur_xy = (1.0 - s_smooth) * self.swing_start_pos[:2] + s_smooth * self.swing_target_pos[:2]
            
            # Parabolic vertical foot clearance: 4 * h_clear * s * (1 - s)
            cur_z = (1.0 - s) * self.swing_start_pos[2] + s * self.swing_target_pos[2] + 4.0 * self.p.h_clear * s * (1.0 - s)
            desired_foot_pos = np.array([cur_xy[0], cur_xy[1], cur_z], dtype=np.float64)
            
            # Touchdown transition condition
            if s >= 1.0 or (s > 0.70 and ((self.swing_leg == "right" and r_contact) or (self.swing_leg == "left" and l_contact))):
                self.state = SteppingState.TOUCHDOWN_ABSORPTION
                self.t_state_start = current_time
                
            return {
                "state": self.state,
                "swing_leg": self.swing_leg,
                "desired_foot_pos": desired_foot_pos,
                "phase": s,
                "target_pos": self.swing_target_pos
            }

        elif self.state == SteppingState.TOUCHDOWN_ABSORPTION:
            # Compliance phase: hold flat ground contact while shifting weight
            if dt_state >= self.p.t_absorb:
                self.state = SteppingState.DUAL_SUPPORT_SETTLE
                self.t_state_start = current_time
                
            return {
                "state": self.state,
                "swing_leg": self.swing_leg,
                "desired_foot_pos": self.swing_target_pos,
                "phase": 1.0,
                "target_pos": self.swing_target_pos
            }

        elif self.state == SteppingState.DUAL_SUPPORT_SETTLE:
            # Stance re-centering
            if dt_state >= 0.20:
                self.state = SteppingState.QUIET_STANCE
                self.swing_leg = None
                
            return {
                "state": self.state,
                "swing_leg": None,
                "desired_foot_pos": self.swing_target_pos,
                "phase": 1.0,
                "target_pos": self.swing_target_pos
            }

        return {
            "state": SteppingState.QUIET_STANCE,
            "swing_leg": None,
            "desired_foot_pos": None,
            "phase": 0.0,
            "target_pos": None
        }

    def get_stance_compensation(self, swing_leg: str, phase: float, ctrl_nom: np.ndarray) -> np.ndarray:
        """
        Computes single-support stance leg weight-shift and lateral balance compensation.
        Shifts CoM over the stance foot sole via stance hip-roll adduction and knee stiffness.
        """
        stance_ctrl = ctrl_nom.copy()
        roll_shift = 0.050 * np.sin(np.pi * phase)
        knee_stiff = 0.060

        if swing_leg == "right":
            # Left leg is stance leg
            stance_ctrl[7] -= roll_shift   # Left hip roll tilts pelvis left over left sole
            stance_ctrl[11] += roll_shift  # Left ankle roll maintains flat sole contact
            stance_ctrl[9] += knee_stiff   # Left knee extension compensation for 2x body load
        else:
            # Right leg is stance leg
            stance_ctrl[1] += roll_shift   # Right hip roll tilts pelvis right over right sole
            stance_ctrl[5] -= roll_shift   # Right ankle roll maintains flat sole contact
            stance_ctrl[3] += knee_stiff   # Right knee extension compensation for 2x body load

        return stance_ctrl

    def inverse_kinematics_leg(self, hip_pos: np.ndarray, target_ankle_pos: np.ndarray) -> np.ndarray:
        """
        Analytical 2-link + spherical hip inverse kinematics for Cyrus 6-DOF leg.
        Target joint angles: [hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll]
        """
        d_vec = target_ankle_pos - hip_pos
        dx, dy, dz = d_vec[0], d_vec[1], d_vec[2]
        
        # Hip roll angle
        hip_roll = np.arctan2(dy, -dz)
        
        # Leg length in sagittal plane
        d_sag = np.sqrt(dx**2 + dz**2)
        L1 = self.p.thigh_len
        L2 = self.p.shin_len
        
        # Clip to reachability
        d_clamped = np.clip(d_sag, 0.15, (L1 + L2) * 0.999)
        
        # Law of cosines for knee angle
        cos_knee = (L1**2 + L2**2 - d_clamped**2) / (2.0 * L1 * L2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee = np.pi - np.arccos(cos_knee) # Knee bends forward in Cyrus (+ direction)
        
        # Hip pitch angle
        alpha = np.arctan2(dx, -dz)
        cos_beta = (L1**2 + d_clamped**2 - L2**2) / (2.0 * L1 * d_clamped)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        hip_pitch = -(alpha + beta) # Hip pitch tilts thigh
        
        # Ankle pitch to keep foot sole parallel to ground
        ankle_pitch = -(hip_pitch + knee)
        ankle_roll = -hip_roll
        hip_yaw = 0.0
        
        return np.array([hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll], dtype=np.float32)
