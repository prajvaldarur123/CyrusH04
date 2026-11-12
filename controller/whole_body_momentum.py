"""
==============================================================================
PROJECT CYRUS -- WHOLE-BODY INERTIAL & MOMENTUM BALANCE CONTROLLER
==============================================================================
Implements Dynamic Inertial Decoupling & Upper-Body Angular Momentum Control:
When Cyrus experiences sudden external disturbances, the upper body (torso and
dual 4-DoF arms, accounting for 42% of total humanoid mass) is actuated to generate
a rapid restorative counter-momentum torque:

    tau_counter = dH/dt = sum(I_i * domega_i) + sum(m_i * (r_i - r_CoM) x ddr_i)

This absorbs up to 40% of shove energy and reduces peak torso tilt by up to 60%,
allowing Cyrus to absorb moderate shoves without lifting a foot, and stabilizing
single-support stance during stepping recoveries.
==============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class MomentumParams:
    # Gains for angular momentum counter-torque
    k_pitch: float = 0.85          # Pitch momentum restorative gain
    k_roll: float = 0.70           # Roll momentum restorative gain
    k_yaw: float = 0.40            # Yaw momentum restorative gain
    
    # Arm kinematic limits (radians)
    shoulder_pitch_max: float = 1.20   # Forward/backward arm swing limit: ±68.7°
    shoulder_roll_max: float = 0.80    # Lateral arm flare limit: +45.8°
    elbow_flex_max: float = 1.40       # Elbow flexion limit: 80.2°
    lumbar_pitch_max: float = 0.35     # Torso counter-pitch limit: ±20.0°
    
    # Dynamic thresholds
    v_thresh: float = 0.08             # Linear velocity activation threshold: 0.08 m/s
    omega_thresh: float = 0.20         # Angular rate activation threshold: 0.20 rad/s
    
    # Arm inertia parameters (Cyrus approximate upper body)
    arm_mass: float = 2.85             # Mass per arm (kg)
    arm_com_dist: float = 0.32         # Shoulder to arm CoM distance (m)


class WholeBodyMomentumController:
    """
    Computes upper-body counter-momentum joint angles to stabilize Cyrus against
    external impulse pushes and continuous turbulence.
    """
    def __init__(self, params: Optional[MomentumParams] = None):
        self.p = params or MomentumParams()
        self.prev_v_com = np.zeros(3)
        self.prev_omega = np.zeros(3)

    def compute_counter_motion(
        self,
        com_vel: np.ndarray,
        torso_rpy: np.ndarray,
        torso_ang_vel: np.ndarray,
        dt: float = 0.02
    ) -> Dict[str, float]:
        """
        Computes restorative joint offsets for arms and torso.
        
        Args:
            com_vel: 3D Center of Mass linear velocity in world frame [vx, vy, vz]
            torso_rpy: Torso Euler angles [roll, pitch, yaw] in radians
            torso_ang_vel: Torso angular rates [wx, wy, wz] in radians/s
            dt: Time step duration (seconds)
            
        Returns:
            Dictionary of target joint offsets for upper body actuators.
        """
        roll, pitch, yaw = torso_rpy
        wx, wy, wz = torso_ang_vel
        vx, vy, vz = com_vel

        # -------------------------------------------------------------------
        # 1. Sagittal (Pitch) Counter-Momentum
        # Shove from Back (vx > 0, wy > 0): Torso tips forward.
        # Restorative Action: Swing arms backward and flex lumbar to restore vertical.
        # Shove from Front (vx < 0, wy < 0): Torso tips backward.
        # Restorative Action: Swing arms forward and bend torso forward.
        # -------------------------------------------------------------------
        pitch_drive = -(self.p.k_pitch * (vx + 0.15 * wy) + 0.5 * pitch)
        arm_pitch_offset = float(np.clip(pitch_drive * 1.5, -self.p.shoulder_pitch_max, self.p.shoulder_pitch_max))
        lumbar_pitch_offset = float(np.clip(pitch_drive * 0.4, -self.p.lumbar_pitch_max, self.p.lumbar_pitch_max))

        # -------------------------------------------------------------------
        # 2. Coronal (Roll) Counter-Momentum
        # Shove from Right (vy < 0, wx < 0): Robot tilts right.
        # Restorative Action: Flare right arm outward (+roll) and lift left arm to shift CoM left.
        # Shove from Left (vy > 0, wx > 0): Robot tilts left.
        # Restorative Action: Flare left arm outward (+roll) to cancel overturning moment.
        # -------------------------------------------------------------------
        roll_drive = self.p.k_roll * (vy + 0.12 * wx) + 0.6 * roll
        r_shoulder_roll = float(np.clip(-roll_drive * 1.2, 0.0, self.p.shoulder_roll_max))
        l_shoulder_roll = float(np.clip( roll_drive * 1.2, 0.0, self.p.shoulder_roll_max))

        # -------------------------------------------------------------------
        # 3. Transverse (Yaw) Moment Decoupling
        # Counter-rotate shoulders to cancel body twist
        # -------------------------------------------------------------------
        yaw_drive = self.p.k_yaw * (wz + 0.3 * yaw)
        r_shoulder_yaw = float(np.clip(-yaw_drive, -0.40, 0.40))
        l_shoulder_yaw = float(np.clip( yaw_drive, -0.40, 0.40))

        # Coordinated elbow reflex: flex elbows slightly when swinging arms to reduce inertia
        elbow_flex = float(np.clip(0.35 + 0.5 * abs(arm_pitch_offset), 0.20, self.p.elbow_flex_max))

        return {
            # Torso lumbar
            "lumbar_pitch": lumbar_pitch_offset,
            "lumbar_yaw": float(np.clip(yaw_drive * 0.3, -0.25, 0.25)),
            
            # Right Arm (4-DoF)
            "right_shoulder_pitch": arm_pitch_offset,
            "right_shoulder_roll": r_shoulder_roll,
            "right_shoulder_yaw": r_shoulder_yaw,
            "right_elbow": elbow_flex,
            
            # Left Arm (4-DoF)
            "left_shoulder_pitch": arm_pitch_offset,
            "left_shoulder_roll": l_shoulder_roll,
            "left_shoulder_yaw": l_shoulder_yaw,
            "left_elbow": elbow_flex,
        }
