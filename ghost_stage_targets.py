"""
Cyrus Humanoid - Ghost Stage Targets Module
Pre-computes the 3D joint landmark coordinates and transparent bone connections
for all 6 Standup Stages from the verified 100% invariant reference trajectory.
"""

import json
import os
import numpy as np
import mujoco

PROJECT_DIR = r"C:\Users\prajval\robotics"
DENSE_PATH = os.path.join(PROJECT_DIR, "cyrus", "ghost_trajectory_dense.json")

# Stage frame indices
STAGE_FRAMES = {
    0: (0,   "Stage 0: Prone Rest on Floor"),
    1: (78,  "Stage 1: Grounded Triceps Push-Up (Knees on Floor)"),
    2: (117, "Stage 1.5: High Plank (Straight Arms, Knees Lifted, Toes Locked)"),
    3: (160, "Stage 2: Athletic Bear Hover & Foot Step (Quadruped Standup Base)"),
    4: (278, "Stage 3: Squat Tuck & Floor Release (Weight on Dual Feet, Upright Guard)"),
    5: (320, "Stage 4: Quad Drive to Standing (Driving Through Heels)"),
    6: (400, "Stage 5: Full Upright Balance Lock (100% Upright Hold)")
}

def load_stage_targets():
    with open(DENSE_PATH, "r") as f:
        dense = json.load(f)

    targets = {}
    for s_idx, (f_idx, s_name) in STAGE_FRAMES.items():
        j = dense[f_idx]["joints"]
        targets[s_idx] = {
            "name": s_name,
            "pelvis": np.array(j["ghost_pelvis"], dtype=np.float64),
            "torso": np.array(j["ghost_torso"], dtype=np.float64),
            "head": np.array(j["ghost_head"], dtype=np.float64),
            "r_palm": np.array(j["ghost_r_palm"], dtype=np.float64),
            "l_palm": np.array(j["ghost_l_palm"], dtype=np.float64),
            "r_elbow": np.array(j["ghost_r_elbow"], dtype=np.float64),
            "l_elbow": np.array(j["ghost_l_elbow"], dtype=np.float64),
            "r_knee": np.array(j["ghost_r_knee"], dtype=np.float64),
            "l_knee": np.array(j["ghost_l_knee"], dtype=np.float64),
            "r_foot": np.array(j["ghost_r_foot"], dtype=np.float64),
            "l_foot": np.array(j["ghost_l_foot"], dtype=np.float64),
            "r_hip": np.array(j["ghost_r_hip"], dtype=np.float64),
            "l_hip": np.array(j["ghost_l_hip"], dtype=np.float64),
            "r_shoulder": np.array(j["ghost_r_shoulder"], dtype=np.float64),
            "l_shoulder": np.array(j["ghost_l_shoulder"], dtype=np.float64)
        }
    return targets

# Color Palette for Ghost Balls & Transparent Parts
C_HEAD    = np.array([0.0, 1.0, 1.0, 0.90])   # Cyan
C_TORSO   = np.array([0.0, 0.85, 0.95, 0.90]) # Bright Cyan
C_PELVIS  = np.array([0.1, 0.9, 0.5, 0.90])  # Emerald Green
C_PALM    = np.array([1.0, 0.8, 0.1, 0.95])   # Gold / Yellow
C_ELBOW   = np.array([1.0, 0.5, 0.1, 0.90])   # Orange
C_KNEE    = np.array([0.95, 0.25, 0.65, 0.90])# Pink / Magenta
C_FOOT    = np.array([0.1, 0.9, 0.5, 0.90])   # Emerald Green
C_BONE    = np.array([0.2, 0.70, 1.0, 0.35])  # Semi-Transparent Glowing Cyan / Blue

def draw_stage_ghost(scn, target_dict, origin_offset=np.zeros(3)):
    """
    Renders 11 ghost joint balls and 12 transparent connecting capsules
    into the MuJoCo viewer user_scn at origin_offset.
    """
    scn.ngeom = 0  # reset custom geometry buffer
    
    t = target_dict
    off = origin_offset
    
    p_head   = t["head"] + off
    p_torso  = t["torso"] + off
    p_pelvis = t["pelvis"] + off
    p_r_palm = t["r_palm"] + off
    p_l_palm = t["l_palm"] + off
    p_r_elb  = t["r_elbow"] + off
    p_l_elb  = t["l_elbow"] + off
    p_r_knee = t["r_knee"] + off
    p_l_knee = t["l_knee"] + off
    p_r_foot = t["r_foot"] + off
    p_l_foot = t["l_foot"] + off
    p_r_sh   = t["r_shoulder"] + off
    p_l_sh   = t["l_shoulder"] + off
    p_r_hip  = t["r_hip"] + off
    p_l_hip  = t["l_hip"] + off

    # 1. Ghost Joint Balls (Spheres)
    spheres = [
        (p_head,   0.075, C_HEAD),
        (p_torso,  0.080, C_TORSO),
        (p_pelvis, 0.085, C_PELVIS),
        (p_r_palm, 0.045, C_PALM),
        (p_l_palm, 0.045, C_PALM),
        (p_r_elb,  0.040, C_ELBOW),
        (p_l_elb,  0.040, C_ELBOW),
        (p_r_knee, 0.048, C_KNEE),
        (p_l_knee, 0.048, C_KNEE),
        (p_r_foot, 0.045, C_FOOT),
        (p_l_foot, 0.045, C_FOOT),
    ]

    for pos, radius, color in spheres:
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0, 0]),
                pos,
                np.eye(3).flatten(),
                color
            )
            scn.ngeom += 1

    # 2. Transparent Limbs / Parts (Capsules)
    limbs = [
        (p_pelvis, p_torso,  0.055),  # Lower Spine
        (p_torso,  p_head,   0.045),  # Upper Spine / Neck
        (p_torso,  p_r_sh,   0.038),  # Right Clavicle
        (p_torso,  p_l_sh,   0.038),  # Left Clavicle
        (p_r_sh,   p_r_elb,  0.040),  # Right Upper Arm
        (p_l_sh,   p_l_elb,  0.040),  # Left Upper Arm
        (p_r_elb,  p_r_palm, 0.035),  # Right Forearm
        (p_l_elb,  p_l_palm, 0.035),  # Left Forearm
        (p_r_hip,  p_r_knee, 0.046),  # Right Thigh
        (p_l_hip,  p_l_knee, 0.046),  # Left Thigh
        (p_r_knee, p_r_foot, 0.042),  # Right Shin
        (p_l_knee, p_l_foot, 0.042),  # Left Shin
    ]

    for a, b, width in limbs:
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
            g.rgba[:] = C_BONE
            scn.ngeom += 1

if __name__ == "__main__":
    t = load_stage_targets()
    print(f"Loaded {len(t)} stage targets successfully:")
    for idx, data in t.items():
        print(f"  Stage {idx}: {data['name']}")
