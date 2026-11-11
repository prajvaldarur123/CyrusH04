"""
==============================================================================
PROJECT CYRUS -- CENTRALIZED BALANCE PIPELINE & RIGOROUS HARNESS
==============================================================================
Centralizes:
1. Cryptographic manifest loading and atomic checkpoint persistence
2. Bilateral symmetry enforcement & Exponential Moving Average (EMA) filtering
3. Joint-specific action scaling (ACTION_SCALE_VEC)
4. Dynamic settled-start verification (Fz == M*g within tolerance)
5. Rigid-body CoM, CoP, Centroidal Moment Pivot (CMP), and momentum metrics
6. Standardized fall detection criteria across all benchmark/test scripts
==============================================================================
"""

import os
import json
import hashlib
import numpy as np
import mujoco

# ============================================================================
# CANONICAL CONSTANTS & ACTION VECTORS
# ============================================================================

ACTION_SCALE_VEC = np.array([
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10,  # Right: yaw, roll, pitch, knee, ank_pitch, ank_roll
    0.10, 0.10, 0.25, 0.25, 0.20, 0.10   # Left:  yaw, roll, pitch, knee, ank_pitch, ank_roll
], dtype=np.float32)

ACTION_SCALE_FLAT = np.full(12, 0.14, dtype=np.float32)

EMA_ALPHA = 0.65

# Nominal Poses (12-DoF lower body)
# Order: [r_yaw, r_roll, r_pitch, r_knee, r_ank_pitch, r_ank_roll,
#         l_yaw, l_roll, l_pitch, l_knee, l_ank_pitch, l_ank_roll]

POSE_A = np.array([
    0.0, 0.0, -0.04,  0.05, -0.01, 0.0,
    0.0, 0.0, -0.04,  0.05, -0.01, 0.0
], dtype=np.float32)

POSE_C = np.array([
    0.0, 0.0, -0.175, 0.350, -0.175, 0.0,
    0.0, 0.0, -0.175, 0.350, -0.175, 0.0
], dtype=np.float32)

# Recommended robust roll gains (Pose C / Stiff Roll)
ROBUST_ROLL_GAINS = {
    "hip_roll_kp": 500.0,
    "hip_roll_kv": 35.0,
    "ankle_roll_kp": 400.0,
    "ankle_roll_kv": 25.0
}

STANDARD_FALL_THRESHOLDS = {
    "min_pelvis_z": 0.55,
    "max_pitch_deg": 30.0,
    "max_roll_deg": 30.0
}


# ============================================================================
# 1. MANIFEST & ATOMIC CHECKPOINT PROTOCOL
# ============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_verified_checkpoint(manifest_path: str, checkpoint_key: str = "reference_baseline"):
    """
    Loads checkpoint file verified against checkpoints/manifest.json.
    Supports top-level keys (e.g. 'reference_baseline') or nested candidate keys ('legacy_production').
    Raises RuntimeError immediately if hash mismatch or file not registered.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"SECURITY AUDIT ERROR: Manifest missing at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entry = None
    if checkpoint_key in manifest:
        entry = manifest[checkpoint_key]
    elif "candidate_policies" in manifest and checkpoint_key in manifest["candidate_policies"]:
        entry = manifest["candidate_policies"][checkpoint_key]
    elif checkpoint_key == "active_production":
        entry = manifest.get("reference_baseline")

    if not entry:
        raise KeyError(f"SECURITY AUDIT ERROR: Key '{checkpoint_key}' not registered in manifest {manifest_path}")

    filename = entry["file"]
    expected_hash = entry["sha256"]

    ckpt_dir = os.path.dirname(manifest_path)
    file_path = os.path.join(ckpt_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SECURITY AUDIT ERROR: Checkpoint file missing: {file_path}")

    actual_hash = compute_file_sha256(file_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SECURITY AUDIT FAILURE: Checkpoint hash mismatch for {filename}!\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Actual SHA-256:   {actual_hash}\n"
            f"Execution halted to prevent loading corrupted or tampered neural weights."
        )

    data = np.load(file_path)
    return data, entry


def save_checkpoint_atomic(target_path: str, arrays_dict: dict):
    """
    Saves an NPZ file atomically using temporary file, fsync, and atomic replace.
    Guarantees OS-level disk flush before renaming to prevent corrupted weights.
    """
    base, ext = os.path.splitext(target_path)
    extension = ext if ext else ".npz"
    tmp_path = f"{base}_tmp_{os.getpid()}{extension}"
    
    np.savez(tmp_path, **arrays_dict)
    actual_tmp = tmp_path if tmp_path.endswith(".npz") else tmp_path + ".npz"
    
    # Flush and fsync to disk before atomic rename
    with open(actual_tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
        
    os.replace(actual_tmp, target_path)


# ============================================================================
# 2. CONTROL TRANSFORMS & EMA FILTER
# ============================================================================

def enforce_bilateral_symmetry(raw_action: np.ndarray) -> np.ndarray:
    """
    Enforces sagittal mirror symmetry across 12-DoF bipedal joints.
    Right leg: [yaw, roll, pitch, knee, ank_pitch, ank_roll]
    Left leg:  [-yaw, roll, pitch, knee, ank_pitch, ank_roll]
    """
    sym_pitch_hip = 0.5 * (raw_action[..., 2] + raw_action[..., 8])
    sym_knee      = 0.5 * (raw_action[..., 3] + raw_action[..., 9])
    sym_pitch_ank = 0.5 * (raw_action[..., 4] + raw_action[..., 10])
    anti_yaw_hip  = 0.5 * (raw_action[..., 0] - raw_action[..., 6])
    sym_roll_hip  = 0.5 * (raw_action[..., 1] + raw_action[..., 7])
    sym_roll_ank  = 0.5 * (raw_action[..., 5] + raw_action[..., 11])

    right_leg = np.stack([anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    left_leg  = np.stack([-anti_yaw_hip, sym_roll_hip, sym_pitch_hip, sym_knee, sym_pitch_ank, sym_roll_ank], axis=-1)
    return np.concatenate([right_leg, left_leg], axis=-1)


class EMAActionFilter:
    """Stateful Exponential Moving Average (EMA) action smoother."""
    def __init__(self, alpha: float = EMA_ALPHA, dim: int = 12):
        self.alpha = float(alpha)
        self.dim = dim
        self.filtered_action = np.zeros(dim, dtype=np.float32)

    def reset(self):
        self.filtered_action.fill(0.0)

    def update(self, action: np.ndarray) -> np.ndarray:
        self.filtered_action = self.alpha * action + (1.0 - self.alpha) * self.filtered_action
        return self.filtered_action.copy()


# ============================================================================
# 3. PHYSICAL IDENTIFICATION & SETTLED EQUILIBRIUM
# ============================================================================

def get_robot_body_ids(model: mujoco.MjModel):
    """Returns list of body IDs excluding world and mocap bodies (e.g. gaze_target)."""
    return [b for b in range(model.nbody) if model.body_mocapid[b] == -1 and b != 0]


def get_robot_mass(model: mujoco.MjModel) -> float:
    """Computes true mass of Cyrus excluding mocap targets."""
    return float(np.sum([model.body_mass[b] for b in get_robot_body_ids(model)]))


def get_robot_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Computes true whole-body CoM position (3D) excluding mocap bodies."""
    body_ids = get_robot_body_ids(model)
    m_tot = get_robot_mass(model)
    return np.sum([model.body_mass[b] * data.xipos[b] for b in body_ids], axis=0) / m_tot


def measure_ground_wrench(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Measures total ground reaction force (Fz, Fx, Fy) and ground CoP.
    Returns: (fz_tot, fx_tot, fy_tot, cop_x, cop_y)
    """
    c_forces = np.zeros(6)
    fz_tot = 0.0
    fx_tot = 0.0
    fy_tot = 0.0
    sum_x_fz = 0.0
    sum_y_fz = 0.0

    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, c_forces)
        R = con.frame.reshape(3, 3)
        fw = R.T @ c_forces[:3]
        if fw[2] > 0.01:
            fz_tot += fw[2]
            fx_tot += fw[0]
            fy_tot += fw[1]
            sum_x_fz += con.pos[0] * fw[2]
            sum_y_fz += con.pos[1] * fw[2]

    cop_x = (sum_x_fz / fz_tot) if fz_tot > 1.0 else 0.0
    cop_y = (sum_y_fz / fz_tot) if fz_tot > 1.0 else 0.0
    return fz_tot, fx_tot, fy_tot, cop_x, cop_y


def verify_settled_equilibrium(model: mujoco.MjModel, data: mujoco.MjData,
                                ctrl: np.ndarray, tol_fz: float = 1.0,
                                tol_qvel: float = 1e-3, max_steps: int = 1000):
    """
    Steps the simulation until physical equilibrium is proven:
    1. Total normal force Fz matches robot weight within tol_fz
    2. Maximum joint velocity |qvel| is below tol_qvel
    Returns: (is_settled, step_count, fz_settled, max_qvel)
    """
    weight = get_robot_mass(model) * 9.80665

    for step in range(max_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        
        fz, _, _, _, _ = measure_ground_wrench(model, data)
        max_v = float(np.max(np.abs(data.qvel)))
        
        if abs(fz - weight) <= tol_fz and max_v <= tol_qvel and step >= 100:
            return True, step, fz, max_v

    fz, _, _, _, _ = measure_ground_wrench(model, data)
    max_v = float(np.max(np.abs(data.qvel)))
    return False, max_steps, fz, max_v


# ============================================================================
# 4. CENTROIDAL MOMENT PIVOT & ANGULAR MOMENTUM
# ============================================================================

def compute_centroidal_angular_momentum(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Computes exact whole-body angular momentum L_com about the center of mass.
    L = sum (m_b * (r_b - r_com) x v_b + I_b_world * w_b)
    """
    body_ids = get_robot_body_ids(model)
    m_tot = get_robot_mass(model)
    r_com = np.sum([model.body_mass[b] * data.xipos[b] for b in body_ids], axis=0) / m_tot

    L = np.zeros(3, dtype=np.float64)
    for b in body_ids:
        mass = model.body_mass[b]
        r = data.xipos[b] - r_com
        w_b = data.cvel[b, :3]
        v_b = data.cvel[b, 3:]
        L += mass * np.cross(r, v_b)

        R = data.ximat[b].reshape(3, 3)
        I_diag = model.body_inertia[b]
        I_world = R @ np.diag(I_diag) @ R.T
        L += I_world @ w_b

    return r_com, L


def compute_cmp_and_margins(model: mujoco.MjModel, data: mujoco.MjData,
                            r_foot_id: int, l_foot_id: int):
    """
    Computes Center of Mass, CoP, Centroidal Moment Pivot (CMP), and instantaneous
    Capture Point relative to ankle joint center.
    """
    r_com = get_robot_com(model, data)
    ank_x = 0.5 * (data.xpos[r_foot_id, 0] + data.xpos[l_foot_id, 0])
    ank_y = 0.5 * (data.xpos[r_foot_id, 1] + data.xpos[l_foot_id, 1])

    fz, fx, fy, cop_x_w, cop_y_w = measure_ground_wrench(model, data)

    # CoP relative to ankle
    cop_x_ank = (cop_x_w - ank_x) * 100.0  # cm
    cop_y_ank = (cop_y_w - ank_y) * 100.0  # cm

    # CMP relative to ankle
    if fz > 1.0:
        cmp_x_w = r_com[0] - (r_com[2] / fz) * fx
        cmp_y_w = r_com[1] - (r_com[2] / fz) * fy
        cmp_x_ank = (cmp_x_w - ank_x) * 100.0  # cm
        cmp_y_ank = (cmp_y_w - ank_y) * 100.0  # cm
    else:
        cmp_x_ank, cmp_y_ank = 0.0, 0.0

    return {
        "com_x_cm": (r_com[0] - ank_x) * 100.0,
        "com_y_cm": (r_com[1] - ank_y) * 100.0,
        "com_z_m": r_com[2],
        "cop_x_cm": cop_x_ank,
        "cop_y_cm": cop_y_ank,
        "cmp_x_cm": cmp_x_ank,
        "cmp_y_cm": cmp_y_ank,
        "fz_N": fz,
        "fx_N": fx,
        "fy_N": fy
    }


# ============================================================================
# 5. STANDARDIZED FALL DETECTOR
# ============================================================================

def check_zero_action_gate(checkpoint_data, obs_dim: int = 141, batch_size: int = 64) -> tuple:
    """
    A7 Zero-Action Gate Rule:
    Evaluates policy weights and verifies that the model is NOT a dummy/quiescent baseline.
    Rejects candidates where:
    1. Action layer weights/biases have norm < 1e-5
    2. Policy output variance across a random test batch is < 1e-6
    Returns: (passes_gate: bool, action_norm: float, action_var: float, reason: str)
    """
    w = checkpoint_data
    if "action_net.weight" not in w:
        return False, 0.0, 0.0, "Missing 'action_net.weight' in checkpoint"

    w_act = w["action_net.weight"]
    b_act = w["action_net.bias"]
    action_norm = float(np.linalg.norm(w_act) + np.linalg.norm(b_act))

    if action_norm < 1e-5:
        return False, action_norm, 0.0, f"Action layer norm ({action_norm:.6e}) is ZERO (quiescent baseline)"

    # Test forward variance across synthetic batch
    rng = np.random.RandomState(42)
    obs_batch = rng.randn(batch_size, obs_dim).astype(np.float32)

    h1 = np.tanh(obs_batch @ w["mlp_extractor.policy_net.0.weight"].T + w["mlp_extractor.policy_net.0.bias"])
    h2 = np.tanh(h1 @ w["mlp_extractor.policy_net.2.weight"].T + w["mlp_extractor.policy_net.2.bias"])
    actions = h2 @ w_act.T + b_act
    action_var = float(np.mean(np.var(actions, axis=0)))

    if action_var < 1e-6:
        return False, action_norm, action_var, f"Action variance ({action_var:.6e}) is CONSTANT (dead policy)"

    return True, action_norm, action_var, "PROMOTABLE (Active neural policy)"


def is_robot_fallen(model: mujoco.MjModel, data: mujoco.MjData, thresholds: dict = None) -> tuple:
    """
    Standardized fall detection check across all testing harnesses:
    1. Pelvis height Z below min threshold (0.55m)
    2. Pelvis roll or pitch exceeding max angle (30 deg)
    3. Non-foot contact check: thighs, pelvis, torso, or knees touching floor
    Returns: (is_fall, reason)
    """
    th = thresholds or STANDARD_FALL_THRESHOLDS
    z = float(data.qpos[2])
    if z < th["min_pelvis_z"]:
        return True, f"Pelvis height ({z:.3f}m) below {th['min_pelvis_z']}m"

    qw, qx, qy, qz = data.qpos[3:7]
    roll = float(np.degrees(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))))
    pitch = float(np.degrees(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))))

    if abs(roll) > th["max_roll_deg"]:
        return True, f"Pelvis roll ({roll:.1f}deg) exceeded {th['max_roll_deg']}deg"
    if abs(pitch) > th["max_pitch_deg"]:
        return True, f"Pelvis pitch ({pitch:.1f}deg) exceeded {th['max_pitch_deg']}deg"

    # Non-foot body contact check with floor (geom 0)
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        # Ignore foot bodies (right_foot, left_foot) and world body (0)
        for b in [b1, b2]:
            if b != 0:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                if body_name not in ["right_foot", "left_foot", "gaze_target"]:
                    # Non-foot body touched ground or collided!
                    if con.geom1 == 0 or con.geom2 == 0:
                        return True, f"Non-foot body collision ({body_name}) with floor"

    return False, "STABLE"
