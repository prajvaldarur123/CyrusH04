import numpy as np


def compute_mechanical_work(torques: np.ndarray, joint_vels: np.ndarray, dt: float) -> float:
    """
    Computes total mechanical work done by actuators:
    E_mech = \int \sum_i |\tau_i(t) * \dot{q}_i(t)| dt (Joules)
    torques: (T, nu) array of applied joint torques
    joint_vels: (T, nu) array of joint velocities in rad/s
    """
    if len(torques) == 0:
        return 0.0
    power = np.abs(torques * joint_vels)  # (T, nu)
    instantaneous_power = np.sum(power, axis=1)  # (T,)
    return float(np.sum(instantaneous_power) * dt)


def compute_mechanical_effort(torques: np.ndarray, dt: float) -> float:
    """
    Computes total mechanical effort (torque squared integral):
    Effort = \int \sum_i \tau_i(t)^2 dt (N^2 * s)
    """
    if len(torques) == 0:
        return 0.0
    torque_sq = np.sum(np.square(torques), axis=1)
    return float(np.sum(torque_sq) * dt)


def compute_step_symmetry(left_step_durations: list, right_step_durations: list) -> float:
    """
    Computes Step Symmetry Index (SI %):
    SI = |T_left_mean - T_right_mean| / (0.5 * (T_left_mean + T_right_mean)) * 100%
    0% = perfect symmetry.
    """
    if not left_step_durations or not right_step_durations:
        return 100.0  # complete asymmetry if one leg never stepped
    mean_l = float(np.mean(left_step_durations))
    mean_r = float(np.mean(right_step_durations))
    denom = 0.5 * (mean_l + mean_r)
    if denom <= 1e-6:
        return 0.0
    return float((abs(mean_l - mean_r) / denom) * 100.0)


def compute_foot_slipping(sole_vels: np.ndarray, normal_forces: np.ndarray, force_threshold: float = 40.0) -> dict:
    """
    Computes foot slipping velocity during the stance phase (when normal force exceeds threshold).
    sole_vels: (T, 3) linear velocity of sole site
    normal_forces: (T,) vertical contact force
    Returns mean and max slip velocity in m/s during stance.
    """
    stance_mask = normal_forces > force_threshold
    if not np.any(stance_mask):
        return {"mean_slip_vel": 0.0, "max_slip_vel": 0.0}
    slip_vels = np.linalg.norm(sole_vels[stance_mask, :2], axis=1)  # horizontal xy velocity
    return {
        "mean_slip_vel": float(np.mean(slip_vels)),
        "max_slip_vel": float(np.max(slip_vels))
    }


def compute_joint_jerk(joint_positions: np.ndarray, dt: float) -> float:
    """
    Computes dimensionless joint jerk as a proxy for high-frequency oscillation:
    Jerk = mean of ||\dddot{q}|| over time.
    """
    if len(joint_positions) < 4:
        return 0.0
    vel = np.gradient(joint_positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def compute_joint_limit_violations(joint_positions: np.ndarray, joint_limits: np.ndarray) -> dict:
    """
    Checks for joint limit violations.
    joint_positions: (T, njnt)
    joint_limits: (njnt, 2) [min, max]
    """
    violations = 0
    max_penetration = 0.0
    for j in range(joint_positions.shape[1]):
        q_min, q_max = joint_limits[j]
        below = q_min - joint_positions[:, j]
        above = joint_positions[:, j] - q_max
        pen_below = np.maximum(0.0, below)
        pen_above = np.maximum(0.0, above)
        v_count = np.sum(pen_below > 1e-4) + np.sum(pen_above > 1e-4)
        violations += int(v_count)
        max_pen_j = max(float(np.max(pen_below)), float(np.max(pen_above)))
        if max_pen_j > max_penetration:
            max_penetration = max_pen_j
    return {
        "violation_count": violations,
        "max_penetration_rad": max_penetration
    }


def compute_capture_point(com_pos: np.ndarray, com_vel: np.ndarray, g: float = 9.80665) -> np.ndarray:
    """
    Computes Instantaneous Capture Point (ICP) based on Linear Inverted Pendulum Model:
    x_cp = x_com + v_com / omega_0, where omega_0 = sqrt(g / z_com)
    """
    z_com = max(0.1, float(com_pos[2]))
    omega_0 = np.sqrt(g / z_com)
    x_cp = com_pos[:2] + com_vel[:2] / omega_0
    return x_cp


def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """
    Converts 3x3 rotation matrix to (roll, pitch, yaw) in radians.
    R: (3, 3) rotation matrix
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(R[2, 0]) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)
