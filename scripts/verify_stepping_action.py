import os
import sys
import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTICS_ROOT = os.path.dirname(PROJECT_ROOT)
for p in [ROBOTICS_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from controller.balance_pipeline import (
    EMAActionFilter,
    get_robot_mass,
    verify_settled_equilibrium,
    ACTION_SCALE_VEC,
    EMA_ALPHA,
    ROBUST_ROLL_GAINS
)

XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "candidate_balance_active.npz")

m = mujoco.MjModel.from_xml_path(XML_PATH)
m.opt.timestep = 0.0025

for side in ["right", "left"]:
    m.actuator(f"{side}_hip_roll_act").gainprm[0] = ROBUST_ROLL_GAINS["hip_roll_kp"]
    m.actuator(f"{side}_hip_roll_act").biasprm[1] = -ROBUST_ROLL_GAINS["hip_roll_kp"]
    m.actuator(f"{side}_hip_roll_act").biasprm[2] = -ROBUST_ROLL_GAINS["hip_roll_kv"]
    m.actuator(f"{side}_ankle_roll_act").gainprm[0] = ROBUST_ROLL_GAINS["ankle_roll_kp"]
    m.actuator(f"{side}_ankle_roll_act").biasprm[1] = -ROBUST_ROLL_GAINS["ankle_roll_kp"]
    m.actuator(f"{side}_ankle_roll_act").biasprm[2] = -ROBUST_ROLL_GAINS["ankle_roll_kv"]

d = mujoco.MjData(m)
pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
r_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
l_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

leg_qpos_adr = [m.jnt(name).qposadr[0] for name in [
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
]]
leg_qvel_adr = [m.jnt(name).dofadr[0] for name in [
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"
]]

ctrl_nom = np.zeros(12, dtype=np.float32)
ctrl_nom[2] = -0.175
ctrl_nom[3] = 0.350
ctrl_nom[4] = -0.175
ctrl_nom[8] = -0.175
ctrl_nom[9] = 0.350
ctrl_nom[10] = -0.175

weights = np.load(CKPT_PATH)

def policy_forward(obs_buf):
    h1 = np.tanh(obs_buf @ weights["mlp_extractor.policy_net.0.weight"].T + weights["mlp_extractor.policy_net.0.bias"])
    h2 = np.tanh(h1 @ weights["mlp_extractor.policy_net.2.weight"].T + weights["mlp_extractor.policy_net.2.bias"])
    raw = h2 @ weights["action_net.weight"].T + weights["action_net.bias"]
    return np.clip(raw, -1.0, 1.0)

def get_obs(prev_act):
    R = d.xmat[torso_id].reshape(3, 3)
    g = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cv = d.cvel[torso_id]
    ang_vel = R.T @ cv[:3]
    lin_vel = R.T @ cv[3:]
    leg_qpos = np.array([d.qpos[a] for a in leg_qpos_adr], dtype=np.float32)
    leg_qvel = np.array([d.qvel[a] for a in leg_qvel_adr], dtype=np.float32)
    r_sole_z = float(d.site_xpos[r_sole_id, 2])
    l_sole_z = float(d.site_xpos[l_sole_id, 2])
    r_contact = 1.0 if r_sole_z < 0.045 else 0.0
    l_contact = 1.0 if l_sole_z < 0.045 else 0.0
    return np.concatenate([
        g, ang_vel, lin_vel, leg_qpos, leg_qvel * 0.1, prev_act, [r_contact, l_contact]
    ]).astype(np.float32)

print("=" * 80)
print(f"VERIFYING STEPPING MOTION WITH LATEST WEIGHTS: {os.path.basename(CKPT_PATH)}")
print("=" * 80)

for angle in [0.0, 90.0, 180.0]:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.825
    d.qpos[3] = 1.0
    for i, adr in enumerate(leg_qpos_adr):
        d.qpos[adr] = float(ctrl_nom[i])
    d.qvel[:] = 0.0
    d.ctrl[:] = ctrl_nom.copy()
    verify_settled_equilibrium(m, d, ctrl_nom, tol_fz=1.0, tol_qvel=1e-3, max_steps=400)

    ema = EMAActionFilter(alpha=EMA_ALPHA, dim=12)
    prev_act = np.zeros(12, dtype=np.float32)
    obs_buf = np.tile(get_obs(prev_act), 3)

    push_mag = 0.70
    pulse_dur = 0.040
    pulse_force = (get_robot_mass(m) * push_mag) / pulse_dur
    fx = pulse_force * np.cos(np.radians(angle))
    fy = pulse_force * np.sin(np.radians(angle))

    max_r_sole_z = 0.0
    max_l_sole_z = 0.0
    max_asym_knee = 0.0
    max_asym_pitch = 0.0

    for step in range(250):  # 5 seconds
        act = policy_forward(obs_buf)
        act_filt = ema.update(act)
        prev_act = act_filt.copy()

        # Check asymmetric differential between right and left leg!
        asym_knee = abs(act_filt[3] - act_filt[9])
        asym_pitch = abs(act_filt[2] - act_filt[8])
        if asym_knee > max_asym_knee: max_asym_knee = asym_knee
        if asym_pitch > max_asym_pitch: max_asym_pitch = asym_pitch

        target = np.clip(ctrl_nom + act_filt * ACTION_SCALE_VEC, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        d.ctrl[:] = target

        for sub in range(8):
            cur_t = step * 0.02 + sub * 0.0025
            if 2.0 <= cur_t < 2.0 + pulse_dur:
                d.xfrc_applied[pelvis_id, 0] = fx
                d.xfrc_applied[pelvis_id, 1] = fy
            else:
                d.xfrc_applied[pelvis_id, 0] = 0.0
                d.xfrc_applied[pelvis_id, 1] = 0.0
            mujoco.mj_step(m, d)

        rz = float(d.site_xpos[r_sole_id, 2])
        lz = float(d.site_xpos[l_sole_id, 2])
        if rz > max_r_sole_z: max_r_sole_z = rz
        if lz > max_l_sole_z: max_l_sole_z = lz

        obs = get_obs(prev_act)
        obs_buf = np.concatenate([obs, obs_buf[:94]])

    print(f"\n--- Push Angle {angle:5.1f}° (Force: {pulse_force:6.1f} N at 0.70 m/s) ---")
    print(f"  Max Right Foot Height: {max_r_sole_z*100.0:5.2f} cm (Nominal contact: ~3.8 cm)")
    print(f"  Max Left Foot Height:  {max_l_sole_z*100.0:5.2f} cm (Nominal contact: ~3.8 cm)")
    print(f"  Max Knee Asymmetry:    {np.degrees(max_asym_knee * 0.25):5.2f}° differential between legs")
    print(f"  Max Hip Pitch Asym:    {np.degrees(max_asym_pitch * 0.25):5.2f}° differential between legs")
    is_asymmetric = (max_asym_knee > 0.01 or max_asym_pitch > 0.01)
    print(f"  --> Asymmetric Independent Leg Motion: {'ACTIVE [✓]' if is_asymmetric else 'SYMMETRIC [X]'}")
