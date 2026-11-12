import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("C:/Users/prajval/robotics/cyrus/models/cyrus.xml")
data = mujoco.MjData(model)

def get_com(m, d):
    mujoco.mj_forward(m, d)
    masses = m.body_mass[1:]
    positions = d.xipos[1:]
    return np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)

# Joints
r_hp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_hip_pitch')
l_hp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_pitch')
r_kn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_knee')
l_kn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_knee')
r_ap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_ankle_pitch')
l_ap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_ankle_pitch')
r_sp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_shoulder_pitch')
l_sp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_pitch')
r_el = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_elbow')
l_el = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_elbow')

# 1. Default XML Configuration (as loaded from cyrus.xml)
mujoco.mj_resetData(model, data)
com_default = get_com(model, data)
pz_default = data.qpos[2]

# 2. Athletic Stance (Pose C - from training/train_balance.py and controller/balance_pipeline.py)
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.825
data.qpos[model.jnt_qposadr[r_hp]] = -0.175
data.qpos[model.jnt_qposadr[l_hp]] = -0.175
data.qpos[model.jnt_qposadr[r_kn]] = 0.350
data.qpos[model.jnt_qposadr[l_kn]] = 0.350
data.qpos[model.jnt_qposadr[r_ap]] = -0.175
data.qpos[model.jnt_qposadr[l_ap]] = -0.175
data.qpos[model.jnt_qposadr[r_sp]] = 0.05
data.qpos[model.jnt_qposadr[l_sp]] = 0.05
data.qpos[model.jnt_qposadr[r_el]] = -0.25
data.qpos[model.jnt_qposadr[l_el]] = -0.25
com_pose_c = get_com(model, data)
pz_pose_c = data.qpos[2]

# 3. Full Standing Lockout (Pose A / Straight Legs)
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.840
data.qpos[model.jnt_qposadr[r_hp]] = 0.0
data.qpos[model.jnt_qposadr[l_hp]] = 0.0
data.qpos[model.jnt_qposadr[r_kn]] = 0.0
data.qpos[model.jnt_qposadr[l_kn]] = 0.0
data.qpos[model.jnt_qposadr[r_ap]] = 0.0
data.qpos[model.jnt_qposadr[l_ap]] = 0.0
com_lockout = get_com(model, data)
pz_lockout = data.qpos[2]

print(f"1. Default XML Configuration:")
print(f"   Pelvis Z = {pz_default:.4f} m | CoM = ({com_default[0]:.6f}, {com_default[1]:.6f}, {com_default[2]:.6f}) m")
print(f"2. Athletic Stance (Pose C):")
print(f"   Pelvis Z = {pz_pose_c:.4f} m | CoM = ({com_pose_c[0]:.6f}, {com_pose_c[1]:.6f}, {com_pose_c[2]:.6f}) m")
print(f"3. Full Standing Lockout:")
print(f"   Pelvis Z = {pz_lockout:.4f} m | CoM = ({com_lockout[0]:.6f}, {com_lockout[1]:.6f}, {com_lockout[2]:.6f}) m")
