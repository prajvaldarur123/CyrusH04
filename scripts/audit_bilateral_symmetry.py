import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("models/cyrus.xml")

# 1. Body Pairs
body_pairs = [
    ('right_shoulder', 'left_shoulder'),
    ('right_upper_arm', 'left_upper_arm'),
    ('right_forearm', 'left_forearm'),
    ('right_hand', 'left_hand'),
    ('right_hip', 'left_hip'),
    ('right_thigh', 'left_thigh'),
    ('right_shin', 'left_shin'),
    ('right_foot', 'left_foot')
]

# Add fingers
for finger in ['thumb_proximal', 'thumb_distal', 'index_proximal', 'index_intermediate', 'index_distal',
               'middle_proximal', 'middle_intermediate', 'middle_distal', 'ring_proximal', 'ring_intermediate',
               'ring_distal', 'pinky_proximal', 'pinky_intermediate', 'pinky_distal']:
    body_pairs.append((f"right_{finger}", f"left_{finger}"))

print("="*85)
print("1. BILATERAL BODY MASS COMPARISON")
print("="*85)
print(f"{'Body Link':<28} | {'Right Mass (kg)':<15} | {'Left Mass (kg)':<15} | {'Delta (kg)':<12} | {'Status'}")
print("-" * 85)
all_mass_pass = True
for r_name, l_name in body_pairs:
    r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r_name)
    l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l_name)
    m_r = m.body_mass[r_id]
    m_l = m.body_mass[l_id]
    delta = abs(m_r - m_l)
    status = "PASS" if delta < 1e-4 else "FAIL"
    if status == "FAIL": all_mass_pass = False
    print(f"{r_name.replace('right_', ''):<28} | {m_r:15.5f} | {m_l:15.5f} | {delta:12.4e} | {status}")

print("\n" + "="*85)
print("2. INERTIAL POSITION AUDIT (Symmetry: x_R = x_L, y_R = -y_L, z_R = z_L)")
print("="*85)
print(f"{'Body Link':<22} | {'Right ipos [x, y, z]':<26} | {'Left ipos [x, y, z]':<26} | {'Status'}")
print("-" * 85)
all_ipos_pass = True
for r_name, l_name in body_pairs[:8]: # Primary links
    r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r_name)
    l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l_name)
    r_ipos = m.body_ipos[r_id]
    l_ipos = m.body_ipos[l_id]
    diff = np.array([abs(r_ipos[0] - l_ipos[0]), abs(r_ipos[1] + l_ipos[1]), abs(r_ipos[2] - l_ipos[2])])
    status = "PASS" if np.all(diff < 1e-4) else "FAIL"
    if status == "FAIL": all_ipos_pass = False
    r_str = f"[{r_ipos[0]:.4f}, {r_ipos[1]:.4f}, {r_ipos[2]:.4f}]"
    l_str = f"[{l_ipos[0]:.4f}, {l_ipos[1]:.4f}, {l_ipos[2]:.4f}]"
    print(f"{r_name.replace('right_', ''):<22} | {r_str:<26} | {l_str:<26} | {status}")

print("\n" + "="*85)
print("3. INERTIA TENSOR AUDIT (Principal Moments: Ixx_R = Ixx_L, Iyy_R = Iyy_L, Izz_R = Izz_L)")
print("="*85)
print(f"{'Body Link':<22} | {'Right Inertia [Ixx, Iyy, Izz]':<28} | {'Left Inertia [Ixx, Iyy, Izz]':<28} | {'Status'}")
print("-" * 85)
all_ine_pass = True
for r_name, l_name in body_pairs[:8]:
    r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r_name)
    l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l_name)
    r_ine = m.body_inertia[r_id]
    l_ine = m.body_inertia[l_id]
    diff = np.abs(r_ine - l_ine)
    status = "PASS" if np.all(diff < 1e-4) else "FAIL"
    if status == "FAIL": all_ine_pass = False
    r_str = f"[{r_ine[0]:.5f}, {r_ine[1]:.5f}, {r_ine[2]:.5f}]"
    l_str = f"[{l_ine[0]:.5f}, {l_ine[1]:.5f}, {l_ine[2]:.5f}]"
    print(f"{r_name.replace('right_', ''):<22} | {r_str:<28} | {l_str:<28} | {status}")

# 4. Joint Locations and Orientations
joint_pairs = [
    ('right_hip_yaw', 'left_hip_yaw'),
    ('right_hip_roll', 'left_hip_roll'),
    ('right_hip_pitch', 'left_hip_pitch'),
    ('right_knee', 'left_knee'),
    ('right_ankle_pitch', 'left_ankle_pitch'),
    ('right_ankle_roll', 'left_ankle_roll'),
    ('right_shoulder_pitch', 'left_shoulder_pitch'),
    ('right_shoulder_roll', 'left_shoulder_roll'),
    ('right_shoulder_yaw', 'left_shoulder_yaw'),
    ('right_elbow', 'left_elbow')
]

print("\n" + "="*85)
print("4. JOINT LOCATIONS & ORIENTATIONS AUDIT (Anchor: x_R=x_L, y_R=-y_L, z_R=z_L; Axis symmetry)")
print("="*85)
print(f"{'Joint Name':<22} | {'Right Axis':<16} | {'Left Axis':<16} | {'Right Range':<14} | {'Left Range':<14}")
print("-" * 85)
for r_jname, l_jname in joint_pairs:
    r_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, r_jname)
    l_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, l_jname)
    r_axis = m.jnt_axis[r_jid]
    l_axis = m.jnt_axis[l_jid]
    r_range = m.jnt_range[r_jid]
    l_range = m.jnt_range[l_jid]
    jname_base = r_jname.replace('right_', '')
    r_ax_str = f"[{r_axis[0]:.1f}, {r_axis[1]:.1f}, {r_axis[2]:.1f}]"
    l_ax_str = f"[{l_axis[0]:.1f}, {l_axis[1]:.1f}, {l_axis[2]:.1f}]"
    r_rng_str = f"[{r_range[0]:.2f}, {r_range[1]:.2f}]"
    l_rng_str = f"[{l_range[0]:.2f}, {l_range[1]:.2f}]"
    print(f"{jname_base:<22} | {r_ax_str:<16} | {l_ax_str:<16} | {r_rng_str:<14} | {l_rng_str:<14}")

# 5. Foot Collision Geometries
print("\n" + "="*85)
print("5. FOOT COLLISION GEOMETRIES AUDIT")
print("="*85)
for gname in ['right_foot_geom', 'left_foot_geom']:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    pos = m.geom_pos[gid]
    size = m.geom_size[gid]
    toe_x = pos[0] + size[0]
    heel_x = pos[0] - size[0]
    total_len = 2 * size[0]
    width = 2 * size[1]
    thickness = 2 * size[2]
    print(f"Geom: {gname}")
    print(f"  Pos:       [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
    print(f"  Half-Size: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}] m")
    print(f"  Toe Edge:  +{toe_x:.4f} m (+{toe_x*100:.1f} cm)")
    print(f"  Heel Edge: {heel_x:.4f} m ({heel_x*100:.1f} cm)")
    print(f"  Sole Span: {total_len:.4f} m ({total_len*100:.1f} cm length) x {width*100:.1f} cm width x {thickness*100:.1f} cm thickness")
