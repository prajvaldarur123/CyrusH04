# ==============================================================================
# CYRUS 150-DIMENSIONAL WHOLE-BODY PROPRIOCEPTIVE SCHEMA OBSERVER
# Implements Agent 3's Whole-Body Proprioception & Dynamic Balance Specification
# ==============================================================================
import numpy as np
import mujoco

class CyrusBodySchemaObserver:
    """
    Extracts the 150-Dimensional Whole-Body Proprioceptive Schema:
    - Group 1: Floating-Base Kinematics (13-D)
    - Group 2: Joint Kinematics across 27 Actuators (54-D)
    - Group 3: Contact Geometry, Activation Sigmas & Normal Forces (12-D)
    - Group 4: Centroidal Equilibrium & Inverted Pendulum Dynamics (8-D)
    - Group 5: Rotational Momentum & Continuous Gyroscopic Pose (9-D)
    - Group 6: Kinematic Safety & Actuation Feedback (54-D)
    Total: 13 + 54 + 12 + 8 + 9 + 54 = 150 Dimensions.
    """

    def __init__(self, model):
        self.model = model
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
        self.r_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
        self.l_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.r_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
        self.l_sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        # Base 27 body joints (excludes fingers)
        self.base_joint_names = [
            'waist_yaw', 'waist_roll', 'waist_pitch',
            'neck_yaw', 'neck_pitch',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]
        self.qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.base_joint_names], dtype=np.int32)
        self.qvel_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.base_joint_names], dtype=np.int32)

        # Control range bounds for 27 actuators
        self.q_min = model.actuator_ctrlrange[:27, 0].astype(np.float32)
        self.q_max = model.actuator_ctrlrange[:27, 1].astype(np.float32)
        self.q_range = self.q_max - self.q_min
        self.q_range = np.where(self.q_range == 0, 1.0, self.q_range)

        self.total_mass = float(np.sum(model.body_mass))
        self.prev_action = np.zeros(27, dtype=np.float32)

        # Precompute geom-to-limb ancestry for contact force routing
        self._build_geom_limb_mapping()

    def _build_geom_limb_mapping(self):
        def get_ancestors(body_id):
            ans = set()
            curr = body_id
            while curr > 0:
                ans.add(curr)
                curr = self.model.body_parentid[curr]
            return ans

        r_hand_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
        l_hand_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_hand')
        r_foot_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_foot')
        l_foot_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_foot')

        self.geom_limb_map = {}
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            anc = get_ancestors(bid)
            anc.add(bid)
            if r_hand_b in anc:
                self.geom_limb_map[gid] = 0  # r_palm
            elif l_hand_b in anc:
                self.geom_limb_map[gid] = 1  # l_palm
            elif r_foot_b in anc:
                self.geom_limb_map[gid] = 2  # r_sole
            elif l_foot_b in anc:
                self.geom_limb_map[gid] = 3  # l_sole

    def compute_contact_normal_forces(self, data):
        """
        Sums vertical/normal contact forces for [r_palm, l_palm, r_sole, l_sole].
        """
        f_z = np.zeros(4, dtype=np.float32)
        c_force = np.zeros(6, dtype=np.float64)
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = con.geom1, con.geom2
            limb_idx = self.geom_limb_map.get(g1, None)
            if limb_idx is None:
                limb_idx = self.geom_limb_map.get(g2, None)
            if limb_idx is not None:
                mujoco.mj_contactForce(self.model, data, i, c_force)
                f_z[limb_idx] += float(abs(c_force[0]))
        return f_z

    def update_action(self, action):
        """Stores previous action (27-dim)."""
        self.prev_action = np.asarray(action[:27], dtype=np.float32).copy()

    def reset(self):
        """Resets previous action."""
        self.prev_action = np.zeros(27, dtype=np.float32)

    def extract_schema(self, data, anchor_x=0.0, anchor_y=0.0, contact_forces=None):
        # 1. Pelvis state (13)
        p_pos = data.qpos[:3].copy()
        p_pos[0] -= anchor_x
        p_pos[1] -= anchor_y
        p_quat = data.qpos[3:7].copy()
        p_lin_vel = data.qvel[:3].copy()
        p_ang_vel = data.qvel[3:6].copy()

        # 2. Joint state (54)
        q = data.qpos[self.qpos_adr].copy()
        q_dot = data.qvel[self.qvel_adr].copy() * 0.1

        # 3. Contact heights, sigmas & forces (12)
        endeff_sites = [self.r_palm_id, self.l_palm_id, self.r_sole_id, self.l_sole_id]
        z_heights = np.array([data.site_xpos[sid, 2] for sid in endeff_sites], dtype=np.float32)

        if contact_forces is None:
            f_z = self.compute_contact_normal_forces(data)
        elif isinstance(contact_forces, dict):
            f_z = np.array([contact_forces.get(k, 0.0) for k in ['r_palm', 'l_palm', 'r_sole', 'l_sole']], dtype=np.float32)
        else:
            f_z = np.asarray(contact_forces, dtype=np.float32)

        sigmas = 1.0 / (1.0 + np.exp(-0.3 * (f_z - 10.0)))
        norm_fz = f_z / (self.total_mass * 9.81)

        # 4. Centroidal CoM & DCM Error (8)
        com = data.subtree_com[0].copy()
        if hasattr(data, 'subtree_linvel'):
            v_com = data.subtree_linvel[0].copy()
        else:
            v_com = data.qvel[:3].copy()

        # Support polygon centroid
        active_mask = (f_z >= 10.0)
        if np.any(active_mask):
            pts = np.array([data.site_xpos[endeff_sites[i], :2] for i in range(4) if active_mask[i]])
            poly_centroid = np.mean(pts, axis=0)
        else:
            poly_centroid = 0.5 * (data.site_xpos[self.r_sole_id, :2] + data.site_xpos[self.l_sole_id, :2])

        com_rel = np.array([com[0] - poly_centroid[0], com[1] - poly_centroid[1], com[2]], dtype=np.float32)

        omega_0 = np.sqrt(9.81 / max(float(com[2]), 0.25))
        icp = com[:2] + v_com[:2] / omega_0
        dcm_err = (icp - poly_centroid).astype(np.float32)

        # 5. Angular momentum & Torso orientation (9)
        if hasattr(data, 'subtree_angmom'):
            h_com_raw = data.subtree_angmom[0].copy()
        else:
            h_com_raw = np.cross(com, v_com * self.total_mass)
        h_com = (h_com_raw / (self.total_mass * np.sqrt(9.81 * (max(float(com[2]), 0.10)**3)))).astype(np.float32)

        r_torso = data.xmat[self.torso_id].reshape(3, 3)
        torso_x = r_torso[:, 0].copy().astype(np.float32)
        torso_z = r_torso[:, 2].copy().astype(np.float32)

        # 6. Joint limit proximity & previous action (54)
        q_norm = 2.0 * (q - self.q_min) / self.q_range - 1.0
        d_limits = (np.sign(q_norm) * np.maximum(0.0, np.abs(q_norm) - 0.75)).astype(np.float32)

        schema_150 = np.concatenate([
            p_pos, p_quat, p_lin_vel, p_ang_vel,       # 13
            q, q_dot,                                  # 54
            z_heights, sigmas, norm_fz,                # 12
            com_rel, v_com, dcm_err,                   # 8
            h_com, torso_x, torso_z,                   # 9
            d_limits, self.prev_action                 # 54
        ]).astype(np.float32)

        assert len(schema_150) == 150, f"Dimension mismatch: {len(schema_150)} != 150"
        return schema_150
