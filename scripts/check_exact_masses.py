import subprocess
import mujoco

res = subprocess.run(["git", "show", "HEAD:models/cyrus_12dof_mjx.xml"], capture_output=True, text=True)
m_orig = mujoco.MjModel.from_xml_string(res.stdout)

for b_id in range(m_orig.nbody):
    b_name = m_orig.body(b_id).name
    b_mass = m_orig.body_mass[b_id]
    b_inertia = m_orig.body_inertia[b_id]
    b_ipos = m_orig.body_ipos[b_id]
    print(f"{b_name:24s} | mass={b_mass:.4f} kg | ipos={b_ipos} | inertia={b_inertia}")
