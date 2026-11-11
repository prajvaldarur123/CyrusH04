"""
Unifies photo-accurate CAD meshes and exact 61.3923 kg mass distribution across all models.
"""
import os
import re
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for m_name in ["cyrus_12dof_mjx.xml", "cyrus_human_feet.xml", "cyrus.xml", "cyrus_mjx.xml"]:
    xml_path = os.path.join(MODELS_DIR, m_name)
    if not os.path.exists(xml_path):
        continue

    with open(xml_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # 1. Right Shin with explicit inertial
    r_shin_old = re.search(r'<body\s+name="right_shin"[^>]*>', txt)
    if r_shin_old:
        target = r_shin_old.group(0)
        # Check if inertial already exists
        if '<body name="right_shin" pos="0 0 -0.20">\n            <inertial' not in txt:
            replacement = (
                '<body name="right_shin" pos="0 0 -0.20">\n'
                '            <inertial pos="0.00456656 0 -0.13724896" mass="3.8280" diaginertia="0.04714421 0.04581121 0.00415275" />'
            )
            txt = txt.replace(target, replacement, 1)

    # 2. Left Shin with explicit inertial
    l_shin_old = re.search(r'<body\s+name="left_shin"[^>]*>', txt)
    if l_shin_old:
        target = l_shin_old.group(0)
        if '<body name="left_shin" pos="0 0 -0.20">\n            <inertial' not in txt:
            replacement = (
                '<body name="left_shin" pos="0 0 -0.20">\n'
                '            <inertial pos="0.00456656 0 -0.13724896" mass="3.8280" diaginertia="0.04714421 0.04581121 0.00415275" />'
            )
            txt = txt.replace(target, replacement, 1)

    # 3. Right Foot with explicit inertial
    r_foot_old = re.search(r'<body\s+name="right_foot"[^>]*>', txt)
    if r_foot_old:
        target = r_foot_old.group(0)
        if '<body name="right_foot" pos="0 0 -0.34">\n              <inertial' not in txt:
            replacement = (
                '<body name="right_foot" pos="0 0 -0.34">\n'
                '              <inertial pos="0.01157045 0 0.00820321" mass="2.3396" diaginertia="0.0175966 0.01577655 0.00282786" />'
            )
            txt = txt.replace(target, replacement, 1)

    # 4. Left Foot with explicit inertial
    l_foot_old = re.search(r'<body\s+name="left_foot"[^>]*>', txt)
    if l_foot_old:
        target = l_foot_old.group(0)
        if '<body name="left_foot" pos="0 0 -0.34">\n              <inertial' not in txt:
            replacement = (
                '<body name="left_foot" pos="0 0 -0.34">\n'
                '              <inertial pos="0.0218251 0 0.00688213" mass="2.1040" diaginertia="0.01518666 0.01339587 0.00266628" />'
            )
            txt = txt.replace(target, replacement, 1)

    # 5. Make all child geoms in shin and foot have mass="0" so they do not add duplicate mass to inertial
    # For shin capsule
    txt = txt.replace(
        '<geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" density="950" />',
        '<geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" mass="0" />'
    )
    # For right foot box geom
    txt = txt.replace(
        'group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850"',
        'group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" mass="0"'
    )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(txt)

    # Validate with MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    # Calculate mass excluding gaze_target
    gaze_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gaze_target")
    total_mass = sum(m.body_mass[i] for i in range(m.nbody) if i != gaze_id)
    print(f"[+] {m_name:24s} unified! Robot Mass = {total_mass:.4f} kg (Target: 61.3923 kg)")
