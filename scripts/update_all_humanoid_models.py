"""
Update all humanoid XML models across the Cyrus project to use the unified
humanoid structure with full CAD foot/ankle meshes and anatomical proportions.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MESH_ASSETS = """    <!-- Parametric CAD Human Foot & Shoe Meshes -->
    <material name="shoe_upper" rgba="0.13 0.14 0.16 1.0" specular="0.38" shininess="0.45" />
    <mesh name="human_foot_right" file="human_foot_right.stl" />
    <mesh name="human_foot_left" file="human_foot_left.stl" />
    <mesh name="foot_sole_right" file="foot_sole_right.stl" />
    <mesh name="foot_sole_left" file="foot_sole_left.stl" />
    <mesh name="ankle_ft_sensor" file="ankle_ft_sensor.stl" />
    <mesh name="shin_clevis_frame" file="shin_clevis_frame.stl" />
  </asset>"""

NEW_RIGHT_FOOT = """            <!-- ANATOMICAL HUMAN-PROPORTIONED SHOE / FOOT (Standing on Middle of Feet) -->
            <body name="right_foot" pos="0 0 -0.34">
              <joint name="right_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="right_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- Ankle 6-Axis FT Sensor Puck -->
              <geom type="mesh" mesh="ankle_ft_sensor" pos="0 0 -0.004" material="joint_metal" contype="0" conaffinity="0" mass="0" />

              <!-- INVISIBLE HIGH-PERFORMANCE PLANAR BASE (group="3" rgba="0 0 0 0") -->
              <geom name="right_foot_geom" type="box" pos="0.0125 0 -0.025" size="0.1375 0.054 0.015" group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- PHOTO-ACCURATE SCULPTED CAD SNEAKER UPPER & CONTOURED RUBBER SOLE -->
              <geom type="mesh" mesh="human_foot_right" material="shoe_upper" contype="0" conaffinity="0" mass="0" />
              <geom type="mesh" mesh="foot_sole_right" material="sole_rubber" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site at world z = 0.000m -->
              <site name="right_sole" pos="0.025 0 -0.040" size="0.006" />
            </body>"""

NEW_LEFT_FOOT = """            <!-- ANATOMICAL HUMAN-PROPORTIONED SHOE / FOOT (Standing on Middle of Feet) -->
            <body name="left_foot" pos="0 0 -0.34">
              <joint name="left_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="left_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- Ankle 6-Axis FT Sensor Puck -->
              <geom type="mesh" mesh="ankle_ft_sensor" pos="0 0 -0.004" material="joint_metal" contype="0" conaffinity="0" mass="0" />

              <!-- INVISIBLE HIGH-PERFORMANCE PLANAR BASE (group="3" rgba="0 0 0 0") -->
              <geom name="left_foot_geom" type="box" pos="0.0050 0 -0.025" size="0.1450 0.056 0.015" group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- PHOTO-ACCURATE SCULPTED CAD SNEAKER UPPER & CONTOURED RUBBER SOLE -->
              <geom type="mesh" mesh="human_foot_left" material="shoe_upper" contype="0" conaffinity="0" mass="0" />
              <geom type="mesh" mesh="foot_sole_left" material="sole_rubber" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site at world z = 0.000m -->
              <site name="left_sole" pos="0.020 0 -0.040" size="0.006" />
            </body>"""

def update_model(xml_name):
    path = os.path.join(MODELS_DIR, xml_name)
    if not os.path.exists(path):
        print(f"[-] Not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Compiler meshdir
    if 'meshdir="meshes"' not in content:
        for old_comp in ['<compiler angle="radian" coordinate="local"/>', '<compiler angle="radian" coordinate="local" />']:
            if old_comp in content:
                content = content.replace(old_comp, '<compiler angle="radian" coordinate="local" meshdir="meshes" autolimits="true" />')
                break

    # 2. Meshes in asset
    if 'human_foot_right.stl' not in content:
        content = content.replace('</asset>', MESH_ASSETS)

    # 3. Replace right foot body
    import re
    r_pattern = r'<body\s+name="right_foot"[^>]*>.*?</body>'
    content = re.sub(r_pattern, NEW_RIGHT_FOOT.strip(), content, flags=re.DOTALL)

    # 4. Replace left foot body
    l_pattern = r'<body\s+name="left_foot"[^>]*>.*?</body>'
    content = re.sub(l_pattern, NEW_LEFT_FOOT.strip(), content, flags=re.DOTALL)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Successfully updated: {xml_name}")

if __name__ == "__main__":
    targets = ["cyrus.xml", "cyrus_mjx.xml", "cyrus_arena_12.xml", "cyrus_ghost_trajectory.xml"]
    for t in targets:
        update_model(t)
