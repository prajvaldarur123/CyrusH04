"""
Updates models/cyrus_human_feet.xml to use the exact CAD foot meshes,
invisible collision plantar primitive, and silver ankle pivot bolts
matching the user's screenshot.
"""
import os
import sys
import re

if sys.platform == "win32":
    PROJECT_ROOT = "C:/Users/prajval/robotics/cyrus"
else:
    PROJECT_ROOT = "/mnt/c/Users/prajval/robotics/cyrus"

SRC_XML = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
DST_XML = os.path.join(PROJECT_ROOT, "models", "cyrus_human_feet.xml")

with open(SRC_XML, "r") as f:
    xml_content = f.read()

# 1. Ensure compiler has meshdir="meshes"
if 'meshdir="meshes"' not in xml_content:
    xml_content = xml_content.replace(
        '<compiler angle="radian" coordinate="local" />',
        '<compiler angle="radian" coordinate="local" meshdir="meshes" autolimits="true" />'
    )

# 2. Add CAD foot meshes and materials in <asset>
mesh_assets = """    <!-- Parametric CAD Human Foot & Shoe Meshes (Matching Reference Screenshot) -->
    <material name="shoe_upper" rgba="0.13 0.14 0.16 1.0" specular="0.38" shininess="0.45" />
    <mesh name="human_foot_right" file="human_foot_right.stl" />
    <mesh name="human_foot_left" file="human_foot_left.stl" />
    <mesh name="foot_sole_right" file="foot_sole_right.stl" />
    <mesh name="foot_sole_left" file="foot_sole_left.stl" />
    <mesh name="ankle_ft_sensor" file="ankle_ft_sensor.stl" />
    <mesh name="shin_clevis_frame" file="shin_clevis_frame.stl" />
  </asset>"""

xml_content = xml_content.replace('</asset>', mesh_assets)

# 3. New Right Shin and Foot with Exact Screenshot Meshes & Invisible High-Friction Planar Base
right_shin_and_foot = """          <body name="right_shin" pos="0 0 -0.20">
            <joint name="right_knee" type="hinge" axis="0 1 0" range="0.0 2.2" damping="3.5" />
            
            <!-- Knee Joint Actuator Hub -->
            <geom type="cylinder" size="0.04 0.035" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" />
            <geom type="cylinder" size="0.036 0.040" zaxis="0 1 0" material="anodized_gold" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.038 0.03" pos="0.045 0 0" material="armor_white" friction="2.0 0.01 0.001" />

            <!-- Shin Structural Core -->
            <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" density="950" />
            
            <!-- Smooth Sculpted Shin Fairing Matching Screenshot -->
            <geom type="mesh" mesh="shin_clevis_frame" material="armor_white" contype="0" conaffinity="0" mass="0" />

            <!-- Brushed Silver Ankle Pivot Bolts (Matching Screenshot Ankle Collar Notches) -->
            <geom type="cylinder" size="0.016 0.005" pos="0 -0.031 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.016 0.005" pos="0  0.031 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

            <!-- ANATOMICAL HUMAN-PROPORTIONED SHOE / FOOT (Standing on Middle of Feet) -->
            <body name="right_foot" pos="0 0 -0.34">
              <joint name="right_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="right_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- Ankle 6-Axis FT Sensor Puck -->
              <geom type="mesh" mesh="ankle_ft_sensor" pos="0 0 -0.004" material="joint_metal" contype="0" conaffinity="0" mass="0" />

              <!-- INVISIBLE HIGH-PERFORMANCE PLANAR BASE (group="3" rgba="0 0 0 0") -->
              <!-- 100% Stable Flat Floor Contact from -12.5cm to +15.0cm WITHOUT Ugly Box Geoms! -->
              <geom name="right_foot_geom" type="box" pos="0.0125 0 -0.025" size="0.1375 0.054 0.015" group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- PHOTO-ACCURATE SCULPTED CAD SNEAKER UPPER & CONTOURED RUBBER SOLE -->
              <geom type="mesh" mesh="human_foot_right" material="shoe_upper" contype="0" conaffinity="0" mass="0" />
              <geom type="mesh" mesh="foot_sole_right" material="sole_rubber" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site at world z = 0.000m -->
              <site name="right_sole" pos="0.020 0 -0.040" size="0.006" />
            </body>
          </body>"""

# 4. New Left Shin and Foot
left_shin_and_foot = """          <body name="left_shin" pos="0 0 -0.20">
            <joint name="left_knee" type="hinge" axis="0 1 0" range="0.0 2.2" damping="3.5" />
            
            <!-- Knee Joint Actuator Hub -->
            <geom type="cylinder" size="0.04 0.035" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" />
            <geom type="cylinder" size="0.036 0.040" zaxis="0 1 0" material="anodized_gold" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.038 0.03" pos="0.045 0 0" material="armor_white" friction="2.0 0.01 0.001" />

            <!-- Shin Structural Core -->
            <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" density="950" />
            
            <!-- Smooth Sculpted Shin Fairing Matching Screenshot -->
            <geom type="mesh" mesh="shin_clevis_frame" material="armor_white" contype="0" conaffinity="0" mass="0" />

            <!-- Brushed Silver Ankle Pivot Bolts (Matching Screenshot Ankle Collar Notches) -->
            <geom type="cylinder" size="0.016 0.005" pos="0 -0.031 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.016 0.005" pos="0  0.031 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

            <!-- ANATOMICAL HUMAN-PROPORTIONED SHOE / FOOT (Standing on Middle of Feet) -->
            <body name="left_foot" pos="0 0 -0.34">
              <joint name="left_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="left_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- Ankle 6-Axis FT Sensor Puck -->
              <geom type="mesh" mesh="ankle_ft_sensor" pos="0 0 -0.004" material="joint_metal" contype="0" conaffinity="0" mass="0" />

              <!-- INVISIBLE HIGH-PERFORMANCE PLANAR BASE (group="3" rgba="0 0 0 0") -->
              <!-- 100% Stable Flat Floor Contact from -12.5cm to +15.0cm WITHOUT Ugly Box Geoms! -->
              <geom name="left_foot_geom" type="box" pos="0.0125 0 -0.025" size="0.1375 0.054 0.015" group="3" rgba="0 0 0 0" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- PHOTO-ACCURATE SCULPTED CAD SNEAKER UPPER & CONTOURED RUBBER SOLE -->
              <geom type="mesh" mesh="human_foot_left" material="shoe_upper" contype="0" conaffinity="0" mass="0" />
              <geom type="mesh" mesh="foot_sole_left" material="sole_rubber" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site at world z = 0.000m -->
              <site name="left_sole" pos="0.020 0 -0.040" size="0.006" />
            </body>
          </body>"""

right_pattern = r'<body name="right_shin" pos="0 0 -0.20">[\s\S]*?</body>\s*</body>'
left_pattern = r'<body name="left_shin" pos="0 0 -0.20">[\s\S]*?</body>\s*</body>'

new_xml = re.sub(right_pattern, right_shin_and_foot, xml_content, count=1)
new_xml = re.sub(left_pattern, left_shin_and_foot, new_xml, count=1)

with open(DST_XML, "w") as f:
    f.write(new_xml)

print(f"[+] Recreated exact screenshot CAD feet in: {DST_XML}")
