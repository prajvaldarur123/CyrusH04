"""
Generates models/cyrus_human_feet.xml with human-proportioned sculpted feet
and ankle joint articulation matching the reference image.
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

# Right leg (shin + foot) definition
right_shin_and_foot = """          <body name="right_shin" pos="0 0 -0.20">
            <joint name="right_knee" type="hinge" axis="0 1 0" range="0.0 2.2" damping="3.5" />
            
            <!-- Knee Joint Actuator Hub -->
            <geom type="cylinder" size="0.04 0.035" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" />
            <geom type="cylinder" size="0.036 0.040" zaxis="0 1 0" material="anodized_gold" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.038 0.03" pos="0.045 0 0" material="armor_white" friction="2.0 0.01 0.001" />

            <!-- Shin Structural Core -->
            <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" density="950" />
            
            <!-- White Frontal & Lateral Shin Armor (Matching Photo) -->
            <geom type="box" size="0.022 0.038 0.12" pos="0.032 0 -0.14" material="armor_white" contype="0" conaffinity="0" />
            <geom type="capsule" fromto="0.028 0 -0.05 0.022 0 -0.24" size="0.028" material="armor_white" contype="0" conaffinity="0" />
            <geom type="capsule" fromto="-0.025 0 -0.06 -0.025 0 -0.22" size="0.032" material="carbon_frame" contype="0" conaffinity="0" />

            <!-- Anterior Metallic Ribbed Tendon / Grille Guard (Matching Photo Lower Shin) -->
            <geom type="box" size="0.008 0.022 0.035" pos="0.038 0 -0.28" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.265" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.280" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.295" material="joint_metal" contype="0" conaffinity="0" mass="0" />

            <!-- Distal Ankle Clevis Bracket (White Armor Fork) -->
            <geom type="box" size="0.025 0.008 0.03" pos="0.0 -0.038 -0.32" material="armor_white" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.008 0.03" pos="0.0 0.038 -0.32" material="armor_white" contype="0" conaffinity="0" mass="0" />

            <!-- Outer Ankle Mechanical Pivot Hub Discs (Matching Photo Side Joint Hub) -->
            <geom type="cylinder" size="0.028 0.006" pos="0 -0.044 -0.34" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.020 0.008" pos="0 -0.045 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.028 0.006" pos="0 0.044 -0.34" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.020 0.008" pos="0 0.045 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

            <!-- HUMAN-PROPORTIONED SCULPTED ROBOTIC FOOT (MATCHING PHOTO) -->
            <!-- Ankle Joint Origin: Placed at pos="0 0 -0.34" (4.0 cm above floor) -->
            <body name="right_foot" pos="0 0 -0.34">
              <joint name="right_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="right_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- PRIMARY TRACTION SOLE (Continuous Stable High-Friction Contact Base) -->
              <!-- Length: 24cm total (-9.0cm heel to +15.0cm toe). Center = +0.030m. Half-length = 0.120m -->
              <geom name="right_foot_geom" type="box" pos="0.030 0 -0.025" size="0.120 0.048 0.015" material="sole_rubber" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- VISUAL SCULPTED HUMAN SHOE / FOOT (MATCHING SCREENSHOT) -->
              <!-- 1. Black Outsole & Midsole Profile -->
              <geom type="box" size="0.122 0.049 0.005" pos="0.030 0 -0.035" material="sole_rubber" contype="0" conaffinity="0" mass="0" />
              <geom type="box" size="0.118 0.047 0.004" pos="0.030 0 -0.026" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

              <!-- 2. Sculpted Heel Counter (Curved Black Heel wrapping behind ankle to -9cm) -->
              <geom type="ellipsoid" size="0.050 0.042 0.026" pos="-0.045 0 -0.010" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="cylinder" size="0.040 0.020" pos="-0.055 0 -0.015" zaxis="0 0 1" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="sphere" size="0.038" pos="-0.055 0 -0.012" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 3. Ankle Cuff / Instep Collar (Where the shin articulates into the shoe) -->
              <geom type="ellipsoid" size="0.042 0.040 0.022" pos="0.000 0 0.005" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="cylinder" size="0.035 0.012" pos="0.000 0 0.012" zaxis="0 0 1" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 4. Sloping Instep / Bridge (Smooth forward slope from ankle to midfoot) -->
              <geom type="box" size="0.045 0.044 0.018" pos="0.045 0 -0.010" euler="0 14 0" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 5. Ball of Foot & Forefoot (Wider anatomical metatarsal area) -->
              <geom type="ellipsoid" size="0.040 0.047 0.018" pos="0.090 0 -0.018" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 6. Anatomical Rounded Toe Box (Sleek tapering toe cap extending to +15cm) -->
              <geom type="ellipsoid" size="0.035 0.042 0.014" pos="0.125 0 -0.022" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="box" size="0.020 0.038 0.010" pos="0.130 0 -0.024" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site: Located at +0.030m (Dead center of support polygon) -->
              <site name="right_sole" pos="0.030 0 -0.040" size="0.01" />
            </body>
          </body>"""

# Left leg (shin + foot) definition
left_shin_and_foot = """          <body name="left_shin" pos="0 0 -0.20">
            <joint name="left_knee" type="hinge" axis="0 1 0" range="0.0 2.2" damping="3.5" />
            
            <!-- Knee Joint Actuator Hub -->
            <geom type="cylinder" size="0.04 0.035" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" />
            <geom type="cylinder" size="0.036 0.040" zaxis="0 1 0" material="anodized_gold" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.038 0.03" pos="0.045 0 0" material="armor_white" friction="2.0 0.01 0.001" />

            <!-- Shin Structural Core -->
            <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.040" material="carbon_frame" density="950" />
            
            <!-- White Frontal & Lateral Shin Armor (Matching Photo) -->
            <geom type="box" size="0.022 0.038 0.12" pos="0.032 0 -0.14" material="armor_white" contype="0" conaffinity="0" />
            <geom type="capsule" fromto="0.028 0 -0.05 0.022 0 -0.24" size="0.028" material="armor_white" contype="0" conaffinity="0" />
            <geom type="capsule" fromto="-0.025 0 -0.06 -0.025 0 -0.22" size="0.032" material="carbon_frame" contype="0" conaffinity="0" />

            <!-- Anterior Metallic Ribbed Tendon / Grille Guard (Matching Photo Lower Shin) -->
            <geom type="box" size="0.008 0.022 0.035" pos="0.038 0 -0.28" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.265" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.280" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.009 0.018 0.004" pos="0.039 0 -0.295" material="joint_metal" contype="0" conaffinity="0" mass="0" />

            <!-- Distal Ankle Clevis Bracket (White Armor Fork) -->
            <geom type="box" size="0.025 0.008 0.03" pos="0.0 -0.038 -0.32" material="armor_white" contype="0" conaffinity="0" mass="0" />
            <geom type="box" size="0.025 0.008 0.03" pos="0.0 0.038 -0.32" material="armor_white" contype="0" conaffinity="0" mass="0" />

            <!-- Outer Ankle Mechanical Pivot Hub Discs (Matching Photo Side Joint Hub) -->
            <geom type="cylinder" size="0.028 0.006" pos="0 -0.044 -0.34" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.020 0.008" pos="0 -0.045 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.028 0.006" pos="0 0.044 -0.34" zaxis="0 1 0" material="joint_metal" contype="0" conaffinity="0" mass="0" />
            <geom type="cylinder" size="0.020 0.008" pos="0 0.045 -0.34" zaxis="0 1 0" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

            <!-- HUMAN-PROPORTIONED SCULPTED ROBOTIC FOOT (MATCHING PHOTO) -->
            <!-- Ankle Joint Origin: Placed at pos="0 0 -0.34" (4.0 cm above floor) -->
            <body name="left_foot" pos="0 0 -0.34">
              <joint name="left_ankle_pitch" type="hinge" axis="0 1 0" range="-0.8 0.8" damping="2.5" />
              <joint name="left_ankle_roll" type="hinge" axis="1 0 0" range="-0.4 0.4" damping="2.5" />

              <!-- PRIMARY TRACTION SOLE (Continuous Stable High-Friction Contact Base) -->
              <!-- Length: 24cm total (-9.0cm heel to +15.0cm toe). Center = +0.030m. Half-length = 0.120m -->
              <geom name="left_foot_geom" type="box" pos="0.030 0 -0.025" size="0.120 0.048 0.015" material="sole_rubber" friction="1.4 0.01 0.005" condim="6" solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2" density="850" />

              <!-- VISUAL SCULPTED HUMAN SHOE / FOOT (MATCHING SCREENSHOT) -->
              <!-- 1. Black Outsole & Midsole Profile -->
              <geom type="box" size="0.122 0.049 0.005" pos="0.030 0 -0.035" material="sole_rubber" contype="0" conaffinity="0" mass="0" />
              <geom type="box" size="0.118 0.047 0.004" pos="0.030 0 -0.026" material="chrome_trim" contype="0" conaffinity="0" mass="0" />

              <!-- 2. Sculpted Heel Counter (Curved Black Heel wrapping behind ankle to -9cm) -->
              <geom type="ellipsoid" size="0.050 0.042 0.026" pos="-0.045 0 -0.010" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="cylinder" size="0.040 0.020" pos="-0.055 0 -0.015" zaxis="0 0 1" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="sphere" size="0.038" pos="-0.055 0 -0.012" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 3. Ankle Cuff / Instep Collar (Where the shin articulates into the shoe) -->
              <geom type="ellipsoid" size="0.042 0.040 0.022" pos="0.000 0 0.005" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="cylinder" size="0.035 0.012" pos="0.000 0 0.012" zaxis="0 0 1" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 4. Sloping Instep / Bridge (Smooth forward slope from ankle to midfoot) -->
              <geom type="box" size="0.045 0.044 0.018" pos="0.045 0 -0.010" euler="0 14 0" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 5. Ball of Foot & Forefoot (Wider anatomical metatarsal area) -->
              <geom type="ellipsoid" size="0.040 0.047 0.018" pos="0.090 0 -0.018" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- 6. Anatomical Rounded Toe Box (Sleek tapering toe cap extending to +15cm) -->
              <geom type="ellipsoid" size="0.035 0.042 0.014" pos="0.125 0 -0.022" material="carbon_frame" contype="0" conaffinity="0" mass="0" />
              <geom type="box" size="0.020 0.038 0.010" pos="0.130 0 -0.024" material="carbon_frame" contype="0" conaffinity="0" mass="0" />

              <!-- Center of Sole Site: Located at +0.030m (Dead center of support polygon) -->
              <site name="left_sole" pos="0.030 0 -0.040" size="0.01" />
            </body>
          </body>"""

right_pattern = r'<body name="right_shin" pos="0 0 -0.20">[\s\S]*?</body>\s*</body>'
left_pattern = r'<body name="left_shin" pos="0 0 -0.20">[\s\S]*?</body>\s*</body>'

new_xml = re.sub(right_pattern, right_shin_and_foot, xml_content, count=1)
new_xml = re.sub(left_pattern, left_shin_and_foot, new_xml, count=1)

with open(DST_XML, "w") as f:
    f.write(new_xml)

print(f"[+] Recreated human-feet model in separate file: {DST_XML}")
