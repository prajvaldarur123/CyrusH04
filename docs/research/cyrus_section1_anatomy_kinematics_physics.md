# Cyrus-H04 Technical Deep-Dive — Section 1
## Robot Anatomy, Kinematics & Physics Models

**Document ID:** CYRUS-ENG-SEC1-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                       CYRUS-H04 KINEMATIC & MASS TOPOLOGY
                               
                               [ Head & Perception ] (1.80 kg)
                                         │
                                [ Upper Torso & IMU ] (11.40 kg)
                                         │
                    ┌────────────────────┴────────────────────┐
              [Right Arm] (2.85 kg)                     [Left Arm] (2.85 kg)
              • Shoulder Pitch (±68.7°)                 • Shoulder Pitch (±68.7°)
              • Shoulder Roll (+45.8°)                  • Shoulder Roll (+45.8°)
              • Shoulder Yaw (±23.0°)                   • Shoulder Yaw (±23.0°)
              • Elbow Flexion (+80.2°)                  • Elbow Flexion (+80.2°)
                                         │
                                [ Lumbar Spine ]
                                • Lumbar Pitch (±20.0°)
                                • Lumbar Yaw (±15.0°)
                                         │
                                 [ Pelvis Core ] (3.80 kg)
                                         │ (z_nominal = 0.825 m)
                    ┌────────────────────┴────────────────────┐
             [Right Leg] (4.375 kg)                    [Left Leg] (4.375 kg)
             • Hip Yaw (±30.0°)                        • Hip Yaw (±30.0°)
             • Hip Roll (±25.0°)                       • Hip Roll (±25.0°)
             • Hip Pitch (-80°/+45°)                   • Hip Pitch (-80°/+45°)
             • Knee Flexion (0°/+120°)                 • Knee Flexion (0°/+120°)
             • Ankle Pitch (±45.0°)                    • Ankle Pitch (±45.0°)
             • Ankle Roll (±25.0°)                     • Ankle Roll (±25.0°)
                    │                                         │
           [Right Sole Complex]                      [Left Sole Complex]
           (21 cm × 10 cm, 0.825 kg)                 (21 cm × 10 cm, 0.825 kg)
```

---

## 1.1 Multibody Mass & Inertia Budgeting

### The Fundamental Design Philosophy
In legged robotics, mass distribution governs the agility, energy efficiency, and dynamic bandwidth of the robot. Cyrus-H04 is engineered following the **Distal Mass Minimization Principle**:
> *Keep mass concentrated near the center of mass (torso/pelvis) and make distal extremities (feet, shins) as lightweight as possible.*

When a leg swings, the rotational inertia about the hip joint scales quadratically with distance:
$$I_{\text{leg}} = \sum m_i r_i^2$$
Every extra 100 grams placed on the foot or ankle increases the motor torque required to swing the leg by up to **$4.5\times$** compared to that same mass placed inside the pelvis. 

```
Upper Body Mass: 18.90 kg (60.1%) ──► Concentrated at CoM (High restorative inertia)
Lower Body Mass: 12.55 kg (39.9%) ──► Tapered downward (Low leg rotational inertia)
Total System:    31.45 kg (100.0%)
```

### Complete Link-by-Link Mass & Inertia Schedule

| Body Link | Mass ($m$) | CoM Offset $[x, y, z]$ (m) | Principal Inertia $I_{xx}$ | Principal Inertia $I_{yy}$ | Principal Inertia $I_{zz}$ | Physical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Torso & Head** | $13.20\,\text{kg}$ | $[0.000, 0.000, +0.280]$ | $0.1850\,\text{kg}\cdot\text{m}^2$ | $0.1620\,\text{kg}\cdot\text{m}^2$ | $0.0980\,\text{kg}\cdot\text{m}^2$ | Houses compute, batteries, and IMU. High inertia acts as an inertial anchor against high-frequency disturbances. |
| **Arms (Dual 4-DoF)** | $5.70\,\text{kg}$ ($2.85\,\text{kg}\times 2$) | $[\pm 0.000, \pm 0.185, -0.150]$ | $0.0320\,\text{kg}\cdot\text{m}^2$ | $0.0380\,\text{kg}\cdot\text{m}^2$ | $0.0120\,\text{kg}\cdot\text{m}^2$ | Lightweight carbon composite with shoulder-concentrated cycloidal drives; allows rapid $50\,\text{Hz}$ counter-swinging. |
| **Pelvis Base** | $3.80\,\text{kg}$ | $[0.000, 0.000, 0.000]$ | $0.0450\,\text{kg}\cdot\text{m}^2$ | $0.0380\,\text{kg}\cdot\text{m}^2$ | $0.0420\,\text{kg}\cdot\text{m}^2$ | Structural bridge between spine and hips. Houses high-power hip roll and pitch motors. |
| **Thigh Link ($\times 2$)** | $2.15\,\text{kg}$ each | $[+0.010, \pm 0.015, -0.165]$ | $0.0240\,\text{kg}\cdot\text{m}^2$ | $0.0220\,\text{kg}\cdot\text{m}^2$ | $0.0065\,\text{kg}\cdot\text{m}^2$ | Concentrates hip pitch/knee actuators high up near the hip axis to minimize swing inertia. |
| **Shin Link ($\times 2$)** | $1.40\,\text{kg}$ each | $[0.000, 0.000, -0.140]$ | $0.0120\,\text{kg}\cdot\text{m}^2$ | $0.0115\,\text{kg}\cdot\text{m}^2$ | $0.0028\,\text{kg}\cdot\text{m}^2$ | High-strength hollow carbon truss. Ankle pitch motors are cable/pushrod-driven from near the knee. |
| **Foot Sole ($\times 2$)** | $0.825\,\text{kg}$ each | $[+0.035, 0.000, -0.025]$ | $0.0018\,\text{kg}\cdot\text{m}^2$ | $0.0042\,\text{kg}\cdot\text{m}^2$ | $0.0051\,\text{kg}\cdot\text{m}^2$ | Structural aluminum/rubber sole. Minimizes touchdown impact shock and prevents bounce upon landing. |
| **Total Humanoid** | **$31.45\,\text{kg}$** | $[0.000, 0.000, +0.783]$ | **—** | **—** | **—** | Calibrated against physical prototype bill of materials (BOM). |

---

## 1.2 12-DoF Lower-Body Kinematic Chain

```
               [ Pelvis Core ]
                      │
            (q0: Hip Yaw: ±30°) ────────► Z-axis rotation (Steering)
                      │
            (q1: Hip Roll: ±25°) ───────► X-axis rotation (Lateral abduction / balance)
                      │
            (q2: Hip Pitch: -80°/+45°) ─► Y-axis rotation (Forward / backward stride)
                      │
              [ Thigh: 0.400 m ]
                      │
            (q3: Knee: 0°/+120°) ───────► Y-axis rotation (Leg extension & shock absorption)
                      │
              [ Shin: 0.340 m ]
                      │
            (q4: Ankle Pitch: ±45°) ────► Y-axis rotation (Foot leveling & plantarflexion)
                      │
            (q5: Ankle Roll: ±25°) ─────► X-axis rotation (Ground contact inversion/eversion)
                      │
               [ Foot Sole ]
```

### The Anthropometric Ratio: Why $L_1 = 0.400\,\text{m}$ and $L_2 = 0.340\,\text{m}$?
Cyrus's leg link lengths are chosen with mathematical precision:
$$\frac{L_{\text{thigh}}}{L_{\text{shin}}} = \frac{0.400\,\text{m}}{0.340\,\text{m}} = 1.176$$
* **Human Biological Equivalent:** In adult human males, the femur-to-tibia ratio is approximately $1.15\text{--}1.18$.
* **Singularity Avoidance:** When walking or balancing, bipeds maintain a slightly bent knee ($\approx 0.35\,\text{rad} = 20^\circ$) to avoid the kinematic singularity at full knee extension ($q_{\text{knee}} = 0^\circ$). At this athletic stance:
  $$z_{\text{pelvis}} = L_1 \cos(q_{\text{hip}}) + L_2 \cos(q_{\text{hip}} + q_{\text{knee}}) \approx 0.825\,\text{m}$$
* **Torque Transmission Balance:** Making the thigh longer than the shin increases ground clearance during swing while keeping the knee actuator closer to the hip, directly slashing the moment of inertia during rapid stepping.

### Actuator Limits & Biomechanical Functions

```
Joint          Range (deg)        Range (rad)        Max Torque     Biomechanical Purpose
──────────────────────────────────────────────────────────────────────────────────────────
Hip Yaw        -30.0° / +30.0°    -0.524 / +0.524     60.0 N·m      Steering, yaw stabilization
Hip Roll       -25.0° / +25.0°    -0.436 / +0.436    120.0 N·m      Lateral push recovery, weight transfer
Hip Pitch      -80.0° / +45.0°    -1.396 / +0.785    120.0 N·m      Forward/backward stride, crouch
Knee Pitch       0.0° / +120.0°    0.000 / +2.094    150.0 N·m      Gravity support, shock absorption
Ankle Pitch    -45.0° / +45.0°    -0.785 / +0.785     80.0 N·m      CoP sagittal shift, push-off
Ankle Roll     -25.0° / +25.0°    -0.436 / +0.436     60.0 N·m      CoP coronal shift, uneven ground adaptation
```

---

## 1.3 8-DoF Upper-Body Inertial Decoupling Chain

```
               [ Upper Torso: 13.20 kg ]
                      │
        ┌─────────────┴─────────────┐
 [ Lumbar Pitch: ±20° ]      [ Lumbar Yaw: ±15° ]
   (Sagittal counter-torque)   (Torso rotational decoupling)
```

In early humanoid designs (e.g., classical ASIMO or basic laboratory bipeds), the upper body was treated as a rigid, static block. This threw away **$60.1\%$ of the robot's physical mass**!

Cyrus-H04 treats the upper body as an active **Reaction Wheel / Control Moment Gyroscope**:

### 1. Lumbar Spine (2-DoF)
* **Lumbar Pitch ($\pm 20.0^\circ$ / $\pm 0.35\,\text{rad}$):**
  When pushed from behind ($v_x > 0$), the torso snaps forward rapidly. By Newton’s Third Law, this exerts an equal and opposite torque on the pelvis:
  $$\tau_{\text{reaction}} = -I_{\text{torso}} \ddot{\theta}_{\text{lumbar}}$$
  This pushes the hips backward, counteracting forward momentum before the legs even need to move.
* **Lumbar Yaw ($\pm 15.0^\circ$ / $\pm 0.26\,\text{rad}$):**
  Cancels rotational ground contact torques when one leg swings forward, eliminating spinning on the stance foot.

### 2. Dual 4-DoF Arms ($5.70\,\text{kg}$ total)
Each arm features:
1. **Shoulder Pitch ($\pm 68.7^\circ$ / $\pm 1.20\,\text{rad}$):** Generates sagittal counter-momentum ($\dot{\mathbf{H}}_y$). Both arms swing backward for forward shoves, forward for backward shoves.
2. **Shoulder Roll ($+45.8^\circ$ / $+0.80\,\text{rad}$):** Lateral flaring. When pushed sideways to the right, the right arm flares outward to inject counter-roll torque ($\dot{\mathbf{H}}_x$) that directly neutralizes the $66.2\,\text{N}\cdot\text{m}$ overturning moment.
3. **Shoulder Yaw ($\pm 23.0^\circ$ / $\pm 0.40\,\text{rad}$):** Cancels torso twisting moments upon dynamic impact.
4. **Elbow Flexion ($+80.2^\circ$ / $+1.40\,\text{rad}$):** Flexes elbows during arm swings to reduce the arm's moment of inertia, allowing faster acceleration with less motor torque.

---

## 1.4 Anthropomorphic Sole Contact Site

```
               ┌───────────────────────▲───────────────────────┐
               │                       │                       │
               │                       │                       │
               │              TOE: +14.0 cm                    │
               │         (Forward Stability Zone)              │
               │                       │                       │
               │                       │                       │
        ◄──────┼───────────────────────O───────────────────────┼──────►
   -5.0 cm     │                  ANKLE AXIS                   │     +5.0 cm
  (Outer Edge) │                       │                       │  (Inner Edge)
               │              HEEL: -7.0 cm                    │
               │         (Backward Stability Zone)             │
               │                       │                       │
               └───────────────────────▼───────────────────────┘
                                TOTAL: 21.0 cm
```

### Biomechanical Geometry
* **Total Sole Length:** $0.210\,\text{m}$ ($21.0\,\text{cm}$)
* **Ankle-to-Toe Distance:** $+0.140\,\text{m}$ ($+14.0\,\text{cm}$)
* **Ankle-to-Heel Distance:** $-0.070\,\text{m}$ ($-7.0\,\text{cm}$)
* **Total Sole Width:** $0.100\,\text{m}$ ($10.0\,\text{cm}$, $\pm 5.0\,\text{cm}$ from centerline)

### Why is the Toe 2x Longer than the Heel (2:1 Ratio)?
In human biomechanics, the distance from ankle to toe is roughly double the distance from ankle to heel:
1. **Human Forward Locomotion Bias:** Humans primarily move forward, requiring a long forward lever arm for the ankle plantarflexor muscles (soleus and gastrocnemius) to exert push-off torque:
   $$\tau_{\text{push-off}} = F_{\text{normal}} \cdot d_{\text{toe}} = (31.45\,\text{kg} \times 9.81\,\text{m/s}^2) \times 0.14\,\text{m} \approx 43.2\,\text{N}\cdot\text{m}$$
2. **Trip Resistance:** When Cyrus takes a forward step, the $14\,\text{cm}$ toe overhang provides a massive polygon of support that prevents forward face-plants.

### Contact Physics & Friction Cone
The sole interacts with the floor through a non-linear Coulomb friction cone:
$$F_{\text{friction}} = \sqrt{F_x^2 + F_y^2} \le \mu F_z$$
* **Friction Coefficient ($\mu$):** Calibrated to $\mu = 0.70$ (industrial gym turf / indoor lab flooring).
* **Contact Parameters:**
  - `solref = [0.004, 1.0]`: Defines contact stiffness time constant ($4.0\,\text{ms}$) and damping ratio ($1.0$ = critically damped). Critically damped contact eliminates unphysical bouncing when the swing foot hits the floor.
  - `solimp = [0.90, 0.95, 0.001]`: Ensures high contact impedance ($90\%\text{--}95\%$) so the foot does not sink or penetrate into the ground mesh.

---

## 1.5 MuJoCo MJX Simulation Model (`models/cyrus_12dof_mjx.xml`)

### Why $2.5\,\text{ms}$ Timestep ($400\,\text{Hz}$ Physics Integration)?
$$\text{Control Rate: } 50\,\text{Hz} \ (dt = 0.020\,\text{s}) \quad \Longleftrightarrow \quad \text{Substeps: } 8 \times 0.0025\,\text{s}$$
* **The Failure of $10\,\text{ms}$ Timesteps:** 
  Standard RL simulations often use $10\,\text{ms}$ or $20\,\text{ms}$ steps for speed. However, for a $31.45\,\text{kg}$ biped with rigid metal soles, a $10\,\text{ms}$ step allows the foot to penetrate $3\text{--}5\,\text{mm}$ into the ground before the solver notices. The resulting restitution force causes violent contact chatter, launching the robot into the air.
* **The Stability of $2.5\,\text{ms}$:** 
  At $dt = 0.0025\,\text{s}$, the maximum penetration at high impact velocity ($1.0\,\text{m/s}$) is less than $0.2\,\text{mm}$. The contact forces remain smooth and realistic.

### GPU Parallel Acceleration via MJX (JAX)
* In `cyrus_12dof_mjx.xml`, all collision meshes are represented as convex geometric primitives (cylinders, boxes, capsules).
* **Convex Collision Detection:** Eliminates expensive mesh-mesh raycasting, allowing pure analytical separating-axis theorem (SAT) calculations on the GPU.
* **Throughput:** $1,024$ parallel humanoids run simultaneously at **$> 25,000$ simulation frames per second** on a single NVIDIA GeForce RTX 3050 Laptop GPU, turning 10 hours of real-world physics into 60 seconds of GPU compute.

---

## Summary of Section 1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 1 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Mass Budget:      31.45 kg total (60.1% upper body, 39.9% lower body)         │
│ 2. Kinematic Links:  Thigh = 0.400 m, Shin = 0.340 m (Ratio 1.176 matches human) │
│ 3. Lower Body DoFs:  12 DoFs (3 Hip + 1 Knee + 2 Ankle per leg)                  │
│ 4. Upper Body DoFs:  8 DoFs (2 Lumbar + 6 Shoulder + 2 Elbow) for active balance │
│ 5. Foot Sole:        21 cm × 10 cm (Toe: +14 cm, Heel: -7 cm, 2:1 human ratio)   │
│ 6. Physics Solver:   MuJoCo MJX GPU at 2.5 ms timestep (8 substeps per 50Hz RL)  │
└──────────────────────────────────────────────────────────────────────────────────┘
```
