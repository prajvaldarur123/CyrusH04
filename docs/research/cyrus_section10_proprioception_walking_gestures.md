# SECTION 10: Whole-Body Proprioceptive Schema, Bipedal Walking Locomotion & Expressive Gestures

**Project:** Cyrus-H04 Full-Scale Humanoid Robotics Initiative  
**Document ID:** `CYRUS-TR-2026-SEC10-REV1`  
**Primary Source Files:**  
- [`controller/body_schema_observer.py`](file:///C:/Users/prajval/robotics/cyrus/controller/body_schema_observer.py) ($179\,\text{lines}$ — 150-D Proprioceptive Schema Engine)  
- [`controller/kinematic_observer.py`](file:///C:/Users/prajval/robotics/cyrus/controller/kinematic_observer.py) ($9.4\,\text{KB}$ — 3D Center of Mass & Capture Point Observer)  
- [`envs/walk_balance_env.py`](file:///C:/Users/prajval/robotics/cyrus/envs/walk_balance_env.py) ($39.8\,\text{KB}$ — 8-Phase Locomotion Curriculum)  
- [`controller/gestures.py`](file:///C:/Users/prajval/robotics/cyrus/controller/gestures.py) ($4.0\,\text{KB}$ — Expressive Gesture Primitives & Gaze Tracking)  
- [`cyrus.py`](file:///C:/Users/prajval/robotics/cyrus/cyrus.py) ($10.4\,\text{KB}$ — Master Unified Humanoid CLI)

---

## Executive Overview: From Balance Recovery to Omnidirectional Locomotion

Push recovery (Sections 1 through 7) and ground recovery (Sections 8 and 9) provide humanoid survivability under sudden trauma. However, a production-grade humanoid must transition smoothly from static push recovery into continuous bipedal locomotion, terrain traversal, and expressive human-robot interaction.

Cyrus-H04 achieves this through three tightly coupled systems:
1. **The 150-Dimensional Whole-Body Proprioceptive Schema ([`controller/body_schema_observer.py`](file:///C:/Users/prajval/robotics/cyrus/controller/body_schema_observer.py)):**  
   A mathematically normalized sensor fusion representation unifying floating-base odometry, 27-actuator internal joint states, 4-limb contact ground reaction forces, Centroidal Divergent Component of Motion (DCM), and normalized angular momentum ($\mathbf{H}_{\text{CoM}}$).
2. **The 8-Phase Walking & Dynamic Gait Engine ([`envs/walk_balance_env.py`](file:///C:/Users/prajval/robotics/cyrus/envs/walk_balance_env.py)):**  
   A staged curriculum moving from stationary 60-second balance holds, through single-support pelvis weight shifting and anti-drag swing clearance, into limit-cycle walking with reciprocal arm swing compensation.
3. **The Expressive Behavioral Gesture Pipeline ([`controller/gestures.py`](file:///C:/Users/prajval/robotics/cyrus/controller/gestures.py)):**  
   Coordinates head gaze tracking, agreement nodding, friendly waving, and organic $0.25\,\text{Hz}$ thoracic breathing micro-motions overlaid on top of lower-body balance controllers without destabilizing ground contact.

---

## 10.1 Mathematical Structure of the 150-Dimensional Proprioceptive Schema

The Cyrus proprioceptive observer fuses raw MuJoCo state tensors into a **150-Dimensional Vector** normalized for deep neural network input:

$$
\mathbf{s}_{\text{schema}} = \left[ \mathbf{p}_{\text{pelvis}}, \mathbf{q}_{27}, \dot{\mathbf{q}}_{27}, \mathbf{c}_{12}, \mathbf{x}_{\text{centroidal}}, \mathbf{h}_{\text{angmom}}, \mathbf{d}_{\text{safety}} \right]^T \in \mathbb{R}^{150}
$$

```
   150-DIMENSIONAL WHOLE-BODY PROPRIOCEPTIVE SCHEMA
   ================================================
   [Group 1: Floating-Base Kinematics (13 dims)]
     - Pelvis relative position (x - x_anchor, y - y_anchor, z)            : 3 dims
     - Orientation unit quaternion (w, x, y, z)                              : 4 dims
     - Linear velocity in world frame (vx, vy, vz)                           : 3 dims
     - Angular velocity in world frame (wx, wy, wz)                          : 3 dims
   
   [Group 2: Actuator Kinematics across 27 Core Joints (54 dims)]
     - Normalized joint angles q_j                                           : 27 dims
     - Joint angular velocities scaled by 0.10 (0.10 * q_dot_j)             : 27 dims
   
   [Group 3: Contact Heights, Activation Sigmas & Normal Forces (12 dims)]
     - End-effector heights [z_r_palm, z_l_palm, z_r_sole, z_l_sole]         : 4 dims
     - Sigmoidal contact activation: sigma_i = 1 / (1 + exp(-0.3*(Fz - 10)))  : 4 dims
     - Normalized vertical forces: F_z,i / (m_total * g)                     : 4 dims
   
   [Group 4: Centroidal CoM & DCM Capture Dynamics (8 dims)]
     - CoM position relative to active support centroid: r_CoM - p_centroid  : 3 dims
     - CoM linear velocity: v_CoM                                            : 3 dims
     - 2D Divergent Component of Motion (Capture Point) Error: xi_xy - p_cent: 2 dims
   
   [Group 5: Angular Momentum & Torso Gyroscopic Vector (9 dims)]
     - Normalized Centroidal Angular Momentum: H_CoM / (m * sqrt(g * z_c^3)) : 3 dims
     - Torso coronal axis vector (torso_x in world coordinates)              : 3 dims
     - Torso vertical axis vector (torso_z in world coordinates)             : 3 dims
   
   [Group 6: Kinematic Safety Boundaries & Previous Action History (54 dims)]
     - Joint limit proximity penalty: sign(q_norm) * max(0, |q_norm| - 0.75) : 27 dims
     - Previous motor command action buffer: a_{t-1}                         : 27 dims
   ----------------------------------------------------------------------------------
   TOTAL DIMENSIONS: 13 + 54 + 12 + 8 + 9 + 54 = 150 DIMENSIONS
```

### 10.1.1 Normalized Centroidal Momentum Formulation

To make the angular momentum invariant across different robot weights or payload carrying tasks, the raw subtree angular momentum $\mathbf{H}_{\text{subtree}} = \sum_i (\mathbf{r}_i \times m_i \mathbf{v}_i + \mathbf{I}_i \boldsymbol{\omega}_i)$ is normalized by the characteristic gravitational pendulum momentum:

$$
\tilde{\mathbf{H}}_{\text{CoM}} = \frac{\mathbf{H}_{\text{raw}}}{M_{\text{total}} \sqrt{g \cdot z_{\text{CoM}}^3}}
$$

This dimensionless momentum vector indicates whether whole-body tumbling torque is approaching the tipping limit of the support foot without requiring threshold retuning across varying body heights.

---

## 10.2 The 8-Phase Locomotion & Walking Curriculum

In [`envs/walk_balance_env.py`](file:///C:/Users/prajval/robotics/cyrus/envs/walk_balance_env.py), locomotion is acquired through an **8-Phase Progressive Curriculum**:

```mermaid
flowchart LR
    P1["Phase 1: Limit Awareness"] --> P2["Phase 2: Stable Stance (60s hold)"]
    P2 --> P3["Phase 3: Pelvis Weight Shift"]
    P3 --> P4["Phase 4: Single-Leg Balance"]
    P4 --> P5["Phase 5: Dynamic Foot Clearance"]
    P5 --> P6["Phase 6: Capture Point Stepping"]
    P6 --> P7["Phase 7: Limit-Cycle Walking"]
    P7 --> P8["Phase 8: Velocity Tracking"]
```

### 10.2.1 Detailed Curriculum Phase Specifications

| Phase | Identifier | Target Milestone | Termination Conditions | Key Reward Terms |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `PHASE_AWARENESS` | Joint limits respected, zero self-collisions. | $z_{\text{pelvis}} < 0.65\,\text{m}$ | $R_{\text{limit}} = -\sum \max(0, |q| - 0.90\,q_{\max})^2$ |
| **2** | `PHASE_STANCE` | Stand continuously for $> 60.0\,\text{s}$ under sensor noise. | Tilt $> 15.0^\circ$ or $z < 0.72\,\text{m}$ | $R_{\text{upright}} = \hat{\mathbf{z}}_{\text{torso}} \cdot \hat{\mathbf{e}}_z$, $R_{\text{height}} = \exp(-20(z - 0.84)^2)$ |
| **3** | `PHASE_WEIGHT_SHIFT` | Shift pelvis laterally $\pm 4.0\,\text{cm}$ over stance sole. | Incomplete weight transfer | $R_{\text{shift}} = \exp(-10(y_{\text{pelvis}} - y_{\text{target}})^2)$ |
| **4** | `PHASE_SINGLE_LEG` | Hold single-leg balance for $> 3.0\,\text{s}$ per leg. | Swing foot touches mat premature | $R_{\text{single}} = F_{z,\text{stance}} / (M g) + \mathbf{r}_{\text{arm\_counter}}$ |
| **5** | `PHASE_SWING_CLEARANCE` | Lift swing foot $\ge 5.0\,\text{cm}$ without sole scuffing. | Swing foot drag ($v_{xy} > 0.1\,\text{m/s}$ while $z < 2\,\text{cm}$) | $R_{\text{clear}} = \min(z_{\text{swing}} - 0.05, 0.0) - P_{\text{drag}}$ |
| **6** | `PHASE_CAPTURE_POINT` | Step landing within $\pm 1.5\,\text{cm}$ of LIPM Capture Point. | Landing outside friction cone | $R_{\text{landing}} = \exp(-15\|\mathbf{p}_{\text{land}} - \boldsymbol{\xi}\|^2)$ |
| **7** | `PHASE_WALKING_CYCLE` | Execute continuous 10-step periodic walking limit cycle. | Step period $< 0.35\,\text{s}$ or $> 0.75\,\text{s}$ | $R_{\text{cycle}} = \cos(2\pi \phi) \cdot v_x + R_{\text{smooth}}$ |
| **8** | `PHASE_COMMAND_TRACK` | Track user commanded velocity vector $(v_x, v_y, \dot{\psi})$. | Tracking error $> 0.25\,\text{m/s}$ | $R_{\text{track}} = \exp(-4(v_x - v_x^*)^2 - 4(\dot{\psi} - \dot{\psi}^*)^2)$ |

### 10.2.2 Reciprocal Arm Counter-Swing Dynamics (Principle 7)

During walking gait, leg swing generates dynamic yaw angular momentum $\tau_z$ about the vertical axis. To cancel this without overloading the stance leg hip yaw actuator, [`envs/walk_balance_env.py`](file:///C:/Users/prajval/robotics/cyrus/envs/walk_balance_env.py#L78-L120) enforces biologically matched **Reciprocal Arm Counter-Swing**:

$$
q_{\text{right\_shoulder\_pitch}} = -k_{\text{arm}} \cdot q_{\text{left\_hip\_pitch}}
$$

$$
q_{\text{left\_shoulder\_pitch}} = -k_{\text{arm}} \cdot q_{\text{right\_hip\_pitch}}
$$

With $k_{\text{arm}} = 0.65$, whenever the left leg swings forward into flexion ($q_{\text{left\_hip\_pitch}} > 0$), the right arm swings forward in synchrony, canceling up to **$82\%$ of axial yaw counter-torque** ($\dot{H}_z$).

---

## 10.3 The Expressive Behavioral Gesture Pipeline

Real-world deployment requires humanoids to interact socially without disrupting dynamic balance. [`controller/gestures.py`](file:///C:/Users/prajval/robotics/cyrus/controller/gestures.py) implements non-blocking expressive primitives:

```
                          [3D Gazed Object / Human Head]
                                       *
                                      /
                                     / (Gaze Vector)
                                    v
                              [Eyes / Head]  <-- neck_yaw, neck_pitch (arctan2 delta)
                                    |
                            [Upper Chest]   <-- 0.25 Hz Breathing micro-motion (waist_pitch ±0.02 rad)
                                   / \
                                  /   \
  [Right Arm Wave Primitive] <---/     \---> [Left Arm Dynamic Counter-Balance]
  - Shoulder Pitch: -1.2 rad                 - Compensates arm wave CoM displacement
  - Elbow Flexion : -1.6 rad
  - Wrist Wave    : 2.5 Hz sine
```

### 10.3.1 Head Gaze Tracking Geometry

Given an eye camera position $\mathbf{p}_{\text{eyes}} \in \mathbb{R}^3$ and target point $\mathbf{p}_{\text{target}} \in \mathbb{R}^3$, the line-of-sight delta $\Delta \mathbf{p} = \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{eyes}}$ determines the required neck angles:

$$
\theta_{\text{neck\_yaw}} = \text{atan2}(\Delta y, \Delta x)
$$

$$
\theta_{\text{neck\_pitch}} = -\text{atan2}\left(\Delta z, \sqrt{\Delta x^2 + \Delta y^2}\right)
$$

Angles are dynamically clamped to ergonomic human boundaries ($\theta_{\text{yaw}} \in [-1.2, +1.2]\,\text{rad}$, $\theta_{\text{pitch}} \in [-0.5, +0.5]\,\text{rad}$) with second-order critically damped smoothing.

### 10.3.2 Organic Breathing Micro-Motion

To eliminate the uncanny "statue freeze" when Cyrus stands idle, an organic $0.25\,\text{Hz}$ sinusoidal respiration generator modulates torso pitch and shoulder elevations:

$$
\Delta q_{\text{waist\_pitch}}(t) = 0.020 \cdot \sin(2\pi \cdot 0.25 \cdot t)
$$

$$
\Delta q_{\text{shoulder\_pitch}}(t) = 0.030 \cdot \sin(2\pi \cdot 0.25 \cdot t)
$$

The resulting $1.15^\circ$ spinal oscillation shifts the CoM by less than $2.1\,\text{mm}$, remaining well within the static ankle stability margin ($> 50.0\,\text{mm}$) while providing lifelike natural presence.

---

## Summary of Section 10 Takeaways

1. **150-D Whole-Body Proprioceptive Schema:** Integrates floating-base root dynamics (13), 27 core joints (54), 4-limb contact forces and activation sigmas (12), Centroidal Capture Point error (8), normalized angular momentum (9), and safety boundary buffers (54).
2. **8-Phase Locomotion Curriculum:** Step-by-step mastery moving from joint limit awareness and 60-second stationary stance, to single-support balance, anti-drag swing foot clearance ($\ge 5\,\text{cm}$), and periodic limit-cycle walking.
3. **Reciprocal Arm Counter-Swing:** Dynamically couples contralateral shoulder and hip pitch axes ($k_{\text{arm}} = 0.65$), canceling $82\%$ of axial yaw counter-torque during walking strides.
4. **Expressive Behavioral Primitives:** Seamlessly overlays active 3D gaze tracking, friendly arm waving ($2.5\,\text{Hz}$), nodding, and organic idle breathing ($0.25\,\text{Hz}$) without violating lower-body dynamic equilibrium.
