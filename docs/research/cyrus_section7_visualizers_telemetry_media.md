# Cyrus-H04 Technical Deep-Dive — Section 7
## Visualizers, Telemetry & Media Recordings

**Document ID:** CYRUS-ENG-SEC7-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                        CYRUS-H04 VISUAL & TELEMETRY ECOSYSTEM
                               
                 ┌──────────────────────────────────────────────┐
                 │    Physics Engine & Policy Inference State   │
                 │    (qpos, qvel, CoM, Capture Point ξ, Tau)   │
                 └──────────────────────┬───────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│  PHYSICS VIDEO PROOFS  │   │  INTERACTIVE 3D WEB    │   │  HEADLESS HUD EXPORTER │
│  (docs/simulations/)   │   │  (docs/visualizers/)   │   │  (scripts/simulate...) │
├────────────────────────┤   ├────────────────────────┤   ├────────────────────────┤
│ • simulation_0.60ms.gif│   │ • ghost_trajectory     │   │ • Off-screen EGL/OSMesa│
│ • simulation_0.70ms.gif│   │ • swarm_simulation     │   │ • Live HUD overlays    │
│ (Omnidirectional proof)│   │ • unitree_live_viewport│   │ • Frame-accurate sync  │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘
```

---

## 7.1 Physics Simulation Video Records (`docs/simulations/`)

In robotics research, mathematical logs and loss curves can easily hide physical failure modes (such as foot sliding, unnatural twitching, or torso hyper-extension). Cyrus-H04 pairs every milestone with **frame-accurate, physics-verified visual recordings**:

### 1. `simulation_0.60ms_push.gif` (File Size: 1.34 MB)
* **Disturbance Scenario:** $0.60\,\text{m/s}$ omnidirectional impulse shoves tested across 16 angles.
* **Biomechanical Validation:**
  - Demonstrates the **Tier 2 Whole-Body Momentum Strategy** absorbing diagonal pushes.
  - Proves the **Tier 3 Capture Point Stepping Reflex** executing clean single-step recoveries without foot scuffing.
  - Highlights stance knee compliance ($+0.120\,\text{rad}$) lowering the pelvis from $0.825\,\text{m}$ to $0.780\,\text{m}$ upon impact.

### 2. `simulation_0.70ms_push.gif` (File Size: 3.70 MB)
* **Disturbance Scenario:** High-velocity violent shock ($0.70\,\text{m/s}$ impulse, equivalent to a physical punch or sprint collision).
* **Biomechanical Validation:**
  - Shows dynamic arm flaring ($+45.8^\circ$) canceling the lateral roll overturning moment.
  - Proves the $+5.0\,\text{cm}$ dynamic flight clearance ($z_{\text{clear}} = 0.090\,\text{m}$) preventing backward stumbles.
  - Demonstrates dual-support settling and smooth recovery back to nominal standing Pose C.

---

## 7.2 Interactive 3D Web Visualizers (`docs/visualizers/`)

To enable researchers and collaborators to inspect Cyrus without installing heavy C++ simulators or CUDA drivers, the project features three standalone, WebGL/Three.js-powered interactive visualizers:

```
docs/visualizers/
├── ghost_trajectory_visualizer.html       # Translucent after-images & flight trails
├── swarm_simulation_visualizer.html       # 1,024 parallel humanoid swarm density grid
└── unitree_live_viewport.html             # Real-time 3D interactive mouse-push viewport
```

### 1. Ghost Trajectory Visualizer (`ghost_trajectory_visualizer.html`)
Renders the temporal evolution of the biped during a push recovery step by displaying **translucent "ghost" after-images**:
* **Temporal Ghosting Equation:** The visualizer samples the robot's kinematic pose at $10\,\text{Hz}$ across a sliding $1.5\,\text{second}$ window. Ghost opacity decays exponentially into the past:
  $$\alpha(t) = \alpha_{\text{base}} \cdot \exp\left(-\lambda (t_{\text{current}} - t)\right)$$
* **Trajectory Flight Arc:** Traces the 3D parabolic trajectory of the ankle center in vibrant cyan, visually demonstrating the $+5.0\,\text{cm}$ dynamic clearance and ground landing accuracy.
* **Capture Point Trail:** Renders the 2D path of $\boldsymbol{\xi}(t)$ on the floor in neon red, proving that the swing foot touchdown point directly encloses the divergent component of motion.

### 2. Swarm Simulation Visualizer (`swarm_simulation_visualizer.html`)
Visualizes all **1,024 parallel humanoids** running on the GPU in a synchronized macroscopic grid:
* **Live Status Heatmap:** Each robot is represented by an interactive cell:
  - 🟢 **Green:** Standing upright ($z \ge 0.75\,\text{m}$, tilt $\le 10^\circ$).
  - 🟡 **Amber:** Actively executing a stepping recovery reflex ($0.65\,\text{m} \le z < 0.75\,\text{m}$, tilt $\le 30^\circ$).
  - 🔴 **Red:** Terminal fall ($z < 0.65\,\text{m}$ or tilt $> 35^\circ$).
* **Batch Telemetry:** Displays instantaneous standing percentage, mean pelvic height, and directional radar survival across the entire 16,384-transition rollout.

### 3. Real-Time 3D Viewport (`unitree_live_viewport.html`)
An interactive 3D digital twin of Cyrus-H04 running at 60 FPS in modern web browsers:
* **Interactive Mouse Perturbator:** Allows the user to click on the humanoid's upper torso and drag to apply an instantaneous 3D force vector $\mathbf{F}_{\text{ext}}$.
* **Live Visual Feedback:** Shows Cyrus immediately responding:
  - Small mouse click $\implies$ Feet stay flat, ankles stiffen (Tier 1 Ankle Strategy).
  - Medium mouse drag $\implies$ Arms swing and torso flexes (Tier 2 Momentum Strategy).
  - Large mouse flick $\implies$ Instantaneous knee dip and explosive stepping recovery (Tier 3 Stepping Reflex).

---

## 7.3 What Was Missing: The 3 Critical Visual & Telemetry Systems

To ensure Section 7 is completely comprehensive, three vital systems operate in the background:

### 1. Headless HUD Video Exporter Pipeline (`scripts/simulate_random_pushes.py`)
Rather than manually screen-recording simulations, Cyrus features an automated headless video generation pipeline:
* **Off-Screen EGL / OSMesa Rendering:** Runs directly on Linux/WSL2 without requiring a physical monitor or X11 window server.
* **Dynamic Head-Up Display (HUD) Overlays:** Every frame is rendered with live telemetry text:
  - Real-time simulation clock ($t = 2.450\,\text{s}$)
  - Shove velocity vector ($v_{\text{push}} = 0.60\,\text{m/s}$ at $\theta = 135.0^\circ$)
  - Pelvis height ($z = 0.784\,\text{m}$)
  - Total ground normal force ($F_z = 308.5\,\text{N}$)
* **Optimized Encoding:** Converts raw frames into lightweight WebP and GIF animations with adaptive color palette quantization, keeping file sizes under $4\,\text{MB}$ for effortless sharing and documentation.

### 2. Directional Vector Arrows & Ground Contact Pressure Heatmap
In the interactive viewports:
* Ground reaction forces are rendered as dynamic 3D vectors emerging from the sole sites. Vector height scales linearly with normal contact force ($1.0\,\text{m} \equiv 1,000\,\text{N}$).
* Foot pressure distribution is rendered as an elliptical heatmap under the $21\,\text{cm} \times 10\,\text{cm}$ sole polygon, showing the exact movement of the Center of Pressure (CoP) from heel to toe.

### 3. GitHub Linguist Documentation Vendoring (`.gitattributes`)
Interactive HTML visualizers contain extensive CSS and Three.js JavaScript libraries ($> 500\,\text{KB}$). When committed to a GitHub repository, GitHub's language detection engine (Linguist) previously misidentified the robotics repository as **"55.9% HTML"**, hiding the Python robotics core!

Cyrus resolved this by engineering custom Linguist override directives in `.gitattributes`:
```gitattributes
# GitHub Linguist Language Overrides
docs/** linguist-documentation
docs/** linguist-vendored
*.html -linguist-detectable
*.html linguist-vendored
*.html linguist-documentation
```
* **Result:** GitHub's repository statistics ignore HTML assets and accurately report **100.0% Python**, reflecting the true robotics and machine learning codebase.

---

## Summary of Section 7

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 7 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Video Records:      simulation_0.60ms.gif (1.34 MB), 0.70ms.gif (3.70 MB)     │
│ 2. Ghost Trajectory:   Translucent temporal after-images & parabolic flight arcs │
│ 3. Swarm Dashboard:    Live 1,024-robot density grid (Green / Amber / Red status)│
│ 4. 3D Viewport:        Interactive click-and-drag mouse push perturbation engine │
│ 5. Headless Exporter:  Automated EGL/OSMesa 60 FPS recording with live HUD stats │
│ 6. Contact Heatmaps:   Real-time ground reaction vectors & CoP pressure ellipses │
│ 7. Linguist Vendoring: .gitattributes overrides preserve 100.0% Python metrics   │
└──────────────────────────────────────────────────────────────────────────────────┘
```
