# Cyrus-H04 Technical Deep-Dive — Section 4
## Hardcore Training & Perturbation Regime

**Document ID:** CYRUS-ENG-SEC4-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                   CHRONOLOGICAL TIMELINE OF A HARDCORE ROLLOUT EPISODE
                       (Horizon: 500 Steps = 10.00 Seconds at 50 Hz)
                       
  Step 0       Step 12         Step 15       Step 35       Step 75       Step 95
  (t=0.00s)    (t=0.24s)       (t=0.30s)     (t=0.70s)     (t=1.50s)     (t=1.90s)
    │             │               │             │             │             │
    ▼             ▼               ▼             ▼             ▼             ▼
┌─────────┐   ┌─────────┐     ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Spawn   │──►│ Contact │────►│ Primary │──►│ Double  │──►│ Primary │──►│ Double  │──► ...
│ Pose C  │   │ Settle  │     │ Shock 1 │   │ Tap 1   │   │ Shock 2 │   │ Tap 2   │
└─────────┘   └─────────┘     └─────────┘   └─────────┘   └─────────┘   └─────────┘
                   │
                   └────────────────────────────────────────────────────────────►
                       Continuous Stochastic Background Turbulence ~ N(0, σ²)
                            (Persistent Horizontal Force Chatter Active)
```

---

## 4.1 Contact Settling Optimization (12 Steps vs. 100 Steps)

### The Problem with the 2.0-Second (100-Step) Delay
In standard physics simulations (MuJoCo, Isaac Gym), humanoids are initialized with their soles slightly above the floor ($1\text{--}2\,\text{cm}$) to prevent mesh interpenetration at timestep zero. 

Earlier implementations waited until **step 100 ($t = 2.00\,\text{s}$)** before applying any external force. This created a catastrophic Reinforcement Learning failure mode known as **"The Statue Overfitting Trap"**:
1. **Pose Penalty Domination:** For the first 99 steps ($2.0\,\text{seconds}$), the PPO policy received maximum reward for keeping joint velocities at zero and holding the nominal standing posture:
   $$r_{\text{standing}} = 1.0 - 0.012 \sum_i (q_i - q_{\text{nom}, i})^2 - 0.002 \sum_i \dot{q}_i^2$$
2. **Distribution Collapse:** The policy learned that the best strategy in life was to **stiffen up like a mannequin**. When a violent push finally struck at $t = 2.0\,\text{s}$, the policy resisted lifting its legs because lifting a foot incurred immediate penalties. The robot locked its knees and tipped over like a falling tree.
3. **Zero Early Shock Robustness:** If Cyrus experienced an impulse at $t = 0.5\,\text{s}$ (e.g. dropped onto an uneven floor or bumped on startup), it instantly collapsed because it had never experienced early perturbations during training.

### The Physics of 12-Step Settling ($0.24\,\text{s}$)
Under Earth gravity ($g = 9.80665\,\text{m/s}^2$), an object dropped from $h_0 = 1.5\,\text{cm}$ falls according to:
$$t_{\text{impact}} = \sqrt{\frac{2 h_0}{g}} = \sqrt{\frac{2 \times 0.015\,\text{m}}{9.80665\,\text{m/s}^2}} \approx 0.055\,\text{seconds}$$
With critically damped sole restitution parameters (`solref = [0.004, 1.0]`), normal contact forces stabilize and ground reaction vibrations decay within $4\times$ the damping time constant ($4 \times 0.04\,\text{s} = 0.16\,\text{s}$).

$$\text{Total Time to Flat Equilibrium} = 0.055\,\text{s} + 0.160\,\text{s} \approx 0.215\,\text{seconds} \quad (\approx 11\text{ steps at } 50\,\text{Hz})$$

* **The Cyrus Formulation:** Contact settling grace period was cut from 100 steps ($2.00\,\text{s}$) down to **exactly 12 steps ($0.24\,\text{s}$)**.
* **Result:** Soles are fully flat, normal forces are active, and Cyrus is forced to defend its balance from the 15th millisecond of life onward.

---

## 4.2 Early Impulse Shock Cascades

```
Shove Schedule (Steps):
Primary Shocks:    15 (0.30s) ──►  75 (1.50s) ──► 140 (2.80s) ──► 220 (4.40s) ──► 300 (6.00s)
Secondary Shocks:  35 (0.70s) ──►  95 (1.90s) ──► 160 (3.20s) ──► 240 (4.80s)
```

In the real world, humanoids are rarely pushed only once into an empty, calm room. Real environments feature:
* Crowded spaces with recurring bumps
* Rebound collisions after taking a step
* Secondary pushes before dual-support equilibrium is fully re-established

### 1. Primary Shock Wave ($t = 0.30\,\text{s}$, Step 15)
Triggered immediately after sole settling ($0.30\,\text{s}$). Cyrus must immediately fire its Capture Point reflex and residual compliance network to execute a recovery step while barely out of its spawn posture.

### 2. Secondary "Double-Tap" Shocks ($t = 0.70\,\text{s}$, Step 35)
Triggered just $20$ steps ($400\,\text{ms}$) after the primary shock. 
* At $t = 0.70\,\text{s}$, Cyrus has just landed its swing foot and is in the middle of weight-transfer absorption.
* The secondary shock hits **before the biped has settled back to quiet standing**.
* This forces the neural network to learn **chained multi-step recoveries**, preventing the robot from getting "stuck" after its first step.

---

## 4.3 Continuous Stochastic Background Turbulence

```
   Torso Velocity Noise Vector:
   v_turb = [ v_turb_x, v_turb_y, 0.0 ]^T
   where:
     v_turb_x ~ N(0, σ²),  v_turb_y ~ N(0, σ²)
     σ = 0.020 · clip(v_push, 0.2, 0.8) m/s
```

### Eliminating the "Quiet Stance" Gradient Sink
In PPO, all transitions in a rollout batch are pooled to compute policy updates:
$$\nabla_\theta J(\pi_\theta) = \frac{1}{B} \sum_{k=1}^B \nabla_\theta \log \pi_\theta(a_k \mid s_k) A_k$$
If 95% of transitions feature zero external force, the gradient is overwhelmed by "do nothing" updates.

### The Background Turbulence Solution
Starting at **step 12 ($t = 0.24\,\text{s}$)** and continuing throughout all $500$ steps of the episode, a continuous stochastic horizontal velocity perturbation is injected into the pelvis at every single control step:
$$\mathbf{v}_{\text{turb}} \sim \mathcal{N}\left(0, \sigma_{\text{turb}}^2 \mathbf{I}_{2\times 2}\right), \quad \sigma_{\text{turb}} = 0.020 \cdot \text{clip}(v_{\text{push}}, 0.2, 0.8)\,\text{m/s}$$

* **Biomechanical Effect:** Cyrus's torso experiences persistent, gentle brownian-motion "wind chatter" ($\pm 25\text{--}40\,\text{N}$ effective disturbance force).
* **Policy Effect:** The robot is physically prohibited from relaxing into a frozen statue. It must continuously modulate ankle stiffness, make micro-adjustments to the CoP, and keep leg reflex trajectories primed for instantaneous activation.

---

## 4.4 360-Degree Omnidirectional Disturbances & Rotational Yaw Torques

```
                        0° (Front Push)
                              ▲
                              │
          315° ◄──────────────┼──────────────► 45°
                \             │             /
                 \            │            /
   (Left Push)    \           │           /    (Right Push)
     270° ◄────────┼──────────O──────────┼────────► 90°
                  /           │           \
                 /            │            \
                /             │             \
          225° ◄──────────────┼──────────────► 135°
                              │
                              ▼
                       180° (Back Push)
```

### 1. Full 360° Azimuth Sampling
In each parallel environment, the shove direction angle $\theta$ is sampled from a continuous uniform distribution:
$$\theta \sim \text{Uniform}(0.0, 2\pi)$$
$$\mathbf{v}_{\text{shove}} = \left[ v_{\text{push}} \cos(\theta), \ v_{\text{push}} \sin(\theta), \ 0 \right]^T$$
This ensures that the policy has no blind spots: Front, Back, Left, Right, and diagonal pushes (Front-Left, Back-Right) are sampled with equal probability.

### 2. Rotational Yaw Moment Decoupling
Real pushes rarely hit exactly through the Center of Mass; they almost always strike off-center (e.g. against one shoulder), inducing an angular yaw twist ($\dot{\omega}_z$).

Cyrus injects an instantaneous rotational velocity impulse:
$$\omega_{\text{yaw}} \sim \text{Uniform}(-0.05, +0.05) \cdot v_{\text{push}}\,\text{rad/s}$$
$$\mathbf{qvel}_{\text{injected}} = \left[ v_x, \ v_y, \ 0, \ 0, \ \omega_{\text{yaw}}, \ 0 \right]^T$$
This forces the policy to learn whole-body coordination: counter-rotating the shoulders and modulating hip yaw to cancel spin before it breaks ground contact friction.

---

## 4.5 What Was Missing: The 3 Vital Additions to Section 4

To ensure Section 4 covers the complete training regime, three critical mechanisms operate inside the loop:

### 1. Dynamic Exploration Noise Decay Schedule
Early in training, the policy needs large exploration noise to explore compliant foot placement. Once basic balance is discovered, high exploration noise causes high-frequency motor vibration (jitter).

Cyrus implements an exponential exploration decay schedule:
$$\sigma_{\text{noise}}(k) = \max\left(0.04, \ 0.08 \times 0.96^{k-1}\right)$$
* **Iteration 1:** $\sigma = 0.080\,\text{rad}$ (high exploration authority to discover landing angles).
* **Iteration 20:** $\sigma = 0.040\,\text{rad}$ (tight, deterministic motor control eliminating standing jitter).

### 2. The Dynamic Curriculum Ladder
Pushes are not applied at maximum intensity from Day 1. The shove velocity $v_{\text{push}}$ follows a zero-tolerance progression ladder:
$$0.40\,\text{m/s} \xrightarrow[\ge 85\% \text{ eval}]{20 \text{ clean streaks}} 0.50\,\text{m/s} \xrightarrow[\ge 85\% \text{ eval}]{20 \text{ clean streaks}} 0.60\,\text{m/s} \xrightarrow[\ge 85\% \text{ eval}]{20 \text{ clean streaks}} 0.80\,\text{m/s} \xrightarrow[\ge 85\% \text{ eval}]{20 \text{ clean streaks}} 1.00\,\text{m/s}$$
* **Zero-Tolerance Gate:** If even a single robot falls during the 16-angle MuJoCo 5.0-second audit, the streak counter resets to 0.

### 3. Dirac-Delta Velocity Jump vs. Multi-Substep Decay
Impulse shoves are injected as a mathematical **Dirac-delta velocity jump** at the start of an RL step:
$$\mathbf{qvel}^+ = \mathbf{qvel}^- + \mathbf{v}_{\text{shove}}$$
Over the subsequent 8 physics substeps ($8 \times 0.0025\,\text{s} = 0.020\,\text{s}$), the natural ground contact friction and joint damping absorb the shock, accurately mimicking the impact dynamics of a physical strike.

---

## Summary of Section 4

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 4 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Contact Settling:   Cut to 12 steps (0.24s); eliminates "statue mode" trap    │
│ 2. Primary Shocks:     Start at step 15 (0.30s), repeat at 75, 140, 220, 300     │
│ 3. Double-Tap Shocks:  Hit at steps 35, 95, 160, 240 to test chained stepping   │
│ 4. Turbulence Noise:   Persistent v_turb ~ N(0, σ²) from step 12 across 500 steps│
│ 5. 360° Omnidirectional: Full uniform azimuth θ ∈ [0, 2π) with yaw torque spin  │
│ 6. Noise Decay:        σ drops from 0.08 -> 0.04 to eliminate standing jitter    │
│ 7. Curriculum Ladder:  0.40 -> 0.50 -> 0.60 -> 0.80 -> 1.00 m/s with 85% gate   │
│ 8. Dirac-Delta Shock:  Instantaneous velocity impulse across 8 MJX substeps      │
└──────────────────────────────────────────────────────────────────────────────────┘
```
