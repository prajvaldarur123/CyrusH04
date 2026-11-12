# Cyrus-H04 Technical Deep-Dive — Section 3
## GPU Swarm Reinforcement Learning Engine

**Document ID:** CYRUS-ENG-SEC3-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                  ASYMMETRIC RESIDUAL ACTOR-CRITIC TRAINING PIPELINE
                      (1,024 Parallel Humanoids on NVIDIA RTX 3050)
                      
   ┌────────────────────────────────────────────────────────┐
   │        1,024 Parallel MuJoCo MJX Environments          │
   │        (Control Rate: 50 Hz, Physics Substeps: 8)      │
   └───────────┬────────────────────────────────┬───────────┘
               │                                │
     [Partial Observation]            [Privileged Ground Truth]
     • 3-Step History (141-dim)       • Ground Reaction Forces
     • IMU Gyro & Gravity Vector      • Friction Coefficient μ
     • Joint Positions & Velocities   • Active Shove Vector v_push
               │                                │
               ▼                                ▼
     ┌───────────────────┐            ┌───────────────────┐
     │   ACTOR NETWORK   │            │  CRITIC NETWORK   │
     │  MLP (256 x 128)  │            │  MLP (256 x 128)  │
     └─────────┬─────────┘            └─────────┬─────────┘
               │                                │
        a_residual (20%)                  Value Baseline V(s)
               │                                │
               ▼                                ▼
     ┌───────────────────┐            ┌───────────────────┐
     │  q_reflex (80%)   │            │   PPO Surrogate   │
     │  (Analytical IK)  │            │   GAE Advantage   │
     └─────────┬─────────┘            └───────────────────┘
               │
               ▼
     u_target = clip(q_reflex + a_residual ⊙ v_scale)
```

---

## 3.1 The Residual Reinforcement Learning Formulation

### Why Pure "Tabula Rasa" RL Fails for Humanoids
In classical machine learning, "tabula rasa" (blank slate) policies initialize a neural network with random Gaussian weights and task it with learning motor positions from pure trial-and-error:
$$\mathbf{u}_{\text{motor}} = \pi_\theta(\mathbf{s})$$

When applied to 12-DoF or 20-DoF full-scale bipedal humanoids, this approach suffers from three fatal limitations:
1. **The Kinematic Search Space Trap:** A randomly initialized Multilayer Perceptron (MLP) does not know that legs must bend at the knee or that ankles must stay parallel to the floor. Discovering non-linear 3D trigonometry (law of cosines, reference frame transformations) requires hundreds of millions of failed steps.
2. **Action Resolution Mismatch:** Quiet standing requires micro-radian precision ($\pm 0.02\text{--}0.05\,\text{rad}$) to prevent jitter, whereas push recovery requires explosive macro-swings ($+0.50\,\text{rad}$ hip pitch, $+0.60\,\text{rad}$ knee). A single action scale cannot produce both.
3. **Training Instability:** If the neural network outputs an uncoordinated leg motion, the robot crashes into the ground, truncating the episode before the policy can learn anything useful.

### The Cyrus Residual Coupling Architecture
Cyrus-H04 solves this by marrying analytical mechanics with deep learning:
$$\mathbf{u}_{\text{target}} = \text{clip}\left( \mathbf{q}_{\text{reflex}}(\mathbf{s}) + \mathbf{a}_{\text{residual}} \odot \mathbf{v}_{\text{scale}}, \ \mathbf{u}_{\text{min}}, \ \mathbf{u}_{\text{max}} \right)$$

* **The Analytical Reflex ($\mathbf{q}_{\text{reflex}}$, $80\%$ of Control Effort):**
  The closed-form 3D Capture Point inverse kinematics solver instantly calculates the exact landing position where physics dictates the foot must go. On Day 1, with zero training, the robot already knows how to step.
* **The Neural Network Residual ($\mathbf{a}_{\text{residual}} \odot \mathbf{v}_{\text{scale}}$, $20\%$ Compliance):**
  The neural network acts as a **compliance and contact compensator**. It does not need to learn how to swing a leg; instead, it learns:
  - Modulating joint stiffness during foot touchdown to absorb ground impact shock.
  - Fine-tuning ankle angle to match uneven ground surfaces.
  - Modulating stance-leg push-off forces against varying ground friction coefficients ($\mu$).

### Joint-Specific Action Scaling Vector ($\mathbf{v}_{\text{scale}}$)
Residual authority is bounded per joint according to physical compliance needs:

```python
RESIDUAL_SCALE_VEC = jnp.array([
    0.08, 0.08, 0.15, 0.18, 0.12, 0.08,  # Right: yaw, roll, pitch, knee, ank_pitch, ank_roll
    0.08, 0.08, 0.15, 0.18, 0.12, 0.08   # Left:  yaw, roll, pitch, knee, ank_pitch, ank_roll
], dtype=jnp.float32)
```
* **Knee ($0.18\,\text{rad} \approx 10.3^\circ$):** Highest residual authority, allowing the network to compliantly yield during high-impact landings.
* **Hip Pitch ($0.15\,\text{rad} \approx 8.6^\circ$):** Adjusts step length for terrain variation.
* **Hip Roll / Yaw ($0.08\,\text{rad} \approx 4.6^\circ$):** Highly constrained residual authority, maintaining structural stiffness against lateral roll-over.

---

## 3.2 Massive Parallel GPU Swarm (1,024 Humanoids in JAX/MJX)

```
        1,024 Environments × 16 Rollout Steps = 16,384 Transitions / PPO Update
```

### The GPU Acceleration Breakthrough
Traditional robotics training (using CPU-based physics engines like standard MuJoCo, Bullet, or ODE) required CPU multiprocessing clusters with dozens of workers communicating via slow inter-process IPC pipes:
* Communication latency between Python and CPU C-libraries created a severe bottleneck.
* Collecting $16,384$ transitions required $45\text{--}90\,\text{seconds}$ on an 8-core CPU.

Cyrus-H04 runs entirely within **Google DeepMind's MuJoCo MJX** implemented in pure JAX:
* **Zero IPC Overhead:** The physics simulation, neural network forward pass, GAE advantage estimation, and PPO gradient backpropagation all execute **inside the GPU's onboard VRAM memory**.
* **Throughput:** $1,024$ humanoids take 16 physics steps ($20\,\text{ms}$ each) simultaneously in **$0.48\,\text{seconds}$**.
* **Total Training Speed:** 1 complete PPO iteration (rollout + 2 SGD epochs + advantage calculation) takes **$< 1.2\,\text{seconds}$** on a standard NVIDIA GeForce RTX 3050 Laptop GPU!

---

## 3.3 Asymmetric Actor-Critic Architecture

In real-world deployment, a humanoid robot only has access to noisy onboard sensors (IMU, joint encoders, foot contact switches). However, during simulation training, the engine has complete omniscience over the physics state.

Cyrus leverages **Asymmetric Actor-Critic (Pinto et al. 2017)**:

```
[ Real World ]  ──► Only Actor runs onboard robot (Reads noisy sensors only)
[ Simulation ]  ──► Critic has full omniscient vision during training (Guides Actor)
```

### 1. The Actor Observation Space ($141$ Dimensions)
To capture temporal dynamics (such as acceleration and rate-of-change) without requiring recurrent neural networks (RNNs/LSTMs), Cyrus uses a **3-step sliding history buffer** ($3 \times 47 = 141$ features):

Each 47-dimensional temporal slice contains:
* **Torso Gravity Vector ($3$ dims):** $\mathbf{g}_{\text{proj}} = \mathbf{R}_{\text{torso}}^T [0, 0, -1]^T$ (indicates tilt direction).
* **Torso Angular Velocity ($3$ dims):** $\boldsymbol{\omega}_{\text{torso}} \in \mathbb{R}^3$ from IMU gyroscope.
* **Torso Linear Velocity ($3$ dims):** $\mathbf{v}_{\text{torso}} \in \mathbb{R}^3$ in body frame.
* **Joint Positions ($12$ dims):** $\mathbf{q}_{\text{act}} \in \mathbb{R}^{12}$ for all 12 leg joints.
* **Scaled Joint Velocities ($12$ dims):** $\dot{\mathbf{q}}_{\text{act}} \times 0.1 \in \mathbb{R}^{12}$ (scaled to balance input variance).
* **Previous Filtered Action ($12$ dims):** $\mathbf{a}_{\text{prev}} \in \mathbb{R}^{12}$ (enforces temporal smoothness).
* **Sole Contact Flags ($2$ dims):** Binary touch indicators $[c_{\text{right}}, c_{\text{left}}]$ from sole contact sensors.

### 2. The Critic Privileged State Space ($171$ Dimensions)
The Critic evaluates the value function $V(s)$ to compute GAE advantages. It receives all 141 actor dimensions **plus 30 privileged physical variables**:
* **True World Floating-Base Position & Orientation ($7$ dims):** $[x, y, z, q_w, q_x, q_y, q_z]$.
* **True Linear & Angular Velocities in World Frame ($6$ dims).**
* **Active Shove Velocity Vector ($3$ dims):** $\mathbf{v}_{\text{shove}} = [v_x, v_y, \omega_z]$ (tells the critic exact push magnitude).
* **Ground Reaction Forces ($6$ dims):** 3D contact forces on right and left soles.
* **Ground Friction Coefficient ($1$ dim):** Real-time terrain friction $\mu \in [0.4, 1.2]$.
* **Center of Mass State ($7$ dims):** True $\mathbf{r}_{\text{CoM}}$, $\mathbf{v}_{\text{CoM}}$, and Capture Point $\boldsymbol{\xi}$.

---

## 3.4 State-Conditioned Dynamic Reward Gating

### The "Statue vs. Sprinter" Gradient Conflict
In classical locomotion RL, reward functions penalize deviations from the nominal stance:
$$\mathcal{P}_{\text{pose}} = 0.012 \sum_{i=1}^{12} (q_i - q_{\text{nom}, i})^2, \quad \mathcal{P}_{\text{load}} = 0.010 (z_{\text{right\_sole}} - z_{\text{left\_sole}})^2$$

Now consider what a physical recovery step requires:
* The pelvis must drop by $4\text{--}6\,\text{cm}$ (violating height reward).
* One foot must lift $8\,\text{cm}$ into the air (violating load asymmetry penalty).
* The hip and knee must rotate by $30^\circ$ ($0.50\,\text{rad}$) (violating pose penalty).

Because humanoids spend $90\%$ of rollout time standing quietly, $90\%$ of the gradient updates tell the neural network: *"Lock your knees, never lift a leg, stay at nominal pose at all costs!"* When pushed, the policy actively fights against stepping.

### The State-Conditioned Solution
Cyrus-H04 implements dynamic reward gating:
$$r_t = r_{\text{height}} + r_{\text{upright}} + r_{\text{alive}} - \underbrace{(1 - \mathbb{I}_{\text{needs\_step}}) \cdot (\mathcal{P}_{\text{pose}} + \mathcal{P}_{\text{load}})}_{\text{Zeroed out during stepping!}} - \mathcal{P}_{\text{action\_rate}}$$

$$\mathbb{I}_{\text{needs\_step}} = \begin{cases} 
1 & \text{if } \boldsymbol{\xi} \text{ exits support polygon} \\ 
0 & \text{if } \boldsymbol{\xi} \text{ is inside support polygon} 
\end{cases}$$

* **When Quiet Standing ($\mathbb{I}_{\text{needs\_step}} = 0$):** Pose and symmetry penalties are active, forcing Cyrus to stand still with zero jitter.
* **When Recovery Stepping ($\mathbb{I}_{\text{needs\_step}} = 1$):** Pose and symmetry penalties are **mathematically zeroed ($0.0$)**. The network is free to lower its hips, crouch, and explosively swing its leg without incurring negative reward penalties!

---

## 3.5 Single-Compile Fast XLA Graph

In JAX, dynamic shapes or variable loop bounds force the XLA compiler to trigger expensive re-compilations (stalling the GPU for 30–60 seconds).

Cyrus-H04 ensures **Zero Re-compilation Overhead**:
1. **Pre-computed Nominal Constants:** Nominal standing poses, joint limits, and kinematic link lengths are baked into static device tensors at initialization.
2. **Fixed Static Dimensions:** All mini-batches ($N = 8,192$) and rollout horizons ($T = 16$) are statically shaped.
3. **Optax Warmup-Cosine Schedule:**
   The learning rate evolves via a pure functional Optax schedule without recompilation:
   $$\eta(t) = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$
   Starting at $\eta_{\text{max}} = 3.0 \times 10^{-4}$ with a 50-update linear warmup.

---

## 3.6 What Was Missing: The 3 Vital Additions to Section 3

### 1. Generalized Advantage Estimation (GAE) with Truncation vs. Terminal Split
* **Terminal Fall:** If pelvis height drops below $0.70\,\text{m}$ or torso tilts beyond $30^\circ$, the state is terminal:
  $$V_{\text{target}} = 0.0 \quad (\text{No future value})$$
* **Timeout Truncation (Step 500):** If the humanoid survives the full 10.0 seconds, the value function bootstraps future rewards:
  $$V_{\text{target}} = V(s_{500})$$
  This prevents the policy from learning a "suicide strategy" near the end of an episode.

### 2. PPO Clipped Objective with Action-Rate Penalty
$$\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] + c_1 \mathcal{L}_V - c_2 \mathcal{S}_{\text{entropy}}$$
where $\epsilon = 0.20$, $c_1 = 0.50$, $c_2 = 0.001$.

### 3. GPU VRAM Memory Budgeting & Host GC
To prevent WSL2 DirectX driver memory exhaustion on laptop GPUs (6.0 GB VRAM):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export MALLOC_TRIM_THRESHOLD_=65536
```
Explicit garbage collection `del m_dict, metrics_cpu` releases host RAM on every iteration, preventing memory leaks during long multi-hour runs.

---

## Summary of Section 3

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 3 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Residual Coupling:  u = clip(q_reflex + a_residual ⊙ v_scale) (80% / 20%)     │
│ 2. Swarm Parallelism:  1,024 humanoids on GPU; 16,384 transitions per iteration  │
│ 3. Actor Input:        141 dims (3-step sliding history of 47 features)          │
│ 4. Critic Input:       171 dims (Omniscient privileged state: forces, μ, v_push) │
│ 5. Dynamic Gating:     P_pose and P_load = 0.0 when needs_step=True              │
│ 6. Single XLA Graph:   Optax warmup-cosine; 0 dynamic recompilations (<1.2s/iter)│
│ 7. GAE Bootstrapping:  λ = 0.95, γ = 0.99 with strict terminal/truncation split  │
│ 8. VRAM Optimization:  Bounded at 60% memory fraction; explicit GC release       │
└──────────────────────────────────────────────────────────────────────────────────┘
```
