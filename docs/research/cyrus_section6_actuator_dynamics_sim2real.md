# Cyrus-H04 Technical Deep-Dive — Section 6
## Actuator Dynamics & Sim-to-Real Hardware Layer

**Document ID:** CYRUS-ENG-SEC6-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                        SIM-TO-REAL HARDWARE DEPLOYMENT PIPELINE
                               
       [ Neural Network + Reflex Controller ] (Evaluated at 50 Hz, dt = 20ms)
                         │ Target Angles: q_des
                         ▼
       ┌─────────────────────────────────────────────────────────┐
       │   EMA Actuator Latency Filter (α = 0.40, 20ms buffer)   │
       │   q_filtered = 0.60 · q_prev + 0.40 · q_des             │
       └─────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
       ┌─────────────────────────────────────────────────────────┐
       │     Torque-Speed Saturation & Back-EMF Limiter          │
       │     τ_avail = τ_max · (1 - |ω| / ω_no_load)             │
       └─────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
       ┌─────────────────────────────────────────────────────────┐
       │         Hardware Safety Guard & Watchdog Layer          │
       │   • Thermal limit check     • Velocity runaway cut-off  │
       │   • Soft-limit clamping     • Lost packet auto-damping  │
       └─────────────────────────┬───────────────────────────────┘
                                 │ CAN-FD / EtherCAT Packets (1000 Hz)
                                 ▼
       ┌─────────────────────────────────────────────────────────┐
       │       Physical High-Torque Brushless Actuators          │
       │       (Cyrus-H04 Physical Bipedal Hardware)             │
       └─────────────────────────────────────────────────────────┘
```

---

## 6.1 Torque-Speed Saturation & Back-EMF Physics

### The "Superhuman Motor" Sim-to-Real Trap
In unconstrained physics simulators, actuators are often modeled as ideal mathematical sources that can instantaneously output maximum torque $\tau_{\text{max}}$ regardless of how fast the joint is rotating.

If an RL policy is trained under this assumption:
* It learns to command $150\,\text{N}\cdot\text{m}$ of torque while the knee is spinning at $25\,\text{rad/s}$ ($> 240\,\text{RPM}$).
* **The Physical Reality:** In an electric permanent-magnet brushless motor (BLDC), spinning the rotor induces a **Back-Electromotive Force (Back-EMF)** voltage:
  $$V_{\text{back-EMF}} = K_e \cdot \omega_{\text{rotor}}$$
  As rotor speed increases, the Back-EMF voltage opposes the motor driver's DC bus supply voltage ($V_{\text{bus}} \approx 48\,\text{V}$). The effective voltage available to push current through the winding drops to:
  $$I_{\text{winding}} = \frac{V_{\text{bus}} - K_e \omega}{R_{\text{winding}}}$$
  Since torque is directly proportional to current ($\tau = K_t I$), the torque an electric motor can produce **drops linearly to zero** as speed approaches no-load velocity!
* **The Consequence:** A policy that relies on high torque at high velocity will violently collapse on physical hardware because real motors cannot produce that torque.

```
       Torque (N·m)
       ▲
τ_max  ┼─────────────┐ ◄── Continuous Peak Torque Zone (Low Speed)
       │             │\
       │             │ \
       │   ALLOWED   │  \  Back-EMF De-rating Boundary
       │   TORQUE    │   \ [ τ_avail = τ_max · (1 - ω/ω_no_load) ]
       │   OPERATING │    \
       │   ENVELOPE  │     \
       │             │      \
       └─────────────┴───────┴──────► Angular Velocity ω (rad/s)
                     ω_rated  ω_no_load
```

### Cyrus Actuator Hardware Specifications

| Joint Axis | Peak Torque $\tau_{\text{max}}$ | Continuous Torque $\tau_{\text{cont}}$ | No-Load Speed $\omega_{\text{no\_load}}$ | Gearhead Reduction | Gearhead Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Knee Pitch** | $150.0\,\text{N}\cdot\text{m}$ | $65.0\,\text{N}\cdot\text{m}$ | $22.0\,\text{rad/s}$ ($210\,\text{RPM}$) | $28:1$ | High-torque Cycloidal Pinwheel |
| **Hip Roll & Pitch** | $120.0\,\text{N}\cdot\text{m}$ | $50.0\,\text{N}\cdot\text{m}$ | $26.0\,\text{rad/s}$ ($248\,\text{RPM}$) | $24:1$ | High-stiffness Cycloidal Drive |
| **Ankle Pitch** | $80.0\,\text{N}\cdot\text{m}$ | $35.0\,\text{N}\cdot\text{m}$ | $30.0\,\text{rad/s}$ ($286\,\text{RPM}$) | $18:1$ | Low-inertia Planetary / Cable |
| **Ankle Roll & Hip Yaw** | $60.0\,\text{N}\cdot\text{m}$ | $25.0\,\text{N}\cdot\text{m}$ | $35.0\,\text{rad/s}$ ($334\,\text{RPM}$) | $16:1$ | Lightweight Planetary Drive |

### Torque-Speed Saturation Enforcement Formula
Cyrus models the physical actuator envelope using a strict de-rating equation:
$$\tau_{\text{available}}(\omega) = \text{clip}\left( \tau_{\text{max}} \left( 1 - \frac{|\dot{q}|}{\omega_{\text{no\_load}}} \right), \ 0.0, \ \tau_{\text{max}} \right)$$
$$\tau_{\text{command}} = \text{clip}\left( K_p (q_{\text{des}} - q) - K_d \dot{q}, \ -\tau_{\text{available}}(\dot{q}), \ +\tau_{\text{available}}(\dot{q}) \right)$$
This guarantees that policy maneuvers trained in simulation never demand unphysical electrical power from real hardware.

---

## 6.2 Domain Randomization Suite

To ensure that neural network policies transfer seamlessly from simulation to physical reality without performance degradation (the "Sim-to-Real Gap"), Cyrus trains against a rigorous **Domain Randomization Suite**:

```
                       DOMAIN RANDOMIZATION DISTRIBUTIONS
                               
  Terrain Friction μ      Payload Mass ΔM       IMU Sensor Noise       Latency Buffer
   ~ Uniform(0.4, 1.2)      ~ Uniform(±3 kg)       ~ N(0, 0.008²)       15ms - 35ms Jitter
        │                        │                      │                      │
        ▼                        ▼                      ▼                      ▼
  [ Ice / Linoleum /       [ Battery / Cargo /    [ Gyro Drift &         [ Communication
    Carpet / Concrete ]      Sensor Rigging ]       Accelerometer Jitter ]  CAN Bus Delay ]
```

### 1. Ground Friction Randomization ($\mu \in [0.40, 1.20]$)
In each parallel environment, ground friction is sampled dynamically:
* $\mu = 0.40$: Represents wet tile, polished marble, or dusty linoleum.
* $\mu = 0.70$: Represents standard rubber gym turf / indoor lab flooring.
* $\mu = 1.20$: Represents high-grip outdoor asphalt or coarse concrete.
* **Result:** The residual policy learns compliance strategies that never assume infinite friction, preventing slip-and-fall accidents on slick floors.

### 2. Sensor Noise & IMU Drift Jitter
Physical MEMS IMUs and optical joint encoders suffer from thermal drift and electrical noise:
* **IMU Gyroscope Noise:** Gaussian noise with standard deviation $\sigma_{\text{gyro}} = 0.008\,\text{rad/s}$ added to angular rate measurements.
* **Torso Projected Gravity Jitter:** $\mathbf{g}_{\text{proj}} + \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, 0.008^2 \mathbf{I}_{3\times 3})$.
* **Joint Encoder Quantization:** Small angular quantization noise simulating 14-bit magnetic absolute encoders ($0.00038\,\text{rad}$ resolution).

### 3. Actuator Latency Simulation (EMA Filtering with $\alpha = 0.40$)
In physical hardware, there is an unavoidable delay between the moment the CPU computes a target position and the moment the motor stator creates magnetic flux:
* CAN bus packet transmission delay ($1\text{--}2\,\text{ms}$)
* Motor driver current loop rise time ($3\text{--}5\,\text{ms}$)
* Mechanical backlash and gearhead rotor acceleration time ($10\text{--}15\,\text{ms}$)

Cyrus explicitly simulates this $20\,\text{ms}$ physical delay using an Exponential Moving Average (EMA) low-pass filter:
$$\mathbf{u}_{\text{applied}}(t) = (1 - \alpha) \mathbf{u}_{\text{applied}}(t-1) + \alpha \mathbf{u}_{\text{target}}(t), \quad \alpha = 0.40$$
* A policy trained without this filter learns high-frequency bang-bang actions that shake the robot to pieces on real hardware.
* With $\alpha = 0.40$, the policy is mathematically forced to produce smooth, dynamically feasible motor trajectories.

---

## 6.3 Real-Time Hardware Control Interface Wrapper (`controller/balance_pipeline.py`)

### Bridging Python Simulation to Physical Bus Hardware
On the physical Cyrus humanoid, motor commands are dispatched over **CAN-FD (Controller Area Network Flexible Data-Rate)** or **EtherCAT** industrial fieldbuses at **$1,000\,\text{Hz}$ ($1\,\text{ms}$ bus cycle)**.

`controller/balance_pipeline.py` acts as the production deployment driver:

```
[ Sensor Inputs from Hardware (1000 Hz) ]
  ├── Optical Joint Encoders (q_act, dq_act)
  ├── 6-Axis IMU (Acc, Gyro, Orientation Quaternion)
  └── Foot Sole Contact Microswitches
            │
            ▼ (Decimated to 50 Hz)
[ Cyrus Balance Pipeline Execution (50 Hz, dt = 20 ms) ]
  1. Capture Point Reflex (LIPM Kinematic IK): q_reflex
  2. Residual RL Policy (Onnx / NumPy Inference): a_residual
  3. Action Coupling: u_target = clip(q_reflex + a_residual ⊙ v_scale)
  4. EMA Latency Smoother: u_cmd = 0.60 · u_prev + 0.40 · u_target
            │
            ▼ (Interpolated to 1000 Hz)
[ CAN-FD / EtherCAT Packet Dispatch (1000 Hz, dt = 1 ms) ]
  Dispatches structured motor frames: [ID, Mode, Pos_Target, Vel_Target, Kp, Kd, Torque_FF]
```

### Motor Frame Command Packet Structure
Each actuator receives a standard Unitree/CAN-FD compliant control packet:
```text
struct MotorCommand {
    uint8_t  motor_id;      // Joint identifier (0 to 11)
    uint8_t  mode;          // 0: Idle, 1: FOC Servo, 2: Damped Brake
    float32  q_target;      // Target angular position (rad)
    float32  dq_target;     // Target angular velocity (rad/s)
    float32  kp;            // Proportional stiffness gain (N·m/rad)
    float32  kd;            // Damping gain (N·m·s/rad)
    float32  tau_feedforward;// Feedforward torque bias (N·m)
};
```

---

## 6.4 What Was Missing: The 3 Critical Sim-to-Real Additions

To ensure the hardware layer is completely robust and prevents hardware destruction, three critical safety mechanisms operate inside the real-time pipeline:

### 1. Reflected Rotor Inertia & Cycloidal Friction
In high-ratio geared actuators, the moment of inertia of the electric rotor spinning at high speed reflects back to the output link magnified by the square of the gear ratio $N$:
$$I_{\text{reflected}} = N^2 \cdot I_{\text{rotor}}$$
For Cyrus's knee actuator ($N = 28$):
$$I_{\text{reflected}} = 28^2 \times (1.2 \times 10^{-4}\,\text{kg}\cdot\text{m}^2) \approx 0.094\,\text{kg}\cdot\text{m}^2$$
This reflected inertia resists sudden leg acceleration. Cyrus's simulation explicitly accounts for this rotor inertia, preventing the network from learning accelerations that real gearboxes cannot mechanically deliver.

### 2. Velocity Runaway Protection & Soft-Limit E-Stop
If a sensor packet drops or a foot slips, motors can experience velocity runaway. The hardware wrapper implements an autonomous soft-stop safety layer:
* **Velocity Watchdog:** If any joint exceeds $|\dot{q}| > 12.0\,\text{rad/s}$ ($> 115\,\text{RPM}$), the pipeline immediately overrides the policy and enters **Passive Damping Mode** ($K_p = 0, K_d = 40.0$), safely decelerating the leg.
* **Soft Joint Clamping:** Actuators have hardware buffers $2.0^\circ$ inside physical hard stops to prevent metal-on-metal collisions.

### 3. Thermal Power Budgeting
Continuous high-torque operation causes electrical winding heat:
$$P_{\text{loss}} = I^2 R = \left( \frac{\tau}{K_t} \right)^2 R$$
The pipeline monitors cumulative $I^2 t$ thermal energy. If continuous torque exceeds $\tau_{\text{cont}}$ for $> 3.0\,\text{seconds}$, the maximum allowable torque $\tau_{\text{max}}$ is gracefully de-rated by $30\%$ to protect motor coils from burning.

---

## Summary of Section 6

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 6 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Torque Envelopes:   Knee: 150 N·m, Hip: 120 N·m, Ankle: 80/60 N·m             │
│ 2. Back-EMF Limiter:   τ_avail = τ_max · (1 - |ω| / ω_no_load); no superhuman torq│
│ 3. Domain Rand:        Friction μ ∈ [0.4, 1.2], Mass ±3kg, IMU noise σ = 0.008   │
│ 4. Latency Modeling:   20ms EMA low-pass filter (α = 0.40) eliminates chatter    │
│ 5. Hardware Pipeline:  CAN-FD / EtherCAT 1000 Hz bridge with 50 Hz RL execution  │
│ 6. Reflected Inertia:  N² · I_rotor modeled to match cycloidal dynamics          │
│ 7. E-Stop Watchdog:    Velocity runaway cut-off (|dq| > 12 rad/s -> zero torque) │
│ 8. Thermal Protection: I²t thermal monitoring prevents coil burnout              │
└──────────────────────────────────────────────────────────────────────────────────┘
```
