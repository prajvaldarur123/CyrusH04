# Cyrus-H04 Technical Deep-Dive — Section 5
## Audit, Forensics & Verification Gatekeepers

**Document ID:** CYRUS-ENG-SEC5-REV1  
**Project:** Cyrus-H04 Full-Scale Autonomous Bipedal Humanoid  
**Author:** Prajval Darur & Robotics Engineering Team  

---

```
                       ZERO-TOLERANCE VERIFICATION PIPELINE
                               
                    ┌──────────────────────────────────────┐
                    │    GPU Swarm Training (JAX / MJX)    │
                    │   1,024 Parallel Robots (16 Steps)   │
                    └──────────────────┬───────────────────┘
                                       │ Candidate Weights (w_cand)
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  5.0-Second Native MuJoCo CPU Audit  │
                    │  (250 Steps @ 50 Hz, 16 Angles 360°) │
                    └──────────────────┬───────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │ Pass Rate >= 85.0% Across 360°?   │
                     └─────────────────┬─────────────────┘
                                       │
                      YES ─────────────┴───────────── NO
                       │                              │
                       ▼                              ▼
          ┌─────────────────────────┐   ┌─────────────────────────┐
          │  PROMOTION APPROVED     │   │   REJECT CANDIDATE      │
          │ • Save to Production    │   │ • Reset Streak to 0     │
          │ • Advance Curriculum    │   │ • Rollback to Safe      │
          │ • Milestone Checkpoint  │   │   Production Weights    │
          └─────────────────────────┘   └─────────────────────────┘
```

---

## 5.1 The 5.0-Second Native MuJoCo Physics Gatekeeper (`run_full_episode_eval`)

### The 16-Step "Illusion Trap" of GPU Rollouts
In massively parallel GPU reinforcement learning, rollouts are collected in short horizons (e.g., $16$ steps = $0.32\,\text{seconds}$) to fit within GPU VRAM:
* When a robot is pushed violently at step 10, it takes approximately $0.60\text{--}1.00\,\text{seconds}$ for the body to physically tip over and hit the floor.
* In a $0.32\,\text{second}$ window, a humanoid that is in the middle of falling may still have its pelvis at $z = 0.75\,\text{m}$ and its torso tilted at only $15^\circ$.
* **The Fatal Flaw:** The GPU training loop evaluates the robot *before it finishes falling*, falsely scoring it as "Standing / Survived"! This led to phantom curriculum promotions where policies that fell over 100% of the time were promoted to higher push speeds.

### The True 5.0-Second Native Physics Verification
Cyrus-H04 completely separates the GPU training loop from the **Verification Gatekeeper**:
1. Every candidate policy is extracted from GPU memory and evaluated inside **native, gold-standard C++ MuJoCo physics** running on the host CPU.
2. **Full Temporal Horizon:** The robot is simulated for a full **$5.0\,\text{seconds}$ ($250$ control steps at $50\,\text{Hz}$, $2,000$ physics integration substeps at $dt = 0.0025\,\text{s}$)**.
3. If the robot loses balance, it has ample physical time to tip over, collide with the floor, and trigger a failure.

### Zero-Tolerance Pass/Fail Thresholds
A robot is marked as **FAILED** if at any millisecond during the 5.0-second episode:
$$\text{Pelvis Height: } z < 0.65\,\text{m} \quad (\text{Nominal: } 0.825\,\text{m})$$
$$\text{Torso Tilt: } \theta_{\text{tilt}} = \sqrt{\theta_{\text{roll}}^2 + \theta_{\text{pitch}}^2} > 35.0^\circ$$

---

## 5.2 The Curriculum Promotion Standard

```
Curriculum Milestone Progression:
0.40 m/s ──────► 0.50 m/s ──────► 0.60 m/s ──────► 0.70 m/s ──────► 0.80 m/s ──────► 1.00 m/s
```

In Cyrus-H04, policy promotion is governed by an automated, unyielding gatekeeper:
* **The 20-Streak Requirement:** The policy must first survive 20 consecutive training iterations on the GPU swarm without collapse.
* **The 85% 360° Physics Audit Standard:**
  The candidate weights are deployed into `run_full_episode_eval` across **16 discrete azimuth angles** uniformly spaced around a circle:
  $$\theta_k = \frac{k}{16} \times 360^\circ, \quad k \in \{0, 1, 2, \dots, 15\}$$
  $$\text{Pass Rate} = \frac{\sum_{k=0}^{15} \mathbb{I}(\text{Survived}_k)}{16} \times 100\%$$
* **Promotion Rule:**
  $$\text{If } \text{Pass Rate} \ge 85.0\% \implies \text{Advance Curriculum Velocity } (+0.05\text{--}0.10\,\text{m/s})$$
  $$\text{If } \text{Pass Rate} < 85.0\% \implies \text{Streak Resets to 0, Training Continues at Current Velocity}$$

---

## 5.3 Directional Forensic Radar

```
                          FRONT (315° - 45°)
                                  ▲
                                  │  100.0%
                                  │
    LEFT (225° - 315°) ◄──────────┼──────────► RIGHT (45° - 135°)
          100.0%                  │                 92.0%
                                  │
                                  ▼  88.0%
                          BACK (135° - 225°)
```

### Deconstructing 360-Degree Fall Metrics
When training humanoids, looking only at an overall survival percentage hides directional blind spots. Cyrus decomposes every failure into an independent **4-Quadrant Forensic Radar**:

| Quadrant | Azimuth Angle Span ($\theta$) | Primary Fall Mechanics | Key Compensating Controller |
| :--- | :--- | :--- | :--- |
| **Front** | $315^\circ \le \theta < 45^\circ$ | Pushed forward; face-plant risk. | Ankle plantarflexion + Backward arm swing + Forward step. |
| **Right** | $45^\circ \le \theta < 135^\circ$ | Pushed left-to-right; rightward roll over. | Left arm lateral flaring + Stance knee dip + Rightward step. |
| **Back** | $135^\circ \le \theta < 225^\circ$ | Pushed backward; heel tip-over, foot scuffing. | Ankle dorsiflexion + Forward arm swing + Backward step with $+5\,\text{cm}$ clearance. |
| **Left** | $225^\circ \le \theta < 315^\circ$ | Pushed right-to-left; leftward roll over. | Right arm lateral flaring + Stance knee dip + Leftward step. |

---

## 5.4 Checkpoint Management & Production Registry (`checkpoints/`)

```
checkpoints/
├── balance_12dof_production.npz          # Gold-standard frozen production policy
├── balance_12dof_weights.npz             # Active running policy weights
├── candidate_balance_active.npz          # Pre-audit candidate weights undergoing verification
├── cyrus_balance_0.40ms_mastered.npz     # Mastered 0.40 m/s milestone
├── cyrus_balance_0.50ms_mastered.npz     # Mastered 0.50 m/s milestone
└── cyrus_balance_0.60ms_mastered.npz     # Mastered 0.60 m/s milestone
```

* **Atomic Checkpointing:** Weights are saved in structured NumPy `.npz` binary tensors with key-value hashes matching the Flax neural network parameter dictionary.
* **Mastered Milestones:** When a curriculum push speed is mastered with $\ge 85\%$ survival on 5.0-second evaluation, a permanent immutable milestone checkpoint is written to disk (e.g. `cyrus_balance_0.60ms_mastered.npz`).
* **Instant Recovery:** If a training run suffers from policy divergence or gradient explosion, the trainer instantly reloads from the last mastered milestone rather than starting from scratch.

---

## 5.5 What Was Missing: The 3 Critical Forensic Discoveries & Fixes

### 1. The Vectorized Reset Shove Memory Bug
During early forensic testing, the telemetry radar showed:
$$\text{Front: } 60\%, \quad \text{Back: } 100\%, \quad \text{Left: } 100\%, \quad \text{Right: } 100\%$$
Every single fall was being recorded as a "Front" fall, while Back/Left/Right had perfect 100% scores, which was physically impossible!

* **The Root Cause:** When an environment experienced a terminal fall and reset, the vectorized environment reset its active push vector to:
  $$\mathbf{v}_{\text{push}} = [0.0, 0.0, 0.0]$$
  When the radar computed the angle:
  $$\theta = \text{atan2}(v_y, v_x) = \text{atan2}(0.0, 0.0) = 0.0^\circ$$
  An angle of $0.0^\circ$ falls directly into the **Front Quadrant** ($315^\circ\text{--}45^\circ$)! Every robot that died anywhere was being categorized as a Front death.
* **The Fix:** The active shove vector is preserved across reset boundaries, and evaluated only when $\text{push\_mag} > 0.05\,\text{m/s}$. Directional reporting became $100\%$ forensic.

### 2. Automatic Rollback Protection
If an aggressive gradient step causes the policy's survival rate to plummet by more than $20\%$ in a single iteration, candidate weights are immediately discarded, and weights are rolled back to `balance_12dof_production.npz`.

### 3. Contact Force Saturation Warning Monitor
During the 5.0-second audit, normal ground reaction forces $F_z$ are monitored:
* If $F_z > 3.0 \times M \cdot g$ ($> 925\,\text{N}$), a **Landing Shock Warning** is flagged, indicating that the leg is landing too rigidly and needs higher compliant damping.

---

## Summary of Section 5

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SECTION 5 SUMMARY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. 5.0-Second Audit:   250 steps (2,000 substeps) in native C++ MuJoCo on CPU    │
│ 2. Failure Criteria:   Pelvis z < 0.65 m or Torso tilt > 35.0°                   │
│ 3. Promotion Standard: Requires ≥ 85.0% survival across all 16 angles (0°-360°)  │
│ 4. Forensic Radar:     Deconstructs Front, Left, Back, Right quadrants           │
│ 5. Reset Memory Fix:   Preserves shove vector across reset; eliminates 0° bias   │
│ 6. Checkpoints:        Production, candidate, and mastered milestone archives    │
│ 7. Rollback Safety:    Automatic recovery if candidate weights degrade           │
│ 8. Force Saturation:   Monitors peak landing impacts (<925 N normal force)       │
└──────────────────────────────────────────────────────────────────────────────────┘
```
