# Cyrus Humanoid (CyrusH04)

Autonomous Bipedal Robotics Platform & Asymmetric Actor-Critic GPU Swarm Control.

---

## Overview

**CyrusH04** is an advanced humanoid robotics project focusing on robust 360° omnidirectional push recovery, dynamic equilibrium, and whole-body proprioceptive balance control.

The system features:
- **Parallel GPU Swarm**: 1,024 concurrent bipedal agents simulated via MuJoCo XLA (`mjx`) on RTX hardware.
- **High-Frequency Control Loop**: 50 Hz actor-critic control rate synchronized with 250–500 Hz physics substeps.
- **Asymmetric Actor-Critic RL**: Privileged critic observations paired with strictly observable proprioceptive actor policies.
- **Bilateral Symmetry Enforcement**: Closed-form mathematical constraints eliminating asymmetric drift.
- **Multi-Verification Audit Engine**: Automated 200-episode stress testing suite evaluating fall rate, survival duration, torque limits, and settling times.

---

## Project Structure

```text
cyrus/
├── models/             # MuJoCo XML models (12-DOF biped, 14-joint hands)
├── controller/         # Real-time balance and locomotion controllers
├── envs/               # Gym/MJX simulation environments
├── training/           # Swarm reinforcement learning scripts
├── checkpoints/        # Production weight matrices (.npz) & ledgers
├── cad/                # CAD references and dimensional layouts
├── benchmark/          # Performance benchmarks and physics tests
├── scripts/            # Helper automation and verification tools
└── tests/              # Unit tests and biomechanical audits
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- MuJoCo 3.x
- JAX / MJX (with CUDA support)

### Run Physical Audit
To verify balance recovery against 200 randomized push episodes:
```bash
python run_gate4_audit.py
```

### Run Live Interactive MuJoCo Viewer
```bash
python watch_balance_live_mujoco.py
```

---

## License
Proprietary & Confidential - Cyrus Robotics Project.
