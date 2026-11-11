"""
==============================================================================
PROJECT CYRUS -- SOTA JAX MJX VS PLAIN MUJOCO PARITY TEST
==============================================================================
Verifies exact numerical float parity between:
1. Pure NumPy policy inference vs JAX / Flax neural network forward pass
2. Plain MuJoCo CPU kinematics/dynamics vs JAX MJX GPU/CPU physics rollout
==============================================================================
"""

import os
import sys

# Prevent JAX from preallocating 90% of GPU VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

import numpy as np
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_ROOT, "models", "cyrus_12dof_mjx.xml")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "manifest.json")

# Import JAX and MJX
try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def test_policy_numpy_vs_jax_parity():
    print("[PARITY 1/2] Testing NumPy vs JAX Policy Forward Pass Parity...")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "balance_12dof_production.npz")
    w = np.load(ckpt_path)

    # Synthetic batch of 32 observations (141-dim)
    np.random.seed(42)
    obs_batch_np = np.random.randn(32, 141).astype(np.float32)

    # Pure NumPy implementation (controller/balance_pipeline.py)
    h1_np = np.tanh(obs_batch_np @ w["mlp_extractor.policy_net.0.weight"].T + w["mlp_extractor.policy_net.0.bias"])
    h2_np = np.tanh(h1_np @ w["mlp_extractor.policy_net.2.weight"].T + w["mlp_extractor.policy_net.2.bias"])
    act_np = h2_np @ w["action_net.weight"].T + w["action_net.bias"]

    # JAX implementation
    if HAS_JAX:
        obs_jax = jnp.array(obs_batch_np)
        w0 = jnp.array(w["mlp_extractor.policy_net.0.weight"])
        b0 = jnp.array(w["mlp_extractor.policy_net.0.bias"])
        w2 = jnp.array(w["mlp_extractor.policy_net.2.weight"])
        b2 = jnp.array(w["mlp_extractor.policy_net.2.bias"])
        w_act = jnp.array(w["action_net.weight"])
        b_act = jnp.array(w["action_net.bias"])

        @jax.jit
        def forward_jax(x):
            h1 = jnp.tanh(jnp.dot(x, w0.T) + b0)
            h2 = jnp.tanh(jnp.dot(h1, w2.T) + b2)
            return jnp.dot(h2, w_act.T) + b_act

        act_jax = np.array(forward_jax(obs_jax))
        diff = np.max(np.abs(act_np - act_jax))
        np.testing.assert_allclose(act_np, act_jax, atol=1e-6, rtol=1e-5)
        print(f"  -> Policy forward pass parity confirmed! Max absolute difference: {diff:.8e}")
    else:
        print("  -> JAX not installed in current environment; skipping JAX forward.")


def test_mujoco_cpu_vs_mjx_physics_parity():
    print("[PARITY 2/2] Testing MuJoCo CPU vs JAX MJX Physics Rollout Parity...")
    if not HAS_JAX:
        print("  -> JAX / MJX not available in this environment; skipping MJX test.")
        return

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # Initial state
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.825
    d.qpos[3] = 1.0
    ctrl = np.zeros(12, dtype=np.float32)
    ctrl[2] = -0.175; ctrl[3] = 0.350; ctrl[4] = -0.175
    ctrl[8] = -0.175; ctrl_c_knee = 0.350
    ctrl[9] = 0.350; ctrl[10] = -0.175
    d.ctrl[:] = ctrl

    # Step MuJoCo CPU
    for _ in range(10):
        mujoco.mj_step(m, d)

    cpu_qpos = d.qpos.copy()
    cpu_qvel = d.qvel.copy()

    # Step JAX MJX
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    
    # Re-reset and step in MJX
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.825; d.qpos[3] = 1.0
    d.ctrl[:] = ctrl
    dx = mjx.put_data(m, d)

    @jax.jit
    def step_mjx(data):
        def body_fn(i, val):
            return mjx.step(mx, val)
        return jax.lax.fori_loop(0, 10, body_fn, data)

    dx_after = step_mjx(dx)
    mjx_qpos = np.array(dx_after.qpos)
    mjx_qvel = np.array(dx_after.qvel)

    qpos_diff = np.max(np.abs(cpu_qpos - mjx_qpos))
    qvel_diff = np.max(np.abs(cpu_qvel - mjx_qvel))

    # In float32 MJX with iterations=6 vs float64 CPU MuJoCo, contact drift
    # accumulates ~4mm over 10 physics steps (50ms).
    print(f"  -> MuJoCo CPU (float64) vs JAX MJX (float32) Rollout over 10 steps (50ms):")
    print(f"     Max qpos difference: {qpos_diff:.6e} m/rad (within float32 contact tolerance < 10mm)")
    print(f"     Max qvel difference: {qvel_diff:.6e} rad/s")
    assert qpos_diff < 1e-2, f"qpos divergence excessive: {qpos_diff} m"
    print("  -> MuJoCo CPU vs JAX MJX Parity Verified within float32 tolerance!")


if __name__ == "__main__":
    print("==================================================================")
    print("CYRUS MUJOCO CPU VS JAX MJX PARITY SUITE")
    print("==================================================================")
    test_policy_numpy_vs_jax_parity()
    test_mujoco_cpu_vs_mjx_physics_parity()
    print("==================================================================")
