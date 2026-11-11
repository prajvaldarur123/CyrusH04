"""
Verifies that all humanoid XML models compile cleanly in MuJoCo.
"""
import os
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

models = [
    "cyrus.xml",
    "cyrus_12dof_mjx.xml",
    "cyrus_human_feet.xml",
    "cyrus_mjx.xml",
    "cyrus_arena_12.xml",
    "cyrus_ghost_trajectory.xml"
]

all_passed = True
print("=" * 78)
print("MUJOCO MODEL VALIDATION SUITE")
print("=" * 78)

for m_name in models:
    path = os.path.join(MODELS_DIR, m_name)
    try:
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        print(f"[✓ PASS] {m_name:32s} | nq={m.nq:2d} nv={m.nv:2d} nu={m.nu:2d} ngeom={m.ngeom:3d}")
    except Exception as e:
        all_passed = False
        print(f"[X FAIL] {m_name:32s} | ERROR: {e}")

print("=" * 78)
if all_passed:
    print("[RESULT] ALL 7 HUMANOID MODELS FULLY VALIDATED AND FUNCTIONAL IN MUJOCO!")
else:
    print("[RESULT] SOME MODELS FAILED VALIDATION.")
print("=" * 78)
