import mujoco
import os

PROJECT_ROOT = "C:/Users/prajval/robotics/cyrus"
MODEL_FILES = [
    "models/cyrus.xml",
    "models/cyrus_12dof_mjx.xml",
    "models/cyrus_human_feet.xml",
]

EXPECTED_MASS = 61.66
TOLERANCE = 0.01  # kg

for filename in MODEL_FILES:
    print("\n" + "=" * 55)
    print(f"MODEL: {filename}")
    print("=" * 55)

    try:
        full_path = os.path.join(PROJECT_ROOT, filename)
        model = mujoco.MjModel.from_xml_path(full_path)

        # Body 0 is the world body, so exclude it.
        total_mass = model.body_mass[1:].sum()

        print(f"Total mass: {total_mass:.4f} kg")
        print(f"Expected mass: {EXPECTED_MASS:.4f} kg")
        print(f"Difference: {total_mass - EXPECTED_MASS:+.4f} kg")

        if abs(total_mass - EXPECTED_MASS) <= TOLERANCE:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

        print(f"\nNumber of bodies: {model.nbody - 1}")

        print("\nLargest body masses:")

        body_data = []

        for i in range(1, model.nbody):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, i
            )
            mass = model.body_mass[i]
            body_data.append((name or f"Body_{i}", mass))

        for name, mass in sorted(
            body_data, key=lambda item: item[1], reverse=True
        )[:15]:
            print(f"{name:30s} {mass:10.4f} kg")

        print("\nMassless bodies:")

        massless = [
            name for name, mass in body_data if mass <= 1e-8
        ]

        if massless:
            for name in massless:
                print(f"  {name}")
        else:
            print("  None")

    except Exception as error:
        print(f"MODEL ERROR: {error}")
