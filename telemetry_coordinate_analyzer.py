import os
import sys
import json
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from cyrus.envs.standup_env import CyrusStandupEnv
from cyrus.envs.walk_balance_env import CyrusWalkBalanceEnv

def analyze_model_rollout(env_type="standup", n_frames=200):
    if env_type == "standup":
        env = CyrusStandupEnv(render_mode=None)
        ckpt = "cyrus/checkpoints/best_model.zip"
        is_walk = False
    else:
        env = CyrusWalkBalanceEnv(render_mode=None, curriculum_phase=1)
        ckpt = "cyrus/checkpoints/walk/best_model.zip"
        is_walk = True

    if not os.path.exists(ckpt):
        ckpt = "cyrus/checkpoints/best_model.zip"

    model = PPO.load(ckpt, device="cpu")
    obs, info = env.reset()

    frames = []
    m = env.model
    d = env.data

    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_upper")
    head_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
    r_palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    l_palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    r_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_sole")
    l_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

    r_shin_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_shin")
    l_shin_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_shin")

    omega_0 = float(np.sqrt(9.81 / 0.84))
    jnt_ids = [m.actuator_trnid[i, 0] for i in range(27)]
    qpos_adr = [m.jnt_qposadr[jid] for jid in jnt_ids]

    for f in range(n_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        pelvis_xyz = [round(float(v), 4) for v in d.xpos[pelvis_id]]
        torso_xyz = [round(float(v), 4) for v in d.xpos[torso_id]]
        head_xyz = [round(float(v), 4) for v in d.xpos[head_id]]
        r_palm_xyz = [round(float(v), 4) for v in d.site_xpos[r_palm_id]]
        l_palm_xyz = [round(float(v), 4) for v in d.site_xpos[l_palm_id]]
        r_sole_xyz = [round(float(v), 4) for v in d.site_xpos[r_sole_id]]
        l_sole_xyz = [round(float(v), 4) for v in d.site_xpos[l_sole_id]]
        r_knee_xyz = [round(float(v), 4) for v in d.xpos[r_shin_id]]
        l_knee_xyz = [round(float(v), 4) for v in d.xpos[l_shin_id]]

        com_xyz = [round(float(v), 4) for v in d.subtree_com[pelvis_id]]
        com_vel = d.qvel[:3]
        cp_x = round(float(com_xyz[0] + com_vel[0] / omega_0), 4)
        cp_y = round(float(com_xyz[1] + com_vel[1] / omega_0), 4)

        torso_mat = d.xmat[torso_id].reshape(3, 3)
        uprightness = round(float(torso_mat[2, 2]), 4)
        pitch = round(float(torso_mat[0, 2]), 4)

        joint_angles = [round(float(v), 3) for v in d.qpos[qpos_adr]]

        frame_data = {
            "frame": f,
            "time_sec": round(f * 0.02, 2),
            "pelvis": pelvis_xyz,
            "torso": torso_xyz,
            "head": head_xyz,
            "r_palm": r_palm_xyz,
            "l_palm": l_palm_xyz,
            "r_sole": r_sole_xyz,
            "l_sole": l_sole_xyz,
            "r_knee": r_knee_xyz,
            "l_knee": l_knee_xyz,
            "com": com_xyz,
            "capture_point": [cp_x, cp_y],
            "uprightness": uprightness,
            "pitch": pitch,
            "reward": round(float(reward), 2),
            "standing_seconds": info.get("standing_seconds", info.get("stand_hold_sec", 0.0)),
            "stage": info.get("stage", info.get("phase_name", "ACTIVE")),
            "faults": info.get("faults", [])
        }
        frames.append(frame_data)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    return frames

if __name__ == "__main__":
    print("[*] Running Frame-by-Frame Coordinate Inspection for Wing 1 (Standup / Recovery)...")
    standup_frames = analyze_model_rollout("standup", n_frames=150)
    print(f"    [+] Collected {len(standup_frames)} physics coordinate frames for Wing 1.")

    print("[*] Running Frame-by-Frame Coordinate Inspection for Wing 2 (Balance & Walk Curriculum)...")
    walk_frames = analyze_model_rollout("walk", n_frames=150)
    print(f"    [+] Collected {len(walk_frames)} physics coordinate frames for Wing 2.")

    telemetry = {
        "wing1_standup": standup_frames,
        "wing2_walk": walk_frames
    }

    out_json = "cyrus/telemetry_frames.json"
    with open(out_json, "w") as fp:
        json.dump(telemetry, fp, indent=2)

    print(f"[+] Successfully wrote complete frame-by-frame coordinates to {out_json}.")
