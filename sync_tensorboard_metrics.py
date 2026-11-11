"""
Cyrus Humanoid - Real-time TensorBoard Syncer
Streams live GPU swarm training telemetry into TensorBoard event files.
Automatically tracks all active task logs.
"""

import os
import re
import time
import glob
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = r"C:\Users\prajval\robotics\cyrus\logs\tensorboard_gpu_swarm"
BRAIN_BASE = r"C:\Users\prajval\.gemini\antigravity\brain"
SWARM_LOG = r"C:\Users\prajval\robotics\cyrus\logs\swarm_train.log"

os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

pattern = re.compile(
    r'\|\s*ITER\s+(\d+)\s*\|\s*TIME\s+([\d:]+)\s*\|\s*DUR\s+([\d\.]+)s/10s.*?\|\s*RETURN:\s*([+-]?[\d\.]+)\s*\|\s*PUSH:\s*([\d\.]+)\s*m/s\s*\|\s*FALLS:\s*([\d\.]+)%'
)

processed_iters = set()

def get_task_logs():
    logs = []
    if os.path.exists(SWARM_LOG):
        logs.append(SWARM_LOG)
    pattern_glob = os.path.join(BRAIN_BASE, "*", ".system_generated", "tasks", "task-*.log")
    logs.extend(glob.glob(pattern_glob))
    return sorted(logs, key=os.path.getmtime, reverse=True)[:15]

def sync_once():
    log_files = get_task_logs()
    new_count = 0
    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if "ITER" not in content:
                continue
            matches = pattern.findall(content)
            for it_str, time_str, dur_str, ret_str, push_str, falls_str in matches:
                it = int(it_str)
                if it in processed_iters:
                    continue
                dur = float(dur_str)
                ret = float(ret_str)
                push = float(push_str)
                falls = float(falls_str)
                survival = 100.0 - falls

                # Write to TensorBoard
                writer.add_scalar("rollout/ep_rew_mean", ret, it)
                writer.add_scalar("rollout/ep_len_sec", dur, it)
                writer.add_scalar("curriculum/push_magnitude_mps", push, it)
                writer.add_scalar("curriculum/fall_rate_pct", falls, it)
                writer.add_scalar("curriculum/survival_rate_pct", survival, it)

                processed_iters.add(it)
                new_count += 1
        except Exception:
            pass

    if new_count > 0:
        writer.flush()
    return new_count

if __name__ == "__main__":
    print(f"[TB Sync] Initializing TensorBoard sync in {LOG_DIR}...")
    initial_count = sync_once()
    print(f"[TB Sync] Processed initial {initial_count} iterations.")
    
    while True:
        time.sleep(3)
        c = sync_once()
        if c > 0:
            print(f"[TB Sync] Synced {c} new iterations (Total: {len(processed_iters)})", flush=True)
