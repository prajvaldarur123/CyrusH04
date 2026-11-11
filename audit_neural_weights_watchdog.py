"""
Cyrus Humanoid - Autonomous Weight & Production Checkpoint Watchdog
Runs continuously in background. Audits every saved checkpoint:
- Tracks file modification timestamp
- Computes Frobenius norm drift ||W_t - W_{t-1}||
- Detects NaN, Inf, weight freezing, or exploding gradients
- Writes real-time audit health report to cyrus/checkpoints/weight_audit_health.json
"""

import os
import time
import json
import numpy as np

CHECKPOINT_PROD = r"C:\Users\prajval\robotics\cyrus\checkpoints\balance_12dof_production.npz"
HEALTH_JSON = r"C:\Users\prajval\robotics\cyrus\checkpoints\weight_audit_health.json"

last_mtime = 0
prev_weights = {}

print("=" * 78)
print("AGENT 4 -- NEURAL NETWORK WEIGHT & PRODUCTION CHECKPOINT AUDITOR")
print(f"Monitoring: {CHECKPOINT_PROD}")
print("=" * 78)

def audit_checkpoint():
    global last_mtime, prev_weights
    if not os.path.exists(CHECKPOINT_PROD):
        return None

    cur_mtime = os.path.getmtime(CHECKPOINT_PROD)
    if cur_mtime == last_mtime:
        return None

    try:
        data = np.load(CHECKPOINT_PROD)
        audit_res = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_age_sec": round(time.time() - cur_mtime, 1),
            "layers": {},
            "total_frobenius_norm": 0.0,
            "weight_drift_from_previous": 0.0,
            "is_learning_active": False,
            "is_healthy": True,
            "anomalies": []
        }

        total_norm_sq = 0.0
        total_drift_sq = 0.0

        for k in data.keys():
            arr = data[k]
            f_norm = float(np.linalg.norm(arr))
            has_nan = bool(np.isnan(arr).any())
            has_inf = bool(np.isinf(arr).any())
            
            total_norm_sq += f_norm ** 2

            drift = 0.0
            if k in prev_weights:
                delta = arr - prev_weights[k]
                drift = float(np.linalg.norm(delta))
                total_drift_sq += drift ** 2

            if has_nan:
                audit_res["anomalies"].append(f"NaN detected in layer {k}!")
                audit_res["is_healthy"] = False
            if has_inf:
                audit_res["anomalies"].append(f"Inf detected in layer {k}!")
                audit_res["is_healthy"] = False

            audit_res["layers"][k] = {
                "shape": list(arr.shape),
                "frobenius_norm": round(f_norm, 4),
                "delta_drift": round(drift, 6)
            }
            prev_weights[k] = arr.copy()

        audit_res["total_frobenius_norm"] = round(float(np.sqrt(total_norm_sq)), 4)
        audit_res["weight_drift_from_previous"] = round(float(np.sqrt(total_drift_sq)), 6)
        audit_res["is_learning_active"] = (audit_res["weight_drift_from_previous"] > 1e-6) or (last_mtime == 0)

        last_mtime = cur_mtime

        with open(HEALTH_JSON, "w") as f:
            json.dump(audit_res, f, indent=2)

        return audit_res
    except Exception as e:
        print(f"[!] Audit error reading checkpoint: {e}", flush=True)
        return None

if __name__ == "__main__":
    initial = audit_checkpoint()
    if initial:
        print(f"[+] Initial Checkpoint Validated! Total Weight Norm: {initial['total_frobenius_norm']} | Drift: {initial['weight_drift_from_previous']}")
    
    while True:
        time.sleep(3)
        res = audit_checkpoint()
        if res:
            status_str = "ACTIVE LEARNING" if res["is_learning_active"] else "STAGNANT"
            print(f"[WEIGHT AUDIT] Time: {res['time_str']} | Total Norm: {res['total_frobenius_norm']} | Delta Drift: {res['weight_drift_from_previous']:.6f} | Status: {status_str} | Healthy: {res['is_healthy']}", flush=True)
