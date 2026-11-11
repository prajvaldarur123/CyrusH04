"""
Cyrus Humanoid - Strict Mastery-Gated Curriculum Manager
Enforces sequential stage progression: the robot trains on a single stage repeatedly
until it achieves it with complete confidence (consecutive mastery streak)
before unlocking and displaying the next stage ghost target.
"""

import json
import os
import time

STATE_FILE = r"C:\Users\prajval\robotics\cyrus\curriculum_state.json"

STAGE_NAMES = {
    0: "Stage 0: Prone Rest on Floor (Palms planted, chin up)",
    1: "Stage 1: Grounded Triceps Push-Up (Chest elevation, knees on floor)",
    2: "Stage 1.5: High Plank (Straight arms, knees lifted, toes locked at back)",
    3: "Stage 2: Athletic Bear Hover & Foot Step (Human-Imitable Quadruped)",
    4: "Stage 3: Squat Tuck & Floor Release (Weight on dual feet, hands in guard)",
    5: "Stage 4: Quad Drive to Standing (Driving through heels)",
    6: "Stage 5: Full Upright Balance Lock (100% Upright Hold)"
}

STAGE_MAX_FRAMES = {
    0: 0,    # Stage 0: Prone Rest (Frame 0)
    1: 78,   # Stage 1: Grounded Triceps Push-Up (Frame 78)
    2: 117,  # Stage 1.5: High Plank (Frame 117)
    3: 160,  # Stage 2: Athletic Bear Hover & Foot Step (Frame 160)
    4: 278,  # Stage 3: Squat Tuck & Floor Release (Frame 278: Full Upright Squat Guard)
    5: 360,  # Stage 4: Quad Drive to Standing (Frame 360)
    6: 400   # Stage 5: Full Upright Balance Lock (Frame 400)
}

class CurriculumManager:
    def __init__(self, streak_required=3):
        self.streak_required = streak_required
        self.unlocked_stage = 2 # Set to Stage 1.5: High Plank
        self.current_streak = 0
        self.total_trials = 0
        self.stage_stats = {
            s: {"trials": 0, "successes": 0, "mastered": False} for s in range(7)
        }
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.unlocked_stage = data.get("unlocked_stage", 2)
                    self.current_streak = data.get("current_streak", 0)
                    self.total_trials = data.get("total_trials", 0)
                    self.streak_required = data.get("streak_required", self.streak_required)
                    stats = data.get("stage_stats", {})
                    for s in range(7):
                        s_str = str(s)
                        if s_str in stats:
                            self.stage_stats[s] = stats[s_str]
                return
            except Exception:
                pass
        self.save_state()

    def save_state(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        data = {
            "unlocked_stage": self.unlocked_stage,
            "current_streak": self.current_streak,
            "streak_required": self.streak_required,
            "total_trials": self.total_trials,
            "stage_name": STAGE_NAMES[self.unlocked_stage],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage_stats": {str(k): v for k, v in self.stage_stats.items()}
        }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def evaluate_stage_criteria(self, stage, metrics_history):
        """
        Evaluates whether the trial satisfied the strict physical criteria for the target stage.
        metrics_history is a list of dicts recorded during the hold phase.
        """
        if not metrics_history:
            return False, "No telemetry recorded during hold phase"

        # Check conditions over 5.0 SECONDS OF SUSTAINED BALANCE HOLD (250 steps at 50Hz)
        min_hold_steps = 250  # Exactly 5.0s of continuous steady hold
        if len(metrics_history) < min_hold_steps:
            return False, f"Hold phase too short ({len(metrics_history)*0.02:.2f}s < 5.0s)"
        recent = metrics_history[-min_hold_steps:]
        
        if stage == 0:
            # Stage 0: Prone Rest on Floor
            for m in recent:
                if m["pelvis_z"] > 0.24:
                    return False, f"Pelvis too high ({m['pelvis_z']:.3f}m > 0.24m)"
                if max(m["r_palm_z"], m["l_palm_z"]) > 0.095:
                    return False, f"Palms not grounded (Palm Z={max(m['r_palm_z'], m['l_palm_z']):.3f}m > 0.095m)"
            return True, "Solid Prone Ground Lock & Palm Base"

        elif stage == 1:
            # Stage 1: Grounded Triceps Push-Up
            for m in recent:
                if m["torso_z"] < 0.25:
                    return False, f"Torso elevation insufficient ({m['torso_z']:.3f}m < 0.25m)"
                if max(m["r_palm_z"], m["l_palm_z"]) > 0.095:
                    return False, f"Palms slipped off floor ({max(m['r_palm_z'], m['l_palm_z']):.3f}m > 0.095m)"
                if m["torso_up"] < 0.30:
                    return False, f"Torso not angled up ({m['torso_up']:.2f} < 0.30)"
            return True, "Sustained Triceps Push-Up & Chest Elevation"

        elif stage == 2:
            # Stage 1.5: High Plank (Straight Arms, Knees Lifted Off Mat, Toes Locked at Back)
            for m in recent:
                if m['torso_z'] < 0.28:
                    return False, f"Plank torso height collapsed ({m['torso_z']:.3f}m < 0.28m)"
                if min(m['r_knee_z'], m['l_knee_z']) < 0.12:
                    return False, f"Plank knees not lifted off mat ({min(m['r_knee_z'], m['l_knee_z']):.3f}m < 0.12m)"
                if max(m['r_palm_z'], m['l_palm_z']) > 0.095:
                    return False, f"Plank palms lost floor contact ({max(m['r_palm_z'], m['l_palm_z']):.3f}m > 0.095m)"
                if max(m['r_sole_z'], m['l_sole_z']) > 0.16:
                    return False, "Plank feet lost ground support at the back"
                span = m.get('span_x', None)
                if span is not None and span < 1.00:
                    return False, f"Plank feet stepped forward prematurely (span = {span:.3f}m < 1.00m)"
            return True, "Rock-Solid High Plank: Straight Arms, Knees Airborne, Toes Planted"

        elif stage == 3:
            # Stage 2: Athletic Bear Hover & Foot Step (Human-Imitable Quadruped)
            # 5-SECOND CONTINUOUS HOLD & 3D GEOMETRY GATES: Exactly matches user's verified blueprint!
            for m in recent:
                if min(m['r_knee_z'], m['l_knee_z']) < 0.060:
                    return False, f"Knees touched floor ({min(m['r_knee_z'], m['l_knee_z']):.3f}m < 0.060m)"
                if m.get('head_z', 0.5) < 0.35:
                    return False, f"Head dropping/faceplant detected ({m.get('head_z', 0):.3f}m < 0.35m)"
                if m['pelvis_z'] < 0.350:
                    return False, f"Pelvis hover collapsed ({m['pelvis_z']:.3f}m < 0.350m)"
                if max(m['r_palm_z'], m['l_palm_z']) > 0.095:
                    return False, f"Palms lost ground support during bear hover ({max(m['r_palm_z'], m['l_palm_z']):.3f}m > 0.095m)"
                if max(m['r_sole_z'], m['l_sole_z']) > 0.16:
                    return False, f"Foot soles lost ground traction during bear hover ({max(m['r_sole_z'], m['l_sole_z']):.3f}m > 0.16m)"
                span = m.get('span_x', None)
                if span is not None and span > 0.88:
                    return False, f"Feet still stuck in plank base (span = {span:.3f}m > 0.88m). Must step feet forward into Bear Hover!"
                if span is not None and span < 0.35:
                    return False, f"Feet stepped too far under hands (span = {span:.3f}m < 0.35m)"
                if 'avg_knee_x' in m and 'avg_sole_x' in m:
                    knee_fwd = m['avg_knee_x'] - m['avg_sole_x']
                    if knee_fwd < 0.10:
                        return False, f"Knees not flexed forward over feet (fwd = {knee_fwd:.3f}m < 0.10m)"
            return True, f"Rock-Solid Athletic Bear Hover: Dual Palms & Soles Grounded, Feet Stepped Forward (span={recent[-1].get('span_x', 0):.2f}m), Knees Hovering Airborne for 5.0 Seconds"

        elif stage == 4:
            # Stage 3: Squat Tuck & Floor Release (Weight on dual feet, hands in defensive guard)
            for m in recent:
                if min(m["r_palm_z"], m["l_palm_z"]) < 0.15:
                    return False, f"Hands still touching floor ({min(m['r_palm_z'], m['l_palm_z']):.3f}m < 0.15m)"
                if max(m["r_sole_z"], m["l_sole_z"]) > 0.135:
                    return False, f"Feet not grounded ({max(m['r_sole_z'], m['l_sole_z']):.3f}m > 0.135m)"
                if min(m["r_knee_z"], m["l_knee_z"]) < 0.10:
                    return False, f"Knees collapsed onto mat ({min(m['r_knee_z'], m['l_knee_z']):.3f}m < 0.10m)"
                if m["pelvis_z"] < 0.26:
                    return False, f"Squat height collapsed ({m['pelvis_z']:.3f}m < 0.26m)"
            return True, "Weight Transferred to Feet & Hands Released into Guard"

        elif stage == 5:
            # Stage 4: Quad Drive to Standing
            for m in recent:
                if m["pelvis_z"] < 0.60:
                    return False, f"Pelvis drive incomplete ({m['pelvis_z']:.3f}m < 0.60m)"
                if m["torso_up"] < 0.70:
                    return False, f"Torso lean unstable ({m['torso_up']:.2f} < 0.70)"
                if max(m["r_sole_z"], m["l_sole_z"]) > 0.065:
                    return False, "Foot slipped during quad drive"
            return True, "Powerful Upward Quad Drive"

        elif stage == 6:
            # Stage 5: Full Upright Balance Lock
            for m in recent:
                if m["pelvis_z"] < 0.80:
                    return False, f"Standing height not reached ({m['pelvis_z']:.3f}m < 0.80m)"
                if m["torso_up"] < 0.90:
                    return False, f"Torso not vertical ({m['torso_up']:.2f} < 0.90)"
                if max(m["r_sole_z"], m["l_sole_z"]) > 0.060:
                    return False, "Foot slipped during balance hold"
            return True, "Full Upright Standing Balance Mastered"

        return False, "Unknown stage"

    def record_trial(self, metrics_history, max_unbroken_hold_steps=0):
        self.total_trials += 1
        st = self.unlocked_stage
        self.stage_stats[st]["trials"] += 1

        if max_unbroken_hold_steps < 250: # Must achieve 250 steps = 5.00s unbroken hold
            success = False
            reason = f"Unbroken hold duration insufficient ({max_unbroken_hold_steps*0.02:.2f}s < 5.00s required)"
        else:
            success, reason = self.evaluate_stage_criteria(st, metrics_history)

        if success:
            self.stage_stats[st]["successes"] += 1
            self.current_streak += 1
            print(f"\n  [+] TRIAL SUCCESS: {reason} | Current Streak: {self.current_streak}/{self.streak_required}", flush=True)

            if self.current_streak >= self.streak_required:
                self.stage_stats[st]["mastered"] = True
                print(f"\n{'='*75}")
                print(f"  [*** STAGE MASTERED WITH FULL CONFIDENCE ***] {STAGE_NAMES[st]}")
                if self.unlocked_stage < 6:
                    self.unlocked_stage += 1
                    self.current_streak = 0
                    print(f"  [>>> UNLOCKING NEXT CURRICULUM GOAL] -> {STAGE_NAMES[self.unlocked_stage]}")
                else:
                    print(f"  [*** 100% COMPLETE GET-UP PROGRAM MASTERED ACROSS ALL 7 STAGES! ***]")
                print(f"{'='*75}\n", flush=True)
        else:
            self.current_streak = 0
            print(f"\n  [-] TRIAL FAILED: {reason} | Streak reset to 0/{self.streak_required} (Must achieve {self.streak_required} in a row)", flush=True)

        self.save_state()
        return success, reason

    def get_stage_max_frame(self):
        """Returns the maximum frame in the reference trajectory for the current unlocked stage."""
        return STAGE_MAX_FRAMES[self.unlocked_stage]
