import numpy as np


class CyrusGestures:
    """
    Expressive behavioral primitives for Cyrus:
    - Head gaze tracking (smooth visual attention to 3D points)
    - Friendly waving
    - Agreement nodding
    - Organic idle breathing micro-motion
    """

    def __init__(self):
        # Active gesture timers and states
        self.wave_active = False
        self.wave_start_time = 0.0
        self.wave_duration = 3.5

        self.nod_active = False
        self.nod_start_time = 0.0
        self.nod_duration = 2.0

        self.gaze_tracking_enabled = True

    def trigger_wave(self, current_time: float):
        """Starts a friendly waving gesture with the right arm."""
        self.wave_active = True
        self.wave_start_time = current_time

    def trigger_nod(self, current_time: float):
        """Starts a head nodding gesture."""
        self.nod_active = True
        self.nod_start_time = current_time

    def compute_gaze(self, eye_pos: np.ndarray, target_pos: np.ndarray) -> tuple[float, float]:
        """
        Calculates neck yaw and pitch angles to direct head gaze toward target_pos.
        """
        delta = target_pos - eye_pos
        # Yaw: angle in horizontal plane
        target_yaw = np.arctan2(delta[1], delta[0])
        # Pitch: angle in vertical plane
        horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
        target_pitch = -np.arctan2(delta[2], horizontal_dist)

        # Clip to comfortable human-like neck limits
        neck_yaw = float(np.clip(target_yaw, -1.2, 1.2))
        neck_pitch = float(np.clip(target_pitch, -0.5, 0.5))

        return neck_yaw, neck_pitch

    def get_offsets(self, current_time: float, eye_pos: np.ndarray = None, target_pos: np.ndarray = None) -> dict:
        """
        Computes joint angle offsets for all active expressive behaviors.
        """
        offsets = {}

        # 1. Subtle Organic Idle Breathing (0.2 Hz breathing frequency)
        breath = np.sin(current_time * 2.0 * np.pi * 0.25)
        offsets["waist_pitch_act"] = 0.02 * breath
        offsets["right_shoulder_pitch_act"] = 0.03 * breath
        offsets["left_shoulder_pitch_act"] = 0.03 * breath

        # 2. Head Gaze Tracking
        if self.gaze_tracking_enabled and eye_pos is not None and target_pos is not None:
            neck_yaw, neck_pitch = self.compute_gaze(eye_pos, target_pos)
            offsets["neck_yaw_act"] = neck_yaw
            offsets["neck_pitch_act"] = neck_pitch

        # 3. Friendly Arm Wave Gesture
        if self.wave_active:
            elapsed = current_time - self.wave_start_time
            if elapsed < self.wave_duration:
                # Raise right arm up into wave posture
                blend_in = min(1.0, elapsed / 0.5)
                blend_out = max(0.0, (self.wave_duration - elapsed) / 0.5)
                blend = min(blend_in, blend_out)

                # Raise shoulder and bend elbow up
                offsets["right_shoulder_pitch_act"] = -1.2 * blend
                offsets["right_shoulder_roll_act"] = -1.1 * blend
                offsets["right_elbow_act"] = -1.6 * blend

                # Oscillate forearm/wrist for wave motion (~2.5 Hz)
                wave_motion = 0.35 * np.sin(elapsed * 2.0 * np.pi * 2.5) * blend
                offsets["right_wrist_act"] = wave_motion
                offsets["right_shoulder_yaw_act"] = 0.2 * wave_motion
            else:
                self.wave_active = False

        # 4. Agreement Nodding Gesture
        if self.nod_active:
            elapsed = current_time - self.nod_start_time
            if elapsed < self.nod_duration:
                blend = np.sin(elapsed / self.nod_duration * np.pi)
                # Nod head up and down twice
                nod_motion = 0.25 * np.sin(elapsed * 2.0 * np.pi * 1.5) * blend
                offsets["neck_pitch_act"] = offsets.get("neck_pitch_act", 0.0) + nod_motion
            else:
                self.nod_active = False

        return offsets
