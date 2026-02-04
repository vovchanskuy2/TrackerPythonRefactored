import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class SmoothLandmark:
    """Сглаживание одной точки с Kalman (x,y,z) и EMA (visibility)."""
    def __init__(self):
        self.kf_states = [self._init_kalman() for _ in range(3)]
        self.visibility = 0.0

    def _init_kalman(self):
        state = np.zeros((2, 1), dtype=np.float64)  # всегда (2,1)
        P = np.eye(2, dtype=np.float64) * 1000.0
        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = 1.0
        Q = np.eye(2, dtype=np.float64) * 0.001
        return {'state': state, 'P': P, 'F': F, 'H': H, 'R': R, 'Q': Q}

    def _kalman_update(self, kf, measurement: float):
        try:
            # Гарантируем, что state имеет форму (2,1)
            state = kf['state']
            if state.shape != (2, 1):
                logging.warning(f"Неправильная форма state: {state.shape} → исправляем")
                state = np.reshape(state, (2, 1))

            # Predict
            kf['state'] = kf['F'] @ state
            kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']

            # Update
            meas = np.array([[measurement]], dtype=np.float64)  # (1,1)
            y = meas - kf['H'] @ kf['state']
            S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
            K = (kf['P'] @ kf['H'].T) / S
            kf['state'] = kf['state'] + K * y
            kf['P'] = (np.eye(2) - K @ kf['H']) @ kf['P']

            return kf['state'][0, 0]
        except Exception as e:
            logging.error(f"Ошибка в Kalman update: {e}, measurement={measurement}")
            return measurement  # fallback — берём сырое значение

    def update(self, x: float, y: float, z: float, visibility: float = 1.0, alpha_vis: float = 0.7):
        self.kf_states[0]['state'] = self._kalman_update(self.kf_states[0], x)
        self.kf_states[1]['state'] = self._kalman_update(self.kf_states[1], y)
        self.kf_states[2]['state'] = self._kalman_update(self.kf_states[2], z)

        vis = float(visibility) if visibility is not None else 1.0
        self.visibility = self.visibility * alpha_vis + vis * (1 - alpha_vis)

    def get(self) -> Tuple[float, float, float, float]:
        return (
            float(self.kf_states[0]['state'][0, 0]),
            float(self.kf_states[1]['state'][0, 0]),
            float(self.kf_states[2]['state'][0, 0]),
            float(self.visibility)
        )

    def reset(self):
        for kf in self.kf_states:
            kf['state'] = np.zeros((2, 1), dtype=np.float64)
            kf['P'] = np.eye(2, dtype=np.float64) * 1000.0
        self.visibility = 0.0


@dataclass
class FrameData:
    pose_landmarks: List[List[Tuple[float, float, float, float]]]
    hand_landmarks: List[List[Tuple[float, float, float, float]]]
    face_landmarks: List[List[Tuple[float, float, float, float]]]
    timestamp: float


class Smoother:
    def __init__(self, buffer_size=10, target_fps=30, process_fps=15,
                 num_poses=1, num_hands=2, num_faces=1,
                 mirror_x=True, invert_z=True):
        self.buffer = deque(maxlen=buffer_size)
        self.smoothed_pose = [[SmoothLandmark() for _ in range(33)] for _ in range(num_poses)]
        self.smoothed_hands = [[[SmoothLandmark() for _ in range(21)] for _ in range(num_hands)] for _ in range(num_poses)]
        self.smoothed_face = [[SmoothLandmark() for _ in range(478)] for _ in range(num_faces)]
        self.last_processed_time = 0.0
        self.last_interpolated_time = 0.0
        self.target_fps = target_fps
        self.process_fps = process_fps
        self.process_interval = 1.0 / process_fps
        self.interpolation_interval = 1.0 / target_fps
        self.num_poses = num_poses
        self.num_hands = num_hands
        self.num_faces = num_faces
        self.mirror_x = mirror_x
        self.invert_z = invert_z

    def add_frame(self, pose_lms, hand_lms, face_lms, timestamp: float):
        frame_pose = []
        frame_hands = []
        frame_face = []

        # Pose
        if pose_lms:
            for pose_landmarks in pose_lms[:self.num_poses]:
                lm_list = []
                for lm in pose_landmarks:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    if self.mirror_x: x = 1.0 - x
                    if self.invert_z: z = -z
                    lm_list.append((x, y, z, vis))
                if len(lm_list) == 33:
                    frame_pose.append(lm_list)
                    for i in range(33):
                        self.smoothed_pose[0][i].update(*lm_list[i][:3], lm_list[i][3])

        # Hands
        if hand_lms and frame_pose:
            for hand_idx, hand_landmarks in enumerate(hand_lms[:self.num_hands]):
                lm_list = []
                for lm in hand_landmarks:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    if self.mirror_x: x = 1.0 - x
                    if self.invert_z: z = -z
                    lm_list.append((x, y, z, vis))
                if len(lm_list) == 21:
                    frame_hands.append(lm_list)
                    for i in range(21):
                        self.smoothed_hands[0][hand_idx][i].update(*lm_list[i][:3], lm_list[i][3])

        # Face
        if face_lms and frame_pose:
            for face_landmarks in face_lms[:self.num_faces]:
                lm_list = []
                for lm in face_landmarks:
                    x = lm.x
                    y = lm.y
                    z = lm.z
                    vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    if self.mirror_x: x = 1.0 - x
                    if self.invert_z: z = -z
                    lm_list.append((x, y, z, vis))
                if len(lm_list) >= 468:
                    lm_list = lm_list[:478]
                    frame_face.append(lm_list)
                    for i in range(min(478, len(lm_list))):
                        self.smoothed_face[0][i].update(*lm_list[i][:3], lm_list[i][3])

        if frame_pose or frame_hands or frame_face:
            self.buffer.append(FrameData(frame_pose, frame_hands, frame_face, timestamp))

    def get_interpolated(self, current_time: float):
        if len(self.buffer) == 0:
            return None, None, None

        if len(self.buffer) == 1:
            d = self.buffer[0]
            return d.pose_landmarks, d.hand_landmarks, d.face_landmarks

        frame1, frame2 = self.buffer[-2], self.buffer[-1]

        if current_time <= frame1.timestamp:
            return frame1.pose_landmarks, frame1.hand_landmarks, frame1.face_landmarks
        if current_time >= frame2.timestamp:
            return frame2.pose_landmarks, frame2.hand_landmarks, frame2.face_landmarks

        t = (current_time - frame1.timestamp) / (frame2.timestamp - frame1.timestamp)

        pose = self._interp(frame1.pose_landmarks[0] if frame1.pose_landmarks else None,
                            frame2.pose_landmarks[0] if frame2.pose_landmarks else None,
                            self.smoothed_pose[0], 33, t)

        hands = []
        for hi in range(self.num_hands):
            h1 = frame1.hand_landmarks[hi] if frame1.hand_landmarks and len(frame1.hand_landmarks) > hi else None
            h2 = frame2.hand_landmarks[hi] if frame2.hand_landmarks and len(frame2.hand_landmarks) > hi else None
            hands.append(self._interp(h1, h2, self.smoothed_hands[0][hi], 21, t))

        face = self._interp(frame1.face_landmarks[0] if frame1.face_landmarks else None,
                            frame2.face_landmarks[0] if frame2.face_landmarks else None,
                            self.smoothed_face[0], 478, t)

        return [pose] if pose else None, [h for h in hands if h], [face] if face else None

    def _interp(self, lm1, lm2, smoothed, n_points, t):
        if lm1 is None or lm2 is None or len(lm1) != n_points or len(lm2) != n_points:
            return None
        result = []
        for i in range(n_points):
            x1, y1, z1, v1 = lm1[i]
            x2, y2, z2, v2 = lm2[i]
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            z = z1 + (z2 - z1) * t
            v = v1 + (v2 - v1) * t
            sx, sy, sz, sv = smoothed[i].get()
            result.append((sx * 0.4 + x * 0.6, sy * 0.4 + y * 0.6, sz * 0.4 + z * 0.6, sv * 0.4 + v * 0.6))
        return result

    def should_process(self, t: float) -> bool:
        if t - self.last_processed_time >= self.process_interval:
            self.last_processed_time = t
            return True
        return False

    def should_interpolate(self, t: float) -> bool:
        if t - self.last_interpolated_time >= self.interpolation_interval:
            self.last_interpolated_time = t
            return True
        return False

    def reset(self):
        self.buffer.clear()
        for group in [self.smoothed_pose[0], self.smoothed_face[0]]:
            for lm in group:
                lm.reset()
        for hand_group in self.smoothed_hands[0]:
            for lm in hand_group:
                lm.reset()