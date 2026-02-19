import numpy as np

VISIBILITY_THRESHOLD = 0.5


def lm_to_array(landmarks):
    """Переводит MediaPipe landmarks в numpy массив"""
    result = []
    for lm in landmarks:
        result.append([lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)])
    return np.array(result)


def filter_visibility(landmarks):
    """Фильтрует точки по visibility"""
    if landmarks is None:
        return None

    mask = landmarks[:, 3] > VISIBILITY_THRESHOLD
    return landmarks, mask


def normalize_pose(pose):
    """Нормализация позы (центр + масштаб)"""
    if pose is None:
        return None

    # центр — между плечами
    left_shoulder = pose[11][:3]
    right_shoulder = pose[12][:3]

    center = (left_shoulder + right_shoulder) / 2.0

    # масштаб — расстояние между плечами
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale < 1e-5:
        scale = 1.0

    normalized = (pose[:, :3] - center) / scale

    return normalized


def process_frame(pose_lms, hand_lms, face_lms):
    """Главная функция обработки"""

    data = {
        "pose": None,
        "hands": None,
        "face": None
    }

    # ---- POSE ----
    if pose_lms:
        pose = lm_to_array(pose_lms[0])
        pose = normalize_pose(pose)
        data["pose"] = pose

    # ---- HANDS ----
    if hand_lms:
        hands = []
        for hand in hand_lms:
            if hand:
                arr = lm_to_array(hand)
                hands.append(arr)
        data["hands"] = hands

    # ---- FACE ----
    if face_lms:
        face = lm_to_array(face_lms[0])
        data["face"] = face

    return data