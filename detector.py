import mediapipe as mp
import logging
import os
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO)

POSE_MODEL_PATH = 'pose_landmarker_full.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
FACE_MODEL_PATH = 'face_landmarker.task'

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28),
    (28, 30), (30, 32), (27, 31), (28, 32)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Упрощённые контуры лица (овал + глаза + рот + брови) — ~30 соединений вместо 100+
# Можно расширить, скопировав из старой документации MediaPipe FACEMESH_CONTOURS
FACE_CONTOURS_SIMPLIFIED = [
    # Овал лица (примерно)
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    # Правый глаз (контур)
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 145),
    (145, 153), (33, 7), (7, 163), (163, 144),
    # Левый глаз
    (263, 387), (387, 385), (385, 362), (362, 380), (380, 373), (373, 380),
    (263, 249), (249, 390), (390, 373),
    # Рот внешний
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409),
    (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), (37, 39),
    (39, 40), (40, 185), (185, 61),
    # Рот внутренний (упрощённо)
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (308, 415),
    (415, 310), (310, 311), (311, 312), (312, 13), (13, 82), (82, 81),
    (81, 42), (42, 183), (183, 78),
]

# Уникальные индексы точек из упрощённых контуров (овал, глаза, губы) — ~93 шт.
SELECTED_FACE_INDICES = sorted(set(idx for pair in FACE_CONTOURS_SIMPLIFIED for idx in pair))

def download_models():
    """Скачивает модели hands и face, если их нет (pose уже должна быть)."""
    models = {
        HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        FACE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    }
    for path, url in models.items():
        if not os.path.exists(path):
            logging.info(f"Скачивание {path}...")
            urlretrieve(url, path)
            logging.info(f"Скачано: {path}")
        else:
            logging.info(f"Модель {path} уже существует.")

def create_detectors(num_poses=1, num_hands=2, num_faces=1, min_conf=0.5):
    """Создаёт детекторы в новом API (Tasks)."""
    try:
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Pose
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        pose_detector = PoseLandmarker.create_from_options(pose_options)

        # Hands
        HandLandmarker = mp.tasks.vision.HandLandmarker
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        hand_detector = HandLandmarker.create_from_options(hand_options)

        # Face
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_face_blendshapes=True,   # Для выражений лица (открытие рта/глаз)
            output_facial_transformation_matrixes=True  # Опционально, для 3D
        )
        face_detector = FaceLandmarker.create_from_options(face_options)

        logging.info("Детекторы (Tasks API) созданы успешно.")
        return pose_detector, hand_detector, face_detector
    except Exception as e:
        logging.error(f"Ошибка создания детекторов: {e}")
        raise