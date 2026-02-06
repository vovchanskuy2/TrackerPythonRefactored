import cv2 #.\venv\Scripts\Activate
import mediapipe as mp
import time
import logging
import argparse
import threading
from detector import create_detectors, download_models, POSE_CONNECTIONS, HAND_CONNECTIONS, FACE_CONTOURS_SIMPLIFIED
from smoother import Smoother
from capture import get_capture
from udp_sender import UDPSender

logging.basicConfig(level=logging.INFO)

def draw_all(frame, pose_lms, hands_lms, face_lms):
    h, w = frame.shape[:2]

    # Pose
    if pose_lms:
        for pose in pose_lms:
            for conn in POSE_CONNECTIONS:
                start, end = conn
                x1, y1, z1, v1 = pose[start]
                x2, y2, z2, v2 = pose[end]
                if v1 < 0.3 or v2 < 0.3:
                    continue
                cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (100, 200, 255), 2)
            for i, (x, y, z, vis) in enumerate(pose):
                if vis < 0.3: continue
                color = (255, 150, 50) if i < 11 else (50, 150, 255) if i < 23 else (50, 255, 50)
                cv2.circle(frame, (int(x*w), int(y*h)), 4, color, -1)

    # Hands
    if hands_lms:
        for hand in hands_lms:
            if not hand: continue
            for conn in HAND_CONNECTIONS:
                start, end = conn
                x1, y1, z1, v1 = hand[start]
                x2, y2, z2, v2 = hand[end]
                if v1 < 0.3 or v2 < 0.3:
                    continue
                cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 100, 100), 2)
            for i, (x, y, z, vis) in enumerate(hand):
                if vis < 0.3: continue
                cv2.circle(frame, (int(x*w), int(y*h)), 3, (100, 255, 100), -1)

    # Face
    if face_lms:
        for face in face_lms:
            for conn in FACE_CONTOURS_SIMPLIFIED:
                start, end = conn
                x1, y1, z1, v1 = face[start]
                x2, y2, z2, v2 = face[end]
                if v1 < 0.3 or v2 < 0.3:
                    continue
                cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (200, 200, 200), 1)
            for i, (x, y, z, vis) in enumerate(face):
                if vis < 0.3: continue
                cv2.circle(frame, (int(x*w), int(y*h)), 1, (255, 255, 0), -1)

    return frame

def detection_thread(pose_det, hand_det, face_det, smoother, cap, stop_event):
    while not stop_event.is_set():
        current_time = time.time()
        if smoother.should_process(current_time):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(current_time * 1000)

            pose_result = pose_det.detect_for_video(mp_image, timestamp_ms)
            
            hand_result = None
            face_result = None

            # Проверяем pose для активации hands и face
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose_lm = pose_result.pose_landmarks[0]
                
                # Hands — если хотя бы одно запястье видно
                left_wrist = pose_lm[15]
                right_wrist = pose_lm[16]
                if left_wrist.visibility > 0.4 or right_wrist.visibility > 0.4:
                    hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

                # Face — если нос виден
                nose = pose_lm[0]
                if nose.visibility > 0.4:
                    face_result = face_det.detect_for_video(mp_image, timestamp_ms)

            # Передаём результаты в smoother
            smoother.add_frame(
                pose_result.pose_landmarks if pose_result else None,
                hand_result.hand_landmarks if hand_result else None,
                face_result.face_landmarks if face_result else None,
                current_time
            )

        time.sleep(0.01)  # небольшая задержка, чтобы не нагружать CPU

def main(args):
    if args.download_models:
        download_models()

    cap = get_capture(source=args.source)
    pose_det, hand_det, face_det = create_detectors()
    smoother = Smoother(buffer_size=10, target_fps=args.target_fps, process_fps=args.process_fps, mirror_x=args.mirror_x, invert_z=args.invert_z)
    udp_sender = UDPSender(ip="127.0.0.1", port=5005)  # localhost:5005 — можно изменить

    stop_event = threading.Event()
    det_thread = threading.Thread(target=detection_thread, args=(pose_det, hand_det, face_det, smoother, cap, stop_event))
    det_thread.start()

    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0

    try:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracking', 1280, 720)
        while True:
            current_time = time.time()
            ret, frame = cap.read()

            if not ret:
                logging.warning("Не удалось получить кадр")
                break
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            if smoother.should_interpolate(current_time):
                pose, hands, face = smoother.get_interpolated(current_time)
                if pose or hands or face:  # отправляем только если есть данные
                    udp_sender.send_pose_data(pose, hands, face, current_time)
                    
            fps_counter += 1
        #    if current_time - last_fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            last_fps_time = current_time  

            display_frame = draw_all(display_frame, pose, hands, face)
            cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

            time.sleep(0.01)

    finally:
        stop_event.set()
        det_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        hand_det.close()
        face_det.close()
        logging.info("Ресурсы освобождены.")
        udp_sender.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracker")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--process_fps", type=int, default=15)
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mirror_x", action="store_true", default=True)
    parser.add_argument("--invert_z", action="store_true", default=True)
    parser.add_argument("--download_models", action="store_true", help="Скачать hands/face модели")
    args = parser.parse_args()
    main(args)