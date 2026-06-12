# main.py
import cv2
import time
import mediapipe as mp

from detector import create_detectors, download_models
from capture import get_capture
from processor import process_frame
from udp_sender import UDPSender


def main():
    download_models()
    cap = get_capture(0)
    pose_det, hand_det, face_det = create_detectors()

    sender = UDPSender()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t = time.time()
        timestamp = int(t * 1000)

        pose_res = pose_det.detect_for_video(mp_image, timestamp)
        hand_res = hand_det.detect_for_video(mp_image, timestamp)
        face_res = face_det.detect_for_video(mp_image, timestamp)

        data = process_frame(
            pose_res.pose_landmarks if pose_res else None,
            hand_res.hand_landmarks if hand_res else None,
            hand_res.handedness if hand_res else None,
            face_res.face_landmarks if face_res else None,
            face_res.face_blendshapes if face_res else None,
            face_res.facial_transformation_matrixes if face_res else None
        )

        data["t"] = t

        sender.send(data)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    sender.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()