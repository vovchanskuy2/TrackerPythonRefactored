import cv2
import time
import mediapipe as mp

from detector import create_detectors, download_models
from capture import get_capture
from processor import process_frame


def main():
    cap = get_capture(0)
    pose_det, hand_det, face_det = create_detectors()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time() * 1000)

        pose_res = pose_det.detect_for_video(mp_image, timestamp)

        hand_res = None
        face_res = None

        if pose_res.pose_landmarks:
            pose_lm = pose_res.pose_landmarks[0]

            # руки если видны запястья
            if pose_lm[15].visibility > 0.4 or pose_lm[16].visibility > 0.4:
                hand_res = hand_det.detect_for_video(mp_image, timestamp)

            # лицо если виден нос
            if pose_lm[0].visibility > 0.4:
                face_res = face_det.detect_for_video(mp_image, timestamp)

        data = process_frame(
            pose_res.pose_landmarks if pose_res else None,
            hand_res.hand_landmarks if hand_res else None,
            face_res.face_landmarks if face_res else None
        )

        # DEBUG вывод
        print("POSE:", None if data["pose"] is None else data["pose"].shape)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()