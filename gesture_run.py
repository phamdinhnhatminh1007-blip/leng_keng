# gesture_run.py - Nhan dien gesture realtime
# Cach dung: python gesture_run.py (sau khi da train xong gesture_model.joblib)
import os
import time
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import joblib

from gesture_utils import GESTURES, normalize_landmarks

CAM_W, CAM_H = 1280, 720
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
GESTURE_MODEL = "gesture_model.joblib"

CONF_THRESHOLD = 0.8   # duoi nguong nay -> coi nhu khong ro chieu
SMOOTH_FRAMES = 7      # lay ket qua pho bien nhat trong N frame gan nhat


def main():
    clf = joblib.load(GESTURE_MODEL)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    history = deque(maxlen=SMOOTH_FRAMES)
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int((time.time() - t0) * 1000))

        label_text = "..."
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])),
                           4, (0, 200, 255), -1)

            vec = normalize_landmarks(landmarks).reshape(1, -1)
            probs = clf.predict_proba(vec)[0]
            pred = int(probs.argmax())
            history.append(pred if probs[pred] >= CONF_THRESHOLD else -1)

            # Smoothing: chon nhan xuat hien nhieu nhat trong N frame
            vote, n = Counter(history).most_common(1)[0]
            if vote != -1 and n >= SMOOTH_FRAMES // 2 + 1:
                label_text = f"{GESTURES[vote]} ({probs[vote]:.0%})"
                # >>> DAT LOGIC GAME TAI DAY, vi du:
                # if vote == 0: cast_skill_1()
        else:
            history.clear()

        cv2.putText(frame, label_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Nhan dien gesture - q de thoat", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
