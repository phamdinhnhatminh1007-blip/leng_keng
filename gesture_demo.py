# main.py - Nhan dien gesture realtime
# Cach dung: python main.py (sau khi da train xong gesture_model.joblib)
import os
import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import joblib

from gesture_utils import GESTURES, normalize_landmarks

CAM_W, CAM_H = 1280, 720
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
GESTURE_MODEL = os.path.join(BASE_DIR, "gesture_model.joblib")

CONF_THRESHOLD = 0.8   # duoi nguong nay -> coi nhu khong ro chieu
SMOOTH_FRAMES = 7      # lay ket qua pho bien nhat trong N frame gan nhat


def flip_landmark_vector(vec):
    """Lat ngang cac moc ban tay quanh co tay (moc 0)."""
    flipped = vec.reshape(21, 2).copy()
    flipped[:, 0] *= -1
    return flipped.reshape(-1)


def predict_with_mirror(clf, vec):
    """Du doan ca dang goc va dang lat, chon ket qua tin cay hon."""
    variants = np.stack((vec, flip_landmark_vector(vec)))
    probabilities = clf.predict_proba(variants)

    best_probability_each_variant = probabilities.max(axis=1)
    variant_index = int(best_probability_each_variant.argmax())
    class_index = int(probabilities[variant_index].argmax())

    label = int(clf.classes_[class_index])
    confidence = float(probabilities[variant_index, class_index])
    used_mirror = variant_index == 1
    return label, confidence, used_mirror


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
        hand_text = "Hand: --"
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])),
                           4, (0, 200, 255), -1)

            vec = normalize_landmarks(landmarks)
            pred, confidence, used_mirror = predict_with_mirror(clf, vec)
            history.append(
                (pred, confidence) if confidence >= CONF_THRESHOLD else (-1, confidence)
            )

            handedness = "Unknown"
            if result.handedness and result.handedness[0]:
                handedness = result.handedness[0][0].category_name
            variant = "mirrored" if used_mirror else "original"
            hand_text = f"Hand: {handedness} | model input: {variant}"

            # Smoothing: chon nhan xuat hien nhieu nhat trong N frame
            vote, n = Counter(label for label, _ in history).most_common(1)[0]
            if vote != -1 and n >= SMOOTH_FRAMES // 2 + 1:
                vote_confidences = [
                    conf for label, conf in history if label == vote
                ]
                mean_confidence = sum(vote_confidences) / len(vote_confidences)
                label_text = f"{GESTURES[vote]} ({mean_confidence:.0%})"
                # >>> DAT LOGIC GAME TAI DAY, vi du:
                # if vote == 0: cast_skill_1()
        else:
            history.clear()

        cv2.putText(frame, label_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, hand_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Nhan dien gesture - q de thoat", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
