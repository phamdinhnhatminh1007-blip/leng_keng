# collect_data.py - Thu thap data gesture
# Cach dung:
#   - Gio gesture truoc webcam
#   - GIU phim so (0-9) de ghi mau cho chieu tuong ung (moi frame = 1 mau)
#   - Thu moi chieu ~200-300 mau, doi goc/khoang cach/tay trai-phai cho da dang
#   - Nhan q de thoat, data luu vao gestures.csv (chay lai se ghi noi tiep)
import os
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_utils import GESTURES, normalize_landmarks

CAM_W, CAM_H = 1280, 720
CSV_PATH = "gestures.csv"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")


def main():
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

    counts = [0] * len(GESTURES)
    # Dem lai so mau da co neu file ton tai
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as fp:
            for row in csv.reader(fp):
                lbl = int(row[0])
                if lbl < len(counts):
                    counts[lbl] += 1

    f = open(CSV_PATH, "a", newline="")
    writer = csv.writer(f)
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, int((time.time() - t0) * 1000))

        key = cv2.waitKey(1) & 0xFF
        hand_ok = bool(result.hand_landmarks)

        if hand_ok:
            landmarks = result.hand_landmarks[0]
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])),
                           4, (0, 200, 255), -1)
            # Giu phim so de ghi mau
            if ord('0') <= key <= ord('9'):
                label = key - ord('0')
                if label < len(GESTURES):
                    vec = normalize_landmarks(landmarks)
                    writer.writerow([label] + [f"{v:.5f}" for v in vec])
                    counts[label] += 1
                    cv2.putText(frame, f"GHI: {GESTURES[label]}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hien thi so mau tung chieu
        for i, name in enumerate(GESTURES):
            cv2.putText(frame, f"[{i}] {name}: {counts[i]}", (10, 140 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "GIU phim so de ghi mau | q: thoat",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Thu thap data gesture", frame)
        if key in (ord('q'), 27):
            break

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Da luu vao", CSV_PATH)
    for i, name in enumerate(GESTURES):
        print(f"  [{i}] {name}: {counts[i]} mau")


if __name__ == "__main__":
    main()
