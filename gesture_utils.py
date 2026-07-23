# gesture_utils.py - dung chung cho collect / train / run
import numpy as np

# Dat ten chieu tai day - phim so khi thu data ung voi index trong list nay
GESTURES = [
    "Chieu 1 - Xoe ban tay",   # phim 0
    "Chieu 2 - Nam dam",       # phim 1
    "Chieu 3 - Dơ ngón fuck",                 # phim 2
    "Chieu 4 - Dơ ngón hi",                 # phim 3
    "Chieu 5 - Dơ ngón cái",                 # phim 4
    "Chieu 6 - Dơ ngón chỏ và ngón út",                 # phim 5
    # them toi da 10 chieu (phim 0-9)
]


def normalize_landmarks(landmarks):
    """Chuyen 21 landmarks thanh vector 42 chieu, bat bien voi vi tri va kich thuoc tay.

    - Dich ve goc: tru toa do co tay (landmark 0)
    - Chia cho khoang cach lon nhat -> khong phu thuoc tay gan/xa camera
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                      # co tay ve goc toa do
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= scale
    return pts.flatten()               # shape (42,)
