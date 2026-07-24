# gesture_utils.py - dung chung cho collect / train / run
import numpy as np

# Dat ten chieu tai day - phim so khi thu data ung voi index trong list nay
GESTURES = [
    "Chieu 1 - Xoe ban tay",   # phim 0
    "Chieu 2 - Nam dam",       # phim 1
    "Chieu 3 - Dơ ngón fuck",                 # phim 2
    "Chieu 4 - Dơ ngón hi",                 # phim 3
    "Chieu 5 - Dơ ngón cái",                 # phim 4
    "Chieu 6 - The Spider-Man",                          # phim 5
    # them toi da 10 chieu (phim 0-9)
]


class FastMLP:
    """Ban inference NumPy nhe cua sklearn MLPClassifier (ReLU + softmax)."""

    def __init__(self, classes, coefs, intercepts):
        self.classes_ = classes
        self.coefs_ = coefs
        self.intercepts_ = intercepts

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            layer_count = int(data["layer_count"])
            classes = data["classes"].copy()
            coefs = [data[f"coef_{i}"].copy() for i in range(layer_count)]
            intercepts = [
                data[f"intercept_{i}"].copy() for i in range(layer_count)
            ]
        return cls(classes, coefs, intercepts)

    def predict_proba(self, features):
        values = np.asarray(features, dtype=self.coefs_[0].dtype)
        for weights, bias in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            values = np.maximum(values @ weights + bias, 0.0)

        logits = values @ self.coefs_[-1] + self.intercepts_[-1]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def save_fast_mlp(model, path):
    """Xuat MLPClassifier sang NPZ de main.py load ma khong import sklearn."""
    payload = {
        "classes": np.asarray(model.classes_),
        "layer_count": np.asarray(len(model.coefs_)),
    }
    for index, (weights, bias) in enumerate(
        zip(model.coefs_, model.intercepts_)
    ):
        payload[f"coef_{index}"] = weights
        payload[f"intercept_{index}"] = bias
    np.savez_compressed(path, **payload)


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


def _joint_angle(points, a, b, c):
    """Goc ABC (do) trong mat phang anh, khong phu thuoc huong ban tay."""
    ba = points[a] - points[b]
    bc = points[c] - points[b]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _finger_extended(points, wrist, mcp, pip, dip, tip):
    """Kiem tra mot ngon co thuc su duoi thang, ke ca khi tay bi xoay."""
    pip_angle = _joint_angle(points, mcp, pip, dip)
    dip_angle = _joint_angle(points, pip, dip, tip)
    tip_reach = np.linalg.norm(points[tip] - points[wrist])
    pip_reach = np.linalg.norm(points[pip] - points[wrist])
    return pip_angle >= 145.0 and dip_angle >= 145.0 and tip_reach >= pip_reach * 1.12


def is_spiderman_pose(landmarks):
    """Chi True khi dung the Spider-Man: cai + tro + ut duoi.

    Day la lop xac minh hinh hoc bo sung cho model ML, nham tranh chieu bang
    bi kich hoat boi cac the tay na na trong du lieu huan luyen.
    """
    if len(landmarks) < 21:
        return False

    points = np.asarray([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    index_up = _finger_extended(points, 0, 5, 6, 7, 8)
    middle_up = _finger_extended(points, 0, 9, 10, 11, 12)
    ring_up = _finger_extended(points, 0, 13, 14, 15, 16)
    pinky_up = _finger_extended(points, 0, 17, 18, 19, 20)

    # Ngon cai co huong van dong khac bon ngon con lai, nen dung hai goc khop
    # va do vuon khoi co tay de nhan biet no dang duoi.
    thumb_mcp_angle = _joint_angle(points, 1, 2, 3)
    thumb_ip_angle = _joint_angle(points, 2, 3, 4)
    thumb_tip_reach = np.linalg.norm(points[4] - points[0])
    thumb_ip_reach = np.linalg.norm(points[3] - points[0])
    palm_size = max(np.linalg.norm(points[9] - points[0]), 1e-6)
    thumb_away_from_palm = np.linalg.norm(points[4] - points[9]) >= palm_size * 0.42
    thumb_up = (
        thumb_mcp_angle >= 140.0
        and thumb_ip_angle >= 145.0
        and thumb_tip_reach >= thumb_ip_reach * 1.08
        and thumb_away_from_palm
    )

    return index_up and pinky_up and thumb_up and not middle_up and not ring_up


def is_lightning_pose(landmarks):
    """Nhan nhanh the chi tay hoac chu V de tung set ma khong cho ML vote lau."""
    if len(landmarks) < 21:
        return False

    points = np.asarray([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    index_up = _finger_extended(points, 0, 5, 6, 7, 8)
    ring_up = _finger_extended(points, 0, 13, 14, 15, 16)
    pinky_up = _finger_extended(points, 0, 17, 18, 19, 20)

    # Ngon giua co the gap (chi tay) hoac duoi (chu V); ap ut va ut phai gap.
    return index_up and not ring_up and not pinky_up
