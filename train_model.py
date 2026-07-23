# train_model.py - Train classifier tu gestures.csv
# Cach dung: python train_model.py
# Yeu cau: pip install scikit-learn joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

from gesture_utils import GESTURES

CSV_PATH = "gestures.csv"
MODEL_OUT = "gesture_model.joblib"


def main():
    data = np.loadtxt(CSV_PATH, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    print(f"Tong {len(y)} mau, {X.shape[1]} dac trung")
    for i in sorted(set(y)):
        name = GESTURES[i] if i < len(GESTURES) else f"label {i}"
        print(f"  [{i}] {name}: {(y == i).sum()} mau")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                          early_stopping=True, random_state=42)
    model.fit(X_train, y_train)

    print("\nKet qua tren tap test:")
    print(classification_report(y_test, model.predict(X_test),
                                zero_division=0))

    joblib.dump(model, MODEL_OUT)
    print("Da luu model:", MODEL_OUT)


if __name__ == "__main__":
    main()
