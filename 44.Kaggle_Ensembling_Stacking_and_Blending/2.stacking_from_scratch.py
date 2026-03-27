from __future__ import annotations

import csv
import math
import random
from typing import Callable, Dict, List


DIABETES_SCHEMA = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_data(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.reader(f))

    if not raw:
        return [], []

    first = raw[0]
    if sum(_is_number(x) for x in first) >= max(1, int(0.8 * len(first))):
        header = DIABETES_SCHEMA[: len(first)]
        data_rows = raw
    else:
        header = first
        data_rows = raw[1:]

    rows = []
    for r in data_rows:
        if len(r) != len(header):
            continue
        rows.append({k: v for k, v in zip(header, r)})

    X = [[
        float(r["BMI"]),
        float(r["Age"]),
        float(r["Glucose"]),
        float(r["BloodPressure"]),
    ] for r in rows]
    y = [int(float(r["Outcome"])) for r in rows]
    return X, y


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    return folds


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))


def fit_logistic(X: List[List[float]], y: List[int], lr: float = 0.00001, epochs: int = 300):
    d = len(X[0])
    w = [0.0] * (d + 1)
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = w[0] + sum(w[j + 1] * xi[j] for j in range(d))
            p = sigmoid(z)
            err = yi - p
            w[0] += lr * err
            for j in range(d):
                w[j + 1] += lr * err * xi[j]
    return w


def predict_logistic(X: List[List[float]], w: List[float]) -> List[float]:
    d = len(X[0])
    out = []
    for xi in X:
        z = w[0] + sum(w[j + 1] * xi[j] for j in range(d))
        out.append(sigmoid(z))
    return out


def fit_stump_feature_threshold(X: List[List[float]], y: List[int], feature: int):
    vals = sorted(x[feature] for x in X)
    thresholds = vals[::max(1, len(vals) // 20)]
    best_thr, best_acc = thresholds[0], -1.0
    for t in thresholds:
        pred = [1 if x[feature] >= t else 0 for x in X]
        acc = sum(a == b for a, b in zip(pred, y)) / len(y)
        if acc > best_acc:
            best_acc = acc
            best_thr = t
    return best_thr


def predict_stump(X: List[List[float]], feature: int, thr: float) -> List[float]:
    return [0.8 if x[feature] >= thr else 0.2 for x in X]


def auc(y_true: List[int], y_prob: List[float]) -> float:
    pos = [p for p, y in zip(y_prob, y_true) if y == 1]
    neg = [p for p, y in zip(y_prob, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = len(pos) * len(neg)
    for pp in pos:
        for pn in neg:
            if pp > pn:
                wins += 1
            elif pp == pn:
                wins += 0.5
    return wins / total


def maybe_plot(scores: Dict[str, float]) -> None:
    labels = list(scores.keys())
    vals = [scores[k] for k in labels]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Scores:")
        for k, v in scores.items():
            print(f"- {k}: {v:.5f}")
        return

    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.title("Model AUC Comparison")
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_data("Y.Kaggle_Data/diabetes.csv")
    n = len(X)
    folds = kfold_indices(n, k=5, seed=123)

    oof_lr = [0.0] * n
    oof_stump1 = [0.0] * n
    oof_stump2 = [0.0] * n

    for valid in folds:
        valid_set = set(valid)
        train = [i for i in range(n) if i not in valid_set]

        X_train, y_train = [X[i] for i in train], [y[i] for i in train]
        X_valid = [X[i] for i in valid]

        w_lr = fit_logistic(X_train, y_train, lr=1e-5, epochs=250)
        p_lr = predict_logistic(X_valid, w_lr)

        thr1 = fit_stump_feature_threshold(X_train, y_train, feature=2)  # glucose
        thr2 = fit_stump_feature_threshold(X_train, y_train, feature=0)  # bmi
        p_s1 = predict_stump(X_valid, 2, thr1)
        p_s2 = predict_stump(X_valid, 0, thr2)

        for idx, p in zip(valid, p_lr):
            oof_lr[idx] = p
        for idx, p in zip(valid, p_s1):
            oof_stump1[idx] = p
        for idx, p in zip(valid, p_s2):
            oof_stump2[idx] = p

    # Blending and stacking meta features.
    blend = [(a + b + c) / 3.0 for a, b, c in zip(oof_lr, oof_stump1, oof_stump2)]
    meta_X = [[oof_lr[i], oof_stump1[i], oof_stump2[i]] for i in range(n)]
    w_meta = fit_logistic(meta_X, y, lr=0.05, epochs=600)
    stack = predict_logistic(meta_X, w_meta)

    scores = {
        "base_lr": auc(y, oof_lr),
        "base_stump_g": auc(y, oof_stump1),
        "base_stump_bmi": auc(y, oof_stump2),
        "blend": auc(y, blend),
        "stack": auc(y, stack),
    }

    print("OOF AUC scores:")
    for k, v in scores.items():
        print(f"- {k}: {v:.5f}")

    maybe_plot(scores)


if __name__ == "__main__":
    main()
