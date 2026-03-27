from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))


def generate_stream(n: int = 4000, drift_at: int = 2200, seed: int = 42):
    random.seed(seed)
    X, y = [], []
    for i in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = 1.0
        if i < drift_at:
            logit = 1.3 * x1 - 1.0 * x2
        else:
            logit = -1.1 * x1 + 1.4 * x2 + 0.2
        p = sigmoid(logit)
        yi = 1 if random.random() < p else 0
        X.append([x1, x2, x3])
        y.append(yi)
    return X, y


def online_predict(x: List[float], w: List[float]) -> float:
    return sigmoid(sum(wi * xi for wi, xi in zip(w, x)))


def run_online(
    X: List[List[float]],
    y: List[int],
    adaptive: bool,
    base_lr: float = 0.03,
    win: int = 120,
    gap_thresh: float = 0.08,
):
    w = [0.0, 0.0, 0.0]
    errors = []
    acc_curve = []
    drift_points = []
    cooldown = 0
    short_win = max(40, win // 2)
    long_win = max(short_win + 20, win * 3)

    for i, (xi, yi) in enumerate(zip(X, y), 1):
        p = online_predict(xi, w)
        pred = 1 if p >= 0.5 else 0
        err = 0 if pred == yi else 1
        errors.append(err)

        # drift detection: recent window vs previous window
        lr = base_lr
        if adaptive and cooldown > 0:
            cooldown -= 1

        if adaptive and len(errors) >= long_win + short_win and i > 900 and cooldown == 0:
            a = sum(errors[-short_win:]) / short_win
            b = sum(errors[-(short_win + long_win) : -short_win]) / long_win
            if a - b > gap_thresh:
                drift_points.append(i)
                lr = base_lr * 3.0
                # mild reset to adapt faster
                w = [0.7 * v for v in w]
                cooldown = 180

        # online logistic update
        grad = yi - p
        for j in range(len(w)):
            w[j] += lr * grad * xi[j]

        # smoothed accuracy
        k = min(200, len(errors))
        acc = 1.0 - (sum(errors[-k:]) / k)
        acc_curve.append(acc)

    return acc_curve, drift_points


def maybe_plot(acc_static, acc_adapt, drift_points):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Final rolling acc static:", round(acc_static[-1], 4))
        print("Final rolling acc adapt :", round(acc_adapt[-1], 4))
        print("Drift points detected   :", drift_points[:10])
        return

    plt.figure(figsize=(9, 4))
    plt.plot(acc_static, label="static online")
    plt.plot(acc_adapt, label="adaptive online")
    for d in drift_points[:10]:
        plt.axvline(d, color="red", alpha=0.15)
    plt.title("Prequential Accuracy Under Concept Drift")
    plt.xlabel("Time step")
    plt.ylabel("Rolling accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = generate_stream(n=4200, drift_at=2300, seed=11)
    acc_static, _ = run_online(X, y, adaptive=False)
    acc_adapt, drifts = run_online(X, y, adaptive=True, base_lr=0.03, win=120, gap_thresh=0.09)

    print("Final static acc:", round(acc_static[-1], 4))
    print("Final adapt  acc:", round(acc_adapt[-1], 4))
    print("Detected drift points:", drifts[:8], "... total", len(drifts))

    maybe_plot(acc_static, acc_adapt, drifts)
