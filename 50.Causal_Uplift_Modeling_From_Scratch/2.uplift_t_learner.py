from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))


def fit_logistic(X: List[List[float]], y: List[int], lr: float = 0.04, epochs: int = 300):
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


def predict_proba(X: List[List[float]], w: List[float]) -> List[float]:
    d = len(X[0])
    out = []
    for xi in X:
        z = w[0] + sum(w[j + 1] * xi[j] for j in range(d))
        out.append(sigmoid(z))
    return out


def synthetic_uplift_data(n: int = 1500, seed: int = 42):
    random.seed(seed)
    X, t, y = [], [], []

    for _ in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        x3 = random.uniform(-1, 1)

        # randomized treatment assignment
        treat = 1 if random.random() < 0.5 else 0

        base = -0.2 + 0.8 * x1 - 0.6 * x2 + 0.3 * x3
        uplift = 0.9 * x1 - 0.5 * x2  # heterogeneous treatment effect
        logit = base + (uplift if treat else 0.0)

        prob = sigmoid(logit)
        outcome = 1 if random.random() < prob else 0

        X.append([x1, x2, x3])
        t.append(treat)
        y.append(outcome)

    return X, t, y


def cumulative_uplift_curve(t: List[int], y: List[int], uplift_scores: List[float], bins: int = 10):
    idx = list(range(len(y)))
    idx.sort(key=lambda i: uplift_scores[i], reverse=True)

    out = []
    for b in range(1, bins + 1):
        k = int(len(y) * b / bins)
        sel = idx[:k]

        yt = [y[i] for i in sel if t[i] == 1]
        yc = [y[i] for i in sel if t[i] == 0]

        rt = sum(yt) / len(yt) if yt else 0.0
        rc = sum(yc) / len(yc) if yc else 0.0
        out.append((b / bins, rt - rc))

    return out


def maybe_plot_curve(curve):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Cumulative uplift curve:")
        for frac, val in curve:
            print(f"top={int(frac*100):2d}% uplift={val:.4f}")
        return

    xs = [x for x, _ in curve]
    ys = [y for _, y in curve]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.title("Cumulative Uplift by Targeting Fraction")
    plt.xlabel("Targeted Population Fraction")
    plt.ylabel("Observed Uplift (treated rate - control rate)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, t, y = synthetic_uplift_data(n=1800, seed=7)

    treated_idx = [i for i, ti in enumerate(t) if ti == 1]
    control_idx = [i for i, ti in enumerate(t) if ti == 0]

    X_t = [X[i] for i in treated_idx]
    y_t = [y[i] for i in treated_idx]
    X_c = [X[i] for i in control_idx]
    y_c = [y[i] for i in control_idx]

    w_t = fit_logistic(X_t, y_t, lr=0.03, epochs=350)
    w_c = fit_logistic(X_c, y_c, lr=0.03, epochs=350)

    p_t = predict_proba(X, w_t)
    p_c = predict_proba(X, w_c)
    uplift = [a - b for a, b in zip(p_t, p_c)]

    curve = cumulative_uplift_curve(t, y, uplift, bins=10)
    maybe_plot_curve(curve)

    print("Top segment observed uplift:")
    for frac, val in curve[:3]:
        print(f"top {int(frac*100)}% -> uplift {val:.4f}")
