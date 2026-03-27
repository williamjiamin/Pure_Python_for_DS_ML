from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))


def make_grouped_data(n: int = 2400, shift: float = 1.2, seed: int = 42):
    random.seed(seed)
    X, y, g = [], [], []

    for i in range(n):
        group = 0 if random.random() < 0.7 else 1
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)

        if group == 0:
            logit = 1.8 * x1 - 1.2 * x2 + random.gauss(0, 0.4)
        else:
            # shifted group relationship
            logit = (1.8 - shift) * x1 + (1.1 + shift * 0.8) * x2 + random.gauss(0, 0.5)

        yi = 1 if random.random() < sigmoid(logit) else 0
        X.append([x1, x2, 1.0])
        y.append(yi)
        g.append(group)

    return X, y, g


def train_erm(X, y, lr=0.06, epochs=60):
    w = [0.0, 0.0, 0.0]
    n = len(X)
    for _ in range(epochs):
        idx = list(range(n))
        random.shuffle(idx)
        for i in idx:
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, X[i])))
            grad = y[i] - p
            for j in range(3):
                w[j] += lr * grad * X[i][j]
    return w


def train_group_robust(X, y, g, lr=0.06, epochs=60, gamma=2.0):
    w = [0.0, 0.0, 0.0]
    n = len(X)

    for _ in range(epochs):
        # estimate group losses under current model
        loss_sum = [0.0, 0.0]
        cnt = [0, 0]
        for xi, yi, gi in zip(X, y, g):
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)))
            p = min(1 - 1e-12, max(1e-12, p))
            loss = -(yi * math.log(p) + (1 - yi) * math.log(1 - p))
            loss_sum[gi] += loss
            cnt[gi] += 1

        loss_mean = [loss_sum[k] / max(1, cnt[k]) for k in [0, 1]]
        # higher weight on higher-loss group
        max_loss = max(loss_mean)
        grp_w = [math.exp(gamma * (lm - max_loss)) for lm in loss_mean]
        s = sum(grp_w)
        grp_w = [v / s for v in grp_w]

        idx = list(range(n))
        random.shuffle(idx)
        for i in idx:
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, X[i])))
            grad = y[i] - p
            weight = grp_w[g[i]]
            for j in range(3):
                w[j] += lr * weight * grad * X[i][j]

    return w


def evaluate(X, y, g, w):
    correct = 0
    corr = [0, 0]
    cnt = [0, 0]

    for xi, yi, gi in zip(X, y, g):
        p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)))
        pred = 1 if p >= 0.5 else 0
        hit = int(pred == yi)
        correct += hit
        corr[gi] += hit
        cnt[gi] += 1

    overall = correct / len(X)
    g0 = corr[0] / max(1, cnt[0])
    g1 = corr[1] / max(1, cnt[1])
    worst = min(g0, g1)
    return overall, g0, g1, worst


def maybe_plot(metrics_erm, metrics_rob):
    labels = ["overall", "group0", "group1", "worst_group"]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("ERM metrics:", [round(v, 4) for v in metrics_erm])
        print("ROB metrics:", [round(v, 4) for v in metrics_rob])
        return

    x = list(range(len(labels)))
    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.15 for i in x], metrics_erm, width=0.3, label="ERM")
    plt.bar([i + 0.15 for i in x], metrics_rob, width=0.3, label="Robust")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title("Average vs Worst-Group Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Xtr, ytr, gtr = make_grouped_data(n=2600, shift=1.3, seed=9)
    Xte, yte, gte = make_grouped_data(n=1200, shift=1.7, seed=10)

    w_erm = train_erm(Xtr, ytr, lr=0.06, epochs=65)
    w_rob = train_group_robust(Xtr, ytr, gtr, lr=0.06, epochs=65, gamma=3.0)

    m_erm = evaluate(Xte, yte, gte, w_erm)
    m_rob = evaluate(Xte, yte, gte, w_rob)

    print("ERM metrics (overall,g0,g1,worst):", tuple(round(v, 4) for v in m_erm))
    print("ROB metrics (overall,g0,g1,worst):", tuple(round(v, 4) for v in m_rob))

    maybe_plot(m_erm, m_rob)
