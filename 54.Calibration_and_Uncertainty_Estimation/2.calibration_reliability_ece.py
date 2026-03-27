from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, x))))


def make_logits_and_labels(n: int = 1200, seed: int = 42) -> Tuple[List[float], List[int]]:
    random.seed(seed)
    logits, labels = [], []

    for _ in range(n):
        x = random.gauss(0, 1)
        true_p = sigmoid(1.2 * x)
        y = 1 if random.random() < true_p else 0

        # intentionally overconfident model logits
        logit = 2.0 * x + random.gauss(0, 0.4)
        logits.append(logit)
        labels.append(y)

    return logits, labels


def probs_from_logits(logits: List[float], T: float = 1.0) -> List[float]:
    return [sigmoid(z / T) for z in logits]


def ece(probs: List[float], labels: List[int], n_bins: int = 10):
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))

    total = len(probs)
    e = 0.0
    rel_x, rel_y, counts = [], [], []
    for b in bins:
        if not b:
            rel_x.append(0.0)
            rel_y.append(0.0)
            counts.append(0)
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        e += (len(b) / total) * abs(acc - conf)
        rel_x.append(conf)
        rel_y.append(acc)
        counts.append(len(b))

    return e, rel_x, rel_y, counts


def nll(probs: List[float], labels: List[int]) -> float:
    s = 0.0
    for p, y in zip(probs, labels):
        p = min(1 - 1e-12, max(1e-12, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(probs)


def tune_temperature(logits: List[float], labels: List[int], grid=None) -> float:
    if grid is None:
        grid = [0.5 + i * 0.05 for i in range(41)]

    best_T, best_nll = grid[0], float("inf")
    for T in grid:
        p = probs_from_logits(logits, T=T)
        val = nll(p, labels)
        if val < best_nll:
            best_nll = val
            best_T = T
    return best_T


def maybe_plot(rel_x1, rel_y1, rel_x2, rel_y2):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Reliability points before:", [(round(x, 3), round(y, 3)) for x, y in zip(rel_x1, rel_y1)])
        print("Reliability points after :", [(round(x, 3), round(y, 3)) for x, y in zip(rel_x2, rel_y2)])
        return

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    plt.plot(rel_x1, rel_y1, marker="o", label="before")
    plt.plot(rel_x2, rel_y2, marker="o", label="after T-scaling")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logits, labels = make_logits_and_labels(n=1400, seed=13)

    split = 900
    log_train, y_train = logits[:split], labels[:split]
    log_test, y_test = logits[split:], labels[split:]

    p_before = probs_from_logits(log_test, T=1.0)
    e_before, x1, y1, _ = ece(p_before, y_test, n_bins=12)

    T = tune_temperature(log_train, y_train)
    p_after = probs_from_logits(log_test, T=T)
    e_after, x2, y2, _ = ece(p_after, y_test, n_bins=12)

    print("Best temperature:", round(T, 3))
    print("ECE before:", round(e_before, 5))
    print("ECE after :", round(e_after, 5))

    maybe_plot(x1, y1, x2, y2)
