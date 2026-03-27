from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logloss(y: List[int], p: List[float]) -> float:
    eps = 1e-12
    total = 0.0
    for yi, pi in zip(y, p):
        pi = min(1 - eps, max(eps, pi))
        total += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
    return total / len(y)


def accuracy(y: List[int], p: List[float], thr: float = 0.5) -> float:
    correct = 0
    for yi, pi in zip(y, p):
        pred = 1 if pi >= thr else 0
        if pred == yi:
            correct += 1
    return correct / len(y)


def make_dataset(n: int = 2600, seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[int] = []

    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1.2)
        x3 = rng.uniform(-2, 2)
        x4 = rng.gauss(0, 1)

        score = (
            1.3 * x1
            - 1.1 * x2
            + 1.7 * x1 * x3
            - 0.9 * abs(x4)
            + (0.8 if x3 > 0.5 else -0.4)
            + rng.gauss(0, 0.8)
        )
        p = sigmoid(score)
        yi = 1 if rng.random() < p else 0

        X.append([x1, x2, x3, x4])
        y.append(yi)

    return X, y


def split_train_valid(X: List[List[float]], y: List[int], valid_ratio: float = 0.25, seed: int = 7):
    idx = list(range(len(y)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - valid_ratio))

    tr, va = idx[:cut], idx[cut:]
    X_tr = [X[i] for i in tr]
    y_tr = [y[i] for i in tr]
    X_va = [X[i] for i in va]
    y_va = [y[i] for i in va]
    return X_tr, y_tr, X_va, y_va


def predict_logits(X: List[List[float]], stumps: List[Dict], lr: float) -> List[float]:
    out = [0.0 for _ in X]
    for stump in stumps:
        f = stump["feature"]
        thr = stump["threshold"]
        lv = stump["left_value"]
        rv = stump["right_value"]
        for i, row in enumerate(X):
            out[i] += lr * (lv if row[f] <= thr else rv)
    return out


def best_first_order_split(
    X: List[List[float]],
    residuals: List[float],
    min_leaf: int,
) -> Dict | None:
    n = len(X)
    p = len(X[0])

    total_sum = sum(residuals)
    total_sq = sum(r * r for r in residuals)
    parent_sse = total_sq - (total_sum * total_sum) / n

    best_gain = -1e18
    best: Dict | None = None

    for j in range(p):
        arr = sorted((X[i][j], residuals[i]) for i in range(n))

        left_n = 0
        left_sum = 0.0
        left_sq = 0.0

        for i in range(n - 1):
            v, r = arr[i]
            left_n += 1
            left_sum += r
            left_sq += r * r

            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            if v == arr[i + 1][0]:
                continue

            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq

            sse_left = left_sq - (left_sum * left_sum) / left_n
            sse_right = right_sq - (right_sum * right_sum) / right_n
            gain = parent_sse - (sse_left + sse_right)

            if gain > best_gain:
                thr = 0.5 * (v + arr[i + 1][0])
                best_gain = gain
                best = {
                    "feature": j,
                    "threshold": thr,
                    "left_value": left_sum / left_n,
                    "right_value": right_sum / right_n,
                    "gain": gain,
                }

    return best


def best_second_order_split(
    X: List[List[float]],
    grads: List[float],
    hess: List[float],
    l2: float,
    gamma: float,
    min_leaf: int,
) -> Dict | None:
    n = len(X)
    p = len(X[0])

    total_g = sum(grads)
    total_h = sum(hess)

    best_gain = -1e18
    best: Dict | None = None

    base_score = (total_g * total_g) / (total_h + l2)

    for j in range(p):
        arr = sorted((X[i][j], grads[i], hess[i]) for i in range(n))

        g_left = 0.0
        h_left = 0.0
        left_n = 0

        for i in range(n - 1):
            v, g_i, h_i = arr[i]
            g_left += g_i
            h_left += h_i
            left_n += 1

            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue
            if v == arr[i + 1][0]:
                continue

            g_right = total_g - g_left
            h_right = total_h - h_left

            gain = 0.5 * (
                (g_left * g_left) / (h_left + l2)
                + (g_right * g_right) / (h_right + l2)
                - base_score
            ) - gamma

            if gain > best_gain:
                thr = 0.5 * (v + arr[i + 1][0])
                best_gain = gain
                best = {
                    "feature": j,
                    "threshold": thr,
                    "left_value": -g_left / (h_left + l2),
                    "right_value": -g_right / (h_right + l2),
                    "gain": gain,
                }

    return best


def train_first_order_boosting(
    X_tr: List[List[float]],
    y_tr: List[int],
    X_va: List[List[float]],
    y_va: List[int],
    rounds: int = 50,
    lr: float = 0.15,
    min_leaf: int = 25,
):
    logits_tr = [0.0 for _ in y_tr]
    stumps: List[Dict] = []

    hist = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "gain": [],
    }

    for _ in range(rounds):
        p_tr = [sigmoid(z) for z in logits_tr]
        residuals = [yi - pi for yi, pi in zip(y_tr, p_tr)]

        stump = best_first_order_split(X_tr, residuals, min_leaf=min_leaf)
        if stump is None:
            break

        stumps.append(stump)

        f = stump["feature"]
        thr = stump["threshold"]
        lv = stump["left_value"]
        rv = stump["right_value"]

        for i, row in enumerate(X_tr):
            logits_tr[i] += lr * (lv if row[f] <= thr else rv)

        p_tr = [sigmoid(z) for z in logits_tr]
        logits_va = predict_logits(X_va, stumps, lr=lr)
        p_va = [sigmoid(z) for z in logits_va]

        hist["train_loss"].append(logloss(y_tr, p_tr))
        hist["valid_loss"].append(logloss(y_va, p_va))
        hist["train_acc"].append(accuracy(y_tr, p_tr))
        hist["valid_acc"].append(accuracy(y_va, p_va))
        hist["gain"].append(stump["gain"])

    return stumps, hist


def train_second_order_boosting(
    X_tr: List[List[float]],
    y_tr: List[int],
    X_va: List[List[float]],
    y_va: List[int],
    rounds: int = 50,
    lr: float = 0.15,
    l2: float = 1.0,
    gamma: float = 0.0,
    min_leaf: int = 25,
):
    logits_tr = [0.0 for _ in y_tr]
    stumps: List[Dict] = []

    hist = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "gain": [],
    }

    for _ in range(rounds):
        p_tr = [sigmoid(z) for z in logits_tr]
        grads = [pi - yi for yi, pi in zip(y_tr, p_tr)]
        hess = [max(1e-6, pi * (1.0 - pi)) for pi in p_tr]

        stump = best_second_order_split(
            X_tr,
            grads,
            hess,
            l2=l2,
            gamma=gamma,
            min_leaf=min_leaf,
        )
        if stump is None:
            break

        stumps.append(stump)

        f = stump["feature"]
        thr = stump["threshold"]
        lv = stump["left_value"]
        rv = stump["right_value"]

        for i, row in enumerate(X_tr):
            logits_tr[i] += lr * (lv if row[f] <= thr else rv)

        p_tr = [sigmoid(z) for z in logits_tr]
        logits_va = predict_logits(X_va, stumps, lr=lr)
        p_va = [sigmoid(z) for z in logits_va]

        hist["train_loss"].append(logloss(y_tr, p_tr))
        hist["valid_loss"].append(logloss(y_va, p_va))
        hist["train_acc"].append(accuracy(y_tr, p_tr))
        hist["valid_acc"].append(accuracy(y_va, p_va))
        hist["gain"].append(stump["gain"])

    return stumps, hist


def maybe_plot(hist_fo: Dict[str, List[float]], hist_so: Dict[str, List[float]]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not installed. Final losses:")
        print("First-order valid loss:", round(hist_fo["valid_loss"][-1], 5))
        print("Second-order valid loss:", round(hist_so["valid_loss"][-1], 5))
        return

    xs_fo = list(range(1, len(hist_fo["train_loss"]) + 1))
    xs_so = list(range(1, len(hist_so["train_loss"]) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    axs[0].plot(xs_fo, hist_fo["train_loss"], label="First-order train", color="#6b7280")
    axs[0].plot(xs_fo, hist_fo["valid_loss"], label="First-order valid", color="#9ca3af")
    axs[0].plot(xs_so, hist_so["train_loss"], label="Second-order train", color="#1d4ed8")
    axs[0].plot(xs_so, hist_so["valid_loss"], label="Second-order valid", color="#60a5fa")
    axs[0].set_title("Logloss Convergence")
    axs[0].set_xlabel("Boosting round")
    axs[0].set_ylabel("Logloss")
    axs[0].legend()

    axs[1].plot(xs_fo, hist_fo["gain"], label="First-order split gain", color="#6b7280")
    axs[1].plot(xs_so, hist_so["gain"], label="Second-order split gain", color="#1d4ed8")
    axs[1].set_title("Split Gain by Round")
    axs[1].set_xlabel("Boosting round")
    axs[1].set_ylabel("Gain")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = make_dataset(n=2600, seed=13)
    X_tr, y_tr, X_va, y_va = split_train_valid(X, y, valid_ratio=0.25, seed=3)

    _, hist_fo = train_first_order_boosting(
        X_tr,
        y_tr,
        X_va,
        y_va,
        rounds=45,
        lr=0.16,
        min_leaf=24,
    )

    _, hist_so = train_second_order_boosting(
        X_tr,
        y_tr,
        X_va,
        y_va,
        rounds=45,
        lr=0.16,
        l2=1.2,
        gamma=0.02,
        min_leaf=24,
    )

    print("First-order final:")
    print(
        {
            "train_loss": round(hist_fo["train_loss"][-1], 5),
            "valid_loss": round(hist_fo["valid_loss"][-1], 5),
            "train_acc": round(hist_fo["train_acc"][-1], 5),
            "valid_acc": round(hist_fo["valid_acc"][-1], 5),
        }
    )

    print("Second-order final:")
    print(
        {
            "train_loss": round(hist_so["train_loss"][-1], 5),
            "valid_loss": round(hist_so["valid_loss"][-1], 5),
            "train_acc": round(hist_so["train_acc"][-1], 5),
            "valid_acc": round(hist_so["valid_acc"][-1], 5),
        }
    )

    maybe_plot(hist_fo, hist_so)
