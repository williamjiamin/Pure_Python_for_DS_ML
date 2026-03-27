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


def auc_score(y: List[int], p: List[float]) -> float:
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(p, y), key=lambda t: t[0])
    rank = 1
    rank_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        pos_here = sum(lbl for _, lbl in pairs[i : j + 1])
        rank_pos += avg_rank * pos_here
        rank += (j - i + 1)
        i = j + 1

    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def standardize_fit(X: List[List[float]]):
    d = len(X[0])
    mean = [0.0] * d
    std = [0.0] * d

    for j in range(d):
        col = [row[j] for row in X]
        m = sum(col) / len(col)
        v = sum((x - m) ** 2 for x in col) / len(col)
        mean[j] = m
        std[j] = math.sqrt(v) + 1e-12

    Xs = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    return Xs, mean, std


def standardize_apply(X: List[List[float]], mean: List[float], std: List[float]):
    d = len(mean)
    return [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]


def fit_logreg(
    X: List[List[float]],
    y: List[int],
    epochs: int = 240,
    lr: float = 0.045,
    l2: float = 3e-4,
    seed: int = 1,
):
    rng = random.Random(seed)
    n = len(y)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    for _ in range(epochs):
        idx = list(range(n))
        rng.shuffle(idx)

        gw = [0.0] * d
        gb = 0.0

        for i in idx:
            z = b
            xi = X[i]
            for j in range(d):
                z += w[j] * xi[j]
            p = sigmoid(z)
            err = p - y[i]

            for j in range(d):
                gw[j] += err * xi[j]
            gb += err

        inv_n = 1.0 / n
        for j in range(d):
            grad = gw[j] * inv_n + l2 * w[j]
            w[j] -= lr * grad
        b -= lr * gb * inv_n

    return w, b


def predict_proba(X: List[List[float]], w: List[float], b: float) -> List[float]:
    out = []
    for row in X:
        z = b
        for j in range(len(w)):
            z += w[j] * row[j]
        out.append(sigmoid(z))
    return out


def simulate_domain(n: int, seed: int, mode: str = "train"):
    rng = random.Random(seed)

    X: List[List[float]] = []
    y: List[int] = []

    for _ in range(n):
        x0 = rng.gauss(0, 1.0)
        x1 = rng.gauss(0, 1.0)
        x2 = rng.gauss(0, 1.0)
        x3 = rng.gauss(0, 1.0)

        if mode == "train":
            x4 = rng.gauss(0.0, 1.0)
            x5 = rng.gauss(0.0, 1.0)
            x6 = rng.gauss(0.0, 1.0)
            x7 = rng.gauss(0.0, 1.0)
            x8 = rng.gauss(0.0, 1.0)
            x9 = rng.gauss(0.0, 1.0)
            spurious_coef = 1.15
        else:
            x4 = rng.gauss(1.4, 1.4)  # shifted + noisier
            x5 = rng.gauss(0.8, 1.3)
            x6 = rng.gauss(-0.6, 1.2)
            x7 = rng.gauss(-1.2, 1.0)
            x8 = rng.gauss(0.4, 1.5)
            x9 = rng.gauss(0.0, 1.0)
            spurious_coef = -0.15  # unreliable out-of-domain

        score = (
            1.35 * x0
            - 1.1 * x1
            + 0.65 * x2 * x3
            + spurious_coef * x4
            + 0.25 * x6
            + rng.gauss(0, 0.8)
        )
        p = sigmoid(score)
        yi = 1 if rng.random() < p else 0

        X.append([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])
        y.append(yi)

    return X, y


def split_fit_valid(X: List[List[float]], y: List[int], ratio: float = 0.72, seed: int = 9):
    idx = list(range(len(y)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    cut = int(len(y) * ratio)
    fit_idx = idx[:cut]
    val_idx = idx[cut:]

    X_fit = [X[i] for i in fit_idx]
    y_fit = [y[i] for i in fit_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    return X_fit, y_fit, X_val, y_val


def select_features(X: List[List[float]], keep: List[int]) -> List[List[float]]:
    return [[row[j] for j in keep] for row in X]


def eval_target_model(
    X_fit: List[List[float]],
    y_fit: List[int],
    X_val: List[List[float]],
    y_val: List[int],
    X_ood: List[List[float]],
    y_ood: List[int],
):
    X_fit_s, mean, std = standardize_fit(X_fit)
    X_val_s = standardize_apply(X_val, mean, std)
    X_ood_s = standardize_apply(X_ood, mean, std)

    w, b = fit_logreg(X_fit_s, y_fit, epochs=240, lr=0.045, l2=3e-4, seed=2)

    p_val = predict_proba(X_val_s, w, b)
    p_ood = predict_proba(X_ood_s, w, b)

    return {
        "valid_auc": auc_score(y_val, p_val),
        "valid_logloss": logloss(y_val, p_val),
        "ood_auc": auc_score(y_ood, p_ood),
        "ood_logloss": logloss(y_ood, p_ood),
    }


def adversarial_weights(X_fit: List[List[float]], X_ood: List[List[float]]):
    X_adv = X_fit + X_ood
    y_adv = [0] * len(X_fit) + [1] * len(X_ood)

    X_adv_s, _, _ = standardize_fit(X_adv)
    w, b = fit_logreg(X_adv_s, y_adv, epochs=260, lr=0.05, l2=3e-4, seed=5)
    p = predict_proba(X_adv_s, w, b)

    return w, b, auc_score(y_adv, p)


def maybe_plot(feature_scores: List[float], ks: List[int], valid_auc: List[float], ood_auc: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not installed. Skipping plots.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))

    axs[0].bar(range(len(feature_scores)), feature_scores, color="#1d4ed8")
    axs[0].set_title("Adversarial Shift Score by Feature")
    axs[0].set_xlabel("Feature index")
    axs[0].set_ylabel("|weight|")

    axs[1].plot(ks, valid_auc, marker="o", label="in-domain valid AUC", color="#9ca3af")
    axs[1].plot(ks, ood_auc, marker="o", label="OOD AUC", color="#1d4ed8")
    axs[1].set_title("Robustness vs Removed Drift Features")
    axs[1].set_xlabel("Top drift features removed")
    axs[1].set_ylabel("AUC")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train_all, y_train_all = simulate_domain(n=4200, seed=8, mode="train")
    X_ood, y_ood = simulate_domain(n=2200, seed=88, mode="ood")

    X_fit, y_fit, X_val, y_val = split_fit_valid(X_train_all, y_train_all, ratio=0.72, seed=3)

    baseline = eval_target_model(X_fit, y_fit, X_val, y_val, X_ood, y_ood)
    print("Baseline (all features):", {k: round(v, 5) for k, v in baseline.items()})

    w_adv, _, adv_auc = adversarial_weights(X_fit, X_ood)
    feature_scores = [abs(v) for v in w_adv]

    ranked = sorted(range(len(feature_scores)), key=lambda j: feature_scores[j], reverse=True)
    print("Adversarial validation AUC (train-vs-ood):", round(adv_auc, 5))
    print("Top drift features:", ranked[:5])

    ks = [0, 1, 2, 3, 4]
    valid_curve = []
    ood_curve = []

    best = None
    for k in ks:
        removed = set(ranked[:k])
        keep = [j for j in range(len(feature_scores)) if j not in removed]

        X_fit_k = select_features(X_fit, keep)
        X_val_k = select_features(X_val, keep)
        X_ood_k = select_features(X_ood, keep)

        metrics = eval_target_model(X_fit_k, y_fit, X_val_k, y_val, X_ood_k, y_ood)
        valid_curve.append(metrics["valid_auc"])
        ood_curve.append(metrics["ood_auc"])

        row = {
            "removed_k": k,
            "kept_features": len(keep),
            "valid_auc": metrics["valid_auc"],
            "ood_auc": metrics["ood_auc"],
            "ood_logloss": metrics["ood_logloss"],
        }
        print({k2: round(v, 5) if isinstance(v, float) else v for k2, v in row.items()})

        if best is None or metrics["ood_auc"] > best["ood_auc"]:
            best = {
                "k": k,
                "keep": keep,
                **metrics,
            }

    print("Best robust configuration:")
    print({
        "remove_top_k": best["k"],
        "kept_features": best["keep"],
        "valid_auc": round(best["valid_auc"], 5),
        "ood_auc": round(best["ood_auc"], 5),
        "ood_logloss": round(best["ood_logloss"], 5),
    })

    maybe_plot(feature_scores, ks, valid_curve, ood_curve)
