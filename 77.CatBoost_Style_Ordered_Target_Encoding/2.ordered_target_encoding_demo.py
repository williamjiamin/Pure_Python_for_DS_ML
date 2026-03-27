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
    rank_sum_pos = 0.0

    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1

        avg_rank = (rank + rank + (j - i)) / 2.0
        pos_here = sum(lbl for _, lbl in pairs[i : j + 1])
        rank_sum_pos += avg_rank * pos_here

        rank += (j - i + 1)
        i = j + 1

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def weighted_choice(cumulative: List[float], rng: random.Random) -> int:
    r = rng.random() * cumulative[-1]
    lo, hi = 0, len(cumulative) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cumulative[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    return lo


def simulate_events(n: int = 7000, n_cat: int = 550, seed: int = 42):
    rng = random.Random(seed)

    cat_weights = [1.0 / ((i + 1) ** 1.08) for i in range(n_cat)]
    cumulative = []
    run = 0.0
    for w in cat_weights:
        run += w
        cumulative.append(run)

    cat_effect = [rng.gauss(0.0, 0.7) for _ in range(n_cat)]

    cats: List[int] = []
    amount: List[float] = []
    night: List[int] = []
    y: List[int] = []

    for t in range(n):
        c = weighted_choice(cumulative, rng)
        a = max(1.0, rng.lognormvariate(3.2, 0.6))
        is_night = 1 if (t % 24 >= 20 or t % 24 <= 5) else 0

        score = (
            -2.2
            + 0.012 * min(a, 200.0)
            + 0.55 * is_night
            + cat_effect[c]
            + rng.gauss(0.0, 0.35)
        )
        p = sigmoid(score)
        label = 1 if rng.random() < p else 0

        cats.append(c)
        amount.append(a)
        night.append(is_night)
        y.append(label)

    return cats, amount, night, y


def train_valid_split(cats, amount, night, y, train_ratio=0.7):
    cut = int(len(y) * train_ratio)
    return (
        cats[:cut],
        amount[:cut],
        night[:cut],
        y[:cut],
        cats[cut:],
        amount[cut:],
        night[cut:],
        y[cut:],
    )


def leaky_target_encoding(cats: List[int], y: List[int], alpha: float = 20.0):
    global_mean = sum(y) / len(y)
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for c, yi in zip(cats, y):
        sums[c] = sums.get(c, 0.0) + yi
        counts[c] = counts.get(c, 0) + 1

    encoded = []
    for c in cats:
        s = sums[c]
        k = counts[c]
        encoded.append((s + alpha * global_mean) / (k + alpha))

    return encoded, sums, counts, global_mean


def ordered_target_encoding(cats: List[int], y: List[int], alpha: float = 20.0):
    global_mean = sum(y) / len(y)
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    encoded = []
    for c, yi in zip(cats, y):
        s = sums.get(c, 0.0)
        k = counts.get(c, 0)
        encoded.append((s + alpha * global_mean) / (k + alpha))

        sums[c] = s + yi
        counts[c] = k + 1

    return encoded, sums, counts, global_mean


def apply_encoding(cats: List[int], sums: Dict[int, float], counts: Dict[int, int], global_mean: float, alpha: float = 20.0):
    out = []
    for c in cats:
        s = sums.get(c, 0.0)
        k = counts.get(c, 0)
        out.append((s + alpha * global_mean) / (k + alpha))
    return out


def standardize_fit(X: List[List[float]]):
    d = len(X[0])
    mean = [0.0] * d
    std = [0.0] * d

    for j in range(d):
        col = [row[j] for row in X]
        m = sum(col) / len(col)
        var = sum((v - m) ** 2 for v in col) / len(col)
        mean[j] = m
        std[j] = math.sqrt(var) + 1e-12

    Xs = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    return Xs, mean, std


def standardize_apply(X: List[List[float]], mean: List[float], std: List[float]):
    d = len(mean)
    return [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]


def train_logreg(
    X: List[List[float]],
    y: List[int],
    epochs: int = 220,
    lr: float = 0.04,
    l2: float = 2e-4,
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

        grad_w = [0.0] * d
        grad_b = 0.0

        for i in idx:
            z = b
            for j in range(d):
                z += w[j] * X[i][j]
            p = sigmoid(z)
            err = p - y[i]

            for j in range(d):
                grad_w[j] += err * X[i][j]
            grad_b += err

        inv_n = 1.0 / n
        for j in range(d):
            grad = grad_w[j] * inv_n + l2 * w[j]
            w[j] -= lr * grad
        b -= lr * grad_b * inv_n

    return w, b


def predict_proba(X: List[List[float]], w: List[float], b: float):
    out = []
    for row in X:
        z = b
        for j in range(len(w)):
            z += w[j] * row[j]
        out.append(sigmoid(z))
    return out


def build_matrix(amount: List[float], night: List[int], enc: List[float]):
    return [[math.log1p(a), float(n), e] for a, n, e in zip(amount, night, enc)]


def evaluate_pipeline(
    X_tr_raw: List[List[float]],
    y_tr: List[int],
    X_va_raw: List[List[float]],
    y_va: List[int],
):
    X_tr, mean, std = standardize_fit(X_tr_raw)
    X_va = standardize_apply(X_va_raw, mean, std)

    w, b = train_logreg(X_tr, y_tr, epochs=230, lr=0.045, l2=2e-4, seed=9)

    p_tr = predict_proba(X_tr, w, b)
    p_va = predict_proba(X_va, w, b)

    return {
        "train_auc": auc_score(y_tr, p_tr),
        "valid_auc": auc_score(y_va, p_va),
        "train_logloss": logloss(y_tr, p_tr),
        "valid_logloss": logloss(y_va, p_va),
        "w": w,
    }


def maybe_plot(leaky_metrics, ordered_metrics, leaky_train_enc, ordered_train_enc, y_train):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not installed. Metrics only.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = ["Leaky TE", "Ordered TE"]
    train_auc = [leaky_metrics["train_auc"], ordered_metrics["train_auc"]]
    valid_auc = [leaky_metrics["valid_auc"], ordered_metrics["valid_auc"]]

    x = [0, 1]
    width = 0.35
    axs[0].bar([v - width / 2 for v in x], train_auc, width=width, label="Train AUC", color="#93c5fd")
    axs[0].bar([v + width / 2 for v in x], valid_auc, width=width, label="Valid AUC", color="#1d4ed8")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_ylim(0.45, 1.0)
    axs[0].set_title("Generalization Gap from Encoding Strategy")
    axs[0].legend()

    ordered_pos = [e for e, yi in zip(ordered_train_enc, y_train) if yi == 1]
    ordered_neg = [e for e, yi in zip(ordered_train_enc, y_train) if yi == 0]
    axs[1].hist(ordered_neg, bins=30, alpha=0.6, label="y=0", color="#9ca3af", density=True)
    axs[1].hist(ordered_pos, bins=30, alpha=0.6, label="y=1", color="#1d4ed8", density=True)
    axs[1].set_title("Ordered Encoding Distribution by Class")
    axs[1].set_xlabel("Encoded category value")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def rare_category_leakage_gap(cats: List[int], leaky_enc: List[float], ordered_enc: List[float]):
    counts: Dict[int, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1

    gaps = [
        abs(a - b)
        for c, a, b in zip(cats, leaky_enc, ordered_enc)
        if counts[c] <= 3
    ]
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)


if __name__ == "__main__":
    cats, amount, night, y = simulate_events(n=7000, n_cat=550, seed=11)
    (
        cats_tr,
        amount_tr,
        night_tr,
        y_tr,
        cats_va,
        amount_va,
        night_va,
        y_va,
    ) = train_valid_split(cats, amount, night, y, train_ratio=0.7)

    leaky_enc_tr, leaky_sums, leaky_counts, global_mean = leaky_target_encoding(cats_tr, y_tr, alpha=18.0)
    leaky_enc_va = apply_encoding(cats_va, leaky_sums, leaky_counts, global_mean, alpha=18.0)

    ordered_enc_tr, ord_sums, ord_counts, ord_global = ordered_target_encoding(cats_tr, y_tr, alpha=18.0)
    ordered_enc_va = apply_encoding(cats_va, ord_sums, ord_counts, ord_global, alpha=18.0)

    X_leaky_tr = build_matrix(amount_tr, night_tr, leaky_enc_tr)
    X_leaky_va = build_matrix(amount_va, night_va, leaky_enc_va)

    X_ordered_tr = build_matrix(amount_tr, night_tr, ordered_enc_tr)
    X_ordered_va = build_matrix(amount_va, night_va, ordered_enc_va)

    leaky_metrics = evaluate_pipeline(X_leaky_tr, y_tr, X_leaky_va, y_va)
    ordered_metrics = evaluate_pipeline(X_ordered_tr, y_tr, X_ordered_va, y_va)

    print("Leaky target encoding metrics:")
    print({k: round(v, 5) for k, v in leaky_metrics.items() if k != "w"})

    print("Ordered target encoding metrics:")
    print({k: round(v, 5) for k, v in ordered_metrics.items() if k != "w"})

    gap = rare_category_leakage_gap(cats_tr, leaky_enc_tr, ordered_enc_tr)
    print("Average leakage gap on rare categories (<=3 rows):", round(gap, 5))

    maybe_plot(leaky_metrics, ordered_metrics, leaky_enc_tr, ordered_enc_tr, y_tr)
