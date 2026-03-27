from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple


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


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    first = rows[0]
    if sum(_is_number(x) for x in first) >= max(1, int(0.8 * len(first))):
        header = DIABETES_SCHEMA[: len(first)]
        data_rows = rows
    else:
        header = first
        data_rows = rows[1:]

    out = []
    for r in data_rows:
        if len(r) != len(header):
            continue
        out.append({k: v for k, v in zip(header, r)})
    return out


def kfold_indices(n: int, k: int = 5, seed: int = 42) -> List[List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    return folds


def logistic_fit(X: List[List[float]], y: List[int], lr: float = 0.03, epochs: int = 250) -> List[float]:
    n, d = len(X), len(X[0])
    w = [0.0] * (d + 1)

    for _ in range(epochs):
        for i in range(n):
            z = w[0] + sum(w[j + 1] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))
            err = y[i] - p
            w[0] += lr * err
            for j in range(d):
                w[j + 1] += lr * err * X[i][j]
    return w


def logistic_predict_prob(X: List[List[float]], w: List[float]) -> List[float]:
    out = []
    d = len(X[0])
    for row in X:
        z = w[0] + sum(w[j + 1] * row[j] for j in range(d))
        p = 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))
        out.append(p)
    return out


def auc_score(y_true: List[int], y_prob: List[float]) -> float:
    pos = [(p, y) for p, y in zip(y_prob, y_true) if y == 1]
    neg = [(p, y) for p, y in zip(y_prob, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = len(pos) * len(neg)
    for pp, _ in pos:
        for pn, _ in neg:
            if pp > pn:
                wins += 1
            elif pp == pn:
                wins += 0.5
    return wins / total


def fold_safe_target_encoding(cat_vals: List[str], y: List[int], folds: List[List[int]]) -> List[float]:
    n = len(cat_vals)
    encoded = [0.0] * n
    global_mean = sum(y) / len(y)

    for valid_idx in folds:
        valid_set = set(valid_idx)
        sum_cnt = defaultdict(lambda: [0.0, 0])

        for i in range(n):
            if i in valid_set:
                continue
            c = cat_vals[i]
            sum_cnt[c][0] += y[i]
            sum_cnt[c][1] += 1

        for i in valid_idx:
            c = cat_vals[i]
            if sum_cnt[c][1] == 0:
                encoded[i] = global_mean
            else:
                encoded[i] = sum_cnt[c][0] / sum_cnt[c][1]

    return encoded


def leaky_target_encoding(cat_vals: List[str], y: List[int]) -> List[float]:
    sum_cnt = defaultdict(lambda: [0.0, 0])
    for c, t in zip(cat_vals, y):
        sum_cnt[c][0] += t
        sum_cnt[c][1] += 1
    return [sum_cnt[c][0] / sum_cnt[c][1] for c in cat_vals]


def maybe_plot_fold_scores(scores_safe: List[float], scores_leaky: List[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Fold scores safe:", [round(s, 4) for s in scores_safe])
        print("Fold scores leaky:", [round(s, 4) for s in scores_leaky])
        return

    xs = list(range(1, len(scores_safe) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, scores_safe, marker="o", label="fold-safe")
    plt.plot(xs, scores_leaky, marker="o", label="leaky")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.title("Fold-safe vs Leaky Target Encoding")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    rows = load_rows("Y.Kaggle_Data/diabetes.csv")
    y = [int(float(r["Outcome"])) for r in rows]

    # Create a synthetic categorical feature from glucose ranges (for demo).
    cat_vals = []
    for r in rows:
        g = float(r["Glucose"])
        if g < 100:
            cat_vals.append("low")
        elif g < 140:
            cat_vals.append("mid")
        else:
            cat_vals.append("high")

    base_features = [[
        float(r["BMI"]),
        float(r["Age"]),
        float(r["Pregnancies"]),
    ] for r in rows]

    folds = kfold_indices(len(rows), k=5, seed=42)

    te_safe = fold_safe_target_encoding(cat_vals, y, folds)
    te_leaky = leaky_target_encoding(cat_vals, y)

    scores_safe, scores_leaky = [], []
    for valid in folds:
        valid_set = set(valid)
        train = [i for i in range(len(rows)) if i not in valid_set]

        X_train_safe = [base_features[i] + [te_safe[i]] for i in train]
        X_valid_safe = [base_features[i] + [te_safe[i]] for i in valid]

        X_train_leaky = [base_features[i] + [te_leaky[i]] for i in train]
        X_valid_leaky = [base_features[i] + [te_leaky[i]] for i in valid]

        y_train = [y[i] for i in train]
        y_valid = [y[i] for i in valid]

        w_safe = logistic_fit(X_train_safe, y_train)
        w_leaky = logistic_fit(X_train_leaky, y_train)

        p_safe = logistic_predict_prob(X_valid_safe, w_safe)
        p_leaky = logistic_predict_prob(X_valid_leaky, w_leaky)

        scores_safe.append(auc_score(y_valid, p_safe))
        scores_leaky.append(auc_score(y_valid, p_leaky))

    print("Mean AUC fold-safe:", round(sum(scores_safe) / len(scores_safe), 5))
    print("Mean AUC leaky    :", round(sum(scores_leaky) / len(scores_leaky), 5))

    maybe_plot_fold_scores(scores_safe, scores_leaky)


if __name__ == "__main__":
    main()
