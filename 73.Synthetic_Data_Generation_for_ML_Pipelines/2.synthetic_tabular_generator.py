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


def load_diabetes(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.reader(f))

    first = raw[0]
    if sum(_is_number(v) for v in first) >= max(1, int(0.8 * len(first))):
        header = DIABETES_SCHEMA[: len(first)]
        rows = raw
    else:
        header = first
        rows = raw[1:]

    out = []
    for r in rows:
        if len(r) != len(header):
            continue
        d = {k: v for k, v in zip(header, r)}
        out.append(d)
    return out


def class_stats(rows, target_col="Outcome"):
    num_cols = [c for c in rows[0].keys() if c != target_col]
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[int(float(r[target_col]))].append(r)

    stats = {}
    for cls, rr in by_cls.items():
        stats[cls] = {}
        for c in num_cols:
            vals = [float(x[c]) for x in rr]
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            stats[cls][c] = (mu, math.sqrt(var) + 1e-6)
    class_prior = {cls: len(rr) / len(rows) for cls, rr in by_cls.items()}
    return stats, class_prior, num_cols


def sample_synthetic(n: int, stats, class_prior, num_cols, seed=42):
    random.seed(seed)
    classes = sorted(class_prior.keys())
    cum = []
    s = 0.0
    for c in classes:
        s += class_prior[c]
        cum.append((s, c))

    def sample_class():
        r = random.random()
        for p, c in cum:
            if r <= p:
                return c
        return classes[-1]

    syn = []
    for _ in range(n):
        cls = sample_class()
        row = {"Outcome": cls}
        for c in num_cols:
            mu, sd = stats[cls][c]
            v = random.gauss(mu, sd)
            row[c] = max(0.0, v)
        syn.append(row)
    return syn


def simple_logistic_fit(X: List[List[float]], y: List[int], lr=0.002, epochs=400):
    d = len(X[0])
    w = [0.0] * (d + 1)
    for _ in range(epochs):
        idx = list(range(len(X)))
        random.shuffle(idx)
        for i in idx:
            z = w[0] + sum(w[j + 1] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-35, min(35, z))))
            err = y[i] - p
            w[0] += lr * err
            for j in range(d):
                w[j + 1] += lr * err * X[i][j]
    return w


def predict_prob(X, w):
    out = []
    d = len(X[0])
    for x in X:
        z = w[0] + sum(w[j + 1] * x[j] for j in range(d))
        out.append(1.0 / (1.0 + math.exp(-max(-35, min(35, z)))))
    return out


def auc(y_true, y_prob):
    pos = [p for p, y in zip(y_prob, y_true) if y == 1]
    neg = [p for p, y in zip(y_prob, y_true) if y == 0]
    wins = 0.0
    total = len(pos) * len(neg)
    for pp in pos:
        for pn in neg:
            if pp > pn:
                wins += 1
            elif pp == pn:
                wins += 0.5
    return wins / max(1, total)


def rows_to_xy(rows, cols):
    X = [[float(r[c]) for c in cols] for r in rows]
    y = [int(float(r["Outcome"])) for r in rows]
    return X, y


def nn_overlap_risk(real_rows, syn_rows, cols, threshold=0.15):
    # simple nearest-neighbor overlap risk in normalized space
    real = [[float(r[c]) for c in cols] for r in real_rows]
    syn = [[float(r[c]) for c in cols] for r in syn_rows]

    # normalize by real std
    d = len(cols)
    means = [sum(x[j] for x in real) / len(real) for j in range(d)]
    stds = []
    for j in range(d):
        v = sum((x[j] - means[j]) ** 2 for x in real) / len(real)
        stds.append(math.sqrt(v) + 1e-6)

    def dist(a, b):
        return math.sqrt(sum(((a[j] - b[j]) / stds[j]) ** 2 for j in range(d)))

    close = 0
    for s in syn:
        mn = min(dist(s, r) for r in real)
        if mn < threshold:
            close += 1
    return close / len(syn)


def main():
    rows = load_diabetes("Y.Kaggle_Data/diabetes.csv")
    random.shuffle(rows)

    split = int(0.7 * len(rows))
    real_train = rows[:split]
    real_test = rows[split:]

    stats, prior, cols = class_stats(real_train)
    syn_train = sample_synthetic(n=len(real_train), stats=stats, class_prior=prior, num_cols=cols, seed=33)

    Xr, yr = rows_to_xy(real_train, cols)
    Xt, yt = rows_to_xy(real_test, cols)
    Xs, ys = rows_to_xy(syn_train, cols)

    w_real = simple_logistic_fit(Xr, yr)
    w_syn = simple_logistic_fit(Xs, ys)

    auc_real = auc(yt, predict_prob(Xt, w_real))
    auc_syn = auc(yt, predict_prob(Xt, w_syn))

    risk = nn_overlap_risk(real_train, syn_train, cols, threshold=0.2)

    print("Train-on-real -> test-on-real AUC:", round(auc_real, 4))
    print("Train-on-syn  -> test-on-real AUC:", round(auc_syn, 4))
    print("Synthetic NN overlap risk       :", round(risk, 4))


if __name__ == "__main__":
    main()
