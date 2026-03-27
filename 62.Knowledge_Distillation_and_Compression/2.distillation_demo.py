from __future__ import annotations

import math
import random
from typing import List, Tuple


Matrix = List[List[float]]


def softmax(logits: List[float], T: float = 1.0) -> List[float]:
    scaled = [z / T for z in logits]
    m = max(scaled)
    ex = [math.exp(z - m) for z in scaled]
    s = sum(ex)
    return [v / s for v in ex]


def cross_entropy(probs: List[float], y: int) -> float:
    p = max(1e-12, min(1 - 1e-12, probs[y]))
    return -math.log(p)


def kl_div(p: List[float], q: List[float]) -> float:
    out = 0.0
    for pi, qi in zip(p, q):
        pi = max(1e-12, pi)
        qi = max(1e-12, qi)
        out += pi * math.log(pi / qi)
    return out


def linear_logits(x: List[float], W: Matrix, b: List[float]) -> List[float]:
    out = []
    for c in range(len(b)):
        out.append(sum(x[i] * W[i][c] for i in range(len(x))) + b[c])
    return out


def make_data(n=1200, d=6, n_cls=3, seed=42):
    random.seed(seed)
    centers = [
        [-0.6, -0.5, 0.2, 0.1, -0.2, 0.3],
        [0.5, 0.6, -0.2, 0.2, 0.4, -0.3],
        [0.1, -0.2, 0.7, -0.6, 0.2, 0.5],
    ]

    X, y = [], []
    for i in range(n):
        c = i % n_cls
        x = [centers[c][j] + random.uniform(-0.9, 0.9) for j in range(d)]
        X.append(x)
        # mild label noise to avoid trivial separability
        if random.random() < 0.08:
            y.append(random.randrange(n_cls))
        else:
            y.append(c)
    return X, y


def train_teacher(X, y, d=6, n_cls=3, epochs=180, lr=0.05, seed=1):
    random.seed(seed)
    W = [[random.uniform(-0.2, 0.2) for _ in range(n_cls)] for _ in range(d)]
    b = [0.0] * n_cls

    for _ in range(epochs):
        idx = list(range(len(X)))
        random.shuffle(idx)
        for i in idx:
            logits = linear_logits(X[i], W, b)
            p = softmax(logits)
            for c in range(n_cls):
                grad = (1.0 if y[i] == c else 0.0) - p[c]
                b[c] += lr * grad
                for f in range(d):
                    W[f][c] += lr * grad * X[i][f]
    return W, b


def train_student(
    X,
    y,
    teacher_W,
    teacher_b,
    d=6,
    n_cls=3,
    epochs=120,
    lr=0.035,
    T=2.0,
    alpha=0.7,
    seed=2,
    use_distill=True,
):
    random.seed(seed)
    # smaller student: low-rank-like projection to fewer hidden dims
    h = 1
    U = [[random.uniform(-0.2, 0.2) for _ in range(h)] for _ in range(d)]
    V = [[random.uniform(-0.2, 0.2) for _ in range(n_cls)] for _ in range(h)]
    b = [0.0] * n_cls

    for _ in range(epochs):
        idx = list(range(len(X)))
        random.shuffle(idx)
        for i in idx:
            x = X[i]

            # forward student logits: x U V
            hvec = [sum(x[f] * U[f][j] for f in range(d)) for j in range(h)]
            s_logits = [sum(hvec[j] * V[j][c] for j in range(h)) + b[c] for c in range(n_cls)]
            s_prob = softmax(s_logits)

            if use_distill:
                t_logits = linear_logits(x, teacher_W, teacher_b)
                t_soft = softmax(t_logits, T=T)
                s_soft = softmax(s_logits, T=T)
            else:
                t_soft = [0.0] * n_cls
                s_soft = [0.0] * n_cls

            # gradient on logits (combined hard + soft)
            g_logits = [0.0] * n_cls
            for c in range(n_cls):
                hard = (1.0 if y[i] == c else 0.0) - s_prob[c]
                if use_distill:
                    soft = (t_soft[c] - s_soft[c]) * (T * T)
                else:
                    soft = 0.0
                g_logits[c] = (1 - alpha) * hard + alpha * soft

            # update V, b, U
            for c in range(n_cls):
                b[c] += lr * g_logits[c]
                for j in range(h):
                    V[j][c] += lr * g_logits[c] * hvec[j]

            for f in range(d):
                for j in range(h):
                    back = sum(g_logits[c] * V[j][c] for c in range(n_cls))
                    U[f][j] += lr * back * x[f]

    return U, V, b


def predict_student(X, U, V, b):
    out = []
    h = len(V)
    for x in X:
        hvec = [sum(x[f] * U[f][j] for f in range(len(x))) for j in range(h)]
        logits = [sum(hvec[j] * V[j][c] for j in range(h)) + b[c] for c in range(len(b))]
        p = softmax(logits)
        out.append(max(range(len(p)), key=lambda c: p[c]))
    return out


def predict_teacher(X, W, b):
    out = []
    for x in X:
        logits = linear_logits(x, W, b)
        p = softmax(logits)
        out.append(max(range(len(p)), key=lambda c: p[c]))
    return out


def acc(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def maybe_plot(a_teacher, a_plain, a_distill):
    labels = ["Teacher", "Student plain", "Student distill"]
    vals = [a_teacher, a_plain, a_distill]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        for l, v in zip(labels, vals):
            print(f"{l}: {v:.4f}")
        return

    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.title("Distillation Accuracy Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = make_data(n=1500, d=6, n_cls=3, seed=13)
    split = 1100
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    tW, tb = train_teacher(Xtr, ytr, d=6, n_cls=3, epochs=160, lr=0.04, seed=4)

    U0, V0, b0 = train_student(Xtr, ytr, tW, tb, use_distill=False, alpha=0.0, T=2.0, epochs=60, lr=0.03, seed=6)
    U1, V1, b1 = train_student(Xtr, ytr, tW, tb, use_distill=True, alpha=0.05, T=1.2, epochs=110, lr=0.03, seed=6)

    a_teacher = acc(yte, predict_teacher(Xte, tW, tb))
    a_plain = acc(yte, predict_student(Xte, U0, V0, b0))
    a_distill = acc(yte, predict_student(Xte, U1, V1, b1))

    print("Teacher acc       :", round(a_teacher, 4))
    print("Student plain acc :", round(a_plain, 4))
    print("Student distill acc:", round(a_distill, 4))

    teacher_params = 6 * 3 + 3
    student_params = 6 * 1 + 1 * 3 + 3
    print("Teacher params:", teacher_params)
    print("Student params:", student_params)
    print("Compression ratio teacher/student:", round(teacher_params / student_params, 4))

    maybe_plot(a_teacher, a_plain, a_distill)
