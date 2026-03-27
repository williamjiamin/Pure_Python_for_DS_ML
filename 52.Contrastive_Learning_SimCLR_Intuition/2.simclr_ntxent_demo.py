from __future__ import annotations

import math
import random
from typing import List, Tuple


Vector = List[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: Vector, b: Vector) -> float:
    return dot(a, b) / max(1e-12, norm(a) * norm(b))


def normalize(v: Vector) -> Vector:
    n = norm(v)
    return [x / max(1e-12, n) for x in v]


def augment(x: Vector, noise: float = 0.08) -> Vector:
    return [v + random.uniform(-noise, noise) for v in x]


def make_data(n: int = 30, d: int = 4, seed: int = 42) -> List[Vector]:
    random.seed(seed)
    centers = [[-1.0, -1.0, 0.5, 0.3], [1.0, 0.8, -0.4, -0.2], [0.0, 1.2, 0.6, -0.8]]
    out = []
    for i in range(n):
        c = centers[i % len(centers)]
        out.append([c[j] + random.uniform(-0.25, 0.25) for j in range(d)])
    return out


def linear_forward(X: List[Vector], W: List[List[float]]) -> List[Vector]:
    Z = []
    for x in X:
        z = [sum(x[j] * W[j][k] for j in range(len(x))) for k in range(len(W[0]))]
        Z.append(normalize(z))
    return Z


def ntxent_loss(Z: List[Vector], tau: float = 0.2) -> float:
    # Z size should be 2N, positives are pairs (2i, 2i+1)
    n2 = len(Z)
    assert n2 % 2 == 0

    def pair_loss(i: int, j: int) -> float:
        num = math.exp(cosine(Z[i], Z[j]) / tau)
        den = 0.0
        for k in range(n2):
            if k == i:
                continue
            den += math.exp(cosine(Z[i], Z[k]) / tau)
        return -math.log(num / max(1e-12, den))

    losses = []
    for p in range(0, n2, 2):
        losses.append(pair_loss(p, p + 1))
        losses.append(pair_loss(p + 1, p))

    return sum(losses) / len(losses)


def finite_diff_grad_W(X2: List[Vector], W: List[List[float]], tau: float = 0.2, h: float = 1e-4):
    d_in, d_out = len(W), len(W[0])
    grad = [[0.0 for _ in range(d_out)] for _ in range(d_in)]

    base = ntxent_loss(linear_forward(X2, W), tau=tau)
    for i in range(d_in):
        for j in range(d_out):
            W2 = [row[:] for row in W]
            W2[i][j] += h
            v2 = ntxent_loss(linear_forward(X2, W2), tau=tau)
            grad[i][j] = (v2 - base) / h
    return grad, base


def maybe_plot(losses: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Loss trajectory:", [round(v, 5) for v in losses[::max(1, len(losses)//10)]])
        return

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("NT-Xent Loss During Training")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(4)
    X = make_data(n=24, d=4, seed=7)

    # build 2N augmented views
    X2 = []
    for x in X:
        X2.append(augment(x, noise=0.1))
        X2.append(augment(x, noise=0.1))

    d_in, d_out = 4, 3
    W = [[random.uniform(-0.2, 0.2) for _ in range(d_out)] for _ in range(d_in)]

    lr = 0.6
    steps = 70
    losses = []
    for _ in range(steps):
        grad, loss = finite_diff_grad_W(X2, W, tau=0.25)
        losses.append(loss)
        for i in range(d_in):
            for j in range(d_out):
                W[i][j] -= lr * grad[i][j]

    print("Initial loss approx:", round(losses[0], 6))
    print("Final loss approx  :", round(losses[-1], 6))
    maybe_plot(losses)
