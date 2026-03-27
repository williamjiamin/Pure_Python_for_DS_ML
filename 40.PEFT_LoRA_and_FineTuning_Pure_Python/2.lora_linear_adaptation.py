from __future__ import annotations

import random
from typing import List, Tuple


Matrix = List[List[float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)]


def mse(y_true: Matrix, y_pred: Matrix) -> float:
    n = len(y_true) * len(y_true[0])
    return sum((a - b) ** 2 for r1, r2 in zip(y_true, y_pred) for a, b in zip(r1, r2)) / n


def zeros(r: int, c: int) -> Matrix:
    return [[0.0 for _ in range(c)] for _ in range(r)]


def random_matrix(r: int, c: int, scale: float = 0.1) -> Matrix:
    return [[random.uniform(-scale, scale) for _ in range(c)] for _ in range(r)]


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def scalar_mul(m: Matrix, s: float) -> Matrix:
    return [[v * s for v in row] for row in m]


def sub(a: Matrix, b: Matrix) -> Matrix:
    return [[x - y for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)]


def mean_abs(m: Matrix) -> float:
    n = len(m) * len(m[0])
    return sum(abs(v) for row in m for v in row) / n


def maybe_plot_curve(values: List[float], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(title)
        for i, v in enumerate(values[::max(1, len(values)//20)]):
            print(f"step={i:03d} loss={v:.6f}")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    n_samples = 120
    in_dim = 6
    out_dim = 5
    rank = 2
    lr = 0.05
    steps = 500
    alpha = 1.0

    # Frozen base weights (pretend pretrained)
    W_base = random_matrix(in_dim, out_dim, scale=0.4)

    # Unknown target transform we want to adapt toward.
    delta_true = random_matrix(in_dim, out_dim, scale=0.15)
    W_target = add(W_base, delta_true)

    X = [random_matrix(1, in_dim, scale=1.0)[0] for _ in range(n_samples)]
    X_m = [x[:] for x in X]
    Y = matmul(X_m, W_target)

    # LoRA params: BA (in_dim x out_dim through in_dim x rank and rank x out_dim)
    B = random_matrix(in_dim, rank, scale=0.01)
    A = random_matrix(rank, out_dim, scale=0.01)

    losses = []
    for _ in range(steps):
        BA = matmul(B, A)
        W_eff = add(W_base, scalar_mul(BA, alpha))
        Y_hat = matmul(X_m, W_eff)
        err = sub(Y_hat, Y)
        loss = mse(Y, Y_hat)
        losses.append(loss)

        # dLoss/dW_eff = X^T * err / n
        Xt = transpose(X_m)
        dW = scalar_mul(matmul(Xt, err), 2.0 / n_samples)

        # dW/dB = A^T, dW/dA = B^T by chain for BA
        At = transpose(A)
        Bt = transpose(B)
        dB = matmul(dW, At)
        dA = matmul(Bt, dW)

        B = sub(B, scalar_mul(dB, lr))
        A = sub(A, scalar_mul(dA, lr))

    BA_final = matmul(B, A)
    print("Final loss:", round(losses[-1], 8))
    print("Mean abs true delta:", round(mean_abs(delta_true), 6))
    print("Mean abs learned BA:", round(mean_abs(BA_final), 6))

    full_params = in_dim * out_dim
    lora_params = in_dim * rank + rank * out_dim
    print("Trainable params full:", full_params)
    print("Trainable params LoRA:", lora_params)
    print("Reduction factor:", round(full_params / lora_params, 3))

    maybe_plot_curve(losses, title="LoRA Training Loss")
