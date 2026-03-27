from __future__ import annotations

import math
import random
from typing import List, Tuple


Matrix = List[List[float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def softmax(row: List[float]) -> List[float]:
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    z = sum(ex)
    return [v / z for v in ex]


def layer_norm(X: Matrix, eps: float = 1e-5) -> Matrix:
    out = []
    for row in X:
        mu = sum(row) / len(row)
        var = sum((x - mu) ** 2 for x in row) / len(row)
        out.append([(x - mu) / math.sqrt(var + eps) for x in row])
    return out


def linear(X: Matrix, W: Matrix) -> Matrix:
    return matmul(X, W)


def split_heads(X: Matrix, n_heads: int) -> List[Matrix]:
    d_model = len(X[0])
    d_head = d_model // n_heads
    heads = []
    for h in range(n_heads):
        s, e = h * d_head, (h + 1) * d_head
        heads.append([row[s:e] for row in X])
    return heads


def combine_heads(heads: List[Matrix]) -> Matrix:
    seq_len = len(heads[0])
    out = []
    for i in range(seq_len):
        row = []
        for h in heads:
            row.extend(h[i])
        out.append(row)
    return out


def causal_mask(seq_len: int) -> Matrix:
    m = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(0.0 if j <= i else -1e9)
        m.append(row)
    return m


def attention(Q: Matrix, K: Matrix, V: Matrix, mask: Matrix) -> Tuple[Matrix, Matrix]:
    d = len(Q[0])
    scores = matmul(Q, transpose(K))
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] = scores[i][j] / math.sqrt(d) + mask[i][j]
    weights = [softmax(row) for row in scores]
    out = matmul(weights, V)
    return out, weights


def mean_entropy(weights: Matrix) -> float:
    e = 0.0
    n = 0
    for row in weights:
        for p in row:
            if p > 1e-12:
                e += -p * math.log(p)
        n += 1
    return e / max(1, n)


def ascii_heatmap(m: Matrix, decimals: int = 2) -> None:
    for row in m:
        print(" ".join(f"{v:.{decimals}f}" for v in row))


def maybe_plot_heatmap(weights: Matrix, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; ASCII heatmap:")
        ascii_heatmap(weights, decimals=2)
        return

    plt.figure(figsize=(5, 4))
    plt.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(7)
    seq_len = 6
    d_model = 8
    n_heads = 2

    X = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_len)]
    Xn = layer_norm(X)

    Wq = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]
    Wk = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]
    Wv = [[random.uniform(-0.5, 0.5) for _ in range(d_model)] for _ in range(d_model)]

    Q = linear(Xn, Wq)
    K = linear(Xn, Wk)
    V = linear(Xn, Wv)

    q_heads = split_heads(Q, n_heads)
    k_heads = split_heads(K, n_heads)
    v_heads = split_heads(V, n_heads)

    mask = causal_mask(seq_len)
    head_outputs = []
    for h in range(n_heads):
        out, w = attention(q_heads[h], k_heads[h], v_heads[h], mask)
        head_outputs.append(out)
        print(f"Head {h} mean attention entropy:", round(mean_entropy(w), 4))
        maybe_plot_heatmap(w, title=f"Head {h} Causal Attention")

    combined = combine_heads(head_outputs)
    print("Combined output shape:", len(combined), "x", len(combined[0]))
