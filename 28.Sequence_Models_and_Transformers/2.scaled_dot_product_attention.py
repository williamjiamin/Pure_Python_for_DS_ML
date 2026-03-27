from __future__ import annotations

import math
from typing import List


Vector = List[float]
Matrix = List[Vector]


def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    out = []
    for row in a:
        out_row = []
        for col in bt:
            out_row.append(sum(x * y for x, y in zip(row, col)))
        out.append(out_row)
    return out


def softmax_row(row: Vector) -> Vector:
    m = max(row)
    ex = [math.exp(x - m) for x in row]
    z = sum(ex)
    return [v / z for v in ex]


def attention(Q: Matrix, K: Matrix, V: Matrix) -> Matrix:
    dk = len(K[0])
    scores = matmul(Q, transpose(K))
    scores = [[x / math.sqrt(dk) for x in row] for row in scores]
    weights = [softmax_row(row) for row in scores]
    return matmul(weights, V)


if __name__ == "__main__":
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0, 2.0], [0.0, 1.0], [3.0, 1.0]]

    out = attention(Q, K, V)
    print("Attention output:")
    for row in out:
        print([round(v, 4) for v in row])
