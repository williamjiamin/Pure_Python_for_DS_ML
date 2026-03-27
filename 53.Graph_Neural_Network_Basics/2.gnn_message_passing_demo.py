from __future__ import annotations

import math
import random
from typing import Dict, List


Vector = List[float]


def relu(v: Vector) -> Vector:
    return [max(0.0, x) for x in v]


def matvec(W: List[List[float]], x: Vector) -> Vector:
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def mean_vec(vecs: List[Vector]) -> Vector:
    d = len(vecs[0])
    out = [0.0] * d
    for v in vecs:
        for i in range(d):
            out[i] += v[i]
    return [x / len(vecs) for x in out]


def cosine(a: Vector, b: Vector) -> float:
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def message_passing(
    X: Dict[int, Vector],
    adj: Dict[int, List[int]],
    W: List[List[float]],
    layers: int = 1,
) -> Dict[int, Vector]:
    H = {k: v[:] for k, v in X.items()}
    for _ in range(layers):
        H2 = {}
        for v in H:
            nbs = adj.get(v, []) + [v]
            agg = mean_vec([H[u] for u in nbs])
            H2[v] = relu(matvec(W, agg))
        H = H2
    return H


def simple_node_classifier(H: Dict[int, Vector], labels: Dict[int, int]) -> float:
    # nearest class prototype classifier
    cls0 = [H[i] for i, y in labels.items() if y == 0]
    cls1 = [H[i] for i, y in labels.items() if y == 1]

    p0 = mean_vec(cls0)
    p1 = mean_vec(cls1)

    correct = 0
    for i, y in labels.items():
        s0 = cosine(H[i], p0)
        s1 = cosine(H[i], p1)
        pred = 1 if s1 >= s0 else 0
        correct += int(pred == y)
    return correct / len(labels)


def maybe_plot_embeddings(H_before: Dict[int, Vector], H_after: Dict[int, Vector], labels: Dict[int, int]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Node embeddings (first 2 dims) before -> after:")
        for i in sorted(H_before.keys()):
            b = H_before[i][:2]
            a = H_after[i][:2]
            print(i, "label", labels[i], "before", [round(x, 3) for x in b], "after", [round(x, 3) for x in a])
        return

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    for i, v in H_before.items():
        c = "tab:blue" if labels[i] == 0 else "tab:orange"
        plt.scatter(v[0], v[1], c=c)
        plt.text(v[0], v[1], str(i), fontsize=8)
    plt.title("Before Message Passing")

    plt.subplot(1, 2, 2)
    for i, v in H_after.items():
        c = "tab:blue" if labels[i] == 0 else "tab:orange"
        plt.scatter(v[0], v[1], c=c)
        plt.text(v[0], v[1], str(i), fontsize=8)
    plt.title("After Message Passing")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(5)

    # toy graph with two communities
    adj = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [1, 2, 4],
        4: [3, 5, 6],
        5: [4, 6],
        6: [4, 5, 7],
        7: [6],
    }

    labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    X = {}
    for i in range(8):
        if labels[i] == 0:
            X[i] = [random.uniform(-1.2, -0.4), random.uniform(0.4, 1.2), random.uniform(-0.2, 0.2)]
        else:
            X[i] = [random.uniform(0.4, 1.2), random.uniform(-1.2, -0.4), random.uniform(-0.2, 0.2)]

    W = [
        [0.8, 0.2, 0.1],
        [0.2, 0.8, -0.1],
        [0.1, -0.1, 0.9],
    ]

    H1 = message_passing(X, adj, W, layers=1)
    H2 = message_passing(X, adj, W, layers=2)

    acc_before = simple_node_classifier(X, labels)
    acc_l1 = simple_node_classifier(H1, labels)
    acc_l2 = simple_node_classifier(H2, labels)

    print("Prototype accuracy before:", round(acc_before, 4))
    print("Prototype accuracy 1-layer:", round(acc_l1, 4))
    print("Prototype accuracy 2-layer:", round(acc_l2, 4))

    maybe_plot_embeddings(X, H1, labels)
