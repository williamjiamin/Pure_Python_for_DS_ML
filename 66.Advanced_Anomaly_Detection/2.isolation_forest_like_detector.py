from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


Point = List[float]


@dataclass
class Node:
    feat: Optional[int] = None
    thr: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    size: int = 0


def c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def build_tree(points: List[Point], depth: int, max_depth: int) -> Node:
    n = len(points)
    node = Node(size=n)
    if n <= 1 or depth >= max_depth:
        return node

    d = len(points[0])
    feat = random.randrange(d)
    vals = [p[feat] for p in points]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return node

    thr = random.uniform(lo, hi)
    left = [p for p in points if p[feat] <= thr]
    right = [p for p in points if p[feat] > thr]
    if not left or not right:
        return node

    node.feat = feat
    node.thr = thr
    node.left = build_tree(left, depth + 1, max_depth)
    node.right = build_tree(right, depth + 1, max_depth)
    return node


def path_length(x: Point, node: Node, depth: int = 0) -> float:
    if node.feat is None or node.left is None or node.right is None:
        return depth + c_factor(node.size)
    if x[node.feat] <= node.thr:
        return path_length(x, node.left, depth + 1)
    return path_length(x, node.right, depth + 1)


class IsolationForestLike:
    def __init__(self, n_trees: int = 80, sample_size: int = 128, seed: int = 42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.trees: List[Node] = []

    def fit(self, X: List[Point]):
        random.seed(self.seed)
        self.trees = []
        max_depth = int(math.ceil(math.log2(self.sample_size)))

        for _ in range(self.n_trees):
            sample = random.sample(X, k=min(self.sample_size, len(X)))
            tree = build_tree(sample, 0, max_depth)
            self.trees.append(tree)

    def score_samples(self, X: List[Point]) -> List[float]:
        cn = c_factor(min(self.sample_size, len(X)))
        out = []
        for x in X:
            h = sum(path_length(x, t) for t in self.trees) / len(self.trees)
            score = 2 ** (-h / max(1e-12, cn))
            out.append(score)
        return out


def zscore_baseline(X: List[Point]) -> List[float]:
    d = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(d)]
    stds = []
    for j in range(d):
        v = sum((row[j] - means[j]) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(v) + 1e-12)

    scores = []
    for row in X:
        z = sum(abs((row[j] - means[j]) / stds[j]) for j in range(d)) / d
        scores.append(z)
    return scores


def generate_data(seed: int = 7) -> Tuple[List[Point], List[int]]:
    random.seed(seed)
    X, y = [], []

    # normal cluster
    for _ in range(380):
        x1 = random.gauss(0, 1.0)
        x2 = random.gauss(0, 1.1)
        X.append([x1, x2])
        y.append(0)

    # secondary dense region
    for _ in range(90):
        x1 = random.gauss(3.0, 0.6)
        x2 = random.gauss(-2.0, 0.7)
        X.append([x1, x2])
        y.append(0)

    # anomalies
    for _ in range(30):
        x1 = random.uniform(-7, 7)
        x2 = random.uniform(-7, 7)
        if abs(x1) < 4 and abs(x2) < 4:
            x1 += random.choice([-4.5, 4.5])
        X.append([x1, x2])
        y.append(1)

    return X, y


def precision_at_k(scores: List[float], labels: List[int], k: int) -> float:
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum(labels[i] for i in idx) / k


def maybe_plot(X: List[Point], labels: List[int], scores: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        print("Top-10 anomaly candidates:")
        for i in top:
            print(i, [round(v, 3) for v in X[i]], "score", round(scores[i], 4), "label", labels[i])
        return

    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(xs, ys, c=scores, cmap="inferno", s=14)
    plt.colorbar(sc, label="anomaly score")
    plt.title("Isolation-like Anomaly Scores")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = generate_data(seed=10)

    model = IsolationForestLike(n_trees=90, sample_size=128, seed=4)
    model.fit(X)
    iso_scores = model.score_samples(X)
    z_scores = zscore_baseline(X)

    for k in [10, 20, 30]:
        p_iso = precision_at_k(iso_scores, y, k)
        p_z = precision_at_k(z_scores, y, k)
        print(f"Precision@{k} isolation={p_iso:.3f} zscore={p_z:.3f}")

    maybe_plot(X, y, iso_scores)
