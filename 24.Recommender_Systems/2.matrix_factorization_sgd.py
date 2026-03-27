from __future__ import annotations

import random
from typing import Dict, List, Tuple


class MatrixFactorization:
    def __init__(self, n_users: int, n_items: int, k: int = 16, lr: float = 0.01, reg: float = 0.01):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_users)]
        self.Q = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_items)]

    def predict(self, u: int, i: int) -> float:
        return sum(a * b for a, b in zip(self.P[u], self.Q[i]))

    def fit(self, interactions: List[Tuple[int, int, float]], epochs: int = 20) -> None:
        for _ in range(epochs):
            random.shuffle(interactions)
            for u, i, r in interactions:
                pred = self.predict(u, i)
                err = r - pred
                for f in range(self.k):
                    pu = self.P[u][f]
                    qi = self.Q[i][f]
                    self.P[u][f] += self.lr * (err * qi - self.reg * pu)
                    self.Q[i][f] += self.lr * (err * pu - self.reg * qi)


def top_k(model: MatrixFactorization, user: int, n_items: int, seen: Dict[int, set], k: int = 5) -> List[int]:
    scored = []
    blocked = seen.get(user, set())
    for item in range(n_items):
        if item in blocked:
            continue
        scored.append((model.predict(user, item), item))
    scored.sort(reverse=True)
    return [item for _, item in scored[:k]]


if __name__ == "__main__":
    # (user, item, rating/implicit score)
    data = [
        (0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0), (1, 2, 1.0),
        (2, 2, 1.0), (2, 3, 1.0), (3, 3, 1.0), (3, 4, 1.0),
    ]
    seen = {0: {0, 1}, 1: {1, 2}, 2: {2, 3}, 3: {3, 4}}

    mf = MatrixFactorization(n_users=4, n_items=5, k=8, lr=0.05, reg=0.01)
    mf.fit(data, epochs=80)

    for u in range(4):
        print(f"User {u} recommendations:", top_k(mf, u, n_items=5, seen=seen, k=2))
