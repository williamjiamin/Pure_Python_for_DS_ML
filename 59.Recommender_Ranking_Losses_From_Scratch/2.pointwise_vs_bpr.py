from __future__ import annotations

import math
import random
from typing import Dict, List, Set, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, x))))


class MFPointwise:
    def __init__(self, n_users: int, n_items: int, k: int = 12, lr: float = 0.04, reg: float = 1e-3, seed: int = 42):
        random.seed(seed)
        self.k = k
        self.lr = lr
        self.reg = reg
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_users)]
        self.Q = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_items)]

    def score(self, u: int, i: int) -> float:
        return sum(a * b for a, b in zip(self.P[u], self.Q[i]))

    def train(self, samples: List[Tuple[int, int, int]], epochs: int = 15):
        for _ in range(epochs):
            random.shuffle(samples)
            for u, i, y in samples:
                s = self.score(u, i)
                p = sigmoid(s)
                err = y - p
                for f in range(self.k):
                    pu = self.P[u][f]
                    qi = self.Q[i][f]
                    self.P[u][f] += self.lr * (err * qi - self.reg * pu)
                    self.Q[i][f] += self.lr * (err * pu - self.reg * qi)


class MFBPR:
    def __init__(self, n_users: int, n_items: int, k: int = 12, lr: float = 0.05, reg: float = 1e-3, seed: int = 42):
        random.seed(seed)
        self.k = k
        self.lr = lr
        self.reg = reg
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_users)]
        self.Q = [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(n_items)]

    def score(self, u: int, i: int) -> float:
        return sum(a * b for a, b in zip(self.P[u], self.Q[i]))

    def train(self, user_pos: Dict[int, Set[int]], n_items: int, steps: int = 20000):
        users = list(user_pos.keys())
        for _ in range(steps):
            u = random.choice(users)
            if not user_pos[u]:
                continue
            i = random.choice(list(user_pos[u]))
            j = random.randrange(n_items)
            while j in user_pos[u]:
                j = random.randrange(n_items)

            xui = self.score(u, i)
            xuj = self.score(u, j)
            x = xui - xuj
            grad = 1.0 - sigmoid(x)

            for f in range(self.k):
                pu = self.P[u][f]
                qi = self.Q[i][f]
                qj = self.Q[j][f]

                self.P[u][f] += self.lr * (grad * (qi - qj) - self.reg * pu)
                self.Q[i][f] += self.lr * (grad * pu - self.reg * qi)
                self.Q[j][f] += self.lr * (-grad * pu - self.reg * qj)


def simulate_interactions(n_users=80, n_items=120, seed=9):
    random.seed(seed)
    # latent preference generator
    U = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(n_users)]
    I = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(n_items)]

    user_pos = {u: set() for u in range(n_users)}
    for u in range(n_users):
        scored = []
        for i in range(n_items):
            s = sum(a * b for a, b in zip(U[u], I[i])) + random.uniform(-0.1, 0.1)
            scored.append((s, i))
        scored.sort(reverse=True)
        for _, i in scored[:10]:
            user_pos[u].add(i)

    return user_pos


def make_pointwise_samples(user_pos: Dict[int, Set[int]], n_items: int, neg_per_pos: int = 2):
    samples = []
    for u, pos_items in user_pos.items():
        for i in pos_items:
            samples.append((u, i, 1))
            for _ in range(neg_per_pos):
                j = random.randrange(n_items)
                while j in pos_items:
                    j = random.randrange(n_items)
                samples.append((u, j, 0))
    return samples


def recall_at_k(model, user_pos: Dict[int, Set[int]], n_items: int, k: int = 10) -> float:
    total = 0.0
    for u, positives in user_pos.items():
        scored = [(model.score(u, i), i) for i in range(n_items)]
        scored.sort(reverse=True)
        topk = [i for _, i in scored[:k]]
        hit = len(set(topk) & positives)
        total += hit / max(1, len(positives))
    return total / len(user_pos)


def maybe_plot(vals_point: List[float], vals_bpr: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Pointwise recalls:", [round(v, 4) for v in vals_point])
        print("BPR recalls     :", [round(v, 4) for v in vals_bpr])
        return

    xs = list(range(1, len(vals_point) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, vals_point, marker="o", label="pointwise")
    plt.plot(xs, vals_bpr, marker="o", label="bpr")
    plt.xlabel("Checkpoint")
    plt.ylabel("Recall@10")
    plt.title("Ranking Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_users, n_items = 70, 100
    user_pos = simulate_interactions(n_users=n_users, n_items=n_items, seed=21)

    point = MFPointwise(n_users, n_items, k=10, lr=0.035, reg=2e-3, seed=2)
    bpr = MFBPR(n_users, n_items, k=10, lr=0.045, reg=2e-3, seed=2)

    samples = make_pointwise_samples(user_pos, n_items, neg_per_pos=2)

    point_recalls, bpr_recalls = [], []
    for _ in range(6):
        point.train(samples, epochs=3)
        bpr.train(user_pos, n_items=n_items, steps=4500)
        point_recalls.append(recall_at_k(point, user_pos, n_items, k=10))
        bpr_recalls.append(recall_at_k(bpr, user_pos, n_items, k=10))

    print("Final Recall@10 pointwise:", round(point_recalls[-1], 4))
    print("Final Recall@10 BPR      :", round(bpr_recalls[-1], 4))
    maybe_plot(point_recalls, bpr_recalls)
