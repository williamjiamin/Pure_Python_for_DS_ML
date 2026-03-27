from __future__ import annotations

import heapq
import math
import random
import time
from typing import Dict, List, Tuple


Point = List[float]


def l2(a: Point, b: Point) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class NSWIndex:
    def __init__(self, M: int = 6, ef_search: int = 20):
        self.M = M
        self.ef_search = ef_search
        self.points: List[Point] = []
        self.graph: Dict[int, List[int]] = {}
        self.entry = None

    def _connect(self, u: int, v: int):
        self.graph.setdefault(u, [])
        if v not in self.graph[u]:
            self.graph[u].append(v)
        if len(self.graph[u]) > self.M:
            p_u = self.points[u]
            self.graph[u].sort(key=lambda x: l2(p_u, self.points[x]))
            self.graph[u] = self.graph[u][: self.M]

    def add(self, p: Point):
        idx = len(self.points)
        self.points.append(p)
        self.graph[idx] = []

        if self.entry is None:
            self.entry = idx
            return

        cands = self.search_internal(p, top_k=self.M, ef=self.ef_search)
        neighbors = [i for _, i in cands]

        for nb in neighbors:
            self._connect(idx, nb)
            self._connect(nb, idx)

    def search_internal(self, q: Point, top_k: int = 10, ef: int = 20):
        if self.entry is None:
            return []

        visited = set()
        cand = []  # min-heap of (distance, node)
        best = []  # max-heap of (-distance, node)

        d0 = l2(q, self.points[self.entry])
        heapq.heappush(cand, (d0, self.entry))
        heapq.heappush(best, (-d0, self.entry))
        visited.add(self.entry)

        while cand:
            d_cur, cur = heapq.heappop(cand)
            worst_best = -best[0][0]
            if d_cur > worst_best and len(best) >= ef:
                break

            for nb in self.graph.get(cur, []):
                if nb in visited:
                    continue
                visited.add(nb)
                d_nb = l2(q, self.points[nb])

                if len(best) < ef or d_nb < -best[0][0]:
                    heapq.heappush(cand, (d_nb, nb))
                    heapq.heappush(best, (-d_nb, nb))
                    if len(best) > ef:
                        heapq.heappop(best)

        out = sorted([(-d, i) for d, i in best])
        return out[:top_k]

    def search(self, q: Point, top_k: int = 5):
        return self.search_internal(q, top_k=top_k, ef=self.ef_search)


def brute_force(points: List[Point], q: Point, top_k: int = 5):
    scored = [(l2(p, q), i) for i, p in enumerate(points)]
    scored.sort()
    return scored[:top_k]


def recall_at_k(exact: List[Tuple[float, int]], approx: List[Tuple[float, int]]) -> float:
    a = {i for _, i in exact}
    b = {i for _, i in approx}
    return len(a & b) / max(1, len(a))


def maybe_plot_tradeoff(recalls: List[float], times: List[float], efs: List[int]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("ef_search tradeoff:")
        for ef, r, t in zip(efs, recalls, times):
            print(f"ef={ef:3d} recall={r:.3f} avg_ms={t:.4f}")
        return

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(efs, recalls, marker="o", label="recall@5")
    ax1.set_xlabel("ef_search")
    ax1.set_ylabel("Recall@5")
    ax2 = ax1.twinx()
    ax2.plot(efs, times, marker="s", color="orange", label="ms/query")
    ax2.set_ylabel("Latency (ms/query)")
    fig.tight_layout()
    plt.title("ANN Tradeoff: ef_search vs recall/latency")
    plt.show()


if __name__ == "__main__":
    random.seed(7)
    n_points = 1200
    dim = 8
    points = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_points)]

    index = NSWIndex(M=8, ef_search=30)
    for p in points:
        index.add(p)

    queries = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(80)]

    efs = [8, 16, 24, 32, 48, 64]
    recalls, latencies = [], []
    for ef in efs:
        index.ef_search = ef
        rs = []
        t0 = time.time()
        for q in queries:
            exact = brute_force(points, q, top_k=5)
            approx = index.search(q, top_k=5)
            rs.append(recall_at_k(exact, approx))
        elapsed = (time.time() - t0) * 1000 / len(queries)
        recalls.append(sum(rs) / len(rs))
        latencies.append(elapsed)

    for ef, r, ms in zip(efs, recalls, latencies):
        print(f"ef={ef:3d} recall@5={r:.3f} ms/query={ms:.4f}")

    maybe_plot_tradeoff(recalls, latencies, efs)
