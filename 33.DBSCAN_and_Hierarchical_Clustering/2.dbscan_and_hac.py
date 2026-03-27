from __future__ import annotations

import math
from typing import List, Tuple


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def region_query(X: List[List[float]], point_idx: int, eps: float) -> List[int]:
    return [i for i, p in enumerate(X) if euclidean(X[point_idx], p) <= eps]


def dbscan(X: List[List[float]], eps: float = 0.8, min_pts: int = 3) -> List[int]:
    n = len(X)
    labels = [-99] * n  # -99 means unvisited
    cluster_id = 0

    for i in range(n):
        if labels[i] != -99:
            continue

        neighbors = region_query(X, i, eps)
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
            continue

        labels[i] = cluster_id
        queue = neighbors[:]
        q_ptr = 0

        while q_ptr < len(queue):
            j = queue[q_ptr]
            q_ptr += 1

            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                continue

            labels[j] = cluster_id
            nbs = region_query(X, j, eps)
            if len(nbs) >= min_pts:
                for nb in nbs:
                    if nb not in queue:
                        queue.append(nb)

        cluster_id += 1

    return labels


def single_linkage_hac(X: List[List[float]], target_k: int = 2) -> List[List[int]]:
    clusters = [[i] for i in range(len(X))]

    def cluster_distance(c1: List[int], c2: List[int]) -> float:
        best = float("inf")
        for i in c1:
            for j in c2:
                d = euclidean(X[i], X[j])
                if d < best:
                    best = d
        return best

    while len(clusters) > target_k:
        best_i, best_j = 0, 1
        best_d = cluster_distance(clusters[0], clusters[1])

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j])
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j

        merged = clusters[best_i] + clusters[best_j]
        new_clusters = []
        for idx, c in enumerate(clusters):
            if idx not in (best_i, best_j):
                new_clusters.append(c)
        new_clusters.append(merged)
        clusters = new_clusters

    return clusters


if __name__ == "__main__":
    X = [
        [1.0, 1.1], [1.3, 1.0], [0.9, 1.2],
        [5.0, 5.2], [5.2, 4.9], [4.8, 5.1],
        [9.0, 1.0],
    ]

    labels = dbscan(X, eps=0.6, min_pts=2)
    print("DBSCAN labels:", labels)

    clusters = single_linkage_hac(X, target_k=2)
    print("HAC clusters:", clusters)
