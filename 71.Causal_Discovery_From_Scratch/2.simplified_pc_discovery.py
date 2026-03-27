from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Dict, List, Set, Tuple


def mean(x: List[float]) -> float:
    return sum(x) / len(x)


def corr(x: List[float], y: List[float]) -> float:
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def regress_residual(y: List[float], z: List[float]) -> List[float]:
    # simple linear residual y ~ a + b z
    mz, my = mean(z), mean(y)
    varz = sum((zi - mz) ** 2 for zi in z) + 1e-12
    cov = sum((zi - mz) * (yi - my) for zi, yi in zip(z, y))
    b = cov / varz
    a = my - b * mz
    return [yi - (a + b * zi) for yi, zi in zip(y, z)]


def partial_corr_single_z(x: List[float], y: List[float], z: List[float]) -> float:
    rx = regress_residual(x, z)
    ry = regress_residual(y, z)
    return corr(rx, ry)


def indep_test(r: float, threshold: float = 0.08) -> bool:
    return abs(r) < threshold


def simulate_data(n: int = 2500, seed: int = 42):
    random.seed(seed)
    # Ground truth DAG: A -> B, A -> C, B -> D, C -> D, C -> E
    A, B, C, D, E = [], [], [], [], []

    for _ in range(n):
        a = random.gauss(0, 1)
        b = 0.9 * a + random.gauss(0, 0.6)
        c = -0.7 * a + random.gauss(0, 0.7)
        d = 0.8 * b + 0.7 * c + random.gauss(0, 0.7)
        e = 0.9 * c + random.gauss(0, 0.6)

        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)
        E.append(e)

    data = {"A": A, "B": B, "C": C, "D": D, "E": E}
    return data


def build_skeleton(data: Dict[str, List[float]], thr=0.08):
    vars_ = list(data.keys())
    edges = {tuple(sorted((u, v))) for u, v in combinations(vars_, 2)}

    # remove unconditional independencies
    for u, v in list(edges):
        r = corr(data[u], data[v])
        if indep_test(r, threshold=thr):
            edges.discard((u, v))

    # remove conditional independencies given one variable
    sep_set = {}
    for u, v in list(edges):
        others = [z for z in vars_ if z not in (u, v)]
        for z in others:
            pr = partial_corr_single_z(data[u], data[v], data[z])
            if indep_test(pr, threshold=thr):
                edges.discard((u, v))
                sep_set[(u, v)] = {z}
                break

    return edges, sep_set


def orient_v_structures(edges: Set[Tuple[str, str]], sep_set: Dict[Tuple[str, str], Set[str]]):
    vars_ = sorted({v for e in edges for v in e})
    undirected = {v: set() for v in vars_}
    for u, v in edges:
        undirected[u].add(v)
        undirected[v].add(u)

    directed = []
    # For non-adjacent X,Y with common neighbor Z and Z not in separating set => X->Z<-Y
    all_pairs = list(combinations(vars_, 2))
    edge_set = {tuple(sorted(e)) for e in edges}
    for x, y in all_pairs:
        if tuple(sorted((x, y))) in edge_set:
            continue
        common = undirected[x] & undirected[y]
        for z in common:
            key = tuple(sorted((x, y)))
            seps = sep_set.get(key, set())
            if z not in seps:
                directed.append((x, z))
                directed.append((y, z))

    return directed


def maybe_plot(edges: Set[Tuple[str, str]], directed: List[Tuple[str, str]]):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception:
        print("Undirected skeleton edges:", sorted(edges))
        print("Oriented edges (v-structures):", sorted(set(directed)))
        return

    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=3)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="#cfe8ff", edge_color="gray")
    if directed:
        DG = nx.DiGraph()
        DG.add_edges_from(set(directed))
        nx.draw_networkx_edges(DG, pos, edge_color="red", arrows=True, width=2)
    plt.title("Discovered Skeleton + V-Structure Orientations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = simulate_data(n=2500, seed=10)
    edges, sep = build_skeleton(data, thr=0.08)
    directed = orient_v_structures(edges, sep)

    print("Skeleton edges:", sorted(edges))
    print("Directed (v-structure) edges:", sorted(set(directed)))

    maybe_plot(edges, directed)
