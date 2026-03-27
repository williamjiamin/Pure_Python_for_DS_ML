from __future__ import annotations

import random
from typing import Dict, List, Tuple


Item = Dict[str, float]


def generate_items(n: int = 120, seed: int = 42) -> List[Item]:
    random.seed(seed)
    items = []
    categories = ["A", "B", "C"]
    for i in range(n):
        c = categories[i % 3]
        base = random.uniform(0.2, 0.9)
        item = {
            "id": i,
            "cat": c,
            "click": max(0.0, min(1.0, base + random.uniform(-0.2, 0.2))),
            "conv": max(0.0, min(1.0, 0.6 * base + random.uniform(-0.25, 0.25))),
            "eng": max(0.0, min(1.0, 0.8 * (1 - base) + random.uniform(-0.2, 0.2))),
        }
        items.append(item)
    return items


def rank_items(items: List[Item], w_click: float, w_conv: float, w_eng: float) -> List[Item]:
    scored = []
    for it in items:
        s = w_click * it["click"] + w_conv * it["conv"] + w_eng * it["eng"]
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored]


def apply_category_quota(ranked: List[Item], k: int = 20, min_per_cat: int = 4) -> List[Item]:
    selected = []
    need = {"A": min_per_cat, "B": min_per_cat, "C": min_per_cat}

    # first pass: satisfy quotas
    for it in ranked:
        c = it["cat"]
        if need[c] > 0 and len(selected) < k:
            selected.append(it)
            need[c] -= 1

    # second pass: fill remaining
    selected_ids = {it["id"] for it in selected}
    for it in ranked:
        if len(selected) >= k:
            break
        if it["id"] in selected_ids:
            continue
        selected.append(it)

    return selected


def evaluate_topk(items: List[Item]) -> Dict[str, float]:
    n = len(items)
    return {
        "ctr": sum(it["click"] for it in items) / n,
        "cvr": sum(it["conv"] for it in items) / n,
        "eng": sum(it["eng"] for it in items) / n,
        "utility": sum(0.45 * it["click"] + 0.35 * it["conv"] + 0.2 * it["eng"] for it in items) / n,
    }


def frontier(items: List[Item], k: int = 20):
    points = []
    for wc in [i / 10 for i in range(1, 9)]:
        for wv in [j / 10 for j in range(1, 9)]:
            we = 1.0 - wc - wv
            if we <= 0:
                continue
            ranked = rank_items(items, wc, wv, we)
            top = ranked[:k]
            m = evaluate_topk(top)
            points.append((wc, wv, we, m["ctr"], m["cvr"], m["eng"], m["utility"]))
    return points


def maybe_plot(points):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Top frontier points by utility:")
        for row in sorted(points, key=lambda x: x[-1], reverse=True)[:8]:
            print(tuple(round(v, 4) if isinstance(v, float) else v for v in row))
        return

    xs = [p[3] for p in points]
    ys = [p[4] for p in points]
    cs = [p[6] for p in points]

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(xs, ys, c=cs, cmap="viridis")
    plt.colorbar(sc, label="Utility")
    plt.xlabel("CTR")
    plt.ylabel("CVR")
    plt.title("Multi-Objective Ranking Frontier")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    items = generate_items(n=150, seed=17)

    ranked = rank_items(items, w_click=0.5, w_conv=0.35, w_eng=0.15)
    top_unconstrained = ranked[:20]
    top_constrained = apply_category_quota(ranked, k=20, min_per_cat=6)

    m_un = evaluate_topk(top_unconstrained)
    m_co = evaluate_topk(top_constrained)

    print("Unconstrained top-20:", {k: round(v, 4) for k, v in m_un.items()})
    print("Constrained top-20  :", {k: round(v, 4) for k, v in m_co.items()})

    cats_un = {"A": 0, "B": 0, "C": 0}
    cats_co = {"A": 0, "B": 0, "C": 0}
    for it in top_unconstrained:
        cats_un[it["cat"]] += 1
    for it in top_constrained:
        cats_co[it["cat"]] += 1

    print("Category counts unconstrained:", cats_un)
    print("Category counts constrained  :", cats_co)

    points = frontier(items, k=20)
    maybe_plot(points)
