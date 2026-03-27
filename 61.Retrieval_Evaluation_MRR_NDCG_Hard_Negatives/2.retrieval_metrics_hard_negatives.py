from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def recall_at_k(ranked_ids: List[int], relevant: set, k: int) -> float:
    topk = set(ranked_ids[:k])
    return len(topk & relevant) / max(1, len(relevant))


def mrr(ranked_ids: List[int], relevant: set) -> float:
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: List[int], gains: Dict[int, float], k: int) -> float:
    out = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        g = gains.get(doc_id, 0.0)
        out += (2 ** g - 1) / math.log2(i + 1)
    return out


def ndcg_at_k(ranked_ids: List[int], gains: Dict[int, float], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, gains, k)
    ideal = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    ideal_ids = [d for d, _ in ideal]
    idcg = dcg_at_k(ideal_ids, gains, k)
    return dcg / idcg if idcg > 0 else 0.0


def make_synthetic(seed: int = 42):
    random.seed(seed)
    n_docs = 300
    dim = 10

    docs = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_docs)]
    queries = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(40)]

    # Define relevance by hidden similarity + noise.
    relevance = {}
    graded = {}
    for qi, q in enumerate(queries):
        scores = [(cosine(q, d) + random.uniform(-0.05, 0.05), i) for i, d in enumerate(docs)]
        scores.sort(reverse=True)
        rel = set(i for _, i in scores[:8])
        gains = {i: (3.0 - rank * 0.3) for rank, (_, i) in enumerate(scores[:8])}
        relevance[qi] = rel
        graded[qi] = gains

    return docs, queries, relevance, graded


def retrieve(query: List[float], docs: List[List[float]]) -> List[int]:
    scored = [(cosine(query, d), i) for i, d in enumerate(docs)]
    scored.sort(reverse=True)
    return [i for _, i in scored]


def add_hard_negatives(
    docs: List[List[float]],
    queries: List[List[float]],
    relevance: Dict[int, set],
    n_hard: int = 6,
):
    # For each query, add near-relevant non-relevant items to candidate pool
    hard_map = {}
    for qi, q in enumerate(queries):
        scored = [(cosine(q, d), i) for i, d in enumerate(docs)]
        scored.sort(reverse=True)
        hard = []
        for s, i in scored:
            if i not in relevance[qi]:
                hard.append(i)
            if len(hard) == n_hard:
                break
        hard_map[qi] = hard
    return hard_map


def evaluate(docs, queries, relevance, graded, hard_map=None, k=10):
    recs, mrrs, ndcgs = [], [], []
    losses = []

    for qi, q in enumerate(queries):
        ranked = retrieve(q, docs)

        # simulate harder eval by forcing hard negatives to front slice
        if hard_map is not None:
            hard = hard_map[qi]
            remaining = [d for d in ranked if d not in hard]
            ranked = hard + remaining

        r = recall_at_k(ranked, relevance[qi], k)
        mr = mrr(ranked, relevance[qi])
        nd = ndcg_at_k(ranked, graded[qi], k)

        recs.append(r)
        mrrs.append(mr)
        ndcgs.append(nd)
        losses.append((qi, mr, r, nd))

    return {
        "recall@10": sum(recs) / len(recs),
        "mrr": sum(mrrs) / len(mrrs),
        "ndcg@10": sum(ndcgs) / len(ndcgs),
        "per_query": losses,
    }


def maybe_plot(before, after):
    labels = ["Recall@10", "MRR", "NDCG@10"]
    b = [before["recall@10"], before["mrr"], before["ndcg@10"]]
    a = [after["recall@10"], after["mrr"], after["ndcg@10"]]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Metric comparison:")
        for name, x, y in zip(labels, b, a):
            print(f"- {name}: baseline={x:.4f} hard_neg={y:.4f}")
        return

    x = range(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.15 for i in x], b, width=0.3, label="baseline")
    plt.bar([i + 0.15 for i in x], a, width=0.3, label="hard negatives")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1)
    plt.title("Retrieval Metrics: Baseline vs Hard Negatives")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    docs, queries, relevance, graded = make_synthetic(seed=9)
    hard_map = add_hard_negatives(docs, queries, relevance, n_hard=7)

    baseline = evaluate(docs, queries, relevance, graded, hard_map=None, k=10)
    hard = evaluate(docs, queries, relevance, graded, hard_map=hard_map, k=10)

    print("Baseline:", {k: round(v, 4) for k, v in baseline.items() if k != "per_query"})
    print("Hard Neg:", {k: round(v, 4) for k, v in hard.items() if k != "per_query"})

    # show worst queries under hard negative setting
    worst = sorted(hard["per_query"], key=lambda x: x[1])[:5]
    print("Worst 5 queries by MRR under hard negatives:", worst)

    maybe_plot(baseline, hard)
