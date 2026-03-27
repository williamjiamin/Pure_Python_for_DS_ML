from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def simulate_user_item_graph(n_users=90, n_items=140, seed=42):
    random.seed(seed)

    # hidden item groups
    item_group = {i: i % 7 for i in range(n_items)}
    user_pref = {u: random.randrange(7) for u in range(n_users)}

    interactions = defaultdict(set)
    for u in range(n_users):
        pref = user_pref[u]
        candidates = [i for i in range(n_items) if item_group[i] == pref]
        negatives = [i for i in range(n_items) if item_group[i] != pref]

        for i in random.sample(candidates, k=10):
            interactions[u].add(i)
        for i in random.sample(negatives, k=4):
            interactions[u].add(i)

    return interactions, n_users, n_items


def train_test_split(interactions: Dict[int, Set[int]], seed=7):
    random.seed(seed)
    train = defaultdict(set)
    test = defaultdict(set)

    for u, items in interactions.items():
        items = list(items)
        random.shuffle(items)
        hold = max(2, len(items) // 5)
        test_items = set(items[:hold])
        train_items = set(items[hold:])
        train[u] = train_items
        test[u] = test_items

    return train, test


def build_graph(train: Dict[int, Set[int]], n_users: int, n_items: int):
    # node ids: users [0..n_users-1], items [n_users..n_users+n_items-1]
    graph = defaultdict(set)
    for u, items in train.items():
        for i in items:
            item_node = n_users + i
            graph[u].add(item_node)
            graph[item_node].add(u)
    return graph


def personalized_pagerank(
    graph,
    start_node: int,
    max_iter: int = 60,
    alpha: float = 0.2,
):
    nodes = list(graph.keys())
    p = {n: 0.0 for n in nodes}
    p[start_node] = 1.0

    for _ in range(max_iter):
        new_p = {n: 0.0 for n in nodes}
        for n in nodes:
            nbrs = list(graph[n])
            if not nbrs:
                continue
            share = p[n] / len(nbrs)
            for nb in nbrs:
                new_p[nb] += (1 - alpha) * share

        new_p[start_node] += alpha
        p = new_p

    return p


def recommend_for_user(u: int, graph, train, n_users: int, n_items: int, k: int = 10, alpha: float = 0.2):
    p = personalized_pagerank(graph, start_node=u, alpha=alpha)

    scored = []
    seen = train[u]
    for i in range(n_items):
        if i in seen:
            continue
        node = n_users + i
        scored.append((p.get(node, 0.0), i))

    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def popularity_baseline(train: Dict[int, Set[int]], n_items: int):
    cnt = [0] * n_items
    for items in train.values():
        for i in items:
            cnt[i] += 1
    rank = sorted(range(n_items), key=lambda i: cnt[i], reverse=True)
    return rank


def hit_recall_at_k(recs: List[int], truth: Set[int]) -> Tuple[float, float]:
    hit = 1.0 if set(recs) & truth else 0.0
    recall = len(set(recs) & truth) / max(1, len(truth))
    return hit, recall


def evaluate(train, test, graph, n_users, n_items, k=10, alpha=0.2):
    hits, recalls = [], []
    for u in range(n_users):
        recs = recommend_for_user(u, graph, train, n_users, n_items, k=k, alpha=alpha)
        h, r = hit_recall_at_k(recs, test[u])
        hits.append(h)
        recalls.append(r)
    return sum(hits) / len(hits), sum(recalls) / len(recalls)


def eval_popularity(train, test, n_users, n_items, k=10):
    pop = popularity_baseline(train, n_items)
    hits, recalls = [], []
    for u in range(n_users):
        recs = [i for i in pop if i not in train[u]][:k]
        h, r = hit_recall_at_k(recs, test[u])
        hits.append(h)
        recalls.append(r)
    return sum(hits) / len(hits), sum(recalls) / len(recalls)


def maybe_plot(alphas, recalls):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("alpha sweep recalls:", [(round(a, 2), round(r, 4)) for a, r in zip(alphas, recalls)])
        return

    plt.figure(figsize=(7, 4))
    plt.plot(alphas, recalls, marker="o")
    plt.xlabel("restart alpha")
    plt.ylabel("Recall@10")
    plt.title("Graph Recommender Sensitivity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    interactions, n_users, n_items = simulate_user_item_graph(n_users=80, n_items=130, seed=17)
    train, test = train_test_split(interactions, seed=19)
    graph = build_graph(train, n_users, n_items)

    hit_pop, rec_pop = eval_popularity(train, test, n_users, n_items, k=10)

    print("Popularity baseline Hit@10:", round(hit_pop, 4), "Recall@10:", round(rec_pop, 4))

    alphas = [0.05, 0.1, 0.2, 0.3, 0.4]
    recalls = []
    for a in alphas:
        hit, rec = evaluate(train, test, graph, n_users, n_items, k=10, alpha=a)
        recalls.append(rec)
        print(f"Graph alpha={a:.2f} -> Hit@10={hit:.4f}, Recall@10={rec:.4f}")

    maybe_plot(alphas, recalls)
