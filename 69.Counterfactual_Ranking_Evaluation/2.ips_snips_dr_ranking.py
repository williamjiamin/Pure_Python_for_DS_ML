from __future__ import annotations

import random
from typing import Dict, List, Tuple


def simulate_environment(n_items: int = 60, seed: int = 42):
    random.seed(seed)
    # latent relevance per segment and item
    relevance = {
        seg: [max(0.0, min(1.0, random.uniform(0.02, 0.5) + (0.25 if i % 5 == seg else 0.0))) for i in range(n_items)]
        for seg in range(5)
    }
    return relevance


def sample_segment() -> int:
    r = random.random()
    if r < 0.2:
        return 0
    if r < 0.4:
        return 1
    if r < 0.6:
        return 2
    if r < 0.8:
        return 3
    return 4


def logging_policy(seg: int, n_items: int, eps: float = 0.25):
    # mostly rank segment-matching items near top, with epsilon exploration
    preferred = [i for i in range(n_items) if i % 5 == seg]
    others = [i for i in range(n_items) if i % 5 != seg]
    ranked = preferred + others

    # epsilon shuffle top region
    if random.random() < eps:
        top = ranked[:12]
        random.shuffle(top)
        ranked = top + ranked[12:]

    # propensity approximation for top-1 shown item
    p = (1 - eps) if ranked[0] in preferred else eps / max(1, len(others))
    p = max(1e-3, min(1.0, p))
    return ranked[0], p


def target_policy(seg: int, n_items: int) -> int:
    # new policy prefers a shifted pattern
    candidates = [i for i in range(n_items) if i % 5 in {seg, (seg + 1) % 5}]
    # deterministic choice for simplicity
    return candidates[0]


def reward_draw(relevance: Dict[int, List[float]], seg: int, item: int) -> int:
    p = relevance[seg][item]
    return 1 if random.random() < p else 0


def direct_model_predict(relevance: Dict[int, List[float]], seg: int, item: int) -> float:
    # pseudo-outcome model with noise / misspecification
    base = relevance[seg][item]
    return max(0.0, min(1.0, 0.9 * base + 0.02))


def collect_logged_data(T: int, relevance, n_items=60, seed=7):
    random.seed(seed)
    logs = []
    for _ in range(T):
        seg = sample_segment()
        item, prop = logging_policy(seg, n_items=n_items, eps=0.25)
        r = reward_draw(relevance, seg, item)
        logs.append((seg, item, prop, r))
    return logs


def ips(logs, target_fn, n_items=60):
    s = 0.0
    for seg, item, prop, r in logs:
        tgt = target_fn(seg, n_items)
        if item == tgt:
            s += r / prop
    return s / len(logs)


def snips(logs, target_fn, n_items=60):
    num = 0.0
    den = 0.0
    for seg, item, prop, r in logs:
        tgt = target_fn(seg, n_items)
        if item == tgt:
            w = 1.0 / prop
            num += w * r
            den += w
    return num / max(1e-12, den)


def doubly_robust(logs, target_fn, relevance, n_items=60):
    s = 0.0
    for seg, item, prop, r in logs:
        tgt = target_fn(seg, n_items)
        q_tgt = direct_model_predict(relevance, seg, tgt)
        q_log = direct_model_predict(relevance, seg, item)
        corr = ((r - q_log) / prop) if item == tgt else 0.0
        s += q_tgt + corr
    return s / len(logs)


def online_ground_truth(relevance, target_fn, n_items=60, T=40000, seed=123):
    random.seed(seed)
    reward = 0
    for _ in range(T):
        seg = sample_segment()
        item = target_fn(seg, n_items)
        reward += reward_draw(relevance, seg, item)
    return reward / T


def maybe_plot(estimates: Dict[str, float], gt: float):
    labels = list(estimates.keys()) + ["ground_truth"]
    vals = list(estimates.values()) + [gt]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Estimates vs GT:")
        for l, v in zip(labels, vals):
            print(l, round(v, 5))
        return

    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.title("Counterfactual Policy Value Estimates")
    plt.ylabel("Estimated Reward")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_items = 60
    relevance = simulate_environment(n_items=n_items, seed=11)
    logs = collect_logged_data(T=14000, relevance=relevance, n_items=n_items, seed=12)

    est_ips = ips(logs, target_policy, n_items=n_items)
    est_snips = snips(logs, target_policy, n_items=n_items)
    est_dr = doubly_robust(logs, target_policy, relevance, n_items=n_items)
    gt = online_ground_truth(relevance, target_policy, n_items=n_items, T=50000, seed=13)

    estimates = {"IPS": est_ips, "SNIPS": est_snips, "DR": est_dr}
    print("Estimates:")
    for k, v in estimates.items():
        print(f"- {k}: {v:.5f}")
    print("Ground truth online:", round(gt, 5))

    maybe_plot(estimates, gt)
