from __future__ import annotations

import random
from typing import Dict, List, Tuple


def sample_segment() -> int:
    r = random.random()
    if r < 0.35:
        return 0
    if r < 0.7:
        return 1
    return 2


def uplift_true(seg: int) -> float:
    return {0: 0.03, 1: 0.08, 2: -0.01}[seg]


def base_response(seg: int) -> float:
    return {0: 0.09, 1: 0.11, 2: 0.07}[seg]


def reward(seg: int, treat: int) -> int:
    p = base_response(seg) + (uplift_true(seg) if treat == 1 else 0.0)
    p = max(0.001, min(0.999, p))
    return 1 if random.random() < p else 0


def run_ab_test(T=12000, treat_prob=0.5, seed=42):
    random.seed(seed)
    total = 0
    for _ in range(T):
        seg = sample_segment()
        tr = 1 if random.random() < treat_prob else 0
        total += reward(seg, tr)
    return total


def run_uplift_policy(T=12000, seed=42):
    random.seed(seed)
    total = 0
    for _ in range(T):
        seg = sample_segment()
        # Treat only if uplift expected positive
        tr = 1 if uplift_true(seg) > 0 else 0
        total += reward(seg, tr)
    return total


def run_hybrid_bandit(T=12000, mix=0.7, seed=42):
    random.seed(seed)

    # Thompson posteriors per segment for treatment arm effect (binary reward modeling)
    alpha_t = {0: 1.0, 1: 1.0, 2: 1.0}
    beta_t = {0: 1.0, 1: 1.0, 2: 1.0}
    alpha_c = {0: 1.0, 1: 1.0, 2: 1.0}
    beta_c = {0: 1.0, 1: 1.0, 2: 1.0}

    total = 0
    cumulative = []

    for _ in range(T):
        seg = sample_segment()

        # Uplift prior signal (could be model estimate in real system)
        uplift_est = uplift_true(seg)
        uplift_action = 1 if uplift_est > 0 else 0

        # Bandit action via Thompson on segment-specific treatment vs control
        p_t = random.betavariate(alpha_t[seg], beta_t[seg])
        p_c = random.betavariate(alpha_c[seg], beta_c[seg])
        bandit_action = 1 if p_t >= p_c else 0

        # Hybrid mix
        if random.random() < mix:
            action = uplift_action
        else:
            action = bandit_action

        r = reward(seg, action)
        total += r

        if action == 1:
            alpha_t[seg] += r
            beta_t[seg] += 1 - r
        else:
            alpha_c[seg] += r
            beta_c[seg] += 1 - r

        cumulative.append(total)

    return total, cumulative


def maybe_plot(mixes: List[float], vals: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Hybrid mix performance:", list(zip([round(m, 2) for m in mixes], [round(v, 4) for v in vals])))
        return

    plt.figure(figsize=(7, 4))
    plt.plot(mixes, vals, marker="o")
    plt.xlabel("Uplift-policy mix weight")
    plt.ylabel("Average reward per round")
    plt.title("Hybrid Uplift+Bandit Policy")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    T = 14000

    ab = run_ab_test(T=T, treat_prob=0.5, seed=9)
    up = run_uplift_policy(T=T, seed=9)

    mixes = [0.0, 0.25, 0.5, 0.75, 1.0]
    per_round = []
    for m in mixes:
        total, _ = run_hybrid_bandit(T=T, mix=m, seed=9)
        per_round.append(total / T)
        print(f"Hybrid mix={m:.2f} total={total} avg={total/T:.5f}")

    print("A/B baseline total:", ab, "avg:", round(ab / T, 5))
    print("Uplift-only total :", up, "avg:", round(up / T, 5))

    maybe_plot(mixes, per_round)
