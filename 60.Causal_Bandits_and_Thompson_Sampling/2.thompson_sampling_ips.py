from __future__ import annotations

import random
from typing import Dict, List, Tuple


def segment_reward_prob(segment: int, arm: int) -> float:
    # contextual/causal heterogeneity by segment
    table = {
        0: {0: 0.08, 1: 0.12, 2: 0.10},
        1: {0: 0.14, 1: 0.10, 2: 0.16},
        2: {0: 0.11, 1: 0.18, 2: 0.09},
    }
    return table[segment][arm]


def sample_segment() -> int:
    r = random.random()
    if r < 0.35:
        return 0
    if r < 0.75:
        return 1
    return 2


def run_thompson(T: int = 12000, n_arms: int = 3, seed: int = 42):
    random.seed(seed)
    alpha = [1.0] * n_arms
    beta = [1.0] * n_arms

    cum_reward = 0
    rewards = []
    log = []  # (segment, chosen_arm, propensity, reward)

    for _t in range(T):
        seg = sample_segment()
        samples = [random.betavariate(alpha[a], beta[a]) for a in range(n_arms)]
        arm = max(range(n_arms), key=lambda a: samples[a])

        # Thompson propensity approximated via Monte Carlo for IPS logging.
        mc = 120
        wins = 0
        for _ in range(mc):
            ss = [random.betavariate(alpha[a], beta[a]) for a in range(n_arms)]
            if arm == max(range(n_arms), key=lambda a: ss[a]):
                wins += 1
        prop = max(1e-3, wins / mc)

        p = segment_reward_prob(seg, arm)
        r = 1 if random.random() < p else 0
        cum_reward += r

        alpha[arm] += r
        beta[arm] += (1 - r)

        rewards.append(cum_reward)
        log.append((seg, arm, prop, r))

    return rewards, log


def run_epsilon_greedy(T: int = 12000, n_arms: int = 3, eps: float = 0.1, seed: int = 42):
    random.seed(seed)
    counts = [0] * n_arms
    values = [0.0] * n_arms

    cum_reward = 0
    rewards = []

    for _t in range(T):
        seg = sample_segment()
        if random.random() < eps:
            arm = random.randrange(n_arms)
        else:
            arm = max(range(n_arms), key=lambda a: values[a])

        p = segment_reward_prob(seg, arm)
        r = 1 if random.random() < p else 0
        cum_reward += r

        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]

        rewards.append(cum_reward)

    return rewards


def ips_eval(log: List[Tuple[int, int, float, int]], policy) -> float:
    s = 0.0
    n = len(log)
    for seg, arm, prop, r in log:
        a_new = policy(seg)
        if a_new == arm:
            s += r / prop
    return s / n


def best_arm_policy(seg: int) -> int:
    # oracle per segment (for evaluation reference)
    best = {
        0: 1,
        1: 2,
        2: 1,
    }
    return best[seg]


def global_best_policy(_seg: int) -> int:
    return 1


def maybe_plot(ts_rewards: List[int], eg_rewards: List[int]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Final reward Thompson:", ts_rewards[-1])
        print("Final reward EpsGreedy:", eg_rewards[-1])
        return

    plt.figure(figsize=(8, 4))
    plt.plot(ts_rewards, label="Thompson")
    plt.plot(eg_rewards, label="Epsilon-Greedy")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Reward")
    plt.title("Bandit Policy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ts_rewards, ts_log = run_thompson(T=9000, n_arms=3, seed=7)
    eg_rewards = run_epsilon_greedy(T=9000, n_arms=3, eps=0.12, seed=7)

    print("Final cumulative reward Thompson:", ts_rewards[-1])
    print("Final cumulative reward EpsGreedy:", eg_rewards[-1])

    ips_oracle = ips_eval(ts_log, best_arm_policy)
    ips_global = ips_eval(ts_log, global_best_policy)

    print("IPS value (oracle contextual policy):", round(ips_oracle, 5))
    print("IPS value (global arm policy)      :", round(ips_global, 5))

    maybe_plot(ts_rewards, eg_rewards)
