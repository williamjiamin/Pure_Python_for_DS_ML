from __future__ import annotations

import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35, min(35, x))))


def policy_prob(theta: List[float], state: List[float], action: int) -> float:
    # binary action policy via logistic model
    z = sum(w * x for w, x in zip(theta, state))
    p1 = sigmoid(z)
    return p1 if action == 1 else (1.0 - p1)


def sample_batch(n: int = 500, seed: int = 42):
    random.seed(seed)
    states, actions, advantages = [], [], []
    for _ in range(n):
        s = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), 1.0]
        # synthetic old policy behavior
        pa = sigmoid(1.1 * s[0] - 0.6 * s[1])
        a = 1 if random.random() < pa else 0
        # synthetic advantage signal
        adv = (0.8 * s[0] - 0.5 * s[1]) + random.uniform(-0.2, 0.2)
        states.append(s)
        actions.append(a)
        advantages.append(adv)
    return states, actions, advantages


def ppo_objective(
    theta: List[float],
    theta_old: List[float],
    states: List[List[float]],
    actions: List[int],
    advs: List[float],
    clip_eps: float,
) -> Tuple[float, float]:
    unclipped_vals = []
    clipped_vals = []

    for s, a, A in zip(states, actions, advs):
        p_new = policy_prob(theta, s, a)
        p_old = policy_prob(theta_old, s, a)
        ratio = p_new / max(1e-12, p_old)

        u = ratio * A
        c = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps) * A
        unclipped_vals.append(u)
        clipped_vals.append(min(u, c))

    return sum(unclipped_vals) / len(unclipped_vals), sum(clipped_vals) / len(clipped_vals)


def finite_diff_grad(
    theta: List[float],
    theta_old: List[float],
    states: List[List[float]],
    actions: List[int],
    advs: List[float],
    clip_eps: float,
    h: float = 1e-4,
) -> List[float]:
    grad = []
    _, base = ppo_objective(theta, theta_old, states, actions, advs, clip_eps)
    for i in range(len(theta)):
        t2 = theta[:]
        t2[i] += h
        _, v2 = ppo_objective(t2, theta_old, states, actions, advs, clip_eps)
        grad.append((v2 - base) / h)
    return grad


def maybe_plot(unclipped_hist: List[float], clipped_hist: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("step unclipped clipped")
        for i in range(0, len(unclipped_hist), max(1, len(unclipped_hist) // 10)):
            print(i, round(unclipped_hist[i], 6), round(clipped_hist[i], 6))
        return

    plt.figure(figsize=(8, 4))
    plt.plot(unclipped_hist, label="Unclipped objective")
    plt.plot(clipped_hist, label="Clipped objective")
    plt.title("PPO Objective Progress")
    plt.xlabel("Optimization Step")
    plt.ylabel("Objective")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    states, actions, advs = sample_batch(n=700, seed=9)

    theta_old = [0.2, -0.1, 0.0]
    theta = theta_old[:]

    lr = 0.3
    clip_eps = 0.2
    steps = 60

    unclipped_hist, clipped_hist = [], []
    for _ in range(steps):
        u, c = ppo_objective(theta, theta_old, states, actions, advs, clip_eps)
        unclipped_hist.append(u)
        clipped_hist.append(c)

        grad = finite_diff_grad(theta, theta_old, states, actions, advs, clip_eps)
        for i in range(len(theta)):
            theta[i] += lr * grad[i]

    print("theta_old:", [round(v, 4) for v in theta_old])
    print("theta_new:", [round(v, 4) for v in theta])
    print("final unclipped:", round(unclipped_hist[-1], 6))
    print("final clipped  :", round(clipped_hist[-1], 6))

    maybe_plot(unclipped_hist, clipped_hist)
