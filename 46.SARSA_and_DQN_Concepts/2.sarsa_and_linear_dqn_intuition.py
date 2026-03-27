from __future__ import annotations

import random
from typing import Dict, List, Tuple


State = Tuple[int, int]
Action = int


class RiskyGrid:
    """Grid where one middle cell has stochastic negative reward transitions."""

    def __init__(self, size: int = 6, slip_prob: float = 0.15):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.risky = (size // 2, size // 2)
        self.slip_prob = slip_prob
        self.state = self.start

    def reset(self) -> State:
        self.state = self.start
        return self.state

    def step(self, action: Action):
        # 0 up, 1 down, 2 left, 3 right
        r, c = self.state
        if random.random() < self.slip_prob:
            action = random.choice([0, 1, 2, 3])

        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        else:
            c += 1

        r = max(0, min(self.size - 1, r))
        c = max(0, min(self.size - 1, c))
        self.state = (r, c)

        done = self.state == self.goal
        reward = 20.0 if done else -0.1
        if self.state == self.risky:
            reward -= 1.0
        return self.state, reward, done


def eps_greedy(Q: Dict[Tuple[State, Action], float], s: State, epsilon: float) -> Action:
    if random.random() < epsilon:
        return random.randint(0, 3)
    return max(range(4), key=lambda a: Q.get((s, a), 0.0))


def train_q_learning(env: RiskyGrid, episodes=1200, alpha=0.15, gamma=0.97, epsilon=0.2):
    Q: Dict[Tuple[State, Action], float] = {}
    rewards = []

    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(250):
            a = eps_greedy(Q, s, epsilon)
            s2, r, done = env.step(a)
            total += r

            best_next = max(Q.get((s2, na), 0.0) for na in range(4))
            old = Q.get((s, a), 0.0)
            Q[(s, a)] = old + alpha * (r + gamma * best_next - old)

            s = s2
            if done:
                break
        rewards.append(total)
    return Q, rewards


def train_sarsa(env: RiskyGrid, episodes=1200, alpha=0.15, gamma=0.97, epsilon=0.2):
    Q: Dict[Tuple[State, Action], float] = {}
    rewards = []

    for _ in range(episodes):
        s = env.reset()
        a = eps_greedy(Q, s, epsilon)
        total = 0.0

        for _ in range(250):
            s2, r, done = env.step(a)
            total += r
            a2 = eps_greedy(Q, s2, epsilon)

            old = Q.get((s, a), 0.0)
            target = r + gamma * Q.get((s2, a2), 0.0)
            Q[(s, a)] = old + alpha * (target - old)

            s, a = s2, a2
            if done:
                break

        rewards.append(total)

    return Q, rewards


def smooth(vals: List[float], w: int = 50) -> List[float]:
    out = []
    for i in range(len(vals)):
        j = max(0, i - w + 1)
        out.append(sum(vals[j : i + 1]) / (i - j + 1))
    return out


def maybe_plot(q_vals: List[float], s_vals: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Final avg reward Q-learning:", round(sum(q_vals[-100:]) / 100, 4))
        print("Final avg reward SARSA    :", round(sum(s_vals[-100:]) / 100, 4))
        return

    plt.figure(figsize=(8, 4))
    plt.plot(smooth(q_vals), label="Q-learning")
    plt.plot(smooth(s_vals), label="SARSA")
    plt.title("Reward Curves: Q-learning vs SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.legend()
    plt.tight_layout()
    plt.show()


def linear_dqn_intuition_step(weights: List[float], features: List[float], target: float, lr: float = 0.05):
    pred = sum(w * x for w, x in zip(weights, features))
    err = pred - target
    for i in range(len(weights)):
        weights[i] -= lr * err * features[i]
    return pred, err


if __name__ == "__main__":
    random.seed(42)
    env = RiskyGrid(size=6, slip_prob=0.18)

    Q_q, rewards_q = train_q_learning(env)
    Q_s, rewards_s = train_sarsa(env)

    print("Q-learning final avg reward:", round(sum(rewards_q[-100:]) / 100, 4))
    print("SARSA final avg reward    :", round(sum(rewards_s[-100:]) / 100, 4))

    maybe_plot(rewards_q, rewards_s)

    # DQN intuition: linear approximation update demo
    w = [0.1, -0.05, 0.02]
    x = [1.0, 0.7, -0.2]
    target_q = 1.4
    for step in range(1, 6):
        pred, err = linear_dqn_intuition_step(w, x, target_q)
        print(f"linear_dqn_step={step} pred={pred:.4f} err={err:.4f} w={[round(v,4) for v in w]}")
