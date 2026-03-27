from __future__ import annotations

import random
from typing import Dict, List, Tuple


class GridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start

    def reset(self) -> Tuple[int, int]:
        self.state = self.start
        return self.state

    def step(self, action: int):
        # actions: 0=up,1=down,2=left,3=right
        r, c = self.state
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1

        r = max(0, min(self.size - 1, r))
        c = max(0, min(self.size - 1, c))
        self.state = (r, c)

        done = self.state == self.goal
        reward = 10.0 if done else -0.1
        return self.state, reward, done


def train_q_learning(
    env: GridWorld,
    episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.2,
) -> Dict[Tuple[Tuple[int, int], int], float]:
    actions = [0, 1, 2, 3]
    Q: Dict[Tuple[Tuple[int, int], int], float] = {}

    def q(s, a):
        return Q.get((s, a), 0.0)

    for _ in range(episodes):
        s = env.reset()
        for _step in range(200):
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                a = max(actions, key=lambda x: q(s, x))

            s_next, r, done = env.step(a)
            best_next = max(q(s_next, na) for na in actions)

            old = q(s, a)
            new = old + alpha * (r + gamma * best_next - old)
            Q[(s, a)] = new

            s = s_next
            if done:
                break

    return Q


def greedy_policy(Q, size=5):
    actions = [0, 1, 2, 3]
    policy = {}
    for r in range(size):
        for c in range(size):
            s = (r, c)
            policy[s] = max(actions, key=lambda a: Q.get((s, a), 0.0))
    return policy


if __name__ == "__main__":
    env = GridWorld(size=5)
    Q = train_q_learning(env, episodes=3000, alpha=0.15, gamma=0.95, epsilon=0.2)
    policy = greedy_policy(Q, size=5)

    arrow = {0: "U", 1: "D", 2: "L", 3: "R"}
    print("Greedy policy:")
    for r in range(5):
        row = []
        for c in range(5):
            if (r, c) == env.goal:
                row.append("G")
            else:
                row.append(arrow[policy[(r, c)]])
        print(" ".join(row))
