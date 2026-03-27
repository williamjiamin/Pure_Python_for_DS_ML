from __future__ import annotations

from typing import List, Tuple


def forward_algorithm(pi, A, B, obs: List[int]) -> float:
    n_states = len(pi)
    alpha = [pi[s] * B[s][obs[0]] for s in range(n_states)]

    for t in range(1, len(obs)):
        new_alpha = [0.0] * n_states
        for j in range(n_states):
            s = 0.0
            for i in range(n_states):
                s += alpha[i] * A[i][j]
            new_alpha[j] = s * B[j][obs[t]]
        alpha = new_alpha

    return sum(alpha)


def viterbi_decode(pi, A, B, obs: List[int]) -> Tuple[float, List[int]]:
    n_states = len(pi)
    T = len(obs)

    dp = [[0.0] * n_states for _ in range(T)]
    bp = [[0] * n_states for _ in range(T)]

    for s in range(n_states):
        dp[0][s] = pi[s] * B[s][obs[0]]

    for t in range(1, T):
        for j in range(n_states):
            best_prob, best_state = -1.0, 0
            for i in range(n_states):
                p = dp[t - 1][i] * A[i][j]
                if p > best_prob:
                    best_prob = p
                    best_state = i
            dp[t][j] = best_prob * B[j][obs[t]]
            bp[t][j] = best_state

    last_state = max(range(n_states), key=lambda s: dp[T - 1][s])
    best_prob = dp[T - 1][last_state]

    path = [last_state]
    for t in range(T - 1, 0, -1):
        path.append(bp[t][path[-1]])
    path.reverse()

    return best_prob, path


if __name__ == "__main__":
    # 0=Rainy, 1=Sunny
    pi = [0.6, 0.4]
    A = [
        [0.7, 0.3],
        [0.4, 0.6],
    ]
    # emissions: 0=walk,1=shop,2=clean
    B = [
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1],
    ]

    obs = [0, 1, 2, 2, 1, 0]

    likelihood = forward_algorithm(pi, A, B, obs)
    best_prob, path = viterbi_decode(pi, A, B, obs)

    print("Sequence likelihood:", round(likelihood, 8))
    print("Viterbi probability:", round(best_prob, 8))
    print("Best hidden path:", path)
