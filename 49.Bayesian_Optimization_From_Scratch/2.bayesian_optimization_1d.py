from __future__ import annotations

import math
import random
from typing import List, Tuple


# ----- basic linear algebra (small matrices) -----
def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*m)]


def inv_matrix(a: List[List[float]]) -> List[List[float]]:
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]

    for i in range(n):
        # pivot
        pivot = i
        for r in range(i + 1, n):
            if abs(aug[r][i]) > abs(aug[pivot][i]):
                pivot = r
        aug[i], aug[pivot] = aug[pivot], aug[i]

        piv = aug[i][i]
        if abs(piv) < 1e-12:
            raise ValueError("Singular matrix")
        for j in range(2 * n):
            aug[i][j] /= piv

        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for j in range(2 * n):
                aug[r][j] -= factor * aug[i][j]

    return [row[n:] for row in aug]


# ----- GP -----
def rbf(x1: float, x2: float, lengthscale: float = 0.25, variance: float = 1.0) -> float:
    return variance * math.exp(-0.5 * ((x1 - x2) / lengthscale) ** 2)


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class GP1D:
    def __init__(self, noise: float = 1e-4, lengthscale: float = 0.25, variance: float = 1.0):
        self.noise = noise
        self.lengthscale = lengthscale
        self.variance = variance
        self.X: List[float] = []
        self.y: List[float] = []
        self.K_inv: List[List[float]] = []

    def fit(self, X: List[float], y: List[float]):
        self.X = X[:]
        self.y = y[:]
        n = len(X)
        K = [[rbf(X[i], X[j], self.lengthscale, self.variance) for j in range(n)] for i in range(n)]
        for i in range(n):
            K[i][i] += self.noise
        self.K_inv = inv_matrix(K)

    def predict(self, x: float) -> Tuple[float, float]:
        k = [rbf(x, xi, self.lengthscale, self.variance) for xi in self.X]
        alpha = matvec(self.K_inv, self.y)
        mu = sum(ki * ai for ki, ai in zip(k, alpha))

        v = matvec(self.K_inv, k)
        kxx = rbf(x, x, self.lengthscale, self.variance) + self.noise
        var = max(1e-12, kxx - sum(ki * vi for ki, vi in zip(k, v)))
        return mu, var


def expected_improvement(mu: float, sigma: float, best: float) -> float:
    if sigma < 1e-12:
        return 0.0
    z = (best - mu) / sigma
    return (best - mu) * normal_cdf(z) + sigma * normal_pdf(z)


def objective(x: float) -> float:
    # Unknown expensive function (simulated)
    noise = random.uniform(-0.01, 0.01)
    return (x - 0.72) ** 2 + 0.1 * math.sin(10 * x) + noise


def maybe_plot(history_x, history_y, grid, mu, sigma):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        best_idx = min(range(len(history_y)), key=lambda i: history_y[i])
        print("Best x:", round(history_x[best_idx], 4), "best y:", round(history_y[best_idx], 5))
        return

    plt.figure(figsize=(9, 4))
    plt.plot(grid, mu, label="GP mean")
    upper = [m + 2 * s for m, s in zip(mu, sigma)]
    lower = [m - 2 * s for m, s in zip(mu, sigma)]
    plt.fill_between(grid, lower, upper, alpha=0.2, label="±2σ")
    plt.scatter(history_x, history_y, c="red", label="observations")
    plt.title("Bayesian Optimization Surrogate")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    # initial design
    X = [0.05, 0.3, 0.55, 0.9]
    y = [objective(x) for x in X]

    gp = GP1D(noise=1e-4, lengthscale=0.18, variance=0.8)

    budget = 20
    for step in range(budget):
        gp.fit(X, y)
        best = min(y)

        grid = [i / 300 for i in range(301)]
        eis = []
        for x in grid:
            mu, var = gp.predict(x)
            sigma = math.sqrt(var)
            eis.append(expected_improvement(mu, sigma, best))

        x_next = grid[max(range(len(grid)), key=lambda i: eis[i])]
        y_next = objective(x_next)
        X.append(x_next)
        y.append(y_next)

    best_idx = min(range(len(y)), key=lambda i: y[i])
    print("Best observed x:", round(X[best_idx], 4))
    print("Best observed y:", round(y[best_idx], 6))

    grid = [i / 300 for i in range(301)]
    mu_sigma = [gp.predict(x) for x in grid]
    mu = [m for m, _ in mu_sigma]
    sigma = [math.sqrt(v) for _, v in mu_sigma]
    maybe_plot(X, y, grid, mu, sigma)
