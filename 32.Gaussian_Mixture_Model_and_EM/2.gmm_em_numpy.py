from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class GaussianMixtureEM:
    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.pi = None
        self.mu = None
        self.var = None  # diagonal covariance (variance vector)

    def _init_params(self, X: np.ndarray) -> None:
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=self.n_components, replace=False)

        self.pi = np.full(self.n_components, 1.0 / self.n_components)
        self.mu = X[idx].copy()
        self.var = np.tile(np.var(X, axis=0, keepdims=True), (self.n_components, 1)) + 1e-6

    @staticmethod
    def _gaussian_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        det = np.prod(var)
        norm = 1.0 / math.sqrt((2 * math.pi) ** d * det)
        diff = X - mu
        expo = -0.5 * np.sum((diff ** 2) / var, axis=1)
        return norm * np.exp(expo)

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        gamma = np.zeros((n, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.pi[k] * self._gaussian_diag(X, self.mu[k], self.var[k])

        gamma_sum = np.sum(gamma, axis=1, keepdims=True) + 1e-12
        gamma /= gamma_sum
        return gamma

    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        n, d = X.shape
        Nk = np.sum(gamma, axis=0) + 1e-12

        self.pi = Nk / n
        self.mu = (gamma.T @ X) / Nk[:, None]

        var = np.zeros((self.n_components, d))
        for k in range(self.n_components):
            diff = X - self.mu[k]
            var[k] = np.sum(gamma[:, [k]] * (diff ** 2), axis=0) / Nk[k]
        self.var = var + 1e-6

    def _log_likelihood(self, X: np.ndarray) -> float:
        probs = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            probs[:, k] = self.pi[k] * self._gaussian_diag(X, self.mu[k], self.var[k])
        return float(np.sum(np.log(np.sum(probs, axis=1) + 1e-12)))

    def fit(self, X: np.ndarray) -> "GaussianMixtureEM":
        self._init_params(X)
        prev_ll = -float("inf")

        for _ in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)
            ll = self._log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def make_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0.0, 0.0], scale=[0.5, 0.7], size=(120, 2))
    b = rng.normal(loc=[3.5, 3.0], scale=[0.6, 0.6], size=(120, 2))
    return np.vstack([a, b])


if __name__ == "__main__":
    X = make_data()
    gmm = GaussianMixtureEM(n_components=2, max_iter=200, tol=1e-5)
    gmm.fit(X)
    labels = gmm.predict(X)

    print("Mixture weights:", np.round(gmm.pi, 4))
    print("Means:\n", np.round(gmm.mu, 4))
    print("Cluster counts:", np.bincount(labels))
