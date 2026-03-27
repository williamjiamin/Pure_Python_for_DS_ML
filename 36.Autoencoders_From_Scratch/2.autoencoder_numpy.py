from __future__ import annotations

import numpy as np


class Autoencoder:
    def __init__(self, in_dim: int, latent_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.2, size=(in_dim, latent_dim))
        self.b1 = np.zeros((1, latent_dim))
        self.W2 = rng.normal(0, 0.2, size=(latent_dim, in_dim))
        self.b2 = np.zeros((1, in_dim))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        h_pre = X @ self.W1 + self.b1
        h = self.sigmoid(h_pre)
        X_hat = h @ self.W2 + self.b2
        cache = (X, h_pre, h, X_hat)
        return X_hat, cache

    @staticmethod
    def mse_loss(X, X_hat):
        return np.mean((X - X_hat) ** 2)

    def backward(self, cache):
        X, h_pre, h, X_hat = cache
        n = X.shape[0]

        dX_hat = 2.0 * (X_hat - X) / n
        dW2 = h.T @ dX_hat
        db2 = np.sum(dX_hat, axis=0, keepdims=True)

        dh = dX_hat @ self.W2.T
        dh_pre = dh * h * (1 - h)
        dW1 = X.T @ dh_pre
        db1 = np.sum(dh_pre, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, grads, lr=0.05, weight_decay=1e-4):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * (dW1 + weight_decay * self.W1)
        self.b1 -= lr * db1
        self.W2 -= lr * (dW2 + weight_decay * self.W2)
        self.b2 -= lr * db2


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=(n, 1))
    x2 = 0.7 * x1 + rng.normal(0, 0.2, size=(n, 1))
    x3 = -0.4 * x1 + 0.5 * x2 + rng.normal(0, 0.2, size=(n, 1))
    X = np.hstack([x1, x2, x3])
    return X


if __name__ == "__main__":
    X = make_data(400)
    model = Autoencoder(in_dim=3, latent_dim=2)

    for epoch in range(1, 501):
        X_hat, cache = model.forward(X)
        loss = model.mse_loss(X, X_hat)
        grads = model.backward(cache)
        model.step(grads, lr=0.05)

        if epoch % 50 == 0:
            print(f"epoch={epoch:03d} recon_loss={loss:.6f}")

    # anomaly score example
    X_noisy = X.copy()
    X_noisy[:10] += 2.5
    X_hat_noisy, _ = model.forward(X_noisy)
    errors = np.mean((X_noisy - X_hat_noisy) ** 2, axis=1)
    print("Top anomaly scores:", np.round(np.sort(errors)[-10:], 4))
