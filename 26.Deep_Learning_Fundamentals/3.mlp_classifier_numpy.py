"""Numpy MLP classifier with manual forward/backward.
This is a bridge from pure Python math to modern DL training loops.
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, out_dim))
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        ex = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        cache = (X, z1, a1, probs)
        return probs, cache

    @staticmethod
    def loss_and_grad(probs: np.ndarray, y: np.ndarray):
        n = len(y)
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n
        return loss, dlogits

    def backward(self, cache, dlogits):
        X, z1, a1, _ = cache
        dW2 = a1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        da1 = dlogits @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, grads, lr: float, weight_decay: float = 0.0):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * (dW1 + weight_decay * self.W1)
        self.b1 -= lr * db1
        self.W2 -= lr * (dW2 + weight_decay * self.W2)
        self.b2 -= lr * db2


def make_toy_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = model.forward(X)
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))


if __name__ == "__main__":
    X, y = make_toy_data(600)
    split = 500
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = MLP(in_dim=2, hidden=32, out_dim=2)
    lr = 0.05

    for epoch in range(1, 401):
        probs, cache = model.forward(X_train)
        loss, dlogits = model.loss_and_grad(probs, y_train)
        grads = model.backward(cache, dlogits)
        model.step(grads, lr=lr, weight_decay=1e-4)

        if epoch % 40 == 0:
            train_acc = accuracy(model, X_train, y_train)
            val_acc = accuracy(model, X_val, y_val)
            print(f"epoch={epoch:03d} loss={loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
