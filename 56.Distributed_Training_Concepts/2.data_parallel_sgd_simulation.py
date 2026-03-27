from __future__ import annotations

import random
from typing import List, Tuple


def make_data(n: int = 3000, seed: int = 42):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-2, 2)
        noise = random.uniform(-0.2, 0.2)
        target = 3.2 * x1 - 1.7 * x2 + 0.5 + noise
        X.append([x1, x2])
        y.append(target)
    return X, y


def mse(X: List[List[float]], y: List[float], w: List[float]) -> float:
    s = 0.0
    for xi, yi in zip(X, y):
        pred = w[0] + w[1] * xi[0] + w[2] * xi[1]
        s += (pred - yi) ** 2
    return s / len(X)


def grad_batch(Xb: List[List[float]], yb: List[float], w: List[float]) -> List[float]:
    g0 = g1 = g2 = 0.0
    n = len(Xb)
    for xi, yi in zip(Xb, yb):
        pred = w[0] + w[1] * xi[0] + w[2] * xi[1]
        err = pred - yi
        g0 += 2 * err
        g1 += 2 * err * xi[0]
        g2 += 2 * err * xi[1]
    return [g0 / n, g1 / n, g2 / n]


def train_single_worker(X, y, epochs=40, batch_size=64, lr=0.05):
    w = [0.0, 0.0, 0.0]
    losses = []
    n = len(X)
    idx = list(range(n))

    for _ in range(epochs):
        random.shuffle(idx)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            Xb = [X[j] for j in b]
            yb = [y[j] for j in b]
            g = grad_batch(Xb, yb, w)
            for k in range(3):
                w[k] -= lr * g[k]
        losses.append(mse(X, y, w))
    return w, losses


def train_sync_data_parallel(X, y, workers=4, epochs=40, batch_per_worker=32, lr=0.05):
    w = [0.0, 0.0, 0.0]
    losses = []
    n = len(X)
    idx = list(range(n))

    global_step_batch = workers * batch_per_worker

    for _ in range(epochs):
        random.shuffle(idx)
        for i in range(0, n, global_step_batch):
            chunk = idx[i : i + global_step_batch]
            if len(chunk) < workers:
                continue

            grads = []
            for wid in range(workers):
                start = wid * batch_per_worker
                end = start + batch_per_worker
                b = chunk[start:end]
                if not b:
                    continue
                Xb = [X[j] for j in b]
                yb = [y[j] for j in b]
                grads.append(grad_batch(Xb, yb, w))

            if not grads:
                continue

            g_avg = [sum(g[k] for g in grads) / len(grads) for k in range(3)]
            for k in range(3):
                w[k] -= lr * g_avg[k]

        losses.append(mse(X, y, w))
    return w, losses


def train_async_style(X, y, workers=4, epochs=40, batch_per_worker=32, lr=0.05, staleness=2):
    # Simulate stale gradients by applying delayed gradients from old weights.
    w = [0.0, 0.0, 0.0]
    losses = []
    n = len(X)
    idx = list(range(n))

    # queue of delayed grads
    pending: List[List[float]] = []

    global_step_batch = workers * batch_per_worker

    for _ in range(epochs):
        random.shuffle(idx)
        for i in range(0, n, global_step_batch):
            chunk = idx[i : i + global_step_batch]
            if len(chunk) < workers:
                continue

            # workers compute grads on current stale snapshot
            snapshot = w[:]
            for wid in range(workers):
                start = wid * batch_per_worker
                end = start + batch_per_worker
                b = chunk[start:end]
                if not b:
                    continue
                Xb = [X[j] for j in b]
                yb = [y[j] for j in b]
                pending.append(grad_batch(Xb, yb, snapshot))

            # apply up to staleness gradients each round
            for _apply in range(min(staleness, len(pending))):
                g = pending.pop(0)
                # Clip stale gradients for numerical stability in this toy simulation.
                g = [max(-1.0, min(1.0, x)) for x in g]
                for k in range(3):
                    w[k] -= lr * g[k]

        losses.append(mse(X, y, w))
    return w, losses


def maybe_plot(ls1, ls2, ls3):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("epoch single sync async")
        for i, (a, b, c) in enumerate(zip(ls1, ls2, ls3), 1):
            if i % max(1, len(ls1) // 10) == 0:
                print(i, round(a, 5), round(b, 5), round(c, 5))
        return

    plt.figure(figsize=(8, 4))
    plt.plot(ls1, label="single-worker")
    plt.plot(ls2, label="sync-data-parallel")
    plt.plot(ls3, label="async-style")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log)")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = make_data(n=2500, seed=8)

    w1, l1 = train_single_worker(X, y, epochs=35, batch_size=64, lr=0.04)
    w2, l2 = train_sync_data_parallel(X, y, workers=4, epochs=35, batch_per_worker=16, lr=0.04)
    w3, l3 = train_async_style(X, y, workers=4, epochs=35, batch_per_worker=16, lr=0.005, staleness=2)

    print("Final MSE single:", round(l1[-1], 6), "weights:", [round(v, 3) for v in w1])
    print("Final MSE sync  :", round(l2[-1], 6), "weights:", [round(v, 3) for v in w2])
    print("Final MSE async :", round(l3[-1], 6), "weights:", [round(v, 3) for v in w3])

    maybe_plot(l1, l2, l3)
