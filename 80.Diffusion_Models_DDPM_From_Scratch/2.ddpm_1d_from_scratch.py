from __future__ import annotations

import math
import random
from typing import List, Tuple


def make_real_data(n: int = 8000, seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.4:
            out.append(rng.gauss(-2.0, 0.55))
        elif r < 0.8:
            out.append(rng.gauss(1.2, 0.45))
        else:
            out.append(rng.gauss(3.0, 0.35))
    return out


def build_schedule(T: int = 60, beta_start: float = 0.0008, beta_end: float = 0.035):
    betas = [0.0] * (T + 1)
    alphas = [1.0] * (T + 1)
    alpha_bar = [1.0] * (T + 1)

    for t in range(1, T + 1):
        frac = (t - 1) / max(1, T - 1)
        beta_t = beta_start + frac * (beta_end - beta_start)
        betas[t] = beta_t
        alphas[t] = 1.0 - beta_t
        alpha_bar[t] = alpha_bar[t - 1] * alphas[t]

    return betas, alphas, alpha_bar


def forward_sample(x0: float, t: int, alpha_bar: List[float], rng: random.Random) -> Tuple[float, float]:
    eps = rng.gauss(0.0, 1.0)
    xt = math.sqrt(alpha_bar[t]) * x0 + math.sqrt(1.0 - alpha_bar[t]) * eps
    return xt, eps


def train_linear_denoiser(
    data: List[float],
    T: int,
    alpha_bar: List[float],
    steps: int = 120000,
    lr: float = 0.012,
    weight_decay: float = 1e-5,
    seed: int = 11,
):
    rng = random.Random(seed)

    # epsilon_hat_t(x_t) = a_t * x_t + b_t
    a = [0.0] * (T + 1)
    b = [0.0] * (T + 1)

    loss_curve = []
    running = 0.0

    for step in range(1, steps + 1):
        x0 = data[rng.randint(0, len(data) - 1)]
        t = rng.randint(1, T)

        xt, eps = forward_sample(x0, t, alpha_bar, rng)
        pred = a[t] * xt + b[t]
        err = pred - eps
        loss = err * err

        grad_a = 2.0 * err * xt + weight_decay * a[t]
        grad_b = 2.0 * err + weight_decay * b[t]

        a[t] -= lr * grad_a
        b[t] -= lr * grad_b

        running += loss
        if step % 1000 == 0:
            loss_curve.append(running / 1000.0)
            running = 0.0

    return a, b, loss_curve


def reverse_sample(
    n_samples: int,
    T: int,
    betas: List[float],
    alphas: List[float],
    alpha_bar: List[float],
    a: List[float],
    b: List[float],
    seed: int = 21,
):
    rng = random.Random(seed)
    xs = [rng.gauss(0.0, 1.0) for _ in range(n_samples)]

    n_trace = min(8, n_samples)
    traces = [[xs[i]] for i in range(n_trace)]

    for t in range(T, 0, -1):
        next_x = []
        for i, xt in enumerate(xs):
            eps_hat = a[t] * xt + b[t]
            coef = betas[t] / math.sqrt(max(1e-12, 1.0 - alpha_bar[t]))
            mean = (xt - coef * eps_hat) / math.sqrt(alphas[t])

            if t > 1:
                z = rng.gauss(0.0, 1.0)
                x_prev = mean + math.sqrt(betas[t]) * z
            else:
                x_prev = mean

            next_x.append(x_prev)
            if i < n_trace:
                traces[i].append(x_prev)

        xs = next_x

    return xs, traces


def mean_std(x: List[float]) -> Tuple[float, float]:
    m = sum(x) / len(x)
    v = sum((v - m) ** 2 for v in x) / len(x)
    return m, math.sqrt(v)


def histogram_density(x: List[float], low: float, high: float, bins: int = 80) -> List[float]:
    width = (high - low) / bins
    h = [0.0] * bins
    for v in x:
        idx = int((v - low) / width)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        h[idx] += 1.0

    total = sum(h)
    if total == 0:
        return h
    return [v / total for v in h]


def histogram_overlap(real: List[float], gen: List[float], low: float, high: float, bins: int = 80) -> float:
    hr = histogram_density(real, low, high, bins)
    hg = histogram_density(gen, low, high, bins)
    return sum(min(a, b) for a, b in zip(hr, hg))


def maybe_plot(loss_curve: List[float], real: List[float], gen: List[float], traces: List[List[float]]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not installed. Skipping plots.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))

    xs = [1000 * (i + 1) for i in range(len(loss_curve))]
    axs[0].plot(xs, loss_curve, color="#1d4ed8")
    axs[0].set_title("Denoiser Training Loss")
    axs[0].set_xlabel("SGD step")
    axs[0].set_ylabel("MSE")

    axs[1].hist(real, bins=80, density=True, alpha=0.55, label="real", color="#9ca3af")
    axs[1].hist(gen, bins=80, density=True, alpha=0.55, label="generated", color="#1d4ed8")
    axs[1].set_title("Real vs Generated Distribution")
    axs[1].legend()

    for tr in traces:
        axs[2].plot(range(len(tr)), tr, alpha=0.75)
    axs[2].set_title("Reverse Trajectories (x_T -> x_0)")
    axs[2].set_xlabel("Reverse step")
    axs[2].set_ylabel("x")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    T = 60
    real_data = make_real_data(n=8000, seed=5)

    betas, alphas, alpha_bar = build_schedule(T=T, beta_start=0.0008, beta_end=0.035)

    a, b, loss_curve = train_linear_denoiser(
        data=real_data,
        T=T,
        alpha_bar=alpha_bar,
        steps=120000,
        lr=0.012,
        weight_decay=1e-5,
        seed=9,
    )

    generated, traces = reverse_sample(
        n_samples=8000,
        T=T,
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        a=a,
        b=b,
        seed=19,
    )

    m_real, s_real = mean_std(real_data)
    m_gen, s_gen = mean_std(generated)
    overlap = histogram_overlap(real_data, generated, low=-5.5, high=5.5, bins=90)

    print("Real moments:", {"mean": round(m_real, 5), "std": round(s_real, 5)})
    print("Generated moments:", {"mean": round(m_gen, 5), "std": round(s_gen, 5)})
    print("Histogram overlap (higher is better, max=1):", round(overlap, 5))

    maybe_plot(loss_curve, real_data, generated, traces)
