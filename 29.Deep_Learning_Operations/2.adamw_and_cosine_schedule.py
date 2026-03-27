from __future__ import annotations

import math
from typing import List


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


class AdamW1D:
    def __init__(self, n_params: int, lr: float = 1e-3, wd: float = 1e-2, b1: float = 0.9, b2: float = 0.999):
        self.lr = lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.eps = 1e-8
        self.m = [0.0] * n_params
        self.v = [0.0] * n_params
        self.t = 0

    def step(self, params: List[float], grads: List[float], lr: float) -> List[float]:
        self.t += 1
        for i in range(len(params)):
            g = grads[i]
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            params[i] -= lr * (m_hat / (math.sqrt(v_hat) + self.eps) + self.wd * params[i])
        return params


if __name__ == "__main__":
    params = [1.5, -2.0]
    target = [0.0, 0.0]
    opt = AdamW1D(n_params=2)

    total_steps = 200
    for step in range(total_steps):
        grads = [2 * (p - t) for p, t in zip(params, target)]
        lr = cosine_lr(step, total_steps, warmup_steps=20, base_lr=0.05, min_lr=0.001)
        params = opt.step(params, grads, lr)

        if step % 25 == 0:
            print(f"step={step:03d} lr={lr:.4f} params={[round(x, 4) for x in params]}")
