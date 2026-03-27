from __future__ import annotations

import math
import random
from typing import List


def quantile_bins(values: List[float], n_bins: int = 10) -> List[float]:
    s = sorted(values)
    if len(s) < n_bins:
        return sorted(set(s))
    bins = []
    for i in range(1, n_bins):
        idx = int(i * (len(s) - 1) / n_bins)
        bins.append(s[idx])
    return sorted(set(bins))


def hist_proportions(values: List[float], cut_points: List[float]) -> List[float]:
    bins = [0] * (len(cut_points) + 1)
    for v in values:
        idx = 0
        while idx < len(cut_points) and v > cut_points[idx]:
            idx += 1
        bins[idx] += 1
    n = len(values)
    return [b / n for b in bins]


def psi(ref: List[float], cur: List[float], n_bins: int = 10) -> float:
    cut_points = quantile_bins(ref, n_bins=n_bins)
    p = hist_proportions(ref, cut_points)
    q = hist_proportions(cur, cut_points)
    out = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, 1e-6)
        qi = max(qi, 1e-6)
        out += (qi - pi) * math.log(qi / pi)
    return out


def status(v: float) -> str:
    if v < 0.1:
        return "stable"
    if v < 0.2:
        return "moderate"
    return "major"


def maybe_plot(values: List[float], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(title, [round(v, 4) for v in values])
        return

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.axhline(0.1, color="orange", linestyle="--", label="0.1")
    plt.axhline(0.2, color="red", linestyle="--", label="0.2")
    plt.title(title)
    plt.xlabel("Window")
    plt.ylabel("PSI")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random.seed(42)

    ref = [random.gauss(0.0, 1.0) for _ in range(1500)]

    windows = []
    for month in range(1, 13):
        mean_shift = (month - 1) * 0.08
        std_scale = 1.0 + (month - 1) * 0.02
        cur = [random.gauss(mean_shift, std_scale) for _ in range(800)]
        windows.append(cur)

    psi_values = [psi(ref, cur, n_bins=10) for cur in windows]

    print("Monthly drift report:")
    for i, v in enumerate(psi_values, 1):
        print(f"month={i:02d} psi={v:.4f} status={status(v)}")

    maybe_plot(psi_values, "Monthly PSI Trend")
