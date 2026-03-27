from __future__ import annotations

import math
import random
from typing import List, Tuple


def make_series(n: int = 450, seed: int = 42) -> List[float]:
    random.seed(seed)
    s = []
    for t in range(n):
        trend = 0.02 * t
        season = 1.8 * math.sin(2 * math.pi * t / 24)
        noise = random.gauss(0, 0.6 + 0.002 * t)  # mild heteroskedastic drift
        s.append(20 + trend + season + noise)
    return s


def fit_ar1(y: List[float]) -> Tuple[float, float]:
    # y_t = a + b y_{t-1}
    x = y[:-1]
    z = y[1:]
    n = len(x)

    mx = sum(x) / n
    mz = sum(z) / n
    cov = sum((xi - mx) * (zi - mz) for xi, zi in zip(x, z))
    var = sum((xi - mx) ** 2 for xi in x) + 1e-12
    b = cov / var
    a = mz - b * mx
    return a, b


def predict_ar1(a: float, b: float, prev: float, horizon: int) -> List[float]:
    out = []
    cur = prev
    for _ in range(horizon):
        cur = a + b * cur
        out.append(cur)
    return out


def quantile(values: List[float], q: float) -> float:
    s = sorted(values)
    idx = int(q * (len(s) - 1))
    return s[idx]


def build_intervals(preds: List[float], residuals: List[float], alpha: float = 0.1):
    lo_q = quantile(residuals, alpha / 2)
    hi_q = quantile(residuals, 1 - alpha / 2)
    lower = [p + lo_q for p in preds]
    upper = [p + hi_q for p in preds]
    return lower, upper


def coverage(y_true: List[float], lower: List[float], upper: List[float]) -> float:
    c = 0
    for y, lo, hi in zip(y_true, lower, upper):
        if lo <= y <= hi:
            c += 1
    return c / len(y_true)


def mean_width(lower: List[float], upper: List[float]) -> float:
    return sum(hi - lo for lo, hi in zip(lower, upper)) / len(lower)


def maybe_plot(y_true, preds, lower, upper):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("first 10 forecast rows:")
        for i in range(min(10, len(y_true))):
            print(i, round(y_true[i], 3), round(preds[i], 3), round(lower[i], 3), round(upper[i], 3))
        return

    plt.figure(figsize=(9, 4))
    xs = list(range(len(y_true)))
    plt.plot(xs, y_true, label="actual")
    plt.plot(xs, preds, label="forecast")
    plt.fill_between(xs, lower, upper, alpha=0.2, label="interval")
    plt.title("Forecast with Prediction Intervals")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    y = make_series(n=460, seed=9)

    train = y[:300]
    calib = y[300:380]
    test = y[380:]

    a, b = fit_ar1(train)

    # Calibration residuals
    calib_preds = []
    prev = train[-1]
    for actual in calib:
        p = a + b * prev
        calib_preds.append(p)
        prev = actual

    residuals = [actual - p for actual, p in zip(calib, calib_preds)]

    # Test rolling forecasts
    test_preds = []
    prev = calib[-1]
    for actual in test:
        p = a + b * prev
        test_preds.append(p)
        prev = actual

    for alpha in [0.2, 0.1, 0.05]:
        lo, hi = build_intervals(test_preds, residuals, alpha=alpha)
        cov = coverage(test, lo, hi)
        width = mean_width(lo, hi)
        print(f"Interval {int((1-alpha)*100)}% -> coverage={cov:.4f}, mean_width={width:.4f}")

    lo90, hi90 = build_intervals(test_preds, residuals, alpha=0.1)
    maybe_plot(test, test_preds, lo90, hi90)
