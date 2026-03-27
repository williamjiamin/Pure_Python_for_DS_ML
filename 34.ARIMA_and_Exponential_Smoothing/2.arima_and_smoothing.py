from __future__ import annotations

from typing import List

import numpy as np


def difference(series: List[float], d: int) -> List[float]:
    out = series[:]
    for _ in range(d):
        out = [out[i] - out[i - 1] for i in range(1, len(out))]
    return out


def invert_difference(history: List[float], diff_forecasts: List[float], d: int) -> List[float]:
    if d == 0:
        return diff_forecasts[:]
    preds = []
    base = history[-1]
    for val in diff_forecasts:
        base = base + val
        preds.append(base)
    return preds


class ARIMAZeroMA:
    """ARIMA(p,d,0) educational implementation."""

    def __init__(self, p: int = 3, d: int = 1):
        self.p = p
        self.d = d
        self.coef = None

    def fit(self, series: List[float]) -> None:
        ds = difference(series, self.d)
        if len(ds) <= self.p:
            raise ValueError("Series too short for given p and d.")

        X, y = [], []
        for t in range(self.p, len(ds)):
            X.append(ds[t - self.p : t])
            y.append(ds[t])
        X = np.asarray(X)
        y = np.asarray(y)

        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    def forecast(self, series: List[float], horizon: int) -> List[float]:
        ds = difference(series, self.d)
        hist = ds[:]
        diff_preds = []

        for _ in range(horizon):
            x = np.asarray(hist[-self.p :])
            pred = float(x @ self.coef)
            hist.append(pred)
            diff_preds.append(pred)

        return invert_difference(series, diff_preds, self.d)


class SimpleExponentialSmoothing:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.level = None

    def fit(self, series: List[float]) -> None:
        level = series[0]
        for y in series[1:]:
            level = self.alpha * y + (1 - self.alpha) * level
        self.level = level

    def forecast(self, horizon: int) -> List[float]:
        return [self.level] * horizon


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


if __name__ == "__main__":
    series = [10, 12, 13, 16, 18, 20, 23, 25, 27, 29, 31, 33]
    train, test = series[:-3], series[-3:]

    ar = ARIMAZeroMA(p=2, d=1)
    ar.fit(train)
    ar_pred = ar.forecast(train, horizon=3)

    ses = SimpleExponentialSmoothing(alpha=0.4)
    ses.fit(train)
    ses_pred = ses.forecast(3)

    print("ARIMA-like pred:", [round(v, 3) for v in ar_pred], "MAE:", round(mae(test, ar_pred), 3))
    print("SES pred      :", [round(v, 3) for v in ses_pred], "MAE:", round(mae(test, ses_pred), 3))
