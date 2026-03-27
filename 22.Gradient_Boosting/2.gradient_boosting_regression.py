"""Tiny gradient boosting regressor with decision stumps.
Pure Python educational implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Stump:
    feature_idx: int
    threshold: float
    left_value: float
    right_value: float

    def predict_one(self, x: Sequence[float]) -> float:
        return self.left_value if x[self.feature_idx] <= self.threshold else self.right_value


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def fit_stump(X: List[List[float]], residuals: List[float]) -> Stump:
    n_features = len(X[0])
    best = None
    best_sse = float("inf")

    for j in range(n_features):
        thresholds = sorted({row[j] for row in X})
        for t in thresholds:
            left = [residuals[i] for i, row in enumerate(X) if row[j] <= t]
            right = [residuals[i] for i, row in enumerate(X) if row[j] > t]
            if not left or not right:
                continue

            lv = mean(left)
            rv = mean(right)
            sse = 0.0
            for i, row in enumerate(X):
                pred = lv if row[j] <= t else rv
                diff = residuals[i] - pred
                sse += diff * diff

            if sse < best_sse:
                best_sse = sse
                best = Stump(j, t, lv, rv)

    if best is None:
        # Fallback when no split found
        avg = mean(residuals)
        best = Stump(0, X[0][0], avg, avg)
    return best


class GradientBoostingRegressor:
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.init_value = 0.0
        self.models: List[Stump] = []

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.init_value = mean(y)
        y_pred = [self.init_value for _ in y]

        for _ in range(self.n_estimators):
            residuals = [yt - yp for yt, yp in zip(y, y_pred)]
            stump = fit_stump(X, residuals)
            self.models.append(stump)

            for i, row in enumerate(X):
                y_pred[i] += self.learning_rate * stump.predict_one(row)

    def predict(self, X: List[List[float]]) -> List[float]:
        preds = [self.init_value for _ in X]
        for stump in self.models:
            for i, row in enumerate(X):
                preds[i] += self.learning_rate * stump.predict_one(row)
        return preds


def mse(y_true: List[float], y_pred: List[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


if __name__ == "__main__":
    X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    y = [1.1, 1.9, 3.2, 3.9, 5.1, 6.2]

    model = GradientBoostingRegressor(n_estimators=20, learning_rate=0.2)
    model.fit(X, y)
    pred = model.predict(X)
    print("MSE:", mse(y, pred))
    print("Pred:", [round(v, 3) for v in pred])
