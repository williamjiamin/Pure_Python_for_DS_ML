from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # +1 means predict +1 when x <= threshold else -1

    def predict_one(self, x: List[float]) -> int:
        pred = 1 if x[self.feature] <= self.threshold else -1
        return pred if self.polarity == 1 else -pred


class AdaBoostClassifier:
    def __init__(self, n_estimators: int = 50) -> None:
        self.n_estimators = n_estimators
        self.learners: List[Tuple[Stump, float]] = []

    def _best_stump(self, X: List[List[float]], y: List[int], w: List[float]) -> Tuple[Stump, float]:
        n_features = len(X[0])
        best_err = float("inf")
        best_stump = None

        for j in range(n_features):
            thresholds = sorted(set(row[j] for row in X))
            for t in thresholds:
                for polarity in (1, -1):
                    stump = Stump(j, t, polarity)
                    err = 0.0
                    for xi, yi, wi in zip(X, y, w):
                        if stump.predict_one(xi) != yi:
                            err += wi
                    if err < best_err:
                        best_err = err
                        best_stump = stump

        return best_stump, best_err

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        n = len(X)
        w = [1.0 / n] * n
        self.learners = []

        for _ in range(self.n_estimators):
            stump, err = self._best_stump(X, y, w)
            err = max(1e-12, min(1 - 1e-12, err))
            alpha = 0.5 * math.log((1 - err) / err)

            # Update weights: upweight misclassified points.
            new_w = []
            for xi, yi, wi in zip(X, y, w):
                hi = stump.predict_one(xi)
                new_w.append(wi * math.exp(-alpha * yi * hi))

            z = sum(new_w)
            w = [wi / z for wi in new_w]
            self.learners.append((stump, alpha))

    def decision_function(self, x: List[float]) -> float:
        return sum(alpha * stump.predict_one(x) for stump, alpha in self.learners)

    def predict(self, X: List[List[float]]) -> List[int]:
        return [1 if self.decision_function(x) >= 0 else -1 for x in X]


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


if __name__ == "__main__":
    X = [[1.0], [2.0], [3.0], [4.0], [6.0], [7.0], [8.0], [9.0]]
    y = [-1, -1, -1, -1, 1, 1, 1, 1]

    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X, y)
    pred = model.predict(X)

    print("Pred:", pred)
    print("Accuracy:", round(accuracy(y, pred), 3))
