from __future__ import annotations

from typing import Callable, List


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def seasonal_naive(train: List[float], horizon: int, season: int = 7) -> List[float]:
    return [train[-season + (i % season)] for i in range(horizon)]


def walk_forward(
    series: List[float],
    min_train_size: int,
    horizon: int,
    forecaster: Callable[[List[float], int], List[float]],
) -> float:
    preds = []
    truth = []

    start = min_train_size
    while start + horizon <= len(series):
        train = series[:start]
        test = series[start : start + horizon]
        f = forecaster(train, horizon)
        preds.extend(f)
        truth.extend(test)
        start += horizon

    return mae(truth, preds)


if __name__ == "__main__":
    # Synthetic weekly seasonal data
    data = [10, 12, 13, 11, 9, 8, 7] * 8
    score = walk_forward(
        series=data,
        min_train_size=21,
        horizon=7,
        forecaster=lambda tr, h: seasonal_naive(tr, h, season=7),
    )
    print("Walk-forward MAE:", score)
