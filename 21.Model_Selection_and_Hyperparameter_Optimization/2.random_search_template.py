"""Simple random search runner (framework-agnostic).

Replace `train_and_eval` with your model training function.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple


def sample_space() -> Dict[str, float]:
    return {
        "learning_rate": 10 ** random.uniform(-4, -1),
        "l2": 10 ** random.uniform(-6, -2),
        "epochs": random.choice([50, 100, 150, 200]),
    }


def train_and_eval(params: Dict[str, float]) -> Tuple[float, float]:
    # TODO: plug in your actual training pipeline and return (score, latency_ms)
    # This placeholder mimics a score surface with noise.
    score = 0.8 - abs(params["learning_rate"] - 0.01) * 2.0
    score -= abs(params["l2"] - 1e-4) * 20.0
    score += random.uniform(-0.01, 0.01)
    latency_ms = random.uniform(2.0, 8.0)
    return score, latency_ms


def random_search(n_trials: int = 40) -> List[Dict[str, float]]:
    history = []
    for trial in range(1, n_trials + 1):
        params = sample_space()
        t0 = time.time()
        score, latency_ms = train_and_eval(params)
        elapsed = (time.time() - t0) * 1000.0

        row = {
            "trial": trial,
            "score": score,
            "latency_ms": latency_ms,
            "eval_ms": elapsed,
            **params,
        }
        history.append(row)

    history.sort(key=lambda x: x["score"], reverse=True)
    return history


def main() -> None:
    results = random_search(40)
    print("Top 5 runs:")
    for r in results[:5]:
        print(r)


if __name__ == "__main__":
    main()
