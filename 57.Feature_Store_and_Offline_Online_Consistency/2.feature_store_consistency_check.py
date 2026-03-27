from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple


Event = Dict[str, float]


def simulate_events(n_users: int = 40, days: int = 30, seed: int = 42) -> List[Event]:
    random.seed(seed)
    events = []
    for day in range(days):
        for u in range(n_users):
            if random.random() < 0.65:
                amount = max(0.0, random.gauss(45 + 0.3 * u, 18))
                events.append({"user_id": u, "day": day, "amount": amount})
    return events


def offline_feature_compute(events: List[Event], snapshot_day: int):
    # Point-in-time correct: only use events <= snapshot_day
    user_amounts = defaultdict(list)
    for e in events:
        if e["day"] <= snapshot_day:
            user_amounts[e["user_id"]].append(e["amount"])

    features = {}
    for u, vals in user_amounts.items():
        features[u] = {
            "tx_count_7d": sum(1 for e in events if e["user_id"] == u and snapshot_day - 6 <= e["day"] <= snapshot_day),
            "avg_amount_30d": sum(vals) / len(vals),
            "max_amount_30d": max(vals),
        }
    return features


def online_feature_compute_with_bug(events: List[Event], snapshot_day: int):
    # Intentionally buggy: includes day+1 leakage for tx_count_7d
    user_amounts = defaultdict(list)
    for e in events:
        if e["day"] <= snapshot_day:
            user_amounts[e["user_id"]].append(e["amount"])

    features = {}
    for u, vals in user_amounts.items():
        features[u] = {
            "tx_count_7d": sum(1 for e in events if e["user_id"] == u and snapshot_day - 6 <= e["day"] <= snapshot_day + 1),
            "avg_amount_30d": sum(vals) / len(vals),
            "max_amount_30d": max(vals),
        }
    return features


def compute_skew(offline_f: Dict[int, Dict[str, float]], online_f: Dict[int, Dict[str, float]]):
    skew = defaultdict(list)
    users = sorted(set(offline_f.keys()) & set(online_f.keys()))

    for u in users:
        for f in offline_f[u]:
            a = offline_f[u][f]
            b = online_f[u][f]
            denom = max(1e-6, abs(a) + abs(b))
            rel = abs(a - b) / denom
            skew[f].append(rel)

    return {k: sum(v) / len(v) if v else 0.0 for k, v in skew.items()}, skew


def maybe_plot(skew_detail: Dict[str, List[float]]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Feature skew (mean relative abs diff):")
        for f, vals in skew_detail.items():
            print(f, round(sum(vals) / len(vals), 6))
        return

    feats = list(skew_detail.keys())
    means = [sum(skew_detail[f]) / len(skew_detail[f]) for f in feats]
    plt.figure(figsize=(7, 4))
    plt.bar(feats, means)
    plt.ylabel("Mean relative skew")
    plt.title("Offline vs Online Feature Skew")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    events = simulate_events(n_users=45, days=35, seed=21)
    snap = 28

    off = offline_feature_compute(events, snapshot_day=snap)
    on = online_feature_compute_with_bug(events, snapshot_day=snap)

    mean_skew, detail = compute_skew(off, on)
    print("Skew summary:")
    for f, v in mean_skew.items():
        status = "ALERT" if v > 0.02 else "ok"
        print(f"- {f}: {v:.5f} [{status}]")

    maybe_plot(detail)
