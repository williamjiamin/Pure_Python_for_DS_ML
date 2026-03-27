from __future__ import annotations

import random
from typing import Dict, List


def simulate_rollout(hours: int = 48, seed: int = 42):
    random.seed(seed)

    traffic_new = 0.05
    rows = []
    phase = "canary"

    for h in range(hours):
        # ramp policy
        if h == 8:
            traffic_new = 0.2
            phase = "ramp1"
        if h == 16:
            traffic_new = 0.5
            phase = "ramp2"
        if h == 24:
            traffic_new = 0.8
            phase = "ramp3"
        if h == 32:
            traffic_new = 1.0
            phase = "full"

        # baseline metrics
        latency = random.gauss(110, 10)
        error_rate = max(0.0, random.gauss(0.009, 0.002))
        conversion = max(0.01, random.gauss(0.125, 0.01))

        # Inject degradation tied to new-model traffic
        if 12 <= h <= 34:
            latency += 90 * traffic_new + random.uniform(0, 20)
            error_rate += 0.03 * traffic_new + random.uniform(0, 0.006)
            conversion -= 0.03 * traffic_new + random.uniform(0, 0.005)

        rows.append(
            {
                "hour": h,
                "phase": phase,
                "traffic_new": traffic_new,
                "latency_ms": latency,
                "error_rate": error_rate,
                "conversion": conversion,
            }
        )

    return rows


def orchestrate_rollback(rows: List[Dict]):
    timeline = []
    rollback_hour = None
    stable_conversion = 0.125

    for r in rows:
        hour = r["hour"]
        if rollback_hour is not None and hour > rollback_hour:
            # Post-rollback, service returns to stable behavior with short settling time.
            r["traffic_new"] = 0.0
            r["phase"] = "rollback_stable"
            hours_since_rollback = hour - rollback_hour
            recovery = min(1.0, hours_since_rollback / 4.0)

            stable_latency = random.gauss(112, 6)
            stable_error = max(0.0, random.gauss(0.009, 0.0015))
            stable_conversion = max(0.01, random.gauss(0.125, 0.006))

            r["latency_ms"] = (1.0 - recovery) * r["latency_ms"] + recovery * stable_latency
            r["error_rate"] = (1.0 - recovery) * r["error_rate"] + recovery * stable_error
            r["conversion"] = (1.0 - recovery) * r["conversion"] + recovery * stable_conversion

        breach = False
        reasons = []
        if r["latency_ms"] > 220:
            breach = True
            reasons.append("latency")
        if r["error_rate"] > 0.03:
            breach = True
            reasons.append("error_rate")
        if stable_conversion - r["conversion"] > 0.02:
            breach = True
            reasons.append("conversion_drop")

        if breach and rollback_hour is None:
            rollback_hour = hour
            timeline.append({"hour": hour, "event": "AUTO_ROLLBACK_TRIGGER", "reasons": reasons})
            timeline.append({"hour": hour + 1, "event": "TRAFFIC_SHIFT_TO_STABLE", "reasons": []})
            timeline.append({"hour": hour + 2, "event": "POST_ROLLBACK_VALIDATION", "reasons": []})

    return rollback_hour, timeline, rows


def summarize(rows: List[Dict], rollback_hour):
    pre = [r for r in rows if rollback_hour is None or r["hour"] <= rollback_hour]
    post = [r for r in rows if rollback_hour is not None and r["hour"] > rollback_hour]

    def avg(rows, k):
        return sum(r[k] for r in rows) / max(1, len(rows))

    out = {
        "avg_latency_pre": avg(pre, "latency_ms"),
        "avg_error_pre": avg(pre, "error_rate"),
        "avg_conv_pre": avg(pre, "conversion"),
    }

    if post:
        out.update(
            {
                "avg_latency_post": avg(post, "latency_ms"),
                "avg_error_post": avg(post, "error_rate"),
                "avg_conv_post": avg(post, "conversion"),
            }
        )
    return out


def maybe_plot(rows: List[Dict], rollback_hour):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Rollback hour:", rollback_hour)
        return

    xs = [r["hour"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    err = [r["error_rate"] for r in rows]
    conv = [r["conversion"] for r in rows]
    traffic = [r["traffic_new"] for r in rows]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(xs, lat); axs[0, 0].axhline(220, color="r", linestyle="--"); axs[0, 0].set_title("Latency")
    axs[0, 1].plot(xs, err); axs[0, 1].axhline(0.03, color="r", linestyle="--"); axs[0, 1].set_title("Error Rate")
    axs[1, 0].plot(xs, conv); axs[1, 0].axhline(0.105, color="r", linestyle="--"); axs[1, 0].set_title("Conversion")
    axs[1, 1].plot(xs, traffic); axs[1, 1].set_title("New Model Traffic Share")

    if rollback_hour is not None:
        for ax in axs.flat:
            ax.axvline(rollback_hour, color="orange", linestyle="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rows = simulate_rollout(hours=48, seed=14)
    rollback_hour, timeline, updated = orchestrate_rollback(rows)

    print("Rollback hour:", rollback_hour)
    print("Timeline:")
    for e in timeline:
        print(e)

    summary = summarize(updated, rollback_hour)
    print("Summary:", {k: round(v, 5) for k, v in summary.items()})

    maybe_plot(updated, rollback_hour)
