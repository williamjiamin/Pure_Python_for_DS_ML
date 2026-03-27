from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def simulate_survival(n: int = 800, seed: int = 42):
    random.seed(seed)
    time = []
    event = []
    group = []

    for i in range(n):
        g = 0 if i < n // 2 else 1
        # group 1 has slightly better survival (lower hazard)
        rate = 0.06 if g == 0 else 0.045

        t_event = random.expovariate(rate)
        t_censor = random.uniform(8, 45)
        t = min(t_event, t_censor)
        e = 1 if t_event <= t_censor else 0

        time.append(t)
        event.append(e)
        group.append(g)

    return time, event, group


def kaplan_meier(times: List[float], events: List[int]):
    pairs = sorted(zip(times, events), key=lambda x: x[0])
    unique_times = sorted(set(t for t, e in pairs if e == 1))

    n_at_risk = len(times)
    S = 1.0
    curve_t = [0.0]
    curve_s = [1.0]

    idx = 0
    for t in unique_times:
        d = 0
        c = 0
        while idx < len(pairs) and pairs[idx][0] <= t + 1e-12:
            if abs(pairs[idx][0] - t) < 1e-12 and pairs[idx][1] == 1:
                d += 1
            elif pairs[idx][0] <= t and pairs[idx][1] == 0:
                c += 1
            idx += 1

        if n_at_risk > 0 and d > 0:
            S *= (1 - d / n_at_risk)
            curve_t.append(t)
            curve_s.append(S)

        n_at_risk -= (d + c)

    return curve_t, curve_s


def median_survival(curve_t: List[float], curve_s: List[float]) -> float:
    for t, s in zip(curve_t, curve_s):
        if s <= 0.5:
            return t
    return float("inf")


def logrank_stat(times: List[float], events: List[int], groups: List[int]) -> float:
    # Simple 2-group log-rank approximation
    unique_times = sorted(set(t for t, e in zip(times, events) if e == 1))

    O1 = E1 = V1 = 0.0
    for t in unique_times:
        at_risk_0 = at_risk_1 = d0 = d1 = 0
        for ti, ei, gi in zip(times, events, groups):
            if ti >= t:
                if gi == 0:
                    at_risk_0 += 1
                else:
                    at_risk_1 += 1
            if abs(ti - t) < 1e-12 and ei == 1:
                if gi == 0:
                    d0 += 1
                else:
                    d1 += 1

        n = at_risk_0 + at_risk_1
        d = d0 + d1
        if n <= 1 or d == 0:
            continue

        O1 += d0
        E1 += d * (at_risk_0 / n)
        V1 += (at_risk_0 * at_risk_1 * d * (n - d)) / (n * n * (n - 1) + 1e-12)

    if V1 <= 1e-12:
        return 0.0
    z = (O1 - E1) / math.sqrt(V1)
    return z


def maybe_plot(t0, s0, t1, s1):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("KM points control:", list(zip([round(x, 2) for x in t0[:8]], [round(y, 3) for y in s0[:8]])))
        print("KM points treat  :", list(zip([round(x, 2) for x in t1[:8]], [round(y, 3) for y in s1[:8]])))
        return

    plt.figure(figsize=(8, 4))
    plt.step(t0, s0, where="post", label="control")
    plt.step(t1, s1, where="post", label="treatment")
    plt.ylim(0, 1.02)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    times, events, groups = simulate_survival(n=900, seed=11)

    t0 = [t for t, g in zip(times, groups) if g == 0]
    e0 = [e for e, g in zip(events, groups) if g == 0]
    t1 = [t for t, g in zip(times, groups) if g == 1]
    e1 = [e for e, g in zip(events, groups) if g == 1]

    c0_t, c0_s = kaplan_meier(t0, e0)
    c1_t, c1_s = kaplan_meier(t1, e1)

    m0 = median_survival(c0_t, c0_s)
    m1 = median_survival(c1_t, c1_s)

    z = logrank_stat(times, events, groups)

    print("Median survival control:", round(m0, 3) if m0 < 1e9 else "not reached")
    print("Median survival treat  :", round(m1, 3) if m1 < 1e9 else "not reached")
    print("Log-rank z-statistic   :", round(z, 4))

    maybe_plot(c0_t, c0_s, c1_t, c1_s)
