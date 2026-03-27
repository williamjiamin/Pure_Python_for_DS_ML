from __future__ import annotations

import math
import random
from typing import List, Tuple


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_prop_z_test(conv_a: int, n_a: int, conv_b: int, n_b: int) -> Tuple[float, float, float]:
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se < 1e-12:
        return p_a, p_b, 1.0

    z = (p_b - p_a) / se
    pval = 2.0 * (1.0 - normal_cdf(abs(z)))
    return p_a, p_b, pval


def run_experiment(
    p_a_true: float,
    p_b_true: float,
    total_per_group: int = 5000,
    checkpoints: int = 10,
    seed: int = 42,
):
    random.seed(seed)

    conv_a = 0
    conv_b = 0
    n_a = 0
    n_b = 0
    records = []

    step = total_per_group // checkpoints
    for k in range(1, checkpoints + 1):
        for _ in range(step):
            n_a += 1
            n_b += 1
            conv_a += 1 if random.random() < p_a_true else 0
            conv_b += 1 if random.random() < p_b_true else 0

        p_a, p_b, pval = two_prop_z_test(conv_a, n_a, conv_b, n_b)
        records.append((k, n_a, p_a, p_b, p_b - p_a, pval))

    return records


def analyze_stopping(records, alpha: float = 0.05):
    naive_stop = None
    corrected_stop = None
    m = len(records)
    alpha_corr = alpha / m

    for k, _, _, _, _, p in records:
        if naive_stop is None and p < alpha:
            naive_stop = k
        if corrected_stop is None and p < alpha_corr:
            corrected_stop = k

    return naive_stop, corrected_stop, alpha_corr


def maybe_plot(records):
    xs = [k for k, *_ in records]
    diffs = [d for *_, d, _ in records]
    pvals = [p for *_, p in records]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("checkpoint diff pvalue")
        for k, d, p in zip(xs, diffs, pvals):
            print(k, round(d, 5), round(p, 6))
        return

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(xs, diffs, marker="o", label="effect (pB-pA)")
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("Effect Size")

    ax2 = ax1.twinx()
    ax2.plot(xs, pvals, marker="s", color="orange", label="p-value")
    ax2.axhline(0.05, color="red", linestyle="--")
    ax2.set_ylabel("P-value")

    fig.tight_layout()
    plt.title("Sequential A/B Monitoring")
    plt.show()


if __name__ == "__main__":
    # Scenario with small true uplift.
    rec = run_experiment(p_a_true=0.10, p_b_true=0.112, total_per_group=6000, checkpoints=12, seed=9)
    naive_stop, corr_stop, alpha_corr = analyze_stopping(rec, alpha=0.05)

    print("Bonferroni corrected alpha per checkpoint:", round(alpha_corr, 6))
    print("Naive sequential stop checkpoint:", naive_stop)
    print("Corrected stop checkpoint       :", corr_stop)

    last = rec[-1]
    _, n, p_a, p_b, diff, p = last
    print("Final n/group:", n)
    print("Final pA, pB:", round(p_a, 4), round(p_b, 4))
    print("Final effect:", round(diff, 4), "Final p-value:", round(p, 6))

    maybe_plot(rec)
