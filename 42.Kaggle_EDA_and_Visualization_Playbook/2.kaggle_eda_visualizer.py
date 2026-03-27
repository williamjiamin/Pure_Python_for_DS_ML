from __future__ import annotations

import csv
import math
from collections import Counter
from typing import Dict, List


DIABETES_SCHEMA = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    first = rows[0]
    # If first row looks numeric, treat file as headerless.
    if sum(_is_number(x) for x in first) >= max(1, int(0.8 * len(first))):
        header = DIABETES_SCHEMA[: len(first)]
        data_rows = rows
    else:
        header = first
        data_rows = rows[1:]

    out = []
    for r in data_rows:
        if len(r) != len(header):
            continue
        out.append({k: v for k, v in zip(header, r)})
    return out


def try_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def numeric_columns(rows: List[Dict[str, str]]) -> List[str]:
    cols = rows[0].keys()
    out = []
    for c in cols:
        ok = 0
        for r in rows[:100]:
            if try_float(r[c]) is not None:
                ok += 1
        if ok >= 90:
            out.append(c)
    return out


def missing_ratio(rows: List[Dict[str, str]], col: str) -> float:
    miss = sum(1 for r in rows if r[col] is None or str(r[col]).strip() == "")
    return miss / len(rows)


def describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": s[0],
        "p25": s[int(0.25 * (n - 1))],
        "p50": s[int(0.50 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "max": s[-1],
    }


def ascii_hist(values: List[float], bins: int = 12, width: int = 40) -> None:
    if not values:
        print("No values")
        return
    lo, hi = min(values), max(values)
    if hi == lo:
        print("constant values")
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / step))
        counts[idx] += 1
    m = max(counts)
    for i, c in enumerate(counts):
        l = lo + i * step
        r = l + step
        n = int((c / m) * width) if m else 0
        print(f"[{l:8.3f},{r:8.3f}) {'#' * n} {c}")


def maybe_plot_hist(values: List[float], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"\n{title} (ASCII)")
        ascii_hist(values, bins=12)
        return

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=12, edgecolor="black")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def target_breakdown(rows: List[Dict[str, str]], target_col: str) -> None:
    c = Counter(r[target_col] for r in rows)
    print("Target distribution:", dict(c))


def main() -> None:
    path = "Y.Kaggle_Data/diabetes.csv"
    rows = read_csv(path)
    print("Rows:", len(rows), "Columns:", len(rows[0]))

    num_cols = numeric_columns(rows)
    print("Numeric columns:", num_cols)

    print("\nMissing ratio:")
    for c in rows[0].keys():
        print(f"- {c}: {missing_ratio(rows, c):.3f}")

    target_col = "Outcome" if "Outcome" in rows[0] else list(rows[0].keys())[-1]
    target_breakdown(rows, target_col)

    print("\nNumeric describe (first 3 cols):")
    for c in num_cols[:3]:
        vals = [try_float(r[c]) for r in rows]
        vals = [v for v in vals if v is not None]
        d = describe(vals)
        print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})
        maybe_plot_hist(vals, title=f"Distribution: {c}")


if __name__ == "__main__":
    main()
