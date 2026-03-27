from __future__ import annotations

import random
from typing import Dict, List


def simulate_monitoring(hours: int = 72, seed: int = 42) -> List[Dict[str, float]]:
    random.seed(seed)
    rows = []
    for h in range(hours):
        latency = random.gauss(110, 15)
        error_rate = max(0.0, random.gauss(0.008, 0.003))
        drift_psi = max(0.0, random.gauss(0.06, 0.03))
        auc = random.gauss(0.81, 0.015)

        # Inject incidents
        if 20 <= h <= 28:
            latency += random.uniform(60, 150)
            error_rate += random.uniform(0.02, 0.05)
        if 36 <= h <= 52:
            drift_psi += random.uniform(0.12, 0.35)
            auc -= random.uniform(0.03, 0.12)
        if 60 <= h <= 66:
            error_rate += random.uniform(0.03, 0.08)

        rows.append(
            {
                "hour": h,
                "latency_ms": latency,
                "error_rate": error_rate,
                "drift_psi": drift_psi,
                "auc": auc,
            }
        )
    return rows


def detect_incidents(rows: List[Dict[str, float]]):
    alerts = []
    for r in rows:
        sev = None
        reasons = []
        if r["error_rate"] > 0.04 or r["latency_ms"] > 250:
            sev = "SEV1"
            reasons.append("service_slo")
        if r["drift_psi"] > 0.2 or r["auc"] < 0.72:
            sev = "SEV2" if sev is None else sev
            reasons.append("model_quality")

        if sev is not None:
            alerts.append({"hour": r["hour"], "severity": sev, "reasons": reasons, **r})
    return alerts


def execute_runbook(alert: Dict[str, float]) -> Dict[str, str]:
    actions = []
    if "service_slo" in alert["reasons"]:
        actions.append("enable_traffic_throttle")
        actions.append("fallback_to_last_stable_model")
    if "model_quality" in alert["reasons"]:
        actions.append("switch_to_shadow_baseline")
        actions.append("start_data_quality_investigation")

    if alert["severity"] == "SEV1":
        actions.append("page_oncall_immediately")

    status = "mitigated" if actions else "monitoring"
    return {
        "hour": str(alert["hour"]),
        "severity": alert["severity"],
        "actions": ",".join(actions),
        "status": status,
    }


def postmortem(alerts: List[Dict[str, float]], runbook_logs: List[Dict[str, str]]) -> Dict[str, str]:
    sev1 = sum(1 for a in alerts if a["severity"] == "SEV1")
    sev2 = sum(1 for a in alerts if a["severity"] == "SEV2")
    root = "combined_service_and_model_drift" if any("service_slo" in a["reasons"] for a in alerts) and any("model_quality" in a["reasons"] for a in alerts) else "single_domain_issue"

    return {
        "total_alerts": str(len(alerts)),
        "sev1_count": str(sev1),
        "sev2_count": str(sev2),
        "root_cause_summary": root,
        "action_items": "improve_auto_rollback, strengthen_data_contracts, add_drift_canary",
    }


def maybe_plot(rows, alerts):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Alert hours:", [a["hour"] for a in alerts[:15]], "...")
        return

    xs = [r["hour"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    err = [r["error_rate"] for r in rows]
    psi = [r["drift_psi"] for r in rows]
    auc = [r["auc"] for r in rows]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs[0, 0].plot(xs, lat)
    axs[0, 0].axhline(250, color="r", linestyle="--")
    axs[0, 0].set_title("Latency")

    axs[0, 1].plot(xs, err)
    axs[0, 1].axhline(0.04, color="r", linestyle="--")
    axs[0, 1].set_title("Error Rate")

    axs[1, 0].plot(xs, psi)
    axs[1, 0].axhline(0.2, color="r", linestyle="--")
    axs[1, 0].set_title("Drift PSI")

    axs[1, 1].plot(xs, auc)
    axs[1, 1].axhline(0.72, color="r", linestyle="--")
    axs[1, 1].set_title("AUC")

    for a in alerts:
        for ax in axs.flat:
            ax.axvline(a["hour"], color="orange", alpha=0.05)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rows = simulate_monitoring(hours=80, seed=15)
    alerts = detect_incidents(rows)

    print("Total alerts:", len(alerts))
    print("First 5 alerts:")
    for a in alerts[:5]:
        print({
            "hour": a["hour"],
            "severity": a["severity"],
            "reasons": a["reasons"],
            "latency_ms": round(a["latency_ms"], 2),
            "error_rate": round(a["error_rate"], 4),
            "drift_psi": round(a["drift_psi"], 4),
            "auc": round(a["auc"], 4),
        })

    runbook_logs = [execute_runbook(a) for a in alerts]
    print("Runbook sample:", runbook_logs[:3])

    pm = postmortem(alerts, runbook_logs)
    print("Postmortem summary:", pm)

    maybe_plot(rows, alerts)
