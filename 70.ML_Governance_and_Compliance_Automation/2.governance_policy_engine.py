from __future__ import annotations

from typing import Dict, List


def sample_registry():
    return [
        {
            "model_id": "credit_risk_v12",
            "owner": "risk-team",
            "model_card": True,
            "pii_review": True,
            "fairness_gap": 0.045,
            "auc": 0.804,
            "drift_psi": 0.08,
            "approver": "ml-governance",
            "last_validation_days": 18,
        },
        {
            "model_id": "ads_ctr_v23",
            "owner": "ads-ranking",
            "model_card": True,
            "pii_review": False,
            "fairness_gap": 0.093,
            "auc": 0.771,
            "drift_psi": 0.16,
            "approver": "",
            "last_validation_days": 44,
        },
        {
            "model_id": "fraud_stream_v7",
            "owner": "fraud-detection",
            "model_card": False,
            "pii_review": True,
            "fairness_gap": 0.031,
            "auc": 0.838,
            "drift_psi": 0.22,
            "approver": "ml-governance",
            "last_validation_days": 61,
        },
    ]


def evaluate_policy(model: Dict) -> Dict:
    checks = {}

    checks["owner_present"] = bool(model.get("owner"))
    checks["model_card"] = bool(model.get("model_card"))
    checks["pii_review"] = bool(model.get("pii_review"))
    checks["approver_present"] = bool(model.get("approver"))

    checks["fairness_ok"] = model.get("fairness_gap", 1.0) <= 0.06
    checks["auc_ok"] = model.get("auc", 0.0) >= 0.78
    checks["drift_ok"] = model.get("drift_psi", 1.0) <= 0.15
    checks["validation_fresh"] = model.get("last_validation_days", 999) <= 30

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = passed / total

    blockers = [k for k, v in checks.items() if not v]
    release = "APPROVE" if score >= 0.9 and not blockers else "BLOCK"

    return {
        "model_id": model["model_id"],
        "score": score,
        "release": release,
        "blockers": blockers,
    }


def remediation_actions(blockers: List[str]) -> List[str]:
    mapping = {
        "model_card": "Create/refresh model card",
        "pii_review": "Complete privacy/PII review",
        "approver_present": "Assign governance approver",
        "fairness_ok": "Run fairness mitigation and re-evaluate",
        "auc_ok": "Retrain/tune model to meet performance floor",
        "drift_ok": "Investigate drift and retrain with fresh data",
        "validation_fresh": "Run new validation suite",
        "owner_present": "Assign accountable model owner",
    }
    return [mapping[b] for b in blockers if b in mapping]


def weekly_report(results: List[Dict]) -> Dict:
    approved = sum(1 for r in results if r["release"] == "APPROVE")
    blocked = len(results) - approved
    avg_score = sum(r["score"] for r in results) / len(results)
    return {
        "models_total": len(results),
        "approved": approved,
        "blocked": blocked,
        "avg_score": round(avg_score, 4),
    }


def maybe_plot(results: List[Dict]):
    labels = [r["model_id"] for r in results]
    vals = [r["score"] for r in results]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        for r in results:
            print(r["model_id"], "score", round(r["score"], 3), "release", r["release"])
        return

    plt.figure(figsize=(8, 4))
    colors = ["green" if r["release"] == "APPROVE" else "red" for r in results]
    plt.bar(labels, vals, color=colors)
    plt.axhline(0.9, color="black", linestyle="--", label="release threshold")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.title("Governance Compliance Scores")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    models = sample_registry()
    results = [evaluate_policy(m) for m in models]

    for r in results:
        print("\nModel:", r["model_id"])
        print("Score:", round(r["score"], 3), "Decision:", r["release"])
        if r["blockers"]:
            print("Blockers:", r["blockers"])
            print("Actions:", remediation_actions(r["blockers"]))

    summary = weekly_report(results)
    print("\nWeekly governance summary:", summary)

    maybe_plot(results)
