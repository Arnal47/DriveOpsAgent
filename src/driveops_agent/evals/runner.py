import json
from pathlib import Path

from ..agent import DriveOpsAgent


def run_evals(root: Path) -> dict:
    cases = [
        json.loads(x) for x in (root / "evals/cases.jsonl").read_text(encoding="utf8").splitlines()
    ]
    results = [DriveOpsAgent(root / "data", root / "reports").run(c["task"]) for c in cases]
    success = sum(r.status.value == "complete" for r in results) / len(results)
    tool_ok = sum(len(r.tool_calls) >= 4 for r in results) / len(results)
    ev_ok = sum(bool(r.evidence) for r in results) / len(results)
    unsupported = sum("Uncertain:" in (r.final_answer or "") for r in results) / len(results)
    report = {
        "cases": len(cases),
        "task_success_rate": success,
        "tool_selection_accuracy": tool_ok,
        "required_evidence_coverage": ev_ok,
        "unsupported_claim_rate": unsupported,
        "average_tool_calls": sum(len(r.tool_calls) for r in results) / len(results),
        "average_steps": sum(r.step_count for r in results) / len(results),
    }
    (root / "reports/v1_eval.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    (root / "reports/v1_eval.md").write_text(
        "# V1 Evaluation\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items()), encoding="utf8"
    )
    return report
