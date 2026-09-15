import json

from ..agent import DriveOpsAgent


def run_evals(root):
    cases = [json.loads(x) for x in (root / "evals/cases.jsonl").read_text().splitlines()]
    rows = []
    for c in cases:
        s = DriveOpsAgent(root / "data", root / "reports").run(c["task"])
        tools = {x.name for x in s.tool_calls}
        linked = {i for cl in s.claims for i in cl.evidence_ids}
        ids = {e.evidence_id for e in s.evidence}
        ok_tools = set(c["expected_tools"]).issubset(tools) and not (
            set(c["forbidden_tools"]) & tools
        )
        ok_ev = set(c["required_evidence"]).issubset(linked) and linked.issubset(ids)
        ok_kw = all(
            k.lower() in (s.final_answer or "").lower() for k in c["expected_claim_keywords"]
        )
        uncertain = any(cl.uncertain for cl in s.claims)
        ok_uncertain = uncertain == c["allowed_uncertainty"]
        rows.append((s, ok_tools, ok_ev, ok_kw and ok_uncertain))
    n = len(rows)
    claims = [c for s, *_ in rows for c in s.claims]
    total = max(1, len(claims))
    report = {
        "cases": n,
        "task_success_rate": sum(x[3] and x[1] and x[2] for x in rows) / n,
        "tool_selection_accuracy": sum(x[1] for x in rows) / n,
        "evidence_coverage": sum(x[2] for x in rows) / n,
        "unsupported_claim_rate": sum(c.unsupported for c in claims) / total,
        "hallucinated_source_rate": sum(
            any(i not in {e.evidence_id for e in s.evidence} for i in c.evidence_ids)
            for s, *_ in rows
            for c in s.claims
        )
        / total,
        "average_tool_calls": sum(len(s.tool_calls) for s, *_ in rows) / n,
    }
    (root / "reports/v1_eval.json").write_text(json.dumps(report, indent=2))
    (root / "reports/v1_eval.md").write_text("\n".join(f"- {k}: {v}" for k, v in report.items()))
    return report
