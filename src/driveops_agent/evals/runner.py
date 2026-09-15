import json

from ..agent import DriveOpsAgent


def run_evals(root):
    cases = [
        json.loads(x) for x in (root / "evals/cases.jsonl").read_text(encoding="utf8").splitlines()
    ]
    results = []
    for c in cases:
        s = DriveOpsAgent(root / "data", root / "reports").run(c["task"])
        tools = {x.name for x in s.tool_calls}
        claim_ev = {e for cl in s.claims for e in cl.evidence_ids}
        sources = {e.evidence_id for e in s.evidence}
        expected = set(c["expected_tools"])
        forbidden = set(c["forbidden_tools"])
        required = set(c["required_evidence"])
        tool_ok = expected.issubset(tools) and not (tools & forbidden)
        evidence_ok = required.issubset(claim_ev) and claim_ev.issubset(sources)
        keyword_ok = any(
            k.lower() in (s.final_answer or "").lower() for k in c["expected_claim_keywords"]
        )
        results.append((s, tool_ok, evidence_ok, keyword_ok))
    n = len(results)
    claims = [cl for s, *_ in results for cl in s.claims]
    total = max(1, len(claims))
    report = {
        "cases": n,
        "task_success_rate": sum(x[3] for x in results) / n,
        "tool_selection_accuracy": sum(x[1] for x in results) / n,
        "evidence_coverage": sum(x[2] for x in results) / n,
        "unsupported_claim_rate": sum(c.unsupported for c in claims) / total,
        "hallucinated_source_rate": sum(
            any(i not in {e.evidence_id for e in s.evidence} for i in c.evidence_ids)
            for s, *_ in results
            for c in s.claims
        )
        / total,
        "average_tool_calls": sum(len(s.tool_calls) for s, *_ in results) / n,
    }
    (root / "reports/v1_eval.json").write_text(json.dumps(report, indent=2))
    (root / "reports/v1_eval.md").write_text("\n".join(f"- {k}: {v}" for k, v in report.items()))
    return report
