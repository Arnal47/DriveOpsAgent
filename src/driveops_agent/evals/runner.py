import json

from ..agent import DriveOpsAgent
from ..external.base import Adapter
from ..models import Status
from ..providers import MockProvider


class FlakyProvider(MockProvider):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.count = 0

    def complete(self, purpose, payload):
        if purpose == "synthesis" and self.fail and self.count == 0:
            self.count += 1
            raise TimeoutError("provider timeout")
        return super().complete(purpose, payload)


def _adapter(mode):
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        if mode == "tool_retry" and len(calls) == 1:
            raise TimeoutError()
        if mode in {"tool_exhausted", "review_required"}:
            raise TimeoutError()
        return {}

    return Adapter(transport, retries=1 if mode == "tool_retry" else 0), calls


def _trace_ok(agent, review):
    events = [json.loads(x) for x in agent.trace.path.read_text().splitlines()]
    names = {e["event_type"] for e in events}
    required = {
        "run_start",
        "provider_call",
        "memory_lookup",
        "memory_save",
        "run_end",
    }
    if review:
        required.add("review_required")
    fields = {"run_id", "step_id", "timestamp", "latency_ms", "success", "error", "evidence_ids"}
    return (
        required.issubset(names)
        and ({"tool_call", "tool_error"} & names)
        and all(fields.issubset(e) for e in events)
    )


def run_evals(root):
    cases = [json.loads(x) for x in (root / "evals/cases.jsonl").read_text().splitlines()]
    rows = []
    v2 = []
    for c in cases:
        mode = c.get("v2_mode")
        provider = FlakyProvider(mode in {"provider_recovery", "provider_invalid_json"})
        adapter, calls = (
            _adapter(mode)
            if mode in {"tool_retry", "tool_exhausted", "review_required"}
            else (None, [])
        )
        agent = DriveOpsAgent(root / "data", root / "reports", provider, external_adapter=adapter)
        s = agent.run(c["task"])
        tools = {x.name for x in s.tool_calls}
        linked = {i for q in s.claims for i in q.evidence_ids}
        ids = {e.evidence_id for e in s.evidence}
        ok_tools = set(c["expected_tools"]).issubset(tools) and not (
            set(c["forbidden_tools"]) & tools
        )
        ok_ev = set(c["required_evidence"]).issubset(linked) and linked.issubset(ids)
        ok_kw = all(
            k.lower() in (s.final_answer or "").lower() for k in c["expected_claim_keywords"]
        )
        review = s.status == Status.NEEDS_REVIEW
        rows.append((s, ok_tools, ok_ev, ok_kw, agent, review, c))
        if mode:
            v2.append((mode, s, agent, calls, review, c))
    base = rows[:16]
    claims = [q for s, *_ in rows for q in s.claims]
    total = max(1, len(claims))
    provider = [x for x in v2 if x[0] in {"provider_recovery", "provider_invalid_json"}]
    tool = [x for x in v2 if x[0] == "tool_retry"]
    reviews = [x for x in v2 if x[0] in {"tool_exhausted", "review_required"}]
    # actual second run establishes memory hit; current evidence remains current provenance
    mem = DriveOpsAgent(root / "data", root / "reports")
    mem.run("CAN timeout")
    m2 = mem.run("CAN timeout")
    memory_ok = any(
        json.loads(x)["event_type"] == "memory_hit" for x in mem.trace.path.read_text().splitlines()
    ) and all(e.provenance == "current" for e in m2.evidence)
    report = {
        "cases": len(cases),
        "task_success_rate": sum(a and b and d for _, a, b, d, *z in base) / len(base),
        "tool_selection_accuracy": sum(a for _, a, *z in base) / len(base),
        "evidence_coverage": sum(b for _, a, b, *z in base) / len(base),
        "unsupported_claim_rate": sum(q.unsupported for q in claims) / total,
        "hallucinated_source_rate": sum(
            any(i not in {e.evidence_id for e in s.evidence} for i in q.evidence_ids)
            for s, *_ in rows
            for q in s.claims
        )
        / total,
        "provider_error_recovery_rate": sum(
            s.status == Status.COMPLETE and any(e["event_type"] == "retry" for e in a.trace.events)
            for _, s, a, *z in provider
        )
        / max(1, len(provider)),
        "tool_failure_recovery_rate": sum(
            s.status == Status.COMPLETE and len(calls) > 1 for _, s, a, calls, *z in tool
        )
        / max(1, len(tool)),
        "review_routing_accuracy": sum(
            review == (c.get("expected_review", True)) for _, s, a, calls, review, c in reviews
        )
        / max(1, len(reviews)),
        "memory_provenance_accuracy": float(memory_ok),
        "trace_completeness": sum(_trace_ok(a, review) for s, x, y, z, a, review, c in rows)
        / len(rows),
    }
    (root / "reports/v1_eval.json").write_text(json.dumps(report, indent=2))
    (root / "reports/v1_eval.md").write_text("\n".join(f"- {k}: {v}" for k, v in report.items()))
    return report
