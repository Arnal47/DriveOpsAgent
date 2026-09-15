from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..agent import DriveOpsAgent
from ..external.base import Adapter
from ..models import Status
from ..providers import MockProvider
from ..providers.base import OpenAICompatibleProvider


def _envelope(content):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def _provider(mode):
    calls = []

    def transport(body, headers, timeout):
        calls.append(1)
        request = json.loads(json.loads(body)["messages"][0]["content"])
        if mode == "provider_timeout" and len(calls) == 1:
            raise TimeoutError("provider timeout")
        if mode == "provider_invalid_json" and len(calls) == 1:
            return b"not valid JSON"
        if mode == "provider_exhausted":
            raise TimeoutError("provider unavailable")
        return _envelope(MockProvider().complete(request["purpose"], request["payload"]))

    return OpenAICompatibleProvider(
        transport,
        max_retries=1,
        api_key="offline",
        base_url="https://offline.invalid",
        model="fake",
    ), calls


def _adapter(mode):
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        if mode == "tool_retry" and len(calls) == 1:
            raise TimeoutError("adapter timeout")
        if mode == "tool_exhausted":
            raise TimeoutError("adapter unavailable")
        return {}

    threshold = 0 if mode == "circuit_open" else 3
    return Adapter(transport, retries=1 if mode == "tool_retry" else 0, threshold=threshold), calls


def _events(agent):
    return [json.loads(line) for line in agent.trace.path.read_text().splitlines()]


def _trace_ok(agent, expected_review=False):
    events = _events(agent)
    names = {e["event_type"] for e in events}
    required = {"run_start", "provider_call", "memory_lookup", "memory_save", "run_end"}
    if not ({"memory_hit", "memory_miss"} & names):
        return False
    if not ({"tool_call", "tool_error", "error"} & names):
        return False
    if expected_review:
        required.add("review_required")
    fields = {
        "run_id",
        "step_id",
        "timestamp",
        "event_type",
        "tool",
        "arguments",
        "latency_ms",
        "success",
        "error",
        "evidence_ids",
    }
    return required.issubset(names) and all(
        fields.issubset(e)
        and e["run_id"] == agent.trace.run_id
        and isinstance(e["latency_ms"], (int, float))
        for e in events
    )


def _seed_stale(agent, task):
    current = agent.registry.call("read_log", {"log_id": "can-timeout-004"})[1][0].model_dump()
    current["snippet"] = "stale historical log said no timeout"
    agent.memory.save(
        "stale-seed",
        {"seed": True},
        [current],
        task=task,
        scenario_id="can-timeout-004",
        status="complete",
    )


def _run_case(root: Path, case: dict, index: int):
    mode = case.get("v2_mode", "")
    provider = MockProvider()
    provider_calls = []
    adapter = None
    adapter_calls = []
    backend = "tfidf"
    if mode in {"provider_recovery", "provider_invalid_json", "provider_retry_exhausted"}:
        provider_mode = {
            "provider_recovery": "provider_timeout",
            "provider_invalid_json": "provider_invalid_json",
            "provider_retry_exhausted": "provider_exhausted",
        }[mode]
        provider, provider_calls = _provider(provider_mode)
    if mode in {"tool_retry", "tool_exhausted", "review_required", "circuit_open"}:
        adapter_mode = "tool_exhausted" if mode == "review_required" else mode
        adapter, adapter_calls = _adapter(adapter_mode)
    if mode in {"bm25", "hybrid"}:
        backend = mode
    reports = root / "reports" / "eval_runs" / f"{index}-{uuid.uuid4()}"
    reports.mkdir(parents=True, exist_ok=True)
    agent = DriveOpsAgent(
        root / "data", reports, provider, external_adapter=adapter, retrieval_backend=backend
    )
    task = case["task"]
    if mode == "memory_hit":
        agent.run(task)
    if mode == "stale_memory":
        _seed_stale(agent, task)
    if mode in {"review_approve", "review_reject"} and "conflict" not in task.lower():
        task = "conflict " + task
    state = agent.run(task)
    routed = state.status == Status.NEEDS_REVIEW
    post_status = state.status
    if mode == "review_approve":
        post_status = agent.review(state.run_id, True).status
    if mode == "review_reject":
        post_status = agent.review(state.run_id, False).status
    events = _events(agent)
    event_names = {e["event_type"] for e in events}
    synth_payloads = [p for purpose, p in getattr(provider, "calls", []) if purpose == "synthesis"]
    memory_context = synth_payloads[-1]["evidence"] if synth_payloads else []
    return {
        "case": case,
        "state": state,
        "agent": agent,
        "routed": routed,
        "post_status": post_status,
        "events": events,
        "event_names": event_names,
        "provider_calls": provider_calls,
        "adapter_calls": adapter_calls,
        "memory_context": memory_context,
    }


def run_evals(root):
    cases = [
        json.loads(x)
        for x in (root / "evals/cases.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]
    if len({(c["task"], c["scenario_id"], c.get("v2_mode")) for c in cases}) != len(cases):
        raise ValueError("eval cases must be unique")
    runs = [_run_case(root, c, i) for i, c in enumerate(cases)]
    successes = []
    tool_scores = []
    evidence_scores = []
    claims = []
    hallucinations = []
    provider_recovery = []
    tool_recovery = []
    review_scores = []
    memory_scores = []
    trace_scores = []
    for run in runs:
        c, s, a = run["case"], run["state"], run["agent"]
        tools = {x.name for x in s.tool_calls} | {
            e.get("tool") for e in run["events"] if e["event_type"] in {"tool_call", "tool_error"}
        }
        ids = {e.evidence_id for e in s.evidence}
        linked = {eid for claim in s.claims for eid in claim.evidence_ids}
        tool_ok = set(c["expected_tools"]).issubset(tools) and not (
            set(c["forbidden_tools"]) & tools
        )
        evidence_ok = set(c["required_evidence"]).issubset(linked) and linked.issubset(ids)
        keyword_ok = all(
            k.lower() in (s.final_answer or "").lower() for k in c["expected_claim_keywords"]
        )
        uncertainty_ok = c["allowed_uncertainty"] or not any(q.uncertain for q in s.claims)
        mode = c.get("v2_mode", "")
        behavior_ok = True
        if mode == "provider_recovery":
            behavior_ok = (
                s.status == Status.COMPLETE
                and "retry" in run["event_names"]
                and len(run["provider_calls"]) > 1
            )
        elif mode == "provider_invalid_json":
            behavior_ok = s.status == Status.COMPLETE and len(run["provider_calls"]) > 1
        elif mode == "provider_retry_exhausted":
            behavior_ok = s.status == Status.FAILED and len(run["provider_calls"]) > 1
        elif mode == "tool_retry":
            behavior_ok = (
                s.status == Status.COMPLETE
                and len(run["adapter_calls"]) > 1
                and "retry" in run["event_names"]
            )
        elif mode in {"tool_exhausted", "review_required", "circuit_open"}:
            behavior_ok = run["routed"] and "tool_error" in run["event_names"]
        elif mode == "memory_hit":
            behavior_ok = "memory_hit" in run["event_names"] and any(
                e.get("provenance") == "memory" for e in run["memory_context"]
            )
        elif mode == "memory_miss":
            behavior_ok = "memory_miss" in run["event_names"]
        elif mode == "stale_memory":
            behavior_ok = run["routed"] and any(
                "current evidence retained" in d for d in s.decisions
            )
        elif mode == "review_approve":
            behavior_ok = run["routed"] and run["post_status"] == Status.COMPLETE
        elif mode == "review_reject":
            behavior_ok = run["routed"] and run["post_status"] == Status.FAILED
        elif mode in {"bm25", "hybrid"}:
            behavior_ok = a.registry.retriever.backend == mode and bool(
                a.registry.retriever.search(c["task"])
            )
        successes.append(tool_ok and evidence_ok and keyword_ok and uncertainty_ok and behavior_ok)
        tool_scores.append(tool_ok)
        evidence_scores.append(evidence_ok)
        claims.extend(s.claims)
        hallucinations.extend(any(eid not in ids for eid in q.evidence_ids) for q in s.claims)
        if mode in {"provider_recovery", "provider_invalid_json"}:
            provider_recovery.append(behavior_ok)
        if mode == "tool_retry":
            tool_recovery.append(behavior_ok)
        if "expected_review" in c:
            review_scores.append(run["routed"] == c["expected_review"])
        if mode in {"memory_hit", "memory_miss", "stale_memory"}:
            memory_items = [e for e in run["memory_context"] if e.get("provenance") == "memory"]
            current_items = [e for e in run["memory_context"] if e.get("provenance") == "current"]
            expected_hit = mode != "memory_miss"
            memory_scores.append(
                (bool(memory_items) == expected_hit)
                and bool(current_items)
                and (mode != "stale_memory" or run["routed"])
            )
        trace_scores.append(_trace_ok(a, run["routed"]))
    total = max(1, len(claims))
    report = {
        "cases": len(cases),
        "unique_cases": len(cases),
        "v1_regression_success": sum(successes[:16]) / 16,
        "v2_task_success": sum(successes[16:]) / max(1, len(successes[16:])),
        "task_success_rate": sum(successes) / len(successes),
        "tool_selection_accuracy": sum(tool_scores) / len(tool_scores),
        "evidence_coverage": sum(evidence_scores) / len(evidence_scores),
        "unsupported_claim_rate": sum(q.unsupported for q in claims) / total,
        "hallucinated_source_rate": sum(hallucinations) / total,
        "provider_error_recovery_rate": sum(provider_recovery) / max(1, len(provider_recovery)),
        "tool_failure_recovery_rate": sum(tool_recovery) / max(1, len(tool_recovery)),
        "review_routing_accuracy": sum(review_scores) / max(1, len(review_scores)),
        "memory_provenance_accuracy": sum(memory_scores) / max(1, len(memory_scores)),
        "trace_completeness": sum(trace_scores) / len(trace_scores),
    }
    (root / "reports/v1_eval.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    (root / "reports/v1_eval.md").write_text(
        "\n".join(f"- {k}: {v}" for k, v in report.items()), encoding="utf8"
    )
    return report
