import time
from pathlib import Path
from types import SimpleNamespace

from .memory.store import MemoryStore
from .models import AgentState, Claim, Evidence, Status, ToolCall
from .planner import make_plan, revise_plan
from .providers import MockProvider
from .tools import ToolRegistry
from .tracing import Trace
from .verification import verify


class DriveOpsAgent:
    def __init__(
        self,
        data_dir: Path,
        reports_dir: Path,
        provider=None,
        max_steps=16,
        external_adapter=None,
        retrieval_backend="tfidf",
    ):
        self.registry = ToolRegistry(data_dir, reports_dir, retrieval_backend=retrieval_backend)
        self.external_adapter = external_adapter
        self.provider = provider or MockProvider()
        self.max_steps = max_steps
        self.cache = {}
        self.memory = MemoryStore(reports_dir / "driveops_memory.sqlite")
        self.trace = None

    def _provider_complete(self, purpose, payload):
        last = None
        for attempt in range(2):
            start = time.monotonic()
            try:
                result = self.provider.complete(purpose, payload)
                self.trace.emit(
                    "provider_call", tool=purpose, latency_ms=(time.monotonic() - start) * 1000
                )
                return result
            except Exception as exc:
                last = exc
                self.trace.emit(
                    "provider_call",
                    tool=purpose,
                    latency_ms=(time.monotonic() - start) * 1000,
                    success=False,
                    error=str(exc),
                )
                if attempt == 0:
                    self.trace.emit("retry", tool=purpose, success=False, error=str(exc))
        raise last

    def _call(self, s, name, args):
        if s.step_count >= self.max_steps:
            raise RuntimeError("maximum execution steps reached")
        key = (name, repr(sorted(args.items())))
        start = time.monotonic()
        try:
            if key in self.cache:
                value, evidence = self.cache[key]
                cached = True
            else:
                value, evidence = self.registry.call(name, args)
                self.cache[key] = (value, evidence)
                cached = False
            s.tool_calls.append(ToolCall(name=name, arguments=args, cached=cached))
            s.evidence += evidence
            s.step_count += 1
            s.observations.append(f"{name}: {value}")
            self.trace.emit(
                "tool_call",
                tool=name,
                arguments=args,
                latency_ms=(time.monotonic() - start) * 1000,
                evidence_ids=[e.evidence_id for e in evidence],
            )
            return value
        except Exception as exc:
            s.tool_calls.append(ToolCall(name=name, arguments=args, success=False))
            s.errors.append(str(exc))
            s.step_count += 1
            self.trace.emit(
                "tool_error",
                tool=name,
                arguments=args,
                latency_ms=(time.monotonic() - start) * 1000,
                success=False,
                error=str(exc),
            )
            raise

    def _persist(self, s):
        summary = self.trace.summary()
        scenario = next(
            (c.arguments.get("log_id", "") for c in s.tool_calls if c.name == "read_log"), ""
        )
        self.trace.emit("memory_save")
        self.memory.save(
            s.run_id,
            summary,
            [e.model_dump() for e in s.evidence],
            s.decisions,
            [c.model_dump() for c in s.tool_calls],
            task=s.user_goal,
            scenario_id=scenario,
            status=s.status.value,
        )
        if s.status == Status.NEEDS_REVIEW:
            self.memory.save_review(s.run_id, "pending", s.model_dump(mode="json"))
        self.trace.emit(
            "run_end",
            success=s.status not in {Status.FAILED, Status.NEEDS_REVIEW},
            error=None if s.status != Status.FAILED else "; ".join(s.errors),
            evidence_ids=[e.evidence_id for e in s.evidence],
        )

    def review(self, run_id, approve):
        saved = self.memory.load_review(run_id)
        if not saved or saved["state"] != "pending":
            raise KeyError("pending review not found")
        state = AgentState.model_validate(saved["payload"])
        state.status = Status.COMPLETE if approve else Status.FAILED
        self.memory.resolve_review(run_id, "approved" if approve else "rejected")
        trace = Trace(self.registry.reports_dir / "traces")
        trace.run_id = run_id
        trace.path = trace.path.with_name(run_id + ".jsonl")
        trace.emit(
            "review_approved" if approve else "review_rejected",
            success=approve,
            error=None if approve else "review rejected: safe termination",
        )
        return state

    def run(self, task):
        self.trace = Trace(self.registry.reports_dir / "traces")
        s = AgentState(run_id=self.trace.run_id, user_goal=task)
        self.trace.emit("run_start")
        start = time.monotonic()
        memories = self.memory.find_similar(task)
        self.trace.emit("memory_lookup", latency_ms=(time.monotonic() - start) * 1000)
        self.trace.emit("memory_hit" if memories else "memory_miss")
        memory_evidence = []
        if memories:
            s.decisions.append("memory context available; current evidence remains authoritative")
        for item in memories:
            for raw in item["evidence"]:
                try:
                    memory_evidence.append(Evidence.model_validate(raw))
                except Exception as exc:
                    s.decisions.append(f"ignored invalid memory evidence: {exc}")
        if memory_evidence:
            s.decisions.append(
                f"retrieved {len(memory_evidence)} historical evidence items with provenance=memory; current evidence is authoritative"
            )
        try:
            proxy = SimpleNamespace(complete=self._provider_complete)
            intent, plan = make_plan(
                task, proxy, ["normal-001", "wheel-speed-002", "pressure-003", "can-timeout-004"]
            )
            s.current_plan = plan
            if self.external_adapter is not None:
                start = time.monotonic()
                try:
                    result = self.external_adapter.request("health", {})
                    attempts = getattr(self.external_adapter, "last_attempts", 1)
                    for _ in range(max(0, attempts - 1)):
                        self.trace.emit(
                            "retry", tool="external.health", success=False, error="adapter retry"
                        )
                    self.trace.emit(
                        "tool_call",
                        tool="external.health",
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
                    s.observations.append(f"external.health: {result}")
                except Exception as exc:
                    attempts = getattr(self.external_adapter, "last_attempts", 1)
                    for _ in range(max(0, attempts - 1)):
                        self.trace.emit(
                            "retry", tool="external.health", success=False, error=str(exc)
                        )
                    self.trace.emit(
                        "tool_error",
                        tool="external.health",
                        latency_ms=(time.monotonic() - start) * 1000,
                        success=False,
                        error=str(exc),
                    )
                    s.errors.append(str(exc))
                    s.status = Status.NEEDS_REVIEW
                    self.trace.emit("review_required", error="external adapter failure")
                    self._persist(s)
                    return s
            proxy = SimpleNamespace(complete=self._provider_complete)
            intent, plan = make_plan(
                task, proxy, ["normal-001", "wheel-speed-002", "pressure-003", "can-timeout-004"]
            )
            s.current_plan = plan
            i = 0
            while i < len(s.current_plan):
                step = s.current_plan[i]
                value = self._call(s, step.tool_name, step.arguments)
                step.status = "complete"
                s.completed_steps.append(step.objective)
                if step.tool_name == "read_log":
                    if value.get("status") == "not found":
                        revise_plan(s, "tool failure", step.arguments["log_id"])
                    elif value.get("dtcs"):
                        s.current_plan.append(
                            type(step)(
                                objective="resolve observed DTC",
                                tool_name="query_dtcs",
                                arguments={"code": value["dtcs"][0]},
                                expected_observation="catalog evidence",
                            )
                        )
                if "conflict" in task.lower() and step.tool_name == "read_log":
                    revise_plan(s, "conflict evidence", step.arguments["log_id"])
                i += 1
            current_by_key = {(e.source, e.artifact_id, e.field): e for e in s.evidence}
            stale = False
            for old in memory_evidence:
                current = current_by_key.get((old.source, old.artifact_id, old.field))
                if current and (old.snippet != current.snippet or old.value != current.value):
                    stale = True
            context = [
                {**e.model_dump(), "untrusted": e.source.endswith(".md")} for e in s.evidence
            ]
            context += [{**e.model_dump(), "untrusted": True} for e in memory_evidence]
            out = self._provider_complete("synthesis", {"task": task, "evidence": context})
            s.claims = [
                Claim(
                    claim_id=f"claim-{i}",
                    text=c["text"],
                    evidence_ids=c["evidence_ids"],
                    confidence=c["confidence"],
                    value=c.get("value"),
                    uncertain=stale or "conflict" in task.lower(),
                )
                for i, c in enumerate(out.get("claims", []), 1)
            ]
            if not any(e.tool == "read_log" for e in s.evidence):
                s.claims = [
                    Claim(
                        claim_id="insufficient-log",
                        text="Insufficient evidence: requested test log was not found.",
                        evidence_ids=[],
                        confidence=0.2,
                        uncertain=True,
                    )
                ]
            verdict = verify(s)
            s.final_answer = (
                out.get("prefix", "Synthesis")
                + ": "
                + " ".join(
                    f"[{c.claim_id}] {c.text} (confidence={c.confidence:.2f})" for c in s.claims
                )
            )
            if intent == "report":
                self._call(
                    s,
                    "generate_report",
                    {
                        "findings": s.final_answer
                        + "\nEvidence: "
                        + ", ".join(e.evidence_id for e in s.evidence)
                        + "\nUncertainty: "
                        + str(any(c.uncertain for c in s.claims))
                    },
                )
            if stale or "conflict" in task.lower():
                s.decisions.append("stale/conflicting memory detected; current evidence retained")
                s.status = Status.NEEDS_REVIEW
                self.trace.emit("review_required", error="stale or conflicting evidence")
            else:
                s.status = (
                    Status.UNCERTAIN
                    if verdict.unsupported_claims or any(c.uncertain for c in s.claims)
                    else Status.COMPLETE
                )
        except Exception as exc:
            s.errors.append(str(exc))
            s.status = Status.NEEDS_REVIEW if self.external_adapter is not None else Status.FAILED
            self.trace.emit(
                "review_required" if s.status == Status.NEEDS_REVIEW else "error",
                success=False,
                error=str(exc),
            )
        self._persist(s)
        return s
