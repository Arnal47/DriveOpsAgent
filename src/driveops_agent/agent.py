import json
from pathlib import Path

from .memory.store import MemoryStore
from .models import AgentState, Claim, Status, ToolCall
from .planner import make_plan, revise_plan
from .providers import MockProvider
from .tools import ToolRegistry
from .tracing import Trace
from .verification import verify


class DriveOpsAgent:
    def __init__(self, data_dir: Path, reports_dir: Path, provider=None, max_steps=16):
        self.registry = ToolRegistry(data_dir, reports_dir)
        self.provider = provider or MockProvider()
        self.max_steps = max_steps
        self.cache = {}
        self.memory = MemoryStore(reports_dir / "driveops_memory.sqlite")
        self.trace = None
        self.pending_reviews = {}

    def _provider_complete(self, purpose, payload):
        for attempt in range(2):
            try:
                start = __import__("time").monotonic()
                result = self.provider.complete(purpose, payload)
                self.trace.emit(
                    "provider_call",
                    tool=purpose,
                    latency_ms=(__import__("time").monotonic() - start) * 1000,
                )
                return result
            except Exception as exc:
                self.trace.emit("retry", tool=purpose, success=False, error=str(exc))
                if attempt == 1:
                    raise

    def _call(self, s, n, a):
        if s.step_count >= self.max_steps:
            raise RuntimeError("maximum execution steps reached")
        k = (n, repr(sorted(a.items())))
        if k in self.cache:
            v, e = self.cache[k]
            cached = True
        else:
            v, e = self.registry.call(n, a)
            self.cache[k] = (v, e)
            cached = False
        s.tool_calls.append(ToolCall(name=n, arguments=a, cached=cached))
        if self.trace:
            self.trace.emit(
                "tool_call", tool=n, arguments=a, evidence_ids=[x.evidence_id for x in e]
            )
        s.evidence += e
        s.step_count += 1
        s.observations.append(f"{n}: {v}")
        return v

    def review(self, run_id, approve):
        state = self.pending_reviews.pop(run_id)
        state.status = Status.COMPLETE if approve else Status.FAILED
        if self.trace and self.trace.run_id == run_id:
            self.trace.emit("review_approved" if approve else "review_rejected", success=approve)
        return state

    def run(self, task):
        self.trace = Trace(self.registry.reports_dir / "traces")
        s = AgentState(run_id=self.trace.run_id, user_goal=task)
        self.trace.emit("run_start")
        prior = self.memory.list()
        self.trace.emit("memory_lookup")
        self.trace.emit("memory_hit" if prior else "memory_miss")
        try:
            self.trace.emit("provider_call", tool="planner")
            intent, plan = make_plan(
                task,
                self.provider,
                ["normal-001", "wheel-speed-002", "pressure-003", "can-timeout-004"],
            )
            s.current_plan = plan
            i = 0
            while i < len(s.current_plan):
                st = s.current_plan[i]
                v = self._call(s, st.tool_name, st.arguments)
                st.status = "complete"
                s.completed_steps.append(st.objective)
                if st.tool_name == "read_log":
                    if v.get("status") == "not found":
                        revise_plan(s, "tool failure", st.arguments["log_id"])
                    elif v.get("dtcs"):
                        s.current_plan.append(
                            type(st)(
                                objective="resolve observed DTC",
                                tool_name="query_dtcs",
                                arguments={"code": v["dtcs"][0]},
                                expected_observation="catalog evidence",
                            )
                        )
                if "conflict" in task.lower() and st.tool_name == "read_log":
                    revise_plan(s, "conflict evidence", st.arguments["log_id"])
                i += 1
            context = []
            for e in s.evidence:
                d = e.model_dump()
                d["untrusted"] = e.source.endswith(".md")
                context.append(d)
            self.trace.emit("provider_call", tool="synthesis")
            out = self._provider_complete("synthesis", {"task": task, "evidence": context})
            s.claims = [
                Claim(
                    claim_id=f"claim-{i}",
                    text=c["text"],
                    evidence_ids=c["evidence_ids"],
                    confidence=c["confidence"],
                    value=c.get("value"),
                    uncertain="conflict" in task.lower(),
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
                findings = (
                    s.final_answer
                    + "\nEvidence: "
                    + ", ".join(e.evidence_id for e in s.evidence)
                    + "\nUncertainty: "
                    + str(any(c.uncertain for c in s.claims))
                )
                self._call(s, "generate_report", {"findings": findings})
            if "conflict" in task.lower():
                s.status = Status.NEEDS_REVIEW
                self.pending_reviews[s.run_id] = s
                self.trace.emit("review_required")
            else:
                s.status = (
                    Status.UNCERTAIN
                    if verdict.unsupported_claims or any(c.uncertain for c in s.claims)
                    else Status.COMPLETE
                )
        except Exception as e:
            s.errors.append(str(e))
            s.status = Status.FAILED
        if self.trace:
            self.trace.emit(
                "run_end",
                success=s.status != Status.FAILED,
                evidence_ids=[e.evidence_id for e in s.evidence],
            )
            self.trace.emit("memory_save")
            self.memory.save(
                s.run_id,
                json.dumps(self.trace.summary()),
                [e.model_dump() for e in s.evidence],
                s.decisions,
            )
        return s
