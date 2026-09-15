from pathlib import Path

from .models import AgentState, Claim, Status, ToolCall
from .planner import make_plan, revise_plan
from .providers import MockProvider
from .tools import ToolRegistry
from .verification import verify


class DriveOpsAgent:
    def __init__(self, data_dir: Path, reports_dir: Path, provider=None, max_steps=16):
        self.registry = ToolRegistry(data_dir, reports_dir)
        self.provider = provider or MockProvider()
        self.max_steps = max_steps
        self.cache = {}

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
        s.evidence += e
        s.step_count += 1
        s.observations.append(f"{n}: {v}")
        return v

    def run(self, task):
        s = AgentState(user_goal=task)
        try:
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
            out = self.provider.complete("synthesis", {"task": task, "evidence": context})
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
            s.status = Status.UNCERTAIN if verdict.unsupported_claims else Status.COMPLETE
        except Exception as e:
            s.errors.append(str(e))
            s.status = Status.FAILED
        return s
