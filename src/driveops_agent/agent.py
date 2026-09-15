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
            logs = [
                self.registry.call("read_log", {"log_id": x})[0]
                for x in ["normal-001", "wheel-speed-002", "pressure-003", "can-timeout-004"]
            ]
            _intent, scenario, plan = make_plan(
                task, self.provider, [x for x in logs if x.get("status") != "not found"]
            )
            s.current_plan = plan
            i = 0
            results = {}
            while i < len(s.current_plan):
                st = s.current_plan[i]
                v = self._call(s, st.tool_name, st.arguments)
                results.setdefault(st.tool_name, []).append(v)
                st.status = "complete"
                s.completed_steps.append(st.objective)
                if isinstance(v, dict) and v.get("status") == "not found":
                    revise_plan(s, "tool not found", scenario)
                i += 1
            log = results.get("read_log", [{}])[0]
            dtc = results.get("query_dtcs", [{}])[0]
            docs = [e for e in s.evidence if e.tool == "search_docs"]
            dtcev = [e for e in s.evidence if e.tool == "query_dtcs"]
            logev = [e for e in s.evidence if e.tool == "read_log"]
            met = [e for e in s.evidence if e.tool == "calculate_metric"]
            if log.get("status") == "not found":
                s.claims = [
                    Claim(
                        claim_id="insufficient",
                        text="Insufficient evidence: requested log was not found.",
                        evidence_ids=[],
                        confidence=0.2,
                        uncertain=True,
                    )
                ]
            elif not log.get("dtcs"):
                s.claims = [
                    Claim(
                        claim_id="no-fault",
                        text="No fault: insufficient evidence supports a fault in this normal run.",
                        evidence_ids=[logev[0].evidence_id],
                        confidence=0.9,
                    )
                ]
            else:
                description = dtc.get("description", "unresolved DTC")
                label = description.split(".")[0].lower()
                ids = [logev[0].evidence_id] + [e.evidence_id for e in dtcev + docs]
                s.claims = [
                    Claim(
                        claim_id="root-cause",
                        text=f"Most likely root cause: {label}.",
                        evidence_ids=ids,
                        confidence=0.82,
                    )
                ]
                if log.get("failsafe"):
                    s.claims.append(
                        Claim(
                            claim_id="failsafe",
                            text="Failsafe activation is supported by the log events.",
                            evidence_ids=[logev[0].evidence_id],
                            confidence=0.9,
                        )
                    )
            if met:
                value = results["calculate_metric"][0]["value"]
                s.claims.append(
                    Claim(
                        claim_id="metric",
                        text=f"Maximum pressure gap is {value} bar.",
                        evidence_ids=[met[0].evidence_id],
                        confidence=0.9,
                    )
                )
            if "conflict" in task.lower():
                revise_plan(s, "evidence conflict", scenario)
            synthesis = self.provider.complete(
                "synthesis",
                {
                    "task": task,
                    "evidence": [e.model_dump() for e in s.evidence],
                    "claims": [c.text for c in s.claims],
                },
            )
            verdict = verify(s)
            s.final_answer = (
                synthesis.get("prefix", "Synthesis")
                + ": "
                + " ".join(
                    f"[{c.claim_id}] {c.text} (confidence={c.confidence:.2f})"
                    for c in verdict.claims
                )
            )
            s.status = Status.UNCERTAIN if verdict.unsupported_claims else Status.COMPLETE
        except Exception as exc:
            s.errors.append(str(exc))
            s.status = Status.FAILED
        return s
