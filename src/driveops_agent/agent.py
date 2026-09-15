from pathlib import Path

from .models import AgentState, Claim, Status, ToolCall
from .planner import make_plan, revise_plan
from .providers import MockProvider
from .tools import ToolRegistry
from .verification import verify


class DriveOpsAgent:
    def __init__(self, data_dir: Path, reports_dir: Path, provider=None, max_steps=12):
        self.registry = ToolRegistry(data_dir, reports_dir)
        self.provider = provider or MockProvider()
        self.max_steps = max_steps
        self.cache = {}

    def _call(self, state, name, args):
        if state.step_count >= self.max_steps:
            raise RuntimeError("maximum execution steps reached")
        key = (name, repr(sorted(args.items())))
        if key in self.cache:
            value, evidence = self.cache[key]
            cached = True
        else:
            value, evidence = self.registry.call(name, args)
            self.cache[key] = (value, evidence)
            cached = False
        state.tool_calls.append(ToolCall(name=name, arguments=args, cached=cached))
        state.evidence.extend(evidence)
        state.step_count += 1
        state.observations.append(f"{name}: {value}")
        return value

    def run(self, task):
        state = AgentState(user_goal=task)
        try:
            scenario, steps = make_plan(task, self.provider)
            state.current_plan = steps
            results = {}
            for step in state.current_plan:
                value = self._call(state, step.tool_name, step.arguments)
                results[step.tool_name] = value
                step.status = "complete"
                state.completed_steps.append(step.objective)
                if isinstance(value, dict) and value.get("status") == "not found":
                    revise_plan(state, f"{step.tool_name} not found")
            log = results.get("read_log", {})
            docs = [e for e in state.evidence if e.tool == "search_docs"]
            dtcs = [e for e in state.evidence if e.tool == "query_dtcs"]
            metrics = [e for e in state.evidence if e.tool == "calculate_metric"]
            root = log.get("root_cause", "unknown")
            ev = ["log-" + scenario] + [e.evidence_id for e in dtcs + docs + metrics]
            if log.get("failsafe"):
                text = f"{root} is the most likely root cause for failsafe in {scenario}."
                state.claims.append(
                    Claim(claim_id="root-cause", text=text, evidence_ids=ev, confidence=0.88)
                )
            elif root == "no-fault":
                state.claims.append(
                    Claim(
                        claim_id="no-fault",
                        text="No fault: there is not enough evidence to support a fault in the normal run.",
                        evidence_ids=["log-" + scenario],
                        confidence=0.9,
                    )
                )
            else:
                state.claims.append(
                    Claim(
                        claim_id="finding",
                        text=f"{root} is observed in {scenario}; failsafe was not asserted.",
                        evidence_ids=ev,
                        confidence=0.75,
                        uncertain=True,
                    )
                )
            if metrics:
                m = metrics[0]
                val = results["calculate_metric"]["value"]
                state.claims.append(
                    Claim(
                        claim_id="metric",
                        text=f"Maximum pressure gap is {val} bar.",
                        evidence_ids=[m.evidence_id],
                        confidence=0.9,
                    )
                )
            self.provider.complete(
                "synthesis",
                {
                    "task": task,
                    "evidence": [e.model_dump() for e in state.evidence],
                    "claims": [c.text for c in state.claims],
                },
            )
            verdict = verify(state)
            state.final_answer = " ".join(
                f"[{c.claim_id}] {c.text} (confidence={c.confidence:.2f})" for c in verdict.claims
            )
            state.status = Status.UNCERTAIN if verdict.unsupported_claims else Status.COMPLETE
        except Exception as exc:
            state.errors.append(str(exc))
            state.status = Status.FAILED
        return state
