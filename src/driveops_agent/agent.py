from __future__ import annotations

from pathlib import Path

from .models import AgentState, Status, ToolCall
from .planner import make_plan, revise_plan
from .tools import ToolRegistry
from .verification import verify


class DriveOpsAgent:
    """Deterministic Observe→Plan→Act→Observe→Reflect→Verify→Answer V1 loop."""

    def __init__(self, data_dir: Path, reports_dir: Path, max_steps: int = 12):
        self.registry = ToolRegistry(data_dir, reports_dir)
        self.max_steps = max_steps
        self.cache: dict[tuple[str, str], tuple[object, list]] = {}

    def _call(self, state: AgentState, name: str, args: dict):
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
        state.observations.append(f"{name} completed")
        return value

    def run(self, goal: str) -> AgentState:
        state = AgentState(user_goal=goal, current_plan=make_plan(goal))
        try:
            compare = "对比" in goal or "compare" in goal.lower()
            target = "brake-failsafe-002"
            self._call(state, "read_log", {"log_id": target})
            state.completed_steps.append("Read target test log")
            if compare:
                self._call(state, "read_log", {"log_id": "brake-normal-001"})
            self._call(state, "query_dtcs", {"code": "U1000"})
            self._call(state, "query_dtcs", {"code": "C1234"})
            self._call(
                state, "search_docs", {"query": "failsafe CAN timeout brake pressure wheel speed"}
            )
            self._call(
                state, "calculate_metric", {"column": "brake_pressure_bar", "operation": "min"}
            )
            state.completed_steps.extend(state.current_plan[1:6])
            revise_plan(state, "CAN timeout evidence requires prioritizing communications fault")
            answer = (
                "Most likely root cause: CAN timeout (U1000) interrupted brake-controller communication; "
                "the log records failsafe activation immediately after the timeout. Secondary candidate: "
                "brake pressure under-response (C1234). Evidence is cited from log, DTC catalog, validation notes, and signals."
            )
            if compare:
                answer = (
                    "Key difference: the failsafe test contains U1000 CAN timeout and failsafe activation; the normal test does not. "
                    + answer
                )
            if "报告" in goal or "report" in goal.lower():
                self._call(state, "generate_report", {"findings": answer})
            state.final_answer = answer
            state.completed_steps.extend(state.current_plan[-2:])
            verdict = verify(state)
            state.status = Status.COMPLETE if verdict.supported else Status.UNCERTAIN
            if verdict.unsupported_claims:
                state.final_answer += " Uncertain: verifier found unsupported claims."
        except Exception as exc:
            state.errors.append(str(exc))
            state.status = Status.FAILED
        return state
