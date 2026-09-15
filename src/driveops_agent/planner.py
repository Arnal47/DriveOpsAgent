from .models import PlanStep


def make_plan(task, provider, scenarios):
    result = provider.complete("plan", {"task": task, "scenarios": scenarios})
    return result["intent"], result["scenario_id"], [PlanStep(**s) for s in result["steps"]]


def revise_plan(state, reason, scenario_id="normal-001"):
    state.decisions.append("plan revised: " + reason)
    if "not found" in reason:
        state.current_plan.append(
            PlanStep(
                objective="fallback documentation retrieval",
                tool_name="search_docs",
                arguments={"query": "diagnostic fallback"},
                expected_observation="fallback evidence",
            )
        )
    if "conflict" in reason:
        state.current_plan.append(
            PlanStep(
                objective="validate conflicting signals",
                tool_name="calculate_metric",
                arguments={
                    "scenario_id": scenario_id,
                    "column": "pressure_gap_bar",
                    "operation": "max",
                },
                expected_observation="conflict metric",
            )
        )
    if "missing evidence" in reason:
        state.current_plan.append(
            PlanStep(
                objective="retrieve missing evidence",
                tool_name="search_docs",
                arguments={"query": "validation evidence"},
                expected_observation="additional evidence",
            )
        )
