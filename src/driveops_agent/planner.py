from .models import PlanStep


def make_plan(task, provider, artifact_ids):
    result = provider.complete(
        "plan",
        {
            "task": task,
            "artifact_ids": artifact_ids,
            "scenario_hints": {
                "wheel": "wheel-speed-002",
                "speed": "wheel-speed-002",
                "pressure": "pressure-003",
                "under-response": "pressure-003",
                "can": "can-timeout-004",
                "timeout": "can-timeout-004",
                "u1000": "can-timeout-004",
                "c1234": "pressure-003",
                "c0035": "wheel-speed-002",
            },
            "tool_schemas": [
                "read_log",
                "query_dtcs",
                "search_docs",
                "calculate_metric",
                "generate_report",
            ],
        },
    )
    return result["intent"], [PlanStep(**step) for step in result["steps"]]


def revise_plan(state, reason, scenario_id="normal-001"):
    state.decisions.append("plan revised: " + reason)
    tool = "search_docs" if "missing" in reason or "failure" in reason else "calculate_metric"
    args = (
        {"query": "fallback validation"}
        if tool == "search_docs"
        else {"scenario_id": scenario_id, "column": "pressure_gap_bar", "operation": "max"}
    )
    state.current_plan.append(
        PlanStep(
            objective="revision verification",
            tool_name=tool,
            arguments=args,
            expected_observation="revision evidence",
        )
    )
