from .models import PlanStep


def make_plan(task, provider, artifact_ids):
    r = provider.complete(
        "plan",
        {
            "task": task,
            "artifact_ids": artifact_ids,
            "tool_schemas": [
                "read_log",
                "query_dtcs",
                "search_docs",
                "calculate_metric",
                "generate_report",
            ],
        },
    )
    return r["intent"], [PlanStep(**x) for x in r["steps"]]


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
