from .models import PlanStep


def make_plan(task, provider):
    result = provider.complete("plan", {"task": task})
    return result["scenario_id"], [PlanStep(**s) for s in result["steps"]]


def revise_plan(state, observation):
    state.decisions.append("plan revised: " + observation)
    if "not found" in observation:
        state.observations.append("revision: catalog evidence unavailable; lower confidence")
