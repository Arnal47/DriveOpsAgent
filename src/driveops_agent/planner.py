from .models import AgentState


def make_plan(goal: str) -> list[str]:
    base = [
        "Read target test log",
        "Find anomaly time points",
        "Query relevant DTCs",
        "Retrieve validation documentation",
        "Analyze vehicle signals",
        "Rank root-cause candidates",
        "Verify evidence",
        "Produce evidence-linked answer",
    ]
    if "对比" in goal or "compare" in goal.lower():
        base[0] = "Read both target test logs"
        base[1] = "Compare anomaly time points"
    return base


def revise_plan(state: AgentState, note: str) -> None:
    state.current_plan.append(f"Revised: {note}")
