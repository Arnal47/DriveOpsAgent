from .models import AgentState, VerificationResult


def verify(state: AgentState) -> VerificationResult:
    claims = [x for x in (state.final_answer or "").split(".") if x.strip()]
    unsupported = [] if state.evidence else claims
    conflicts = (
        ["Conflicting evidence detected"]
        if "normal" in (state.final_answer or "").lower()
        and "failsafe" in (state.final_answer or "").lower()
        else []
    )
    return VerificationResult(
        supported=not unsupported and not state.errors,
        unsupported_claims=unsupported,
        conflicts=conflicts,
        incomplete_plan=len(state.completed_steps) < 3,
    )
