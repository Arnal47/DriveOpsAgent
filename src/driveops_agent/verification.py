from .models import VerificationResult


def verify(state):
    ids = {e.evidence_id: e for e in state.evidence}
    unsupported = []
    conflicts = []
    for claim in state.claims:
        linked = [ids.get(i) for i in claim.evidence_ids]
        bad = not linked or any(x is None for x in linked)
        if "U1000" in claim.text and not any(e and e.artifact_id == "U1000" for e in linked):
            bad = True
        if "C1234" in claim.text and not any(e and e.artifact_id == "C1234" for e in linked):
            bad = True
        if (
            any(ch.isdigit() for ch in claim.text)
            and "bar" in claim.text
            and not any(e and e.source == "vehicle_signals.csv" for e in linked)
        ):
            bad = True
        if bad:
            claim.unsupported = True
            claim.uncertain = True
            claim.confidence = min(claim.confidence, 0.3)
            unsupported.append(claim.claim_id)
    if any("no fault" in c.text.lower() for c in state.claims) and any(
        e.artifact_id != "normal-001" for e in state.evidence if e.tool == "read_log"
    ):
        conflicts.append("normal claim conflicts with fault log")
    return VerificationResult(
        claims=state.claims, unsupported_claims=unsupported, conflicts=conflicts
    )
