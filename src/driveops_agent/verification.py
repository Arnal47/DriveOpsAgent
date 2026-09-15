from .models import VerificationResult


def verify(state):
    ids = {e.evidence_id: e for e in state.evidence}
    bad, conflicts = [], []
    for claim in state.claims:
        linked = [ids.get(eid) for eid in claim.evidence_ids]
        sources = {e.source for e in linked if e}
        failed = not linked or any(e is None for e in linked)
        if "Most likely" in claim.text and len(sources) < 2 and not claim.uncertain:
            failed = True
        if claim.value is None and "bar" in claim.text.lower() and not any(e and e.source == "vehicle_signals.csv" for e in linked):`n            failed = True`n        if claim.value is not None:
            metric_values = [e.value for e in linked if e and e.source == "vehicle_signals.csv"]
            if metric_values != [claim.value]:
                failed = True
        if claim.uncertain:
            claim.confidence = min(claim.confidence, 0.45)
            conflicts.append(claim.claim_id)
        if failed:
            claim.unsupported = True
            claim.uncertain = True
            claim.confidence = min(claim.confidence, 0.3)
            bad.append(claim.claim_id)
    return VerificationResult(claims=state.claims, unsupported_claims=bad, conflicts=conflicts)
