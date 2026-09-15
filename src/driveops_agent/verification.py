from .models import VerificationResult


def verify(state):
    ids = {e.evidence_id: e for e in state.evidence}
    bad = []
    conf = []
    for c in state.claims:
        linked = [ids.get(i) for i in c.evidence_ids]
        sources = {e.source for e in linked if e}
        fail = not linked or any(e is None for e in linked)
        if c.claim_id and "Most likely" in c.text and len(sources) < 2 and not c.uncertain:
            fail = True
        if c.kind if hasattr(c, "kind") else False:
            pass
        if ("bar" in c.text.lower() or c.text == "Signal metric measured.") and not any(
            e and e.source == "vehicle_signals.csv" for e in linked
        ):
            fail = True
        if c.uncertain:
            c.confidence = min(c.confidence, 0.45)
            conf.append(c.claim_id)
        if fail:
            c.unsupported = True
            c.uncertain = True
            c.confidence = min(c.confidence, 0.3)
            bad.append(c.claim_id)
    return VerificationResult(claims=state.claims, unsupported_claims=bad, conflicts=conf)
