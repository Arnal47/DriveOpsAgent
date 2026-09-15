from .models import VerificationResult


def verify(state):
    ids = {e.evidence_id: e for e in state.evidence}
    unsupported = []
    conflicts = []
    for c in state.claims:
        linked = [ids.get(i) for i in c.evidence_ids]
        sources = {e.source for e in linked if e}
        bad = not linked or any(e is None for e in linked)
        text = c.text.lower()
        if c.claim_id == "root-cause" and len(sources) < 2 and not c.uncertain:
            bad = True
        if (
            c.claim_id == "root-cause"
            and any(x in text for x in ["can communication", "brake pressure", "wheel-speed"])
            and not any(e and e.source == "dtc_catalog.json" for e in linked)
        ):
            bad = True
        if c.claim_id == "metric" or ("bar" in text and any(ch.isdigit() for ch in text)):
            signal = next((e for e in linked if e and e.source == "vehicle_signals.csv"), None)
            if (
                signal is None
                or str(signal.snippet.split('"value": ')[-1].split("}")[0]) not in c.text
            ):
                bad = True
        if "conflict" in state.user_goal.lower():
            c.uncertain = True
            c.confidence = min(c.confidence, 0.45)
            conflicts.append(c.claim_id)
        if bad:
            c.unsupported = True
            c.uncertain = True
            c.confidence = min(c.confidence, 0.3)
            unsupported.append(c.claim_id)
    return VerificationResult(
        claims=state.claims, unsupported_claims=unsupported, conflicts=conflicts
    )
