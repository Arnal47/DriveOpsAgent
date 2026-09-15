import json
import time
import uuid
from pathlib import Path


class Trace:
    def __init__(self, root):
        self.run_id = str(uuid.uuid4())
        self.path = Path(root) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []

    def emit(
        self,
        event_type,
        tool=None,
        arguments=None,
        success=True,
        error=None,
        evidence_ids=None,
        latency_ms=0,
    ):
        e = {
            "run_id": self.run_id,
            "step_id": len(self.events) + 1,
            "timestamp": time.time(),
            "event_type": event_type,
            "tool": tool,
            "arguments": arguments or {},
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
            "evidence_ids": evidence_ids or [],
        }
        self.events.append(e)
        self.path.open("a").write(json.dumps(e) + "\n")

    def summary(self):
        return {
            "total_steps": len(self.events),
            "tool_calls": sum(e["tool"] is not None for e in self.events),
            "failures": sum(not e["success"] for e in self.events),
            "total_latency": sum(e["latency_ms"] for e in self.events),
            "evidence_count": sum(len(e["evidence_ids"]) for e in self.events),
        }
