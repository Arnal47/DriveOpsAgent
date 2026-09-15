from dataclasses import dataclass
from time import monotonic


@dataclass
class ToolError(Exception):
    code: str
    message: str
    retryable: bool = True


class Adapter:
    def __init__(self, transport, timeout=1, retries=2, threshold=3):
        self.transport = transport
        self.timeout = timeout
        self.retries = retries
        self.threshold = threshold
        self.failures = 0
        self.last_attempts = 0

    def discover(self):
        return self.transport("discover", {})

    def request(self, tool, args):
        if self.failures >= self.threshold:
            self.last_attempts = 0
            raise ToolError("circuit_open", "adapter circuit open", False)
        last = None
        self.last_attempts = 0
        for _ in range(self.retries + 1):
            self.last_attempts += 1
            try:
                start = monotonic()
                out = self.transport(tool, args, timeout=self.timeout)
                out["latency_ms"] = (monotonic() - start) * 1000
                out["untrusted"] = True
                self.failures = 0
                return out
            except Exception as exc:
                last = exc
                self.failures += 1
        raise ToolError("unavailable", str(last))
