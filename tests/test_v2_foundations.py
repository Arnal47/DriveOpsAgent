import json
from pathlib import Path

import pytest

from driveops_agent.agent import DriveOpsAgent
from driveops_agent.providers import MockProvider

ROOT = Path(__file__).parents[1]

from driveops_agent.external.base import Adapter, ToolError
from driveops_agent.memory.store import MemoryStore
from driveops_agent.tracing import Trace


@pytest.mark.parametrize(
    "tool", ["list_logs", "read_log", "query_signals", "search_docs", "query_dtcs"]
)
def test_adapter_discovery_and_requests(tool):
    def transport(name, args, timeout=None):
        return {"tool": name, "payload": args}

    assert Adapter(transport).request(tool, {})["untrusted"]


def test_adapter_retry_success():
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError()
        return {}

    assert Adapter(transport, retries=1).request("x", {})["untrusted"] and len(calls) == 2


def test_adapter_circuit_breaker():
    def bad(*args, **kwargs):
        raise TimeoutError()

    a = Adapter(bad, retries=0, threshold=1)
    with pytest.raises(ToolError):
        a.request("x", {})
    with pytest.raises(ToolError):
        a.request("x", {})


def test_memory_persistence(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite")
    m.save("r1", "summary", [{"evidence_id": "e"}])
    assert m.list() and "memory" in m.show("r1")[0][0]


def test_memory_clear(tmp_path):
    m = MemoryStore(tmp_path / "m.sqlite")
    m.save("r", "x")
    m.clear()
    assert not m.list()


def test_trace_schema_and_isolation(tmp_path):
    a, b = Trace(tmp_path), Trace(tmp_path)
    a.emit("tool", tool="read_log", evidence_ids=["e"])
    row = json.loads(a.path.read_text().splitlines()[0])
    assert row["run_id"] == a.run_id and a.run_id != b.run_id and "latency_ms" in row


def test_trace_summary(tmp_path):
    t = Trace(tmp_path)
    t.emit("tool", tool="x", latency_ms=3)
    assert t.summary()["tool_calls"] == 1


@pytest.mark.parametrize("event", ["provider_call", "memory_hit", "review_required"])
def test_trace_event_types(tmp_path, event):
    t = Trace(tmp_path)
    t.emit(event)
    assert t.events[0]["event_type"] == event


class FlakySynthesisProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.failures = 0

    def complete(self, purpose, payload):
        if purpose == "synthesis" and self.failures == 0:
            self.failures += 1
            raise TimeoutError("provider timeout")
        return super().complete(purpose, payload)


def test_provider_timeout_recovery_through_agent(tmp_path):
    provider = FlakySynthesisProvider()
    agent = DriveOpsAgent(ROOT / "data", tmp_path, provider)
    state = agent.run("CAN timeout")
    assert state.run_id == agent.trace.run_id
    assert state.status.value == "complete"
    assert any(event["event_type"] == "retry" for event in agent.trace.events) is True


def test_agent_routes_conflict_to_review(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    assert agent.run("conflict CAN timeout").status.value == "needs_review"


def test_review_approve_transition(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    state = agent.run("conflict CAN timeout")
    assert agent.review(state.run_id, True).status.value == "complete"


def test_review_reject_transition(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    state = agent.run("conflict CAN timeout")
    assert agent.review(state.run_id, False).status.value == "failed"


def test_trace_contains_required_events(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    state = agent.run("CAN timeout")
    assert state.run_id == agent.trace.run_id
    import json

    events = [json.loads(x) for x in agent.trace.path.read_text().splitlines()]
    kinds = {e["event_type"] for e in events}
    assert {
        "run_start",
        "memory_lookup",
        "memory_miss",
        "provider_call",
        "tool_call",
        "memory_save",
        "run_end",
    }.issubset(kinds)
    assert all(
        {
            "run_id",
            "step_id",
            "timestamp",
            "latency_ms",
            "success",
            "error",
            "evidence_ids",
        }.issubset(e)
        for e in events
    )
