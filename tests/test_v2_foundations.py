import json

import pytest

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
