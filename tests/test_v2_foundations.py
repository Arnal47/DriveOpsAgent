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


def test_adapter_retry_through_agent(tmp_path):
    calls = []

    def transport(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError()
        return {}

    from driveops_agent.external.base import Adapter

    state = DriveOpsAgent(
        ROOT / "data", tmp_path, MockProvider(), external_adapter=Adapter(transport, retries=1)
    ).run("CAN timeout")
    assert state.status.value == "complete" and len(calls) == 2


def test_adapter_exhausted_routes_review(tmp_path):
    from driveops_agent.external.base import Adapter

    def transport(*args, **kwargs):
        raise TimeoutError()

    state = DriveOpsAgent(
        ROOT / "data", tmp_path, MockProvider(), external_adapter=Adapter(transport, retries=0)
    ).run("CAN timeout")
    assert state.status.value == "needs_review"


def test_agent_memory_hit_and_provenance(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    first = agent.run("CAN timeout")
    second = agent.run("CAN timeout")
    import json

    events = [json.loads(x)["event_type"] for x in agent.trace.path.read_text().splitlines()]
    assert first.run_id != second.run_id and "memory_hit" in events
    assert all(e.provenance == "current" for e in second.evidence)


def test_current_evidence_overrides_stale_memory(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    agent.memory.save("old", "stale CAN", [{"evidence_id": "old", "source": "memory"}])
    state = agent.run("conflict CAN timeout")
    assert state.status.value == "needs_review"
    assert all(e.provenance == "current" for e in state.evidence)
    assert "current evidence" in state.decisions[0]


def _provider_response(content):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_openai_provider_timeout_recovery_through_agent(tmp_path):
    from driveops_agent.providers.base import OpenAICompatibleProvider

    calls = []

    def transport(body, headers, timeout):
        calls.append(1)
        request = json.loads(json.loads(body)["messages"][0]["content"])
        if len(calls) == 1:
            raise TimeoutError("first attempt")
        if request["purpose"] == "plan":
            return _provider_response(MockProvider().complete("plan", request["payload"]))
        return _provider_response(MockProvider().complete("synthesis", request["payload"]))

    provider = OpenAICompatibleProvider(
        transport, max_retries=1, api_key="x", base_url="https://offline", model="fake"
    )
    agent = DriveOpsAgent(ROOT / "data", tmp_path, provider)
    state = agent.run("CAN timeout")
    assert state.status.value == "complete" and any(
        e["event_type"] == "retry" for e in agent.trace.events
    )


def test_openai_provider_invalid_json_recovery_through_agent(tmp_path):
    from driveops_agent.providers.base import OpenAICompatibleProvider

    calls = []

    def transport(body, headers, timeout):
        calls.append(1)
        request = json.loads(json.loads(body)["messages"][0]["content"])
        if len(calls) == 1:
            return b"not-json"
        return _provider_response(MockProvider().complete(request["purpose"], request["payload"]))

    provider = OpenAICompatibleProvider(
        transport, max_retries=1, api_key="x", base_url="https://offline", model="fake"
    )
    assert (
        DriveOpsAgent(ROOT / "data", tmp_path, provider).run("CAN timeout").status.value
        == "complete"
    )


def test_openai_provider_retry_exhausted_through_agent(tmp_path):
    from driveops_agent.providers.base import OpenAICompatibleProvider

    def transport(*args):
        raise TimeoutError("offline")

    provider = OpenAICompatibleProvider(
        transport, max_retries=1, api_key="x", base_url="https://offline", model="fake"
    )
    assert (
        DriveOpsAgent(ROOT / "data", tmp_path, provider).run("CAN timeout").status.value == "failed"
    )


def test_review_state_restores_in_new_process_object(tmp_path):
    pending = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).run("conflict CAN timeout")
    restored = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).review(pending.run_id, True)
    assert restored.status.value == "complete"


def test_memory_store_saves_all_run_artifacts(tmp_path):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())
    state = agent.run("CAN timeout")
    details = agent.memory.details(state.run_id)
    assert (
        details["tool_calls"]
        and details["evidence"]
        and details["decisions"] == []
        and details["summaries"]
    )


@pytest.mark.parametrize("backend", ["bm25", "hybrid"])
def test_retrieval_v2_backend_hits_expected_document(tmp_path, backend):
    agent = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider(), retrieval_backend=backend)
    hits = agent.registry.retriever.search("wheel speed mismatch validation")
    assert hits and any("validation" in hit["source_id"] for hit in hits)


def test_openai_provider_invalid_schema_exhausted_through_agent(tmp_path):
    from driveops_agent.providers.base import OpenAICompatibleProvider

    def transport(*args):
        return _provider_response({"unexpected": True})

    provider = OpenAICompatibleProvider(
        transport, max_retries=1, api_key="x", base_url="https://offline", model="fake"
    )
    state = DriveOpsAgent(ROOT / "data", tmp_path, provider).run("CAN timeout")
    assert (
        state.status.value == "failed" and "retry_exhausted" not in state.final_answer
        if state.final_answer
        else state.errors
    )
