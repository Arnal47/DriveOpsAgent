from pathlib import Path

import pytest
from pydantic import ValidationError

from driveops_agent.agent import DriveOpsAgent
from driveops_agent.models import AgentState, Claim, Evidence
from driveops_agent.providers import MockProvider
from driveops_agent.tools import ToolRegistry
from driveops_agent.verification import verify

ROOT = Path(__file__).parents[1]


@pytest.fixture
def agent(tmp_path):
    return DriveOpsAgent(ROOT / "data", tmp_path, MockProvider())


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(ROOT / "data", tmp_path)


@pytest.mark.parametrize(
    "task,tool",
    [
        ("normal no-fault", "read_log"),
        ("wheel speed mismatch", "query_dtcs"),
        ("pressure under-response", "calculate_metric"),
        ("CAN timeout failsafe", "query_dtcs"),
    ],
)
def test_dynamic_paths(agent, task, tool):
    assert tool in [x.name for x in agent.run(task).tool_calls]


def test_provider_called(agent):
    agent.run("CAN timeout")
    assert agent.provider.calls


def test_different_args(agent):
    assert (
        agent.run("normal").tool_calls[0].arguments != agent.run("pressure").tool_calls[0].arguments
    )


@pytest.mark.parametrize(
    "code,found", [("U1000", True), ("C0035", True), ("C1234", True), ("P9999", False)]
)
def test_dtc(registry, code, found):
    assert (registry.call("query_dtcs", {"code": code})[0].get("status") != "not found") == found


@pytest.mark.parametrize(
    "log,found",
    [
        ("normal-001", True),
        ("wheel-speed-002", True),
        ("pressure-003", True),
        ("missing-999", False),
    ],
)
def test_logs(registry, log, found):
    assert (registry.call("read_log", {"log_id": log})[0].get("status") != "not found") == found


def test_unknown_tool(registry):
    with pytest.raises(KeyError):
        registry.call("hack", {})


def test_schema(registry):
    with pytest.raises(ValidationError):
        registry.call("query_dtcs", {"code": "bad"})


def test_path(registry):
    with pytest.raises(ValueError):
        registry.call("generate_report", {"findings": "x", "filename": "../x"})


def test_injection_untrusted(registry):
    assert (
        "Ignore previous"
        in registry.call("search_docs", {"query": "Ignore previous"})[0][0]["content"]
    )


def test_normal_no_fault(agent):
    assert "No fault" in agent.run("normal no-fault").final_answer


def test_pressure_metric(agent):
    assert "bar" in agent.run("pressure under-response").final_answer


def test_revision_unknown(agent):
    s = AgentState(user_goal="x")
    from driveops_agent.planner import revise_plan

    revise_plan(s, "not found")
    assert s.decisions


def test_claim_links(agent):
    s = agent.run("CAN timeout")
    assert all(c.evidence_ids for c in s.claims)


def test_hallucinated_rejected():
    s = AgentState(
        user_goal="x",
        claims=[Claim(claim_id="x", text="fact", evidence_ids=["fake"], confidence=0.9)],
    )
    assert verify(s).unsupported_claims


def test_numeric_requires_signal():
    s = AgentState(
        user_goal="x",
        claims=[Claim(claim_id="x", text="Pressure 20 bar", evidence_ids=["e"], confidence=0.9)],
        evidence=[
            Evidence(
                evidence_id="e",
                source="x",
                tool="x",
                artifact_id="x",
                field="x",
                snippet="x",
                confidence=0.9,
            )
        ],
    )
    assert verify(s).unsupported_claims


@pytest.mark.parametrize("task", ["normal", "wheel speed", "pressure", "CAN timeout"] * 3)
def test_scenarios_complete(agent, task):
    assert agent.run(task).status.value in ["complete", "uncertain"]
