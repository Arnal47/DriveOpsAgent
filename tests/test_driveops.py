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


class AlternateSynthesisProvider(MockProvider):
    def complete(self, purpose, payload):
        result = super().complete(purpose, payload)
        if purpose == "synthesis":
            return {
                "prefix": "Altered synthesis",
                "claims": [
                    {
                        "text": "Provider-selected conclusion.",
                        "evidence_ids": [payload["evidence"][0]["evidence_id"]],
                        "confidence": 0.7,
                    }
                ],
            }
        return result


def test_metric_mutation_changes_final_answer(tmp_path):
    data = tmp_path / "data"
    import shutil

    shutil.copytree(ROOT / "data", data)
    signals = data / "vehicle_signals.csv"
    signals.write_text(signals.read_text().replace("pressure-003,2,28,1", "pressure-003,2,99,1"))
    answer = (
        DriveOpsAgent(data, tmp_path, MockProvider()).run("pressure under-response").final_answer
    )
    assert "99.0 bar" in answer


def test_synthesis_mutation_changes_final_answer(tmp_path):
    answer = (
        DriveOpsAgent(ROOT / "data", tmp_path, AlternateSynthesisProvider())
        .run("CAN timeout")
        .final_answer
    )
    assert "Provider-selected conclusion." in answer


def test_compare_executes_two_logs_and_links_both(tmp_path):
    state = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).run("compare CAN timeout")
    logs = [call for call in state.tool_calls if call.name == "read_log"]
    assert len(logs) >= 2 and logs[0].arguments != logs[1].arguments
    difference = next(claim for claim in state.claims if "Comparison" in claim.text)
    assert {"log-can-timeout-004", "log-normal-001"}.issubset(difference.evidence_ids)


def test_report_markdown_contains_claim_evidence_and_uncertainty(tmp_path):
    state = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).run("report CAN timeout")
    report = (tmp_path / "driveops_report.md").read_text()
    assert state.claims[0].text in report
    assert state.claims[0].evidence_ids[0] in report
    assert "Uncertainty:" in report


def test_unknown_dtc_calls_p9999(tmp_path):
    state = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).run("unknown DTC")
    assert any(c.name == "query_dtcs" and c.arguments["code"] == "P9999" for c in state.tool_calls)
    assert any("not found" in x for x in state.observations)


def test_missing_log_calls_missing_id(tmp_path):
    state = DriveOpsAgent(ROOT / "data", tmp_path, MockProvider()).run("missing log evidence")
    assert any(
        c.name == "read_log" and c.arguments["log_id"] == "missing-999" for c in state.tool_calls
    )
