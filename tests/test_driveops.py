from pathlib import Path

import pytest
from pydantic import ValidationError

from driveops_agent.agent import DriveOpsAgent
from driveops_agent.models import AgentState, Status
from driveops_agent.planner import make_plan
from driveops_agent.providers import MockProvider, OpenAICompatibleProvider
from driveops_agent.tools import ToolRegistry
from driveops_agent.verification import verify

ROOT = Path(__file__).parents[1]


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(ROOT / "data", tmp_path)


@pytest.fixture
def agent(tmp_path):
    return DriveOpsAgent(ROOT / "data", tmp_path)


def test_mock_provider():
    assert MockProvider().complete("x") == "offline mock response"


def test_openai_requires_env(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleProvider()


def test_registry_has_five_tools(registry):
    assert len(registry.schemas) == 5


def test_schema_validation(registry):
    with pytest.raises(ValidationError):
        registry.call("read_log", {"log_id": "bad"})


def test_dtc_retrieval(registry):
    assert registry.call("query_dtcs", {"code": "U1000"})[0]["severity"] == "high"


def test_log_reading(registry):
    assert registry.call("read_log", {"log_id": "brake-failsafe-002"})[0]["failsafe"]


@pytest.mark.parametrize("op", ["min", "max", "mean", "delta", "threshold_exceedance", "duration"])
def test_metrics(registry, op):
    assert (
        "value"
        in registry.call(
            "calculate_metric", {"column": "brake_pressure_bar", "operation": op, "threshold": 12}
        )[0]
    )


def test_retrieval(registry):
    assert registry.call("search_docs", {"query": "CAN timeout failsafe"})[0][0]["chunk_id"]


def test_evidence_linking(registry):
    assert registry.call("read_log", {"log_id": "brake-normal-001"})[1][0].source


def test_report_generation(registry, tmp_path):
    assert Path(registry.call("generate_report", {"findings": "x"})[0]["path"]).exists()


def test_path_traversal(registry):
    with pytest.raises(ValueError):
        registry.call("generate_report", {"findings": "x", "filename": "../bad.md"})


def test_prompt_injection_is_data(registry):
    assert (
        "Ignore previous"
        in registry.call("search_docs", {"query": "Ignore previous instructions"})[0][0]["content"]
    )


def test_planner():
    assert len(make_plan("why")) >= 8


def test_agent_loop(agent):
    state = agent.run("为什么这次制动测试进入 failsafe？")
    assert state.status == Status.COMPLETE and state.step_count >= 5


def test_duplicate_cache(agent):
    s = AgentState(user_goal="x")
    agent._call(s, "query_dtcs", {"code": "U1000"})
    agent._call(s, "query_dtcs", {"code": "U1000"})
    assert s.tool_calls[-1].cached


def test_max_steps(agent):
    agent.max_steps = 1
    assert agent.run("x").status == Status.FAILED


def test_failure_recovery(agent):
    agent.registry.schemas.pop("read_log")
    assert agent.run("x").status == Status.FAILED


def test_verifier_detects_unsupported():
    assert not verify(AgentState(user_goal="x", final_answer="claim")).supported


@pytest.mark.parametrize(
    "task",
    [
        "为什么这次制动测试进入 failsafe？",
        "根据日志和 DTC，给出最可能的根因候选并排序。",
        "对比两次制动测试，找出关键差异。",
        "生成带证据引用的验证报告。",
    ],
)
def test_e2e_tasks(agent, task):
    assert agent.run(task).status == Status.COMPLETE
