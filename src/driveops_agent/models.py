from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
    NEEDS_REVIEW = "needs_review"
    RUNNING = "running"
    COMPLETE = "complete"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class Evidence(BaseModel):
    evidence_id: str
    source: str
    tool: str
    artifact_id: str
    field: str
    snippet: str
    confidence: float = Field(ge=0, le=1)
    value: float | None = None


class Claim(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str] = []
    confidence: float = Field(ge=0, le=1)
    value: float | None = None
    uncertain: bool = False
    unsupported: bool = False


class PlanStep(BaseModel):
    objective: str
    tool_name: str
    arguments: dict[str, Any]
    expected_observation: str
    status: str = "pending"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    cached: bool = False
    success: bool = True


class AgentState(BaseModel):
    user_goal: str
    current_plan: list[PlanStep] = []
    completed_steps: list[str] = []
    tool_calls: list[ToolCall] = []
    observations: list[str] = []
    evidence: list[Evidence] = []
    errors: list[str] = []
    claims: list[Claim] = []
    final_answer: str | None = None
    status: Status = Status.RUNNING
    step_count: int = 0
    decisions: list[str] = []


class VerificationResult(BaseModel):
    claims: list[Claim]
    unsupported_claims: list[str] = []
    conflicts: list[str] = []
