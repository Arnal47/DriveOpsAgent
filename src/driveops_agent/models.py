from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Status(str, Enum):
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


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    cached: bool = False
    success: bool = True


class AgentState(BaseModel):
    user_goal: str
    current_plan: list[str] = []
    completed_steps: list[str] = []
    tool_calls: list[ToolCall] = []
    observations: list[str] = []
    evidence: list[Evidence] = []
    errors: list[str] = []
    final_answer: str | None = None
    status: Status = Status.RUNNING
    step_count: int = 0
    decisions: list[str] = []


class VerificationResult(BaseModel):
    supported: bool
    unsupported_claims: list[str] = []
    conflicts: list[str] = []
    incomplete_plan: bool = False
