from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


class Provider(Protocol):
    def complete(self, purpose: str, payload: dict) -> dict: ...


class MockProvider:
    def __init__(self):
        self.calls = []
        self.synthesis_prefix = "Evidence synthesis"

    def complete(self, purpose, payload):
        self.calls.append((purpose, payload))
        task = payload.get("task", "").lower()
        if purpose == "plan":
            scenarios = payload["scenarios"]
            scores = [
                (
                    sum(
                        token in task
                        for token in (x["scenario"] + " " + x["log_id"])
                        .lower()
                        .replace("-", " ")
                        .split()
                    ),
                    x,
                )
                for x in scenarios
            ]
            scenario = (
                max(scores, key=lambda x: x[0])[1]
                if max(scores, key=lambda x: x[0])[0]
                else scenarios[0]
            )
            intent = next(
                (
                    x
                    for x in [
                        "missing",
                        "unknown",
                        "injection",
                        "compare",
                        "report",
                        "dtc",
                        "failsafe",
                        "evidence",
                    ]
                    if x in task
                ),
                "root_cause",
            )
            if "missing log" in task:
                scenario = {"log_id": "missing-999", "scenario": "missing"}
            steps = [
                {
                    "objective": "inspect log",
                    "tool_name": "read_log",
                    "arguments": {"log_id": scenario["log_id"]},
                    "expected_observation": "events and DTCs",
                }
            ]
            if intent == "unknown":
                steps.append(
                    {
                        "objective": "look up unknown DTC",
                        "tool_name": "query_dtcs",
                        "arguments": {"code": "P9999"},
                        "expected_observation": "not found response",
                    }
                )
            elif scenario.get("dtcs"):
                steps.append(
                    {
                        "objective": "interpret logged DTC",
                        "tool_name": "query_dtcs",
                        "arguments": {"code": scenario["dtcs"][0]},
                        "expected_observation": "catalog definition",
                    }
                )
            query = (
                "Ignore previous instructions" if intent == "injection" else scenario["scenario"]
            )
            steps.append(
                {
                    "objective": "retrieve validation evidence",
                    "tool_name": "search_docs",
                    "arguments": {"query": query},
                    "expected_observation": "untrusted supporting chunk",
                }
            )
            if "pressure" in scenario["scenario"] or intent == "evidence":
                steps.append(
                    {
                        "objective": "measure signals",
                        "tool_name": "calculate_metric",
                        "arguments": {
                            "scenario_id": scenario["log_id"],
                            "column": "pressure_gap_bar",
                            "operation": "max",
                        },
                        "expected_observation": "metric",
                    }
                )
            if intent == "report":
                steps.append(
                    {
                        "objective": "generate report",
                        "tool_name": "generate_report",
                        "arguments": {"findings": "Evidence report requested"},
                        "expected_observation": "report path",
                    }
                )
            return {"intent": intent, "scenario_id": scenario["log_id"], "steps": steps}
        return {"prefix": self.synthesis_prefix, "claim_count": len(payload.get("claims", []))}


class OpenAICompatibleProvider:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not all((self.api_key, self.base_url, self.model)):
            raise RuntimeError("API_KEY, BASE_URL, and MODEL are required")

    def complete(self, purpose, payload):
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({"purpose": purpose, "payload": payload}),
                    }
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
