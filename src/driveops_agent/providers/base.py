from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol


class Provider(Protocol):
    def complete(self, purpose: str, payload: dict) -> dict: ...


class MockProvider:
    """Deterministic offline provider: derives plans from supplied state, never a canned answer."""

    def __init__(self):
        self.calls = []

    def complete(self, purpose, payload):
        self.calls.append((purpose, payload))
        task = payload.get("task", "").lower()
        scenario = payload.get("scenario_id") or self._scenario(task)
        if purpose == "plan":
            base = [
                {
                    "objective": "Inspect selected test log",
                    "tool_name": "read_log",
                    "arguments": {"log_id": scenario},
                    "expected_observation": "scenario events and DTCs",
                },
                {
                    "objective": "Retrieve validation evidence",
                    "tool_name": "search_docs",
                    "arguments": {"query": scenario.replace("-", " ")},
                    "expected_observation": "relevant technical chunk",
                },
            ]
            if scenario != "normal-001":
                base.insert(
                    1,
                    {
                        "objective": "Resolve scenario DTC",
                        "tool_name": "query_dtcs",
                        "arguments": {"code": self._dtc(scenario)},
                        "expected_observation": "catalog definition",
                    },
                )
            if scenario == "pressure-003":
                base.append(
                    {
                        "objective": "Measure pressure gap",
                        "tool_name": "calculate_metric",
                        "arguments": {
                            "scenario_id": scenario,
                            "column": "pressure_gap_bar",
                            "operation": "max",
                        },
                        "expected_observation": "pressure gap metric",
                    }
                )
            return {"scenario_id": scenario, "steps": base}
        return {"summary": f"Synthesized from {len(payload.get('evidence', []))} evidence records."}

    def _scenario(self, task):
        if any(x in task for x in ["normal", "正常", "no-fault"]):
            return "normal-001"
        if any(x in task for x in ["wheel", "轮速"]):
            return "wheel-speed-002"
        if any(x in task for x in ["pressure", "压力"]):
            return "pressure-003"
        return "can-timeout-004"

    def _dtc(self, s):
        return {
            "wheel-speed-002": "C0035",
            "pressure-003": "C1234",
            "can-timeout-004": "U1000",
        }.get(s, "")


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
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
