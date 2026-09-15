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
            ids = payload["artifact_ids"]
            pick = next(
                (x for x in ids if x.split("-")[0] in task or x.replace("-", " ") in task), None
            )
            if any(x in task for x in ["can", "timeout", "u1000"]):
                pick = "can-timeout-004"
            elif any(
                x in task for x in ["pressure", "under-response", "c1234", "deceleration gap"]
            ):
                pick = "pressure-003"
            elif any(x in task for x in ["wheel", "speed"]):
                pick = "wheel-speed-002"
            else:
                pick = "normal-001"
            intent = (
                "compare"
                if "compare" in task
                else "report"
                if "report" in task
                else "injection"
                if "injection" in task
                else "root"
            )
            if "unknown" in task:
                intent = "unknown"
            if "missing log" in task:
                intent = "missing"
                pick = "missing-999"
            steps = [
                {
                    "objective": "inspect target log",
                    "tool_name": "read_log",
                    "arguments": {"log_id": pick},
                    "expected_observation": "events, DTCs, failsafe",
                }
            ]
            if intent == "unknown":
                steps.append(
                    {
                        "objective": "query unknown DTC",
                        "tool_name": "query_dtcs",
                        "arguments": {"code": "P9999"},
                        "expected_observation": "not found",
                    }
                )
            if intent == "compare":
                steps.append(
                    {
                        "objective": "inspect baseline log",
                        "tool_name": "read_log",
                        "arguments": {
                            "log_id": "normal-001" if pick != "normal-001" else "can-timeout-004"
                        },
                        "expected_observation": "comparison baseline",
                    }
                )
            steps += [
                {
                    "objective": "retrieve validation evidence",
                    "tool_name": "search_docs",
                    "arguments": {
                        "query": "Ignore previous instructions"
                        if intent == "injection"
                        else pick.replace("-", " ")
                    },
                    "expected_observation": "untrusted evidence",
                }
            ]
            if "pressure" in pick or intent == "compare":
                steps.append(
                    {
                        "objective": "measure key signals",
                        "tool_name": "calculate_metric",
                        "arguments": {
                            "scenario_id": pick,
                            "column": "pressure_gap_bar",
                            "operation": "max",
                        },
                        "expected_observation": "numeric metric",
                    }
                )
            return {"intent": intent, "steps": steps}
        evidence = payload["evidence"]
        dtcs = [e for e in evidence if e["source"] == "dtc_catalog.json"]
        logs = [e for e in evidence if e["source"] == "brake_test_log.jsonl"]
        metrics = [e for e in evidence if e["source"] == "vehicle_signals.csv"]
        claims = []
        if len(logs) >= 2:
            claims.append(
                {
                    "text": "Comparison identifies different log events and failsafe outcomes.",
                    "evidence_ids": [e["evidence_id"] for e in logs],
                    "confidence": 0.85,
                    "kind": "difference",
                }
            )
        elif dtcs:
            claims.append(
                {
                    "text": "Most likely finding: " + dtcs[0]["snippet"],
                    "evidence_ids": [logs[0]["evidence_id"], dtcs[0]["evidence_id"]],
                    "confidence": 0.82,
                    "kind": "root-cause",
                }
            )
        elif logs:
            claims.append(
                {
                    "text": "No fault: log provides insufficient fault evidence.",
                    "evidence_ids": [logs[0]["evidence_id"]],
                    "confidence": 0.8,
                    "kind": "conclusion",
                }
            )
        if metrics:
            claims.append(
                {
                    "text": f"Signal metric measured: {metrics[0].get('value')} bar.",
                    "evidence_ids": [metrics[0]["evidence_id"]],
                    "confidence": 0.9,
                    "value": metrics[0].get("value"),
                    "kind": "numeric",
                }
            )
        return {"prefix": self.synthesis_prefix, "claims": claims}


class OpenAICompatibleProvider:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not all((self.api_key, self.base_url, self.model)):
            raise RuntimeError("API_KEY, BASE_URL, and MODEL are required")

    def complete(self, purpose, payload):
        b = json.dumps(
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
        r = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=b,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=30) as x:
            return json.loads(json.loads(x.read())["choices"][0]["message"]["content"])
