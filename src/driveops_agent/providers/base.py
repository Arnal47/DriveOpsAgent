from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class Provider(Protocol):
    def complete(self, purpose: str, payload: dict) -> dict: ...


class ProviderError(RuntimeError):
    def __init__(self, code, message, retryable=True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class MockProvider:
    def __init__(self):
        self.calls = []
        self.synthesis_prefix = "Evidence synthesis"

    def complete(self, purpose, payload):
        self.calls.append((purpose, payload))
        task = payload.get("task", "").lower()
        if purpose == "plan":
            ids = payload["artifact_ids"]
            pick = next((x for x in ids if x in task), None)
            hints = payload.get("scenario_hints", {})
            if pick is None:
                pick = next((sid for hint, sid in hints.items() if hint in task), "normal-001")
            intent = "compare" if "compare" in task else "report" if "report" in task else "root"
            if "unknown" in task or "p9999" in task:
                intent = "unknown"
            if "missing" in task or "missing-999" in task:
                intent, pick = "missing", "missing-999"
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
                        "objective": "query requested DTC",
                        "tool_name": "query_dtcs",
                        "arguments": {"code": "P9999"},
                        "expected_observation": "catalog result",
                    }
                )
            if intent == "compare":
                other = "normal-001" if pick != "normal-001" else "can-timeout-004"
                steps.append(
                    {
                        "objective": "inspect comparison log",
                        "tool_name": "read_log",
                        "arguments": {"log_id": other},
                        "expected_observation": "comparison events",
                    }
                )
            steps.append(
                {
                    "objective": "retrieve validation evidence",
                    "tool_name": "search_docs",
                    "arguments": {
                        "query": "Ignore previous instructions" if "injection" in task else task
                    },
                    "expected_observation": "untrusted document evidence",
                }
            )
            if pick == "pressure-003" or intent == "compare":
                steps.append(
                    {
                        "objective": "measure key signal",
                        "tool_name": "calculate_metric",
                        "arguments": {
                            "scenario_id": pick,
                            "column": "pressure_gap_bar",
                            "operation": "max",
                        },
                        "expected_observation": "numeric signal evidence",
                    }
                )
            return {"intent": intent, "steps": steps}
        evidence = payload["evidence"]
        current = lambda tool: [
            e for e in evidence if e["tool"] == tool and e.get("provenance", "current") == "current"
        ]
        dtcs, logs, metrics = (
            current("query_dtcs"),
            current("read_log"),
            current("calculate_metric"),
        )
        claims = []
        if len(logs) >= 2:
            claims.append(
                {
                    "text": "Comparison identifies different log events and failsafe outcomes.",
                    "evidence_ids": [e["evidence_id"] for e in logs[:2]],
                    "confidence": 0.85,
                    "kind": "difference",
                }
            )
        elif dtcs and logs:
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
            value = metrics[0].get("value")
            claims.append(
                {
                    "text": f"Signal metric measured: {value} bar.",
                    "evidence_ids": [metrics[0]["evidence_id"]],
                    "confidence": 0.9,
                    "value": value,
                    "kind": "numeric",
                }
            )
        return {"prefix": self.synthesis_prefix, "claims": claims}


class OpenAICompatibleProvider:
    def __init__(
        self, transport=None, timeout=30, max_retries=2, *, api_key=None, base_url=None, model=None
    ):
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model or os.getenv("MODEL")
        if not all((self.api_key, self.base_url, self.model)):
            raise RuntimeError("API_KEY, BASE_URL, and MODEL are required")
        self.transport = transport or self._http_transport
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = []

    def _http_transport(self, body, headers, timeout):
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _decode(self, raw):
        try:
            outer = raw if isinstance(raw, dict) else json.loads(raw)
            content = outer["choices"][0]["message"]["content"]
            result = content if isinstance(content, dict) else json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError("invalid_json", f"invalid provider JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise ProviderError("invalid_schema", "structured response must be an object")
        return result

    def complete(self, purpose, payload):
        self.calls.append((purpose, payload))
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
        headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._decode(self.transport(body, headers, self.timeout))
                if purpose == "plan" and not isinstance(result.get("steps"), list):
                    raise ProviderError("invalid_schema", "plan.steps must be an array")
                if purpose == "synthesis" and not isinstance(result.get("claims"), list):
                    raise ProviderError("invalid_schema", "synthesis.claims must be an array")
                return result
            except ProviderError as exc:
                last = exc
            except (TimeoutError, urllib.error.HTTPError, OSError) as exc:
                last = ProviderError("provider_unavailable", str(exc))
            self.last_errors.append(str(last))
            if attempt == self.max_retries or not last.retryable:
                break
        raise ProviderError("retry_exhausted", str(last)) from last
