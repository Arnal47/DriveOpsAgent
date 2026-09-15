from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import Evidence
from ..retrieval import OfflineRetriever


class SearchDocsArgs(BaseModel):
    query: str = Field(min_length=1)


class ReadLogArgs(BaseModel):
    log_id: Literal["brake-normal-001", "brake-failsafe-002"]


class DtcArgs(BaseModel):
    code: str = Field(pattern=r"^[A-Z][0-9]{4}$")


class MetricArgs(BaseModel):
    column: str
    operation: Literal["min", "max", "mean", "delta", "threshold_exceedance", "duration"]
    threshold: float | None = None


class ReportArgs(BaseModel):
    findings: str = Field(min_length=1)
    filename: str = "driveops_report.md"


class ToolRegistry:
    def __init__(self, data_dir: Path, reports_dir: Path):
        self.data_dir, self.reports_dir = data_dir, reports_dir
        self.retriever = OfflineRetriever(data_dir)
        self.schemas = {
            "search_docs": SearchDocsArgs,
            "read_log": ReadLogArgs,
            "query_dtcs": DtcArgs,
            "calculate_metric": MetricArgs,
            "generate_report": ReportArgs,
        }

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, list[Evidence]]:
        if name not in self.schemas:
            raise KeyError(f"unknown tool: {name}")
        parsed = self.schemas[name](**args)
        return getattr(self, f"_{name}")(parsed)

    def _search_docs(self, a):
        hits = self.retriever.search(a.query)
        ev = [
            Evidence(
                evidence_id=f"doc-{i}",
                source=h["source_id"],
                tool="search_docs",
                artifact_id=h["chunk_id"],
                field="content",
                snippet=h["content"][:220],
                confidence=min(0.95, h["score"] + 0.4),
            )
            for i, h in enumerate(hits)
        ]
        return hits, ev

    def _read_log(self, a):
        rows = [
            json.loads(x) for x in (self.data_dir / "brake_test_log.jsonl").read_text().splitlines()
        ]
        row = next(x for x in rows if x["log_id"] == a.log_id)
        return row, [
            Evidence(
                evidence_id=f"log-{a.log_id}",
                source="brake_test_log.jsonl",
                tool="read_log",
                artifact_id=a.log_id,
                field="events",
                snippet=json.dumps(row["events"]),
                confidence=0.95,
            )
        ]

    def _query_dtcs(self, a):
        items = json.loads((self.data_dir / "dtc_catalog.json").read_text())
        row = next((x for x in items if x["code"] == a.code), None)
        return row, (
            []
            if row is None
            else [
                Evidence(
                    evidence_id=f"dtc-{a.code}",
                    source="dtc_catalog.json",
                    tool="query_dtcs",
                    artifact_id=a.code,
                    field="description",
                    snippet=row["description"],
                    confidence=0.9,
                )
            ]
        )

    def _calculate_metric(self, a):
        rows = list(csv.DictReader((self.data_dir / "vehicle_signals.csv").open()))
        vals = [float(r[a.column]) for r in rows]
        op = a.operation
        if op == "min":
            value = min(vals)
        elif op == "max":
            value = max(vals)
        elif op == "mean":
            value = sum(vals) / len(vals)
        elif op == "delta":
            value = vals[-1] - vals[0]
        elif op == "threshold_exceedance":
            value = sum(x > (a.threshold or 0) for x in vals)
        else:
            value = sum(1 for x in vals if x > (a.threshold or 0)) * 0.1
        result = {"column": a.column, "operation": op, "value": value}
        ev = [
            Evidence(
                evidence_id=f"metric-{a.column}-{op}",
                source="vehicle_signals.csv",
                tool="calculate_metric",
                artifact_id=a.column,
                field=op,
                snippet=json.dumps(result),
                confidence=0.9,
            )
        ]
        return result, ev

    def _generate_report(self, a):
        path = (self.reports_dir / a.filename).resolve()
        if path.parent != self.reports_dir.resolve() or ".." in a.filename:
            raise ValueError("unsafe report path")
        path.write_text(f"# DriveOps Analysis Report\n\n{a.findings}\n", encoding="utf8")
        return {"path": str(path)}, [
            Evidence(
                evidence_id="report-1",
                source="reports",
                tool="generate_report",
                artifact_id=path.name,
                field="markdown",
                snippet=a.findings[:220],
                confidence=0.8,
            )
        ]
