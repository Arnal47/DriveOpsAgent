import csv
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Evidence
from ..retrieval import OfflineRetriever


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)


class LogArgs(BaseModel):
    log_id: str = Field(pattern=r"^[a-z0-9-]+$")


class DtcArgs(BaseModel):
    code: str = Field(pattern=r"^[A-Z][0-9]{4}$")


class MetricArgs(BaseModel):
    scenario_id: str
    column: str
    operation: Literal["min", "max", "mean", "delta", "threshold_exceedance", "duration"]
    threshold: float | None = None


class ReportArgs(BaseModel):
    findings: str
    filename: str = "driveops_report.md"


class ToolRegistry:
    def __init__(self, data_dir: Path, reports_dir: Path):
        self.data_dir, self.reports_dir = data_dir, reports_dir
        self.retriever = OfflineRetriever(data_dir)
        self.schemas = {
            "search_docs": SearchArgs,
            "read_log": LogArgs,
            "query_dtcs": DtcArgs,
            "calculate_metric": MetricArgs,
            "generate_report": ReportArgs,
        }

    def call(self, name, args):
        if name not in self.schemas:
            raise KeyError("unknown tool: " + name)
        return getattr(self, "_" + name)(self.schemas[name](**args))

    def _search_docs(self, a):
        hits = self.retriever.search(a.query)
        ev = [
            Evidence(
                evidence_id="doc-" + h["chunk_id"],
                source=h["source_id"],
                tool="search_docs",
                artifact_id=h["chunk_id"],
                field="content",
                snippet=h["content"][:300],
                confidence=min(0.95, 0.4 + h["score"]),
            )
            for h in hits
        ]
        return hits, ev

    def _read_log(self, a):
        rows = [
            json.loads(x) for x in (self.data_dir / "brake_test_log.jsonl").read_text().splitlines()
        ]
        row = next((x for x in rows if x["log_id"] == a.log_id), None)
        if row is None:
            return {"status": "not found", "log_id": a.log_id}, []
        return row, [
            Evidence(
                evidence_id="log-" + a.log_id,
                source="brake_test_log.jsonl",
                tool="read_log",
                artifact_id=a.log_id,
                field="events",
                snippet=json.dumps(row["events"]),
                confidence=0.95,
            )
        ]

    def _query_dtcs(self, a):
        row = next(
            (
                x
                for x in json.loads((self.data_dir / "dtc_catalog.json").read_text())
                if x["code"] == a.code
            ),
            None,
        )
        if row is None:
            return {"status": "not found", "code": a.code}, []
        return row, [
            Evidence(
                evidence_id="dtc-" + a.code,
                source="dtc_catalog.json",
                tool="query_dtcs",
                artifact_id=a.code,
                field="description",
                snippet=row["description"],
                confidence=0.9,
            )
        ]

    def _calculate_metric(self, a):
        rows = [
            r
            for r in csv.DictReader((self.data_dir / "vehicle_signals.csv").open())
            if r["scenario_id"] == a.scenario_id
        ]
        if not rows:
            raise ValueError("unknown scenario")
        vals = [float(r[a.column]) for r in rows]
        op = a.operation
        value = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "delta": vals[-1] - vals[0],
            "threshold_exceedance": sum(v > (a.threshold or 0) for v in vals),
            "duration": 0.1 * sum(v > (a.threshold or 0) for v in vals),
        }[op]
        result = {"scenario_id": a.scenario_id, "column": a.column, "operation": op, "value": value}
        return result, [
            Evidence(
                evidence_id=f"metric-{a.scenario_id}-{a.column}-{op}",
                source="vehicle_signals.csv",
                tool="calculate_metric",
                artifact_id=a.scenario_id,
                field=a.column,
                snippet=json.dumps(result),
                confidence=0.9,
            )
        ]

    def _generate_report(self, a):
        p = (self.reports_dir / a.filename).resolve()
        if p.parent != self.reports_dir.resolve() or ".." in a.filename:
            raise ValueError("unsafe report path")
        p.write_text("# DriveOps Report\n\n" + a.findings)
        return {"path": str(p)}, []
