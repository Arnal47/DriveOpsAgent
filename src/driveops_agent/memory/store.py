from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        create table if not exists sessions(run_id text primary key, summary text, task text not null default '', scenario_id text not null default '', status text not null default '');
        create table if not exists evidence(run_id text, payload text);
        create table if not exists decisions(run_id text, payload text);
        create table if not exists tool_calls(run_id text, payload text);
        create table if not exists summaries(run_id text, payload text);
        create table if not exists reviews(run_id text primary key, state text, payload text);
        """)
        columns = {r[1] for r in self.db.execute("pragma table_info(sessions)")}
        for name in ("task", "scenario_id", "status"):
            if name not in columns:
                self.db.execute(f"alter table sessions add column {name} text not null default ''")
        self.db.commit()

    def save(
        self,
        run_id,
        summary,
        evidence=(),
        decisions=(),
        tool_calls=(),
        *,
        task="",
        scenario_id="",
        status="",
    ):
        summary = summary if isinstance(summary, str) else json.dumps(summary)
        self.db.execute(
            "insert or replace into sessions(run_id,summary,task,scenario_id,status) values(?,?,?,?,?)",
            (run_id, summary, task, scenario_id, status),
        )
        for table in ("evidence", "decisions", "tool_calls", "summaries"):
            self.db.execute(f"delete from {table} where run_id=?", (run_id,))
        self.db.executemany(
            "insert into evidence values(?,?)",
            [(run_id, json.dumps({**e, "provenance": "memory"})) for e in evidence],
        )
        self.db.executemany(
            "insert into decisions values(?,?)", [(run_id, json.dumps(d)) for d in decisions]
        )
        self.db.executemany(
            "insert into tool_calls values(?,?)", [(run_id, json.dumps(c)) for c in tool_calls]
        )
        self.db.execute("insert into summaries values(?,?)", (run_id, summary))
        self.db.commit()

    def find_similar(self, task, scenario_id="", limit=3):
        words = {w for w in task.lower().replace("-", " ").split() if len(w) > 2}
        ranked = []
        for row in self.db.execute(
            "select run_id,task,scenario_id,status,summary from sessions order by rowid desc"
        ):
            overlap = len(
                words
                & set(
                    (row["task"] + " " + (row["summary"] or "")).lower().replace("-", " ").split()
                )
            )
            score = overlap + (2 if scenario_id and row["scenario_id"] == scenario_id else 0)
            if score:
                item = dict(row)
                item["score"] = score
                item["evidence"] = [
                    {**json.loads(e[0]), "provenance": "memory"}
                    for e in self.db.execute(
                        "select payload from evidence where run_id=?", (row["run_id"],)
                    )
                ]
                ranked.append(item)
        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:limit]

    def save_review(self, run_id, state, payload):
        self.db.execute(
            "insert or replace into reviews values(?,?,?)", (run_id, state, json.dumps(payload))
        )
        self.db.commit()

    def load_review(self, run_id):
        row = self.db.execute(
            "select state,payload from reviews where run_id=?", (run_id,)
        ).fetchone()
        return (
            None if row is None else {"state": row["state"], "payload": json.loads(row["payload"])}
        )

    def resolve_review(self, run_id, state):
        self.db.execute("update reviews set state=? where run_id=?", (state, run_id))
        self.db.commit()

    def list(self):
        return [tuple(r) for r in self.db.execute("select run_id,summary from sessions")]

    def show(self, run_id):
        return [
            tuple(r)
            for r in self.db.execute("select payload from evidence where run_id=?", (run_id,))
        ]

    def details(self, run_id):
        session = self.db.execute("select * from sessions where run_id=?", (run_id,)).fetchone()
        if session is None:
            return {}
        out = dict(session)
        for table in ("tool_calls", "evidence", "decisions", "summaries"):
            out[table] = [
                json.loads(r[0])
                for r in self.db.execute(f"select payload from {table} where run_id=?", (run_id,))
            ]
        return out

    def clear(self):
        self.db.executescript(
            "delete from sessions;delete from evidence;delete from decisions;delete from tool_calls;delete from summaries;delete from reviews;"
        )
        self.db.commit()
