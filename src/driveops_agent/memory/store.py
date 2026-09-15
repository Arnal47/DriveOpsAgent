import json
import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(
            "create table if not exists sessions(run_id text primary key, summary text);create table if not exists evidence(run_id text, payload text);create table if not exists decisions(run_id text, payload text);create table if not exists tool_calls(run_id text, payload text);create table if not exists summaries(run_id text, payload text);"
        )

    def save(self, run_id, summary, evidence=(), decisions=(), tool_calls=()):
        self.db.execute("insert or replace into sessions values(?,?)", (run_id, summary))
        self.db.executemany(
            "insert into evidence values(?,?)",
            [(run_id, json.dumps({"provenance": "memory", **e})) for e in evidence],
        )
        self.db.commit()

    def list(self):
        return self.db.execute("select run_id,summary from sessions").fetchall()

    def show(self, run_id):
        return self.db.execute("select payload from evidence where run_id=?", (run_id,)).fetchall()

    def clear(self):
        self.db.executescript(
            "delete from sessions;delete from evidence;delete from decisions;delete from tool_calls;delete from summaries;"
        )
        self.db.commit()
