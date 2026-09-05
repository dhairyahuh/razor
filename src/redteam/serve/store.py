"""SQLite persistence for runs, alerts and cases.

Why not just read the parquet
-----------------------------
The artefacts on disk are the source of truth and this does not replace them. But an alert
queue is a paging, filtering, sorting workload over a few hundred thousand rows, and doing
that by loading a parquet file per request is the kind of thing that works in a demo of ten
rows and falls over in a demo of a hundred thousand - which is the demo that happens in front
of people.

SQLite is chosen over a server database for the reason it is usually chosen: the deployment
story is a file. A reviewer clones the repository, starts one process, and the queue works.
There is no container to orchestrate and no credential to distribute, and for a read-mostly
workload with one writer it is not a compromise.

Case state is here too. An alert a human has dispositioned is different from one nobody has
looked at, and that distinction cannot live in the parquet because the parquet is an
immutable record of what the model said.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    run_name      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    seed          INTEGER,
    git_sha       TEXT,
    config_hash   TEXT,
    library_hash  TEXT,
    headline      TEXT,
    provenance    TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    run_id            TEXT NOT NULL,
    txn_id            TEXT NOT NULL,
    timestamp         TEXT,
    amount            REAL,
    rail              TEXT,
    score             REAL,
    decision          INTEGER,
    model_alert       INTEGER,
    intent_block      INTEGER,
    control_block     INTEGER,
    is_fraud          INTEGER,
    attack_vector_id  TEXT,
    reason_codes      TEXT,
    PRIMARY KEY (run_id, txn_id)
);

-- Ordered by descending score because that is how a queue is worked, and a queue that has
-- to sort a hundred thousand rows per page is the thing this table exists to avoid.
CREATE INDEX IF NOT EXISTS idx_alerts_queue ON alerts (run_id, decision, score DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_vector ON alerts (run_id, attack_vector_id);

CREATE TABLE IF NOT EXISTS cases (
    run_id      TEXT NOT NULL,
    txn_id      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    disposition TEXT,
    note        TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, txn_id)
);
"""

#: Columns copied from the scored frame into the alerts table.
ALERT_COLUMNS = ["txn_id", "timestamp", "amount", "rail", "score", "decision",
                 "model_alert", "intent_block", "control_block", "is_fraud",
                 "attack_vector_id", "reason_codes"]


def _bind(value: Any) -> Any:
    """Coerce one dataframe cell into something sqlite3 will accept as a parameter.

    The frame arrives with numpy scalars, pandas timestamps and, for reason codes, a list.
    sqlite3 binds none of those, and the failure is a runtime error at the end of a long
    pipeline stage, so it is worth being exhaustive here rather than trusting dtypes.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


class Store:
    """A thin, explicit wrapper. No ORM, because the schema is nine columns wide."""

    def __init__(self, path: Path | str = "data/redteam.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL so a long-running read (a reviewer paging the queue) does not block the write
        # that finishes a pipeline run.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writes ------------------------------------------------------------
    def record_run(self, provenance, headline: Dict[str, Any]) -> None:
        p = provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, run_name, created_at, seed, git_sha, "
                "config_hash, library_hash, headline, provenance) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (p["run_id"], p["run_name"], p["created_at"], p["seed"], p["git_sha"],
                 p["config_hash"], p["library_hash"], json.dumps(headline, default=str),
                 json.dumps(p, default=str)),
            )

    def record_alerts(self, run_id: str, scored: pd.DataFrame) -> int:
        """Persist a scored test set. Idempotent, so re-running a stage does not duplicate."""
        frame = scored.copy()
        for column in ALERT_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[ALERT_COLUMNS]
        frame.insert(0, "run_id", run_id)
        rows = [tuple(_bind(v) for v in row)
                for row in frame.itertuples(index=False, name=None)]
        with self.connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO alerts (run_id, {', '.join(ALERT_COLUMNS)}) "
                f"VALUES ({', '.join('?' * (len(ALERT_COLUMNS) + 1))})",
                rows,
            )
        return len(rows)

    def set_case(self, run_id: str, txn_id: str, *, status: str = "open",
                 disposition: Optional[str] = None, note: Optional[str] = None) -> None:
        import time

        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cases (run_id, txn_id, status, disposition, note, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, txn_id, status, disposition, note,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )

    # -- reads -------------------------------------------------------------
    def runs(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT run_id, run_name, created_at, seed, git_sha, headline FROM runs "
                "ORDER BY created_at DESC")]

    def run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["headline"] = json.loads(out["headline"] or "{}")
        out["provenance"] = json.loads(out["provenance"] or "{}")
        return out

    def queue(self, run_id: str, *, limit: int = 50, offset: int = 0,
              alerts_only: bool = True, vector: Optional[str] = None,
              min_score: float = 0.0) -> List[Dict[str, Any]]:
        """One page of the alert queue, highest score first."""
        clauses = ["a.run_id = ?"]
        params: List[Any] = [run_id]
        if alerts_only:
            clauses.append("a.decision = 1")
        if vector:
            clauses.append("a.attack_vector_id = ?")
            params.append(vector)
        if min_score > 0:
            clauses.append("a.score >= ?")
            params.append(float(min_score))
        params.extend([int(limit), int(offset)])

        sql = (
            "SELECT a.*, c.status, c.disposition FROM alerts a "
            "LEFT JOIN cases c ON c.run_id = a.run_id AND c.txn_id = a.txn_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY a.score DESC LIMIT ? OFFSET ?"
        )
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def counts(self, run_id: str) -> Dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS scored, "
                "SUM(decision) AS alerts, "
                "SUM(is_fraud) AS fraud, "
                "SUM(CASE WHEN decision = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) AS caught "
                "FROM alerts WHERE run_id = ?", (run_id,)).fetchone()
        return {k: int(v or 0) for k, v in dict(row).items()}
