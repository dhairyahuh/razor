"""Serving a completed run's artefacts over HTTP, so a browser can read what a run produced.

The rest of the serving layer is about the *deployed model*: score a payment, fire a genome
at it, ask what is loaded. This module is about the *run* - the thirty-odd CSVs, the two
JSON summaries and the parquet working data that a finished pipeline leaves in
``artifacts/<run>/`` and ``data/<run>/``. Those are two genuinely different data sources and
this is the missing half rather than a second way of doing the first.

Three properties matter more than the endpoint list.

**Every payload names the file it came from.** A number on a screen with no provenance
invites the reader to wonder whether it was computed or chosen, and on a synthetic-data
submission that suspicion is the default. So a table is never returned as a bare array: it
arrives as ``{"source": "defend_per_vector_recall.csv", "available": true, "rows": [...]}``,
and the interface is expected to render the source next to the chart. A reader who wants to
check a figure should be one click from the file that contains it.

**A missing artefact is a value, not an exception.** Runs are partial all the time - a
``--no-loop`` run has no loop tables, a ``--shallow`` run has no importance, and a run that
was interrupted has whatever it got to. If a missing CSV raised, one absent file would take
down a whole route. Instead ``available`` goes false, ``rows`` is empty, and the panel that
needed it can say which file it wanted while every other panel on the page still renders.

**Nothing here computes a metric.** Every number served is read from a file that the
pipeline wrote and the report quotes. If this module started deriving figures of its own,
the site and the report would eventually disagree in front of whoever was reading them. The
two exceptions are joins rather than derivations - attaching measured recall to a vector,
and ordering vectors by residual risk - and both are marked where they happen.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "the serving layer needs FastAPI: pip install 'fastapi>=0.115' 'uvicorn>=0.32'."
    ) from exc

from ..schema import (
    AGENTIC_COLUMNS,
    AUTH_COLUMNS,
    BEHAVIOUR_COLUMNS,
    CUSTOMER_COLUMNS,
    DEVICE_COLUMNS,
    INSTRUCTION_COLUMNS,
    LABEL_COLUMNS,
    META_COLUMNS,
    PAYEE_COLUMNS,
    TIME_COLUMNS,
)

#: The schema blocks, in the order an analyst reads a payment: what was instructed, how it
#: was authenticated, what the device and the person looked like, who was being paid, what
#: the agent was doing, who the customer is. Mirrors :mod:`redteam.schema` rather than
#: re-grouping, so a column added there appears here without anyone remembering to.
SCHEMA_BLOCKS: Dict[str, List[str]] = {
    "instruction": INSTRUCTION_COLUMNS,
    "authentication": AUTH_COLUMNS,
    "device": DEVICE_COLUMNS,
    "behaviour": BEHAVIOUR_COLUMNS,
    "counterparty": PAYEE_COLUMNS,
    "agentic": AGENTIC_COLUMNS,
    "customer": CUSTOMER_COLUMNS,
    "time": TIME_COLUMNS,
}

#: Page size ceiling for the transaction endpoint. The test window is tens of thousands of
#: rows and the client virtualises, so pages are small and frequent by design.
MAX_PAGE = 500

#: How many sibling payments to return with a transaction. A mule ring can run to hundreds;
#: an analyst needs to see that it is a ring, not to scroll it.
MAX_SIBLINGS = 40


def _clean(value: Any) -> Any:
    """Make one cell safe for strict JSON.

    ``json.dumps`` will happily emit bare ``NaN``, which no browser can parse, and every
    numeric column in these frames has missing values somewhere. Numpy scalars need
    unwrapping for the same reason: they serialise as objects rather than numbers.
    """
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_clean(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """A dataframe as JSON-safe records."""
    if frame is None or frame.empty:
        return []
    return [{k: _clean(v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]


@dataclass
class ArtifactStore:
    """Cached, read-only access to what the runs on disk contain."""

    artifacts_root: Path = Path("artifacts")
    data_root: Path = Path("data")

    def __post_init__(self) -> None:
        self.artifacts_root = Path(self.artifacts_root)
        self.data_root = Path(self.data_root)
        # Keyed by (path, mtime) so an artefact rewritten by a fresh run is picked up
        # without a restart, which matters because a demo may well regenerate one.
        self._cache: Dict[tuple, Any] = {}

    # -- discovery -----------------------------------------------------
    def run_ids(self) -> List[str]:
        """Directories that look like a run, newest first.

        ``serving`` is excluded: it holds the deployed bundle, not a run.
        """
        if not self.artifacts_root.exists():
            return []
        found = [p for p in self.artifacts_root.iterdir()
                 if p.is_dir() and p.name != "serving" and any(p.glob("*.csv"))]
        return [p.name for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]

    def exists(self, run: str) -> bool:
        return (self.artifacts_root / run).is_dir()

    def require(self, run: str) -> str:
        if not self.exists(run):
            raise HTTPException(404, f"no run '{run}' in {self.artifacts_root}. "
                                     f"Available: {self.run_ids() or 'none'}")
        return run

    # -- reading -------------------------------------------------------
    def _cached(self, path: Path, load) -> Any:
        if not path.exists():
            return None
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._cache:
            # One entry per file per version. Runs are a handful of files and the frames are
            # the same ones the report already held in memory, so this is bounded by the
            # size of the run rather than by traffic.
            self._cache[key] = load(path)
        return self._cache[key]

    def frame(self, run: str, name: str) -> Optional[pd.DataFrame]:
        return self._cached(self.artifacts_root / run / name, pd.read_csv)

    def parquet(self, run: str, name: str) -> Optional[pd.DataFrame]:
        return self._cached(self.data_root / run / name, pd.read_parquet)

    def blob(self, run: str, name: str) -> Optional[Any]:
        return self._cached(self.artifacts_root / run / name,
                            lambda p: json.loads(p.read_text(encoding="utf-8")))

    def text(self, run: str, name: str) -> Optional[str]:
        return self._cached(self.artifacts_root / run / name,
                            lambda p: p.read_text(encoding="utf-8"))

    # -- envelopes -----------------------------------------------------
    def table(self, run: str, name: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
        """One CSV as a named, self-describing payload.

        ``available`` distinguishes "this run did not produce that table" from "that table
        is empty", which are different facts and lead to different empty states.
        """
        frame = self.frame(run, name)
        if frame is None:
            return {"source": name, "available": False, "rows": [], "n_rows": 0}
        shown = frame.head(limit) if limit else frame
        return {"source": name, "available": True, "rows": records(shown),
                "n_rows": int(len(frame)),
                "truncated": bool(limit and len(frame) > limit)}

    def provenance(self, run: str) -> Dict[str, Any]:
        """Run identity, from the summary the pipeline wrote.

        Degrades rather than failing: an interrupted run has no ``run_summary.json`` and is
        still worth browsing, so the run directory name stands in for an id and the missing
        fields are null. A UI that prints "git sha: unknown" is honest; one that 500s is not.
        """
        summary = self.blob(run, "run_summary.json") or {}
        stamped = summary.get("provenance") or {}
        return {
            "run": run,
            "run_id": stamped.get("run_id"),
            "run_name": stamped.get("run_name", run),
            "seed": stamped.get("seed"),
            "git_sha": stamped.get("git_sha"),
            "git_dirty": stamped.get("git_dirty"),
            "config_hash": stamped.get("config_hash"),
            "library_hash": stamped.get("library_hash"),
            "generated_at": stamped.get("created_at"),
            "environment": stamped.get("environment") or {},
        }

    def envelope(self, run: str, **payload: Any) -> Dict[str, Any]:
        return {"provenance": self.provenance(run), **payload}


# --------------------------------------------------------------------------------------
# Joins the report does not already do
# --------------------------------------------------------------------------------------

def _measured_recall(store: ArtifactStore, run: str) -> Dict[str, Dict[str, Any]]:
    """Per-vector recall, keyed by vector id, for attaching to the taxonomy."""
    frame = store.frame(run, "defend_per_vector_recall.csv")
    if frame is None or frame.empty:
        return {}
    return {str(r["attack_vector_id"]): {k: _clean(v) for k, v in r.items()}
            for r in frame.to_dict("records")}


def _residual_risk(vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank the taxonomy by risk that the defence has not been shown to cover.

    ``risk_score x (1 - measured_recall)``, which is the one table in this system a bank
    would actually act on: it puts the vectors that are both dangerous and undetected at the
    top, and it is the bridge from the Identify pillar to the Defend one.

    A vector with no measurement is not scored as zero risk and not dropped either. It is
    carried with ``measured_recall`` null and ``basis`` saying why - "documented only, never
    simulated" is a different and more alarming state than "simulated and caught", and
    collapsing the two would hide exactly the gap the table exists to show.
    """
    out = []
    for vector in vectors:
        recall = vector.get("measured_recall")
        rows = vector.get("measured_rows")
        if recall is None:
            basis = ("documented only, not simulated" if not vector.get("simulated")
                     else "simulated but absent from the test window")
            residual = float(vector["risk_score"])
        else:
            basis = (f"measured on {int(rows)} test rows" if rows
                     else "measured")
            residual = float(vector["risk_score"]) * (1.0 - float(recall))
        out.append({
            "id": vector["id"], "name": vector["name"], "family": vector["family"],
            "risk_score": vector["risk_score"], "severity": vector["severity"],
            "simulated": vector["simulated"],
            "measured_recall": recall, "measured_rows": rows,
            "recall_lo95": vector.get("recall_lo95"),
            "n_sufficient": vector.get("n_sufficient"),
            "residual_risk": round(residual, 4), "basis": basis,
        })
    return sorted(out, key=lambda r: -r["residual_risk"])


# --------------------------------------------------------------------------------------
# The router
# --------------------------------------------------------------------------------------

def create_router(store: ArtifactStore) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["artifacts"])

    @router.get("/runs")
    def list_runs() -> List[Dict[str, Any]]:
        """Every run on disk, with just enough to choose between them."""
        out = []
        for run in store.run_ids():
            provenance = store.provenance(run)
            summary = store.blob(run, "run_summary.json") or {}
            generate = summary.get("generate") or {}
            defend = (summary.get("defend") or {}).get("headline") or {}
            out.append({
                **provenance,
                "n_txns": generate.get("transactions"),
                "fraud": generate.get("fraud"),
                "fraud_rate": generate.get("fraud_rate"),
                "vectors_present": generate.get("vectors_present"),
                "recall": defend.get("recall"),
                "complete": bool(summary),
                "has_loop": store.frame(run, "loop_rounds.csv") is not None,
            })
        return out

    @router.get("/runs/{run}/summary")
    def run_summary(run: str) -> Dict[str, Any]:
        """The whole ``run_summary.json``, plus the headline as a table."""
        store.require(run)
        summary = store.blob(run, "run_summary.json")
        return store.envelope(
            run,
            source="run_summary.json",
            available=summary is not None,
            summary=_clean_tree(summary or {}),
            headline=store.table(run, "defend_headline.csv"),
            intervals=store.table(run, "defend_headline_intervals.csv"),
            split=store.table(run, "defend_split.csv"),
            has_report=store.text(run, "REPORT.md") is not None,
        )

    @router.get("/runs/{run}/identify")
    def identify(run: str) -> Dict[str, Any]:
        """The taxonomy, with each vector's measured recall attached.

        Vectors come from the YAML rather than ``identify_vectors.csv`` because the CSV
        export drops ``kill_chain``, ``description`` and ``channels``, and the kill-chain
        matrix is the whole point of this pillar.
        """
        from ..identify.library import load_library, summarise

        store.require(run)
        library = load_library()
        measured = _measured_recall(store, run)

        vectors = []
        for v in library.vectors:
            hit = measured.get(v.id, {})
            vectors.append({
                "id": v.id, "name": v.name, "family": v.family,
                "description": v.description, "genai_enablers": v.genai_enablers,
                "kill_chain": v.kill_chain, "rails": v.rails, "channels": v.channels,
                "victim": v.victim, "signals": v.signals, "controls": v.controls,
                "simulated": bool(v.simulated), "generator": v.generator,
                "severity": v.severity, "prevalence": v.prevalence,
                "detection_difficulty": v.detection_difficulty,
                "risk_score": v.risk_score, "liability_note": v.liability_note or None,
                # Joined from the defence, so the Identify screen can show what the Defend
                # screen measured without the reader having to hold two pages in their head.
                "measured_recall": hit.get("recall"),
                "measured_rows": hit.get("rows"),
                "recall_lo95": hit.get("recall_lo95"),
                "recall_hi95": hit.get("recall_hi95"),
                "n_sufficient": hit.get("n_sufficient"),
                "value_at_risk": hit.get("value"),
            })

        summary = summarise(library)
        return store.envelope(
            run,
            source="src/redteam/identify/attack_library.yaml",
            library_version=library.version,
            kill_chain_stages=library.kill_chain,
            families=[{"family": k, **v} for k, v in (summary["families"] or {}).items()],
            family_meta=library.families,
            vectors=vectors,
            residual_risk=_residual_risk(vectors),
            signal_usage=store.table(run, "identify_signal_usage.csv"),
            counts={
                "total": summary["total_vectors"],
                "simulated": summary["simulated_vectors"],
                "documented_only": summary["total_vectors"] - summary["simulated_vectors"],
                "measured": sum(1 for v in vectors if v["measured_recall"] is not None),
            },
            discovered=_discovered(store, run),
        )

    @router.get("/runs/{run}/fidelity")
    def fidelity(run: str) -> Dict[str, Any]:
        """How honest the generated data is, including the checks that failed."""
        store.require(run)
        summary = store.blob(run, "run_summary.json") or {}
        generate = summary.get("generate") or {}
        return store.envelope(
            run,
            scores=store.table(run, "generate_fidelity_scores.csv"),
            single_feature_auc=store.table(run, "generate_single_feature_auc.csv"),
            per_vector=store.table(run, "generate_per_vector.csv"),
            ablation=store.table(run, "defend_ablation_grid.csv"),
            overall=generate.get("fidelity"),
            flags=generate.get("flags") or [],
            warnings=generate.get("warnings") or [],
            zeroed=generate.get("zeroed_checks") or [],
            # The two separability probes. The gap between them is the reason most of the
            # generator work exists, and it is the number `/generate` leads with.
            raw_separability_recall=generate.get("joint_separability_recall"),
            derived_separability_recall=generate.get("derived_separability_recall"),
            counts={
                "transactions": generate.get("transactions"),
                "fraud": generate.get("fraud"),
                "fraud_rate": generate.get("fraud_rate"),
                "agent_context_bundles": generate.get("agent_context_bundles"),
                "transcripts": generate.get("transcripts"),
            },
        )

    @router.get("/runs/{run}/defend")
    def defend(run: str) -> Dict[str, Any]:
        """Everything the defence measured, one named table per artefact."""
        store.require(run)
        summary = store.blob(run, "run_summary.json") or {}
        return store.envelope(
            run,
            headline=store.table(run, "defend_headline.csv"),
            intervals=store.table(run, "defend_headline_intervals.csv"),
            split=store.table(run, "defend_split.csv"),
            operating_curve=store.table(run, "defend_operating_curve.csv"),
            per_vector=store.table(run, "defend_per_vector_recall.csv"),
            per_family=store.table(run, "defend_per_family_recall.csv"),
            false_positives=store.table(run, "defend_false_positives.csv"),
            fairness=store.table(run, "defend_fairness.csv"),
            baselines=store.table(run, "defend_baselines.csv"),
            expert_rules=store.table(run, "defend_expert_rules.csv"),
            cost_curve=store.table(run, "defend_cost_curve.csv"),
            cost_summary=store.table(run, "defend_cost_summary.csv"),
            evasion=store.table(run, "defend_evasion_delta.csv"),
            guards=store.table(run, "defend_guard_layers.csv"),
            guard_leakage=store.table(run, "defend_guard_leakage.csv"),
            intent_coverage=store.table(run, "defend_intent_coverage.csv"),
            control_coverage=store.table(run, "defend_agent_control_coverage.csv"),
            ceiling_bound=store.table(run, "defend_scoped_token_loss_bound.csv"),
            zero_day=store.table(run, "defend_zero_day.csv"),
            injection_unseen_family=store.table(run, "defend_injection_unseen_family.csv"),
            injection_unseen_phrasing=store.table(run,
                                                  "defend_injection_unseen_phrasing.csv"),
            vishing_unseen_script=store.table(run, "defend_vishing_unseen_script.csv"),
            vishing_unseen_wording=store.table(run, "defend_vishing_unseen_wording.csv"),
            worst_slices=store.table(run, "defend_worst_slices.csv"),
            ablation=store.table(run, "defend_ablation_grid.csv"),
            null_control=store.table(run, "defend_null_control.csv"),
            lift=store.table(run, "defend_decile_lift.csv"),
            importance=store.table(run, "defend_feature_importance.csv", limit=40),
            narratives=store.table(run, "defend_alert_narratives.csv"),
            guard_reports={
                "injection": (summary.get("defend") or {}).get("injection_guard"),
                "vishing": (summary.get("defend") or {}).get("vishing_guard"),
                "media": (summary.get("defend") or {}).get("media_guard"),
            },
        )

    @router.get("/runs/{run}/loop")
    def loop(run: str) -> Dict[str, Any]:
        """The arms race: rounds, tactics, what blue did, and what got written back."""
        store.require(run)
        summary = store.blob(run, "run_summary.json") or {}
        return store.envelope(
            run,
            rounds=store.table(run, "loop_rounds.csv"),
            tactics=store.table(run, "loop_tactics.csv"),
            blue_moves=store.table(run, "loop_blue_moves.csv"),
            control_gaps=store.table(run, "loop_control_gaps.csv"),
            transfer_matrix=store.table(run, "loop_transfer_matrix.csv"),
            cost_model=store.table(run, "loop_cost_model.csv"),
            convergence=summary.get("loop_convergence"),
            discovered=_discovered(store, run),
        )

    @router.get("/runs/{run}/transactions")
    def transactions(run: str,
                     cursor: int = Query(0, ge=0),
                     limit: int = Query(100, ge=1, le=MAX_PAGE),
                     rail: Optional[str] = None,
                     vector: Optional[str] = None,
                     decision: Optional[str] = Query(
                         None, pattern="^(alert|approve|fraud|missed|false_positive)$"),
                     hard_negative: Optional[bool] = None,
                     min_score: float = Query(0.0, ge=0.0, le=1.0),
                     max_score: float = Query(1.0, ge=0.0, le=1.0),
                     min_amount: float = Query(0.0, ge=0.0),
                     max_amount: Optional[float] = None) -> Dict[str, Any]:
        """A filtered page of the scored test window, cursor-paginated.

        Cursor rather than offset because the client virtualises a table of tens of
        thousands of rows and scrolls it continuously; an offset scheme re-derives the
        filter on every page and drifts if anything underneath changes.
        """
        store.require(run)
        frame = store.frame(run, "defend_scored_test_set.csv")
        if frame is None:
            return store.envelope(run, source="defend_scored_test_set.csv",
                                  available=False, rows=[], next_cursor=None, total=0)

        view = frame
        if rail:
            view = view[view["rail"] == rail]
        if vector:
            view = view[view["attack_vector_id"] == vector]
        if hard_negative is not None:
            view = view[view["is_hard_negative"] == int(hard_negative)]
        if decision == "alert":
            view = view[view["decision"] == 1]
        elif decision == "approve":
            view = view[view["decision"] == 0]
        elif decision == "fraud":
            view = view[view["is_fraud"] == 1]
        elif decision == "missed":
            view = view[(view["is_fraud"] == 1) & (view["decision"] == 0)]
        elif decision == "false_positive":
            view = view[(view["is_fraud"] == 0) & (view["decision"] == 1)]
        view = view[(view["score"] >= min_score) & (view["score"] <= max_score)]
        view = view[view["amount"] >= min_amount]
        if max_amount is not None:
            view = view[view["amount"] <= max_amount]

        total = int(len(view))
        page = view.iloc[cursor:cursor + limit]
        nxt = cursor + limit
        return store.envelope(
            run,
            source="defend_scored_test_set.csv",
            available=True,
            rows=records(page),
            total=total,
            cursor=cursor,
            next_cursor=nxt if nxt < total else None,
        )

    @router.get("/runs/{run}/transactions/{txn_id}")
    def transaction(run: str, txn_id: str) -> Dict[str, Any]:
        """One payment, in as much detail as the run preserved."""
        store.require(run)
        scored = store.frame(run, "defend_scored_test_set.csv")
        if scored is None:
            raise HTTPException(404, "this run has no scored test set")
        hit = scored[scored["txn_id"] == txn_id]
        if hit.empty:
            raise HTTPException(404, f"no transaction {txn_id} in run {run}")
        verdict = {k: _clean(v) for k, v in hit.iloc[0].items()}

        full = _full_row(store, run, txn_id)
        blocks: Dict[str, List[Dict[str, Any]]] = {}
        if full is not None:
            baselines = store.frame(run, "defend_benign_baselines.csv")
            reference = ({str(r["column"]): r for r in baselines.to_dict("records")}
                         if baselines is not None else {})
            for block, columns in SCHEMA_BLOCKS.items():
                fields = []
                for column in columns:
                    if column not in full:
                        continue
                    stats = reference.get(column)
                    fields.append({
                        "column": column,
                        "value": _clean(full[column]),
                        # The legitimate distribution beside the value, so a deviation reads
                        # at a glance instead of requiring the reader to know the field.
                        "benign_median": _clean(stats["median"]) if stats is not None else None,
                        "benign_p05": _clean(stats["p05"]) if stats is not None else None,
                        "benign_p95": _clean(stats["p95"]) if stats is not None else None,
                    })
                if fields:
                    blocks[block] = fields

        return store.envelope(
            run,
            source="defend_scored_test_set.csv + data/{run}/transactions.parquet",
            txn_id=txn_id,
            verdict=verdict,
            reason_codes=[c.strip() for c in str(verdict.get("reason_codes") or "").split(",")
                          if c.strip()],
            blocks=blocks,
            counterfactuals=_counterfactuals(store, run, txn_id),
            meta={k: _clean(full[k]) for k in META_COLUMNS if full is not None and k in full},
            labels={k: _clean(full[k]) for k in LABEL_COLUMNS
                    if full is not None and k in full},
            siblings=_siblings(store, run, full),
            agent_bundle=_agent_bundle(store, run, txn_id),
            transcript=_transcript(store, run, txn_id),
        )

    @router.get("/runs/{run}/report")
    def report(run: str) -> Dict[str, Any]:
        """``REPORT.md`` as text, for the evidence page to render."""
        store.require(run)
        text = store.text(run, "REPORT.md")
        return store.envelope(run, source="REPORT.md", available=text is not None,
                              markdown=text or "")

    @router.get("/runs/{run}/file/{name}")
    def raw_file(run: str, name: str) -> FileResponse:
        """One artefact, byte for byte, as it was written.

        Every number the interface shows names the file it came from, and a name a reader
        cannot open is a citation they have to take on trust. This endpoint is what turns
        those labels into links, so checking a figure is one click rather than a request
        for the repository.

        Only files directly inside the run directory are reachable: the name is reduced to
        its last path component before use, which drops ``..`` and absolute paths, and the
        resolved path is then required to sit under the run. Nothing here is behind auth,
        so a traversal would expose the host filesystem.
        """
        store.require(run)
        directory = (store.artifacts_root / run).resolve()
        path = (directory / Path(name).name).resolve()
        if path.parent != directory or not path.is_file():
            raise HTTPException(404, f"no artefact '{name}' in run '{run}'")
        return FileResponse(path, filename=path.name,
                            media_type=_MEDIA_TYPES.get(path.suffix, "application/octet-stream"))

    return router


# Served inline where a browser can render it, so clicking a source link shows the CSV in
# a tab rather than dropping a download the reader then has to find.
_MEDIA_TYPES = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


# --------------------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------------------

def _clean_tree(value: Any) -> Any:
    """Recursively JSON-safe a nested structure read from disk.

    Older runs were written before the report started sanitising its own JSON, so a
    ``run_summary.json`` sitting on disk may still contain ``NaN``. Reading it back into
    Python succeeds and re-serialising it to a browser does not, and the failure surfaces as
    an unparseable response rather than as anything that names the cause.
    """
    if isinstance(value, dict):
        return {k: _clean_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_tree(v) for v in value]
    return _clean(value)


def _full_row(store: ArtifactStore, run: str, txn_id: str) -> Optional[pd.Series]:
    """The complete schema row, from the featured test set if present or the raw stream.

    ``test_features.parquet`` is preferred because it carries the derived and guard columns
    as well, which is what a counterfactual needs. Runs made before that file existed fall
    back to ``transactions.parquet``, which still has every schema block the inspector shows.
    """
    for name in ("test_features.parquet", "transactions.parquet"):
        frame = store.parquet(run, name)
        if frame is None:
            continue
        hit = frame[frame["txn_id"] == txn_id]
        if not hit.empty:
            return hit.iloc[0]
    return None


def _counterfactuals(store: ArtifactStore, run: str, txn_id: str) -> Dict[str, Any]:
    """What single change would have cleared this alert, if any one change would.

    Three states, and they are genuinely different answers rather than degrees of the same
    one. ``available: false`` means the run never computed counterfactuals. ``sampled:
    false`` means it did, but this payment was not in the bounded sample - deriving one per
    alert across the whole window is not free, so the pipeline takes a few hundred. An empty
    ``rows`` for a payment that *was* sampled is the interesting case: no single actionable
    field would have flipped the decision, so the alert rests on a combination, and saying
    that is more informative than saying nothing.
    """
    frame = store.frame(run, "defend_counterfactuals.csv")
    if frame is None:
        return {"source": "defend_counterfactuals.csv", "available": False,
                "sampled": False, "rows": []}
    hit = frame[frame["txn_id"] == txn_id]
    # A sampled alert with nothing that would flip it is written as one row with a null
    # column, so its absence from the results is recorded rather than inferred.
    found = hit[hit["column"].notna()] if "column" in hit.columns else hit
    return {
        "source": "defend_counterfactuals.csv",
        "available": True,
        "sampled": bool(len(hit)),
        "rows": records(found),
    }


def _siblings(store: ArtifactStore, run: str, row: Optional[pd.Series]) -> Dict[str, Any]:
    """Other payments in the same campaign or mule ring.

    The unit an analyst should be working is the case, not the payment. Forty transfers into
    one ring is one investigation; presented as forty queue items it is forty decisions, each
    made without the context that would settle it.
    """
    if row is None:
        return {"available": False, "campaign": [], "ring": []}
    scored = store.frame(run, "defend_scored_test_set.csv")
    source = store.parquet(run, "test_features.parquet")
    if source is None:
        source = store.parquet(run, "transactions.parquet")
    if source is None:
        return {"available": False, "campaign": [], "ring": []}

    scores = ({str(r["txn_id"]): r for r in scored.to_dict("records")}
              if scored is not None else {})

    def gather(column: str) -> List[Dict[str, Any]]:
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
            return []
        kin = source[(source[column] == value) & (source["txn_id"] != row["txn_id"])]
        out = []
        for _, other in kin.head(MAX_SIBLINGS).iterrows():
            verdict = scores.get(str(other["txn_id"]), {})
            out.append({
                "txn_id": other["txn_id"],
                "timestamp": _clean(other.get("timestamp")),
                "amount": _clean(other.get("amount")),
                "rail": _clean(other.get("rail")),
                "is_fraud": _clean(other.get("is_fraud")),
                "score": _clean(verdict.get("score")),
                "decision": _clean(verdict.get("decision")),
                "in_test_window": str(other["txn_id"]) in scores,
            })
        return out

    return {"available": True,
            "campaign_id": _clean(row.get("campaign_id")),
            "mule_ring_id": _clean(row.get("mule_ring_id")),
            "campaign": gather("campaign_id"),
            "ring": gather("mule_ring_id")}


def _agent_bundle(store: ArtifactStore, run: str, txn_id: str) -> Optional[Dict[str, Any]]:
    """The context window the agent read, with the injected span located exactly.

    Offsets come from the generator, which recorded them where it inserted the payload.
    Nothing here searches the text for the payload: several obfuscations rewrite it after
    composition, so a search would either miss or highlight the wrong run of characters, and
    an interface that points confidently at the wrong span is worse than one that points at
    nothing.
    """
    corpus = store.parquet(run, "agent_corpus.parquet")
    if corpus is None:
        return None
    hit = corpus[corpus["txn_id"] == txn_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    start = int(row["payload_start"]) if "payload_start" in row else -1
    end = int(row["payload_end"]) if "payload_end" in row else -1
    return {
        "source": f"data/{run}/agent_corpus.parquet",
        "text": str(row["text"]),
        "has_injection": bool(row["has_injection"]),
        "payload_family": _clean(row.get("payload_family")),
        "obfuscation": _clean(row.get("obfuscation")),
        "payload_template": _clean(row.get("payload_template")),
        # Null rather than -1 when absent, so a client cannot slice by it accidentally.
        "payload_start": start if start >= 0 else None,
        "payload_end": end if end >= 0 else None,
        "payload_text": str(row["text"])[start:end] if start >= 0 else None,
    }


def _transcript(store: ArtifactStore, run: str, txn_id: str) -> Optional[Dict[str, Any]]:
    """The call or chat attached to this payment, if the generator produced one."""
    frame = store.parquet(run, "transcripts.parquet")
    if frame is None:
        return None
    hit = frame[frame["txn_id"] == txn_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "source": f"data/{run}/transcripts.parquet",
        "transcript_id": _clean(row.get("transcript_id")),
        "channel": _clean(row.get("channel")),
        "text": str(row["text"]),
        "is_coercive": bool(row.get("is_coercive", 0)),
        "scam_script": _clean(row.get("scam_script")),
    }


def _discovered(store: ArtifactStore, run: str) -> Dict[str, Any]:
    """Vectors the loop found and wrote back into the taxonomy.

    The moment the loop closes, so it is served from both ``/identify`` and ``/loop``
    rather than living on one of them: the write-back panel needs it to show the arrow
    leaving, and the atlas needs it to show the arrow arriving.
    """
    text = store.text(run, "discovered_vectors.yaml")
    if text is None:
        return {"source": "discovered_vectors.yaml", "available": False, "vectors": []}
    try:
        import yaml

        parsed = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001 - a malformed artefact must not take down the route
        return {"source": "discovered_vectors.yaml", "available": False, "vectors": [],
                "error": "could not parse"}
    return {
        "source": "discovered_vectors.yaml",
        "available": True,
        "version": parsed.get("version"),
        "note": parsed.get("note"),
        "vectors": parsed.get("vectors") or [],
        "yaml": text,
    }
