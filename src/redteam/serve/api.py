"""The HTTP surface: score a payment, drive the loop, inspect what is deployed.

The endpoint that matters is ``POST /score``. Everything else exists so that a reviewer can
*operate* the system rather than read a report about it - launch a generation run, fire a
red-team genome at the live model, watch recall fall and then recover.

Three design decisions worth stating, because each is the kind a payments audience checks.

**Scoring is stateless and the derived features are the caller's problem.** A single payment
arriving over HTTP has no history, and this system's model is mostly history: velocity
windows, counterparty novelty, graph position. A service that silently filled those with
zeros would return a confident number computed from a fiction. So ``/score`` accepts the
derived features when the caller has them and reports ``feature_completeness`` when it does
not, and the response says plainly how much of the matrix was actually populated. That is
the honest interface for a component that would sit behind a real feature store.

**The decision is the union of a model alert and the deterministic guards**, exactly as in
the batch pipeline, and the response separates them. "Declined because the intent artefact
named a different payee" is an auditable reason a bank can defend to a customer and a
regulator. "Declined because the score was 0.94" is not, and conflating the two costs the
first one its value.

**Nothing here trains.** The service loads a bundle produced by the offline pipeline. A
request cannot cause a fit, so no request can change what any other request gets back.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "the serving layer needs FastAPI: pip install 'fastapi>=0.115' 'uvicorn>=0.32'. "
        "The offline pipeline does not require it."
    ) from exc

from ..config import Config
from ..provenance import stamp
from ..schema import DEFAULTS, observable_columns
from .artifacts import ArtifactStore, create_router
from .bundle import ServingBundle
from .store import Store

#: Where the service looks for its artefact unless told otherwise.
DEFAULT_BUNDLE = Path("artifacts/serving/bundle.pkl")

#: Requests per minute per client. Deliberately modest and deliberately present: an
#: unauthenticated scoring endpoint with no limit is a free oracle for anyone who wants to
#: map the decision boundary, which is a threat this repository models explicitly as
#: MODEL-ORACLE-PROBING. Leaving it off while shipping an attack that exploits it would be
#: an odd thing to do.
RATE_LIMIT_PER_MINUTE = 120


# --------------------------------------------------------------------------------------
# Request and response shapes
# --------------------------------------------------------------------------------------

class PaymentRequest(BaseModel):
    """One payment, as a caller would present it at authorisation time.

    Only ``amount`` and ``rail`` are required. Everything else defaults to its schema value,
    which is what a sparse caller gets and why the response reports completeness.
    """

    amount: float = Field(..., gt=0, le=1e9, description="Payment amount in minor-unit-free rupees")
    rail: str = Field(
        ...,
        pattern="^(upi|imps|neft|rtgs|card)$",
        description="upi, imps, neft, rtgs or card",
    )
    channel: str = Field("mobile_app", description="mobile_app, web, pos, ivr, agent_api, ...")
    timestamp: Optional[str] = Field(None, description="ISO 8601; defaults to now")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any further schema or derived columns the caller can supply. "
                    "Unknown keys are rejected rather than ignored.",
    )

    model_config = {"json_schema_extra": {"examples": [{
        "amount": 48000.0,
        "rail": "upi",
        "channel": "mobile_app",
        "features": {
            "payee_is_first_time": 1,
            "payee_account_age_days": 2,
            "call_in_progress": 1,
            "f_payer_txn_count_1h": 4,
        },
    }]}}


class ScoreResponse(BaseModel):
    score: float
    threshold: float
    model_alert: bool
    intent_block: bool
    control_block: bool
    decision: str
    reason_codes: List[str]
    feature_completeness: float
    features_supplied: int
    features_expected: int
    latency_ms: float
    provenance: Dict[str, Any]


class AttackRequest(BaseModel):
    """One red-team genome, fired at the deployed model."""

    vector_id: str = Field(...,
                           description="Attack vector to mutate, e.g. APP-SAFE-ACCOUNT-SCREENSHARE")
    levers: List[str] = Field(
        default_factory=list,
        description="Evasion levers to pull; empty pulls every one of them, which is the "
                    "strongest available mutation rather than a typical one",
    )
    rows: int = Field(200, ge=20, le=5000,
                      description="Requested event count. The generator works in whole "
                                  "episodes, so the realised row count is reported back and "
                                  "will usually exceed this.")


class CaseUpdate(BaseModel):
    status: str = Field("open", pattern="^(open|in_review|closed)$")
    disposition: Optional[str] = Field(None, pattern="^(fraud|genuine|inconclusive)$")
    note: Optional[str] = Field(None, max_length=2000)


# --------------------------------------------------------------------------------------
# Application state
# --------------------------------------------------------------------------------------

class ServiceState:
    """Loaded once at startup. Holds the bundle, the store and the in-flight run registry."""

    def __init__(self, bundle_path: Path = DEFAULT_BUNDLE,
                 store_path: Path = Path("data/redteam.sqlite")):
        self.bundle_path = Path(bundle_path)
        self.bundle: Optional[ServingBundle] = None
        self.store = Store(store_path)
        self.load_error: str = ""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._hits: Dict[str, List[float]] = {}
        self.reload()

    def reload(self) -> None:
        try:
            self.bundle = ServingBundle.load(self.bundle_path)
            self.load_error = ""
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            # A missing bundle is the normal state of a fresh clone, not a crash. /health
            # reports it and /score refuses; every other endpoint still works, so a reviewer
            # can read the model card and the run history before building one.
            self.bundle = None
            self.load_error = str(exc)

    def require_bundle(self) -> ServingBundle:
        if self.bundle is None:
            raise HTTPException(
                status_code=503,
                detail=(f"no serving bundle at {self.bundle_path}: {self.load_error or 'absent'}. "
                        "Run `python -m redteam run --quick` to build one."),
            )
        return self.bundle

    def check_rate(self, client: str) -> None:
        now = time.monotonic()
        hits = [t for t in self._hits.get(client, []) if now - t < 60.0]
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit of {RATE_LIMIT_PER_MINUTE} requests per minute exceeded",
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        self._hits[client] = hits


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def build_row(payload: PaymentRequest, columns: List[str]) -> tuple[pd.DataFrame, int]:
    """Turn a request into a single-row frame the detector can read.

    Unknown keys raise rather than being dropped. A caller who misspells
    ``payee_account_age_days`` and gets a 200 back has been told their feature was used when
    it was silently replaced by a default, and that is a far worse failure than a 422.
    """
    known = set(DEFAULTS) | set(observable_columns()) | set(columns)
    supplied = dict(payload.features)
    unknown = sorted(k for k in supplied if k not in known)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown feature columns: {unknown[:8]}. They would have been silently "
                   f"replaced by defaults, which would misreport what was scored.",
        )

    row: Dict[str, Any] = dict(DEFAULTS)
    row.update({
        "amount": float(payload.amount),
        "rail": payload.rail,
        "channel": payload.channel,
        "timestamp": pd.Timestamp(payload.timestamp) if payload.timestamp
                     else pd.Timestamp.utcnow().tz_localize(None),
    })
    row.update(supplied)

    frame = pd.DataFrame([row])
    # Derived columns the caller did not supply are left missing rather than zeroed. The
    # boosting ensemble treats NaN as "unknown" and routes on it; filling zeros would assert
    # "this payer has made no payments in the last hour", which is a different and confident
    # claim about a payer nobody looked up.
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)],
                          axis=1)
    return frame, len(supplied)


def _flag(frame: pd.DataFrame, column: str) -> bool:
    """Read a deterministic guard flag off the request frame.

    Absent and NaN both mean the caller never ran that control, which is not the same as the
    control clearing the payment. Both read as false here, and the response reports
    completeness so the difference is visible rather than implied.
    """
    if column not in frame.columns:
        return False
    value = frame[column].iloc[0]
    return bool(value) if pd.notna(value) else False


def score_payment(bundle: ServingBundle, payload: PaymentRequest) -> ScoreResponse:
    started = time.perf_counter()
    frame, supplied = build_row(payload, bundle.columns)

    score = float(bundle.detector.predict_proba(frame)[0])
    model_alert = score >= bundle.threshold

    intent = _flag(frame, "intent_guard_blocked")
    control = _flag(frame, "agent_control_blocked")

    reasons: List[str] = []
    if model_alert:
        if bundle.importance.empty:
            # A bundle built by a --shallow run has no permutation importance, and reason
            # codes are derived from it. Saying so beats returning an empty list, which
            # reads as "the model alerted for no reason" rather than "this build cannot
            # tell you the reason".
            reasons = ["REASONS_UNAVAILABLE_SHALLOW_BUILD"]
        else:
            raw = bundle.detector.reason_codes(frame, bundle.importance)
            # A reason naming a feature the caller never supplied reads as evidence and is
            # the opposite: `payee_name_match_score=nan` says the model was not told, not
            # that the name failed to match. Dropped here rather than in the detector,
            # because in the batch pipeline every column is populated and the case does not
            # arise.
            reasons = [r.strip() for r in str(raw[0]).split(",")
                       if r.strip() and not r.strip().endswith("=nan")]
    if intent:
        reasons.insert(0, "INTENT_GUARD_BLOCK")
    if control:
        reasons.insert(0, "AGENT_CONTROL_BLOCK")

    # Completeness over the derived layer specifically. A caller supplying the raw payment
    # message and nothing else has a complete *message* and an almost empty *matrix*, and it
    # is the second number that determines whether the score means anything.
    derived = [c for c in bundle.columns if c.startswith(("f_", "g_"))]
    present = sum(1 for c in derived if c in payload.features)
    completeness = round(present / len(derived), 4) if derived else 1.0

    decision = "decline" if (model_alert or intent or control) else "approve"
    return ScoreResponse(
        score=round(score, 6),
        threshold=round(float(bundle.threshold), 6),
        model_alert=model_alert,
        intent_block=intent,
        control_block=control,
        decision=decision,
        reason_codes=reasons,
        feature_completeness=completeness,
        features_supplied=supplied,
        features_expected=len(bundle.columns),
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        provenance={"run_id": bundle.provenance.run_id,
                    "git_sha": bundle.provenance.git_sha,
                    "config_hash": bundle.provenance.config_hash},
    )


# --------------------------------------------------------------------------------------
# The application
# --------------------------------------------------------------------------------------

def create_app(bundle_path: Path = DEFAULT_BUNDLE,
               store_path: Path = Path("data/redteam.sqlite"),
               artifacts_root: Path = Path("artifacts"),
               data_root: Path = Path("data")) -> FastAPI:
    state = ServiceState(bundle_path, store_path)

    app = FastAPI(
        title="Razor — AI Risk Manager for Payment Fraud",
        version="1.0.0",
        description=(
            "Defensive fraud detection ensemble trained and evaluated on synthetic payment "
            "data. Scores payments, reports precision and recall on a held-out test set, and "
            "exposes audit artefacts. **Not fit for scoring real customer traffic**: no "
            "threshold here transfers to a real portfolio. See `GET /model-card` for the "
            "full statement of intended use and limitations."
        ),
    )
    app.state.service = state

    # The UI is served from a different origin during development (Vite on :5173) and from
    # the same one in the packaged container. Permissive here because the service holds no
    # credentials, no session and no customer data - it scores synthetic payments - so the
    # thing CORS protects does not exist. Anything that did would need this narrowed.
    # --- global error handler ---------------------------------------------------
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """Catch-all: return structured JSON instead of a 500 stack trace."""
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{type(exc).__name__}: {exc}",
                "hint": "If this persists, run `python -m redteam run --quick` to rebuild the serving bundle.",
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    def service() -> ServiceState:
        return state

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path in ("/score", "/attack", "/simulate"):
            client = request.client.host if request.client else "unknown"
            try:
                state.check_rate(client)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code,
                                    content={"detail": exc.detail},
                                    headers=exc.headers or {})
        return await call_next(request)

    # -- operational ---------------------------------------------------
    @app.get("/health", tags=["operational"])
    def health() -> Dict[str, Any]:
        """Liveness plus what is actually loaded.

        Reports the bundle's run id rather than only "ok", because a green health check on a
        service quietly serving last week's model is worse than a red one.
        """
        return {
            "status": "ok" if state.bundle is not None else "degraded",
            "bundle_loaded": state.bundle is not None,
            "bundle_path": str(state.bundle_path),
            "detail": state.load_error or None,
            "serving_run_id": state.bundle.provenance.run_id if state.bundle else None,
            "git_sha": state.bundle.provenance.git_sha if state.bundle else None,
        }

    @app.get("/model-card", tags=["operational"])
    def model_card(svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        """The governance view, generated from the deployed bundle so it cannot disagree."""
        return svc.require_bundle().model_card()

    @app.post("/reload", tags=["operational"])
    def reload_bundle(svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        svc.reload()
        return health()

    # -- scoring -------------------------------------------------------
    @app.post("/score", response_model=ScoreResponse, tags=["scoring"])
    def score(payload: PaymentRequest,
              svc: ServiceState = Depends(service)) -> ScoreResponse:
        """Score one payment.

        The response separates the model's alert from the deterministic guard blocks. That
        separation is the point: a decline with a named guard reason is defensible to a
        customer and a regulator, and a decline with a probability is not.
        """
        return score_payment(svc.require_bundle(), payload)

    @app.post("/score/batch", tags=["scoring"])
    def score_batch(payloads: List[PaymentRequest],
                    svc: ServiceState = Depends(service)) -> List[ScoreResponse]:
        if len(payloads) > 500:
            raise HTTPException(422, "batch limit is 500 payments")
        bundle = svc.require_bundle()
        return [score_payment(bundle, p) for p in payloads]

    # -- runs and the queue --------------------------------------------
    @app.get("/runs", tags=["runs"])
    def list_runs(svc: ServiceState = Depends(service)) -> List[Dict[str, Any]]:
        return svc.store.runs()

    @app.get("/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        record = svc.store.run(run_id)
        if record is None:
            raise HTTPException(404, f"no run {run_id}")
        record["counts"] = svc.store.counts(run_id)
        return record

    @app.get("/runs/{run_id}/alerts", tags=["runs"])
    def alert_queue(run_id: str,
                    limit: int = Query(50, ge=1, le=500),
                    offset: int = Query(0, ge=0),
                    alerts_only: bool = True,
                    vector: Optional[str] = None,
                    min_score: float = Query(0.0, ge=0.0, le=1.0),
                    svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        """One page of the alert queue, highest score first."""
        rows = svc.store.queue(run_id, limit=limit, offset=offset,
                               alerts_only=alerts_only, vector=vector, min_score=min_score)
        return {"run_id": run_id, "offset": offset, "limit": limit,
                "counts": svc.store.counts(run_id), "alerts": rows}

    @app.put("/runs/{run_id}/cases/{txn_id}", tags=["runs"])
    def update_case(run_id: str, txn_id: str, update: CaseUpdate,
                    svc: ServiceState = Depends(service)) -> Dict[str, str]:
        svc.store.set_case(run_id, txn_id, status=update.status,
                           disposition=update.disposition, note=update.note)
        return {"run_id": run_id, "txn_id": txn_id, "status": update.status}

    # -- driving the pipeline ------------------------------------------
    @app.post("/simulate", status_code=202, tags=["red team"])
    async def simulate(quick: bool = True, seed: Optional[int] = None,
                       svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        """Launch a generation-and-defence run in the background.

        Returns a job id immediately rather than blocking, because the quick config is
        minutes and the default config is far longer. Progress streams from
        ``GET /jobs/{id}/events``.
        """
        job_id = f"job-{int(time.time() * 1000):x}"
        svc.jobs[job_id] = {"status": "queued", "stage": None, "events": [],
                            "started_at": time.time()}
        asyncio.get_running_loop().run_in_executor(
            None, _run_pipeline_job, svc, job_id, bool(quick), seed
        )
        return {"job_id": job_id, "status": "queued",
                "events": f"/jobs/{job_id}/events"}

    @app.get("/jobs/{job_id}", tags=["red team"])
    def job_status(job_id: str, svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        job = svc.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return {k: v for k, v in job.items() if k != "events"} | {
            "event_count": len(job["events"])}

    @app.get("/jobs/{job_id}/events", tags=["red team"])
    async def job_events(job_id: str, svc: ServiceState = Depends(service)):
        """Server-sent events for stage progress.

        A long run behind a spinner is indistinguishable from a hung one, and during a demo
        that ambiguity is expensive.
        """
        if job_id not in svc.jobs:
            raise HTTPException(404, f"no job {job_id}")

        async def stream():
            sent = 0
            while True:
                job = svc.jobs.get(job_id, {})
                events = job.get("events", [])
                while sent < len(events):
                    yield f"data: {json.dumps(events[sent])}\n\n"
                    sent += 1
                if job.get("status") in ("done", "failed"):
                    yield f"data: {json.dumps({'stage': 'end', 'status': job.get('status')})}\n\n"
                    return
                await asyncio.sleep(0.4)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/attack", tags=["red team"])
    def attack(request: AttackRequest,
               svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        """Mutate one attack vector and re-score it against the deployed model.

        The demo moment, and also the honest one: it reports recall before and after on the
        *same* rows, so the number is a like-for-like comparison rather than two campaigns of
        different size being set beside each other.
        """
        bundle = svc.require_bundle()
        return _run_attack(bundle, request)

    # -- the run's artefacts -------------------------------------------
    # A separate router over a separate data source: the endpoints above serve the *deployed
    # model*, these serve what a completed *run* wrote to disk. Mounted here rather than in
    # its own application so a reviewer has one base URL and one OpenAPI schema.
    app.include_router(create_router(ArtifactStore(artifacts_root, data_root)))

    # -- aliases -------------------------------------------------------
    # The read-only surface lives under /api; these put the interactive endpoints beside it
    # so a browser client has one prefix for everything. Aliases rather than renames,
    # because the bare paths are what the existing tests and the CLI already use.
    @app.get("/api/health", tags=["operational"])
    def api_health() -> Dict[str, Any]:
        state_of_play = health()
        store = ArtifactStore(artifacts_root, data_root)
        runs = store.run_ids()
        return {**state_of_play,
                # "live" means a bundle is loaded and /score will answer. Without one the
                # site can still render every artefact, and saying so plainly is what lets
                # the UI show an honest mode banner rather than guessing.
                "mode": "live" if state_of_play["bundle_loaded"] else "demo",
                "runs": runs,
                "default_run": runs[0] if runs else None}

    @app.get("/api/model-card", tags=["operational"])
    def api_model_card(svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        return svc.require_bundle().model_card()

    @app.post("/api/score", response_model=ScoreResponse, tags=["scoring"])
    def api_score(payload: PaymentRequest,
                  svc: ServiceState = Depends(service)) -> ScoreResponse:
        return score_payment(svc.require_bundle(), payload)

    @app.post("/api/attack", tags=["red team"])
    def api_attack(request: AttackRequest,
                   svc: ServiceState = Depends(service)) -> Dict[str, Any]:
        return _run_attack(svc.require_bundle(), request)

    @app.get("/api/stream/{job_id}", tags=["red team"])
    async def api_stream(job_id: str, svc: ServiceState = Depends(service)):
        return await job_events(job_id, svc)

    @app.get("/api/metrics", tags=["operational"])
    def api_metrics() -> Dict[str, Any]:
        """The submission headline: precision, recall, F1 on the held-out test set.

        Reads from the latest run's results.json so a judge can hit one URL and
        see every metric Track 02 asks for.
        """
        results_path = artifacts_root / "quick" / "results.json"
        if not results_path.exists():
            results_path = next(artifacts_root.glob("*/results.json"), None)
        if results_path is None or not results_path.exists():
            raise HTTPException(404, "No results.json found. Run `python -m redteam run --quick` first.")

        results = json.loads(results_path.read_text())
        defend = results.get("defend", {})
        intervals = results.get("defend_intervals", [])
        ablation = results.get("ablation", [])
        null_ctrl = results.get("null_control", [])

        return {
            "track": "02 — AI Risk Manager",
            "held_out_test_set": {
                "rows": defend.get("rows"),
                "fraud": defend.get("fraud"),
                "fraud_rate": defend.get("fraud_rate"),
            },
            "headline_metrics": {
                "precision": defend.get("precision"),
                "recall": defend.get("recall"),
                "f1": defend.get("f1"),
                "roc_auc": defend.get("roc_auc"),
                "pr_auc": defend.get("pr_auc"),
                "false_positive_rate": defend.get("false_positive_rate"),
                "threshold": defend.get("threshold"),
            },
            "value_metrics": {
                "value_recall": defend.get("value_recall"),
                "value_at_risk_inr": defend.get("value_at_risk"),
                "value_saved_inr": defend.get("value_saved"),
            },
            "confidence_intervals": intervals,
            "ablation_grid": ablation,
            "null_control": null_ctrl,
            "provenance": results.get("provenance"),
        }

    return app


# --------------------------------------------------------------------------------------
# Background work
# --------------------------------------------------------------------------------------

def _run_pipeline_job(svc: ServiceState, job_id: str, quick: bool,
                      seed: Optional[int]) -> None:
    """Run generate-and-defend in a worker thread, emitting progress events."""
    from ..defend.pipeline import run_defence
    from ..generate.campaign import generate_dataset
    from ..identify.library import load_library
    from .bundle import from_artifacts

    job = svc.jobs.setdefault(job_id, {"status": "queued", "stage": None, "events": [],
                                       "started_at": time.time()})

    def emit(stage: str, **extra: Any) -> None:
        job["stage"] = stage
        job["events"].append({"stage": stage, "at": round(time.time() - job["started_at"], 2),
                              **extra})

    try:
        job["status"] = "running"
        root = Path(__file__).resolve().parents[3]
        cfg = Config.load(root / "configs" / ("quick.yaml" if quick else "default.yaml"))
        if seed is not None:
            cfg.seed = int(seed)
        cfg.run_name = job_id

        emit("identify")
        library = load_library()

        emit("generate")
        result = generate_dataset(cfg, library)
        emit("generated", rows=len(result.transactions),
             fraud=int(result.transactions["is_fraud"].sum()))

        emit("defend")
        artifacts = run_defence(result.transactions, result.corpus, cfg, verbose=False,
                                with_zero_day=False, with_importance=True,
                                transcripts=result.transcripts)
        head = artifacts.headline.to_dict()
        emit("defended", recall=head.get("recall"), fpr=head.get("false_positive_rate"))

        provenance = stamp(cfg)
        svc.store.record_run(provenance, head)
        svc.store.record_alerts(provenance.run_id, artifacts.scored)
        from_artifacts(artifacts, provenance).save(svc.bundle_path)
        svc.reload()

        emit("stored", run_id=provenance.run_id)
        job["status"] = "done"
        job["run_id"] = provenance.run_id
        job["headline"] = head
    except Exception as exc:  # noqa: BLE001 - a job must record its failure, not vanish
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["events"].append({"stage": "error", "detail": job["error"]})


def _run_attack(bundle: ServingBundle, request: AttackRequest) -> Dict[str, Any]:
    """Generate a small campaign for one vector, mutate it, and compare recall."""
    from ..generate.attacks.model_attacks import EVASION_MOVES
    from ..identify.library import load_library

    library = load_library()
    vector = next((v for v in library.vectors if v.id == request.vector_id), None)
    if vector is None:
        raise HTTPException(404, f"no attack vector {request.vector_id}")

    available = {name for name, _ in EVASION_MOVES}
    unknown = [lever for lever in request.levers if lever not in available]
    if unknown:
        raise HTTPException(422, f"unknown levers: {unknown}. Available: {sorted(available)}")

    cfg = Config.load(Path(__file__).resolve().parents[3] / "configs" / "quick.yaml")
    baseline, mutated = _campaign_pair(cfg, library, vector, request)
    if baseline is None:
        raise HTTPException(
            422, f"{request.vector_id} produced no rows at this size; try a larger `rows`")

    before = bundle.detector.predict_proba(baseline) >= bundle.threshold
    after = bundle.detector.predict_proba(mutated) >= bundle.threshold
    return {
        "vector_id": request.vector_id,
        "levers": request.levers or sorted({name for name, _ in EVASION_MOVES}),
        "rows": int(len(baseline)),
        "recall_before": round(float(before.mean()), 4),
        "recall_after": round(float(after.mean()), 4),
        "recall_lost": round(float(before.mean() - after.mean()), 4),
        "still_caught": int(after.sum()),
        "provenance": {"run_id": bundle.provenance.run_id},
        "note": ("Both figures are measured on the same rows, before and after mutation, so "
                 "the difference is attributable to the levers rather than to sampling. The "
                 "campaign is built against a freshly simulated benign backdrop and the two "
                 "content guards (prompt injection, transcripts) are not run here, so "
                 "absolute recall will differ from the batch report; the delta is the number "
                 "to read."),
    }


def _campaign_pair(cfg, library, vector, request: AttackRequest):
    """A single vector's campaign, before and after the mutation, feature-built both ways.

    Built against the same benign backdrop both times. The derived layer is history-dependent,
    so scoring the attack rows alone would measure a population that does not exist; the
    backdrop is what makes velocity and counterparty novelty mean anything.
    """
    from ..generate.attacks import build_context, generator_for
    from ..generate.attacks.model_attacks import apply_evasion
    from ..generate.benign import simulate_benign
    from ..generate.campaign import _merge
    from ..generate.entities import build_population

    rng = np.random.default_rng(cfg.seed)
    population = build_population(cfg, rng)
    benign = simulate_benign(cfg, population, rng)
    ctx = build_context(cfg, population, benign, library, rng)

    attacks = generator_for(vector).generate(ctx, vector, request.rows)
    if attacks is None or attacks.empty:
        return None, None

    mutated_rows = attacks.copy()
    apply_evasion(np.random.default_rng(cfg.seed + 1), mutated_rows,
                  share=1.0, strength=0.7, moves=request.levers or None)

    def built(frame: pd.DataFrame) -> pd.DataFrame:
        out = _featurise(_merge(benign, frame))
        return out[out["is_fraud"] == 1]

    return built(attacks), built(mutated_rows)


def _featurise(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the batch pipeline's feature stack for an ad-hoc campaign.

    The tabular, graph and deterministic-guard layers are all rebuilt, because each is a pure
    function of the rows. The two *content* guards are not: the injection and vishing
    classifiers score an agent corpus and a transcript bank that this endpoint does not
    generate, so their columns are left at their schema defaults. That understates the
    defence for the two agentic and social-engineering vectors that those guards catch, which
    is the conservative direction for a red-team endpoint to err in - it will not claim a
    lever defeated a control the caller never ran.
    """
    from ..defend.agent_controls import apply_controls
    from ..defend.intent_guard import apply_guard
    from ..features import build_features
    from ..features.graph import build_graph_features
    from ..generate.enrich import enrich

    out = build_graph_features(build_features(enrich(frame)))
    out = apply_controls(apply_guard(out))
    for column, default in (("agent_injection_score", 0.0), ("agent_injection_flag", 0),
                            ("vishing_score", 0.0), ("vishing_flag", 0),
                            ("media_artefact_score", 0.0), ("media_artefact_flag", 0)):
        if column not in out.columns:
            out[column] = default
    return out


app = create_app()


def get_app() -> FastAPI:  # pragma: no cover - uvicorn entry point
    return app
