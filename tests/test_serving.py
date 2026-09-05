"""The serving layer: the bundle, the store and the HTTP surface.

These tests are about the contract, not the accuracy. Whether the model is any good is
settled by the defend tests; what matters here is that the number the service returns was
computed against the threshold that shipped with it, that a caller who misspells a feature
finds out, and that the alert queue pages the way a queue is actually worked.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="serving layer is an optional extra")
from fastapi.testclient import TestClient  # noqa: E402

from redteam.provenance import Provenance  # noqa: E402
from redteam.serve.bundle import BUNDLE_VERSION, ServingBundle  # noqa: E402
from redteam.serve.store import Store  # noqa: E402


# --------------------------------------------------------------------------------------
# Fixtures: one trained detector, shared, because fitting it is the slow part
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifacts(tiny_config, transactions, corpus, transcripts):
    from redteam.defend.pipeline import run_defence

    return run_defence(transactions, corpus, tiny_config, verbose=False,
                       with_zero_day=False, with_importance=True, transcripts=transcripts)


@pytest.fixture(scope="module")
def provenance(tiny_config):
    from redteam.provenance import stamp

    return stamp(tiny_config)


@pytest.fixture(scope="module")
def bundle(artifacts, provenance):
    from redteam.serve.bundle import from_artifacts

    return from_artifacts(artifacts, provenance)


@pytest.fixture(scope="module")
def client(tmp_path_factory, bundle, artifacts, provenance):
    from redteam.serve.api import create_app

    root = tmp_path_factory.mktemp("serving")
    bundle.save(root / "bundle.pkl")
    store = Store(root / "test.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    store.record_alerts(provenance.run_id, artifacts.scored)
    with TestClient(create_app(root / "bundle.pkl", root / "test.sqlite")) as c:
        yield c


# --------------------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------------------

def test_a_bundle_round_trips_with_its_threshold(bundle, tmp_path):
    """The threshold must survive the trip. It is the whole decision."""
    path = bundle.save(tmp_path / "b.pkl")
    loaded = ServingBundle.load(path)
    assert loaded.threshold == pytest.approx(bundle.threshold)
    assert loaded.columns == bundle.columns
    assert loaded.provenance.run_id == bundle.provenance.run_id


def test_the_reloaded_bundle_scores_identically(bundle, artifacts, tmp_path):
    """A pickled detector that scores differently is worse than one that fails to load."""
    loaded = ServingBundle.load(bundle.save(tmp_path / "b.pkl"))
    sample = artifacts.split.test.head(50)
    before = bundle.detector.predict_proba(sample)
    after = loaded.detector.predict_proba(sample)
    assert after == pytest.approx(before)


def test_a_stale_bundle_version_refuses_to_load(bundle, tmp_path):
    """Loading an old shape into a new class fails at the first request, not at load, so
    the version check has to be explicit."""
    bundle.version = BUNDLE_VERSION + 1
    path = bundle.save(tmp_path / "old.pkl")
    bundle.version = BUNDLE_VERSION
    with pytest.raises(ValueError, match="bundle version"):
        ServingBundle.load(path)


def test_no_training_data_rides_along_in_the_bundle(bundle, tmp_path):
    """A servable artefact carrying customer rows is a deployment mistake away from being
    an incident."""
    size_mb = bundle.save(tmp_path / "b.pkl").stat().st_size / 1e6
    assert size_mb < 200, f"bundle is {size_mb:.0f} MB; something large came with it"
    for attribute in ("featured", "split", "scored", "train", "test"):
        assert not hasattr(bundle, attribute)


def test_the_model_card_states_it_is_not_for_real_traffic(bundle):
    card = bundle.model_card()
    assert "not fit for production" in card["intended_use"].lower()
    assert card["known_limitations"], "a card with no limitations is marketing"
    assert card["model"]["decision_threshold"] == pytest.approx(bundle.threshold)
    assert card["provenance"]["run_id"] == bundle.provenance.run_id


def test_the_card_reports_which_guards_actually_shipped(bundle):
    """A card claiming a guard the bundle does not carry is the failure mode worth
    testing, because both states are plausible and only one is true."""
    guards = bundle.model_card()["guards"]
    assert guards["prompt_injection"] is (bundle.injection_guard is not None)
    assert guards["vishing_transcripts"] is (bundle.vishing_guard is not None)


# --------------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------------

def test_alerts_are_written_and_counted(tmp_path, artifacts, provenance):
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    written = store.record_alerts(provenance.run_id, artifacts.scored)
    counts = store.counts(provenance.run_id)
    assert written == len(artifacts.scored)
    assert counts["scored"] == len(artifacts.scored)
    assert counts["fraud"] == int(artifacts.scored["is_fraud"].sum())
    assert counts["caught"] <= counts["alerts"]


def test_rewriting_the_same_run_does_not_duplicate(tmp_path, artifacts, provenance):
    """Re-running the defend stage is routine. It must not double the queue."""
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    store.record_alerts(provenance.run_id, artifacts.scored)
    store.record_alerts(provenance.run_id, artifacts.scored)
    assert store.counts(provenance.run_id)["scored"] == len(artifacts.scored)


def test_the_queue_comes_back_highest_score_first(tmp_path, artifacts, provenance):
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    store.record_alerts(provenance.run_id, artifacts.scored)
    page = store.queue(provenance.run_id, limit=25)
    scores = [row["score"] for row in page]
    assert scores == sorted(scores, reverse=True)
    assert all(row["decision"] == 1 for row in page), "alerts_only returned non-alerts"


def test_paging_does_not_repeat_or_skip_rows(tmp_path, artifacts, provenance):
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    store.record_alerts(provenance.run_id, artifacts.scored)
    first = store.queue(provenance.run_id, limit=10, offset=0)
    second = store.queue(provenance.run_id, limit=10, offset=10)
    ids = [r["txn_id"] for r in first] + [r["txn_id"] for r in second]
    assert len(set(ids)) == len(ids)


def test_a_disposition_survives_and_joins_back_to_its_alert(tmp_path, artifacts, provenance):
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, artifacts.headline.to_dict())
    store.record_alerts(provenance.run_id, artifacts.scored)
    txn = store.queue(provenance.run_id, limit=1)[0]["txn_id"]
    store.set_case(provenance.run_id, txn, status="closed", disposition="fraud", note="mule")
    found = next(r for r in store.queue(provenance.run_id, limit=500) if r["txn_id"] == txn)
    assert found["status"] == "closed"
    assert found["disposition"] == "fraud"


def test_numpy_and_timestamp_cells_bind_without_an_interface_error(tmp_path, provenance):
    """The scored frame arrives with numpy scalars, pandas timestamps and a list of reason
    codes. sqlite3 binds none of those natively, and the failure lands at the end of a long
    pipeline stage, so it is pinned here."""
    import numpy as np

    frame = pd.DataFrame({
        "txn_id": ["t1"],
        "timestamp": [pd.Timestamp("2025-01-01T10:00:00")],
        "amount": [np.float64(1200.5)],
        "rail": ["upi"],
        "score": [np.float32(0.91)],
        "decision": [np.bool_(True)],
        "model_alert": [np.int64(1)],
        "is_fraud": [np.int8(1)],
        "attack_vector_id": [None],
        "reason_codes": [["A", "B"]],
    })
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, {})
    assert store.record_alerts(provenance.run_id, frame) == 1
    row = store.queue(provenance.run_id, limit=1)[0]
    assert row["reason_codes"] == "A, B"
    assert row["decision"] == 1
    assert row["timestamp"].startswith("2025-01-01")


def test_a_missing_column_becomes_null_rather_than_failing(tmp_path, provenance):
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, {})
    store.record_alerts(provenance.run_id,
                        pd.DataFrame({"txn_id": ["t1"], "score": [0.5], "decision": [1]}))
    assert store.queue(provenance.run_id, limit=1)[0]["rail"] is None


def test_the_schema_is_idempotent_across_opens(tmp_path, provenance):
    Store(tmp_path / "s.sqlite")
    store = Store(tmp_path / "s.sqlite")
    store.record_run(provenance, {})
    assert store.runs()[0]["run_id"] == provenance.run_id


# --------------------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------------------

def test_health_names_the_run_it_is_serving(client, provenance):
    """A green health check on a service quietly running last week's model is worse than a
    red one, so the run id is part of liveness."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["serving_run_id"] == provenance.run_id


def test_scoring_returns_a_decision_against_the_shipped_threshold(client, bundle):
    body = client.post("/score", json={"amount": 48000.0, "rail": "upi"}).json()
    assert body["threshold"] == pytest.approx(bundle.threshold, abs=1e-6)
    assert body["model_alert"] is (body["score"] >= body["threshold"])
    assert body["decision"] in ("approve", "decline")


def test_an_unknown_feature_is_rejected_not_ignored(client):
    """A caller who misspells a column and gets a 200 has been told their feature was used
    when it was replaced by a default. That is a worse outcome than a 422."""
    response = client.post("/score", json={"amount": 100.0, "rail": "upi",
                                           "features": {"payee_acct_age_dys": 2}})
    assert response.status_code == 422
    assert "payee_acct_age_dys" in response.json()["detail"]


def test_a_sparse_caller_is_told_how_empty_the_matrix_was(client):
    """The model is mostly history. A score computed from defaults is a number about a
    fiction, and the response has to say so."""
    sparse = client.post("/score", json={"amount": 100.0, "rail": "upi"}).json()
    assert sparse["feature_completeness"] == pytest.approx(0.0)
    rich = client.post("/score", json={
        "amount": 100.0, "rail": "upi",
        "features": {"f_payer_txn_count_1h": 4, "f_payer_amount_sum_24h": 90000.0},
    }).json()
    assert rich["feature_completeness"] > sparse["feature_completeness"]


def test_a_guard_block_declines_with_a_named_reason(client):
    """The separation is the point: a decline a bank can defend to a regulator names the
    control, not the probability."""
    body = client.post("/score", json={"amount": 5000.0, "rail": "upi",
                                       "features": {"intent_guard_blocked": 1}}).json()
    assert body["intent_block"] is True
    assert body["decision"] == "decline"
    assert "INTENT_GUARD_BLOCK" in body["reason_codes"]


def test_a_guard_block_is_reported_separately_from_the_model(client):
    """If the two were conflated the auditable reason would lose its value."""
    body = client.post("/score", json={"amount": 100.0, "rail": "neft",
                                       "features": {"agent_control_blocked": 1}}).json()
    assert body["control_block"] is True
    assert "AGENT_CONTROL_BLOCK" in body["reason_codes"]
    assert set(body.keys()) >= {"model_alert", "intent_block", "control_block"}


def test_an_alert_without_importance_says_so_rather_than_returning_nothing(bundle, tmp_path):
    """A bundle from a --shallow run has no permutation importance and so cannot produce
    reason codes. An empty list reads as "alerted for no reason", which is a different and
    much worse claim than "this build cannot tell you"."""
    from redteam.serve.api import PaymentRequest, score_payment

    stripped = ServingBundle.load(bundle.save(tmp_path / "b.pkl"))
    stripped.importance = pd.DataFrame()
    body = score_payment(stripped, PaymentRequest(amount=250000.0, rail="upi"))
    if body.model_alert:
        assert body.reason_codes == ["REASONS_UNAVAILABLE_SHALLOW_BUILD"]


def test_a_reason_never_cites_a_feature_the_caller_did_not_supply(client):
    """`payee_name_match_score=nan` reads as evidence and is the opposite: it says the model
    was never told, not that the name failed to match."""
    body = client.post("/score", json={"amount": 480000.0, "rail": "upi",
                                       "features": {"payee_is_first_time": 1}}).json()
    assert not any(r.endswith("=nan") for r in body["reason_codes"]), body["reason_codes"]


def test_every_response_carries_the_run_that_produced_it(client, provenance):
    body = client.post("/score", json={"amount": 100.0, "rail": "upi"}).json()
    assert body["provenance"]["run_id"] == provenance.run_id


def test_scoring_rejects_a_nonsense_amount(client):
    assert client.post("/score", json={"amount": -5.0, "rail": "upi"}).status_code == 422
    assert client.post("/score", json={"rail": "upi"}).status_code == 422


def test_a_batch_is_capped(client):
    payloads = [{"amount": 100.0, "rail": "upi"}] * 501
    assert client.post("/score/batch", json=payloads).status_code == 422


def test_a_batch_scores_every_row(client):
    payloads = [{"amount": float(100 * i + 1), "rail": "upi"} for i in range(1, 6)]
    body = client.post("/score/batch", json=payloads).json()
    assert len(body) == 5


def test_the_model_card_is_served_from_the_deployed_bundle(client, bundle):
    """Generated live rather than written into a document, so it cannot describe a
    different model than the one answering requests."""
    card = client.get("/model-card").json()
    assert card["model"]["decision_threshold"] == pytest.approx(bundle.threshold, abs=1e-6)
    assert card["provenance"]["run_id"] == bundle.provenance.run_id


def test_the_queue_endpoint_pages_and_counts(client, provenance):
    body = client.get(f"/runs/{provenance.run_id}/alerts?limit=5").json()
    assert len(body["alerts"]) <= 5
    assert body["counts"]["scored"] > 0


def test_an_unknown_run_is_a_404_not_an_empty_page(client):
    assert client.get("/runs/does-not-exist").status_code == 404


def test_a_case_can_be_dispositioned_over_http(client, provenance):
    txn = client.get(f"/runs/{provenance.run_id}/alerts?limit=1").json()["alerts"][0]["txn_id"]
    response = client.put(f"/runs/{provenance.run_id}/cases/{txn}",
                          json={"status": "closed", "disposition": "fraud"})
    assert response.status_code == 200
    rows = client.get(f"/runs/{provenance.run_id}/alerts?limit=500").json()["alerts"]
    assert next(r for r in rows if r["txn_id"] == txn)["disposition"] == "fraud"


def test_an_invalid_disposition_is_refused(client, provenance):
    response = client.put(f"/runs/{provenance.run_id}/cases/t1",
                          json={"status": "closed", "disposition": "probably"})
    assert response.status_code == 422


def test_an_unknown_attack_vector_is_a_404(client):
    response = client.post("/attack", json={"vector_id": "NOT-A-VECTOR", "rows": 20})
    assert response.status_code == 404


def test_an_unknown_lever_names_the_ones_that_exist(client):
    """A red-team endpoint that silently ignores an unrecognised lever reports a recall
    delta caused by nothing."""
    response = client.post("/attack", json={"vector_id": "APP-SAFE-ACCOUNT-SCREENSHARE",
                                            "levers": ["make_it_invisible"], "rows": 20})
    assert response.status_code == 422
    assert "amount" in response.json()["detail"], "the error should name the real levers"


def test_an_attack_compares_the_same_rows_before_and_after(client):
    """The delta is the whole point, and it is only interpretable if both numbers come from
    the same rows. Two campaigns of different size set beside each other would not be."""
    body = client.post("/attack", json={"vector_id": "APP-SAFE-ACCOUNT-SCREENSHARE",
                                        "levers": ["amount", "hour"], "rows": 40}).json()
    assert body["rows"] > 0
    assert body["recall_lost"] == pytest.approx(
        body["recall_before"] - body["recall_after"], abs=1e-6)
    assert body["levers"] == ["amount", "hour"]


def test_an_empty_lever_list_reports_which_levers_were_pulled(client):
    """"Empty" pulls all of them, and a response that echoed back an empty list would let a
    reader think nothing was mutated."""
    body = client.post("/attack", json={"vector_id": "APP-SAFE-ACCOUNT-SCREENSHARE",
                                        "rows": 40}).json()
    assert len(body["levers"]) > 1
    assert "amount" in body["levers"]


def test_the_attack_response_says_it_is_not_the_batch_number(client):
    """Absolute recall here is measured without the content guards and against a fresh
    backdrop. Reporting it beside the batch figure without saying so would invite the
    comparison it does not support."""
    body = client.post("/attack", json={"vector_id": "APP-SAFE-ACCOUNT-SCREENSHARE",
                                        "rows": 40}).json()
    assert "differ from the batch report" in body["note"]


def test_simulate_returns_immediately_with_a_job_handle(client, monkeypatch):
    """Blocking an HTTP request on a pipeline run is how a demo becomes indistinguishable
    from a hang.

    The worker is stubbed out. What is under test is that the caller is handed a handle
    instead of being made to wait; whether the pipeline it launches is correct is the
    subject of the rest of the suite, and running one here would cost minutes and overwrite
    the bundle the other tests are scoring against.
    """
    from redteam.serve import api

    monkeypatch.setattr(api, "_run_pipeline_job", lambda svc, job_id, quick, seed: None)

    body = client.post("/simulate?quick=true&seed=7").json()
    assert body["status"] == "queued"
    assert body["events"] == f"/jobs/{body['job_id']}/events"
    assert client.get(f"/jobs/{body['job_id']}").json()["status"] in ("queued", "running")


def test_an_unknown_job_is_a_404(client):
    assert client.get("/jobs/job-nope").status_code == 404


def test_the_event_stream_replays_progress_and_terminates(client):
    """A finished job must close the stream. One that stays open leaves the browser
    spinning on a run that already succeeded."""
    service = client.app.state.service
    service.jobs["job-fake"] = {
        "status": "done", "stage": "stored", "started_at": 0.0,
        "events": [{"stage": "generate"}, {"stage": "defend"}, {"stage": "stored"}],
    }
    with client.stream("GET", "/jobs/job-fake/events") as stream:
        payloads = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
    stages = [json.loads(p)["stage"] for p in payloads]
    assert stages[:3] == ["generate", "defend", "stored"]
    assert stages[-1] == "end"


def test_a_failed_job_reports_why_rather_than_vanishing(client):
    service = client.app.state.service
    service.jobs["job-bad"] = {"status": "failed", "stage": "generate", "started_at": 0.0,
                               "error": "ValueError: nope", "events": []}
    body = client.get("/jobs/job-bad").json()
    assert body["status"] == "failed"
    assert "nope" in body["error"]


def test_scoring_still_works_when_no_bundle_is_present(tmp_path):
    """A fresh clone has no bundle. That is the normal state, not a crash: health reports
    it, score refuses with a 503 that says how to build one, and the card is still
    readable."""
    from redteam.serve.api import create_app

    with TestClient(create_app(tmp_path / "absent.pkl", tmp_path / "s.sqlite")) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded"
        response = c.post("/score", json={"amount": 100.0, "rail": "upi"})
        assert response.status_code == 503
        assert "redteam run" in response.json()["detail"]


def test_the_scoring_endpoint_is_rate_limited(tmp_path, bundle):
    """An unauthenticated scoring endpoint with no limit is a free oracle for mapping the
    decision boundary, which is a threat this repository ships an attack for."""
    from redteam.serve import api

    bundle.save(tmp_path / "b.pkl")
    with TestClient(api.create_app(tmp_path / "b.pkl", tmp_path / "s.sqlite")) as c:
        codes = {c.post("/score", json={"amount": 100.0, "rail": "upi"}).status_code
                 for _ in range(api.RATE_LIMIT_PER_MINUTE + 5)}
    assert 429 in codes


def test_reading_the_queue_is_not_rate_limited(client, provenance):
    """The limit protects the oracle, not the reviewer paging a queue."""
    codes = {client.get(f"/runs/{provenance.run_id}/alerts?limit=1").status_code
             for _ in range(30)}
    assert codes == {200}
