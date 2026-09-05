"""Tests for the read-only artefact surface the browser client consumes.

Built against a synthetic run directory rather than whatever happens to be in
``artifacts/``, so the suite passes on a fresh clone and so the partial-run cases can be
constructed deliberately. Those are the interesting ones: a run missing half its tables is
the normal state of a ``--shallow`` or interrupted pipeline, and the contract this module
has to keep is that a missing file degrades one panel instead of a whole route.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from redteam.serve.artifacts import ArtifactStore, create_router  # noqa: E402


def _strict(response) -> dict:
    """Parse exactly as a browser would: no NaN, no Infinity, no leniency.

    ``response.json()`` uses Python's decoder, which accepts all three and would let an
    unparseable payload through the test and fail in the UI instead.
    """
    def reject(token):
        raise AssertionError(f"response contains the non-JSON constant {token!r}")

    return json.loads(response.content.decode(), parse_constant=reject)


@pytest.fixture
def run_dir(tmp_path):
    """A minimal but realistic run: some tables present, some deliberately absent."""
    artifacts = tmp_path / "artifacts" / "r1"
    data = tmp_path / "data" / "r1"
    artifacts.mkdir(parents=True)
    data.mkdir(parents=True)

    pd.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "amount": [100.0, 90_000.0, 5_000.0],
        "rail": ["UPI_P2P", "IMPS", "UPI_P2P"],
        "is_fraud": [0, 1, 0],
        "attack_vector_id": ["", "APP-PIG-BUTCHERING", ""],
        "is_hard_negative": [0, 0, 1],
        "score": [0.01, 0.97, 0.44],
        "model_alert": [0, 1, 0],
        "intent_block": [0, 0, 0],
        "control_block": [0, 0, 0],
        "injection_flag": [0, 0, 0],
        "decision": [0, 1, 0],
        "reason_codes": ["", "payee_is_first_time=1, amount=90000", ""],
    }).to_csv(artifacts / "defend_scored_test_set.csv", index=False)

    pd.DataFrame({
        "attack_vector_id": ["APP-PIG-BUTCHERING"], "family": ["app_scam"], "rows": [1],
        "recall": [1.0], "median_score": [0.97], "value": [90_000.0],
        "value_caught": [90_000.0], "value_recall": [1.0], "recall_lo95": [0.21],
        "recall_hi95": [1.0], "n_sufficient": [0],
    }).to_csv(artifacts / "defend_per_vector_recall.csv", index=False)

    pd.DataFrame({"metric": ["recall"], "value": [1.0]}).to_csv(
        artifacts / "defend_headline.csv", index=False)

    pd.DataFrame({
        "column": ["amount", "payee_account_age_days"], "n": [2, 2],
        "mean": [2_550.0, 400.0], "p05": [100.0, 10.0], "p25": [100.0, 10.0],
        "median": [2_550.0, 400.0], "p75": [5_000.0, 800.0], "p95": [5_000.0, 800.0],
        "share_zero": [0.0, 0.0],
    }).to_csv(artifacts / "defend_benign_baselines.csv", index=False)

    # The full row source the inspector reads, with a campaign two of the rows share.
    pd.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "customer_id": ["C1", "C2", "C3"],
        "campaign_id": ["", "CMP-1", "CMP-1"],
        "mule_ring_id": ["", "RING-1", ""],
        "amount": [100.0, 90_000.0, 5_000.0],
        "rail": ["UPI_P2P", "IMPS", "UPI_P2P"],
        "channel": ["mobile_app", "mobile_app", "web"],
        "auth_method": ["upi_pin", "otp_sms", "upi_pin"],
        "payee_account_age_days": [800.0, 2.0, 400.0],
        "payee_is_first_time": [0, 1, 0],
        "customer_age_band_ord": [1, 4, 2],
        "is_fraud": [0, 1, 0],
    }).to_parquet(data / "transactions.parquet", index=False)

    (artifacts / "run_summary.json").write_text(json.dumps({
        "provenance": {"run_id": "abc123", "run_name": "r1", "seed": 7,
                       "git_sha": "deadbeef", "config_hash": "cfg1",
                       "created_at": "2026-01-04T00:00:00Z", "environment": {}},
        "generate": {"transactions": 3, "fraud": 1, "fraud_rate": 0.333, "flags": [],
                     "warnings": ["a warning that must survive to the client"]},
        "defend": {"headline": {"recall": 1.0}},
    }), encoding="utf-8")

    return tmp_path


@pytest.fixture
def client(run_dir):
    app = FastAPI()
    app.include_router(create_router(
        ArtifactStore(run_dir / "artifacts", run_dir / "data")))
    return TestClient(app)


# --------------------------------------------------------------------------------------
# Discovery and provenance
# --------------------------------------------------------------------------------------

def test_runs_are_discovered_from_disk(client):
    runs = _strict(client.get("/api/runs"))
    assert [r["run"] for r in runs] == ["r1"]
    assert runs[0]["run_id"] == "abc123"
    assert runs[0]["has_loop"] is False


def test_every_response_carries_provenance(client):
    for endpoint in ("summary", "identify", "fidelity", "defend", "loop", "transactions"):
        body = _strict(client.get(f"/api/runs/r1/{endpoint}"))
        provenance = body["provenance"]
        assert provenance["run_id"] == "abc123"
        assert provenance["git_sha"] == "deadbeef"
        assert provenance["config_hash"] == "cfg1"
        assert provenance["generated_at"] == "2026-01-04T00:00:00Z"


def test_an_unknown_run_is_a_404_naming_what_exists(client):
    response = client.get("/api/runs/nope/defend")
    assert response.status_code == 404
    assert "r1" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# The contract that keeps a route alive when an artefact is missing
# --------------------------------------------------------------------------------------

def test_a_missing_table_is_a_value_not_an_error(client):
    """This run has no loop, no fairness and no cost curve. Every route must still load."""
    body = _strict(client.get("/api/runs/r1/defend"))
    assert body["per_vector"]["available"] is True
    for absent in ("fairness", "cost_curve", "baselines", "zero_day"):
        assert body[absent]["available"] is False
        assert body[absent]["rows"] == []
        # Named, so the empty state can say which file it wanted.
        assert body[absent]["source"].endswith(".csv")

    loop = _strict(client.get("/api/runs/r1/loop"))
    assert loop["rounds"]["available"] is False
    assert loop["discovered"]["available"] is False


def test_every_table_names_its_source_file(client):
    body = _strict(client.get("/api/runs/r1/defend"))
    tables = [v for v in body.values() if isinstance(v, dict) and "source" in v]
    assert len(tables) > 15
    assert all(t["source"] for t in tables)


# --------------------------------------------------------------------------------------
# Strict JSON, which is the whole reason the sanitiser exists
# --------------------------------------------------------------------------------------

def test_missing_numbers_serialise_as_null_not_nan(client, run_dir):
    """A single NaN makes the whole response unparseable in a browser."""
    pd.DataFrame({
        "layer": ["raw", "+ tabular"], "features": [10, 20],
        "recall": [0.5, 0.6], "delta_recall": [float("nan"), 0.1],
    }).to_csv(run_dir / "artifacts" / "r1" / "defend_ablation_grid.csv", index=False)

    body = _strict(client.get("/api/runs/r1/defend"))
    rows = body["ablation"]["rows"]
    assert rows[0]["delta_recall"] is None
    assert rows[1]["delta_recall"] == pytest.approx(0.1)


# --------------------------------------------------------------------------------------
# Identify: the join and the honest gap
# --------------------------------------------------------------------------------------

def test_vectors_carry_kill_chain_which_the_csv_export_drops(client):
    body = _strict(client.get("/api/runs/r1/identify"))
    assert body["counts"]["total"] == 67
    assert body["counts"]["simulated"] + body["counts"]["documented_only"] == 67
    assert len(body["kill_chain_stages"]) == 8
    assert all(v["kill_chain"] for v in body["vectors"])
    assert all(v["description"] for v in body["vectors"])


def test_measured_recall_is_joined_onto_the_taxonomy(client):
    body = _strict(client.get("/api/runs/r1/identify"))
    by_id = {v["id"]: v for v in body["vectors"]}
    assert by_id["APP-PIG-BUTCHERING"]["measured_recall"] == 1.0
    assert by_id["APP-PIG-BUTCHERING"]["measured_rows"] == 1
    # Everything else in this run was never measured, and must read as unmeasured rather
    # than as zero recall.
    unmeasured = [v for v in body["vectors"] if v["id"] != "APP-PIG-BUTCHERING"]
    assert all(v["measured_recall"] is None for v in unmeasured)


def test_residual_risk_distinguishes_undetected_from_unmeasured(client):
    """The distinction the table exists to make. A vector nobody simulated is not a vector
    the defence was shown to handle, and scoring it as zero residual risk would say it was."""
    body = _strict(client.get("/api/runs/r1/identify"))
    residual = body["residual_risk"]
    assert residual == sorted(residual, key=lambda r: -r["residual_risk"])

    measured = next(r for r in residual if r["id"] == "APP-PIG-BUTCHERING")
    assert measured["residual_risk"] == 0.0
    assert "measured on 1 test rows" in measured["basis"]

    documented = next(r for r in residual if not r["simulated"])
    assert documented["measured_recall"] is None
    assert documented["residual_risk"] == documented["risk_score"]
    assert "documented only" in documented["basis"]


def test_fidelity_warnings_reach_the_client(client):
    """The zero-without-a-flag state is the most misleading one the report can be in, so it
    must not be dropped on the way to the screen."""
    body = _strict(client.get("/api/runs/r1/fidelity"))
    assert body["warnings"] == ["a warning that must survive to the client"]


# --------------------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------------------

def test_transactions_filter_and_paginate(client):
    body = _strict(client.get("/api/runs/r1/transactions?decision=fraud"))
    assert body["total"] == 1
    assert body["rows"][0]["txn_id"] == "T2"
    assert body["next_cursor"] is None

    page = _strict(client.get("/api/runs/r1/transactions?limit=2"))
    assert len(page["rows"]) == 2
    assert page["next_cursor"] == 2

    assert _strict(client.get("/api/runs/r1/transactions?hard_negative=true"))["total"] == 1
    assert _strict(client.get("/api/runs/r1/transactions?min_amount=50000"))["total"] == 1
    assert _strict(client.get("/api/runs/r1/transactions?rail=UPI_P2P"))["total"] == 2


def test_a_transaction_is_grouped_by_schema_block(client):
    body = _strict(client.get("/api/runs/r1/transactions/T2"))
    assert body["verdict"]["score"] == pytest.approx(0.97)
    assert body["reason_codes"] == ["payee_is_first_time=1", "amount=90000"]

    blocks = body["blocks"]
    assert "instruction" in blocks and "counterparty" in blocks
    payee = {f["column"]: f for f in blocks["counterparty"]}
    assert payee["payee_account_age_days"]["value"] == 2.0
    # The legitimate baseline travels with the value so the deviation reads at a glance.
    assert payee["payee_account_age_days"]["benign_median"] == 400.0


def test_case_siblings_group_the_campaign_and_the_ring(client):
    body = _strict(client.get("/api/runs/r1/transactions/T2"))
    siblings = body["siblings"]
    assert siblings["campaign_id"] == "CMP-1"
    assert [s["txn_id"] for s in siblings["campaign"]] == ["T3"]
    # T3 shares the campaign but not the ring.
    assert siblings["ring"] == []


def test_an_unknown_transaction_is_a_404(client):
    assert client.get("/api/runs/r1/transactions/NOPE").status_code == 404


def test_a_sampled_alert_with_no_counterfactual_is_distinguishable_from_an_unsampled_one(
        client, run_dir):
    """Three states, and the interface says something different about each.

    "We never computed these", "this alert was outside the sample" and "we looked and no
    single change would have flipped it" are different answers. The last one is the most
    interesting and the easiest to lose, because it looks like absence.
    """
    absent = _strict(client.get("/api/runs/r1/transactions/T2"))["counterfactuals"]
    assert absent["available"] is False

    pd.DataFrame([
        # T2 was sampled and nothing flipped it: recorded as a row with no column.
        {"txn_id": "T2", "column": None, "from": None, "to": None,
         "score_after": None, "normalised_move": None, "phrase": None},
        {"txn_id": "T3", "column": "amount", "from": 90000.0, "to": 12000.0,
         "score_after": 0.4, "normalised_move": 0.8, "phrase": "Approved if amount INR 12,000 or less."},
    ]).to_csv(run_dir / "artifacts" / "r1" / "defend_counterfactuals.csv", index=False)

    looked = _strict(client.get("/api/runs/r1/transactions/T2"))["counterfactuals"]
    assert looked["available"] is True
    assert looked["sampled"] is True
    assert looked["rows"] == []

    found = _strict(client.get("/api/runs/r1/transactions/T3"))["counterfactuals"]
    assert found["sampled"] is True
    assert found["rows"][0]["column"] == "amount"


# --------------------------------------------------------------------------------------
# Artefact download
# --------------------------------------------------------------------------------------

def test_an_artefact_can_be_fetched_byte_for_byte(client, run_dir):
    """Source labels in the interface are links. A citation nobody can open is a request to
    take the number on trust, which is the opposite of what naming the file is for."""
    response = client.get("/api/runs/r1/file/defend_headline.csv")
    assert response.status_code == 200
    assert response.text == (run_dir / "artifacts" / "r1" / "defend_headline.csv").read_text()
    assert response.headers["content-type"].startswith("text/csv")


def test_the_file_endpoint_cannot_escape_the_run_directory(client, run_dir):
    """Nothing here is behind auth, so a traversal would expose the host filesystem."""
    secret = run_dir / "artifacts" / "secret.txt"
    secret.write_text("not yours")

    for attempt in ("../secret.txt", "..%2Fsecret.txt", "/etc/passwd", "..", "."):
        response = client.get(f"/api/runs/r1/file/{attempt}")
        assert response.status_code in (404, 307), attempt
        assert "not yours" not in response.text
