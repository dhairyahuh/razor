"""One full pass through all three pillars, plus the CLI that drives them.

Slow by nature - it trains the real ensemble - so it is a single test rather than many,
and it asserts the things that would make the whole run untrustworthy rather than
re-checking component behaviour covered elsewhere.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from redteam.cli import main
from redteam.defend.dataset import input_columns
from redteam.defend.pipeline import run_defence
from redteam.schema import LABEL_COLUMNS, META_COLUMNS


@pytest.fixture(scope="module")
def artifacts(transactions, corpus, tiny_config):
    return run_defence(transactions, corpus, tiny_config, verbose=False,
                       with_zero_day=False, with_importance=False)


def test_detector_beats_chance_without_reading_the_answer(artifacts):
    head = artifacts.headline
    assert head.roc_auc > 0.85, head.to_dict()
    assert head.recall > 0.3, head.to_dict()
    # The operating point is a budget, so the realised rate has to respect it.
    assert head.false_positive_rate <= 0.01, head.to_dict()

    trained_on = set(artifacts.detector.columns)
    assert trained_on.isdisjoint(LABEL_COLUMNS)
    assert trained_on.isdisjoint(META_COLUMNS)
    assert trained_on == set(input_columns(artifacts.split.train))


def test_scores_are_probabilities_and_reproducible(artifacts, tiny_config):
    scores = artifacts.detector.predict_proba(artifacts.split.test)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    again = artifacts.detector.predict_proba(artifacts.split.test)
    np.testing.assert_allclose(scores, again)


def test_evaluation_tables_are_populated(artifacts):
    assert not artifacts.per_vector.empty
    assert not artifacts.per_family.empty
    assert not artifacts.operating_curve.empty
    assert not artifacts.false_positives.empty

    # Recall on hard negatives is reported separately because that is where customer
    # harm lands; both segments must be present or the number is an average of nothing.
    segments = set(artifacts.false_positives["segment"])
    assert {"ordinary_legitimate", "hard_negative"} <= segments


def test_operating_curve_trades_recall_against_alerts(artifacts):
    curve = artifacts.operating_curve.sort_values("target_fpr")
    assert curve["recall"].is_monotonic_increasing
    assert curve["alerts_per_10k"].is_monotonic_increasing
    assert curve["threshold"].is_monotonic_decreasing


def test_hard_negatives_are_harder_than_ordinary_traffic(artifacts):
    """If they were not, they would not be doing their job as false-positive traps."""
    rates = artifacts.false_positives.set_index("segment")["fp_rate"]
    assert rates["hard_negative"] > rates["ordinary_legitimate"]


def test_the_pipeline_reports_chance_on_shuffled_labels(transactions, corpus, tiny_config):
    """The leakage regression guard, and the only one that cannot be argued with.

    Every other check reasons about whether a particular feature is fair. This one destroys
    the relationship between features and label entirely and demands the pipeline notice. If
    feature construction, the temporal split or the threshold calibration ever begins to
    condition on the label - the classic ways a synthetic-data result becomes worthless - this
    fails, whatever the leak's mechanism and whether or not anyone thought to look for it.
    """
    from redteam.defend.evaluate import label_shuffle_control
    from redteam.defend.model import FraudDetector
    from redteam.defend.pipeline import prepare

    _, split, _, _ = prepare(transactions, corpus, tiny_config, verbose=False)
    control = label_shuffle_control(
        split,
        lambda **kw: FraudDetector(seed=tiny_config.seed,
                                   target_fpr=tiny_config.defence.target_fpr,
                                   max_iter=60, n_estimators=40, **kw),
        target_fpr=tiny_config.defence.target_fpr,
        seed=tiny_config.seed,
    )
    assert not control.empty
    row = control.iloc[0]

    # Chance, within the noise of a deliberately small test window. ROC AUC is the tightest
    # of the three because it does not depend on the estimated threshold.
    assert abs(row["roc_auc"] - 0.5) < 0.12, row.to_dict()
    assert row["recall"] < 0.10, row.to_dict()
    assert row["pr_auc"] < 5 * row["expected_pr_auc"] + 0.01, row.to_dict()


def test_ablation_grid_is_monotone_and_credits_the_raw_schema(artifacts, transactions,
                                                              corpus, tiny_config):
    """The derived layers must help, and the raw payment message must not be useless.

    Two failure modes, opposite and both fatal to the submission's credibility. If the raw
    schema alone scored as well as the full matrix, every derived feature here would be
    decoration. If it scored at chance while the full matrix scored highly, the result would
    rest entirely on engineered columns and the leak objection would be unanswerable.
    """
    from redteam.defend.evaluate import ablation_grid
    from redteam.defend.model import FraudDetector

    grid = ablation_grid(
        artifacts.split,
        lambda **kw: FraudDetector(seed=tiny_config.seed,
                                   target_fpr=tiny_config.defence.target_fpr,
                                   max_iter=60, n_estimators=40, **kw),
        target_fpr=tiny_config.defence.target_fpr,
    )
    assert not grid.empty
    assert list(grid["layer"])[0] == "raw_schema"
    # Feature counts must grow, confirming the rows really are cumulative.
    assert grid["features"].is_monotonic_increasing

    raw = float(grid.iloc[0]["recall"])
    full = float(grid.iloc[-1]["recall"])
    assert raw > 0.15, f"the raw payment schema should carry real signal: {grid.to_dict()}"
    assert full >= raw, f"derived features should not hurt: {grid.to_dict()}"


def test_decision_is_the_union_of_model_and_deterministic_blocks(artifacts):
    scored = artifacts.scored
    expected = (
        (scored["model_alert"] == 1)
        | (scored["intent_block"] == 1)
        | (scored["control_block"] == 1)
    ).astype(int)
    pd.testing.assert_series_equal(scored["decision"], expected, check_names=False)


@pytest.mark.slow
def test_cli_run_produces_a_report_and_artifacts(tmp_path, monkeypatch):
    config = tmp_path / "cli.yaml"
    config.write_text(
        "seed: 5\n"
        "run_name: clitest\n"
        f"data_dir: {tmp_path / 'data'}\n"
        f"artifacts_dir: {tmp_path / 'artifacts'}\n"
        "population: {n_customers: 500, n_merchants: 80, n_psps: 6}\n"
        "benign: {n_days: 12, base_txns_per_customer_per_day: 1.0}\n"
        "attacks: {target_fraud_rate: 0.012}\n"
        "defence: {test_days: 4, calibration_days: 2, max_iter: 60,\n"
        "          forest_n_estimators: 30, zero_day_vectors: 1}\n"
        "loop: {rounds: 1, population: 2, survivors: 1, min_fraud_rows: 6}\n",
        encoding="utf-8",
    )
    assert main(["run", "--config", str(config), "--shallow"]) == 0

    root = tmp_path / "artifacts" / "clitest"
    assert (root / "REPORT.md").exists()
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "## Identify" in report and "## Generate" in report and "## Defend" in report

    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 5
    assert summary["identify"]["total_vectors"] > 60

    for name in ("identify_vectors.csv", "generate_per_vector.csv",
                 "defend_per_vector_recall.csv", "defend_headline.csv"):
        assert (root / name).exists(), name

    # The dataset is written so `defend` and `loop` can be re-run without regenerating.
    assert (tmp_path / "data" / "clitest" / "transactions.parquet").exists()
    assert (tmp_path / "data" / "clitest" / "agent_corpus.parquet").exists()

    # results.json is the single source of truth every consumer reads, so its shape is
    # asserted rather than left to whoever writes the next thing that depends on it.
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert results["schema_version"] >= 1
    assert results["run"]["seed"] == 5
    assert results["identify"]["vectors"] > 60
    assert results["generate"]["transactions"] > 0
    assert 0.0 <= results["defend"]["recall"] <= 1.0
    # Both separability probes must be machine-readable, since the raw-versus-derived gap is
    # the central fidelity claim and a reader should not have to parse prose for it.
    assert results["generate"]["raw_separability_recall"] is not None
    assert results["generate"]["derived_separability_recall"] is not None


def test_cli_rejects_an_unknown_stage():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_cli_rejects_an_unknown_sweep_parameter():
    """Sweep names are an explicit allowlist, so a typo fails loudly rather than silently.

    Resolving a dotted path onto the config would accept `attacks.signl_emission`, set an
    attribute nobody reads, and produce a flat curve that looks like a finding.
    """
    with pytest.raises(SystemExit) as exc:
        main(["sweep", "--sweep", "signl_emission=0.5,0.9"])
    assert "signal_emission" in str(exc.value), "the error should list the valid names"

    with pytest.raises(SystemExit):
        main(["sweep", "--sweep", "no_equals_sign"])
