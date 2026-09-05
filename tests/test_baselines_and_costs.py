"""Tests for the three panels that make a detection number defensible.

The comparison table, the fairness breakdown and the cost curve all share a failure mode:
each can produce a number that is arithmetically correct and rhetorically dishonest. A
baseline that alerts on nothing scores zero recall and flatters the model. A fairness table
computed only on cohorts with plenty of data hides the cohort with three rows. A net-benefit
figure quoted without its price list is a sales number. The tests here are aimed at those
failures rather than at the arithmetic, which is the easy part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redteam.defend.baselines import (
    BUDGET_FLOOR,
    BUDGET_TOLERANCE,
    EXPERT_RULES,
    _operating_point,
    baseline_comparison,
    expert_rule_firing,
)
from redteam.defend.costs import cost_curve, cost_summary
from redteam.defend.evaluate import fairness_breakdown
from redteam.defend.explain import ACTIONABLE, benign_baselines, counterfactual, phrase


# --------------------------------------------------------------------------------------
# Choosing an operating point a coarse score can actually express
# --------------------------------------------------------------------------------------

def test_a_rule_too_blunt_for_the_budget_is_reported_not_silenced():
    """The regression this module exists to prevent.

    ``payee_is_first_time`` fires on several percent of legitimate payments. The only
    threshold honouring a 0.5% budget is one above every score, which alerts on nothing and
    posts 0% recall. Published beside a model at 72% that reads as a rout, when what it
    actually says is that the comparison was never run.
    """
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(120, int), np.zeros(9_880, int)])
    # Fires on 80% of fraud and 5% of legitimate traffic: a genuinely useful rule that
    # simply cannot be operated at a 0.5% false-positive budget.
    flag = np.concatenate([rng.random(120) < 0.8, rng.random(9_880) < 0.05]).astype(float)

    threshold = _operating_point(flag, y, target_fpr=0.005)
    alert = flag >= threshold

    assert alert.sum() > 0, "the rule must be judged at a point where it alerts on something"
    assert alert[y == 1].mean() > 0.5, "its real recall must survive into the table"
    assert alert[y == 0].mean() > 0.005 * BUDGET_TOLERANCE, (
        "and its real false-positive rate must be reported as outside the budget"
    )


def test_a_continuous_score_is_held_to_the_budget_exactly():
    """The tolerance is for flagging, not for selecting. The deployed model does not get to
    overshoot its own budget by 60% and still be described as operating at it."""
    rng = np.random.default_rng(1)
    y = np.concatenate([np.ones(200, int), np.zeros(9_800, int)])
    scores = np.concatenate([rng.normal(3.0, 1.0, 200), rng.normal(0.0, 1.0, 9_800)])

    threshold = _operating_point(scores, y, target_fpr=0.005)
    assert (scores >= threshold)[y == 0].mean() <= 0.005


def test_comparable_budget_rejects_a_detector_that_barely_alerts():
    """A row alerting at a tenth of the budget is not comparable, however low its FPR."""
    assert BUDGET_FLOOR < 1.0 < BUDGET_TOLERANCE
    target = 0.005
    assert not (target * BUDGET_FLOOR <= target * 0.05 <= target * BUDGET_TOLERANCE)
    assert target * BUDGET_FLOOR <= target * 0.9 <= target * BUDGET_TOLERANCE


# --------------------------------------------------------------------------------------
# The comparison table on real data
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored_split(split, tiny_config):
    """A fitted detector and its test-window scores, shared by the tables below."""
    from redteam.defend.model import FraudDetector

    detector = FraudDetector(seed=tiny_config.seed, target_fpr=tiny_config.defence.target_fpr,
                             max_iter=tiny_config.defence.max_iter,
                             n_estimators=tiny_config.defence.forest_n_estimators)
    detector.fit(split.train, split.calibration)
    return split, detector, detector.predict_proba(split.test)


def test_every_baseline_is_measured_on_the_same_rows(scored_split, tiny_config):
    split, _, scores = scored_split
    table = baseline_comparison(split, scores, target_fpr=tiny_config.defence.target_fpr,
                                seed=tiny_config.seed)
    assert not table.empty
    assert table["rows"].nunique() == 1, "a comparison across different denominators is not one"
    assert (table["model"] == "ensemble (deployed)").sum() == 1
    assert table["model"].iloc[0] == "ensemble (deployed)", "the reference belongs first"
    # Every row alerted on something, so no row is a vacuous zero.
    assert (table["alerts_per_10k"] > 0).all()
    assert table["recall"].between(0, 1).all()


def test_the_comparison_carries_its_own_uncertainty(scored_split, tiny_config):
    """A margin over a baseline on a hundred fraud rows needs an interval or it is noise."""
    split, _, scores = scored_split
    table = baseline_comparison(split, scores, target_fpr=tiny_config.defence.target_fpr,
                                seed=tiny_config.seed)
    for column in ("recall_lo95", "recall_hi95", "n_sufficient"):
        assert column in table.columns
    assert (table["recall_lo95"] <= table["recall"]).all()
    assert (table["recall"] <= table["recall_hi95"]).all()


def test_the_rule_set_reports_which_rules_are_dead_weight(scored_split):
    split, _, _ = scored_split
    firing = expert_rule_firing(split.test)
    assert len(firing) == len(EXPERT_RULES)
    assert set(firing["rule"]) == {name for name, _ in EXPERT_RULES}
    # Hit rates are rates.
    assert firing["fraud_hit_rate"].dropna().between(0, 1).all()
    assert firing["legitimate_hit_rate"].dropna().between(0, 1).all()


# --------------------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------------------

def test_fairness_reports_burden_and_protection_together(scored_split):
    """Either number alone is misleading. `fp_rate` without `recall` invites the reply that
    the cohort is riskier; `recall` without `fp_rate` hides who pays for it."""
    split, detector, scores = scored_split
    table = fairness_breakdown(split.test, scores, detector.threshold)
    assert not table.empty
    assert set(table["dimension"]) == {"age_band", "digital_literacy"}
    for column in ("fp_rate", "recall", "victimisation_rate", "fp_rate_lo95",
                   "recall_lo95", "recall_n_sufficient"):
        assert column in table.columns

    # The two rates have different denominators and must not share a sufficiency flag: a
    # cohort can have 9,000 legitimate rows and 4 fraud rows, and the recall computed on the
    # second is the cell a reader needs warning about.
    assert (table["legitimate_rows"] >= table["fraud_rows"]).any()
    assert table["n_sufficient"].isin([0, 1]).all()
    assert table["recall_n_sufficient"].isin([0, 1]).all()


def test_thin_cohorts_are_kept_and_flagged_rather_than_dropped(scored_split):
    """A cohort with four fraud rows is a finding about the population, not a row to hide."""
    split, detector, scores = scored_split
    table = fairness_breakdown(split.test, scores, detector.threshold)
    thin = table[table["recall_n_sufficient"] == 0]
    if not thin.empty:
        # Present, and carrying an interval wide enough to warn the reader off it.
        assert thin["recall_hi95"].sub(thin["recall_lo95"]).min() > 0.15


# --------------------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------------------

def test_the_cost_curve_finds_a_minimum_and_marks_the_deployed_point(scored_split):
    split, detector, scores = scored_split
    curve = cost_curve(split.test, scores, detector.threshold)
    assert not curve.empty
    assert curve["is_optimal"].sum() == 1
    assert curve["is_deployed"].sum() == 1
    # The marked optimum really is the minimum of the column it claims to minimise.
    optimal = curve.loc[curve["is_optimal"] == 1, "total_cost"].iloc[0]
    assert optimal == curve["total_cost"].min()
    # Recall rises as the threshold falls, monotonically, or the curve is not a curve.
    ordered = curve.sort_values("threshold")
    assert (ordered["recall"].diff().dropna() <= 1e-9).all()


def test_net_benefit_is_measured_against_doing_nothing(scored_split):
    split, detector, scores = scored_split
    curve = cost_curve(split.test, scores, detector.threshold)
    summary = cost_summary(curve)
    values = summary.set_index("metric")["value"]

    do_nothing = values["do_nothing_cost"]
    deployed = values["cost_at_deployed_threshold"]
    assert values["net_benefit_at_deployed"] == pytest.approx(do_nothing - deployed, abs=1.0)
    # The loss-minimising point cannot be worse than the deployed one, by construction.
    assert values["cost_at_loss_minimising_threshold"] <= deployed + 1e-6
    assert values["cost_of_budget_constraint"] >= -1e-6


def test_the_price_list_travels_with_the_answer(scored_split):
    """A rupee figure without its assumptions is unreviewable, so the assumptions ship in
    the same artefact rather than in prose somewhere else."""
    split, detector, scores = scored_split
    summary = cost_summary(cost_curve(split.test, scores, detector.threshold))
    metrics = set(summary["metric"])
    assert {"reimbursement_share", "analyst_review", "false_decline"} <= metrics
    assert summary[summary["metric"] == "reimbursement_share"]["note"].iloc[0].startswith(
        "assumption")


# --------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------

def test_benign_baselines_describe_legitimate_traffic_only(split):
    table = benign_baselines(split.train)
    assert not table.empty
    assert (table["p05"] <= table["median"]).all()
    assert (table["median"] <= table["p95"]).all()
    # Label columns must not leak in as "inputs" with a distribution attached.
    assert "is_fraud" not in set(table["column"])
    assert "attack_vector_id" not in set(table["column"])


def test_a_counterfactual_actually_flips_the_model(scored_split):
    """Not a story about the decision: a re-scored perturbation that crosses the threshold."""
    split, detector, scores = scored_split
    baselines = benign_baselines(split.train)
    alerted = split.test[scores >= detector.threshold]
    if alerted.empty:
        pytest.skip("no alerts in this window to explain")

    found = 0
    for i in range(min(len(alerted), 12)):
        row = alerted.iloc[[i]]
        results = counterfactual(row, detector.predict_proba, detector.threshold, baselines)
        for entry in results:
            assert entry["score_after"] < detector.threshold
            assert entry["column"] in ACTIONABLE
            assert entry["from"] != entry["to"]
            assert phrase(entry).startswith("Approved if")
        found += len(results)
    # Not every alert has a single-field counterfactual, and that is a real answer, but
    # across a dozen alerts the machinery must produce at least one or it is not working.
    assert found > 0


def test_a_payment_below_the_threshold_has_no_counterfactual(scored_split):
    split, detector, scores = scored_split
    baselines = benign_baselines(split.train)
    approved = split.test[scores < detector.threshold]
    row = approved.iloc[[0]]
    assert counterfactual(row, detector.predict_proba, detector.threshold, baselines) == []
