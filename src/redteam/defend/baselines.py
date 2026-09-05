"""What the ensemble is actually worth: the same test window, scored by simpler things.

An absolute recall figure is close to meaningless on synthetic data. The generator decides
how separable fraud is, so a high number can be read as evidence about the model or as
evidence about the generator, and a reader has no way to tell which. What survives that
ambiguity is a *comparison*: the same rows, the same false-positive budget, a simpler
detector. If a logistic regression on identical features gets within a point of the stacked
ensemble, then the ensemble is not the contribution and saying so is worth more than the
headline. If a single rule a bank could write on a whiteboard recovers most of the fraud,
the honest conclusion is that this dataset is easy.

Three comparators, chosen because each answers a different sceptical question:

``payee_is_first_time``
    One column, no fitting. This is the question "did you need machine learning at all?",
    and for push-payment fraud it is not a rhetorical one - counterparty novelty is the
    single most predictive field in the schema, and a bank already has it.

``expert_rules``
    Ten hand-written conditions of the kind a fraud team actually deploys, scored by how
    many fire. This is the real incumbent. Most institutions run rules, not models, and
    beating a well-chosen rule set is the claim that matters commercially.

``logistic_regression``
    The same feature matrix the ensemble sees, fitted linearly. This isolates what the
    non-linearity and the ensembling buy, separately from what the feature engineering
    buys - the ablation grid already prices the features, and this prices the model.

**Every comparator is evaluated at the same false-positive budget as the ensemble**, because
a recall number quoted at an unstated alert volume is not comparable to anything. Two of the
three cannot always reach that budget: a binary rule fires at whatever rate it fires at, and
a ten-valued rule count can only step between a handful of operating points. Rather than
quietly compare a 2% FPR rule against a 0.4% FPR model, each row reports the false-positive
rate it actually realised, and ``comparable_budget`` marks whether the row met the target.
A reader who ignores that column will draw a wrong conclusion, so it is not optional.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import MatrixBuilder, Split, input_columns
from .stats import add_wilson

#: How far above the target false-positive rate a row may sit and still count as a
#: like-for-like comparison. Generous, because the alternative to a tolerance is declaring
#: every discrete-scored baseline incomparable and printing a table of blanks.
BUDGET_TOLERANCE = 1.6

#: ...and how far below. A detector alerting at a tenth of the budget is not "comfortably
#: within budget", it is operating somewhere else entirely, and its recall is not comparable
#: to a detector spending the whole allowance. Both bounds are needed: without the lower one,
#: a rule that cannot be tuned and therefore alerts on nothing scores a false-positive rate
#: of zero, passes an upper-bound-only test, and gets published as a comparable baseline with
#: 0% recall - a row that looks like a devastating result for the baseline and is in fact a
#: statement that the comparison was never made.
BUDGET_FLOOR = 0.34


def _one_rule(df: pd.DataFrame) -> np.ndarray:
    """Counterparty novelty alone."""
    return pd.to_numeric(df.get("payee_is_first_time", 0), errors="coerce").fillna(0).to_numpy(float)


#: The incumbent. Each entry is a name and a predicate over the frame, and the score is the
#: number of conditions met. Weighted equally on purpose: a rule set with tuned weights is a
#: model with extra steps, and the point of this comparator is to represent what a team
#: writes down without fitting anything.
EXPERT_RULES: List[Tuple[str, Callable[[pd.DataFrame], np.ndarray]]] = [
    ("new_payee_large_amount",
     lambda d: (_num(d, "payee_is_first_time") == 1) & (_num(d, "amount") > 25_000)),
    ("payee_added_minutes_ago",
     lambda d: _num(d, "payee_added_minutes_ago", np.inf) < 30),
    ("screen_share_during_payment",
     lambda d: _num(d, "screen_share_active") == 1),
    ("remote_access_tool_present",
     lambda d: _num(d, "remote_access_app_detected") == 1),
    ("live_call_while_paying",
     lambda d: (_num(d, "call_in_progress") == 1) & (_num(d, "amount") > 10_000)),
    ("payee_name_mismatch",
     lambda d: _num(d, "payee_name_match_score", 1.0) < 0.6),
    ("recent_sim_swap",
     lambda d: _num(d, "sim_swap_recency_days", np.inf) < 7),
    ("new_device_at_night",
     lambda d: (_num(d, "device_is_new") == 1) & (_num(d, "is_night") == 1)),
    ("mule_shaped_beneficiary",
     lambda d: _num(d, "payee_inbound_unique_payers_24h") >= 8),
    ("irrevocable_rail_new_payee_high_value",
     lambda d: (_num(d, "is_irrevocable_rail") == 1) & (_num(d, "payee_is_first_time") == 1)
               & (_num(d, "amount") > 50_000)),
]


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(dtype=float)


def _expert_rules(df: pd.DataFrame) -> np.ndarray:
    fired = np.zeros(len(df), dtype=float)
    for _, rule in EXPERT_RULES:
        fired += np.asarray(rule(df), dtype=float)
    return fired


def expert_rule_firing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-rule hit rate on fraud and on legitimate traffic.

    Reported alongside the comparison because "the rules got 61%" invites the immediate
    question of which rule, and because a rule that fires on 4% of legitimate payments is
    not deployable however much fraud it finds.
    """
    if df.empty:
        return pd.DataFrame()
    fraud = df["is_fraud"].to_numpy() == 1
    rows = []
    for name, rule in EXPERT_RULES:
        hit = np.asarray(rule(df), dtype=bool)
        rows.append({
            "rule": name,
            "fires_on_fraud": int(hit[fraud].sum()),
            "fraud_hit_rate": round(float(hit[fraud].mean()), 4) if fraud.any() else np.nan,
            "fires_on_legitimate": int(hit[~fraud].sum()),
            "legitimate_hit_rate": round(float(hit[~fraud].mean()), 4) if (~fraud).any() else np.nan,
        })
    out = pd.DataFrame(rows)
    # Lift is the number a fraud lead reads first: how much more often this fires on fraud
    # than on everything else. A rule below 1.0 is actively anti-predictive.
    out["lift"] = (out["fraud_hit_rate"] / out["legitimate_hit_rate"].replace(0, np.nan)).round(1)
    return out.sort_values("lift", ascending=False, na_position="last").reset_index(drop=True)


def _logistic(split: Split, seed: int) -> np.ndarray:
    """A linear model on the ensemble's own feature matrix.

    Median imputation and standardisation are done here rather than in a pipeline object
    because the matrix is already numeric by the time it arrives and the two statistics have
    to be learned on train and applied to test, which is three lines either way.
    """
    columns = input_columns(split.train)
    builder = MatrixBuilder(columns).fit(split.train)
    x_train = builder.transform(split.train)
    x_test = builder.transform(split.test)

    centre = np.nanmedian(x_train, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, centre)
    x_test = np.where(np.isfinite(x_test), x_test, centre)

    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - centre) / scale
    x_test = (x_test - centre) / scale

    model = LogisticRegression(
        max_iter=2000, C=0.5, solver="lbfgs",
        # Fraud is under one percent of the rows. Without rebalancing the fit converges on
        # the majority class and the comparison becomes a straw man rather than a baseline.
        class_weight="balanced", random_state=seed,
    )
    model.fit(x_train, split.train["is_fraud"].to_numpy().astype(int))
    return model.predict_proba(x_test)[:, 1]


def _operating_point(scores: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    """The threshold to judge a detector at, chosen so coarse scores stay comparable.

    :func:`~redteam.defend.thresholds.threshold_for_budget` guarantees the realised
    false-positive rate never exceeds the budget, which is exactly right for setting a
    deployed model's cut and exactly wrong for a rule. ``payee_is_first_time`` fires on
    several percent of legitimate payments, so the only cut honouring a 0.5% budget is one
    above the highest score, which alerts on nothing and scores zero recall. Published
    beside a model at 72%, that row reads as a crushing win. It is not a result at all - it
    says the rule cannot be run at this budget, which is a different and much less
    flattering claim to make on the model's behalf.

    So the search here runs over the operating points the score can actually express. It
    takes the most sensitive cut that stays within tolerance of the budget, and if even the
    strictest non-degenerate cut overshoots, it takes that one and lets the realised rate be
    reported as what it is. Every row then describes a detector that alerted on something,
    and ``comparable_budget`` says whether the alert volume was close enough to the model's
    for the recalls to be set side by side.
    """
    scores = np.asarray(scores, dtype=float)
    negatives = np.sort(scores[y == 0])
    if negatives.size == 0:
        return float(np.min(scores))

    # Candidate cuts are the distinct score values: `score >= v` for each v is every
    # partition the score can express, and nothing between two adjacent values differs.
    values = np.unique(scores)
    # Negatives at or above each cut, via the sorted array rather than a comparison matrix,
    # which on the ensemble's 70k distinct scores would be several gigabytes.
    above = negatives.size - np.searchsorted(negatives, values, side="left")
    fprs = above / negatives.size

    within = np.flatnonzero(fprs <= target_fpr)
    if within.size:
        # Lowest qualifying cut: false-positive rate falls as the threshold rises, so this
        # is the most sensitive point that still respects the budget. For a continuous score
        # this reproduces `threshold_for_budget` exactly, which is the intent - the deployed
        # model must be judged at its real operating point and not at a tolerated overshoot.
        return float(values[within[0]])
    # Nothing the score can express fits the budget. The strictest non-degenerate cut is the
    # only honest choice, and the realised rate beside it says how far outside it lands.
    return float(values[-1])


def baseline_comparison(split: Split, ensemble_scores: np.ndarray, *,
                        target_fpr: float = 0.005, seed: int = 0) -> pd.DataFrame:
    """The ensemble against three simpler detectors on the same rows and the same budget."""
    test = split.test
    if test.empty or test["is_fraud"].sum() == 0:
        return pd.DataFrame()

    y = test["is_fraud"].to_numpy().astype(int)
    amounts = test["amount"].to_numpy(dtype=float)
    at_risk = float(amounts[y == 1].sum())

    candidates: Dict[str, Tuple[np.ndarray, str, int]] = {
        "ensemble (deployed)": (np.asarray(ensemble_scores, dtype=float),
                                "stacked boosting + forest + isolation",
                                len(input_columns(split.train))),
        "logistic_regression": (_logistic(split, seed),
                                "linear, same feature matrix, balanced classes",
                                len(input_columns(split.train))),
        "expert_rules": (_expert_rules(test),
                         f"{len(EXPERT_RULES)} hand-written conditions, unweighted count",
                         len(EXPERT_RULES)),
        "payee_is_first_time": (_one_rule(test),
                                "one column, no fitting",
                                1),
    }

    rows = []
    for name, (scores, note, n_features) in candidates.items():
        thr = _operating_point(scores, y, target_fpr)
        alert = scores >= thr
        realised_fpr = float(alert[y == 0].mean()) if (y == 0).any() else np.nan
        caught = float(amounts[(y == 1) & alert].sum())
        # A constant score vector has no ordering to measure, which happens if a rule never
        # fires on this window. Reported as missing rather than as 0.5.
        rankable = float(np.ptp(scores)) > 0
        rows.append({
            "model": name,
            "recall": round(float(alert[y == 1].mean()), 4),
            "precision": round(float(y[alert].mean()), 4) if alert.any() else 0.0,
            "value_recall": round(caught / at_risk, 4) if at_risk else np.nan,
            "alerts_per_10k": round(float(alert.mean()) * 10_000, 1),
            "realised_fpr": round(realised_fpr, 5),
            "target_fpr": target_fpr,
            "comparable_budget": int(target_fpr * BUDGET_FLOOR
                                     <= realised_fpr <= target_fpr * BUDGET_TOLERANCE),
            "roc_auc": round(float(roc_auc_score(y, scores)), 5) if rankable else np.nan,
            "pr_auc": round(float(average_precision_score(y, scores)), 5) if rankable else np.nan,
            "rows": int(y.sum()),
            "n_features": n_features,
            "note": note,
        })

    out = pd.DataFrame(rows)
    out = add_wilson(out, "recall", "rows")
    # Deployed model first, then descending recall, so the table reads as a comparison
    # against a reference rather than as a ranking the ensemble happens to top.
    out["_order"] = np.where(out["model"] == "ensemble (deployed)", 0, 1)
    return (out.sort_values(["_order", "recall"], ascending=[True, False])
            .drop(columns="_order").reset_index(drop=True))
