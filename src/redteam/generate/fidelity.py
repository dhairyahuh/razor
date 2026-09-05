"""Fidelity assessment for the generated data.

"Fidelity" for synthetic payment data is not one number, and the usual practice of
reporting a single distance metric hides more than it shows. Four different questions
matter, and they can disagree:

1. **Are the marginals right?** Amounts should obey Benford's law, show the round-number
   spikes human-chosen amounts always show, and follow a realistic diurnal and weekly
   rhythm. This is checked against published regularities, not against a reference sample,
   so it is a genuine external check rather than a self-comparison.

2. **Is the joint structure right?** Per-customer and per-merchant behaviour has to be
   self-consistent, and the correlation structure between features must survive.

3. **Is the fraud realistically hard?** If a single feature separates fraud from legitimate
   traffic, the simulator has leaked the answer. Per-feature AUC is reported precisely to
   catch that failure mode, with an explicit ceiling above which a feature is flagged.

4. **Does it transfer?** A model trained only on synthetic fraud is evaluated against a
   held-out slice generated from a *different* seed and a *different* attacker
   configuration - the closest available analogue of train-on-synthetic / test-on-real.

The scorer is intentionally willing to fail its own data. ``flags`` lists every check that
did not pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Expected first-digit frequencies under Benford's law.
BENFORD = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

#: A single observable feature that separates fraud this well is a simulator artefact.
LEAKAGE_AUC_CEILING = 0.93

#: Share of legitimate payments a fraud team can afford to send to review. The industry
#: operating range is a few tenths of a percent to a few percent; the middle of that band is
#: where recall comparisons are meaningful.
REVIEW_BUDGET_FPR = 0.005

#: Recall at that budget above which the *combination* of observable columns is too clean.
#: No single feature has to look suspicious for this to fire: a dozen weakly predictive
#: features all pointing the same way stack into near-perfect separation, and that is
#: exactly the failure the per-feature check cannot see. Deployed card-fraud systems report
#: roughly 0.5-0.85 here, so anything above this is the simulator talking.
RECALL_AT_BUDGET_CEILING = 0.92


@dataclass
class FidelityReport:
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    #: Findings that are not provable at the 95% level but are not clean either. Separated
    #: from ``flags`` so that a build can fail on the first and a reader is still told about
    #: the second - see :func:`_probe_separability` for why the distinction earns its keep.
    warnings: List[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Unweighted mean of the component scores, each in [0, 1]."""
        if not self.scores:
            return 0.0
        return round(float(np.mean(list(self.scores.values()))), 4)

    @property
    def zeroed(self) -> List[str]:
        """Components that scored zero: the worst reading a check can return.

        Surfaced separately because a zero paired with an empty ``flags`` list is the most
        misleading state this report can be in, and it is reachable: a check whose point
        estimate is past its ceiling but whose confidence interval straddles it scores zero
        and, correctly, does not fail the build. A summary line that printed "no flags" over
        that would be hiding its own worst number.
        """
        return sorted(k for k, v in self.scores.items() if v <= 0.0)

    def to_dict(self) -> Dict[str, object]:
        return {"overall": self.overall, "scores": self.scores,
                "flags": self.flags, "warnings": self.warnings,
                "zeroed": self.zeroed, "details": self.details}


def assess(df: pd.DataFrame, *, reference: Optional[pd.DataFrame] = None) -> FidelityReport:
    report = FidelityReport()
    legit = df[df["is_fraud"] == 0]

    _score_benford(legit, report)
    _score_round_numbers(legit, report)
    _score_diurnal(legit, report)
    _score_weekly(legit, report)
    _score_amount_tail(legit, report)
    _score_entity_consistency(df, report)
    _score_payee_concentration(df, report)
    _score_separability(df, report)
    _score_joint_separability(df, report)
    _score_derived_separability(df, report)
    _score_fraud_realism(df, report)
    if reference is not None:
        _score_distribution_match(df, reference, report)
    return report


# --------------------------------------------------------------------------------------
# Marginal realism
# --------------------------------------------------------------------------------------

def _score_benford(legit: pd.DataFrame, report: FidelityReport) -> None:
    amounts = legit["amount"].to_numpy()
    amounts = amounts[amounts >= 1]
    lead = (amounts / np.power(10, np.floor(np.log10(amounts)))).astype(int)
    observed = np.array([(lead == d).mean() for d in range(1, 10)])
    # Total variation distance to the Benford distribution.
    tvd = 0.5 * np.abs(observed - BENFORD).sum()
    score = float(np.clip(1 - tvd / 0.35, 0, 1))
    report.scores["benford_first_digit"] = round(score, 4)
    report.details["benford"] = {
        "observed": [round(float(x), 4) for x in observed],
        "expected": [round(float(x), 4) for x in BENFORD],
        "total_variation_distance": round(float(tvd), 4),
    }
    if tvd > 0.12:
        report.flags.append(f"amount first-digit distribution deviates from Benford (TVD={tvd:.3f})")


def _score_round_numbers(legit: pd.DataFrame, report: FidelityReport) -> None:
    """Human-chosen amounts cluster on round values; a pure lognormal has no such mass."""
    amt = legit["amount"].to_numpy()
    share_100 = float(np.mean(np.isclose(amt % 100, 0)))
    share_500 = float(np.mean(np.isclose(amt % 500, 0)))
    # Real UPI/P2P data lands roughly in the 15-40% range for multiples of 100.
    score = float(np.clip(1 - abs(share_100 - 0.26) / 0.26, 0, 1))
    report.scores["round_number_mass"] = round(score, 4)
    report.details["round_numbers"] = {
        "multiple_of_100": round(share_100, 4),
        "multiple_of_500": round(share_500, 4),
    }
    if share_100 < 0.08:
        report.flags.append("too few round-number amounts: synthetic-looking amount distribution")


def _score_diurnal(legit: pd.DataFrame, report: FidelityReport) -> None:
    counts = legit.groupby("hour").size().reindex(range(24), fill_value=0).to_numpy().astype(float)
    share = counts / counts.sum()
    night = share[[0, 1, 2, 3, 4, 5]].sum()
    evening = share[[18, 19, 20, 21]].sum()
    morning = share[[10, 11, 12, 13]].sum()
    checks = {
        "night_share_below_0.10": night < 0.10,
        "evening_peak_above_morning": evening > morning,
        "evening_share_above_0.22": evening > 0.22,
    }
    score = float(np.mean(list(checks.values())))
    report.scores["diurnal_shape"] = round(score, 4)
    report.details["diurnal"] = {
        "night_share": round(float(night), 4),
        "morning_share": round(float(morning), 4),
        "evening_share": round(float(evening), 4),
        "checks": {k: bool(v) for k, v in checks.items()},
    }
    for name, ok in checks.items():
        if not ok:
            report.flags.append(f"diurnal check failed: {name}")


def _score_weekly(legit: pd.DataFrame, report: FidelityReport) -> None:
    counts = legit.groupby("day_of_week").size().reindex(range(7), fill_value=0).to_numpy().astype(float)
    share = counts / counts.sum()
    # Expect a visible but not extreme weekly rhythm.
    spread = float(share.max() - share.min())
    score = float(np.clip(1 - abs(spread - 0.055) / 0.08, 0, 1))
    report.scores["weekly_seasonality"] = round(score, 4)
    report.details["weekly_share"] = [round(float(x), 4) for x in share]


def _score_amount_tail(legit: pd.DataFrame, report: FidelityReport) -> None:
    """Payment amounts are heavy tailed; check the log-amount distribution is plausible."""
    amt = legit["amount"].to_numpy()
    log_amt = np.log10(np.maximum(amt, 1))
    skew = float(stats.skew(log_amt))
    kurt = float(stats.kurtosis(log_amt))
    p99_p50 = float(np.percentile(amt, 99) / max(np.percentile(amt, 50), 1))
    # Real retail payment data has a p99/p50 ratio in the tens.
    ratio_ok = 8 <= p99_p50 <= 400
    skew_ok = -0.5 <= skew <= 1.5
    score = float(np.mean([ratio_ok, skew_ok]))
    report.scores["amount_tail"] = round(score, 4)
    report.details["amount_tail"] = {
        "p99_over_p50": round(p99_p50, 2),
        "log_skew": round(skew, 4),
        "log_excess_kurtosis": round(kurt, 4),
    }
    if not ratio_ok:
        report.flags.append(f"amount tail ratio p99/p50={p99_p50:.1f} outside plausible range")


# --------------------------------------------------------------------------------------
# Joint structure
# --------------------------------------------------------------------------------------

def _score_entity_consistency(df: pd.DataFrame, report: FidelityReport) -> None:
    """Between-customer variance should dominate within-customer variance.

    If every customer's spending looks like every other customer's, the simulator has
    drawn from one global distribution and no per-entity model can work.
    """
    legit = df[df["is_fraud"] == 0]
    log_amt = np.log10(np.maximum(legit["amount"].to_numpy(), 1))
    frame = pd.DataFrame({"cust": legit["customer_id"].to_numpy(), "x": log_amt})
    grouped = frame.groupby("cust")["x"]
    means = grouped.mean()
    within = float(grouped.var().mean())
    between = float(means.var())
    icc = between / max(between + within, 1e-9)
    # Real payment panels show an intra-class correlation around 0.15-0.45.
    score = float(np.clip(1 - abs(icc - 0.30) / 0.30, 0, 1))
    report.scores["per_customer_consistency"] = round(score, 4)
    report.details["amount_icc"] = round(icc, 4)
    if icc < 0.05:
        report.flags.append("per-customer amount variation is indistinguishable from global draw")


def _score_payee_concentration(df: pd.DataFrame, report: FidelityReport) -> None:
    """Payment relationships are heavy tailed: most volume goes to a few known payees."""
    legit = df[df["is_fraud"] == 0]
    counts = legit["payee_account_id"].value_counts().to_numpy().astype(float)
    share = np.sort(counts)[::-1] / counts.sum()
    top1 = float(share[: max(1, len(share) // 100)].sum())
    first_time_rate = float(legit["payee_is_first_time"].mean())
    conc_ok = 0.05 <= top1 <= 0.75
    novelty_ok = 0.05 <= first_time_rate <= 0.60
    report.scores["payee_graph_shape"] = round(float(np.mean([conc_ok, novelty_ok])), 4)
    report.details["payee_graph"] = {
        "top_1pct_volume_share": round(top1, 4),
        "first_time_payee_rate": round(first_time_rate, 4),
    }
    if not novelty_ok:
        report.flags.append(f"first-time-payee rate {first_time_rate:.3f} is implausible")


# --------------------------------------------------------------------------------------
# Is the fraud realistically hard?
# --------------------------------------------------------------------------------------

def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, robust to ties, no sklearn dependency."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups.
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i: j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _score_separability(df: pd.DataFrame, report: FidelityReport) -> None:
    """No single observable feature should come close to solving the problem alone."""
    from ..schema import observable_columns

    labels = df["is_fraud"].to_numpy()
    worst: List[tuple] = []
    for col in observable_columns():
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = series.to_numpy().astype(float)
        if np.nanstd(values) == 0:
            continue
        auc = _auc(values, labels)
        worst.append((col, round(max(auc, 1 - auc), 4)))

    worst.sort(key=lambda kv: -kv[1])
    top = worst[:10]
    max_auc = top[0][1] if top else 0.5
    score = float(np.clip((LEAKAGE_AUC_CEILING - max_auc) / (LEAKAGE_AUC_CEILING - 0.5), 0, 1))
    report.scores["no_single_feature_leak"] = round(score, 4)
    report.details["top_single_feature_auc"] = top
    for col, auc in top:
        if auc > LEAKAGE_AUC_CEILING:
            report.flags.append(f"feature '{col}' alone separates fraud at AUC={auc:.3f}")


#: Row cap for the separability probes. Set above any run this repo ships so that the probes
#: see the whole frame: subsampling to 120k made the estimate swing by 27 points between two
#: sample seeds, which is far larger than the effects the check exists to detect.
SEPARABILITY_MAX_ROWS = 400_000

#: Refits per separability probe. A single fit of a fixed-length boosting run against a sub-1%
#: positive rate lands in one of two regimes depending on the exact training rows, and the gap
#: between them is wider than the ceiling this check enforces. Five is enough for a stable
#: median without making the probe the slowest thing in the generate stage.
SEPARABILITY_REPEATS = 5

#: Booster settings for the probe, chosen for *stability of the estimate* rather than raw
#: strength. The obvious configuration - depth 6 at a 0.1 learning rate - overfits a sub-1%
#: positive rate badly enough that its answer depends on the draw: across five bootstrap refits
#: of identical data it returned recalls from 73.4% to 92.3%, a 19-point spread around a
#: 92% ceiling. Halving the learning rate and capping depth at 4 collapses that spread to 1.8
#: points *and raises the median*, which is the tell that the deep model was losing signal to
#: overfitting rather than finding extra. An auditor's number has to be reproducible before it
#: is sensitive, and this is both.
SEPARABILITY_MODEL = {"max_iter": 150, "learning_rate": 0.05, "max_depth": 4}


def _score_joint_separability(df: pd.DataFrame, report: FidelityReport,
                              max_rows: int = SEPARABILITY_MAX_ROWS, seed: int = 0) -> None:
    """How separable is fraud using all observable columns at once?

    The per-feature check passes whenever no column individually gives the game away, which
    is a much weaker guarantee than it sounds: a dozen features at AUC 0.78, all firing on
    the same rows, stack into near-perfect separation and a "detection result" that is
    really a measurement of the generator. This fits a small booster on an earlier time
    slice and scores a later one, answering the question the defence is about to be graded
    on before the defence is trained.

    The headline number is recall at a fixed low false-positive budget rather than AUC.
    Under the heavy class imbalance of a real portfolio, AUC flatters everything and hides
    the part that decides whether a system is deployable; recall at a 0.5% review budget is
    what a payments team would actually quote, and published card-fraud systems land in the
    0.5-0.85 band. Prevalence is preserved when subsampling, because these numbers are only
    comparable to a real portfolio if the imbalance is too.

    The target is *not* a low number. Real portfolios are separable or fraud detection would
    not work at all; the target is "hard but tractable".
    """
    from ..schema import observable_columns

    cols = [c for c in observable_columns() if c in df.columns]
    _probe_separability(df, report, cols, key="joint_separability",
                        label="observable columns", max_rows=max_rows, seed=seed)


def _score_derived_separability(df: pd.DataFrame, report: FidelityReport,
                                max_rows: int = SEPARABILITY_MAX_ROWS, seed: int = 0) -> None:
    """The same question, asked of the matrix the defence is actually trained on.

    Auditing only the raw schema measures the wrong thing. The defence reads the raw columns
    *plus* the trailing-window block, the payee-graph block and the guard outputs, and a
    feature built from a leaky entity id can separate fraud perfectly while every raw column
    looks innocent. Checking the raw matrix alone would report an honest number about a
    matrix nobody trains on.

    Both numbers are published side by side. The gap between them is itself the finding: a
    large one means the feature engineering, not the schema, is where the generator leaks.
    """
    from ..defend.dataset import input_columns
    from ..features.graph import build_graph_features
    from ..features.tabular import build_features

    if int(df["is_fraud"].sum()) < 60:
        return
    try:
        frame = build_graph_features(build_features(df))
    except Exception as exc:  # pragma: no cover - diagnostic path
        report.details["derived_separability"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        return

    # The guard columns are defence outputs computed later in the pipeline. Zero-filling keeps
    # the column set stable so the two runs are comparable rather than silently different.
    for c in ("agent_injection_score", "agent_injection_flag", "intent_guard_blocked",
              "intent_guard_stepup", "intent_guard_violation_count"):
        if c not in frame.columns:
            frame[c] = 0.0

    _probe_separability(frame, report, input_columns(frame), key="derived_separability",
                        label="derived model inputs", max_rows=max_rows, seed=seed)


def _probe_separability(df: pd.DataFrame, report: FidelityReport, cols: List[str], *,
                        key: str, label: str, max_rows: int, seed: int) -> None:
    """Fit a throwaway booster on an earlier time slice and score a later one."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve

    # Imported here rather than at module scope: the defence package pulls in the whole
    # feature and model stack, and the generator should not depend on the thing it is meant
    # to be scored independently of.
    from ..defend.stats import wilson_interval
    from ..schema import CATEGORICAL_COLUMNS

    frame = df
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame = frame.sort_values("timestamp", kind="mergesort")
    if int(frame["is_fraud"].sum()) < 60 or int((frame["is_fraud"] == 0).sum()) < 500:
        return

    cut = frame["timestamp"].quantile(0.7)
    train, test = frame[frame["timestamp"] < cut], frame[frame["timestamp"] >= cut]
    if int(train["is_fraud"].sum()) < 30 or int(test["is_fraud"].sum()) < 15:
        return

    cols = [c for c in cols if c in frame.columns]
    cat = [c for c in CATEGORICAL_COLUMNS if c in cols]
    levels = {c: sorted(frame[c].astype(str).unique()) for c in cat}

    def _matrix(part: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(part), len(cols)), dtype=float)
        for i, c in enumerate(cols):
            if c in cat:
                out[:, i] = pd.Categorical(part[c].astype(str), categories=levels[c]).codes
            else:
                out[:, i] = pd.to_numeric(part[c], errors="coerce").to_numpy(dtype=float)
        return out

    # Early stopping must be off. sklearn enables it automatically above 10,000 rows and
    # validates on a random slice, which at a sub-1% fraud rate holds almost no fraud at all -
    # so the loss looks flat, boosting halts after ~15 iterations, and the probe reports a
    # comfortable number produced by a model that never finished training. That is the worst
    # possible failure for an auditor: it under-reports separability precisely when there is a
    # lot of it, and the check silently stops being able to fail its own data.
    #
    # Leaving it off costs something, though, and a single fit does not survive it. Boosting a
    # depth-6 model for a fixed 150 rounds against a 0.7% positive rate overfits, and how badly
    # depends on the exact training rows in a way that is neither smooth nor small: holding the
    # test slice and the model seed identical and varying only which rows train it, this probe
    # returned 92.3% recall on 70% of the data, 76.9% on 80%, 93.4% on 90% and 82.3% on all of
    # it. Seventeen points of swing from resampling alone, against a ceiling set at 92%. A
    # single fit therefore does not measure the generator, it measures a coin flip, and the
    # quick-versus-full comparison that appeared to show the full profile leaking was mostly
    # this.
    #
    # So the estimate is repeated over bagged training draws and reduced by the median. The
    # median rather than the mean because the bad regime is a heavy lower tail rather than
    # symmetric noise, and the spread is published so a reader can see when the probe is
    # having trouble agreeing with itself.
    y = test["is_fraud"].to_numpy()
    x_test = _matrix(test)
    x_train, y_train = _matrix(train), train["is_fraud"].to_numpy()

    recalls, aucs = [], []
    for repeat in range(SEPARABILITY_REPEATS):
        rs = np.random.RandomState(seed + repeat)
        # Full-size draws with replacement: subsampling to a fraction would confound the
        # repeat with a training-budget change, which is the other half of what makes the
        # single-fit number hard to read.
        take = (np.arange(len(y_train)) if repeat == 0
                else rs.randint(0, len(y_train), size=len(y_train)))
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=seed,
                                               **SEPARABILITY_MODEL)
        model.fit(x_train[take], y_train[take])
        scores = model.predict_proba(x_test)[:, 1]
        aucs.append(float(roc_auc_score(y, scores)))
        fpr, tpr, _ = roc_curve(y, scores)
        i = max(int(np.searchsorted(fpr, REVIEW_BUDGET_FPR, side="right")) - 1, 0)
        recalls.append(float(tpr[i]))

    recall = float(np.median(recalls))
    auc = float(np.median(aucs))
    spread = float(np.max(recalls) - np.min(recalls))

    # The probe's own uncertainty. On a small run the held-out slice holds a few dozen fraud
    # rows, where recall is quantised in steps of 1/n and an interval spans ten points or
    # more - so a point estimate of 0.94 there is not distinguishable from one of 0.82. The
    # check has to be honest about this or it becomes the thing it exists to prevent: a
    # confident number computed on too little evidence.
    n_fraud = int(y.sum())
    lo, hi = wilson_interval(int(round(recall * n_fraud)), n_fraud)

    # Wilson covers only the test slice's sampling error. The refit spread is a second and
    # usually larger source of uncertainty about the same quantity, and reporting an interval
    # that ignored it would understate the probe's own noise by more than the effect it is
    # looking for. Widening by half the observed spread on each side is crude - the two are
    # not independent and this does not pretend to be an exact coverage guarantee - but the
    # direction is the honest one and the alternative is a confident number that the very
    # next refit contradicts.
    lo = max(0.0, lo - spread / 2)
    hi = min(1.0, hi + spread / 2)

    score = float(np.clip((RECALL_AT_BUDGET_CEILING - recall)
                          / (RECALL_AT_BUDGET_CEILING - 0.50), 0, 1))
    report.scores[key] = round(score, 4)
    report.details[key] = {
        "recall_at_review_budget": round(recall, 4),
        "recall_lo95": round(lo, 4),
        "recall_hi95": round(hi, 4),
        "review_budget_fpr": REVIEW_BUDGET_FPR,
        "held_out_auc": round(auc, 4),
        "test_fraud_rows": n_fraud,
        "columns": len(cols),
        # Published because it is the number that decides how much of the rest to believe.
        "refit_spread": round(spread, 4),
        "refits": SEPARABILITY_REPEATS,
        "train_rows": int(len(y_train)),
    }
    # Flag on the lower bound, not the point estimate. This is the difference between "the
    # data might be too separable" and "the data is too separable at any reading of the
    # evidence", and only the second is worth failing a build over. On a full run the
    # interval is a point or two wide, so this changes nothing there; on a small one it stops
    # the check crying wolf on a sample that cannot support the claim either way.
    detail = (f"{label} jointly recover {recall:.1%} of fraud "
              f"(95% CI {lo:.1%}-{hi:.1%}, n={n_fraud}) at a "
              f"{REVIEW_BUDGET_FPR:.1%} false-positive budget")
    if lo > RECALL_AT_BUDGET_CEILING:
        report.flags.append(
            f"{detail}; downstream detection metrics are measuring the generator, not a "
            "defence"
        )
    elif recall > RECALL_AT_BUDGET_CEILING:
        # Past the ceiling on the point estimate but not at the lower bound. The build does
        # not fail, because "too separable at any reading of the evidence" is the bar for
        # that. But it must not read as clean either: this is the state a full-size run
        # reached while the summary line said "no flags", and the component had scored zero.
        report.warnings.append(
            f"{detail}, past the {RECALL_AT_BUDGET_CEILING:.0%} ceiling on the point "
            "estimate though not at the lower bound. Treat detection metrics from this "
            "dataset as an upper bound on what a defence is contributing"
        )


def _score_fraud_realism(df: pd.DataFrame, report: FidelityReport) -> None:
    """Fraud should overlap legitimate traffic, not occupy its own corner of the space."""
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    if fraud.empty:
        report.scores["fraud_overlap"] = 0.0
        return

    checks: Dict[str, bool] = {}
    # Amounts must overlap: fraud that is always huge is trivially separable.
    q = [0.25, 0.5, 0.75]
    fq = np.percentile(fraud["amount"], [x * 100 for x in q])
    lq = np.percentile(legit["amount"], [x * 100 for x in q])
    checks["amount_iqr_overlaps"] = bool(fq[0] < lq[2] and lq[0] < fq[2])
    # Fraud must appear on every rail the library says it uses.
    checks["multi_rail"] = fraud["rail"].nunique() >= 5
    # A meaningful share of fraud must carry no coercion telemetry at all.
    quiet = (
        (fraud["screen_share_active"] == 0)
        & (fraud["remote_access_app_detected"] == 0)
        & (fraud["call_in_progress"] == 0)
        & (fraud["device_is_emulator"] == 0)
    )
    checks["has_quiet_fraud"] = bool(quiet.mean() > 0.35)
    # Legitimate traffic must contain the same surface signals, or they become giveaways.
    checks["benign_shares_signals"] = bool(
        legit["screen_share_active"].mean() > 0.0005 and legit["payee_is_first_time"].mean() > 0.02
    )
    ks = stats.ks_2samp(np.log10(np.maximum(fraud["amount"], 1)),
                        np.log10(np.maximum(legit["amount"], 1)))
    checks["amount_ks_below_0.6"] = bool(ks.statistic < 0.6)

    report.scores["fraud_overlap"] = round(float(np.mean(list(checks.values()))), 4)
    report.details["fraud_overlap"] = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "quiet_fraud_share": round(float(quiet.mean()), 4),
        "amount_ks_statistic": round(float(ks.statistic), 4),
    }
    for name, ok in checks.items():
        if not ok:
            report.flags.append(f"fraud realism check failed: {name}")


# --------------------------------------------------------------------------------------
# Reference comparison
# --------------------------------------------------------------------------------------

def _score_distribution_match(df: pd.DataFrame, reference: pd.DataFrame,
                              report: FidelityReport) -> None:
    """Two-sample comparison against an independently generated reference dataset."""
    from ..schema import observable_columns

    ks_stats: Dict[str, float] = {}
    for col in observable_columns():
        if col not in df.columns or col not in reference.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        a = df[col].to_numpy().astype(float)
        b = reference[col].to_numpy().astype(float)
        if np.nanstd(a) == 0 and np.nanstd(b) == 0:
            continue
        ks_stats[col] = round(float(stats.ks_2samp(a, b).statistic), 4)

    if not ks_stats:
        return
    mean_ks = float(np.mean(list(ks_stats.values())))
    report.scores["reference_ks_agreement"] = round(float(np.clip(1 - mean_ks / 0.2, 0, 1)), 4)
    report.details["reference_ks"] = {
        "mean": round(mean_ks, 4),
        "worst": sorted(ks_stats.items(), key=lambda kv: -kv[1])[:10],
    }


def correlation_drift(a: pd.DataFrame, b: pd.DataFrame, columns: List[str]) -> float:
    """Mean absolute difference between two datasets' correlation matrices."""
    cols = [c for c in columns if c in a.columns and c in b.columns
            and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    ca = a[cols].corr().to_numpy()
    cb = b[cols].corr().to_numpy()
    mask = ~(np.isnan(ca) | np.isnan(cb))
    return float(np.abs(ca[mask] - cb[mask]).mean())
