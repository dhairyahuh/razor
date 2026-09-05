"""Evaluation that reflects how a payment risk engine is actually judged.

A single AUC on a 0.75%-fraud dataset is close to meaningless: it is dominated by the easy
majority of attacks and says nothing about the operating point anyone would deploy. The
metrics here are the ones a fraud operations team and a model risk committee ask for.

* **Recall at a fixed false-positive budget.** The alert queue is a fixed resource.
* **Value-weighted recall.** Catching a hundred ₹400 card tests is not equivalent to
  missing one ₹8 lakh APP transfer. Money stopped is the number the business owns.
* **Per-vector recall.** The mean hides the vectors that get through. A defence that is
  99% overall and 4% on agentic hijack has one specific, exploitable hole.
* **Hard-negative false-positive rate.** Legitimate payments that look alarming - a genuine
  first-time large transfer to a new payee - are where customer harm actually happens.
* **Evasion delta.** Recall on adversarially-perturbed fraud minus recall on the unmodified
  version of the same vectors, which is the red team's direct score against the blue team.
* **Leave-one-vector-out.** The zero-day proxy: retrain with a vector entirely absent and
  measure what the model catches with no labelled example of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import FraudDetector
from .stats import (
    MIN_ROWS_FOR_INTERPRETATION,
    add_wilson,
    bootstrap_metric,
    partial_auc,
    wilson_interval,
)
from .thresholds import threshold_for_budget


@dataclass
class Headline:
    rows: int
    fraud: int
    fraud_rate: float
    roc_auc: float
    partial_auc_2pct: float
    pr_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    alerts_per_10k: float
    value_recall: float
    value_at_risk: float
    value_saved: float
    brier: float

    def to_dict(self) -> Dict[str, float]:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def headline_metrics(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                     threshold: float) -> Headline:
    alert = (scores >= threshold).astype(int)
    caught_value = float(amounts[(y == 1) & (alert == 1)].sum())
    at_risk = float(amounts[y == 1].sum())
    return Headline(
        rows=int(len(y)),
        fraud=int(y.sum()),
        fraud_rate=float(y.mean()),
        roc_auc=float(roc_auc_score(y, scores)),
        # The number to read instead of ROC AUC. See stats.partial_auc for why the full curve
        # flatters every model at this imbalance.
        partial_auc_2pct=float(partial_auc(y, scores, max_fpr=0.02)),
        pr_auc=float(average_precision_score(y, scores)),
        threshold=float(threshold),
        precision=float(precision_score(y, alert, zero_division=0)),
        recall=float(recall_score(y, alert, zero_division=0)),
        f1=float(f1_score(y, alert, zero_division=0)),
        false_positive_rate=float(alert[y == 0].mean()),
        alerts_per_10k=float(alert.mean() * 10_000),
        value_recall=float(caught_value / at_risk) if at_risk else float("nan"),
        value_at_risk=round(at_risk, 2),
        value_saved=round(caught_value, 2),
        brier=float(brier_score_loss(y, scores)),
    )


def headline_intervals(y: np.ndarray, scores: np.ndarray, target_fpr: float,
                       n_resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap intervals for the metrics the submission leads with.

    Recall is recomputed against a threshold re-derived inside each resample, because the
    operating point is estimated from the same finite sample as everything else and treating
    it as exact would report a narrower interval than the evidence supports.
    """
    def _recall_at_budget(yy: np.ndarray, ss: np.ndarray) -> float:
        thr = threshold_for_budget(ss[yy == 0], target_fpr)
        return float((ss[yy == 1] >= thr).mean())

    metrics = {
        f"recall_at_{target_fpr:g}_fpr": _recall_at_budget,
        "pr_auc": lambda yy, ss: float(average_precision_score(yy, ss)),
        "partial_auc_2pct": lambda yy, ss: partial_auc(yy, ss, max_fpr=0.02),
        "roc_auc": lambda yy, ss: float(roc_auc_score(yy, ss)),
    }
    rows = []
    for name, fn in metrics.items():
        out = bootstrap_metric(y, scores, fn, n_resamples=n_resamples, seed=seed)
        rows.append({"metric": name, **out})
    return pd.DataFrame(rows)


def operating_curve(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                    fprs=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    """Recall and review workload across the false-positive budgets worth considering."""
    negatives = scores[y == 0]
    rows = []
    at_risk = float(amounts[y == 1].sum())
    for fpr in fprs:
        thr = threshold_for_budget(negatives, fpr)
        alert = scores >= thr
        rows.append(
            {
                "target_fpr": fpr,
                "threshold": round(thr, 6),
                "recall": round(float(alert[y == 1].mean()), 4),
                "precision": round(float(y[alert].mean()) if alert.any() else 0.0, 4),
                "value_recall": round(float(amounts[(y == 1) & alert].sum() / at_risk), 4) if at_risk else np.nan,
                "alerts_per_10k": round(float(alert.mean() * 10_000), 1),
            }
        )
    return pd.DataFrame(rows)


def per_vector_recall(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Detection broken out by attack vector, worst first."""
    fraud = test["is_fraud"].to_numpy() == 1
    alert = scores >= threshold
    frame = test.loc[fraud, ["attack_vector_id", "fraud_type", "amount", "evasion_applied"]].copy()
    frame["alert"] = alert[fraud]
    frame["score"] = scores[fraud]

    grouped = frame.groupby("attack_vector_id", sort=False).agg(
        family=("fraud_type", "first"),
        rows=("alert", "size"),
        recall=("alert", "mean"),
        median_score=("score", "median"),
        value=("amount", "sum"),
        value_caught=("amount", lambda s: float(s[frame.loc[s.index, "alert"]].sum())),
    ).reset_index()
    grouped["value_recall"] = (grouped["value_caught"] / grouped["value"].clip(lower=1)).round(4)
    grouped["recall"] = grouped["recall"].round(4)
    grouped["median_score"] = grouped["median_score"].round(4)
    grouped["value"] = grouped["value"].round(2)
    grouped["value_caught"] = grouped["value_caught"].round(2)
    grouped = add_wilson(grouped, "recall", "rows")
    return grouped.sort_values("recall").reset_index(drop=True)


def per_family_recall(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    fraud = test["is_fraud"].to_numpy() == 1
    alert = scores >= threshold
    frame = test.loc[fraud, ["fraud_type", "amount"]].copy()
    frame["alert"] = alert[fraud]
    out = frame.groupby("fraud_type").agg(
        rows=("alert", "size"), recall=("alert", "mean"), value=("amount", "sum")
    ).reset_index()
    out["recall"] = out["recall"].round(4)
    out["value"] = out["value"].round(2)
    out = add_wilson(out, "recall", "rows")
    return out.sort_values("recall").reset_index(drop=True)


def false_positive_breakdown(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Where the false positives land. Hard negatives are the ones that hurt customers."""
    legit = test["is_fraud"].to_numpy() == 0
    alert = scores >= threshold
    frame = test.loc[legit, ["is_hard_negative", "rail", "channel", "amount"]].copy()
    frame["alert"] = alert[legit]

    by_kind = frame.groupby("is_hard_negative").agg(
        rows=("alert", "size"), false_positives=("alert", "sum"), fp_rate=("alert", "mean")
    ).reset_index()
    by_kind["segment"] = by_kind["is_hard_negative"].map({0: "ordinary_legitimate", 1: "hard_negative"})
    by_kind["fp_rate"] = by_kind["fp_rate"].round(5)

    by_rail = frame.groupby("rail").agg(
        rows=("alert", "size"), false_positives=("alert", "sum"), fp_rate=("alert", "mean")
    ).reset_index()
    by_rail["segment"] = "rail:" + by_rail["rail"]
    by_rail["fp_rate"] = by_rail["fp_rate"].round(5)

    cols = ["segment", "rows", "false_positives", "fp_rate"]
    return pd.concat([by_kind[cols], by_rail[cols]], ignore_index=True)


#: Age band labels in the order the population generator assigns them, so an ordinal column
#: can be printed as the thing it means. Kept here rather than imported from the config
#: because the evaluation must be able to read an artefact written by a run whose config is
#: no longer in the process.
AGE_BANDS = ["18-24", "25-34", "35-49", "50-64", "65+"]


def fairness_breakdown(test: pd.DataFrame, scores: np.ndarray,
                       threshold: float) -> pd.DataFrame:
    """False-positive burden and detection benefit, per customer cohort.

    A single global threshold does not fall equally on everyone, and on this problem there
    is a specific reason to look: the cohorts that scams target are the ones whose ordinary
    behaviour least resembles the confident-user norm the model learns from. Hesitation,
    slow form fill, a first transfer to a new payee, a phone call during the payment - these
    are the fraud signals, and they are also what an eighty-year-old making a legitimate
    payment for the first time looks like. If the model is doing what the training
    distribution encourages, the burden lands on the elderly and the digitally unconfident.

    So this table reports two numbers per cohort rather than one. ``fp_rate`` is the burden:
    how often a legitimate payment from this cohort is stopped. ``recall`` is the benefit:
    how much of the fraud aimed at this cohort is caught. Reporting the burden alone invites
    the reply that the cohort is riskier and the friction is earned; reporting both makes the
    trade explicit and lets a reader see whether the people paying the most friction are the
    ones getting the most protection. ``victimisation_rate`` is the third leg - whether the
    cohort is disproportionately attacked in the first place.

    Emitted whether or not it looks good. A fairness panel that only appears when it is
    flattering is marketing.
    """
    if test.empty:
        return pd.DataFrame()

    alert = np.asarray(scores) >= threshold
    fraud = test["is_fraud"].to_numpy() == 1
    frame = pd.DataFrame({"alert": alert, "fraud": fraud,
                          "amount": test["amount"].to_numpy(dtype=float)})

    dimensions: Dict[str, pd.Series] = {}
    if "customer_age_band_ord" in test.columns:
        ordinals = pd.to_numeric(test["customer_age_band_ord"], errors="coerce")
        dimensions["age_band"] = ordinals.map(
            lambda v: AGE_BANDS[int(v)] if pd.notna(v) and 0 <= int(v) < len(AGE_BANDS)
            else "unknown"
        ).to_numpy()
    if "customer_digital_literacy" in test.columns:
        literacy = pd.to_numeric(test["customer_digital_literacy"], errors="coerce")
        # Fixed cut points rather than deciles of this window's own distribution, so a
        # cohort label means the same thing across runs and can be compared between them.
        # `qcut` would silently redefine "lowest decile" every time the population changed.
        dimensions["digital_literacy"] = pd.cut(
            literacy, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        ).astype(str).to_numpy()

    rows = []
    for dimension, values in dimensions.items():
        frame["_segment"] = values
        for segment, part in frame.groupby("_segment", sort=True):
            legit = part[~part["fraud"]]
            attacked = part[part["fraud"]]
            rows.append({
                "dimension": dimension,
                "segment": str(segment),
                "rows": int(len(part)),
                "legitimate_rows": int(len(legit)),
                "false_positives": int(legit["alert"].sum()),
                "fp_rate": round(float(legit["alert"].mean()), 5) if len(legit) else np.nan,
                "fraud_rows": int(len(attacked)),
                "victimisation_rate": round(float(len(attacked) / len(part)), 5) if len(part) else np.nan,
                "recall": round(float(attacked["alert"].mean()), 4) if len(attacked) else np.nan,
                "value_at_risk": round(float(attacked["amount"].sum()), 2),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = add_wilson(out, "fp_rate", "legitimate_rows", prefix="fp_rate")
    # Recall gets its own interval on its own denominator. Sharing `n_sufficient` between a
    # rate computed on 9,000 legitimate rows and one computed on 11 fraud rows would attach
    # the wrong confidence to the second, which is the cell a reader most needs warning about.
    recall_bounds = out.apply(
        lambda r: wilson_interval(int(round((r["recall"] or 0) * r["fraud_rows"])),
                                  int(r["fraud_rows"])) if r["fraud_rows"] else (np.nan, np.nan),
        axis=1, result_type="expand")
    out["recall_lo95"] = recall_bounds[0].round(4)
    out["recall_hi95"] = recall_bounds[1].round(4)
    out["recall_n_sufficient"] = (out["fraud_rows"] >= MIN_ROWS_FOR_INTERPRETATION).astype(int)
    return out.sort_values(["dimension", "segment"]).reset_index(drop=True)


def evasion_delta(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Red team's scoreboard: how much recall the adversarial perturbation bought.

    Compared within vector, because the vectors chosen for evasion are not a random sample.
    """
    fraud = test[test["is_fraud"] == 1].copy()
    fraud["alert"] = scores[test["is_fraud"].to_numpy() == 1] >= threshold
    targeted = fraud[fraud["attack_vector_id"].isin(
        fraud.loc[fraud["evasion_applied"] == 1, "attack_vector_id"].unique()
    )]
    if targeted.empty:
        return pd.DataFrame()
    out = targeted.groupby(["attack_vector_id", "evasion_applied"]).agg(
        rows=("alert", "size"), recall=("alert", "mean")
    ).reset_index()
    pivot = out.pivot(index="attack_vector_id", columns="evasion_applied", values="recall")
    counts = out.pivot(index="attack_vector_id", columns="evasion_applied", values="rows")
    pivot.columns = [f"recall_evasion_{c}" for c in pivot.columns]
    counts.columns = [f"rows_evasion_{c}" for c in counts.columns]
    joined = pivot.join(counts).reset_index()
    if "recall_evasion_0" in joined and "recall_evasion_1" in joined:
        joined["recall_drop"] = (joined["recall_evasion_0"] - joined["recall_evasion_1"]).round(4)
    return joined.round(4).sort_values("recall_drop", ascending=False, na_position="last")


def leave_one_vector_out(split, detector_factory, vectors: Optional[List[str]] = None,
                         min_rows: int = 40, max_vectors: int = 12) -> pd.DataFrame:
    """Retrain with one vector fully removed and measure zero-shot recall on it.

    This is the closest offline proxy for a novel attack. Anything the model still catches
    is being caught by generalisable structure - velocity, counterparty shape, delegated
    authority anomalies - rather than by having memorised the vector.

    The folds are independent and dominate the pipeline's runtime, which makes them look
    like an obvious thing to parallelise. They are not: the estimators inside already use
    every core, so process workers oversubscribe the CPU while each pickles its own copy of
    a large frame. Measured, that is four times slower than running them one after another.
    """
    train, calibration, test = split.train, split.calibration, split.test
    counts = train.loc[train["is_fraud"] == 1, "attack_vector_id"].value_counts()
    candidates = vectors or [v for v, n in counts.items() if n >= min_rows][:max_vectors]
    legit_test = test[test["is_fraud"] == 0]
    novelty_ref = legit_test.sample(min(4000, len(legit_test)), random_state=0)

    rows = []
    for vector in candidates:
        held_test = test[(test["attack_vector_id"] == vector) & (test["is_fraud"] == 1)]
        if len(held_test) < 10:
            continue
        tr = train[train["attack_vector_id"] != vector]
        cal = calibration[calibration["attack_vector_id"] != vector]
        if int(tr["is_fraud"].sum()) < 20 or int(cal["is_fraud"].sum()) < 10:
            continue

        detector = detector_factory().fit(tr, cal)
        eval_frame = pd.concat([legit_test, held_test])
        scores = detector.predict_proba(eval_frame)
        y = eval_frame["is_fraud"].to_numpy().astype(int)

        rows.append(
            {
                "held_out_vector": vector,
                "held_out_rows": len(held_test),
                "zero_shot_recall": round(float((scores[y == 1] >= detector.threshold).mean()), 4),
                "zero_shot_roc_auc": round(float(roc_auc_score(y, scores)), 4),
                "novelty_percentile": round(
                    float(
                        (
                            detector.novelty_scores(held_test)[:, None]
                            > detector.novelty_scores(novelty_ref)[None, :]
                        ).mean()
                    ),
                    4,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["held_out_vector", "held_out_rows", "zero_shot_recall",
                                     "zero_shot_roc_auc", "novelty_percentile"])
    frame = add_wilson(pd.DataFrame(rows), "zero_shot_recall", "held_out_rows")
    return frame.sort_values("zero_shot_recall").reset_index(drop=True)


def worst_slices(test: pd.DataFrame, scores: np.ndarray, threshold: float,
                 min_rows: int = 25, top: int = 8) -> pd.DataFrame:
    """Search the slice space for where the defence is weakest, and report the worst.

    The per-vector table answers "which attack gets through". This answers the harder and more
    operationally useful question: *which kind of payment* gets through, regardless of which
    attack produced it. A model can be strong on every named vector and still have a hole at,
    say, mid-value NEFT payments from long-tenured accounts - because that combination is
    where the benign distribution is densest and the decision boundary is therefore most
    conservative. An attacker who finds that slice does not need a new vector; they just need
    to route existing fraud through it.

    Finding these automatically is a red-team capability in its own right, so it runs as part
    of the defence report rather than being left to whoever reads the CSVs. Slices thinner
    than ``min_rows`` fraud examples are excluded, because the minimum of a noisy statistic
    over many thin slices is always near zero and says nothing.
    """
    fraud = test[test["is_fraud"] == 1].copy()
    if fraud.empty:
        return pd.DataFrame()
    fraud["alert"] = scores[test["is_fraud"].to_numpy() == 1] >= threshold

    amounts = fraud["amount"].to_numpy(dtype=float)
    # Fixed money bands rather than quantiles, so a slice means the same thing across runs
    # and can be compared between them.
    fraud["_amount_band"] = pd.cut(
        amounts, [-np.inf, 500, 5_000, 50_000, 200_000, np.inf],
        labels=["<500", "500-5k", "5k-50k", "50k-2L", ">2L"],
    )
    ages = fraud.get("payee_account_age_days", pd.Series(np.nan, index=fraud.index))
    fraud["_payee_age_band"] = pd.cut(
        ages.to_numpy(dtype=float), [-np.inf, 7, 90, 730, np.inf],
        labels=["<1w", "1w-3m", "3m-2y", ">2y"],
    )

    dimensions = {
        "rail": "rail",
        "channel": "channel",
        "amount_band": "_amount_band",
        "payee_age_band": "_payee_age_band",
    }
    present = {k: v for k, v in dimensions.items() if v in fraud.columns}

    rows = []
    # Singletons and pairs. Triples and beyond fragment the sample past the point where any
    # estimate survives, and the pairs already surface the interactions that matter.
    from itertools import combinations
    combos = [(k,) for k in present] + list(combinations(present, 2))
    for combo in combos:
        cols = [present[k] for k in combo]
        grouped = fraud.groupby(cols, sort=False, observed=True).agg(
            fraud_rows=("alert", "size"),
            recall=("alert", "mean"),
            value_at_risk=("amount", "sum"),
            value_missed=("amount", lambda s: float(s[~fraud.loc[s.index, "alert"]].sum())),
        )
        for key, row in grouped.iterrows():
            if row["fraud_rows"] < min_rows:
                continue
            values = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    "slice": " & ".join(f"{d}={v}" for d, v in zip(combo, values)),
                    "dimensions": len(combo),
                    "fraud_rows": int(row["fraud_rows"]),
                    "recall": round(float(row["recall"]), 4),
                    "value_at_risk": round(float(row["value_at_risk"]), 2),
                    "value_missed": round(float(row["value_missed"]), 2),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = add_wilson(pd.DataFrame(rows), "recall", "fraud_rows")
    return frame.sort_values("recall").head(top).reset_index(drop=True)


def ablation_grid(split, detector_factory, target_fpr: float = 0.005) -> pd.DataFrame:
    """Retrain on each cumulative feature layer and report what it bought.

    This is the table that explains the headline rather than merely stating it. A defence
    reporting high recall on synthetic data invites one obvious objection - that the
    generator is leaking through the features - and the only honest answer is to show how
    much of the performance survives when the derived layers are removed. If the raw payment
    message alone gets most of the way, the derived features are decoration; if it gets
    nowhere, they are the contribution.
    """
    from .dataset import column_layers

    train, calibration, test = split.train, split.calibration, split.test
    if int(train["is_fraud"].sum()) < 20 or int(test["is_fraud"].sum()) < 20:
        return pd.DataFrame()

    layers = column_layers(train)
    available = set(layers["+ guards (full)"])
    y = test["is_fraud"].to_numpy().astype(int)
    amounts = test["amount"].to_numpy().astype(float)
    at_risk = float(amounts[y == 1].sum())

    rows, previous = [], None
    for name, columns in layers.items():
        if not columns:
            continue
        # Withhold the complement rather than passing an allowlist, so the features are still
        # built from the whole frame and only the model's view narrows. Trimming the frame
        # instead would change the derived columns and measure something else entirely.
        withheld = tuple(sorted(available - set(columns)))
        detector = detector_factory(drop_columns=withheld).fit(train, calibration)
        scores = detector.predict_proba(test)
        thr = threshold_for_budget(scores[y == 0], target_fpr)
        alert = scores >= thr
        recall = float(alert[y == 1].mean())
        rows.append(
            {
                "layer": name,
                "features": len(columns),
                "recall": round(recall, 4),
                "delta_recall": round(recall - previous, 4) if previous is not None else np.nan,
                "pr_auc": round(float(average_precision_score(y, scores)), 4),
                "partial_auc_2pct": round(float(partial_auc(y, scores, max_fpr=0.02)), 4),
                "value_recall": round(float(amounts[(y == 1) & alert].sum() / at_risk), 4)
                if at_risk else np.nan,
            }
        )
        previous = recall
    return pd.DataFrame(rows)


def label_shuffle_control(split, detector_factory, target_fpr: float = 0.005,
                          seed: int = 0) -> pd.DataFrame:
    """Refit on permuted labels and confirm the pipeline reports chance.

    The one check that cannot be fooled by a subtle leak. If any part of the feature
    construction, the split or the threshold calibration is quietly conditioning on the
    label, a model trained on shuffled labels will still beat chance, and no amount of
    reasoning about individual features will reveal it. Recall at a fixed false-positive
    budget should land near the budget itself, and PR AUC near prevalence.
    """
    rng = np.random.default_rng(seed)
    train, calibration, test = split.train.copy(), split.calibration.copy(), split.test.copy()
    for frame in (train, calibration, test):
        frame["is_fraud"] = rng.permutation(frame["is_fraud"].to_numpy())
    if int(train["is_fraud"].sum()) < 20:
        return pd.DataFrame()

    detector = detector_factory().fit(train, calibration)
    scores = detector.predict_proba(test)
    y = test["is_fraud"].to_numpy().astype(int)
    thr = threshold_for_budget(scores[y == 0], target_fpr)
    prevalence = float(y.mean())
    return pd.DataFrame([
        {
            "check": "label_shuffle_null",
            "recall": round(float((scores[y == 1] >= thr).mean()), 4),
            "expected_recall": target_fpr,
            "pr_auc": round(float(average_precision_score(y, scores)), 5),
            "expected_pr_auc": round(prevalence, 5),
            "roc_auc": round(float(roc_auc_score(y, scores)), 4),
            "expected_roc_auc": 0.5,
        }
    ])


def guard_contribution(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """What each layer catches on agentic traffic, and what only the union catches.

    Reported as a layered breakdown because the three controls are not interchangeable: the
    intent guard is deterministic and disputable, the injection guard reads content, the
    model reads behaviour. Knowing which one fires changes the response.
    """
    agentic = test[test["initiated_by_agent"] == 1].copy()
    if agentic.empty:
        return pd.DataFrame()
    agentic["model_alert"] = scores[test["initiated_by_agent"].to_numpy() == 1] >= threshold
    fraud = agentic[agentic["is_fraud"] == 1]
    if fraud.empty:
        return pd.DataFrame()

    intent = fraud.get("intent_guard_blocked", pd.Series(0, index=fraud.index)).to_numpy() == 1
    injection = fraud.get("agent_injection_flag", pd.Series(0, index=fraud.index)).to_numpy() == 1
    model = fraud["model_alert"].to_numpy()

    rows = []
    for vector, idx in fraud.groupby("attack_vector_id").groups.items():
        pos = fraud.index.get_indexer(idx)
        rows.append(
            {
                "attack_vector_id": vector,
                "rows": len(pos),
                "intent_guard": round(float(intent[pos].mean()), 4),
                "injection_guard": round(float(injection[pos].mean()), 4),
                "tabular_model": round(float(model[pos].mean()), 4),
                "any_layer": round(float((intent[pos] | injection[pos] | model[pos]).mean()), 4),
                "model_only": round(
                    float((model[pos] & ~intent[pos] & ~injection[pos]).mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("any_layer").reset_index(drop=True)


def guard_leakage(featured: pd.DataFrame) -> pd.DataFrame:
    """Audit each guard output as a standalone classifier of the label.

    This exists because a real defect got all the way to a published headline of 100% recall
    and nothing in the evaluation could see it. The fidelity probes deliberately zero-fill
    the guard columns, since guards are defence outputs computed long after generation, so
    the only artefact that showed the problem was the ablation grid - one row, in one table,
    that a reader has to already suspect something to look at.

    A guard column is a legitimate feature: a content classifier that reads a scam transcript
    *should* be informative. What it must not be is a *label*. Two ways it becomes one:

    * its score is near-perfect because a single author wrote both classes of the corpus it
      was fitted on, so it separates synthetic text rather than real coercion;
    * its telemetry is only ever present on fraud, so the model learns "a transcript exists"
      instead of anything the transcript says.

    The second is the one that bit. It is reported here as ``p_fraud_given_present`` beside
    the base rate, because a column that is 95% predictive purely by being populated is a
    leak whatever its AUC says.
    """
    from .dataset import EXTRA_INPUTS

    #: Guards whose telemetry is either present or absent independently of the score, and the
    #: column that records which. A genuine call scores near zero and rounds to the same 0.0
    #: as a payment nobody phoned about, so reading presence off the score would measure
    #: "flagged as coercive" and call it coverage.
    COVERAGE = {"vishing_score": "vishing_covered", "vishing_flag": "vishing_covered",
                "agent_injection_score": "initiated_by_agent",
                "agent_injection_flag": "initiated_by_agent"}

    y = featured["is_fraud"].to_numpy().astype(int)
    base = float(y.mean())
    rows: List[Dict[str, object]] = []
    for column in EXTRA_INPUTS:
        if column not in featured.columns:
            continue
        values = pd.to_numeric(featured[column], errors="coerce").fillna(0.0).to_numpy(float)
        if np.allclose(values, values[0]):
            continue
        cover = COVERAGE.get(column)
        if cover and cover in featured.columns:
            present = pd.to_numeric(featured[cover], errors="coerce").fillna(0).to_numpy() != 0
        else:
            present = values != 0.0
        rows.append({
            "guard_column": column,
            "single_feature_auc": round(float(roc_auc_score(y, values)), 4)
                                  if y.min() != y.max() else float("nan"),
            "coverage": round(float(present.mean()), 4),
            "p_fraud_given_present": round(float(y[present].mean()), 4) if present.any()
                                     else float("nan"),
            "base_fraud_rate": round(base, 4),
            # A column present on fraud and almost nothing else is a label wearing a feature's
            # name, regardless of how good the classifier behind it is.
            "presence_lift": round(float(y[present].mean() / base), 2)
                             if present.any() and base > 0 else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values("single_feature_auc", ascending=False)
            .reset_index(drop=True))


def score_lift_table(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                     bins: int = 10) -> pd.DataFrame:
    """Decile lift: the format fraud strategy teams actually read."""
    order = np.argsort(-scores)
    y_sorted, amt_sorted = y[order], amounts[order]
    chunks = np.array_split(np.arange(len(y)), bins)
    base_rate = y.mean()
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "decile": i,
                "rows": len(chunk),
                "fraud": int(y_sorted[chunk].sum()),
                "fraud_rate": round(float(y_sorted[chunk].mean()), 5),
                "lift": round(float(y_sorted[chunk].mean() / base_rate), 2) if base_rate else np.nan,
                "fraud_value": round(float(amt_sorted[chunk][y_sorted[chunk] == 1].sum()), 2),
            }
        )
    return pd.DataFrame(rows)
