"""Assembling the three defence layers into one scored decision.

Order matters and is not arbitrary:

1. **Derived features** - causal velocity, counterparty and behavioural aggregates, then
   nightly graph snapshots. Both are computed on the full stream before splitting, because
   both are already causal by construction; recomputing them per split would change a
   transaction's features depending on which split it landed in, which is worse.
2. **Injection guard** - fitted on training-window bundles only, then scores every bundle.
   Its output becomes ``agent_injection_score``, an ordinary model input with ordinary
   errors.
3. **Intent guard** - deterministic, no fitting, so it runs on everything. Its verdict is
   both a hard block and a model feature.
4. **Tabular ensemble** - reads all of the above.

The final decision is the union of the deterministic block and the model alert. Keeping
them separate rather than folding the guard into the score is deliberate: a payment
declined for ``PAYEE_MISMATCH`` has a defensible, auditable reason, and a payment declined
for a 0.94 model score does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..features import build_features, build_graph_features
from .dataset import Split, temporal_split
from .evaluate import (
    Headline,
    ablation_grid,
    evasion_delta,
    fairness_breakdown,
    false_positive_breakdown,
    guard_contribution,
    guard_leakage,
    headline_intervals,
    headline_metrics,
    label_shuffle_control,
    leave_one_vector_out,
    operating_curve,
    per_family_recall,
    per_vector_recall,
    score_lift_table,
    worst_slices,
)
from .baselines import baseline_comparison, expert_rule_firing
from .costs import cost_curve, cost_summary
from .deepfake_guard import DeepfakeGuard
from .explain import benign_baselines, counterfactual, phrase
from .injection_guard import (
    InjectionGuard,
    InjectionGuardReport,
    leave_one_family_out,
    unseen_phrasing_holdout,
)
from .vishing_guard import VishingGuard, held_out_script_recall, unseen_wording_recall
from .agent_controls import apply_controls, ceiling_loss_bound, control_coverage
from .intent_guard import apply_guard, coverage as intent_coverage
from .model import FraudDetector


@dataclass
class Prepared:
    """The feature frame and every fitted guard, before the tabular model sees any of it.

    Returned as a record rather than a tuple because the guard count has now grown past the
    point where positional unpacking at three call sites is safe to change.
    """

    featured: pd.DataFrame
    split: Split
    injection: Optional[InjectionGuard]
    injection_report: Optional[InjectionGuardReport]
    vishing: Optional[VishingGuard] = None
    media: Optional[DeepfakeGuard] = None

    def __iter__(self):
        # Kept unpackable so `df, split, guard, report = prepare(...)` still reads naturally
        # in the closed loop, which only ever wanted those four.
        return iter((self.featured, self.split, self.injection, self.injection_report))


@dataclass
class DefenceArtifacts:
    """Everything the defence produced, ready to be written to disk or reported."""

    detector: FraudDetector
    injection_guard: Optional[InjectionGuard]
    split: Split
    featured: pd.DataFrame
    scored: pd.DataFrame
    headline: Headline
    operating_curve: pd.DataFrame
    per_vector: pd.DataFrame
    per_family: pd.DataFrame
    false_positives: pd.DataFrame
    evasion: pd.DataFrame
    guards: pd.DataFrame
    intent_coverage: pd.DataFrame
    control_coverage: pd.DataFrame
    ceiling_bound: pd.DataFrame
    injection_report: Optional[InjectionGuardReport]
    lift: pd.DataFrame
    slices: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    zero_day: pd.DataFrame = field(default_factory=pd.DataFrame)
    ablation: pd.DataFrame = field(default_factory=pd.DataFrame)
    null_control: pd.DataFrame = field(default_factory=pd.DataFrame)
    guard_leakage: pd.DataFrame = field(default_factory=pd.DataFrame)
    injection_zero_shot: pd.DataFrame = field(default_factory=pd.DataFrame)
    injection_paraphrase: pd.DataFrame = field(default_factory=pd.DataFrame)
    vishing_report: Optional[Dict[str, object]] = None
    vishing_zero_shot: pd.DataFrame = field(default_factory=pd.DataFrame)
    vishing_unseen_wording: pd.DataFrame = field(default_factory=pd.DataFrame)
    media_report: Optional[Dict[str, object]] = None
    # What the ensemble is worth against simpler detectors, who carries the false-positive
    # burden, and what the operating point costs in money. None of the three changes the
    # model; all three change how defensible its numbers are.
    baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    expert_rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    fairness: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Legitimate-traffic distribution per input, so the inspector can show a field against
    #: what normal looks like instead of as a bare number.
    benign_baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The single change that would have flipped each of a sample of alerts. Precomputed
    #: rather than served on demand because deriving one needs the fitted model, and the
    #: interface has to answer "why this payment" with no model loaded and no network.
    counterfactuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    # The fitted objects, not just their scorecards. The serving bundle needs the guard
    # itself, and a run that reported a guard but could not ship it would be a report about
    # a system nobody can deploy.
    vishing_guard: Optional[VishingGuard] = None
    media_guard: Optional[DeepfakeGuard] = None

    def summary(self) -> Dict[str, object]:
        worst = self.per_vector.head(5)[["attack_vector_id", "recall"]].to_dict("records")
        return {
            "headline": self.headline.to_dict(),
            "weakest_vectors": worst,
            "injection_guard": self.injection_report.to_dict() if self.injection_report else None,
            "vishing_guard": self.vishing_report,
            "media_guard": self.media_report,
        }


def prepare(df: pd.DataFrame, corpus: pd.DataFrame, cfg: Config,
            *, verbose: bool = False,
            transcripts: Optional[pd.DataFrame] = None) -> "Prepared":
    """Feature engineering plus every guard, returning a model-ready frame and split."""
    if verbose:
        print("  features: tabular")
    df = build_features(df)
    if verbose:
        print("  features: account graph snapshots")
    df = build_graph_features(df)

    split = temporal_split(df, test_days=cfg.defence.test_days,
                           calibration_days=cfg.defence.calibration_days)

    guard: Optional[InjectionGuard] = None
    report: Optional[InjectionGuardReport] = None
    if not corpus.empty:
        if verbose:
            print("  guard: prompt-injection content classifier")
        guard = InjectionGuard(seed=cfg.seed)
        try:
            report = guard.fit(
                corpus,
                train_end=split.calibration["timestamp"].min(),
                test_start=split.test["timestamp"].min(),
            )
            df = guard.score_transactions(df, corpus)
        except ValueError:
            # Too few injected bundles in the training window to fit honestly. Better to
            # ship the feature at its schema default than to fit on the test window.
            guard, report = None, None
            df["agent_injection_flag"] = 0

    vishing: Optional[VishingGuard] = None
    if transcripts is not None and not transcripts.empty:
        if verbose:
            print("  guard: vishing and grooming transcript classifier")
        vishing = VishingGuard(seed=cfg.seed)
        try:
            vishing.fit(
                transcripts,
                train_end=split.calibration["timestamp"].min(),
                test_start=split.test["timestamp"].min(),
            )
            df = vishing.score_transactions(df, transcripts)
        except ValueError:
            vishing = None
    if "vishing_score" not in df.columns:
        df["vishing_score"] = 0.0
        df["vishing_flag"] = 0

    if verbose:
        print("  guard: cryptographic intent verification")
    df = apply_guard(df)

    if verbose:
        print("  guard: agent identity, token binding, spend ceiling")
    df = apply_controls(df)

    # Re-split now that guard columns exist, using the same time boundaries.
    split = temporal_split(df, test_days=cfg.defence.test_days,
                           calibration_days=cfg.defence.calibration_days)

    if verbose:
        print("  guard: synthetic-media artefact consistency")
    media = DeepfakeGuard(seed=cfg.seed).fit(split.train, split.test)
    if media.usable:
        df = media.score_transactions(df)
        split = temporal_split(df, test_days=cfg.defence.test_days,
                               calibration_days=cfg.defence.calibration_days)
    else:
        df["media_artefact_score"] = 0.0
        df["media_artefact_flag"] = 0
        split = temporal_split(df, test_days=cfg.defence.test_days,
                               calibration_days=cfg.defence.calibration_days)

    return Prepared(df, split, guard, report, vishing, media)


def run_defence(df: pd.DataFrame, corpus: pd.DataFrame, cfg: Config, *,
                verbose: bool = True, with_zero_day: bool = True,
                with_importance: bool = True,
                transcripts: Optional[pd.DataFrame] = None) -> DefenceArtifacts:
    prep = prepare(df, corpus, cfg, verbose=verbose, transcripts=transcripts)
    df, split, guard, report = prep.featured, prep.split, prep.injection, prep.injection_report

    if verbose:
        print("  model: stacked ensemble (boosting + forest + isolation)")
    detector = FraudDetector(
        seed=cfg.seed,
        target_fpr=cfg.defence.target_fpr,
        max_iter=cfg.defence.max_iter,
        n_estimators=cfg.defence.forest_n_estimators,
    ).fit(split.train, split.calibration)

    test = split.test
    scores = detector.predict_proba(test)
    y = test["is_fraud"].to_numpy().astype(int)
    amounts = test["amount"].to_numpy().astype(float)
    thr = detector.threshold

    scored = test[["txn_id", "timestamp", "amount", "rail", "is_fraud",
                   "attack_vector_id", "is_hard_negative"]].copy()
    # Carried through so a reader of the scored set can see *who* and *how*, not only how
    # much. An operations console listing amounts with no party or channel is a column of
    # numbers, and the channel in particular is what distinguishes an agent-initiated
    # payment from one a human tapped - which is the distinction most of this run is about.
    for column in ("payer_id", "payee_id", "channel", "initiated_by_agent"):
        if column in test.columns:
            scored[column] = test[column].to_numpy()
    scored["score"] = np.round(scores, 6)
    scored["model_alert"] = (scores >= thr).astype(int)
    scored["intent_block"] = test.get("intent_guard_blocked", 0)
    scored["control_block"] = test.get("agent_control_blocked", 0)
    scored["injection_flag"] = test.get("agent_injection_flag", 0)
    # The union of the model and every deterministic control that can prove its case. The
    # controls are in the decision rather than only in the report because a control nobody
    # acts on is documentation, and because each one stops a vector the model is weakest on.
    scored["decision"] = (
        (scored["model_alert"] == 1)
        | (scored["intent_block"] == 1)
        | (scored["control_block"] == 1)
    ).astype(int)

    # Always present, so a --shallow run writes the same schema as a deep one. Emitting the
    # column only when importance was computed meant two runs of the same pipeline produced
    # defend_scored_test_set.csv files that could not be compared or concatenated.
    scored["reason_codes"] = ""

    importance = pd.DataFrame()
    if with_importance:
        if verbose:
            print("  explain: permutation importance on the ensemble")
        sample = _stratified_sample(split.calibration, n=6000, seed=cfg.seed)
        importance = detector.global_importance(sample, n_repeats=2)
        alerts = scored["model_alert"] == 1
        scored.loc[alerts, "reason_codes"] = detector.reason_codes(test[alerts.to_numpy()], importance)

    artifacts = DefenceArtifacts(
        detector=detector,
        injection_guard=guard,
        split=split,
        featured=df,
        scored=scored,
        headline=headline_metrics(y, scores, amounts, thr),
        operating_curve=operating_curve(y, scores, amounts),
        per_vector=per_vector_recall(test, scores, thr),
        per_family=per_family_recall(test, scores, thr),
        false_positives=false_positive_breakdown(test, scores, thr),
        evasion=evasion_delta(test, scores, thr),
        guards=guard_contribution(test, scores, thr),
        guard_leakage=guard_leakage(test),
        intent_coverage=intent_coverage(test),
        control_coverage=control_coverage(test),
        ceiling_bound=ceiling_loss_bound(test),
        injection_report=report,
        lift=score_lift_table(y, scores, amounts),
        slices=worst_slices(test, scores, thr),
        intervals=headline_intervals(y, scores, cfg.defence.target_fpr, seed=cfg.seed),
        importance=importance,
        vishing_report=prep.vishing.report.to_dict()
                       if prep.vishing is not None and prep.vishing.report else None,
        media_report=prep.media.report.to_dict()
                     if prep.media is not None and prep.media.report else None,
        vishing_guard=prep.vishing,
        media_guard=prep.media,
        fairness=fairness_breakdown(test, scores, thr),
        benign_baselines=benign_baselines(split.train),
    )

    if verbose:
        print("  compare: one rule, expert rules, logistic regression")
    artifacts.baselines = baseline_comparison(split, scores,
                                              target_fpr=cfg.defence.target_fpr, seed=cfg.seed)
    artifacts.expert_rules = expert_rule_firing(test)

    if verbose:
        print("  economics: expected loss across the threshold range")
    artifacts.cost_curve = cost_curve(test, scores, thr)
    artifacts.cost_summary = cost_summary(artifacts.cost_curve)

    if verbose:
        print("  explain: counterfactuals for a sample of alerts")
    artifacts.counterfactuals = _counterfactual_table(
        detector, test, scores, thr, artifacts.benign_baselines, seed=cfg.seed,
    )

    # Half-size estimators for the retraining studies. Each fits a fresh model, and the point
    # of these tables is the comparison between rows rather than the absolute level.
    def _lean(**kwargs) -> FraudDetector:
        return FraudDetector(seed=cfg.seed, target_fpr=cfg.defence.target_fpr,
                             max_iter=max(120, cfg.defence.max_iter // 2),
                             n_estimators=max(80, cfg.defence.forest_n_estimators // 2),
                             **kwargs)

    if with_zero_day:
        if verbose:
            print("  stress: leave-one-vector-out retraining")
        artifacts.zero_day = leave_one_vector_out(
            split, _lean, max_vectors=cfg.defence.zero_day_vectors,
        )
        if verbose:
            print("  stress: feature-layer ablation grid")
        artifacts.ablation = ablation_grid(split, _lean, target_fpr=cfg.defence.target_fpr)
        if verbose:
            print("  control: label-shuffle null")
        artifacts.null_control = label_shuffle_control(
            split, _lean, target_fpr=cfg.defence.target_fpr, seed=cfg.seed
        )
        if not corpus.empty:
            artifacts.injection_zero_shot = leave_one_family_out(corpus, seed=cfg.seed)
            artifacts.injection_paraphrase = unseen_phrasing_holdout(corpus, seed=cfg.seed)
        if transcripts is not None and not transcripts.empty:
            if verbose:
                print("  stress: leave-one-scam-script-out on the transcript guard")
            artifacts.vishing_zero_shot = held_out_script_recall(transcripts, seed=cfg.seed)
            if prep.vishing is not None:
                if verbose:
                    print("  stress: held-out conversational vocabulary")
                artifacts.vishing_unseen_wording = unseen_wording_recall(
                    prep.vishing, seed=cfg.seed)

    return artifacts


#: How many alerts get a counterfactual. Each one costs a batch of predictions against the
#: full ensemble, so this is bounded by the demo's patience rather than by the data: a few
#: hundred is enough that a reviewer opening alerts at random keeps finding one.
COUNTERFACTUAL_SAMPLE = 400


def _counterfactual_table(detector: FraudDetector, test: pd.DataFrame, scores: np.ndarray,
                          threshold: float, baselines: pd.DataFrame, *,
                          seed: int) -> pd.DataFrame:
    """The smallest single change that would have cleared each of a sample of alerts.

    Precomputed here rather than derived on request. A counterfactual needs the fitted
    model, and the interface has to be able to answer "why was this stopped, and what would
    have changed it" with no model loaded, no server and no network - which is the situation
    the packaged demo is in. Computing them once, at the point where the model is already in
    memory, is also far cheaper than reloading it per click.

    Sampled across attack vectors rather than taken from the top of the file, so the alerts
    that have one are not all the same kind of attack. An alert with no row in the result is
    a real answer: no single actionable field would have flipped it, and the decision rests
    on a combination.
    """
    if baselines.empty or test.empty:
        return pd.DataFrame()

    alerts = test[scores >= threshold]
    if alerts.empty:
        return pd.DataFrame()

    if len(alerts) > COUNTERFACTUAL_SAMPLE:
        # Proportional within each vector, so a vector contributing 2% of alerts keeps
        # roughly 2% of the sample instead of being crowded out by the common ones.
        group = "attack_vector_id" if "attack_vector_id" in alerts.columns else None
        if group is not None:
            fraction = COUNTERFACTUAL_SAMPLE / len(alerts)
            alerts = (alerts.groupby(group, group_keys=False, dropna=False)
                      .apply(lambda g: g.sample(max(1, round(len(g) * fraction)),
                                                random_state=seed)))
        if len(alerts) > COUNTERFACTUAL_SAMPLE:
            alerts = alerts.sample(COUNTERFACTUAL_SAMPLE, random_state=seed)

    rows = []
    for _, alert in alerts.iterrows():
        one = alert.to_frame().T
        found = counterfactual(one, detector.predict_proba, threshold, baselines)
        if not found:
            # Recorded rather than omitted. "We looked and no single field would have
            # changed this" and "we never looked at this one" are different answers, and a
            # reader can only tell them apart if the first one leaves a trace.
            rows.append({"txn_id": alert["txn_id"], "column": None, "from": None, "to": None,
                         "score_after": None, "normalised_move": None, "phrase": None})
            continue
        for entry in found:
            rows.append({
                "txn_id": alert["txn_id"],
                "column": entry["column"],
                "from": entry["from"],
                "to": entry["to"],
                "score_after": entry["score_after"],
                "normalised_move": entry["normalised_move"],
                "phrase": phrase(entry),
            })

    return pd.DataFrame(rows)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep every fraud row, subsample the legitimate majority.

    Permutation importance on 90k rows is minutes of wall clock for information that is
    stable at a few thousand. Keeping all the fraud preserves the signal that matters.
    """
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    take = max(n - len(fraud), 500)
    if len(legit) > take:
        legit = legit.sample(take, random_state=seed)
    return pd.concat([fraud, legit]).sort_values("timestamp").reset_index(drop=True)
