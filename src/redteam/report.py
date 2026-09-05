"""Turning a run into artefacts a reader can audit.

Every table the pipeline produces is written as CSV next to a single markdown report that
narrates them. The markdown is generated from the same numbers, never hand-written, so it
cannot drift from what the code actually measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .defend.pipeline import DefenceArtifacts
from .genai.ideate import to_yaml_block
from .generate.fidelity import FidelityReport
from .identify.library import AttackLibrary, summarise
from .provenance import stamp

# Bumped whenever a key in results.json moves or changes meaning, so a consumer reading it
# can refuse a shape it does not understand instead of quietly finding None.
RESULTS_SCHEMA_VERSION = 1


def _joint_separability_note(joint: Dict[str, object]) -> str:
    """Narrate the check that per-feature leakage cannot make.

    Kept separate from the per-feature table because it answers a different question and is
    easy to misread as a duplicate of it: features can each look innocent and still stack.
    """
    if not joint:
        return ""
    recall = float(joint["recall_at_review_budget"])
    budget = float(joint["review_budget_fpr"])
    return (
        "\nPassing that check is necessary but not sufficient. Features that each look "
        "innocent can still stack, so a throwaway booster is trained on an earlier slice of "
        "this data and scored on a later one, before the real defence exists. It recovers "
        f"**{recall:.1%}** of fraud at a {budget:.1%} false-positive budget "
        f"(ROC AUC {float(joint['held_out_auc']):.4f} on "
        f"{int(joint['test_fraud_rows'])} held-out fraud rows). Recall at a fixed budget is "
        "quoted rather than AUC because at this class imbalance AUC flatters everything; "
        "deployed card-fraud systems report roughly 0.5-0.85 here, and above 0.92 this "
        "scorer flags its own data as too clean to draw conclusions from.\n"
    )


def _is_blank(value) -> bool:
    """Empty, or a mapping whose every value is None. See ``_write_results_json``."""
    if value is None or (isinstance(value, (list, dict, str)) and not value):
        return True
    return isinstance(value, dict) and all(v is None for v in value.values())


def _write(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value):
    """Replace non-finite floats with null, recursively.

    ``json.dumps`` emits bare ``NaN``, ``Infinity`` and ``-Infinity`` tokens by default.
    Python reads them back happily, which is why this went unnoticed, but they are not
    JSON - RFC 8259 has no such literals - and every strict parser rejects them, including
    the browser's ``JSON.parse``. A single ``NaN`` anywhere in the file therefore takes the
    whole artefact from "readable by any consumer" to "readable only by us", and these files
    exist precisely so that other things can read them.

    NaN is not rare here either: ``delta_recall`` is undefined on the first ablation row and
    ``recall_on_llm_written`` is undefined whenever the LLM cache ships empty, which is the
    default. Both are genuinely absent values, so null is also the more accurate encoding.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _dump_json(payload, path: Path) -> None:
    """Serialise with ``allow_nan=False`` so a future regression fails here, loudly.

    The sanitiser above already removes every non-finite value; the flag is what stops the
    next person adding a metric that quietly reintroduces one.
    """
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False),
                    encoding="utf-8")


def _fidelity_caveat(fidelity) -> str:
    """Say plainly when a check scored zero without failing the build.

    That combination is reachable and it is the most misleading state the report can be in:
    a check whose point estimate is past its ceiling but whose confidence interval straddles
    it scores zero and, correctly, raises no flag. A reader skimming for "flags" sees a clean
    run. On a full-size run this is exactly what happened, and the detection numbers
    underneath it were substantially the generator's.
    """
    warnings = list(getattr(fidelity, "warnings", []))
    zeroed = list(getattr(fidelity, "zeroed", []))
    if not warnings and not zeroed:
        return ""
    lines = ["> **Read the flags line with the next two facts beside it.**\n>\n"]
    for warning in warnings:
        lines.append(f"> - {warning}.\n")
    if zeroed:
        names = ", ".join(f"`{z}`" for z in zeroed)
        lines.append(
            f"> - {names} scored **zero**, the worst reading the check can return. A zero "
            "without a flag means the point estimate is past the ceiling while the "
            "confidence interval still straddles it: not provable, not clean.\n"
        )
    return "".join(lines) + ">\n\n"


def _md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "_no rows_\n"
    shown = frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "| " + " | ".join("---" for _ in shown.columns) + " |"
    body = [
        "| " + " | ".join(_cell(v) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    out = "\n".join([header, rule, *body])
    if len(frame) > max_rows:
        out += f"\n\n_{len(frame) - max_rows} further rows in the CSV._"
    return out + "\n"


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.2e}"
    return str(value).replace("|", "/")


class RunWriter:
    """Collects artefacts from each stage and renders the final report."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.artifacts_path / cfg.run_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.sections: List[str] = []
        self.provenance = stamp(cfg)
        self.stage_timings: Dict[str, float] = {}
        self.meta: Dict[str, object] = {
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "provenance": self.provenance.to_dict(),
        }

    def record_timing(self, stage: str, seconds: float) -> None:
        """Per-stage wall clock, so the latency claim in the write-up has a source."""
        self.stage_timings[stage] = round(float(seconds), 3)
        self.meta["stage_timings_seconds"] = dict(self.stage_timings)

    # -- stages ------------------------------------------------------------
    def identify(self, library: AttackLibrary, problems: List[str]) -> None:
        summary = summarise(library)
        self.meta["identify"] = summary
        coverage = pd.DataFrame(library.coverage_rows())
        _write(coverage, self.root / "identify_vectors.csv")

        families = pd.DataFrame(
            [{"family": k, **v} for k, v in summary["families"].items()]
        ).sort_values("total", ascending=False)
        _write(families, self.root / "identify_families.csv")

        signals = pd.DataFrame(
            [{"signal": k, "vectors_using_it": v} for k, v in library.signal_usage().items()]
        )
        _write(signals, self.root / "identify_signal_usage.csv")

        top = coverage.sort_values("risk_score", ascending=False).head(12)[
            ["id", "family", "risk_score", "severity", "detection_difficulty", "simulated"]
        ]
        self.sections.append(
            "## Identify\n\n"
            f"The taxonomy holds **{summary['total_vectors']} distinct attack vectors** across "
            f"{len(summary['families'])} families, of which **{summary['simulated_vectors']} are "
            "wired to a generator and appear in the data**. Every vector declares the observable "
            "signals it would leave in the transaction schema, and the loader rejects any signal "
            "that is not a real column, so the taxonomy cannot describe evidence the pipeline "
            "could not actually produce.\n\n"
            f"Validation: **{'no problems' if not problems else str(len(problems)) + ' problems'}**.\n\n"
            "Vectors by family:\n\n" + _md_table(families) + "\n"
            "Highest-risk vectors (0.4x severity + 0.25x prevalence + 0.35x detection difficulty):\n\n"
            + _md_table(top)
        )

    def ideate(self, result, cache_stats=None) -> None:
        """The LLM ideation agent's proposals, accepted and rejected alike."""
        table = result.table()
        _write(table, self.root / "identify_llm_proposals.csv")
        summary = result.summary()
        self.meta["ideate"] = dict(summary)
        if cache_stats is not None:
            self.meta["ideate"]["llm_cache"] = cache_stats.to_dict()

        accepted = result.accepted
        yaml_block = to_yaml_block(accepted) if accepted else ""
        if yaml_block:
            (self.root / "identify_llm_proposals.yaml").write_text(yaml_block,
                                                                  encoding="utf-8")

        self.sections.append(
            "## Identify: the ideation agent\n\n"
            "The taxonomy above is hand-authored, which is a respectable artefact and not an "
            "AI system. This stage makes ideation something the machine runs: the model is "
            "shown the schema, the existing taxonomy and the families with thin coverage, and "
            "asked for vectors that are not already there.\n\n"
            "The prompt is not the interesting part - the gate is. Every proposal is parsed "
            "into an `AttackVector` and put through `AttackLibrary.validate()`, the same check "
            "the committed library passes. Acceptance requires it to be well-formed, novel, "
            "and to reference **signals that exist in the schema**. That last condition does "
            "most of the work: it separates a vector this system can simulate and measure from "
            "a paragraph of plausible threat prose. A model that invents "
            "`deepfake_confidence_score` is rejected, because no column carries it.\n\n"
            f"**Proposed {summary['proposed']}, accepted {summary['accepted']}, "
            f"rejected {summary['rejected']} - an accept rate of "
            f"{summary['accept_rate']:.0%}.** The rejections are published because an accept "
            "rate computed over only the acceptances is 1.0 by construction:\n\n"
            + _md_table(table, max_rows=12) + "\n"
            + ("Accepted proposals are written to `identify_llm_proposals.yaml` for review "
               "rather than merged automatically, and are marked `simulated: false`. An agent "
               "that both proposes vectors and marks them simulated would grow the headline "
               "count without anything having been built.\n"
               if yaml_block else "")
        )

    def generate(self, txns: pd.DataFrame, corpus: pd.DataFrame, per_vector: pd.DataFrame,
                 fidelity: FidelityReport,
                 transcripts: Optional[pd.DataFrame] = None,
                 judged=None) -> None:
        _write(per_vector, self.root / "generate_per_vector.csv")
        details = fidelity.details

        scores = pd.DataFrame(
            [{"check": k, "score": v} for k, v in fidelity.scores.items()]
        ).sort_values("score")
        _write(scores, self.root / "generate_fidelity_scores.csv")

        leakage = pd.DataFrame(details.get("top_single_feature_auc", []),
                               columns=["feature", "single_feature_auc"])
        _write(leakage, self.root / "generate_single_feature_auc.csv")

        self.meta["generate"] = {
            "transactions": int(len(txns)),
            "fraud": int(txns["is_fraud"].sum()),
            "fraud_rate": round(float(txns["is_fraud"].mean()), 5),
            "vectors_present": int(txns.loc[txns["is_fraud"] == 1, "attack_vector_id"].nunique()),
            "agent_context_bundles": int(len(corpus)),
            "fidelity": fidelity.overall,
            "flags": fidelity.flags,
            "warnings": getattr(fidelity, "warnings", []),
            "zeroed_checks": getattr(fidelity, "zeroed", []),
            # Both separability probes, so the raw-versus-derived gap that motivated most of
            # the generator work is machine-readable rather than buried in the narrative.
            "joint_separability_recall": (details.get("joint_separability") or {}).get(
                "recall_at_review_budget"),
            "derived_separability_recall": (details.get("derived_separability") or {}).get(
                "recall_at_review_budget"),
            "transcripts": int(len(transcripts)) if transcripts is not None else 0,
            "llm_judge": judged.to_dict() if judged else None,
        }

        judge_note = ""
        if judged:
            _write(judged.table(), self.root / "generate_llm_judge.csv")
            j = judged.to_dict()
            judge_note = (
                "\n### Can a reader tell these rows are synthetic?\n\n"
                "Every check above is a statistic the generator could be tuned to satisfy, "
                "and a generator that satisfies all of them can still emit rows that are "
                "obviously fake to anyone who has looked at payment data, because the "
                "implausibility lives in the *combination* rather than in any marginal. So a "
                "language model was shown "
                f"{j['rows_judged']} unlabelled legitimate rows and asked to pick out the "
                "synthetic ones. Chance is 50% and chance is the target; it scored "
                f"**{j['discriminator_accuracy']:.0%}**, calling "
                f"{j['share_called_synthetic']:.0%} of them synthetic. Verdict: "
                f"{j['verdict']}.\n\n"
                "Two caveats, because this measurement is easy to overclaim. There is no "
                "real payment data in this repository, so the model judges against a "
                "description of Indian retail payment traffic drawn from published NPCI and "
                "RBI aggregates rather than against held-out reality - this measures "
                "plausibility to a knowledgeable reader, not distributional truth. And only "
                "legitimate rows are shown, since fraud is *supposed* to look unusual and "
                "including it would turn a fidelity check into a detection one.\n"
                + ("\nThe tells it cited:\n\n"
                   + "\n".join(f"- {t}" for t in j["tells_cited"]) + "\n"
                   if j["tells_cited"] else "")
            )

        transcript_note = ""
        if transcripts is not None and not transcripts.empty:
            coercive = int(transcripts["is_coercive"].sum())
            llm_share = float((transcripts["source"] == "llm").mean())
            transcript_note = (
                f"\nA third plane carries **{len(transcripts):,} call and chat transcripts** "
                f"({coercive:,} coercive, the rest ordinary conversations about money). This "
                "is the only plane that sees the *cause* of an authorised push payment rather "
                "than its consequences: the tabular model's entire evidence that somebody was "
                "on the telephone is a single `call_in_progress` bit, which the closed loop "
                "shows an attacker can simply stop emitting. A conversation is much harder to "
                "sanitise, because the scam has to establish authority, manufacture urgency, "
                "discourage verification and redirect the money, and an operator who drops "
                "those has no scam left. "
                f"{llm_share:.0%} of transcripts are model-written rather than composed from "
                "templates, so the guard's numbers are not a measurement of template "
                "coverage. Both classes are assembled the same way, and the legitimate half "
                "deliberately contains deadlines, first-time payees and large amounts.\n"
            )

        overlap = details.get("fraud_overlap", {})
        self.sections.append(
            "## Generate\n\n"
            f"**{len(txns):,} transactions** over {self.cfg.benign.n_days} days across "
            f"{len(self.cfg.benign.rail_mix)} payment rails, of which "
            f"**{int(txns['is_fraud'].sum()):,} are fraudulent** "
            f"({txns['is_fraud'].mean():.2%}), spanning "
            f"{txns.loc[txns['is_fraud'] == 1, 'attack_vector_id'].nunique()} distinct vectors. "
            f"A second data plane carries **{len(corpus):,} agent context bundles** - the "
            "catalogue text, tool descriptions and reviews an agent read on its way to "
            "checkout.\n"
            + transcript_note
            + "\n"
            f"Composite fidelity score: **{fidelity.overall}**"
            + (f" (flags: {', '.join(fidelity.flags)})" if fidelity.flags else " (no flags)")
            + ".\n\n"
            + _fidelity_caveat(fidelity)
            + _md_table(scores)
            + "\nThe single most important realism check is that no individual feature gives "
            "the game away. If one column separated fraud from legitimate traffic, every "
            "detection number downstream would be measuring the generator rather than the "
            "model.\n\n"
            + _md_table(leakage, max_rows=10)
            + _joint_separability_note(details.get("joint_separability", {}))
            + f"\nFraud also has to overlap legitimate traffic where it should: "
            f"**{overlap.get('quiet_fraud_share', float('nan')):.1%}** of fraudulent payments "
            "emit no loud tell at all, and the amount distributions of fraud and legitimate "
            "traffic differ by a KS statistic of only "
            f"{overlap.get('amount_ks_statistic', float('nan')):.3f}.\n"
            + judge_note
        )

    @staticmethod
    def _ablation_note(grid: pd.DataFrame) -> str:
        """Say something when a layer costs recall instead of adding it.

        A negative row is a real result and the temptation is to leave it as a number in a
        table and move on. It is worth a sentence, because the reason is not obvious and the
        alternative reading - that the layer is worthless - is wrong.
        """
        losing = grid[grid["delta_recall"] < -0.01] if "delta_recall" in grid else grid.iloc[0:0]
        if losing.empty:
            return ""
        names = ", ".join(f"`{r}`" for r in losing["layer"])
        return (
            f"\nOne layer costs recall here rather than adding it ({names}), and it is "
            "published as it came out. Columns that are near-constant on a test window this "
            "small buy the ensemble more variance than signal. That is not the same as the "
            "layer being worthless: the deterministic guards contribute through the "
            "*decision*, as the union of a model alert and a provable block, and this grid "
            "measures the model alone. It does mean their value on this sample size is in "
            "the reasons they give rather than in the recall they add.\n"
        )

    def defend(self, artifacts: DefenceArtifacts, narratives=None) -> None:
        head = artifacts.headline
        self.meta["defend"] = artifacts.summary()
        self.meta["alert_narratives"] = narratives.to_dict() if narratives else None
        self.meta["defend_intervals"] = artifacts.intervals.to_dict("records")
        self.meta["ablation"] = artifacts.ablation.to_dict("records")
        self.meta["guard_leakage"] = artifacts.guard_leakage.to_dict("records")
        self.meta["null_control"] = artifacts.null_control.to_dict("records")

        _write(artifacts.split.describe(), self.root / "defend_split.csv")
        _write(artifacts.operating_curve, self.root / "defend_operating_curve.csv")
        _write(artifacts.per_vector, self.root / "defend_per_vector_recall.csv")
        _write(artifacts.per_family, self.root / "defend_per_family_recall.csv")
        _write(artifacts.false_positives, self.root / "defend_false_positives.csv")
        _write(artifacts.evasion, self.root / "defend_evasion_delta.csv")
        _write(artifacts.guards, self.root / "defend_guard_layers.csv")
        _write(artifacts.intent_coverage, self.root / "defend_intent_coverage.csv")
        _write(artifacts.control_coverage, self.root / "defend_agent_control_coverage.csv")
        _write(artifacts.ceiling_bound, self.root / "defend_scoped_token_loss_bound.csv")
        _write(artifacts.lift, self.root / "defend_decile_lift.csv")
        _write(artifacts.importance, self.root / "defend_feature_importance.csv")
        _write(artifacts.zero_day, self.root / "defend_zero_day.csv")
        _write(artifacts.intervals, self.root / "defend_headline_intervals.csv")
        _write(artifacts.slices, self.root / "defend_worst_slices.csv")
        _write(artifacts.ablation, self.root / "defend_ablation_grid.csv")
        _write(artifacts.guard_leakage, self.root / "defend_guard_leakage.csv")
        _write(artifacts.null_control, self.root / "defend_null_control.csv")
        _write(artifacts.injection_zero_shot, self.root / "defend_injection_unseen_family.csv")
        _write(artifacts.injection_paraphrase, self.root / "defend_injection_unseen_phrasing.csv")
        _write(artifacts.vishing_zero_shot, self.root / "defend_vishing_unseen_script.csv")
        _write(artifacts.vishing_unseen_wording,
               self.root / "defend_vishing_unseen_wording.csv")
        _write(artifacts.scored, self.root / "defend_scored_test_set.csv")
        _write(artifacts.counterfactuals, self.root / "defend_counterfactuals.csv")
        _write(artifacts.baselines, self.root / "defend_baselines.csv")
        _write(artifacts.expert_rules, self.root / "defend_expert_rules.csv")
        _write(artifacts.fairness, self.root / "defend_fairness.csv")
        _write(artifacts.cost_curve, self.root / "defend_cost_curve.csv")
        _write(artifacts.cost_summary, self.root / "defend_cost_summary.csv")
        _write(artifacts.benign_baselines, self.root / "defend_benign_baselines.csv")

        headline = pd.DataFrame([{
            "metric": k, "value": v} for k, v in head.to_dict().items()])
        _write(headline, self.root / "defend_headline.csv")

        text = [
            "## Defend\n",
            f"Trained on the first {len(artifacts.split.train):,} transactions, calibrated on the "
            f"next {len(artifacts.split.calibration):,} and evaluated on the final "
            f"{len(artifacts.split.test):,}. The split is by wall-clock time, never at random: "
            "fraud arrives in campaigns, and a random split puts half a mule ring in training "
            "and half in test, which turns the score into a measurement of nothing.\n",
            _md_table(artifacts.split.describe()),
            f"\nAt an operating point of {self.cfg.defence.target_fpr:.2%} false positives, the "
            f"ensemble catches **{head.recall:.1%} of fraudulent payments** and "
            f"**{head.value_recall:.1%} of the money at risk**, at a realised false-positive rate "
            f"of {head.false_positive_rate:.3%} and {head.alerts_per_10k:.0f} alerts per 10,000 "
            f"payments. PR AUC is {head.pr_auc:.4f} and partial AUC over the usable 0-2% "
            f"false-positive region is {head.partial_auc_2pct:.4f}.\n",
            f"\nThe full ROC AUC is {head.roc_auc:.4f}, and it is reported here only because "
            "its absence would be conspicuous. At 0.75% prevalence it integrates over "
            "operating points that would alert on one payment in three, so most of its area "
            "comes from a region no institution would deploy, and it reads near 1.0 for models "
            "that differ substantially where it matters. The partial AUC above, and the "
            "operating curve below, are the numbers to judge this on.\n",
            "\nBecause the alert queue is a fixed resource, the useful view is recall across "
            "the false-positive budgets a team might actually be given:\n",
            _md_table(artifacts.operating_curve),
            "\n### Where the defence is weakest\n",
            "The average hides the holes. These are the vectors with the lowest recall, and "
            "they are the ones that feed the next round of red-team work. Every rate carries a "
            "95% Wilson interval, and `n_sufficient = 0` marks a cell computed on fewer than "
            "20 examples - a recall of 1.0000 on three rows is arithmetic, not evidence, and "
            "those rows are kept visible rather than deleted because a thin cell is itself a "
            "fact about the generator's mix:\n",
            _md_table(artifacts.per_vector.head(10)),
            "\n### False positives\n",
            "Hard negatives are legitimate payments engineered to look alarming - a genuine "
            "first large transfer to a new payee, made at night. They are where customer harm "
            "actually happens, so they are reported separately from ordinary traffic:\n",
            _md_table(artifacts.false_positives, max_rows=12),
        ]

        if not artifacts.slices.empty:
            text += [
                "\n### Automatically mined weak slices\n",
                "The per-vector table answers which *attack* gets through. This answers which "
                "*kind of payment* gets through, whatever produced it, by searching every "
                "rail, channel, amount band and payee-age band and every pair of them for the "
                "worst recall on at least 25 examples. It matters because an attacker who "
                "finds a soft slice does not need a new vector - they only need to route "
                "existing fraud through it:\n",
                _md_table(artifacts.slices),
            ]

        if not artifacts.intervals.empty:
            text += [
                "\n### How much of this is noise\n",
                "Every headline figure is a point estimate on one test window from one seed. "
                "These are 1,000-resample stratified bootstrap intervals, with the threshold "
                "re-derived inside each resample - holding it fixed would pretend the operating "
                "point were known exactly and report a narrower interval than the evidence "
                "supports:\n",
                _md_table(artifacts.intervals),
            ]

        if not artifacts.ablation.empty:
            text += [
                "\n### What each feature layer actually buys\n",
                "The obvious objection to a high score on synthetic data is that the generator "
                "is leaking through the features. The only honest answer is to remove them and "
                "publish the result. Each row retrains the full ensemble seeing only that "
                "layer's columns; the features are still built from the whole stream, and only "
                "the model's view narrows, because trimming the frame instead would change the "
                "derived columns and measure a different question:\n",
                _md_table(artifacts.ablation),
                "\n`raw_schema` is what a bank already has in the payment message itself. "
                "Everything above it is work this repository does, and the `delta_recall` "
                "column is the size of that contribution.\n",
                self._ablation_note(artifacts.ablation),
            ]

        if not artifacts.guard_leakage.empty:
            text += [
                "\n### Are the guard outputs features, or are they the label?\n",
                "This table exists because the distinction failed once, silently, and reached "
                "a published headline of 100% recall. A transcript classifier fitted on a "
                "corpus one person wrote separates that corpus almost perfectly, so its score "
                "is close to binary; if the transcripts themselves are then only ever attached "
                "to fraud, the tabular model inherits a column that *is* the answer. The "
                "fidelity probes cannot see this, because they zero-fill guard outputs by "
                "design - guards run long after generation.\n",
                "So each guard column is audited on its own. `single_feature_auc` asks whether "
                "the score separates fraud unaided. `presence_lift` asks the sharper question: "
                "how much more likely a payment is to be fraudulent purely because the column "
                "is populated at all. A guard whose telemetry only ever appears on fraud is a "
                "label wearing a feature's name, however good the classifier behind it is:\n",
                _md_table(artifacts.guard_leakage),
                "\nThe deterministic controls sit at the top of this table by construction and "
                "that is not a defect: `intent_guard_blocked` fires only when a presented "
                "artefact contradicts the settlement request, which cannot happen on "
                "legitimate traffic. A high `presence_lift` on a *content* guard is the one to "
                "worry about.\n",
            ]

        if not artifacts.null_control.empty:
            text += [
                "\n### The null control\n",
                "The same pipeline, refit on shuffled labels. If any part of the feature "
                "construction, the temporal split or the threshold calibration were quietly "
                "conditioning on the label, this row would beat chance and no amount of "
                "reasoning about individual features would have revealed it. Recall should "
                "land near the false-positive budget and PR AUC near prevalence:\n",
                _md_table(artifacts.null_control),
            ]

        if not artifacts.guards.empty:
            text += [
                "\n### The three layers on agentic traffic\n",
                "The layers are not interchangeable. The intent guard is deterministic and "
                "produces a reason an investigator can defend in a dispute; the injection guard "
                "reads content; the model reads behaviour. Vectors where the injection guard "
                "scores zero are the ones that carry no adversarial text at all - a counterfeit "
                "storefront, a replayed token, a spoofed crawler identity - and no content "
                "classifier will ever catch them:\n",
                _md_table(artifacts.guards),
            ]

        if not artifacts.intent_coverage.empty:
            text += [
                "\n### What the intent guard declines, and what it only flags\n",
                "A decline requires a presented artefact to contradict the settlement request, "
                "so the block rate on the `(legitimate)` row is zero by construction. A missing "
                "artefact is a different thing: it describes an attacker, but it also describes "
                "every agent integration that predates the protocol, so it raises friction and "
                "feeds the model instead of declining on its own:\n",
                _md_table(artifacts.intent_coverage, max_rows=12),
            ]

        if not artifacts.control_coverage.empty:
            text += [
                "\n### The three delegated-authority controls, measured separately\n",
                "The intent guard verifies *what the user asked for*, and its coverage table "
                "above reports 0% against three vectors for one reason: their intent chains "
                "are perfectly valid. These controls verify *who is asking* and *what the "
                "credential is good for* instead.\n",
                "Reported per control rather than in aggregate, because the useful finding is "
                "that they barely overlap. Web Bot Auth catches the impersonator, which asserts "
                "an enrolled identity and cannot sign for it. Token binding catches the replay. "
                "Scope catches the confused deputy and the poisoned tool. The counterfeit "
                "storefront is untouched by all three, because every credential it is handed "
                "is genuine - and that row is the honest boundary of this whole approach:\n",
                _md_table(artifacts.control_coverage, max_rows=14),
            ]

        if not artifacts.ceiling_bound.empty:
            text += [
                "\n### What a per-transaction spend cap is worth\n",
                "A cap detects nothing. Once the agent is genuinely hijacked and the token it "
                "presents is genuinely its own, verification has nothing left to check and the "
                "only remaining question is how much authority the credential carried. A "
                "standing delegation is worth the account; a per-purchase scoped token is "
                "worth the cap. Reported as exposure rather than as a detection rate, with the "
                "share of legitimate payments the cap would truncate published beside it:\n",
                _md_table(artifacts.ceiling_bound),
            ]

        if artifacts.injection_report is not None:
            r = artifacts.injection_report
            text += [
                "\n### Prompt-injection guard\n",
                f"In-distribution: ROC AUC {r.roc_auc:.4f}, recall {r.recall_at_1pct_fpr:.1%} at a "
                "1% false-positive budget. **That number should not be believed on its own.** "
                "Payloads are generated from templates, so a bag-of-n-grams model can memorise "
                "the exact strings. The measurement that matters is recall on phrasings held out "
                "of training entirely:\n",
                _md_table(artifacts.injection_paraphrase),
                "\nAnd on an entire payload family withheld from training:\n",
                _md_table(artifacts.injection_zero_shot),
            ]

        if artifacts.vishing_report:
            v = artifacts.vishing_report
            text += [
                "\n### Vishing and grooming guard\n",
                "The only layer that reads the *cause* of an authorised push payment. Every "
                "tabular feature for coercion - `call_in_progress`, `screen_share_active`, the "
                "hesitation counters - is a proxy for a conversation, and each is one bit an "
                "operator can choose not to leave behind; suppressing that telemetry is among "
                "the cheapest levers the closed loop finds. The conversation itself resists "
                "that, because the scam has to establish authority, manufacture urgency, "
                "discourage verification and redirect the money in order to work at all.\n",
                f"\nROC AUC {v['roc_auc']:.4f}, recall {v['recall_at_1pct_fpr']:.1%} at a 1% "
                "false-positive budget on genuine conversations. **Those numbers are not "
                "evidence of field performance and should not be read as any.** The reason "
                "is structural rather than fixable: one author wrote both classes of this "
                "corpus, so the boundary between them is perfectly consistent by "
                "construction, and a classifier recovers a perfectly consistent boundary "
                "exactly. Several corpus artefacts that would have produced the same score "
                "for worse reasons were found and removed - disjoint openers, customer-side "
                "turns that appeared only in scam calls, a length gap between the classes - "
                "and the score did not move, which is itself the finding.\n",
                "\nWhat the evaluations below *can* establish is the narrower claim that the "
                "guard is not simply memorising strings. Each withholds something and asks "
                "whether recall survives without it.\n",
                "\nThe false-positive budget is stated on genuine calls rather than on all "
                "traffic deliberately. Every negative is a near-twin of a scam: a real "
                "fraud-team callback against an impersonated one, a real courier against a "
                "redelivery scam, a genuine relative asking for help, a legitimate first "
                "payment to a builder who has changed banks. A guard that could not separate "
                "those would flag the bank's own outbound calls, and the cost of that is not "
                "analyst time, it is customers who stop answering the phone.\n",
            ]
            if not artifacts.vishing_unseen_wording.empty:
                text += [
                    "\nThe strongest of them re-renders both classes in a vocabulary held "
                    "out of training entirely - different openers, different coercion "
                    "phrasings, different benign scenarios, different customer replies - and "
                    "re-derives the threshold on the unseen negatives, since an unfamiliar "
                    "genuine call is as likely to surprise the model as an unfamiliar scam:\n",
                    _md_table(artifacts.vishing_unseen_wording),
                ]
            if not artifacts.vishing_zero_shot.empty:
                text += [
                    "\nAnd each row here withholds an entire scam pretext from training, "
                    "which is the offline analogue of a new social-engineering story "
                    "appearing. This one is the weaker test of the two, because the coercion "
                    "vocabulary is shared across pretexts and survives the holdout intact:\n",
                    _md_table(artifacts.vishing_zero_shot),
                ]
            wrote_llm = v["recall_on_llm_written"] == v["recall_on_llm_written"]
            provenance = (
                f"Recall on the model-written share of the corpus is "
                f"{v['recall_on_llm_written']:.0%}, against "
                f"{v['recall_on_template_written']:.0%} on the template-composed share."
                if wrote_llm else
                "No model-written transcripts were available in this run, because the LLM "
                "cache ships empty."
            )
            text += [
                f"\n{provenance} Text from a genuinely different author is the only thing "
                "that would turn this section's numbers into a measurement, and populating "
                "`redteam.genai.transcript_bank` is the single highest-value thing that "
                "could be done to this plane. Until then, treat the transcript guard as a "
                "demonstrated *mechanism* - a third data plane, wired end to end into the "
                "feature matrix and the decision - and not as a demonstrated accuracy.\n",
            ]

        if artifacts.media_report:
            m = artifacts.media_report
            text += [
                "\n### Synthetic-media artefact scoring\n",
                "**This is not a deepfake detector, and the distinction matters.** A real one "
                "operates on media - spectral artefacts, frame inconsistency, rPPG residuals - "
                "and this system has no media, only the scores a capture SDK returned. What "
                "this layer does is sit above such a stack and ask whether a capture's "
                "outputs are jointly consistent with genuine ones: a poor real capture "
                "degrades liveness and match together, while a cloned voice presented against "
                "a real liveness check does not.\n",
                f"\nFitted on {m['fitted_on_genuine_captures']:,} genuine captures from the "
                "training window only, with no fraud label at any point, so it cannot have "
                "memorised which vectors are attacks. It covers the "
                f"{m['capture_coverage']:.1%} of payments that carry any biometric capture at "
                "all and says nothing about the rest; recall on that covered slice is "
                f"{m['recall_at_1pct_fpr']:.1%} at a 1% false-positive budget on genuine "
                "captures. Against an anti-forensically generated capture whose vendor scores "
                "look ordinary it will degrade, which is the correct behaviour to report "
                "rather than to hide.\n",
            ]

        if not artifacts.baselines.empty:
            deployed = artifacts.baselines[
                artifacts.baselines["model"] == "ensemble (deployed)"]
            simpler = artifacts.baselines[
                (artifacts.baselines["model"] != "ensemble (deployed)")
                & (artifacts.baselines["comparable_budget"] == 1)]
            gap = ""
            if not deployed.empty and not simpler.empty:
                best = simpler.iloc[0]
                margin = float(deployed["recall"].iloc[0]) - float(best["recall"])
                gap = (
                    f"\nThe closest comparable baseline is **{best['model']}** at "
                    f"{best['recall']:.1%}, so the ensemble is worth **{margin:+.1%} recall** "
                    "over it at the same false-positive budget. Read that margin rather than "
                    "the absolute number: on synthetic data the absolute level is a property "
                    "of the generator, and only the difference between detectors on identical "
                    "rows says anything about the detectors.\n"
                )
            text += [
                "\n### Against simpler detectors\n",
                "The same test window, the same false-positive budget, scored by things a "
                "bank either already has or could build in an afternoon. `comparable_budget` "
                "marks the rows that actually reached the target false-positive rate - a "
                "binary rule fires at whatever rate it fires at, and comparing its recall "
                "against a model tuned to a budget it never met would be a rigged "
                "comparison:\n",
                _md_table(artifacts.baselines[[
                    "model", "recall", "recall_lo95", "recall_hi95", "precision",
                    "value_recall", "alerts_per_10k", "realised_fpr", "comparable_budget",
                    "n_features"]]),
                gap,
            ]
            if not artifacts.expert_rules.empty:
                text += [
                    "\nThe rule set itself, rule by rule, on fraud and on legitimate "
                    "traffic. A rule with lift near 1.0 is contributing alert volume and "
                    "nothing else:\n",
                    _md_table(artifacts.expert_rules),
                ]

        if not artifacts.fairness.empty:
            fair = artifacts.fairness
            burden = ""
            ages = fair[(fair["dimension"] == "age_band") & (fair["legitimate_rows"] > 0)]
            if len(ages) > 1:
                worst = ages.loc[ages["fp_rate"].idxmax()]
                best = ages.loc[ages["fp_rate"].idxmin()]
                ratio = (float(worst["fp_rate"]) / float(best["fp_rate"])
                         if float(best["fp_rate"]) > 0 else float("nan"))
                burden = (
                    f"\nThe **{worst['segment']}** band carries a false-positive rate of "
                    f"{worst['fp_rate']:.3%} against {best['fp_rate']:.3%} for "
                    f"**{best['segment']}**"
                    + (f", a factor of {ratio:.1f}" if ratio == ratio else "")
                    + f". Fraud aimed at {worst['segment']} is detected at "
                    f"{worst['recall']:.1%}. Whether that trade is acceptable is a policy "
                    "question and not a modelling one, but it cannot be answered by anyone "
                    "who has not been shown the two numbers together.\n"
                )
            text += [
                "\n### Who carries the friction\n",
                "One global threshold does not fall equally on everyone. The signals that "
                "identify a scam - hesitation, a slow form fill, a first transfer to a new "
                "payee, a phone call in progress - are also what an unconfident customer "
                "making an ordinary payment looks like, and those customers are the ones "
                "the scams target. `fp_rate` is the burden, `recall` is the protection, and "
                "`victimisation_rate` is how much the cohort is attacked in the first "
                "place:\n",
                _md_table(fair[[
                    "dimension", "segment", "legitimate_rows", "fp_rate", "fp_rate_lo95",
                    "fp_rate_hi95", "fraud_rows", "victimisation_rate", "recall",
                    "recall_n_sufficient"]]),
                burden,
            ]

        if not artifacts.cost_summary.empty:
            summary_rows = artifacts.cost_summary.set_index("metric")["value"]
            net = float(summary_rows.get("net_benefit_at_deployed", float("nan")))
            constraint = float(summary_rows.get("cost_of_budget_constraint", float("nan")))
            text += [
                "\n### What the operating point is worth\n",
                "A threshold is a business decision, so it is priced here rather than "
                "described. Costs come from the same `DefenderCosts` price list the "
                "co-evolution loop optimises against, so the two halves of this report "
                "cannot quietly disagree about what an alert costs. Every constant in it is "
                "an assumption and is printed alongside the result:\n",
                _md_table(artifacts.cost_summary),
                f"\nAgainst approving every payment and reimbursing the losses, the deployed "
                f"operating point is worth **INR {net:,.0f}** on this test window. The "
                f"loss-minimising threshold would save a further INR {constraint:,.0f}, "
                "which is the price of running the alert queue at a fixed size instead of "
                "at the size the arithmetic prefers. Both figures inherit every assumption "
                "in the price list above; substitute an institution's own numbers and they "
                "move.\n",
            ]

        if not artifacts.zero_day.empty:
            text += [
                "\n### Zero-day proxy\n",
                "Each row retrains the whole ensemble with one attack vector removed from "
                "training, then measures what it catches with no labelled example of it. This is "
                "the closest offline analogue of a genuinely novel attack, and anything caught "
                "here is caught by generalisable structure rather than memorisation:\n",
                _md_table(artifacts.zero_day),
            ]

        if not artifacts.evasion.empty:
            text += [
                "\n### Cost of evasion\n",
                "Recall on adversarially tuned fraud against the untuned version of the same "
                "vectors:\n",
                _md_table(artifacts.evasion, max_rows=12),
            ]

        if not artifacts.importance.empty:
            text += [
                "\n### What the model is using\n",
                "Permutation importance measured against the ensemble output, not any single "
                "learner:\n",
                _md_table(artifacts.importance.head(15)),
            ]

        if narratives:
            _write(narratives.table(), self.root / "defend_alert_narratives.csv")
            text += [
                "\n### What the analyst actually reads\n",
                "Reason codes are correct, auditable and nearly unreadable at queue speed - "
                "`f_payer_amount_sum_1h, payee_account_age_days, f_payer_new_payees_7d` is "
                "accurate and tells a reviewer working three hundred alerts a shift almost "
                "nothing. These summaries are generated from the reason codes and the "
                "payment's raw facts, for the top of the queue:\n",
                _md_table(narratives.table().drop(columns=["txn_id"]), max_rows=8),
                "\nGenerated **after** the decision and incapable of changing it. If the "
                "model writing them hallucinates, the score, the alert and the reason codes "
                "beside them are all unaffected - which is why this is the one place in the "
                "system where a language model is safe to put in the path at all, and why "
                "the machine-checkable reason codes are stored next to the prose rather "
                "than replaced by it.\n",
            ]

        self.sections.append("\n".join(text))

    def loop(self, result) -> None:
        from .loop.economics import AttackerCosts, DefenderCosts

        summary = result.summary()
        _write(summary, self.root / "loop_rounds.csv")
        _write(result.tactics, self.root / "loop_tactics.csv")
        _write(result.blue_options, self.root / "loop_blue_moves.csv")
        _write(result.control_gaps, self.root / "loop_control_gaps.csv")
        _write(result.transfer.reset_index() if not result.transfer.empty else result.transfer,
               self.root / "loop_transfer_matrix.csv")
        _write(pd.concat([AttackerCosts().as_frame().assign(side="attacker"),
                          DefenderCosts().as_frame().assign(side="defender")]),
               self.root / "loop_cost_model.csv")
        if result.discovered_yaml:
            (self.root / "discovered_vectors.yaml").write_text(
                result.discovered_yaml, encoding="utf-8")

        self.meta["loop"] = summary.to_dict("records")
        self.meta["loop_convergence"] = result.convergence

        best = (result.tactics.sort_values("attacker_profit", ascending=False).head(8)
                if not result.tactics.empty and "attacker_profit" in result.tactics.columns
                else pd.DataFrame())
        rounds = summary["round"].max() if not summary.empty else 0

        text = [
            "## Closed loop",
            "",
            "Each round the detector is frozen; the red team probes it, fits a surrogate copy "
            "from the approve/decline answers, searches for a tactic that profits against that "
            "copy, and commits the survivors to the stream; then the blue team picks its "
            "cheapest response. Both sides optimise money, not accuracy. That is the whole "
            "design: a search rewarded for evasion alone converges on tiny harmless payments "
            "and declares victory, because the cheapest way to look benign is to be benign.",
            "",
            "The attacker's move set has a cosmetic half - amount, hour, session pacing, which "
            "coercion tells to leave behind - and a structural half that changes the plan: the "
            "rail, the channel, the victim cohort, how many payments the demand is split into "
            "and how far apart, how many drop accounts collect it and how hard they are reused, "
            "how long those accounts are seasoned first, and whether to buy an aged account with "
            "real inbound history instead of minting a fresh mule. Each lever has a price, and "
            "some are paid in conversion rather than cash: an operator who cannot stay on the "
            "phone loses a third of their victims, so switching off the coercion telemetry is "
            "not free even though it costs nothing to do.",
            "",
            "Targeted figures cover only the vectors the surviving tactics touched. The overall "
            "figures sit beside them precisely because averaging over fifty vectors hides an "
            "attacker who has comprehensively broken one.",
            "",
            _md_table(summary.drop(columns=["surviving_tactics"], errors="ignore")),
        ]

        verdict = (result.convergence or {}).get("verdict")
        if verdict:
            slope = (result.convergence or {}).get("recall_slope_per_round")
            text += [
                "",
                f"**Convergence over {rounds} rounds: {verdict}.**"
                + (f" Targeted recall moves {slope:+.4f} per round on a least-squares fit."
                   if slope is not None else ""),
            ]

        if not best.empty:
            text += [
                "",
                "The most profitable tactics the search found. `attacker_roi` is profit per "
                "rupee of tooling, accounts and operator time; a negative value is a tactic "
                "that evaded the model and still lost the attacker money, which is a result "
                "worth having and one a recall-only fitness function could never express.",
                "",
                _md_table(best),
            ]

        if not result.blue_options.empty:
            chosen = summary[["round", "blue_move", "blue_total_cost_inr"]] \
                if "blue_move" in summary.columns else pd.DataFrame()
            text += [
                "",
                "### What the defence chose, and what it cost",
                "",
                "Blue is not allowed to simply retrain every round. Five responses are priced "
                "against the same held-out window - hold and absorb the losses, re-cut the "
                "threshold, add a deterministic rule, add step-up friction on a slice, or "
                "retrain - and the cheapest total wins. Cost is fraud losses at the PSR "
                "reimbursement split, plus analyst review, plus false declines, plus customer "
                "friction, plus the engineering and model-risk cost of shipping a change.",
                "",
                _md_table(chosen),
                "",
                "Full option table in `loop_blue_moves.csv`.",
            ]

        if not result.transfer.empty:
            text += [
                "",
                "### Do the tactics transfer?",
                "",
                "Recall of each round's surviving tactics against each round's detector. Read "
                "along a row: a tactic that still slips past detectors trained several rounds "
                "later is a durable capability, and one whose recall climbs is evidence the "
                "defence generalised rather than memorised.",
                "",
                _md_table(result.transfer.reset_index()),
            ]

        if not result.control_gaps.empty:
            text += [
                "",
                "### Which controls the attacker walked through",
                "",
                "Every vector in the library declares the controls that ought to stop it. For "
                "each surviving tactic those controls are marked defeated or intact, which "
                "turns the search log into a defensive roadmap. A control is only marked "
                "defeated when a lever the tactic actually pulled neutralises it; anything "
                "unmapped is reported as intact, so the bias runs against the red team.",
                "",
                _md_table(result.control_gaps.head(12)),
            ]

        if result.discovered:
            text += [
                "",
                "### Back into the taxonomy",
                "",
                f"{len(result.discovered)} surviving tactics were emitted as new library "
                f"entries in `discovered_vectors.yaml`, each validated against the same schema "
                f"gate every hand-authored vector passes "
                f"({len(result.discovery_rejections)} rejected). Each carries its provenance: "
                "the round that found it, the parent vector it mutated from, the recall it "
                "achieved against the deployed model and what it cost to run. This is the "
                "return path that makes the three pillars a loop rather than a pipeline.",
                "",
                _md_table(pd.DataFrame([
                    {"id": d.vector.id, "parent": d.provenance["parent_vector"],
                     "round": d.provenance["round"],
                     "recall_vs_deployed": d.provenance["recall_against_deployed_model"],
                     "attacker_roi": d.provenance["attacker_roi"]}
                    for d in result.discovered[:10]
                ])),
            ]

        text += [
            "",
            "```",
            "round N   red probes the frozen model  ->  surrogate  ->  genome search",
            "             |                                              |",
            "             |                                    surviving tactics",
            "             v                                              |",
            "        control gap  <-------------------------------------+",
            "             |                                              |",
            "             v                                              v",
            "  defensive roadmap                          discovered_vectors.yaml",
            "                                                            |",
            "                                              round N+1 target weighting",
            "```",
            "",
            "The cost assumptions every rupee figure above depends on are written to "
            "`loop_cost_model.csv`. They are assumptions, not measurements; substitute your "
            "own institution's numbers and the ROI column moves accordingly.",
        ]

        self.sections.append("\n".join(text))

    def sweep(self, result) -> None:
        _write(result.runs, self.root / "sweep_runs.csv")
        _write(result.summary, self.root / "sweep_summary.csv")
        self.meta["sweep"] = result.to_dict()

        varied = [c for c in result.runs.columns if c in ("seed",) or c in result.label]
        self.sections.append(
            "## Repeated runs\n\n"
            f"Sweep: **{result.label}**. Each row is a complete generate-and-train cycle, so "
            "the spread below is the variation between independently generated worlds and not "
            "just retraining noise on one dataset. Every headline number elsewhere in this "
            "report is a single draw from this distribution, and should be read with the "
            "spread here attached to it.\n\n"
            + _md_table(result.summary)
            + "\nPer-run detail:\n\n"
            + _md_table(result.runs.drop(columns=[c for c in ("rows", "fraud") if c in
                                                 result.runs.columns], errors="ignore"))
        )

    # -- output ------------------------------------------------------------
    def write(self) -> Path:
        _dump_json(self.meta, self.root / "run_summary.json")
        self._write_results_json()
        self.cfg.save(self.root / "config.yaml")

        p = self.provenance
        timings = ""
        if self.stage_timings:
            total = sum(self.stage_timings.values())
            parts = ", ".join(f"{k} {v:.0f}s" for k, v in self.stage_timings.items())
            timings = f"\nStage timings: {parts} (total {total:.0f}s).\n"

        header = (
            f"# Red Teaming AI for Payment Fraud - run `{self.cfg.run_name}`\n\n"
            f"Every table below is generated from the run's own CSV artefacts in this "
            "directory; none of it is written by hand.\n\n"
            f"{p.summary_line()}\n"
            + ("\n**This run was produced from a modified working tree and cannot be "
               "reproduced from the recorded commit.**\n" if p.git_dirty else "")
            + f"\nThe run id is derived from the code, config, seed and taxonomy rather than "
            "from the clock, so an identical rerun produces an identical id. Environment: "
            + ", ".join(f"{k} {v}" for k, v in p.environment.items()
                        if k in ("python", "numpy", "pandas", "sklearn"))
            + ".\n"
            + timings
        )
        report = self.root / "REPORT.md"
        report.write_text("\n\n".join([header, *self.sections]), encoding="utf-8")
        return report

    def _write_results_json(self) -> None:
        """One schema-versioned file holding every number anything downstream quotes.

        Without this, the write-up, the report narrative and any demo each read a different
        CSV and slowly disagree - and the disagreement surfaces in front of whoever is
        reading, not in front of whoever could fix it. The version field exists so a consumer
        can fail loudly on a shape it does not understand rather than silently reading a key
        that has moved.
        """
        identify = self.meta.get("identify") or {}
        generate = self.meta.get("generate") or {}
        defend = self.meta.get("defend") or {}
        payload = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "provenance": self.provenance.to_dict(),
            "stage_timings_seconds": dict(self.stage_timings),
            "run": {
                "name": self.cfg.run_name,
                "seed": self.cfg.seed,
                "customers": self.cfg.population.n_customers,
                "days": self.cfg.benign.n_days,
                "target_fraud_rate": self.cfg.attacks.target_fraud_rate,
                "target_fpr": self.cfg.defence.target_fpr,
            },
            "identify": {
                "vectors": identify.get("total_vectors"),
                "simulated": identify.get("simulated_vectors"),
                "families": len(identify.get("families") or {}),
            },
            "generate": {
                "transactions": generate.get("transactions"),
                "fraud": generate.get("fraud"),
                "fraud_rate": generate.get("fraud_rate"),
                "fidelity": generate.get("fidelity"),
                "fidelity_flags": generate.get("flags"),
                "raw_separability_recall": generate.get("joint_separability_recall"),
                "derived_separability_recall": generate.get("derived_separability_recall"),
            },
            "defend": defend.get("headline") or {},
            "defend_intervals": self.meta.get("defend_intervals") or [],
            "ablation": self.meta.get("ablation") or [],
            "guard_leakage": self.meta.get("guard_leakage") or [],
            "null_control": self.meta.get("null_control") or [],
            "loop": self.meta.get("loop") or [],
            "sweep": (self.meta.get("sweep") or {}).get("summary") or [],
        }
        # A stage that did not run contributes nothing rather than a block of nulls, so a
        # consumer can distinguish "not measured in this run" from "measured as zero".
        payload = {k: v for k, v in payload.items()
                   if k in ("schema_version", "run") or not _is_blank(v)}
        _dump_json(payload, self.root / "results.json")
