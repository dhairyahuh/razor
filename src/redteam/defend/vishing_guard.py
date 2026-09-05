"""Coercion classifier over call and chat transcripts.

This is the third data plane of the defence, and the only one that sees the cause of an
authorised push payment rather than its consequences. The tabular model's evidence that
somebody was on the telephone is ``call_in_progress`` - one bit, which the closed loop
demonstrates an attacker can simply switch off at a cost of some conversion. The
conversation is far harder to sanitise, because the scam has to make its moves to work at
all: establish authority, manufacture urgency, discourage verification, and direct money
somewhere new.

The classifier is deliberately the same shape as the injection guard - word and character
n-grams into a calibrated linear model - for a reason worth stating. Character n-grams are
what survive rewording. A cloned voice reading LLM-written script will not reuse the corpus
vocabulary, but it will still be an imperative addressed to somebody being walked through
their own banking app, and that register has structure.

Reading the numbers, and why they are all near 1.0
--------------------------------------------------
Every evaluation in this module returns a near-perfect score, including the ones designed
specifically to break it, and that needs saying plainly rather than being presented as a
result. **The limit is the corpus, not the guard.** One author wrote both classes, so the
boundary between them is perfectly consistent by construction, and a linear model over
n-grams recovers a perfectly consistent boundary exactly. No amount of additional template
writing fixes this, because the additional templates have the same author.

Several artefacts that would have produced the same score for worse reasons *were* found and
removed while building this, and they are worth listing because each is a trap the next
person will otherwise re-enter:

* the two classes had disjoint openers, so the first sentence was sufficient;
* customer-side turns appeared only in scam transcripts, making "sounds like somebody being
  walked through their banking app" a free label that transferred across held-out vocabulary
  as a *register*;
* the legitimate class was systematically longer.

Fixing all three moved the score by nothing, which is itself the finding: the residual
separability is the semantic content the author put there deliberately.

So the two holdouts below establish a narrower claim than they appear to.
:func:`unseen_wording_recall` re-renders both classes in a vocabulary withheld from training
and rules out string memorisation. :func:`held_out_script_recall` withholds a whole pretext
and is the weaker of the two, because the coercion vocabulary is shared across pretexts and
survives that holdout intact. Neither can rule out shared authorship. Only text from a
different generator can, which is what :mod:`redteam.genai.transcript_bank` is for and why
the report states the model-written share beside every figure.

The false-positive number matters more here than anywhere else in the system. Every
legitimate transcript is a near-twin of a scam one: a real fraud-team callback against an
impersonated one, a real delivery message against a redelivery scam, a genuine first payment
to a builder who changed banks. A guard that cannot separate those would flag the bank's own
outbound calls, and the cost of that is not an analyst's time, it is customers who stop
answering the telephone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .injection_guard import normalise
from .thresholds import threshold_for_budget


def _build_pipeline(seed: int) -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                           max_features=60_000, sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=120_000, sublinear_tf=True)
    base = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, max_iter=60, tol=1e-4,
                         class_weight="balanced", random_state=seed)
    return Pipeline([
        ("features", FeatureUnion([("word", word), ("char", char)])),
        ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3)),
    ])


@dataclass
class VishingGuardReport:
    train_rows: int
    test_rows: int
    roc_auc: float
    pr_auc: float
    recall_at_1pct_fpr: float
    threshold: float
    per_script_recall: Dict[str, float]
    llm_written_recall: float
    template_recall: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "recall_at_1pct_fpr": round(self.recall_at_1pct_fpr, 4),
            "threshold": round(self.threshold, 4),
            "per_script_recall": {k: round(v, 4) for k, v in self.per_script_recall.items()},
            "recall_on_llm_written": round(self.llm_written_recall, 4),
            "recall_on_template_written": round(self.template_recall, 4),
        }


class VishingGuard:
    """Trains on transcripts from the training window and scores unseen ones."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.pipeline: Optional[Pipeline] = None
        self.threshold: float = 0.5
        self.report: Optional[VishingGuardReport] = None

    def fit(self, corpus: pd.DataFrame, train_end: pd.Timestamp,
            test_start: pd.Timestamp) -> VishingGuardReport:
        """Fit on the training window only.

        Same wall-clock boundaries as the tabular model. Training this guard on future
        transcripts would leak into the tabular model through ``vishing_score``, which is the
        quiet way a text plane can contaminate a temporal split.
        """
        ts = pd.to_datetime(corpus["timestamp"])
        train = corpus[ts < train_end]
        test = corpus[ts >= test_start]
        if train.empty or int(train["is_coercive"].sum()) < 5:
            raise ValueError("insufficient coercive transcripts in the training window")

        self.pipeline = _build_pipeline(self.seed)
        self.pipeline.fit([normalise(t) for t in train["text"]],
                          train["is_coercive"].to_numpy())

        if test.empty:
            self.threshold = 0.5
            self.report = VishingGuardReport(len(train), 0, float("nan"), float("nan"),
                                             float("nan"), 0.5, {}, float("nan"),
                                             float("nan"))
            return self.report

        y = test["is_coercive"].to_numpy()
        scores = self.predict_proba([str(t) for t in test["text"]])
        self.threshold = threshold_for_budget(scores[y == 0], 0.01)
        flagged = scores >= self.threshold

        per_script = {
            str(s): float(flagged[(test["scam_script"] == s).to_numpy() & (y == 1)].mean())
            for s in sorted(test.loc[y == 1, "scam_script"].unique())
        }
        # Recall split by who wrote the transcript. If the guard does markedly worse on the
        # model-written half then the template half was teaching it templates, which is the
        # defect this whole generative layer exists to expose.
        source = test["source"].to_numpy() if "source" in test.columns else np.array([""] * len(test))
        llm = (source == "llm") & (y == 1)
        template = (source == "template") & (y == 1)

        self.report = VishingGuardReport(
            train_rows=len(train),
            test_rows=len(test),
            roc_auc=float(roc_auc_score(y, scores)) if y.min() != y.max() else float("nan"),
            pr_auc=float(average_precision_score(y, scores)) if y.min() != y.max() else float("nan"),
            recall_at_1pct_fpr=float(flagged[y == 1].mean()) if (y == 1).any() else float("nan"),
            threshold=float(self.threshold),
            per_script_recall=per_script,
            llm_written_recall=float(flagged[llm].mean()) if llm.any() else float("nan"),
            template_recall=float(flagged[template].mean()) if template.any() else float("nan"),
        )
        return self.report

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("guard is not fitted")
        return self.pipeline.predict_proba([normalise(t) for t in texts])[:, 1]

    def score_transactions(self, df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
        """Write ``vishing_score`` onto the transaction frame.

        The score attaches to the whole episode, not just the payment the transcript was
        anchored to. A scam call produces several transfers and the bank hears one
        conversation, so scoring only the first payment would model a capability nobody has
        and would understate the guard on exactly the multi-payment episodes it helps most
        with.

        Payments with no observed conversation get 0.0 rather than NaN: "no coercive call was
        recorded" is a true statement about them, not a missing one. It is also, deliberately,
        the same value a scam gets if the bank never captured the call - the guard cannot
        help where there is no telemetry, and pretending otherwise would inflate it.
        """
        out = df.copy()
        out["vishing_score"] = 0.0
        out["vishing_flag"] = 0
        # Whether a recording exists at all, kept separate from what it said. A genuine call
        # scores near zero and rounds to the same 0.0 as a payment nobody phoned about, so
        # without this column the two are indistinguishable and any audit of the guard's
        # coverage silently measures "scored as coercive" instead. Not a model input - it is
        # very nearly `call_in_progress`, which the model already reads - but the leakage
        # probe needs it to ask whether the telemetry's presence is doing the classifying.
        out["vishing_covered"] = 0
        if corpus.empty or self.pipeline is None:
            return out

        scores = self.predict_proba([str(t) for t in corpus["text"]])
        anchored = dict(zip(corpus["txn_id"].to_numpy(), scores))

        by_txn = out["txn_id"].map(anchored)
        if "campaign_id" in out.columns:
            # Spread each transcript's score across its episode, then keep the strongest
            # signal available for any given payment.
            #
            # Only across *real* episodes. `campaign_id` defaults to the empty string, so
            # every legitimate payment in the book shares one value; grouping on it without
            # excluding the default pooled thirty-five thousand unrelated payments into a
            # single episode and handed all of them the highest-scoring genuine transcript in
            # the run. Fraud rows belonging to an episode with no captured call kept 0.0, so
            # `vishing_score == 0` became a near-perfect *inverted* label and the guard layer
            # took headline recall from 71% to 100% on its own. The ablation grid is what
            # exposed it; nothing else in the evaluation could see it, because the fidelity
            # probes zero-fill the guard columns by design.
            campaign = out["campaign_id"].astype("string")
            real = campaign.notna() & (campaign.str.strip() != "")
            grouped = pd.DataFrame({"campaign_id": campaign.where(real), "score": by_txn})
            per_campaign = grouped.dropna(subset=["campaign_id", "score"]) \
                                  .groupby("campaign_id")["score"].max()
            spread = campaign.where(real).map(per_campaign)
            by_txn = by_txn.fillna(pd.Series(spread.to_numpy(), index=out.index))

        out["vishing_covered"] = by_txn.notna().astype(int).to_numpy()
        out["vishing_score"] = np.round(by_txn.fillna(0.0).astype(float).to_numpy(), 5)
        out["vishing_flag"] = (out["vishing_score"].to_numpy() >= self.threshold).astype(int)
        return out


def unseen_wording_recall(guard: "VishingGuard", seed: int = 0) -> pd.DataFrame:
    """Score the fitted guard on a corpus written in vocabulary it has never seen.

    This is the number to quote, and the reason is worth stating rather than leaving to the
    reader. Every scam transcript in the training corpus expresses its redirection using one
    of a handful of catalogued phrasings, and no genuine transcript ever does. A bag-of-
    n-grams model therefore reaches a perfect in-distribution score by memorising a few
    strings, and holding out an entire *pretext* does not disturb that at all, because the
    coercion vocabulary is shared across pretexts and survives the holdout intact. That is
    why leave-one-script-out also reports near-perfect recall and why neither figure is
    evidence on its own.

    What a deployed guard meets is a model-written script: the same four moves, expressed in
    words nobody catalogued. Here both classes are re-rendered in disjoint wording, so the
    false-positive budget is set on unseen negatives too, and the recall that survives is
    attributable to structure rather than to vocabulary.
    """
    from ..generate.transcripts import build_holdout_corpus

    if guard.pipeline is None:
        return pd.DataFrame()
    holdout = build_holdout_corpus(np.random.default_rng(seed))
    scores = guard.predict_proba([str(t) for t in holdout["text"]])
    y = holdout["is_coercive"].to_numpy()

    # Re-thresholded on the unseen negatives rather than reusing the fitted threshold. The
    # question is whether the guard can still hit a 1% false-positive budget at all on
    # unfamiliar language, not whether an old cut-point happens to transfer.
    thr = threshold_for_budget(scores[y == 0], 0.01)
    flagged = scores >= thr
    rows = [{
        "evaluation": "all held-out wording",
        "rows": int(len(holdout)),
        "threshold_at_1pct_unseen_fpr": round(float(thr), 4),
        "recall": round(float(flagged[y == 1].mean()), 4),
        "false_positive_rate": round(float(flagged[y == 0].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
    }]
    for script in sorted(holdout.loc[y == 1, "scam_script"].unique()):
        mask = (holdout["scam_script"] == script).to_numpy() & (y == 1)
        rows.append({
            "evaluation": script,
            "rows": int(mask.sum()),
            "threshold_at_1pct_unseen_fpr": round(float(thr), 4),
            "recall": round(float(flagged[mask].mean()), 4),
            "false_positive_rate": float("nan"),
            "roc_auc": float("nan"),
        })
    return pd.DataFrame(rows)


def held_out_script_recall(corpus: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Hold out a whole scam script and see whether the guard still recognises coercion.

    The honest question for this guard. Social-engineering pretexts turn over constantly -
    a new government scheme, a new courier, a new investment platform - and a defence that
    only knows the pretexts in its training data will be perpetually one script behind. What
    should generalise is the structure underneath: urgency, isolation, and an instruction to
    send money somewhere new. This measures whether it does.
    """
    scripts = sorted(s for s in corpus["scam_script"].unique() if s != "none")
    benign = corpus[corpus["is_coercive"] == 0]
    rows: List[Dict[str, object]] = []
    for held in scripts:
        positives = corpus[corpus["is_coercive"] == 1]
        train = pd.concat([benign, positives[positives["scam_script"] != held]])
        test_pos = positives[positives["scam_script"] == held]
        if len(test_pos) < 5 or int(train["is_coercive"].sum()) < 5:
            continue
        pipe = _build_pipeline(seed)
        pipe.fit([normalise(t) for t in train["text"]], train["is_coercive"].to_numpy())

        benign_scores = pipe.predict_proba([normalise(t) for t in benign["text"]])[:, 1]
        thr = threshold_for_budget(benign_scores, 0.01)
        held_scores = pipe.predict_proba([normalise(t) for t in test_pos["text"]])[:, 1]
        rows.append({
            "held_out_script": held,
            "held_out_rows": len(test_pos),
            "threshold_at_1pct_benign_fpr": round(float(thr), 4),
            "zero_shot_recall": round(float((held_scores >= thr).mean()), 4),
            "median_score": round(float(np.median(held_scores)), 4),
        })
    return pd.DataFrame(rows).sort_values("zero_shot_recall").reset_index(drop=True) \
        if rows else pd.DataFrame()
