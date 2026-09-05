"""Content classifier for indirect prompt injection in agent context bundles.

This is the second data plane of the defence. It never sees a transaction; it reads the
catalogue text, tool descriptions and review corpus the agent consumed on the way to
checkout, and emits a calibrated probability that the context contains an instruction
aimed at the agent rather than information aimed at the buyer.

Two design choices matter:

*Normalisation before vectorisation.* Zero-width joiners, doubled spacing and HTML comment
wrappers are cheap evasions that any deployed filter strips. Normalising them is standard
hygiene, not generator knowledge. What survives normalisation - reversed base64-ish
wrappers, payloads buried inside review text - is left for the model to earn.

*Character n-grams alongside word n-grams.* A word-only model memorises payload vocabulary
and collapses the moment an attacker rewrites the sentence. Character n-grams pick up the
structural tells - directive syntax, bracketed pseudo-tags, imperative-to-machine framing -
that survive rewording. :func:`leave_one_family_out` measures exactly that, by training
with one payload family removed and testing on it.

The output is written back onto the transaction as ``agent_injection_score``, a real
classifier score with real errors, so the tabular model consumes an honest feature.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .thresholds import threshold_for_budget

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
HTML_COMMENT = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)
MULTISPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Undo the cheap evasions before the model ever sees the string."""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    text = HTML_COMMENT.sub(r"\1", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip().lower()


def _build_pipeline(seed: int) -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60_000,
        sublinear_tf=True, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=120_000,
        sublinear_tf=True,
    )
    # A log-loss SGD learner rather than lbfgs/liblinear logistic regression. The decision
    # boundary is the same family; the difference is that this one stays inside scikit-learn's
    # Cython sparse routines instead of dispatching to the platform BLAS, which on macOS
    # arm64 faults on matrices this wide. It also supports ``partial_fit``, so the guard can
    # be updated online as the closed loop discovers new payload families.
    base = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-6, max_iter=60, tol=1e-4,
        class_weight="balanced", random_state=seed,
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word), ("char", char)])),
            ("clf", CalibratedClassifierCV(base, method="isotonic", cv=3)),
        ]
    )


@dataclass
class InjectionGuardReport:
    train_rows: int
    test_rows: int
    roc_auc: float
    pr_auc: float
    recall_at_1pct_fpr: float
    threshold: float
    per_obfuscation_recall: Dict[str, float]
    per_family_recall: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "recall_at_1pct_fpr": round(self.recall_at_1pct_fpr, 4),
            "threshold": round(self.threshold, 4),
            "per_obfuscation_recall": {k: round(v, 4) for k, v in self.per_obfuscation_recall.items()},
            "per_family_recall": {k: round(v, 4) for k, v in self.per_family_recall.items()},
        }


class InjectionGuard:
    """Trains on agent context bundles and scores unseen ones."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.pipeline: Optional[Pipeline] = None
        self.threshold: float = 0.5
        self.report: Optional[InjectionGuardReport] = None

    def fit(self, corpus: pd.DataFrame, train_end: pd.Timestamp,
            test_start: pd.Timestamp) -> InjectionGuardReport:
        """Fit on the training window only; evaluate on the held-out window.

        The boundaries are the same wall-clock ones the tabular model uses. Training the
        text guard on future bundles would leak straight into the tabular model through
        ``agent_injection_score``, which is the subtler of the two ways this pipeline could
        have quietly cheated.
        """
        ts = pd.to_datetime(corpus["timestamp"])
        train = corpus[ts < train_end]
        test = corpus[ts >= test_start]
        if train.empty or int(train["has_injection"].sum()) < 5:
            raise ValueError("insufficient injected bundles in the training window")

        self.pipeline = _build_pipeline(self.seed)
        self.pipeline.fit([normalise(t) for t in train["text"]], train["has_injection"].to_numpy())

        if test.empty:
            self.threshold = 0.5
            self.report = InjectionGuardReport(len(train), 0, float("nan"), float("nan"),
                                               float("nan"), 0.5, {}, {})
            return self.report

        y = test["has_injection"].to_numpy()
        scores = self.predict_proba([str(t) for t in test["text"]])
        self.threshold = _threshold_at_fpr(y, scores, target_fpr=0.01)

        flagged = scores >= self.threshold
        per_obf = {
            str(mode): float(flagged[(test["obfuscation"] == mode).to_numpy() & (y == 1)].mean())
            for mode in sorted(test.loc[y == 1, "obfuscation"].unique())
        }
        per_family = {
            str(fam): float(flagged[(test["payload_family"] == fam).to_numpy() & (y == 1)].mean())
            for fam in sorted(test.loc[y == 1, "payload_family"].unique())
        }

        self.report = InjectionGuardReport(
            train_rows=len(train),
            test_rows=len(test),
            roc_auc=float(roc_auc_score(y, scores)) if y.min() != y.max() else float("nan"),
            pr_auc=float(average_precision_score(y, scores)) if y.min() != y.max() else float("nan"),
            recall_at_1pct_fpr=float(flagged[y == 1].mean()) if (y == 1).any() else float("nan"),
            threshold=float(self.threshold),
            per_obfuscation_recall=per_obf,
            per_family_recall=per_family,
        )
        return self.report

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("guard is not fitted")
        return self.pipeline.predict_proba([normalise(t) for t in texts])[:, 1]

    def score_transactions(self, df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
        """Write ``agent_injection_score`` onto the transaction frame.

        Non-agentic rows get 0.0 rather than NaN: for a human-initiated payment the
        statement "no adversarial agent context was observed" is true, not missing.
        """
        out = df.copy()
        out["agent_injection_score"] = 0.0
        out["agent_injection_flag"] = 0
        if corpus.empty:
            return out
        linked = corpus[corpus["linked"] == 1]
        if linked.empty:
            return out
        scores = self.predict_proba([str(t) for t in linked["text"]])
        mapping = dict(zip(linked["txn_id"].to_numpy(), scores))
        ids = out["txn_id"].to_numpy()
        out["agent_injection_score"] = np.round(
            [mapping.get(i, 0.0) for i in ids], 5
        )
        out["agent_injection_flag"] = (
            out["agent_injection_score"].to_numpy() >= self.threshold
        ).astype(int)
        return out

    def top_terms(self, k: int = 20) -> pd.DataFrame:
        """Most injection-indicative word features, for the model card."""
        if self.pipeline is None:
            raise RuntimeError("guard is not fitted")
        union: FeatureUnion = self.pipeline.named_steps["features"]
        word_vec: TfidfVectorizer = dict(union.transformer_list)["word"]
        calibrated: CalibratedClassifierCV = self.pipeline.named_steps["clf"]
        coefs = np.mean(
            [c.estimator.coef_[0] for c in calibrated.calibrated_classifiers_], axis=0
        )
        names = np.asarray(union.get_feature_names_out())
        word_mask = np.array([n.startswith("word__") for n in names])
        w_names = names[word_mask]
        w_coefs = coefs[word_mask]
        order = np.argsort(w_coefs)[::-1][:k]
        return pd.DataFrame(
            {
                "term": [w_names[i].replace("word__", "") for i in order],
                "weight": np.round(w_coefs[order], 4),
            }
        )


def unseen_phrasing_holdout(corpus: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Train on half the payload phrasings per family, test on the other half.

    This is the number to read, and it is the reason the in-distribution figures in
    :class:`InjectionGuardReport` should be treated as an upper bound rather than a result.
    Generated payloads come from a finite set of templates, so a bag-of-n-grams model can
    memorise the exact strings and report a perfect score that would evaporate the first
    time an attacker rewrote a sentence. Holding out whole phrasings forces the model to
    generalise across surface form within a family, which is the closest offline analogue
    of an attacker paraphrasing a known payload.
    """
    benign = corpus[corpus["has_injection"] == 0]
    injected = corpus[corpus["has_injection"] == 1]
    if injected.empty or "payload_template" not in corpus.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for family, group in injected.groupby("payload_family"):
        templates = sorted(group["payload_template"].unique())
        if len(templates) < 4:
            continue
        held = set(rng.choice(templates, size=max(1, len(templates) // 2), replace=False))
        train_pos = group[~group["payload_template"].isin(held)]
        test_pos = group[group["payload_template"].isin(held)]
        # Other families stay in training: an attacker paraphrasing one payload family does
        # not remove the defender's knowledge of the others.
        other = injected[injected["payload_family"] != family]
        train = pd.concat([benign, train_pos, other])
        if len(test_pos) < 10 or train_pos.empty:
            continue

        pipe = _build_pipeline(seed)
        pipe.fit([normalise(t) for t in train["text"]], train["has_injection"].to_numpy())
        benign_scores = pipe.predict_proba([normalise(t) for t in benign["text"]])[:, 1]
        thr = threshold_for_budget(benign_scores, 0.01)
        held_scores = pipe.predict_proba([normalise(t) for t in test_pos["text"]])[:, 1]
        rows.append(
            {
                "payload_family": family,
                "held_out_phrasings": len(held),
                "held_out_rows": len(test_pos),
                "recall_on_unseen_phrasing": round(float((held_scores >= thr).mean()), 4),
                "median_score": round(float(np.median(held_scores)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("recall_on_unseen_phrasing").reset_index(drop=True)


def leave_one_family_out(corpus: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Hold out an entire payload family and see whether the guard still finds it.

    This is the honest question for a content filter. In-distribution recall measures
    vocabulary memorisation; recall on a family the model has never seen measures whether
    it learned the shape of an instruction aimed at a machine. The gap between the two
    numbers is the guard's real exposure to a novel payload.
    """
    families = sorted(f for f in corpus["payload_family"].unique() if f != "none")
    benign = corpus[corpus["has_injection"] == 0]
    rows = []
    for held in families:
        train = pd.concat(
            [benign, corpus[(corpus["has_injection"] == 1) & (corpus["payload_family"] != held)]]
        )
        test_pos = corpus[(corpus["has_injection"] == 1) & (corpus["payload_family"] == held)]
        if len(test_pos) < 5 or int(train["has_injection"].sum()) < 5:
            continue
        pipe = _build_pipeline(seed)
        pipe.fit([normalise(t) for t in train["text"]], train["has_injection"].to_numpy())

        # Threshold set on held-out benign text so the false-positive budget is comparable.
        benign_scores = pipe.predict_proba([normalise(t) for t in benign["text"]])[:, 1]
        thr = threshold_for_budget(benign_scores, 0.01)
        held_scores = pipe.predict_proba([normalise(t) for t in test_pos["text"]])[:, 1]
        rows.append(
            {
                "held_out_family": held,
                "held_out_rows": len(test_pos),
                "threshold_at_1pct_benign_fpr": round(thr, 4),
                "zero_shot_recall": round(float((held_scores >= thr).mean()), 4),
                "median_score": round(float(np.median(held_scores)), 4),
            }
        )
    return pd.DataFrame(rows)


def _threshold_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    return threshold_for_budget(scores[y == 0], target_fpr)
