"""The tabular detector: a stacked ensemble with an unsupervised channel.

Three learners, chosen because they fail differently:

* **Gradient-boosted trees** carry the load. They handle the mixed categorical/numeric
  matrix natively, tolerate missing values, and pick up the threshold interactions that
  dominate payment risk ("new payee AND irrevocable rail AND above the customer's p95").
* **A random forest** on the same matrix. Correlated with the booster but not identical;
  it is markedly steadier on the low-support attack vectors where the booster overfits a
  handful of episodes.
* **An isolation forest fitted on legitimate training traffic only.** This is the channel
  that matters for the closed loop. The supervised pair can only recognise attacks shaped
  like the ones in the training window; the novelty score keeps rising for a vector nobody
  has labelled yet, which is precisely the zero-day case.

The three are combined by a logistic meta-learner fitted on the *calibration* slice, which
neither base learner saw. That gives a stack whose weights are estimated out-of-sample and
whose output is a usable probability rather than a raw score.

Operating point is chosen by false-positive budget, not by F1. A payment risk engine is
provisioned in alerts per day; "maximise F1" is not a sentence anyone in a fraud operations
team can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from .dataset import MatrixBuilder, input_columns
from .thresholds import threshold_for_budget

TARGET_FPR = 0.005
"""Half a percent of legitimate payments. On the volumes a mid-size PSP runs, anything
looser is an operations budget nobody has."""


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class FraudDetector:
    seed: int = 0
    target_fpr: float = TARGET_FPR
    max_iter: int = 400
    n_estimators: int = 300

    builder: Optional[MatrixBuilder] = None
    drop_columns: Tuple[str, ...] = ()
    """Inputs to withhold, for ablation. Empty means "read everything available".

    Ablation needs to be a first-class capability rather than something a caller fakes by
    pre-trimming the frame: the feature builders need columns that the model must not read,
    so dropping them upstream changes the features too and measures the wrong thing.
    """

    columns: List[str] = field(default_factory=list)
    hgb: Optional[HistGradientBoostingClassifier] = None
    rf: Optional[RandomForestClassifier] = None
    iso: Optional[IsolationForest] = None
    meta: Optional[LogisticRegression] = None
    threshold: float = 0.5
    legit_stats: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------------------------
    # Fitting
    # ----------------------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "FraudDetector":
        withheld = set(self.drop_columns)
        self.columns = [c for c in input_columns(train) if c not in withheld]
        if not self.columns:
            raise ValueError("drop_columns withheld every available input")
        self.builder = MatrixBuilder(self.columns).fit(train)

        x_train = self.builder.transform(train)
        y_train = train["is_fraud"].to_numpy().astype(int)
        if y_train.sum() < 20:
            raise ValueError("training window contains too few fraud rows to fit a detector")

        cat_mask = self.builder.categorical_mask
        self.hgb = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            max_bins=255,
            categorical_features=cat_mask,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=self.seed,
        )
        self.hgb.fit(x_train, y_train)

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.seed,
        )
        self.rf.fit(np.nan_to_num(x_train, nan=-1.0), y_train)

        # Fitted on legitimate rows only: this learner's job is to describe normality, and
        # showing it fraud would defeat the point.
        self.iso = IsolationForest(
            n_estimators=200,
            max_samples=min(8192, len(x_train)),
            contamination="auto",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.iso.fit(np.nan_to_num(x_train[y_train == 0], nan=-1.0))

        self._fit_meta(calibration)
        self._fit_legit_stats(train)
        return self

    def _fit_meta(self, calibration: pd.DataFrame) -> None:
        x_cal = self.builder.transform(calibration)
        y_cal = calibration["is_fraud"].to_numpy().astype(int)
        z_cal = self._base_scores(x_cal)

        if y_cal.sum() >= 10:
            self.meta = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=self.seed
            )
            self.meta.fit(z_cal, y_cal)
            proba = self.meta.predict_proba(z_cal)[:, 1]
        else:  # pragma: no cover - only on pathologically small runs
            self.meta = None
            proba = _sigmoid(z_cal[:, 0])

        self.threshold = threshold_for_budget(proba[y_cal == 0], self.target_fpr)

    def _fit_legit_stats(self, train: pd.DataFrame) -> None:
        """Per-feature legitimate baseline, used to explain individual alerts."""
        legit = train[train["is_fraud"] == 0]
        x = self.builder.transform(legit)
        self.legit_stats = pd.DataFrame(
            {
                "column": self.columns,
                "mean": np.nanmean(x, axis=0),
                "std": np.nanstd(x, axis=0) + 1e-9,
            }
        ).set_index("column")

    # ----------------------------------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------------------------------

    def _base_scores(self, x: np.ndarray) -> np.ndarray:
        x_filled = np.nan_to_num(x, nan=-1.0)
        p_hgb = self.hgb.predict_proba(x)[:, 1]
        p_rf = self.rf.predict_proba(x_filled)[:, 1]
        # score_samples is higher for normal points; negate so larger means more anomalous.
        novelty = -self.iso.score_samples(x_filled)
        return np.column_stack([_logit(p_hgb), _logit(p_rf), novelty])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self.builder.transform(df)
        z = self._base_scores(x)
        if self.meta is None:  # pragma: no cover
            return _sigmoid(z[:, 0])
        return self.meta.predict_proba(z)[:, 1]

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full scoring output: probability, decision and per-channel diagnostics."""
        x = self.builder.transform(df)
        z = self._base_scores(x)
        proba = self.meta.predict_proba(z)[:, 1] if self.meta is not None else _sigmoid(z[:, 0])
        return pd.DataFrame(
            {
                "txn_id": df["txn_id"].to_numpy(),
                "score": np.round(proba, 6),
                "alert": (proba >= self.threshold).astype(int),
                "gbm_logit": np.round(z[:, 0], 4),
                "rf_logit": np.round(z[:, 1], 4),
                "novelty": np.round(z[:, 2], 4),
            }
        )

    def novelty_scores(self, df: pd.DataFrame) -> np.ndarray:
        x = np.nan_to_num(self.builder.transform(df), nan=-1.0)
        return -self.iso.score_samples(x)

    # ----------------------------------------------------------------------------------
    # Explanation
    # ----------------------------------------------------------------------------------

    def global_importance(self, sample: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
        """Permutation importance against the ensemble output.

        Measured on the ensemble rather than on a single learner, because the ensemble is
        what makes the decision. Permutation is slow but it is the only importance here
        that is not a tree-structure artefact.
        """
        from sklearn.metrics import roc_auc_score

        y = sample["is_fraud"].to_numpy().astype(int)
        if y.min() == y.max():
            return pd.DataFrame(columns=["column", "importance"])
        base = roc_auc_score(y, self.predict_proba(sample))
        rng = np.random.default_rng(self.seed)
        rows = []
        for c in self.columns:
            drops = []
            original = sample[c].to_numpy().copy()
            for _ in range(n_repeats):
                shuffled = sample.copy()
                shuffled[c] = rng.permutation(original)
                drops.append(base - roc_auc_score(y, self.predict_proba(shuffled)))
            rows.append({"column": c, "importance": float(np.mean(drops))})
        out = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return out.reset_index(drop=True)

    def reason_codes(self, df: pd.DataFrame, importance: pd.DataFrame, k: int = 4) -> List[str]:
        """Why this payment alerted, in terms an investigator can act on.

        The score is a global importance weight times the row's standardised deviation from
        the legitimate baseline. It is an approximation of a Shapley attribution, and it is
        used deliberately: it costs one matrix operation per batch, which is the difference
        between reason codes that ship inside a 300ms authorisation budget and reason codes
        that only ever appear in an offline notebook.
        """
        x = self.builder.transform(df)
        weights = (
            importance.set_index("column")["importance"].reindex(self.columns).fillna(0.0).to_numpy()
        )
        weights = np.maximum(weights, 0.0)
        mean = self.legit_stats["mean"].reindex(self.columns).to_numpy()
        std = self.legit_stats["std"].reindex(self.columns).to_numpy()
        deviation = np.abs((np.nan_to_num(x, nan=0.0) - mean) / std)
        contribution = deviation * weights

        order = np.argsort(-contribution, axis=1)[:, :k]
        cols = np.asarray(self.columns)
        out = []
        for i in range(len(df)):
            picks = [j for j in order[i] if contribution[i, j] > 0]
            out.append(", ".join(f"{cols[j]}={_fmt(df.iloc[i][cols[j]])}" for j in picks))
        return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3g}"
    return str(v)
