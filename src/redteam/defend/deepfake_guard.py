"""Synthetic-media artefact scoring over the biometric capture telemetry.

Why the GenAI framing needs a defensive half
--------------------------------------------
The threat research this system is built from spends most of its length on generation: voice
clones from fifteen seconds of audio, virtual camera injection past liveness checks, diffusion
models that now reproduce the photoplethysmographic pulse signal the industry adopted
specifically to catch them. Every one of those is represented on the offensive side here -
``ATO-VOICE-CLONE-IVR``, the camera-injection vectors, the document forgery family all write
``voice_match_score``, ``liveness_score`` and ``doc_ocr_confidence``.

Nothing read them back. The tabular model consumed those columns as ordinary numbers, which
means the system modelled the attack and not the countermeasure, and a panel asking "so what
does your defence do about deepfakes" would have got an answer about gradient boosting.

What this actually is, and is not
---------------------------------
This is **not** a deepfake detector. A real one operates on the media - spectral artefacts,
frame-level inconsistency, rPPG residuals - and this system has no media, only the scores a
capture SDK returned. Claiming otherwise would be the most dishonest thing in the repository.

What it is, is the layer above such a detector: given the *outputs* of the biometric stack,
does this capture sit where genuine captures sit? Two things make that worth doing.

* **Genuine captures are messy in a characteristic way.** Real biometric matching produces
  a broad distribution with correlated components - a poor-lighting capture degrades the
  liveness score and the match score together. A synthesised one is often *too clean* on one
  axis and inconsistent across axes, because the generator optimised the thing the attacker
  could see.
* **It is trained only on legitimate captures.** A one-class model over the training window's
  genuine media telemetry, scoring novelty. It never sees a fraud label, so it cannot memorise
  which vectors are attacks, and it degrades honestly on an attack whose capture profile
  happens to look ordinary - which is exactly what an anti-forensically generated one will do.

The score is written back as ``media_artefact_score``, and the report states plainly that it
is a consistency check over vendor outputs rather than media forensics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import QuantileTransformer

from .thresholds import threshold_for_budget

#: The capture telemetry a biometric stack returns. Present on every row; meaningful only
#: where a capture actually happened.
MEDIA_COLUMNS: List[str] = [
    "voice_match_score",
    "liveness_score",
    "biometric_match_score",
    "doc_ocr_confidence",
]


def has_capture(df: pd.DataFrame) -> np.ndarray:
    """Rows where at least one biometric modality was actually captured.

    Everything else scores zero. A card payment at a terminal has no voice sample, and
    scoring its absent voice match as anomalous would turn this guard into a channel
    detector - which the model already has, for free, in the channel column.
    """
    present = np.zeros(len(df), dtype=bool)
    for column in MEDIA_COLUMNS:
        if column in df.columns:
            present |= df[column].to_numpy().astype(float) > 0
    return present


@dataclass
class DeepfakeGuardReport:
    fitted_on: int
    scored: int
    threshold: float
    coverage: float
    """Share of all rows carrying any biometric capture at all. The guard's ceiling: it can
    say nothing about the rest, and a headline that averaged over them would be diluted
    nonsense."""

    recall_at_1pct_fpr: float
    per_vector_recall: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "fitted_on_genuine_captures": self.fitted_on,
            "scored_captures": self.scored,
            "capture_coverage": round(self.coverage, 4),
            "threshold": round(self.threshold, 4),
            "recall_at_1pct_fpr": round(self.recall_at_1pct_fpr, 4),
            "per_vector_recall": {k: round(v, 4) for k, v in self.per_vector_recall.items()},
        }


class DeepfakeGuard:
    """One-class novelty scoring over biometric capture telemetry."""

    def __init__(self, seed: int = 0, n_estimators: int = 120):
        self.seed = seed
        self.n_estimators = n_estimators
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[QuantileTransformer] = None
        self._columns: List[str] = []
        self.threshold: float = 1.0
        self.report: Optional[DeepfakeGuardReport] = None

    def fit(self, train: pd.DataFrame, test: Optional[pd.DataFrame] = None) -> "DeepfakeGuard":
        """Fit on genuine captures in the training window only.

        Labels are used to *select* the fitting set, not to supervise it. That is the same
        arrangement any bank has: it knows which of its historical captures were later
        confirmed fraudulent and can exclude them, and it has no labels at all for the
        traffic it is about to score.
        """
        self._columns = [c for c in MEDIA_COLUMNS if c in train.columns]
        if not self._columns:
            return self

        genuine = train[(train["is_fraud"] == 0) & has_capture(train)]
        if len(genuine) < 200:
            return self

        raw = self._matrix(genuine)
        # Quantile-mapped before the forest. The four scores have wildly different shapes -
        # one is a near-degenerate spike at zero, another is a broad beta - and an isolation
        # forest on the raw values spends most of its splits on that scale difference rather
        # than on the joint structure that actually distinguishes a capture.
        self._scaler = QuantileTransformer(
            n_quantiles=min(1_000, len(genuine)), output_distribution="normal",
            random_state=self.seed,
        ).fit(raw)
        self._model = IsolationForest(
            n_estimators=self.n_estimators, contamination=0.02,
            random_state=self.seed, n_jobs=1,
        ).fit(self._scaler.transform(raw))

        # Threshold on genuine captures so the false-positive budget is stated in the units
        # that matter: share of real customers asked to re-verify.
        self.threshold = float(threshold_for_budget(self.score(genuine), 0.01))
        self._report(train, test, len(genuine))
        return self

    @property
    def usable(self) -> bool:
        return self._model is not None

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Artefact score in [0, 1]; higher means less like a genuine capture."""
        if self._model is None or df.empty:
            return np.zeros(len(df))
        raw = self._matrix(df)
        # `score_samples` is higher for inliers, so it is negated and squashed. The scale
        # constant only affects readability of the number, not the ordering or the threshold.
        inlier = self._model.score_samples(self._scaler.transform(raw))
        return 1.0 / (1.0 + np.exp(8.0 * (inlier + 0.5)))

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write ``media_artefact_score`` onto the frame, zero where nothing was captured."""
        out = df.copy()
        out["media_artefact_score"] = 0.0
        out["media_artefact_flag"] = 0
        if self._model is None:
            return out
        captured = has_capture(out)
        if not captured.any():
            return out
        scores = np.zeros(len(out))
        scores[captured] = self.score(out[captured])
        out["media_artefact_score"] = np.round(scores, 5)
        out["media_artefact_flag"] = ((scores >= self.threshold) & captured).astype(int)
        return out

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        base = df[self._columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        # Cross-modality consistency, which is where a synthesised capture gives itself away
        # more often than on any single score. A genuine poor capture degrades together; a
        # cloned voice presented against a real liveness check does not.
        extras = []
        if {"voice_match_score", "liveness_score"} <= set(self._columns):
            voice = df["voice_match_score"].to_numpy(float)
            liveness = df["liveness_score"].to_numpy(float)
            extras.append(voice - liveness)
        if {"biometric_match_score", "liveness_score"} <= set(self._columns):
            extras.append(df["biometric_match_score"].to_numpy(float)
                          - df["liveness_score"].to_numpy(float))
        if extras:
            base = np.column_stack([base] + extras)
        return base

    def _report(self, train: pd.DataFrame, test: Optional[pd.DataFrame],
                fitted_on: int) -> None:
        frame = test if test is not None and not test.empty else train
        captured = has_capture(frame)
        coverage = float(captured.mean()) if len(frame) else 0.0
        subset = frame[captured]
        if subset.empty:
            self.report = DeepfakeGuardReport(fitted_on, 0, self.threshold, coverage,
                                              float("nan"), {})
            return

        scores = self.score(subset)
        flagged = scores >= self.threshold
        fraud = subset["is_fraud"].to_numpy() == 1
        per_vector: Dict[str, float] = {}
        if fraud.any():
            per_vector = {
                str(v): float(flagged[(subset["attack_vector_id"] == v).to_numpy() & fraud].mean())
                for v in sorted(subset.loc[fraud, "attack_vector_id"].unique())
            }
        self.report = DeepfakeGuardReport(
            fitted_on=fitted_on,
            scored=int(len(subset)),
            threshold=self.threshold,
            coverage=coverage,
            recall_at_1pct_fpr=float(flagged[fraud].mean()) if fraud.any() else float("nan"),
            per_vector_recall=per_vector,
        )
