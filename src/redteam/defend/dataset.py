"""Turning a generated campaign into a leakage-free training matrix.

Two rules are enforced here rather than trusted to discipline elsewhere.

*Only observable columns.* The label block, the red-team provenance columns and every
internal ``_``-prefixed field are dropped by construction. The model physically cannot see
``attack_vector_id`` or ``evasion_applied``.

*Temporal splits, never random ones.* Fraud is bursty and campaign-structured: a random
split puts half of a mule ring in train and half in test, and the resulting AUC is a
measurement of nothing. The split here is by wall-clock time, with the calibration slice
sitting between train and test so thresholds are set on data the model did not fit and the
test window still sits strictly in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..schema import CATEGORICAL_COLUMNS, LABEL_COLUMNS, META_COLUMNS, observable_columns

DERIVED_PREFIXES = ("f_", "g_")
EXTRA_INPUTS = ["agent_injection_score", "agent_injection_flag",
                "intent_guard_blocked", "intent_guard_stepup",
                "intent_guard_violation_count",
                "agent_control_blocked", "agent_control_stepup",
                # The two GenAI-native planes. Both are guard *outputs* rather than raw
                # telemetry, so they belong in the ablation grid's guard layer where their
                # contribution is priced separately from what a bank already has.
                "vishing_score", "vishing_flag",
                "media_artefact_score", "media_artefact_flag"]


@dataclass
class Split:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("calibration", self.calibration), ("test", self.test)):
            rows.append(
                {
                    "split": name,
                    "rows": len(part),
                    "fraud": int(part["is_fraud"].sum()),
                    "fraud_rate": round(float(part["is_fraud"].mean()), 5) if len(part) else 0.0,
                    "vectors": int(part.loc[part["is_fraud"] == 1, "attack_vector_id"].nunique()),
                    "start": part["timestamp"].min(),
                    "end": part["timestamp"].max(),
                }
            )
        return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame, *, test_days: int = 10, calibration_days: int = 5) -> Split:
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    end = df["timestamp"].max()
    test_start = end - pd.Timedelta(days=test_days)
    calib_start = test_start - pd.Timedelta(days=calibration_days)

    train = df[df["timestamp"] < calib_start]
    calibration = df[(df["timestamp"] >= calib_start) & (df["timestamp"] < test_start)]
    test = df[df["timestamp"] >= test_start]
    return Split(train.reset_index(drop=True), calibration.reset_index(drop=True), test.reset_index(drop=True))


def input_columns(df: pd.DataFrame) -> List[str]:
    """Every column the model is permitted to read."""
    forbidden = set(META_COLUMNS) | set(LABEL_COLUMNS)
    cols: List[str] = []
    for c in df.columns:
        if c in forbidden or c.startswith("_"):
            continue
        if c in ("intent_guard_reasons", "agent_control_reasons", "agent_control_ceiling"):
            continue
        if c in observable_columns() or c.startswith(DERIVED_PREFIXES) or c in EXTRA_INPUTS:
            cols.append(c)
    return cols


def column_layers(df: pd.DataFrame) -> "OrderedDict[str, List[str]]":
    """The model's inputs split into the four layers, cumulatively.

    Used by the ablation grid. The interesting question is not which features are important
    but which *layer* is: the raw schema is what a bank already has in a payment message, and
    everything above it is work this project did. A grid that shows the derived layers adding
    little would be a finding worth reporting, and one that shows them adding almost
    everything is the honest explanation for why the headline number is what it is.
    """
    from collections import OrderedDict

    full = input_columns(df)
    raw = [c for c in full if not c.startswith(DERIVED_PREFIXES) and c not in EXTRA_INPUTS]
    tabular = [c for c in full if c.startswith("f_")]
    graph = [c for c in full if c.startswith("g_")]
    guards = [c for c in full if c in EXTRA_INPUTS]

    layers: "OrderedDict[str, List[str]]" = OrderedDict()
    layers["raw_schema"] = raw
    layers["+ tabular"] = raw + tabular
    layers["+ graph"] = raw + tabular + graph
    layers["+ guards (full)"] = raw + tabular + graph + guards
    return layers


def categorical_columns(cols: List[str]) -> List[str]:
    return [c for c in CATEGORICAL_COLUMNS if c in cols]


class MatrixBuilder:
    """Encodes the frame into a float matrix, with categories learned once on train."""

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.categoricals = categorical_columns(columns)
        self.categories: dict = {}

    def fit(self, df: pd.DataFrame) -> "MatrixBuilder":
        for c in self.categoricals:
            self.categories[c] = pd.Index(sorted(df[c].astype(str).unique()))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # Filled column-by-column into a preallocated array rather than assembled as a
        # DataFrame: this runs once per candidate in the co-evolution search, and repeated
        # column insertion on a 180-column frame is the single slowest thing in that loop.
        out = np.empty((len(df), len(self.columns)), dtype=float)
        for i, c in enumerate(self.columns):
            if c in self.categoricals:
                # Unseen categories become NaN, which the gradient-boosted trees route as
                # missing. A previously unknown PSP is genuinely unknown, not category 0.
                codes = self.categories[c].get_indexer(df[c].astype(str))
                out[:, i] = np.where(codes < 0, np.nan, codes)
            else:
                out[:, i] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        return out

    @property
    def categorical_mask(self) -> np.ndarray:
        return np.array([c in self.categoricals for c in self.columns], dtype=bool)

    def n_categories(self) -> int:
        return max((len(v) for v in self.categories.values()), default=0)
