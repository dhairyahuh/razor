"""Two things an analyst needs that a score and a reason code do not provide.

**What normal looks like.** A reason code naming ``payee_account_age_days`` tells an analyst
which field mattered. It does not tell them that the value on this payment is 2 when the
legitimate population sits at 840. Without the comparison the field is a number on a screen;
with it, the deviation reads at a glance. So the baselines here summarise the legitimate
distribution of every numeric input, computed on the training window only - a baseline that
included the test window would be describing data the deployed model never saw.

**What would have had to be different.** The counterfactual answers the question a customer
asks on the phone: not "why is this risky" but "what would have made it fine". It is
computed by actually re-scoring perturbed copies of the payment through the deployed model,
so the answer is a property of the model rather than a story told about it.

One caveat travels with every counterfactual and the UI is expected to print it: this is a
statement about the model's decision surface, not about the world. "Approved if the
beneficiary had been on file three days" means the model would not have alerted, not that
the payment would have been safe. On a fraud that used an aged mule account the two come
apart completely, and that gap is the honest description of what a counterfactual is worth.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset import input_columns

#: Fields a counterfactual is allowed to move, with the direction that would reduce risk and
#: the plain-English frame for the sentence. Restricted on purpose. The model reads 170-odd
#: columns and most of them are derived aggregates whose counterfactual is meaningless -
#: "approved if the payer's 24-hour velocity had been 3 instead of 7" is not something anyone
#: can act on or verify. These are the fields an analyst can check against a document, a call
#: recording or the beneficiary record.
ACTIONABLE: Dict[str, Dict[str, object]] = {
    "payee_account_age_days": {
        "label": "the beneficiary account had been open",
        "unit": "days", "direction": "increase",
    },
    "payee_added_minutes_ago": {
        "label": "the beneficiary had been on file",
        "unit": "minutes", "direction": "increase",
    },
    "payee_is_first_time": {
        "label": "the payer had used this beneficiary before",
        "unit": "flag", "direction": "decrease",
    },
    "amount": {
        "label": "the amount had been",
        "unit": "currency", "direction": "decrease",
    },
    "call_in_progress": {
        "label": "no call had been in progress",
        "unit": "flag", "direction": "decrease",
    },
    "screen_share_active": {
        "label": "no screen share had been active",
        "unit": "flag", "direction": "decrease",
    },
    "remote_access_app_detected": {
        "label": "no remote-access tool had been present",
        "unit": "flag", "direction": "decrease",
    },
    "payee_name_match_score": {
        "label": "the beneficiary name had matched at",
        "unit": "score", "direction": "increase",
    },
    "device_is_new": {
        "label": "the device had been recognised",
        "unit": "flag", "direction": "decrease",
    },
    "auth_attempts": {
        "label": "authentication had succeeded first time",
        "unit": "count", "direction": "decrease",
    },
}

#: Values tried per field, as quantiles of the legitimate distribution. Sweeping the benign
#: range rather than an arbitrary grid keeps every candidate a value some real legitimate
#: payment actually had, so the answer cannot be "approved if the amount had been ₹-4,000".
SWEEP_QUANTILES = (0.05, 0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 0.95)


def benign_baselines(train: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Distribution of every numeric model input across legitimate training traffic."""
    if train.empty:
        return pd.DataFrame()
    legit = train[train["is_fraud"] == 0] if "is_fraud" in train.columns else train
    if legit.empty:
        return pd.DataFrame()

    cols = columns if columns is not None else input_columns(train)
    rows = []
    for column in cols:
        if column not in legit.columns:
            continue
        values = pd.to_numeric(legit[column], errors="coerce").dropna()
        # Categorical columns arrive as strings and coerce to an empty series. They are not
        # summarisable as quantiles and the UI renders them as categories instead.
        if values.empty:
            continue
        quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({
            "column": column,
            "n": int(values.size),
            "mean": round(float(values.mean()), 4),
            "p05": round(float(quantiles.loc[0.05]), 4),
            "p25": round(float(quantiles.loc[0.25]), 4),
            "median": round(float(quantiles.loc[0.50]), 4),
            "p75": round(float(quantiles.loc[0.75]), 4),
            "p95": round(float(quantiles.loc[0.95]), 4),
            "share_zero": round(float((values == 0).mean()), 4),
        })
    return pd.DataFrame(rows)


def counterfactual(row: pd.DataFrame, predict: Callable[[pd.DataFrame], np.ndarray],
                   threshold: float, baselines: pd.DataFrame,
                   max_results: int = 3) -> List[Dict[str, object]]:
    """Single-field changes that would have taken this payment below the alert threshold.

    Every candidate is scored through ``predict``, which is the deployed model, in one
    batched call. Returns the changes that actually flip the decision, smallest first by how
    far the field had to move relative to the legitimate spread - so "on file three days
    instead of four minutes" outranks "amount reduced by 96%", which is technically a
    counterfactual and practically useless.

    An empty list is a real and interesting answer: it means no single actionable field
    would have changed the outcome, and the alert rests on a combination.
    """
    if row.empty or baselines.empty:
        return []

    base = row.iloc[[0]]
    if float(predict(base)[0]) < threshold:
        return []

    stats = baselines.set_index("column")
    candidates: List[Dict[str, object]] = []
    frames: List[pd.DataFrame] = []

    for column, meta in ACTIONABLE.items():
        if column not in base.columns or column not in stats.index:
            continue
        current = pd.to_numeric(pd.Series([base[column].iloc[0]]), errors="coerce").iloc[0]
        if pd.isna(current):
            continue

        reference = stats.loc[column]
        if meta["unit"] == "flag":
            trials = [0.0] if current else []
        else:
            legit = pd.to_numeric(reference[["p05", "p25", "median", "p75", "p95"]],
                                  errors="coerce").dropna()
            if legit.empty:
                continue
            trials = sorted({round(float(v), 4) for v in legit})
            trials = ([t for t in trials if t > current] if meta["direction"] == "increase"
                      else [t for t in trials if t < current])

        for value in trials:
            trial = base.copy()
            trial[column] = value
            frames.append(trial)
            candidates.append({"column": column, "from": float(current), "to": float(value),
                               "label": meta["label"], "unit": meta["unit"],
                               "spread": float(reference["p95"] - reference["p05"]) or 1.0})

    if not frames:
        return []

    scores = np.asarray(predict(pd.concat(frames, ignore_index=True)), dtype=float)
    flipped = []
    for candidate, score in zip(candidates, scores):
        if score >= threshold:
            continue
        # Distance in units of the legitimate spread, so fields on different scales compare.
        move = abs(candidate["to"] - candidate["from"]) / max(abs(candidate["spread"]), 1e-9)
        flipped.append({**{k: v for k, v in candidate.items() if k != "spread"},
                        "score_after": round(float(score), 6),
                        "normalised_move": round(float(move), 4)})

    flipped.sort(key=lambda c: c["normalised_move"])
    # One per field: five variations on "a larger beneficiary age would have helped" is one
    # finding, and the smallest sufficient change is the only one worth stating.
    seen, unique = set(), []
    for entry in flipped:
        if entry["column"] in seen:
            continue
        seen.add(entry["column"])
        unique.append(entry)
    return unique[:max_results]


def phrase(entry: Dict[str, object], currency: str = "INR") -> str:
    """Render one counterfactual as the sentence an analyst would say out loud."""
    label, unit, to = entry["label"], entry["unit"], entry["to"]
    if unit == "flag":
        return f"Approved if {label}."
    if unit == "currency":
        return f"Approved if {label} {currency} {to:,.0f} or less."
    if unit == "count":
        return f"Approved if {label}."
    if unit == "score":
        return f"Approved if {label} {to:.2f} or above."
    value = int(round(float(to)))
    return f"Approved if {label} {value:,} {unit}."
