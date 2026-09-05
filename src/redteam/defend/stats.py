"""Uncertainty for every number the report prints.

A per-vector recall computed on three rows can only take the values 0, 1/3, 2/3 or 1. Printing
`1.0000` is not wrong so much as it is meaningless, and a reader who does not check the row
count beside it will draw a conclusion the data cannot support. The functions here attach an
interval to every rate and suppress the cells too thin to interpret at all.

Wilson intervals rather than the textbook normal approximation, because the normal one is
worst exactly where this dataset lives: proportions near 0 or 1 on small samples, where it
happily produces bounds below zero or above one. Wilson stays inside [0, 1] and keeps roughly
nominal coverage down to single-digit counts.

For the headline numbers the uncertainty is not a simple binomial - recall at a fixed
false-positive budget depends on a threshold that is itself estimated from the negatives - so
those are bootstrapped instead, resampling rows and recomputing the threshold inside each
resample. Holding the threshold fixed across resamples would understate the interval by
pretending the operating point were known exactly.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

# Below this many trials a rate is reported but flagged as uninterpretable. Twenty is the
# conventional floor at which a Wilson interval on a proportion is narrow enough to order
# two cells against each other; under it, the interval spans most of the unit range.
MIN_ROWS_FOR_INTERPRETATION = 20

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion, clipped to [0, 1]."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    k, n = float(successes), float(trials)
    centre = (k + 0.5 * z * z) / (n + z * z)
    spread = (z / (n + z * z)) * np.sqrt(k * (n - k) / n + 0.25 * z * z)
    return (float(max(0.0, centre - spread)), float(min(1.0, centre + spread)))


def add_wilson(frame: pd.DataFrame, rate_col: str, count_col: str,
               prefix: str = "") -> pd.DataFrame:
    """Attach lower/upper bounds and a low-n flag beside an existing rate column.

    Operates on the rate rather than a success count so it can be applied to tables that
    have already aggregated, which is every table in the evaluation module.
    """
    if frame.empty or rate_col not in frame or count_col not in frame:
        return frame
    out = frame.copy()
    rates = out[rate_col].to_numpy(dtype=float)
    counts = out[count_col].to_numpy(dtype=float)
    successes = np.rint(np.nan_to_num(rates) * counts).astype(int)

    bounds = [wilson_interval(s, int(n)) for s, n in zip(successes, counts)]
    name = prefix or rate_col
    out[f"{name}_lo95"] = np.round([b[0] for b in bounds], 4)
    out[f"{name}_hi95"] = np.round([b[1] for b in bounds], 4)
    # Not dropped: a vector with four examples in the test window is itself a finding about
    # the generator's mix, and hiding the row would hide that too.
    out["n_sufficient"] = (counts >= MIN_ROWS_FOR_INTERPRETATION).astype(int)
    return out


def bootstrap_metric(y: np.ndarray, scores: np.ndarray,
                     metric: Callable[[np.ndarray, np.ndarray], float],
                     n_resamples: int = 1000, seed: int = 0,
                     stratified: bool = True) -> Dict[str, float]:
    """Percentile bootstrap for a metric of (labels, scores).

    Stratified by class by default. An unstratified resample of a 0.75%-prevalence dataset
    varies the number of positives by several percent between draws, so the resulting interval
    mixes sampling noise in the metric with sampling noise in the prevalence - two different
    questions. Stratifying answers the one that was asked.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return {"point": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "resamples": 0}

    draws = []
    for _ in range(n_resamples):
        if stratified:
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                  rng.choice(neg, len(neg), replace=True)])
        else:
            idx = rng.integers(0, len(y), len(y))
        try:
            draws.append(float(metric(y[idx], scores[idx])))
        except (ValueError, ZeroDivisionError):
            continue

    if not draws:
        return {"point": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "resamples": 0}
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "point": round(float(metric(y, scores)), 5),
        "lo95": round(float(np.percentile(arr, 2.5)), 5),
        "hi95": round(float(np.percentile(arr, 97.5)), 5),
        "resamples": int(len(arr)),
    }


def partial_auc(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.02) -> float:
    """Standardised partial AUC over [0, max_fpr], McClish-corrected to [0.5, 1].

    The full ROC AUC integrates over false-positive rates up to 100%, which on this problem
    means most of its area comes from operating points that would alert on one payment in
    three. No institution deploys there, so the region is not merely uninformative - it is
    where a model can bank most of a flattering-looking score. Restricting to the first two
    percent measures the only part of the curve anyone can use.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / max(tps[-1], 1)
    fpr = fps / max(fps[-1], 1)

    # Prepend the origin and interpolate the curve exactly at the cut so the area does not
    # depend on whether a threshold happens to land on max_fpr.
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    keep = fpr <= max_fpr
    fpr_c = np.concatenate([fpr[keep], [max_fpr]])
    tpr_c = np.concatenate([tpr[keep], [float(np.interp(max_fpr, fpr, tpr))]])

    area = float(np.trapezoid(tpr_c, fpr_c)) if hasattr(np, "trapezoid") else float(
        np.trapz(tpr_c, fpr_c)
    )
    # McClish: rescale so a random model reads 0.5 rather than shrinking with max_fpr.
    minimum = 0.5 * max_fpr * max_fpr
    maximum = max_fpr
    if maximum <= minimum:
        return float("nan")
    return float(0.5 * (1.0 + (area - minimum) / (maximum - minimum)))


def paired_bootstrap_difference(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                                metric: Callable[[np.ndarray, np.ndarray], float],
                                n_resamples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Interval on metric(b) - metric(a), resampling the same rows for both.

    Pairing matters: the two score vectors are highly correlated because they come from
    models trained on overlapping data, and comparing two independent intervals throws that
    correlation away and declares differences insignificant that are not. Used for loop
    round-over-round comparisons, where the README already concedes a single transaction is
    worth two to three points and the swings being reported are that size.
    """
    y = np.asarray(y).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return {"difference": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "p_two_sided": float("nan")}

    diffs = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            diffs.append(float(metric(y[idx], b[idx])) - float(metric(y[idx], a[idx])))
        except (ValueError, ZeroDivisionError):
            continue
    if not diffs:
        return {"difference": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
                "p_two_sided": float("nan")}

    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    observed = float(metric(y, b)) - float(metric(y, a))
    # Bootstrap p-value: the mass of the resampled difference distribution on the far side
    # of zero, doubled. Approximate, and reported as such.
    share = float((arr <= 0).mean()) if observed > 0 else float((arr >= 0).mean())
    return {
        "difference": round(observed, 5),
        "lo95": round(float(np.percentile(arr, 2.5)), 5),
        "hi95": round(float(np.percentile(arr, 97.5)), 5),
        "p_two_sided": round(min(1.0, 2.0 * share), 4),
    }


def summarise_seeds(frames: Sequence[Dict[str, float]]) -> pd.DataFrame:
    """Mean, sd and range for each metric across repeated runs with different seeds.

    One seed produces one dataset, and the metrics are properties of that dataset as much as
    of the model. Reporting mean and spread across seeds is the difference between a number
    and an estimate.
    """
    if not frames:
        return pd.DataFrame()
    frame = pd.DataFrame(list(frames))
    numeric = frame.select_dtypes(include="number")
    out = pd.DataFrame({
        "metric": numeric.columns,
        "mean": numeric.mean().to_numpy().round(5),
        "sd": numeric.std(ddof=1).to_numpy().round(5) if len(frame) > 1 else np.nan,
        "min": numeric.min().to_numpy().round(5),
        "max": numeric.max().to_numpy().round(5),
        "runs": len(frame),
    })
    return out.reset_index(drop=True)
