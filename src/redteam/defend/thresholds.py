"""Choosing an operating point from a false-positive budget."""

from __future__ import annotations

import numpy as np


def threshold_for_budget(negative_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose realised false-positive rate stays within budget.

    A plain quantile is wrong whenever scores are heavily tied, which they are as soon as a
    calibrator pins most negatives to exactly the same value: ``quantile(negatives, 0.99)``
    returns that value, and a ``score >= threshold`` test then alerts on every negative
    while appearing to honour a 1% budget. Working from the sorted order and stepping just
    above the cutoff keeps the guarantee intact.
    """
    negatives = np.sort(np.asarray(negative_scores, dtype=float))
    if negatives.size == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * negatives.size))
    idx = max(negatives.size - allowed - 1, 0)
    return float(np.nextafter(negatives[idx], np.inf))
