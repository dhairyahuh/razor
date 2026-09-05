"""Red's stolen copy of the detector, trained on approve/decline answers.

The threat this models
----------------------
``MODEL-ORACLE-PROBING`` is in the attack library and the generator emits amount ladders for
it, but until now nothing in the system actually *did* anything with the answers. That made
the vector decorative: a shape the defence could learn, with no consequence attached.

This is the consequence. An attacker who can submit payments and observe whether they clear
has a free labelling oracle for the bank's model. Every declined probe is a positive, every
cleared one a negative, and a few thousand of them are enough to fit a serviceable copy. The
attacker then optimises against their own copy - offline, instantly, and without tripping a
single velocity alarm - and only submits the tactics that already work. This is how adaptive
attackers behave in practice, and it is why a defence that is only tested against a fixed
attack set is being flattered.

It also happens to be the search's compute budget. Scoring a genome against the real
pipeline costs a feature rebuild; scoring it against the surrogate costs a matrix multiply.
The loop generates a large population, screens it through the surrogate, and spends the
expensive evaluation only on the handful that look promising. So the same mechanism that
makes the red team more realistic is what buys the round count that makes the curve mean
anything.

What is deliberately withheld
-----------------------------
The surrogate is fitted on the same *feature matrix* the defender uses, which is generous to
the attacker - a real one would have to reconstruct features from raw payment fields. That
generosity is intentional and stated: an attacker model that is too weak proves nothing when
the defence survives it. What the attacker does not get is the defender's labels, its
threshold, or its internals. It sees a binary decision per probe, which is exactly what
submitting a payment tells you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..defend.dataset import input_columns
from ..defend.model import FraudDetector


@dataclass
class SurrogateReport:
    """How good the attacker's copy actually is, which is a finding in its own right."""

    probes: int
    agreement: float
    """Share of held-out probes where the surrogate's decision matches the real oracle."""

    positive_rate: float
    """Share of probes the real oracle declined. A near-zero rate means the attacker
    learned almost nothing, because a classifier needs both classes."""

    def describe(self) -> dict:
        return {
            "oracle_probes": self.probes,
            "surrogate_agreement": round(self.agreement, 4),
            "oracle_decline_rate": round(self.positive_rate, 4),
        }


class SurrogateOracle:
    """A model of the defender's decision boundary, fitted from decisions alone."""

    def __init__(self, seed: int = 0, max_iter: int = 60) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self._model: Optional[HistGradientBoostingClassifier] = None
        self._columns: list = []
        self.report: Optional[SurrogateReport] = None

    def probe(self, detector: FraudDetector, frame: pd.DataFrame, *,
              budget: int = 4_000, rng: Optional[np.random.Generator] = None) -> "SurrogateOracle":
        """Submit ``budget`` payments, record approve/decline, and fit on the answers.

        The probe set is drawn from traffic the attacker could plausibly generate or observe
        rather than from the defender's held-out window, and it is capped: an attacker who
        could submit unlimited probes without being rate-limited would not need a surrogate,
        they would just brute-force the live system.
        """
        rng = rng or np.random.default_rng(self.seed)
        if frame.empty:
            return self

        take = min(budget, len(frame))
        # Over-weight fraud rows: the attacker probes with their own traffic, so their
        # sample is nothing like the bank's 1% prevalence and a uniform draw would hand them
        # a probe set with almost no declines in it.
        weights = np.where(frame["is_fraud"].to_numpy() == 1, 12.0, 1.0)
        picks = rng.choice(len(frame), size=take, replace=False, p=weights / weights.sum())
        probes = frame.iloc[picks]

        declined = (detector.predict_proba(probes) >= detector.threshold).astype(int)
        positive_rate = float(declined.mean())
        self._columns = [c for c in input_columns(probes) if c in probes.columns]

        # Both classes have to be present or there is nothing to separate.
        if positive_rate <= 0.01 or positive_rate >= 0.99 or take < 200:
            self._model = None
            self.report = SurrogateReport(take, float("nan"), positive_rate)
            return self

        split = int(take * 0.75)
        X = probes[self._columns]
        model = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=0.12, max_leaf_nodes=24,
            random_state=self.seed,
        ).fit(_numeric(X.iloc[:split]), declined[:split])

        predicted = model.predict(_numeric(X.iloc[split:]))
        agreement = float((predicted == declined[split:]).mean())

        self._model = model
        self.report = SurrogateReport(take, agreement, positive_rate)
        return self

    @property
    def usable(self) -> bool:
        return self._model is not None

    def decline_probability(self, frame: pd.DataFrame) -> np.ndarray:
        """The attacker's own estimate of being stopped. Higher is worse for the attacker."""
        if self._model is None or frame.empty:
            return np.full(len(frame), 0.5)
        missing = [c for c in self._columns if c not in frame.columns]
        if missing:
            return np.full(len(frame), 0.5)
        return self._model.predict_proba(_numeric(frame[self._columns]))[:, 1]


def _numeric(frame: pd.DataFrame) -> np.ndarray:
    """Categoricals as integer codes; the surrogate does not need the defender's encoder.

    An attacker reverse-engineering a model does not get the bank's preprocessing pipeline,
    and using it here would overstate their capability. Ordinal codes on a tree model are a
    fair approximation of what they could build themselves.
    """
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object or str(out[column].dtype) == "category":
            out[column] = pd.factorize(out[column])[0].astype(float)
    return out.to_numpy(dtype=float)
