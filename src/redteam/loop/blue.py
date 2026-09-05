"""Blue's move set, priced. A game where one side cannot move is not a game.

The problem with retrain-only
-----------------------------
In the original loop the defender's entire repertoire was "fit the model again on data that
now contains the attack". That is not what a fraud function does when a new tactic lands,
and it makes the co-evolution one-sided in a way that flatters both parties: red gets to
choose among twelve levers, blue gets a single button, and whether the button works is
mostly a question of how much of the new tactic ended up in the training window.

Real responses to a new tactic, in ascending order of how long they take to ship:

* **Hold.** Absorb the losses. Genuinely the right answer when the tactic is rare or
  low-value, and a defence that never chooses it is over-fitting to whatever it saw last.
* **Re-threshold.** Free and immediate. Trades false positives for recall along the existing
  score, and needs no model change or sign-off.
* **Add a rule.** A day's work. Catches the specific tactic exactly, carries a permanent
  maintenance cost, and stops working the moment the attacker moves off the condition.
* **Add friction.** Step up authentication on a slice rather than declining it. Cheap per
  event and it does not lose the customer outright, which makes it the right tool for a
  slice that is suspicious but not conclusive.
* **Retrain.** Weeks, including model-risk sign-off. The only response that generalises,
  and by far the most expensive.

Each is evaluated on the same held-out window and priced with the same
:class:`~redteam.loop.economics.DefenderCosts`, and blue picks the cheapest total. That is
the actual decision a fraud function makes, and it produces a defensible answer to the
question a card network will ask: not "what is your recall" but "what did the last point of
recall cost you, and was it worth buying".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..defend.dataset import Split
from ..defend.model import FraudDetector
from .economics import DefenderCosts, defender_cost

#: Share of fraudulent payments abandoned when the customer is stepped up. Not 1.0: an
#: attacker coaching a victim through a one-time code defeats step-up routinely, which is
#: the whole reason authorised push payment fraud works, and a model that assumed friction
#: was a perfect block would recommend friction everywhere.
STEP_UP_DEFEAT_RATE = 0.28

#: Share of genuine customers who abandon rather than complete a step-up. Small, but it is
#: multiplied by a very large denominator, which is what makes broad friction expensive.
STEP_UP_ABANDON_RATE = 0.04


@dataclass
class BlueMove:
    """One candidate response, with the numbers that justify choosing it."""

    name: str
    description: str
    recall: float
    false_positive_rate: float
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    detector: Optional[FraudDetector] = None
    threshold: Optional[float] = None
    rule: Optional["Rule"] = None
    step_up_rule: Optional["Rule"] = None

    @property
    def total_cost(self) -> float:
        return float(sum(self.cost_breakdown.values()))

    def describe(self) -> Dict[str, object]:
        return {
            "move": self.name,
            "detail": self.description,
            "recall": round(self.recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 5),
            **{f"cost_{k}": v for k, v in self.cost_breakdown.items()},
            "total_cost": round(self.total_cost, 2),
        }


@dataclass(frozen=True)
class Rule:
    """A one- or two-condition deterministic rule, of the kind a fraud analyst writes."""

    conditions: Tuple[Tuple[str, str, float], ...]

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(frame), dtype=bool)
        for column, operator, value in self.conditions:
            if column not in frame.columns:
                return np.zeros(len(frame), dtype=bool)
            series = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            with np.errstate(invalid="ignore"):
                mask &= (series >= value) if operator == ">=" else (series <= value)
        return mask

    def describe(self) -> str:
        return " AND ".join(f"{c} {o} {v:g}" for c, o, v in self.conditions)


def choose_move(*, split: Split, frozen: FraudDetector, retrained: FraudDetector,
                targeted: Sequence[str], costs: DefenderCosts,
                seed: int = 0) -> Tuple[BlueMove, pd.DataFrame]:
    """Evaluate every response to the round's surviving tactics and return the cheapest.

    ``retrained`` is passed in already fitted rather than fitted here, because the caller
    needs it for the round's reported recall regardless of whether blue ends up choosing to
    ship it. Fitting it twice would double the most expensive step in the loop to answer a
    question the caller has already answered.
    """
    test = split.test
    moves: List[BlueMove] = [
        _hold(test, frozen, costs),
        _rethreshold(split, frozen, costs),
        _retrain(test, retrained, costs),
    ]

    rule = _mine_rule(split, frozen, targeted)
    if rule is not None:
        moves.append(_add_rule(test, frozen, rule, costs))
        moves.append(_add_friction(test, frozen, rule, costs, seed))

    table = pd.DataFrame([m.describe() for m in moves])
    best = min(moves, key=lambda m: m.total_cost)
    return best, table


# --------------------------------------------------------------------------------------
# The moves
# --------------------------------------------------------------------------------------

def _outcome(test: pd.DataFrame, alert: np.ndarray, costs: DefenderCosts, *,
             extra: Optional[Dict[str, float]] = None,
             step_up: Optional[np.ndarray] = None,
             seed: int = 0) -> Tuple[float, float, Dict[str, float]]:
    """Recall, false positive rate and the itemised bill for one operating point.

    Step-up is modelled as a third outcome alongside approve and decline, because collapsing
    it into either one is what makes friction look like a free win. A stepped-up fraudulent
    payment is usually but not always stopped; a stepped-up genuine payment usually but not
    always completes, and the ones that do not are as expensive as an outright decline.
    Which specific rows fall on either side is drawn from a seeded generator rather than
    from row position, so the result does not correlate with the order the frame happens to
    be sorted in.
    """
    rng = np.random.default_rng(seed)
    fraud = test["is_fraud"].to_numpy() == 1
    amount = test["amount_inr" if "amount_inr" in test.columns
                  else "amount"].to_numpy().astype(float)

    caught = alert.copy()
    n_step_ups = 0
    n_abandoned = 0
    if step_up is not None:
        pending = step_up & ~alert
        stepped_fraud = pending & fraud
        caught = caught | (stepped_fraud
                           & (rng.random(len(test)) >= STEP_UP_DEFEAT_RATE))
        stepped_genuine = pending & ~fraud
        n_step_ups = int(stepped_genuine.sum())
        n_abandoned = int(rng.binomial(n_step_ups, STEP_UP_ABANDON_RATE))

    recall = float(caught[fraud].mean()) if fraud.any() else float("nan")
    # Reported on outright declines only. A step-up is friction, not a false positive, and
    # counting it as one would make the two moves indistinguishable in the headline column.
    fpr = float(alert[~fraud].mean()) if (~fraud).any() else float("nan")

    breakdown = defender_cost(
        fraud_amount_missed=float(amount[fraud & ~caught].sum()),
        n_alerts=int(alert.sum()),
        # Genuine customers stopped outright, plus the ones who gave up at the step-up. Both
        # are lost payments and both cost the same; the rest of the stepped-up population is
        # priced far lower, which is the entire argument for using friction.
        n_false_positives=int((alert & ~fraud).sum()) + n_abandoned,
        n_step_ups=n_step_ups - n_abandoned,
        retrains=0, rules=0, costs=costs,
    )
    breakdown.update(extra or {})
    return recall, fpr, breakdown


def _hold(test: pd.DataFrame, frozen: FraudDetector, costs: DefenderCosts) -> BlueMove:
    alert = frozen.predict_proba(test) >= frozen.threshold
    recall, fpr, breakdown = _outcome(test, alert, costs)
    return BlueMove("hold", "absorb the losses, change nothing", recall, fpr, breakdown,
                    detector=frozen, threshold=frozen.threshold)


def _rethreshold(split: Split, frozen: FraudDetector,
                 costs: DefenderCosts) -> BlueMove:
    """Re-cut the existing score at whatever point minimises total cost.

    Chosen on the calibration window and *reported* on the test window, because a threshold
    tuned on the data it is scored against is not a defence, it is a look-ahead.
    """
    calibration = split.calibration
    scores_cal = frozen.predict_proba(calibration)
    candidates = np.quantile(scores_cal, np.linspace(0.80, 0.9995, 40))

    best_threshold, best_cost = frozen.threshold, np.inf
    for threshold in candidates:
        _, _, breakdown = _outcome(calibration, scores_cal >= threshold, costs)
        total = sum(breakdown.values())
        if total < best_cost:
            best_threshold, best_cost = float(threshold), total

    test = split.test
    alert = frozen.predict_proba(test) >= best_threshold
    recall, fpr, breakdown = _outcome(test, alert, costs)
    return BlueMove("rethreshold", f"move the cut to {best_threshold:.4f}", recall, fpr,
                    breakdown, detector=frozen, threshold=best_threshold)


def _retrain(test: pd.DataFrame, retrained: FraudDetector,
             costs: DefenderCosts) -> BlueMove:
    alert = retrained.predict_proba(test) >= retrained.threshold
    recall, fpr, breakdown = _outcome(test, alert, costs,
                                      extra={"model_changes": costs.retrain})
    return BlueMove("retrain", "refit on data containing the new tactics", recall, fpr,
                    breakdown, detector=retrained, threshold=retrained.threshold)


def _add_rule(test: pd.DataFrame, frozen: FraudDetector, rule: Rule,
              costs: DefenderCosts) -> BlueMove:
    alert = (frozen.predict_proba(test) >= frozen.threshold) | rule.evaluate(test)
    recall, fpr, breakdown = _outcome(test, alert, costs,
                                      extra={"model_changes": costs.rule_maintenance})
    return BlueMove("add_rule", f"decline where {rule.describe()}", recall, fpr, breakdown,
                    detector=frozen, threshold=frozen.threshold, rule=rule)


def _add_friction(test: pd.DataFrame, frozen: FraudDetector, rule: Rule,
                  costs: DefenderCosts, seed: int = 0) -> BlueMove:
    alert = frozen.predict_proba(test) >= frozen.threshold
    recall, fpr, breakdown = _outcome(test, alert, costs, step_up=rule.evaluate(test),
                                      seed=seed,
                                      extra={"model_changes": costs.rule_maintenance})
    return BlueMove("add_friction", f"step up where {rule.describe()}", recall, fpr,
                    breakdown, detector=frozen, threshold=frozen.threshold,
                    step_up_rule=rule)


# --------------------------------------------------------------------------------------
# Rule mining
# --------------------------------------------------------------------------------------

def _mine_rule(split: Split, frozen: FraudDetector,
               targeted: Sequence[str], max_conditions: int = 2) -> Optional[Rule]:
    """Find a short numeric condition that fires on the missed tactic and rarely elsewhere.

    Mined on the *calibration* window so the test window stays honest, and restricted to the
    fraud the frozen model actually missed - a rule that re-catches what the model already
    catches buys nothing and still costs maintenance.

    Greedy and deliberately shallow. A deeper search would find conditions with better
    calibration numbers and worse everything else; an analyst writing a rule by hand gets
    one or two clauses, and the point of this move is to be the thing an analyst can
    actually ship on the day.
    """
    frame = split.calibration
    if frame.empty or not len(targeted):
        return None

    scores = frozen.predict_proba(frame)
    missed = ((frame["is_fraud"].to_numpy() == 1)
              & frame["attack_vector_id"].isin(list(targeted)).to_numpy()
              & (scores < frozen.threshold))
    legit = frame["is_fraud"].to_numpy() == 0
    if missed.sum() < 8:
        return None

    numeric = [c for c in frame.columns
               if c.startswith(("f_", "g_")) and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return None

    conditions: List[Tuple[str, str, float]] = []
    mask = np.ones(len(frame), dtype=bool)
    for _ in range(max_conditions):
        best, best_lift = None, 0.0
        for column in numeric:
            values = frame[column].to_numpy().astype(float)
            finite = np.isfinite(values)
            if not (finite & missed & mask).any():
                continue
            for operator, quantile in ((">=", 0.75), ("<=", 0.25)):
                cut = float(np.nanquantile(values[missed & mask & finite], quantile))
                fires = (values >= cut) if operator == ">=" else (values <= cut)
                fires &= finite
                hit = float((fires & missed & mask).sum()) / max(int((missed & mask).sum()), 1)
                noise = float((fires & legit & mask).mean())
                # Precision-like objective: catching 60% of the missed tactic while firing
                # on 3% of genuine traffic is a bad rule, and this arithmetic says so.
                lift = hit - 25.0 * noise
                if lift > best_lift:
                    best, best_lift = (column, operator, cut), lift
        if best is None:
            break
        conditions.append(best)
        mask &= Rule((best,)).evaluate(frame)

    if not conditions:
        return None
    rule = Rule(tuple(conditions))
    fires = rule.evaluate(frame)
    # A rule that fires on more than one genuine payment in two hundred is not shippable at
    # any recall, so it is not offered as a move at all.
    if not fires[missed].any() or float(fires[legit].mean()) > 0.005:
        return None
    return rule
