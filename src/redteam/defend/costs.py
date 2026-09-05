"""The operating point as a money decision rather than a metric.

Recall at a fixed false-positive budget is the right way to *compare* detectors, but it is
not how the threshold gets chosen in an institution. The budget is itself a proxy: someone
decided the alert queue can absorb so many reviews a day, and that decision encoded an
implicit price on an analyst's hour and on a wrongly declined customer. Making those prices
explicit turns the threshold from a hyperparameter into an arithmetic problem with an
answer, and it lets the report say the one thing a business reader actually wants: what is
this worth, in rupees.

Two numbers come out of that and they are different.

**The loss-minimising threshold** is where total cost bottoms out. It will usually not equal
the deployed threshold, because the deployed one is set to a false-positive budget instead,
and the gap between them is worth showing rather than hiding - it is the price of running
the queue at a fixed size.

**Net benefit** is the deployed operating point measured against doing nothing at all. Not
against a perfect detector, which is unachievable, and not against a random one, which
nobody would deploy: against the counterfactual where every payment is approved and every
fraud is reimbursed. That is the decision an institution is actually making when it funds a
fraud function, and it is the only framing in which a single rupee figure is meaningful.

The price list is :class:`redteam.loop.economics.DefenderCosts`, reused deliberately rather
than redefined here, so that the cost the defence reports and the cost the co-evolution loop
optimises against cannot drift apart. Every constant in it is an assumption and the curve
inherits that: a reader who disagrees with ₹900 for a false decline should be able to see
the number, change it, and get a different answer. The artefact therefore carries the price
list alongside the curve, and the UI is expected to show it. A net-benefit figure quoted
without its assumptions is a sales number.

Fixed costs - retraining, rule maintenance - are excluded. They do not vary with the
threshold, so including them would shift every row by the same constant and make the curve
harder to read while changing nothing about where its minimum sits.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: How many thresholds to price. Dense enough that the minimum is located to within a
#: rounding error of the score scale, cheap enough to recompute on every run.
GRID = 240


def cost_curve(test: pd.DataFrame, scores: np.ndarray, deployed_threshold: float,
               costs: Optional[object] = None) -> pd.DataFrame:
    """Total defence cost across the threshold range, itemised.

    One row per candidate threshold. ``is_deployed`` and ``is_optimal`` mark the two rows a
    reader needs to find, so the UI does not have to re-derive them and risk disagreeing
    with the report about which point is which.
    """
    # Imported inside the function because `redteam.loop` pulls in the co-evolution module,
    # which imports this package's pipeline. At module scope that is a genuine cycle; here
    # it resolves after both packages have finished loading.
    from ..loop.economics import DefenderCosts, defender_cost

    prices = costs if costs is not None else DefenderCosts()
    if test.empty or "is_fraud" not in test.columns:
        return pd.DataFrame()

    scores = np.asarray(scores, dtype=float)
    y = test["is_fraud"].to_numpy().astype(int)
    amounts = test["amount"].to_numpy(dtype=float)
    if y.sum() == 0:
        return pd.DataFrame()

    fraud_value_total = float(amounts[y == 1].sum())

    # Candidate thresholds from the score distribution itself rather than a linear sweep of
    # [0, 1]: scores on a 0.8%-prevalence problem pile up near zero, and a uniform grid
    # would spend most of its rows in a region no threshold would ever be set.
    grid = np.unique(np.quantile(scores, np.linspace(0.90, 0.99995, GRID)))
    grid = np.unique(np.concatenate([grid, [float(deployed_threshold)]]))

    rows: List[Dict[str, float]] = []
    for thr in grid:
        alert = scores >= thr
        missed_value = float(amounts[(y == 1) & ~alert].sum())
        itemised = defender_cost(
            fraud_amount_missed=missed_value,
            n_alerts=int(alert.sum()),
            n_false_positives=int((alert & (y == 0)).sum()),
            # Step-up is a separate control in this system rather than a band of the score,
            # so no payment is charged friction here. Priced at zero rather than omitted so
            # the itemisation matches `defender_cost`'s signature and a future
            # score-banded step-up policy has somewhere to land.
            n_step_ups=0,
            retrains=0, rules=0, costs=prices,
        )
        total = float(sum(itemised.values()))
        rows.append({
            "threshold": round(float(thr), 6),
            "recall": round(float(alert[y == 1].mean()), 4),
            "false_positive_rate": round(float(alert[y == 0].mean()), 5),
            "alerts": int(alert.sum()),
            "alerts_per_10k": round(float(alert.mean()) * 10_000, 1),
            "fraud_value_missed": round(missed_value, 2),
            "fraud_losses": itemised["fraud_losses"],
            "alert_review": itemised["alert_review"],
            "false_declines": itemised["false_declines"],
            "total_cost": round(total, 2),
        })

    curve = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    # Cost of approving everything: the whole fraud bill, no review, no declines. The
    # baseline every other row is implicitly measured against.
    do_nothing = float(fraud_value_total * prices.reimbursement_share)
    curve["net_benefit"] = (do_nothing - curve["total_cost"]).round(2)

    optimal = int(curve["total_cost"].idxmin())
    deployed = int((curve["threshold"] - float(deployed_threshold)).abs().idxmin())
    curve["is_optimal"] = 0
    curve["is_deployed"] = 0
    curve.loc[optimal, "is_optimal"] = 1
    curve.loc[deployed, "is_deployed"] = 1
    return curve


def cost_summary(curve: pd.DataFrame, costs: Optional[object] = None) -> pd.DataFrame:
    """The four figures the cost view leads with, as a metric/value table.

    Separated from the curve because these are the numbers that get quoted, and a caller
    that has to locate ``is_deployed`` itself to quote them will eventually locate it
    differently from the way the report did.
    """
    from ..loop.economics import DefenderCosts

    if curve.empty:
        return pd.DataFrame()
    prices = costs if costs is not None else DefenderCosts()

    deployed = curve[curve["is_deployed"] == 1].iloc[0]
    optimal = curve[curve["is_optimal"] == 1].iloc[0]
    do_nothing = float(deployed["net_benefit"] + deployed["total_cost"])

    rows = [
        {"metric": "do_nothing_cost", "value": round(do_nothing, 2),
         "note": "every payment approved, reimbursed at the PSR share"},
        {"metric": "cost_at_deployed_threshold", "value": float(deployed["total_cost"]),
         "note": f"threshold {deployed['threshold']:.4f}, "
                 f"{deployed['alerts_per_10k']:.0f} alerts per 10k"},
        {"metric": "net_benefit_at_deployed", "value": float(deployed["net_benefit"]),
         "note": "rupees saved against approving everything"},
        {"metric": "cost_at_loss_minimising_threshold", "value": float(optimal["total_cost"]),
         "note": f"threshold {optimal['threshold']:.4f}, "
                 f"{optimal['alerts_per_10k']:.0f} alerts per 10k"},
        {"metric": "net_benefit_at_loss_minimising", "value": float(optimal["net_benefit"]),
         "note": "the best this price list can do at any threshold"},
        {"metric": "cost_of_budget_constraint",
         "value": round(float(deployed["total_cost"] - optimal["total_cost"]), 2),
         "note": "what running to a fixed alert budget costs versus minimising loss"},
        {"metric": "reimbursement_share", "value": prices.reimbursement_share,
         "note": "assumption: UK PSR 50/50 sending/receiving split"},
        {"metric": "analyst_review", "value": prices.analyst_review,
         "note": "assumption: fully loaded cost of working one alert"},
        {"metric": "false_decline", "value": prices.false_decline,
         "note": "assumption: lost margin plus churn risk on one wrong decline"},
    ]
    return pd.DataFrame(rows)
