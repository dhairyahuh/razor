"""What each side of the arms race pays, so the loop optimises profit rather than evasion.

Why the loop needs a price list
-------------------------------
A co-evolution search whose fitness is ``1 - recall`` has one dominant strategy and it is a
degenerate one: make the fraud small, slow and harmless. Every evasion lever moves an attack
toward the benign distribution, and the cheapest way to look benign is to *be* benign. A
search rewarded purely for evasion will happily converge on ₹800 transfers to accounts it
already owns, report a triumphant 0.02 recall, and have discovered nothing, because nobody
runs a fraud operation to lose money slowly.

Real attackers are firms. They hold inventory (mule accounts), pay for tooling
(subscriptions to the dark-LLM services in the threat research), pay per unit for
consumables (residential proxy egress, seasoning float), and staff operator time. They
choose the tactic with the best return on that spend. Once the search is scored on **profit
after cost** instead of evasion, the degenerate strategy prices itself out: shrinking the
amount shrinks the numerator, and the aged accounts that defeat counterparty-novelty
scoring are the single most expensive line item on the sheet.

The defender is a firm too, and this is the framing a card network actually reasons in. Blue
does not maximise recall; blue minimises the sum of fraud losses, analyst review cost and
customer friction. A control that adds two points of recall and ten points of false
positives is a bad trade and the arithmetic here says so out loud.

On the numbers
--------------
Every constant below is an assumption, not a measurement, and the whole point of putting
them in a dataclass is that they are visible and changeable rather than buried in a fitness
function. The cost model is written into ``loop_cost_model.csv`` with every run, so a reader
can see exactly what the ROI column was conditioned on and substitute their own institution's
figures. Orders of magnitude are anchored where the threat research gives one: synthetic
identity kits around ₹400, dark-LLM subscriptions at $200/month, aged accounts as the
premium tier of mule brokerage.

``reimbursement_share`` encodes the UK Payment Systems Regulator's October 2024 mandate,
under which the sending and receiving PSPs split reimbursement 50/50. It is the reason a
sending institution's loss from an undetected authorised push payment is roughly half the
transferred amount rather than zero, and it is why inbound mule detection stopped being
somebody else's problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttackerCosts:
    """The red team's price list, in rupees.

    Two kinds of price appear here. Most levers cost **money**: an account to buy, a proxy
    to rent, an operator to pay. A few cost **conversion** instead, and those are the
    interesting ones. Suppressing the coercion telemetry means the operator is no longer
    screen-sharing with the victim and talking them through the screens, so fewer victims
    complete the payment. That is a real cost and it does not appear on any invoice, but a
    model that ignores it would conclude that hiding coercion tells is free, and then the
    search would switch them all off in round one and stop learning anything.
    """

    # --- fixed, per campaign -----------------------------------------------------------
    tooling_subscription: float = 1_500.0
    """Amortised share of a dark-LLM / phishing-kit subscription. FraudGPT was advertised at
    $200 a month; spread over the campaigns one subscription supports, this is the slice."""

    # --- per distinct counterparty account ---------------------------------------------
    fresh_mule_account: float = 600.0
    """A newly opened drop account with a synthetic identity behind it."""

    aged_mule_account: float = 3_500.0
    """An account with real inbound history, rented or bought from a broker. Roughly six
    times a fresh one, which is the premium the market charges precisely because it is what
    defeats counterparty-novelty features."""

    named_mule_account: float = 1_200.0
    """A drop account whose registered name matches the story being told to the victim, so
    Confirmation of Payee returns a match. Costs more than a fresh account because the
    identity has to be built to order rather than taken off the shelf."""

    payee_pre_registration: float = 40.0
    """Holding cost of registering a beneficiary days ahead of use instead of minutes."""

    seasoning_payment: float = 120.0
    """One benign-looking inbound payment used to give a drop account a history. The float
    has to come from somewhere and some of it does not come back."""

    # --- per fraudulent payment --------------------------------------------------------
    residential_proxy: float = 25.0
    """Egress through residential IP space rather than a hosting ASN, priced per payment."""

    operator_minutes: float = 15.0
    """Human operator time for a hand-paced session rather than a script."""

    rail_fee: float = 8.0
    """Scheme fee on each payment. Only material when the tactic structures one payment
    into many, which is exactly when it should be."""

    # --- conversion penalties, as a multiplier on value extracted -----------------------
    conversion_loss_no_call: float = 0.35
    """Share of victims lost when the operator cannot stay on the phone."""

    conversion_loss_no_remote: float = 0.20
    """Share lost when the operator cannot drive the victim's screen."""

    conversion_loss_per_extra_payment: float = 0.04
    """Share lost per additional payment a victim is asked to make. Every extra step is
    another chance for the victim to stop, which is the real limit on structuring."""

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in asdict(self).items()]
        )


@dataclass(frozen=True)
class DefenderCosts:
    """The blue team's price list, in rupees.

    Recall is not the objective. A bank that declined every payment would have perfect
    recall and no customers. These are the three currencies a fraud function actually
    trades between, and blue's move set is scored on their sum.
    """

    analyst_review: float = 250.0
    """Fully loaded cost of a human working one alert."""

    false_decline: float = 900.0
    """Lost margin plus the churn risk of wrongly stopping a genuine payment. Deliberately
    several times the review cost: a declined customer is far more expensive than a
    reviewed one, which is why the threshold search is not symmetric."""

    step_up_friction: float = 40.0
    """Abandonment and support cost of asking a genuine customer for extra authentication.
    Cheap per event, which is what makes friction a good move on a narrow slice and a
    terrible one applied broadly."""

    retrain: float = 75_000.0
    """Engineering, validation and model-risk sign-off for shipping a new model."""

    rule_maintenance: float = 25_000.0
    """Annualised cost of one more hand-written rule in the estate, amortised to a round.
    Non-zero because rules are never removed and every one of them has to be re-justified."""

    reimbursement_share: float = 0.5
    """Share of an undetected authorised push payment the sending institution reimburses,
    per the UK PSR mandate. Applied to the value of fraud that got through."""

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in asdict(self).items()]
        )


#: Which evasion levers cost money, and on what basis. Levers absent from this map are free:
#: choosing a busy hour or a non-round amount costs an attacker nothing at all.
_PER_PAYMENT_LEVERS: Mapping[str, str] = {
    "form_fill_duration_s": "operator_minutes",
    "ip_asn_is_hosting": "residential_proxy",
    "vpn_or_proxy": "residential_proxy",
}

#: Levers that buy a better class of account rather than a per-payment consumable.
_PER_ACCOUNT_LEVERS: Mapping[str, str] = {
    "payee_name_match_score": "named_mule_account",
    "_payee_reg_lag_s": "payee_pre_registration",
}

#: Levers that suppress a coercion tell, and the conversion each one costs.
_CONVERSION_LEVERS: Mapping[str, str] = {
    "call_in_progress": "conversion_loss_no_call",
    "screen_share_active": "conversion_loss_no_remote",
    "remote_access_app_detected": "conversion_loss_no_remote",
}


@dataclass
class CampaignEconomics:
    """The full profit-and-loss for one tactic run over one campaign."""

    value_at_risk: float
    """Total rupees the campaign attempts to move."""

    value_reaching_mule: float
    """Of that, what survives the conversion penalties the tactic incurred."""

    value_extracted: float
    """Of that, what the detector failed to stop."""

    cost: float
    profit: float
    roi: float
    """Profit per rupee spent. Negative means the tactic loses the attacker money."""

    cost_breakdown: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> Dict[str, object]:
        return {
            "value_at_risk": round(self.value_at_risk, 2),
            "value_reaching_mule": round(self.value_reaching_mule, 2),
            "value_extracted": round(self.value_extracted, 2),
            "attacker_cost": round(self.cost, 2),
            "attacker_profit": round(self.profit, 2),
            "attacker_roi": round(self.roi, 4),
        }


def conversion_multiplier(moves: tuple, structuring: int, costs: AttackerCosts) -> float:
    """How much of the attempted value actually reaches the mule, given the tactic.

    Penalties compound multiplicatively rather than adding, because they are independent
    chances for the victim to walk away. Adding them would let three levers take away 105%
    of the conversion, which is arithmetic no fraud operation has ever managed.
    """
    multiplier = 1.0
    seen: set = set()
    for move in moves:
        attribute = _CONVERSION_LEVERS.get(move)
        # Screen-share and remote-access suppression are the same operational loss counted
        # twice if both levers are pulled, so the penalty applies once.
        if attribute is None or attribute in seen:
            continue
        seen.add(attribute)
        multiplier *= 1.0 - getattr(costs, attribute)
    extra_payments = max(structuring - 1, 0)
    multiplier *= (1.0 - costs.conversion_loss_per_extra_payment) ** extra_payments
    return float(np.clip(multiplier, 0.0, 1.0))


def campaign_cost(*, moves: tuple, n_payments: int, n_distinct_payees: int,
                  n_aged_accounts: int, seasoning_payments: int,
                  structuring: int, costs: AttackerCosts) -> Dict[str, float]:
    """Itemise what the tactic cost to run, so the report can show the invoice.

    Returned as a breakdown rather than a total because "the aged accounts were 70% of the
    spend" is the finding, and a single number hides it.
    """
    breakdown: Dict[str, float] = {"tooling": costs.tooling_subscription}

    fresh = max(n_distinct_payees - n_aged_accounts, 0)
    named = fresh if "payee_name_match_score" in moves else 0
    breakdown["mule_accounts"] = (
        named * costs.named_mule_account + (fresh - named) * costs.fresh_mule_account
    )
    breakdown["aged_accounts"] = n_aged_accounts * costs.aged_mule_account
    breakdown["seasoning"] = seasoning_payments * costs.seasoning_payment

    if "_payee_reg_lag_s" in moves:
        breakdown["pre_registration"] = n_distinct_payees * costs.payee_pre_registration

    per_payment = 0.0
    charged: set = set()
    for move in moves:
        attribute = _PER_PAYMENT_LEVERS.get(move)
        # Hosting-ASN and VPN suppression are one proxy purchase, not two.
        if attribute is None or attribute in charged:
            continue
        charged.add(attribute)
        per_payment += getattr(costs, attribute)
    breakdown["per_payment_levers"] = per_payment * n_payments
    breakdown["rail_fees"] = costs.rail_fee * n_payments * max(structuring, 1)

    return {k: round(v, 2) for k, v in breakdown.items() if v > 0}


def score_campaign(*, amounts: np.ndarray, detected: np.ndarray, moves: tuple,
                   structuring: int, n_distinct_payees: int, n_aged_accounts: int,
                   seasoning_payments: int, costs: AttackerCosts) -> CampaignEconomics:
    """Price one tactic against one detector's verdicts.

    ``detected`` is the detector's alert decision per payment. Value extracted counts only
    the payments that got through, scaled by the conversion the tactic could sustain - so a
    tactic that evades perfectly by making the victim hang up earns nothing, which is the
    behaviour the whole cost model exists to produce.
    """
    amounts = np.asarray(amounts, dtype=float)
    detected = np.asarray(detected, dtype=bool)
    value_at_risk = float(amounts.sum())

    multiplier = conversion_multiplier(moves, structuring, costs)
    reaching = value_at_risk * multiplier
    extracted = float(amounts[~detected].sum()) * multiplier

    breakdown = campaign_cost(
        moves=moves, n_payments=int(amounts.size), n_distinct_payees=n_distinct_payees,
        n_aged_accounts=n_aged_accounts, seasoning_payments=seasoning_payments,
        structuring=structuring, costs=costs,
    )
    total = float(sum(breakdown.values()))
    profit = extracted - total
    return CampaignEconomics(
        value_at_risk=value_at_risk,
        value_reaching_mule=reaching,
        value_extracted=extracted,
        cost=total,
        profit=profit,
        # Guarded because a tactic with no purchased inputs still pays the subscription, so
        # the denominator is never actually zero - but a future price list could zero it.
        roi=float(profit / total) if total > 0 else float("nan"),
        cost_breakdown=breakdown,
    )


def defender_cost(*, fraud_amount_missed: float, n_alerts: int, n_false_positives: int,
                  n_step_ups: int, retrains: int, rules: int,
                  costs: DefenderCosts) -> Dict[str, float]:
    """Total cost of running the defence at a given operating point, itemised.

    The fraud term is the reimbursed share rather than the full transferred value, which is
    what the sending institution actually carries under the PSR split.
    """
    return {
        "fraud_losses": round(fraud_amount_missed * costs.reimbursement_share, 2),
        "alert_review": round(n_alerts * costs.analyst_review, 2),
        "false_declines": round(n_false_positives * costs.false_decline, 2),
        "customer_friction": round(n_step_ups * costs.step_up_friction, 2),
        "model_changes": round(retrains * costs.retrain + rules * costs.rule_maintenance, 2),
    }
