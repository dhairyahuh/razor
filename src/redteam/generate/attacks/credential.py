"""Credential, mandate and liquidation vectors that are pure episode profiles.

These sit apart from ``card.py``, ``rail.py`` and ``mule.py`` because those modules exist for
attacks whose shape is genuinely idiosyncratic - a fan-out graph, a burst concentrated on one
merchant, a probe sequence that adapts. Everything here is fully described by the six
decisions :class:`EpisodeProfile` already encodes, so writing a bespoke generator for them
would add code without adding fidelity.

What they have in common as a group is that none of them defeats an authentication control.
Each one either abuses a credential the bank issued correctly, a mandate the customer granted
knowingly, or a cash-out route that is entirely legal on its own. That makes them the hardest
kind of fraud to catch from the payment alone, and the reason they are worth simulating: the
evidence is almost never in the transaction, it is in what the counterparty does next.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import AttackContext, const, dist
from .profiles import EpisodeProfile, ProfileAttackGenerator, pois, u


def _customer_present(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
    """No agent, no delegated authority: an ordinary human-initiated payment."""
    df["initiated_by_agent"] = 0
    df["intent_token_present"] = 0
    df["intent_signature_valid"] = 0
    df["agent_injection_score"] = 0.0
    df["mfa_passed"] = 1


def _terminal_hop(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
    """Cash-out leg: the payer is the mule, not a victim.

    Money has already been laundered through one or more hops and is now leaving the banking
    system. The payer account is criminal-controlled, so its own history is short and its
    outbound-to-inbound ratio is the signal - not anything about the instruction itself.
    """
    _customer_present(ctx, df, seq)
    n = len(df)
    # A cash-out account is recently opened or recently repurposed, and holds nothing: it
    # exists to be emptied. Deliberately noisy rather than constant, because a column with a
    # single value under attack is a generator artefact rather than a fraud signal.
    df["account_balance_ratio"] = np.round(np.clip(ctx.rng.beta(5.0, 1.8, n), 0.02, 1.0), 5)
    df["customer_tenure_days"] = np.round(
        np.clip(ctx.rng.lognormal(3.6, 1.1, n), 2.0, 900.0), 1
    )
    df["prior_fraud_reports"] = (ctx.rng.random(n) < 0.06).astype(int)


PROFILES: Dict[str, EpisodeProfile] = {
    "CARD-TOKEN-PROVISIONING-FRAUD": EpisodeProfile(
        fraud_type="card_fraud",
        rails=["CARD_CP", "WALLET"],
        amount_multiplier=(1.5, 18.0),
        payments=(2, 6),
        episode_span_hours=(0.05, 8.0),
        victim={"by_vulnerability": 0.5},
        # The stolen card is provisioned into a wallet on the attacker's handset. From then
        # on every payment is a *card-present* tokenised transaction with full cryptographic
        # validity - which is why it clears cleanly and why the only evidence sits in the
        # provisioning event rather than in any of the payments that follow.
        counterparty="real_merchant",
        real_merchant_mccs=(5732, 5651, 5947, 4511, 5814),
        round_bias=0.20,
        signals={
            "device_is_new": const(1, 0.88),
            "device_age_days": dist(u(0.0, 2.5), 0.85),
            "recent_credential_change_h": dist(u(0.2, 20.0), 0.80),
            # Wallet provisioning is frequently authorised over the phone, and a cloned
            # voice clears the threshold with less headroom than a genuine speaker.
            "voice_match_score": dist(u(0.66, 0.90), 0.55),
            "geo_velocity_kmh": dist(u(120, 900), 0.50),
            "distance_from_home_km": dist(u(60, 1200), 0.55),
        },
        post=_customer_present,
    ),
    "CARD-SUBSCRIPTION-TRAP-FARM": EpisodeProfile(
        fraud_type="card_fraud",
        rails=["CARD_CNP"],
        amount_multiplier=(0.05, 0.6),
        payments=(4, 12),
        # Weeks, not hours. The whole design is to stay below the amount at which a customer
        # bothers to dispute and below the frequency at which a rule notices.
        episode_span_hours=(160.0, 900.0),
        channel="recurring",
        victim={"by_vulnerability": 0.7, "corporate": False},
        # Many shell storefronts behind one operator. Each looks like a small legitimate
        # digital merchant; the pattern only exists across them.
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(5968, 7311, 5815),
        counterfeit_domain_age=(20.0, 300.0),
        round_bias=0.15,
        amount_cap=3_000.0,
        amount_floor=39.0,
        signals={
            # Nothing about the individual charge is remarkable. That is the attack: the
            # victim genuinely entered their card once, so authentication, device and
            # geography are all correct, and the fraud is the recurrence itself.
            "form_fill_duration_s": dist(u(2.0, 9.0), 0.35),
        },
        post=_customer_present,
    ),
    "CARD-3DS-FRICTIONLESS-ABUSE": EpisodeProfile(
        fraud_type="card_fraud",
        rails=["CARD_CNP"],
        # Kept deliberately under the value at which most issuers stop offering the
        # frictionless path. Staying below the step-up threshold *is* the attack.
        amount_multiplier=(0.3, 2.2),
        payments=(3, 9),
        episode_span_hours=(0.3, 40.0),
        victim={"by_vulnerability": 0.3},
        counterparty="real_merchant",
        real_merchant_mccs=(5815, 5399, 7994, 5968),
        round_bias=0.10,
        amount_cap=30_000.0,
        signals={
            # The issuer's risk engine sees a low-value payment from a plausible session and
            # declines to challenge. Every payment is therefore authenticated to the network's
            # satisfaction with no cardholder interaction at all - and the absence of friction
            # is the only artefact, which no per-transaction rule can act on.
            "step_up_triggered": const(0, 0.94),
            "auth_method": const("3ds2", 0.90),
            "device_is_new": const(1, 0.55),
            "session_duration_s": dist(u(20.0, 150.0), 0.70),
            "hesitation_events": const(0, 0.80),
        },
        post=_customer_present,
    ),
    "RAIL-MANDATE-ABUSE": EpisodeProfile(
        fraud_type="card_fraud",
        rails=["UPI_P2M", "NEFT", "CARD_CNP"],
        # A mandate authorised for a small amount, then drawn at a much larger one. The
        # customer approved a recurring claim; they did not approve this size of claim.
        amount_multiplier=(2.5, 26.0),
        payments=(1, 4),
        episode_span_hours=(24.0, 720.0),
        channel="recurring",
        victim={"by_vulnerability": 0.8, "corporate": False},
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(5968, 7311, 6300),
        counterfeit_domain_age=(15.0, 240.0),
        round_bias=0.30,
        signals={
            # No live session at all: a standing instruction executing on schedule. Every
            # behavioural and device feature is absent by construction, which is precisely
            # why mandate abuse evades a defence built around session telemetry.
            "session_duration_s": const(0.0, 0.95),
            "form_fill_duration_s": const(0.0, 0.95),
            "step_up_triggered": const(0, 0.90),
        },
        post=_customer_present,
    ),
    "RAIL-SETTLEMENT-WINDOW-TIMING": EpisodeProfile(
        fraud_type="card_fraud",
        rails=["NEFT", "SEPA_INST"],
        amount_multiplier=(3.0, 30.0),
        payments=(2, 6),
        episode_span_hours=(0.2, 10.0),
        channel="web",
        victim={"by_vulnerability": 0.2, "corporate": True},
        # Submitted so the instruction lands after the last review cycle of the day and
        # settles before the next one begins, buying hours before anyone looks at it. On a
        # deferred-net rail that window is real; the payment itself is unremarkable.
        night_bias=0.72,
        cross_border_p=0.55,
        round_bias=0.25,
        signals={
            "iso20022_unstructured_ratio": dist(u(0.45, 0.9), 0.55),
            "iso20022_remittance_len": dist(u(20.0, 70.0), 0.50),
        },
        post=_customer_present,
    ),
    "MULE-CRYPTO-OFFRAMP": EpisodeProfile(
        fraud_type="mule_network",
        rails=["IMPS", "UPI_P2P", "SEPA_INST"],
        amount_multiplier=(1.5, 16.0),
        payments=(2, 7),
        episode_span_hours=(0.1, 20.0),
        victim={"by_vulnerability": 0.0},
        # The fiat leg is the only part a bank or card network ever sees, and it is worth
        # simulating precisely because the chain analytics everyone reaches for cannot see
        # it. The counterparty is a real, registered exchange - not a counterfeit anything -
        # so payee reputation is no help at all.
        counterparty="real_merchant",
        real_merchant_mccs=(6051, 6211),
        night_bias=0.35,
        round_bias=0.55,
        signals={
            "payee_is_high_risk_category": const(1, 0.90),
            "is_cross_border": const(1, 0.45),
        },
        post=_terminal_hop,
    ),
    "MULE-GIFT-CARD-LIQUIDATION": EpisodeProfile(
        fraud_type="mule_network",
        rails=["CARD_CNP", "WALLET", "UPI_P2M"],
        # Deliberately fragmented: stored-value products carry per-purchase caps, so the
        # cash-out is many small buys rather than one large one. The fragmentation is
        # forced by the product, not chosen to evade - which is what makes it convincing.
        amount_multiplier=(0.15, 1.4),
        payments=(5, 16),
        episode_span_hours=(0.2, 30.0),
        victim={"by_vulnerability": 0.0},
        counterparty="real_merchant",
        real_merchant_mccs=(5947, 5815, 7994),
        round_bias=0.80,
        amount_cap=50_000.0,
        signals={
            "payee_is_high_risk_category": const(1, 0.45),
        },
        post=_terminal_hop,
    ),
}


class CredentialGenerator(ProfileAttackGenerator):
    name = "credential"
    PROFILES = PROFILES
