"""Agentic commerce: attacks on delegated payment authority.

These vectors break the assumption every conventional fraud model rests on - that an
anomalous payment looks anomalous. When an AI agent holds a scoped payment credential, a
hijacked transaction is executed by the right agent, from the right session, against a
merchant the user's profile supports, with a valid token. Nothing in the device, geo,
velocity or behavioural blocks is unusual, because none of those blocks describe the thing
that was actually compromised: the agent's context window.

The generator therefore emits two linked planes:

* the **transaction**, carrying the delegated-authority envelope (intent artefact, token
  scope, tool-call count, untrusted-content volume);
* an internal ``_agent_payload`` tag naming the injection family, which
  :mod:`redteam.generate.agentic_corpus` expands into the actual adversarial text the
  agent ingested.

Note what the generator deliberately does *not* set: ``agent_injection_score``. That
column is the output of the defence's own content classifier, produced in the defence
pipeline and joined back. Letting the red team write it would hand the blue team a
perfect oracle and turn the hardest vector in the library into a one-feature lookup.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import AttackContext, const, dist
from .profiles import EpisodeProfile, ProfileAttackGenerator, pois, u

AGENT_MCCS = (5399, 5651, 5732, 5815, 5411)
GIFT_CARD_MCCS = (5947, 5815)


def _agent_footprint(ctx: AttackContext, df: pd.DataFrame, payload: str,
                     *, registered_p: float = 0.9, reputation=(0.55, 0.95),
                     asserts_identity_p: Optional[float] = None) -> None:
    """Everything true of any agent-initiated payment, hijacked or not."""
    n = len(df)
    rng = ctx.rng
    df["initiated_by_agent"] = 1
    df["channel"] = "agent_api"
    registered = rng.random(n) < registered_p
    df["agent_is_registered"] = registered.astype(int)

    # A hijacked-but-genuine agent still holds its own key, so it asserts and signs
    # normally: the payment is fraudulent and the identity is not. That is what makes
    # injection hard, and it is why identity verification is reported as a separate control
    # rather than folded into the headline.
    #
    # An impersonator is the opposite case. It asserts a registered identity - copying the
    # user-agent string is the entire attack - and cannot produce a signature over the
    # request, because it does not have the key that identity is enrolled with.
    asserted = registered if asserts_identity_p is None else rng.random(n) < asserts_identity_p
    signed = asserted & registered
    df["agent_identity_asserted"] = asserted.astype(int)
    df["agent_request_signature_valid"] = signed.astype(int)
    df["agent_reputation"] = np.round(rng.uniform(*reputation, n), 4)
    # No human is present, so there is no human telemetry to observe.
    df["keystroke_flight_mean_ms"] = 0.0
    df["keystroke_flight_cv"] = 0.0
    df["typing_burstiness"] = 0.0
    df["pointer_entropy"] = 0.0
    df["hesitation_events"] = 0
    df["amount_field_corrections"] = 0
    df["payee_field_pasted"] = 0
    df["form_fill_duration_s"] = np.round(rng.gamma(2.0, 0.9, n), 3)
    df["biometric_match_score"] = 0.0
    df["voice_match_score"] = 0.0
    df["liveness_score"] = 0.0
    df["call_in_progress"] = 0
    df["screen_share_active"] = 0
    df["_agent_payload"] = payload
    df["agent_injection_score"] = 0.0  # filled by the defence's content classifier


def _hijack(payload: str, **kw):
    def _post(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
        _agent_footprint(ctx, df, payload, **kw)
    return _post


PROFILES: Dict[str, EpisodeProfile] = {
    "AGENTIC-PROMPT-INJECTION-PAYEE": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "UPI_P2M", "RTP_FEDNOW"],
        amount_multiplier=(1.0, 14.0),
        payments=(1, 3),
        episode_span_hours=(0.01, 0.4),
        victim={"by_vulnerability": 0.0, "agentic": True},
        counterparty="mule",
        round_bias=0.25,
        signals={
            # The hijack's irreducible artefact: the settled payee is not the payee the
            # user's signed intent named.
            "intent_payee_match": const(0, 0.92),
            "intent_token_present": const(1, 0.95),
            "intent_signature_valid": const(1, 0.85),
            "intent_amount_delta_ratio": dist(u(0.05, 0.9), 0.55),
            # Reading a poisoned catalogue means ingesting far more third-party text and
            # making far more tool calls than an ordinary checkout.
            "agent_untrusted_content_tokens": dist(u(6_000, 48_000), 0.80),
            "agent_tool_calls": dist(pois(19.0), 0.75),
        },
        post=_hijack("payee_swap"),
    ),
    "AGENTIC-MCP-TOOL-POISONING": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "RTP_FEDNOW"],
        amount_multiplier=(1.5, 22.0),
        payments=(1, 4),
        episode_span_hours=(0.01, 2.0),
        victim={"by_vulnerability": 0.0, "agentic": True},
        counterparty="mule",
        signals={
            # A poisoned tool description drives the agent outside its granted scope and
            # into a long tail of unplanned tool invocations.
            "spt_scope_violation": const(1, 0.85),
            "agent_tool_calls": dist(pois(34.0), 0.85),
            "intent_signature_valid": const(0, 0.55),
            "intent_token_present": const(1, 0.8),
            "intent_payee_match": const(0, 0.7),
            "agent_untrusted_content_tokens": dist(u(2_000, 20_000), 0.6),
        },
        post=_hijack("tool_poisoning"),
    ),
    "AGENTIC-CONFUSED-DEPUTY": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "RTP_FEDNOW", "SEPA_INST"],
        amount_multiplier=(2.0, 26.0),
        payments=(1, 2),
        victim={"by_vulnerability": 0.0, "agentic": True},
        counterparty="mule",
        cross_border_p=0.3,
        signals={
            # A fully trusted, well-reputed agent - borrowing authority it was never
            # delegated. Reputation-based defences actively work against you here.
            "spt_scope_violation": const(1, 0.90),
            "intent_payee_match": const(0, 0.75),
            "intent_token_present": const(1, 0.9),
            "intent_signature_valid": const(1, 0.9),
            "agent_tool_calls": dist(pois(12.0), 0.6),
        },
        post=_hijack("confused_deputy", registered_p=0.98, reputation=(0.82, 0.99)),
    ),
    "AGENTIC-COUNTERFEIT-STOREFRONT": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "UPI_P2M"],
        amount_multiplier=(0.6, 5.0),
        payments=(1, 2),
        victim={"by_vulnerability": 0.0, "agentic": True},
        counterparty="counterfeit_merchant",
        counterfeit_mccs=AGENT_MCCS,
        counterfeit_domain_age=(1.0, 40.0),
        counterfeit_agent_optimised_p=0.97,
        round_bias=0.1,
        signals={
            # Nothing about the delegation is broken. The agent followed its instructions
            # exactly and bought from a merchant that does not exist. Detection has to come
            # from merchant provenance, not from the agent envelope.
            "intent_payee_match": const(1, 1.0),
            "intent_signature_valid": const(1, 1.0),
            "intent_token_present": const(1, 1.0),
            "merchant_is_agent_optimised": const(1, 1.0),
            "agent_untrusted_content_tokens": dist(u(3_000, 18_000), 0.6),
        },
        post=_hijack("aeo_storefront"),
    ),
    "AGENTIC-SPT-REPLAY": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "UPI_P2M", "RTP_FEDNOW"],
        amount_multiplier=(1.0, 12.0),
        payments=(2, 7),
        episode_span_hours=(0.02, 8.0),
        victim={"by_vulnerability": 0.0, "agentic": True},
        counterparty="mule",
        signals={
            # A scoped, time-bound token used repeatedly and long after issuance.
            "spt_reuse_count": dist(lambda rng, k: rng.integers(3, 22, k), 0.88),
            "intent_age_s": dist(u(4_000, 130_000), 0.80),
            "intent_token_present": const(1, 1.0),
            "intent_signature_valid": const(1, 0.65),
            "intent_payee_match": const(0, 0.55),
            "spt_scope_violation": const(1, 0.45),
            "ip_asn_is_hosting": const(1, 0.55),
        },
        post=_hijack("spt_replay", registered_p=0.75, reputation=(0.35, 0.85)),
    ),
    "AGENTIC-IMPERSONATED-CRAWLER": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP", "UPI_P2M"],
        amount_multiplier=(0.05, 0.5),
        payments=(6, 30),
        episode_span_hours=(0.05, 4.0),
        victim={"by_vulnerability": 0.0},
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(5815, 5399),
        counterfeit_domain_age=(10.0, 600.0),
        counterfeit_agent_optimised_p=0.80,
        amount_cap=900.0,
        amount_floor=1.0,
        round_bias=0.05,
        signals={
            # Spoofing a trusted shopping agent buys frictionless API access, but the
            # attacker cannot forge registration or reputation - only the user-agent string.
            "agent_is_registered": const(0, 0.85),
            "agent_reputation": dist(u(0.02, 0.30), 0.85),
            "ip_asn_is_hosting": const(1, 0.80),
            "intent_token_present": const(0, 0.85),
            "intent_signature_valid": const(0, 0.9),
            "intent_payee_match": const(0, 0.9),
            "agent_tool_calls": dist(pois(45.0), 0.8),
            "auth_attempts": dist(lambda rng, k: rng.integers(1, 4, k), 0.6),
        },
        # Asserting a trusted crawler's identity on almost every request is the attack, so
        # the assertion rate is high while the registration rate stays low. Web Bot Auth is
        # the control that separates the two.
        post=_hijack("crawler_spoof", registered_p=0.12, reputation=(0.02, 0.3),
                     asserts_identity_p=0.93),
    ),
    "AGENTIC-CART-STUFFING": EpisodeProfile(
        fraud_type="agentic_compromise",
        rails=["CARD_CNP"],
        amount_multiplier=(1.2, 4.5),
        payments=(1, 2),
        victim={"by_vulnerability": 0.0, "agentic": True},
        # The order is placed at the merchant the user actually wanted. Only the injected
        # extra line item is fraudulent, which is why merchant reputation cannot help.
        counterparty="real_merchant",
        real_merchant_mccs=GIFT_CARD_MCCS,
        round_bias=0.5,
        signals={
            # The user's actual purchase went through correctly. The fraud is the delta
            # between the signed intent and what was finally charged.
            "intent_amount_delta_ratio": dist(u(0.22, 1.6), 0.90),
            "intent_payee_match": const(1, 0.8),
            "intent_token_present": const(1, 0.95),
            "intent_signature_valid": const(1, 0.9),
            "agent_untrusted_content_tokens": dist(u(4_000, 26_000), 0.7),
        },
        post=_hijack("cart_stuffing"),
    ),
}


class AgenticGenerator(ProfileAttackGenerator):
    name = "agentic"
    PROFILES = PROFILES
