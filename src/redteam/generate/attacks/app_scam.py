"""Authorized Push Payment scams: the victim authenticates correctly and pays.

Every control that verifies *who* is paying passes cleanly here - correct device, correct
credential, correct biometric, correct home IP. The only thing that is wrong is *why*.
So what distinguishes a voice-clone family emergency from an invoice redirection is not
the payment mechanism but the victim cohort, the amount envelope, the counterparty, and
which coercion telemetry leaks out of the session.

Several profiles emit multi-payment episodes because that is how the scams actually run:
grooming escalates over days, safe-account drains inside a single session.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import AttackContext, const, dist
from .profiles import EpisodeProfile, ProfileAttackGenerator, pois, u


def _authorised_by_victim(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
    """APP scams are, by definition, correctly authenticated by the real customer."""
    df["mfa_passed"] = 1
    df["auth_attempts"] = np.minimum(df["auth_attempts"].to_numpy(), 2)
    df["initiated_by_agent"] = 0
    df["intent_token_present"] = 0
    df["intent_signature_valid"] = 0
    df["agent_injection_score"] = 0.0


PROFILES: Dict[str, EpisodeProfile] = {
    "APP-VOICE-CLONE-FAMILY": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST"],
        amount_multiplier=(2.5, 22.0),
        payments=(1, 2),
        victim={"by_vulnerability": 1.2, "corporate": False, "min_age_band": 2},
        night_bias=0.22,
        signals={
            # A live call during the payment is the defining artefact of vishing.
            "call_in_progress": const(1, 0.88),
            # Panic compresses deliberation but multiplies typos.
            "form_fill_duration_s": dist(u(4.0, 22.0), 0.75),
            "amount_field_corrections": dist(pois(2.4), 0.70),
            "hesitation_events": dist(pois(3.2), 0.65),
            "payee_field_pasted": const(1, 0.60),
        },
        post=_authorised_by_victim,
    ),
    "APP-VOICE-CLONE-EXEC": EpisodeProfile(
        fraud_type="app_scam",
        rails=["RTP_FEDNOW", "SEPA_INST", "IMPS", "NEFT"],
        amount_multiplier=(4.0, 45.0),
        victim={"by_vulnerability": 0.3, "corporate": True},
        cross_border_p=0.42,
        channel="web",
        signals={
            "call_in_progress": const(1, 0.70),
            "form_fill_duration_s": dist(u(20.0, 90.0), 0.5),
            "step_up_triggered": const(1, 0.30),
            "iso20022_unstructured_ratio": dist(u(0.55, 0.95), 0.5),
        },
        post=_authorised_by_victim,
    ),
    "APP-DEEPFAKE-VIDEO-CALL": EpisodeProfile(
        fraud_type="app_scam",
        rails=["RTP_FEDNOW", "SEPA_INST", "NEFT"],
        amount_multiplier=(10.0, 120.0),
        payments=(2, 5),
        episode_span_hours=(0.5, 26.0),
        victim={"by_vulnerability": 0.2, "corporate": True},
        cross_border_p=0.60,
        channel="web",
        round_bias=0.60,
        signals={
            # A long authenticated session: the victim sat through the entire call.
            "session_duration_s": dist(u(900.0, 5400.0), 0.80),
            "call_in_progress": const(1, 0.55),
            "hesitation_events": dist(pois(1.2), 0.40),
        },
        post=_authorised_by_victim,
    ),
    "APP-BEC-INVOICE-REDIRECT": EpisodeProfile(
        fraud_type="app_scam",
        rails=["NEFT", "IMPS", "SEPA_INST", "RTP_FEDNOW"],
        amount_multiplier=(3.0, 30.0),
        payments=(1, 3),
        episode_span_hours=(2.0, 72.0),
        victim={"by_vulnerability": 0.15, "corporate": True},
        cross_border_p=0.35,
        channel="web",
        round_bias=0.15,  # invoice amounts are precise, not round
        signals={
            # The attacker reuses the real supplier's trading name, so Confirmation of
            # Payee very nearly matches - which is exactly why CoP alone does not save you.
            "payee_name_match_score": dist(u(0.72, 0.97), 0.85),
            "iso20022_remittance_len": dist(u(38.0, 120.0), 0.70),
            "iso20022_field_entropy": dist(u(4.3, 5.4), 0.60),
        },
        post=_authorised_by_victim,
    ),
    "APP-SAFE-ACCOUNT-SCREENSHARE": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST"],
        amount_multiplier=(3.0, 40.0),
        payments=(1, 4),
        escalating=True,
        escalation_factor=1.4,
        episode_span_hours=(0.05, 1.2),
        victim={"by_vulnerability": 1.4, "corporate": False},
        signals={
            "screen_share_active": const(1, 0.72),
            "remote_access_app_detected": const(1, 0.55),
            "accessibility_service_active": const(1, 0.45),
            "call_in_progress": const(1, 0.90),
            "session_duration_s": dist(u(600.0, 4200.0), 0.85),
            "hesitation_events": dist(pois(4.0), 0.70),
            "payee_field_pasted": const(1, 0.50),
        },
        post=_authorised_by_victim,
    ),
    "APP-PIG-BUTCHERING": EpisodeProfile(
        fraud_type="app_scam",
        rails=["IMPS", "UPI_P2P", "RTP_FEDNOW", "SEPA_INST"],
        amount_multiplier=(0.8, 12.0),
        payments=(3, 9),
        escalating=True,
        episode_span_hours=(24.0, 600.0),
        victim={"by_vulnerability": 1.0, "corporate": False},
        round_bias=0.55,
        signals={
            # No coercion telemetry whatsoever: the victim is calm and believes they are
            # investing. Per-payment, these look benign; only the escalating sequence and
            # the counterparty give them away.
            "hesitation_events": dist(pois(0.4), 0.60),
            "payee_is_high_risk_category": const(1, 0.40),
        },
        post=_authorised_by_victim,
    ),
    "APP-DEEPFAKE-INVESTMENT": EpisodeProfile(
        fraud_type="app_scam",
        rails=["IMPS", "UPI_P2M", "SEPA_INST", "CARD_CNP"],
        amount_multiplier=(2.0, 25.0),
        payments=(1, 3),
        escalating=True,
        episode_span_hours=(1.0, 200.0),
        victim={"by_vulnerability": 0.8, "corporate": False},
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(6051, 6211),
        counterfeit_domain_age=(1.0, 60.0),
        round_bias=0.60,
        post=_authorised_by_victim,
    ),
    "APP-PURCHASE-SCAM": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2M", "CARD_CNP", "WALLET"],
        amount_multiplier=(0.7, 6.0),
        victim={"by_vulnerability": 0.6, "corporate": False},
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(5399, 5651, 5732),
        counterfeit_domain_age=(1.0, 45.0),
        round_bias=0.20,
        signals={
            # Too-good-to-be-true pricing shortens deliberation rather than lengthening it.
            "form_fill_duration_s": dist(u(3.0, 18.0), 0.50),
        },
        post=_authorised_by_victim,
    ),
    "APP-COLLECT-REQUEST-ABUSE": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "UPI_P2M"],
        amount_multiplier=(0.5, 8.0),
        payments=(1, 2),
        victim={"by_vulnerability": 1.3, "corporate": False},
        round_bias=0.50,
        amount_cap=100_000.0,
        signals={
            # The victim believes they are *receiving* money, so they approve almost
            # instantly and without hesitation - the inverse of the usual scam profile.
            "form_fill_duration_s": dist(u(1.5, 8.0), 0.85),
            "hesitation_events": dist(pois(0.2), 0.70),
        },
        post=_authorised_by_victim,
    ),
    "APP-REFUND-OVERPAYMENT": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "IMPS", "RTP_FEDNOW"],
        amount_multiplier=(1.5, 14.0),
        payments=(1, 2),
        channel="mobile_app",
        victim={"by_vulnerability": 1.5, "corporate": False, "min_age_band": 2},
        round_bias=0.30,
        signals={
            # The pretext inverts the usual direction of travel: the victim is told a refund
            # was overpaid and is asked to *return* the difference. That makes the amount
            # oddly precise rather than round, and the payment feels like a correction rather
            # than a transfer, so hesitation is low despite a long coached session.
            "screen_share_active": const(1, 0.78),
            "remote_access_app_detected": const(1, 0.50),
            "call_in_progress": const(1, 0.92),
            "session_duration_s": dist(u(700.0, 3600.0), 0.80),
            "hesitation_events": dist(pois(1.1), 0.55),
        },
        post=_authorised_by_victim,
    ),
    "APP-GOVT-TAX-IMPERSONATION": EpisodeProfile(
        fraud_type="app_scam",
        rails=["IMPS", "NEFT", "UPI_P2M", "SEPA_INST"],
        amount_multiplier=(1.2, 16.0),
        payments=(1, 2),
        # A demand for money converts far better when the money is actually there, so the
        # campaign is scheduled for payday. Benign volume peaks then too, which is the point.
        salary_bias=0.45,
        victim={"by_vulnerability": 1.1, "corporate": False, "min_age_band": 3},
        round_bias=0.25,
        signals={
            # Authority pretexts produce compliance, not panic: the victim believes they are
            # settling a genuine liability, so the session is unremarkable except for its
            # length and the amount being named to them.
            "call_in_progress": const(1, 0.66),
            "form_fill_duration_s": dist(u(25.0, 140.0), 0.55),
            "payee_field_pasted": const(1, 0.55),
            "payee_name_match_score": dist(u(0.55, 0.88), 0.60),
        },
        post=_authorised_by_victim,
    ),
    "APP-CHARITY-DISASTER": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2M", "CARD_CNP"],
        amount_multiplier=(0.3, 2.6),
        payments=(1, 2),
        victim={"by_vulnerability": 0.5, "corporate": False},
        # A fake appeal runs through a storefront, not a personal account, and the tell is a
        # freshly registered domain taking small donations from many unrelated payers.
        counterparty="counterfeit_merchant",
        counterfeit_mccs=(8398, 5399),
        counterfeit_domain_age=(1.0, 21.0),
        round_bias=0.72,
        amount_cap=50_000.0,
        signals={
            # Nothing coercive at all. The donor is a willing participant acting on genuine
            # sympathy, which is what makes this one of the harder profiles: only the
            # counterparty's youth and its fan-in shape carry any evidence.
            "form_fill_duration_s": dist(u(8.0, 40.0), 0.45),
        },
        post=_authorised_by_victim,
    ),
    "APP-JOB-TASK-SCAM": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "IMPS", "WALLET"],
        amount_multiplier=(0.25, 5.0),
        payments=(3, 8),
        # The deposit ladder: small "task fees" that grow as the victim is shown fake
        # earnings. Same escalation mechanic as pig butchering on a much shorter clock,
        # because the victim expects to be paid within days rather than months.
        escalating=True,
        escalation_factor=1.7,
        episode_span_hours=(18.0, 260.0),
        victim={"by_vulnerability": 1.2, "corporate": False, "max_age_band": 3},
        round_bias=0.62,
        signals={
            "hesitation_events": dist(pois(0.6), 0.50),
            "app_backgrounded_count": dist(pois(2.2), 0.50),
        },
        post=_authorised_by_victim,
    ),
    "APP-REVERSE-VISHING-CALLBACK": EpisodeProfile(
        fraud_type="app_scam",
        rails=["UPI_P2P", "IMPS", "CARD_CNP"],
        amount_multiplier=(2.0, 20.0),
        payments=(1, 2),
        # The victim dials the attacker, having been primed by a fake invoice or renewal
        # notice. That inversion matters: no inbound call to trace, and every caller-ID and
        # inbound-call heuristic is looking the wrong way.
        channel="ivr",
        auth_method="voice_print",
        victim={"by_vulnerability": 1.3, "corporate": False, "min_age_band": 2},
        night_bias=0.12,
        round_bias=0.40,
        signals={
            "call_in_progress": const(1, 0.95),
            "session_duration_s": dist(u(600.0, 3000.0), 0.85),
            "screen_share_active": const(1, 0.35),
            # Authentication genuinely passes - this is the real customer on the phone,
            # correctly verified. It is what separates this from a voice-clone takeover,
            # where the biometric is the thing being defeated.
            "voice_match_score": dist(u(0.88, 0.99), 0.85),
            "hesitation_events": dist(pois(1.8), 0.55),
        },
        post=_authorised_by_victim,
    ),
    "APP-GEN-ALPHA-IAP": EpisodeProfile(
        fraud_type="app_scam",
        rails=["CARD_CNP", "WALLET", "UPI_P2M"],
        amount_multiplier=(0.4, 4.0),
        payments=(3, 14),
        episode_span_hours=(0.2, 30.0),
        victim={"by_vulnerability": 0.4, "corporate": False, "max_age_band": 3},
        # The purchases land at real, established gaming and gift-card merchants. Nothing
        # about the counterparty is suspicious; the anomaly is entirely behavioural.
        counterparty="real_merchant",
        real_merchant_mccs=(7994, 5947, 5815),
        night_bias=0.35,
        round_bias=0.35,
        amount_cap=25_000.0,
        signals={
            # The account holder's own device, home network and stored credential, but a
            # different human at the screen: faster and far more erratic than the
            # registered behavioural profile. Device and geo intelligence see nothing.
            "keystroke_flight_mean_ms": dist(u(70.0, 130.0), 0.70),
            "keystroke_flight_cv": dist(u(0.55, 0.95), 0.70),
            "typing_burstiness": dist(u(0.70, 0.98), 0.60),
            "app_backgrounded_count": dist(pois(4.0), 0.60),
        },
        post=_authorised_by_victim,
    ),
}


class AppScamGenerator(ProfileAttackGenerator):
    name = "app_scam"
    PROFILES = PROFILES
