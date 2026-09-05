"""Account takeover: the payment is unauthorised, so identity telemetry is the tell.

The interesting design tension across this family is that the strongest ATO signals are
mutually exclusive. A SIM-swap takeover screams "new device, impossible travel"; an
accessibility-overlay hijack runs on the victim's own handset from their own living room
and has no device anomaly at all. A defence tuned only on device novelty catches the first
and is blind to the second, which is precisely why both are simulated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import AttackContext, const, dist
from .profiles import EpisodeProfile, ProfileAttackGenerator, pois, u


def _unauthorised(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
    """Common footprint of a takeover: no agent, and the balance is the target."""
    df["initiated_by_agent"] = 0
    df["intent_token_present"] = 0
    df["intent_signature_valid"] = 0
    # Takeovers drain toward the available balance rather than tracking normal spend.
    balance = ctx.pop.customers["balance"].to_numpy()[df["_customer_idx"].to_numpy()]
    share = ctx.rng.uniform(0.25, 0.95, len(df))
    blended = 0.55 * df["amount"].to_numpy() + 0.45 * balance * share
    df["amount"] = np.round(np.maximum(blended, 100.0), 2)
    df["account_balance_ratio"] = np.round(
        np.clip(df["amount"].to_numpy() / np.maximum(balance, 100.0), 0, 50), 5
    )


def _fresh_device(ctx: AttackContext, df: pd.DataFrame, seq: np.ndarray) -> None:
    """Attacker hardware: a device with no history against this customer."""
    _unauthorised(ctx, df, seq)
    n = len(df)
    fresh = ctx.tell(n, 0.85)
    age = df["device_age_days"].to_numpy().astype(float)
    age[fresh] = ctx.rng.uniform(0, 1.5, int(fresh.sum()))
    df["device_age_days"] = np.round(age, 3)
    df["device_is_new"] = (age < 7).astype(int)
    # A crew works from a handful of handsets, not one per victim and not one shared across
    # every campaign in the run. Both extremes are detectable for the wrong reason.
    n_fresh = int(fresh.sum())
    if n_fresh:
        minted = ctx.device_ids(n_fresh, n_devices=max(1, int(np.ceil(n_fresh / 3.5))))
        ids = df["device_id"].to_numpy().copy()
        ids[fresh] = minted
        df["device_id"] = ids


PROFILES: Dict[str, EpisodeProfile] = {
    "ATO-VOICE-CLONE-IVR": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["IMPS", "NEFT", "RTP_FEDNOW", "SEPA_INST"],
        amount_multiplier=(3.0, 30.0),
        payments=(1, 3),
        episode_span_hours=(0.1, 6.0),
        victim={"by_vulnerability": 0.5},
        channel="ivr",
        auth_method="voice_print",
        round_bias=0.35,
        signals={
            # Clones clear the voiceprint threshold, but usually with less headroom than a
            # genuine speaker - a margin, not a wall.
            "voice_match_score": dist(u(0.68, 0.93), 0.90),
            # The agent was talked into a reset moments before the payment.
            "recent_credential_change_h": dist(u(0.05, 4.0), 0.75),
            "auth_attempts": dist(pois(2.0), 0.45),
            "auth_latency_ms": dist(u(28_000, 190_000), 0.60),
            "step_up_triggered": const(1, 0.35),
        },
        post=_unauthorised,
    ),
    "ATO-SIM-SWAP-OTP": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["IMPS", "UPI_P2P", "RTP_FEDNOW"],
        amount_multiplier=(4.0, 40.0),
        payments=(1, 5),
        escalating=False,
        episode_span_hours=(0.02, 3.0),
        victim={"by_vulnerability": 0.4},
        auth_method="otp_sms",
        night_bias=0.30,
        signals={
            "sim_swap_recency_days": dist(u(0.2, 6.0), 0.92),
            "recent_credential_change_h": dist(u(0.1, 36.0), 0.80),
            # OTPs now land instantly on the attacker's SIM instead of traversing to the
            # victim's handset, which shortens the observed delivery-to-entry gap.
            "otp_delivery_delay_s": dist(u(0.5, 3.0), 0.55),
            "ip_country_mismatch": const(1, 0.35),
            "geo_velocity_kmh": dist(u(400, 1400), 0.55),
            "distance_from_home_km": dist(u(150, 3000), 0.60),
        },
        post=_fresh_device,
    ),
    "ATO-BEHAVIOURAL-MIMICRY": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["CARD_CNP", "IMPS", "UPI_P2P", "WALLET"],
        amount_multiplier=(2.0, 18.0),
        payments=(1, 4),
        episode_span_hours=(0.01, 0.6),
        victim={"by_vulnerability": 0.2},
        # Mimicry takeovers monetise by buying resellable goods at ordinary merchants.
        counterparty="real_merchant",
        real_merchant_mccs=(5732, 5651, 5947, 4511),
        round_bias=0.30,
        signals={
            # Synthetic trajectories inject jitter to clear variance thresholds, but the
            # injected variance is itself too well-behaved: the coefficient of variation
            # lands in a narrow band no real hand produces consistently.
            "keystroke_flight_cv": dist(u(0.16, 0.30), 0.80),
            "typing_burstiness": dist(u(0.18, 0.34), 0.70),
            "pointer_entropy": dist(u(5.4, 6.6), 0.65),
            "form_fill_duration_s": dist(u(6.0, 16.0), 0.70),
            "hesitation_events": dist(pois(0.15), 0.75),
            "amount_field_corrections": const(0, 0.85),
        },
        post=_fresh_device,
    ),
    "ATO-AITM-SESSION-THEFT": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["CARD_CNP", "IMPS", "RTP_FEDNOW"],
        amount_multiplier=(2.5, 22.0),
        payments=(1, 4),
        episode_span_hours=(0.01, 1.5),
        victim={"by_vulnerability": 0.3},
        channel="web",
        counterparty="real_merchant",
        real_merchant_mccs=(5732, 6051, 5947, 4511, 5651),
        signals={
            # The session is fully authenticated - it was relayed through the real bank -
            # but it now originates from proxy infrastructure.
            "ip_asn_is_hosting": const(1, 0.80),
            "vpn_or_proxy": const(1, 0.72),
            "ip_country_mismatch": const(1, 0.50),
            "geo_velocity_kmh": dist(u(600, 2200), 0.65),
            "distance_from_home_km": dist(u(800, 9000), 0.70),
            "auth_latency_ms": dist(u(300, 1500), 0.55),
        },
        post=_fresh_device,
    ),
    "ATO-KBA-OSINT-LLM": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["IMPS", "NEFT", "CARD_CNP"],
        amount_multiplier=(2.5, 24.0),
        payments=(1, 3),
        episode_span_hours=(0.05, 4.0),
        victim={"by_vulnerability": 0.6},
        # Knowledge-based verification over the phone, so the channel is the phone and the
        # credential is whatever the rail normally uses. Deliberately *not* given an
        # auth_method of its own: a categorical level that only ever appears under attack is
        # a perfect detector of the simulator, which is the artefact class this generator
        # spent most of its effort removing.
        channel="ivr",
        round_bias=0.35,
        signals={
            # The distinguishing footprint is *how well* verification goes. A genuine
            # customer fumbles their own security answers surprisingly often; an attacker
            # reading a model-assembled dossier does not. Passing first time, quickly, is
            # the anomaly - which makes this one of the few vectors where the defence has to
            # treat unusually clean authentication as evidence rather than reassurance.
            "auth_attempts": const(1, 0.90),
            "auth_latency_ms": dist(u(1_500, 9_000), 0.80),
            "recent_credential_change_h": dist(u(0.05, 2.5), 0.72),
            "step_up_triggered": const(0, 0.70),
        },
        post=_unauthorised,
    ),
    "ATO-PASSKEY-DOWNGRADE": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["IMPS", "RTP_FEDNOW", "CARD_CNP"],
        amount_multiplier=(3.0, 26.0),
        payments=(1, 3),
        episode_span_hours=(0.02, 2.5),
        victim={"by_vulnerability": 0.5},
        # The attack is the fallback: the account has phishing-resistant authentication
        # available, and the attacker forces the flow onto the weaker method the bank kept
        # for lost-device recovery. The payment then authenticates entirely legitimately.
        auth_method="otp_sms",
        night_bias=0.28,
        signals={
            "recent_credential_change_h": dist(u(0.1, 8.0), 0.85),
            "step_up_triggered": const(1, 0.55),
            "auth_attempts": dist(pois(2.6), 0.60),
            "otp_delivery_delay_s": dist(u(1.0, 12.0), 0.45),
        },
        post=_fresh_device,
    ),
    "ATO-ACCESSIBILITY-OVERLAY": EpisodeProfile(
        fraud_type="account_takeover",
        rails=["UPI_P2P", "UPI_P2M", "IMPS", "WALLET"],
        amount_multiplier=(2.0, 20.0),
        payments=(1, 6),
        escalating=True,
        escalation_factor=1.25,
        episode_span_hours=(0.01, 2.0),
        victim={"by_vulnerability": 1.1, "corporate": False},
        night_bias=0.45,
        signals={
            # Everything device-level is perfect: real handset, real SIM, home IP. The
            # only artefacts are the malware's own footprint and machine-paced input.
            "accessibility_service_active": const(1, 0.88),
            "device_is_rooted": const(1, 0.35),
            "remote_access_app_detected": const(1, 0.30),
            "app_backgrounded_count": dist(pois(5.0), 0.65),
            "form_fill_duration_s": dist(u(1.0, 5.0), 0.75),
            "hesitation_events": const(0, 0.80),
            "payee_field_pasted": const(1, 0.75),
        },
        post=_unauthorised,
    ),
}


class AtoGenerator(ProfileAttackGenerator):
    name = "ato"
    PROFILES = PROFILES
