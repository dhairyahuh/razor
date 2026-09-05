"""Card, merchant and dispute abuse.

The structural difference from the APP and ATO families is *where the burst appears*.
A takeover concentrates many payments on one payer; card testing concentrates many payers
on one merchant. Any velocity feature keyed only on the customer is blind to it, which is
why the feature layer computes merchant-side aggregates as well.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...identify.library import AttackVector
from .base import (
    AttackContext,
    AttackGenerator,
    finalise,
    reset_labels,
    select_victims,
    set_rail,
    template_rows,
    to_counterfeit_merchant,
    to_real_merchant,
)
from .profiles import clamp_timestamps

LOW_FRICTION_MCCS = (5815, 5399, 7994, 4814)
HIGH_VALUE_MCCS = (5732, 5651, 4511, 7011)


class CardGenerator(AttackGenerator):
    name = "card"
    vector_ids = (
        "CARD-TESTING-BIN-ENUM",
        "CARD-DISTRIBUTED-MICROCHARGE",
        "CARD-FIRST-PARTY-DEEPFAKE-REFUND",
        "CARD-TRANSACTION-LAUNDERING",
    )

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        if n_events <= 0:
            return pd.DataFrame()
        if vector.id == "CARD-TESTING-BIN-ENUM":
            return self._card_testing(ctx, vector, n_events, concentrated=True)
        if vector.id == "CARD-DISTRIBUTED-MICROCHARGE":
            return self._card_testing(ctx, vector, n_events, concentrated=False)
        if vector.id == "CARD-FIRST-PARTY-DEEPFAKE-REFUND":
            return self._first_party(ctx, vector, n_events)
        return self._transaction_laundering(ctx, vector, n_events)

    # -- authorisation probing ------------------------------------------------------------
    def _card_testing(self, ctx: AttackContext, vector: AttackVector, n_events: int,
                      *, concentrated: bool) -> pd.DataFrame:
        """Validate stolen PANs with micro-authorisations.

        ``concentrated`` is the classic loud version: hundreds of probes against a single
        low-friction merchant inside a few minutes. The distributed variant is the same
        attack deliberately spread thin to stay under every per-merchant velocity rule -
        the same fraud with the opposite statistical signature.
        """
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = ctx.cap(int(rng.integers(25, 140) if concentrated else rng.integers(8, 30)), minimum=6)
            # Each probe is a different stolen card, i.e. a different payer.
            payers = select_victims(ctx, n, by_vulnerability=0.0)
            df = template_rows(ctx, payers)
            df = reset_labels(df)
            set_rail(ctx, df, ["CARD_CNP"], keep_channel=True)
            df["channel"] = rng.choice(["web", "agent_api"], size=n, p=[0.6, 0.4])
            df["auth_method"] = rng.choice(["cvv_only", "3ds2"], size=n, p=[0.82, 0.18])

            if concentrated:
                span_s = rng.uniform(120, 2_400)
                n_merch = 1
            else:
                span_s = rng.uniform(6 * 3600, 6 * 86400)
                n_merch = max(2, n // 3)
            offsets = np.sort(rng.uniform(0, span_s, n))
            # Card testing runs when the acquirer's risk team is thinnest, but it still has to
            # sit inside the calendar real traffic occupies.
            start_s = ctx.episode_start(span_s, night_bias=0.30)

            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

            # Probes are aimed at real, low-friction merchants: the attacker wants a
            # working authorisation endpoint, not a storefront of their own.
            to_real_merchant(ctx, df, mccs=LOW_FRICTION_MCCS, n_distinct=n_merch)

            # Probe amounts are tiny and often identical across attempts.
            df["amount"] = np.round(rng.choice([1.0, 5.0, 10.0, 20.0, 49.0, 99.0], size=n,
                                               p=[0.18, 0.22, 0.24, 0.16, 0.12, 0.08]), 2)
            df["account_balance_ratio"] = np.round(rng.uniform(0.00001, 0.001, n), 6)

            _machine_client(ctx, df, hosting_p=0.85 if concentrated else 0.45)
            df["auth_attempts"] = rng.integers(1, 4, n)
            df["step_up_triggered"] = (rng.random(n) < 0.1).astype(int)
            df["mfa_passed"] = (rng.random(n) < 0.35).astype(int)

            frames.append(finalise(df, vector, campaign, "card_fraud"))
        return pd.concat(frames, ignore_index=True)

    # -- first-party dispute abuse ---------------------------------------------------------
    def _first_party(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        """Legitimate-looking purchases that will be falsely disputed by voice clone.

        The authorisation itself is unremarkable, which is the point. What is observable at
        authorisation time is the *account's* dispute posture: an elevated history of prior
        claims, high-resale merchandise, and a support interaction attached to the order
        whose voiceprint sits just inside tolerance.
        """
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = int(rng.integers(2, 9))
            payer = select_victims(ctx, 1, by_vulnerability=0.0, corporate=False)
            df = template_rows(ctx, np.repeat(payer, n))
            df = reset_labels(df)
            set_rail(ctx, df, ["CARD_CNP", "CARD_CP"])

            offsets = np.sort(rng.uniform(0, rng.uniform(3600, 60 * 3600), n))
            start_s = ctx.episode_start(float(offsets[-1]), night_bias=0.22)
            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

            to_real_merchant(ctx, df, mccs=HIGH_VALUE_MCCS, n_distinct=int(rng.integers(1, 4)))
            df["amount"] = np.round(np.exp(rng.uniform(np.log(6_000), np.log(180_000), n)), 2)
            # Serial disputers carry a visible claims history.
            df["prior_fraud_reports"] = rng.integers(2, 9, n)
            # A support call is attached to the order; the clone clears the voiceprint
            # with less headroom than the genuine account holder normally does.
            m = ctx.tell(n, 0.7)
            voice = df["voice_match_score"].to_numpy().astype(float)
            voice[m] = rng.uniform(0.66, 0.92, int(m.sum()))
            df["voice_match_score"] = np.round(voice, 4)
            frames.append(finalise(df, vector, campaign, "first_party"))
        return pd.concat(frames, ignore_index=True)

    # -- front-merchant laundering ---------------------------------------------------------
    def _transaction_laundering(self, ctx: AttackContext, vector: AttackVector,
                                n_events: int) -> pd.DataFrame:
        """A benign-looking storefront processing volume for a prohibited business.

        The declared MCC and the observed behaviour disagree: a "grocery" merchant taking
        tens of thousands per ticket, cross-border, from payers who never come back.
        """
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = ctx.cap(int(rng.integers(15, 70)), minimum=6)
            payers = select_victims(ctx, n, by_vulnerability=0.0)
            df = template_rows(ctx, payers)
            df = reset_labels(df)
            set_rail(ctx, df, ["CARD_CNP"], keep_channel=True)
            df["channel"] = "web"

            offsets = np.sort(rng.uniform(0, rng.uniform(2, 9) * 86400, n))
            start_s = ctx.episode_start(float(offsets[-1]))
            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

            # Declared as an innocuous category; behaves nothing like one.
            to_counterfeit_merchant(ctx, df, mccs=(5411, 5912, 5999, 8299),
                                    domain_age=(30.0, 700.0), reuse=0.92,
                                    agent_optimised_p=0.1)
            df["payee_is_high_risk_category"] = 0
            df["amount"] = np.round(np.exp(rng.uniform(np.log(9_000), np.log(240_000), n)), 2)
            df["is_cross_border"] = (rng.random(n) < 0.55).astype(int)
            df["mfa_passed"] = 1
            frames.append(finalise(df, vector, campaign, "card_fraud"))
        return pd.concat(frames, ignore_index=True)


def _machine_client(ctx: AttackContext, df: pd.DataFrame, *, hosting_p: float) -> None:
    """Strip the human telemetry a bot cannot produce and stamp the infrastructure tells."""
    n = len(df)
    rng = ctx.rng
    df["ip_asn_is_hosting"] = (rng.random(n) < hosting_p * ctx.emission / 0.72).astype(int)
    df["vpn_or_proxy"] = (rng.random(n) < 0.6).astype(int)
    df["device_is_emulator"] = (rng.random(n) < 0.45).astype(int)
    df["device_is_new"] = 1
    df["device_age_days"] = np.round(rng.uniform(0, 2, n), 3)
    df["keystroke_flight_mean_ms"] = 0.0
    df["keystroke_flight_cv"] = 0.0
    df["typing_burstiness"] = 0.0
    df["pointer_entropy"] = np.round(rng.uniform(0.0, 1.2, n), 3)
    df["hesitation_events"] = 0
    df["amount_field_corrections"] = 0
    df["form_fill_duration_s"] = np.round(rng.uniform(0.2, 2.5, n), 3)
    df["session_duration_s"] = np.round(rng.uniform(1.0, 25.0, n), 2)
    df["biometric_match_score"] = 0.0
    df["liveness_score"] = 0.0
    df["initiated_by_agent"] = 0
