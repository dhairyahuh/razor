"""Payment rail and messaging-infrastructure abuse.

Two of these vectors are structural rather than behavioural and need bespoke construction:
an instant-rail burst is one payer fanning out to many payees inside the settlement
window, and QR tampering is many payers converging on one payee at a single physical
location. Both are invisible to per-transaction scoring and only appear once you look at
the shape of the flow.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ...identify.library import AttackVector
from ...schema import IRREVOCABLE_RAILS
from .base import (
    AttackContext,
    finalise,
    reset_labels,
    select_victims,
    set_rail,
    template_rows,
    to_mule,
)
from .base import const, dist
from .profiles import EpisodeProfile, ProfileAttackGenerator, clamp_timestamps, u

PROFILES: Dict[str, EpisodeProfile] = {
    "RAIL-ISO20022-FIELD-POISONING": EpisodeProfile(
        fraud_type="infrastructure_abuse",
        rails=["SEPA_INST", "RTP_FEDNOW", "NEFT"],
        amount_multiplier=(2.0, 30.0),
        payments=(1, 4),
        episode_span_hours=(0.5, 48.0),
        victim={"by_vulnerability": 0.1, "corporate": True},
        counterparty="mule",
        cross_border_p=0.65,
        channel="web",
        round_bias=0.1,
        signals={
            # Syntactically valid, semantically loaded: stuffed remittance blocks with
            # near-random filler that shifts the feature distribution the AML model learns.
            "iso20022_remittance_len": dist(u(140.0, 420.0), 0.85),
            "iso20022_field_entropy": dist(u(5.2, 7.4), 0.80),
            "iso20022_unstructured_ratio": dist(u(0.78, 0.99), 0.80),
        },
    ),
    "RAIL-COP-HOMOGLYPH-EVASION": EpisodeProfile(
        fraud_type="infrastructure_abuse",
        rails=["SEPA_INST", "RTP_FEDNOW", "IMPS", "NEFT"],
        amount_multiplier=(2.5, 28.0),
        payments=(1, 3),
        victim={"by_vulnerability": 0.5},
        counterparty="mule",
        round_bias=0.3,
        signals={
            # The whole point of the vector: clear the fuzzy name-match threshold while
            # naming a different legal entity.
            "payee_name_match_score": dist(u(0.86, 0.99), 0.90),
            # The script-mixing detector fires only when it is actually deployed and the
            # attacker has not normalised around it.
            "payee_name_homoglyph_flag": const(1, 0.45),
        },
    ),
    "RAIL-CROSS-BORDER-CORRIDOR-ARBITRAGE": EpisodeProfile(
        fraud_type="infrastructure_abuse",
        rails=["SEPA_INST", "RTP_FEDNOW", "NEFT"],
        amount_multiplier=(3.0, 35.0),
        payments=(2, 6),
        episode_span_hours=(0.2, 30.0),
        victim={"by_vulnerability": 0.1},
        counterparty="mule",
        mule_reuse=0.8,
        cross_border_p=1.0,
        channel="web",
        round_bias=0.2,
        signals={
            "iso20022_unstructured_ratio": dist(u(0.6, 0.95), 0.6),
            "iso20022_remittance_len": dist(u(10.0, 40.0), 0.5),
        },
    ),
}


class RailGenerator(ProfileAttackGenerator):
    name = "rail"
    PROFILES = PROFILES

    @property
    def vector_ids(self):  # type: ignore[override]
        return tuple(PROFILES) + ("RAIL-RTP-VELOCITY-BURST", "RAIL-STATIC-QR-TAMPERING")

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        if vector.id == "RAIL-RTP-VELOCITY-BURST":
            return self._velocity_burst(ctx, vector, n_events)
        if vector.id == "RAIL-STATIC-QR-TAMPERING":
            return self._qr_tampering(ctx, vector, n_events)
        return super().generate(ctx, vector, n_events)

    # -- one payer, many payees, inside the settlement window -----------------------------
    def _velocity_burst(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        """Split a drained balance across many instant transfers in one short window.

        The individual payments are unremarkable in size. The intervention window on an
        instant rail is measured in hundreds of milliseconds, so the only defence is
        cumulative intra-session state - which is exactly what this stresses.
        """
        if n_events <= 0:
            return pd.DataFrame()
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = ctx.cap(int(rng.integers(4, 18)), minimum=3)
            payer = select_victims(ctx, 1, by_vulnerability=0.6)
            df = template_rows(ctx, np.repeat(payer, n))
            df = reset_labels(df)
            set_rail(ctx, df, ["UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST"])

            burst_s = rng.uniform(45, 900)
            offsets = np.sort(rng.uniform(0, burst_s, n))
            start_s = ctx.episode_start(burst_s, night_bias=0.28)
            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

            # Fan out across distinct mules rather than repeating one payee.
            to_mule(ctx, df, reuse=0.25)

            balance = ctx.pop.customers["balance"].to_numpy()[int(payer[0])]
            slice_share = rng.dirichlet(np.full(n, 2.5)) * rng.uniform(0.55, 0.98)
            df["amount"] = np.round(np.maximum(balance * slice_share, 100.0), 2)
            df["account_balance_ratio"] = np.round(
                df["amount"].to_numpy() / max(balance, 100.0), 5
            )
            df["is_irrevocable_rail"] = np.isin(df["rail"].to_numpy(), list(IRREVOCABLE_RAILS)).astype(int)
            df["session_duration_s"] = np.round(rng.uniform(60, 1200, n), 1)
            df["initiated_by_agent"] = 0
            frames.append(finalise(df, vector, campaign, "infrastructure_abuse"))
        return pd.concat(frames, ignore_index=True)

    # -- many payers, one payee, one physical location ------------------------------------
    def _qr_tampering(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        """A tampered static merchant QR silently redirects in-person payments.

        From each payer's point of view this is an ordinary local merchant payment: right
        shop, right amount, right neighbourhood. The anomaly is entirely on the receiving
        side - a young account absorbing many one-off payers in a tight geographic cluster.
        """
        if n_events <= 0:
            return pd.DataFrame()
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = ctx.cap(int(rng.integers(12, 60)), minimum=6)
            payers = select_victims(ctx, n, by_vulnerability=0.0)
            df = template_rows(ctx, payers)
            df = reset_labels(df)
            set_rail(ctx, df, ["UPI_P2M"], keep_channel=True)
            df["channel"] = rng.choice(["mobile_app", "pos"], size=n, p=[0.75, 0.25])

            offsets = np.sort(rng.uniform(0, rng.uniform(1, 4) * 86400, n))
            # A tampered QR is scanned by ordinary customers going about their day, so this
            # one carries no night bias at all.
            start_s = ctx.episode_start(float(offsets[-1]))
            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

            # A single tampered QR means a single receiving account for the whole cluster.
            to_mule(ctx, df, reuse=0.0)
            single = df["payee_account_id"].iat[0]
            df["payee_account_id"] = single
            df["payee_psp"] = ctx.mules.psp_of[single]
            df["mule_ring_id"] = ctx.mules.ring_of[single]
            df["payee_account_age_days"] = round(ctx.mules.opened_days_ago[single], 1)

            # Ordinary in-person ticket sizes, paid close to home.
            df["amount"] = np.round(np.exp(rng.normal(np.log(420), 0.85, n)), 2)
            df["distance_from_home_km"] = np.round(np.abs(rng.normal(3.5, 3.0, n)), 3)
            df["geo_velocity_kmh"] = np.round(np.abs(rng.normal(8.0, 6.0, n)), 2)
            df["mcc"] = int(rng.choice([5411, 5814, 5499, 5812]))
            df["initiated_by_agent"] = 0
            frames.append(finalise(df, vector, campaign, "infrastructure_abuse"))
        return pd.concat(frames, ignore_index=True)
