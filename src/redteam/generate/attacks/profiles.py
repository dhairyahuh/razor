"""Profile-driven episode generator shared by most attack families.

Almost every payment attack decomposes into the same six decisions: who is targeted, on
which rail, over how many payments, into what counterparty, at what amount envelope, and
which telemetry leaks. Encoding that as a declarative :class:`EpisodeProfile` means a new
vector usually costs a dozen lines of data rather than a new generator, which is what
allows the library to carry 39 simulated vectors instead of five.

Anything genuinely idiosyncratic — a mule fan-out graph, an adversarial evasion search —
either uses the ``post`` hook or gets its own generator module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...identify.library import AttackVector
from .. import timing
from .base import (
    AttackContext,
    AttackGenerator,
    SignalSpec,
    apply_signals,
    finalise,
    reset_labels,
    retime,
    scale_amount,
    select_victims,
    set_rail,
    template_rows,
    to_counterfeit_merchant,
    to_mule,
    to_real_merchant,
)


@dataclass
class EpisodeProfile:
    """Declarative description of one attack vector's observable footprint."""

    fraud_type: str
    rails: Sequence[str]
    amount_multiplier: Tuple[float, float]
    payments: Tuple[int, int] = (1, 1)
    """Inclusive range of payments in a single episode."""

    escalating: bool = False
    """If true, later payments in an episode grow - the grooming / drain signature."""

    escalation_factor: float = 1.85
    episode_span_hours: Tuple[float, float] = (0.02, 0.5)
    victim: Dict[str, object] = field(default_factory=dict)
    signals: Dict[str, SignalSpec] = field(default_factory=dict)
    night_bias: float = 0.0
    salary_bias: float = 0.0
    """Share of payments pulled onto the days around payday. See ``base.retime``."""

    channel: Optional[str] = None
    auth_method: Optional[str] = None
    counterparty: str = "mule"
    """mule | counterfeit_merchant | real_merchant | keep.

    ``keep`` leaves the donor transaction's own counterparty untouched.
    """

    real_merchant_mccs: Sequence[int] = ()
    counterfeit_mccs: Sequence[int] = (5399,)
    counterfeit_domain_age: Tuple[float, float] = (1.0, 120.0)
    counterfeit_agent_optimised_p: float = 0.2
    cross_border_p: Optional[float] = None
    round_bias: float = 0.45
    amount_cap: Optional[float] = None
    amount_floor: float = 50.0
    mule_reuse: float = 0.62
    post: Optional[Callable[[AttackContext, pd.DataFrame, np.ndarray], None]] = None
    """Hook receiving (ctx, dataframe, sequence-within-episode) for family-specific work."""


class ProfileAttackGenerator(AttackGenerator):
    """Generic generator driven by a ``PROFILES`` mapping on the subclass."""

    PROFILES: Dict[str, EpisodeProfile] = {}

    @property
    def vector_ids(self) -> Sequence[str]:  # type: ignore[override]
        return tuple(self.PROFILES)

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        profile = self.PROFILES[vector.id]
        if n_events <= 0:
            return pd.DataFrame()

        victims = select_victims(ctx, n_events, **profile.victim)  # type: ignore[arg-type]

        lo, hi = profile.payments
        # Do not let a single long episode blow through the remaining fraud budget.
        hi = max(lo, min(hi, ctx.budget_hint))
        counts = ctx.rng.integers(lo, hi + 1, n_events)
        episode_of = np.repeat(np.arange(n_events), counts)
        seq = np.concatenate([np.arange(c) for c in counts])
        total = int(counts.sum())
        if total == 0:
            return pd.DataFrame()

        df = template_rows(ctx, victims[episode_of])
        df = reset_labels(df)

        set_rail(ctx, df, profile.rails, keep_channel=bool(profile.channel))
        if profile.channel:
            df["channel"] = profile.channel
        if profile.auth_method:
            df["auth_method"] = profile.auth_method

        retime(ctx, df, night_bias=profile.night_bias, jitter_days=2.0,
               salary_bias=profile.salary_bias)
        spread_episode(ctx, df, episode_of, seq, profile.episode_span_hours)

        if profile.counterparty == "mule":
            to_mule(ctx, df, reuse=profile.mule_reuse)
        elif profile.counterparty == "real_merchant":
            to_real_merchant(ctx, df, mccs=profile.real_merchant_mccs)
        elif profile.counterparty == "counterfeit_merchant":
            to_counterfeit_merchant(
                ctx, df,
                mccs=profile.counterfeit_mccs,
                domain_age=profile.counterfeit_domain_age,
                agent_optimised_p=profile.counterfeit_agent_optimised_p,
            )

        scale_amount(ctx, df, *profile.amount_multiplier, round_bias=profile.round_bias,
                     cap=profile.amount_cap, floor=profile.amount_floor)
        if profile.escalating:
            df["amount"] = np.round(
                df["amount"].to_numpy() * np.power(profile.escalation_factor, seq), 2
            )

        if profile.cross_border_p is not None:
            df["is_cross_border"] = (ctx.rng.random(total) < profile.cross_border_p).astype(int)

        apply_signals(ctx, df, profile.signals)
        if profile.post is not None:
            profile.post(ctx, df, seq)

        campaign = np.array([ctx.campaign_id(vector.id) for _ in range(n_events)], dtype=object)
        df = finalise(df, vector, "", profile.fraud_type)
        df["campaign_id"] = campaign[episode_of]
        return df


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def spread_episode(ctx: AttackContext, df: pd.DataFrame, episode_of: np.ndarray,
                   seq: np.ndarray, span_hours: Tuple[float, float]) -> None:
    """Lay the payments of one episode out over a realistic window."""
    if len(df) == 0:
        return
    n_episodes = int(episode_of.max()) + 1
    span = ctx.rng.uniform(*span_hours, size=n_episodes)[episode_of]
    sizes = np.bincount(episode_of, minlength=n_episodes)[episode_of]
    frac = np.where(sizes > 1, seq / np.maximum(sizes - 1, 1), 0.0)
    offset_s = frac * span * 3600.0 + ctx.rng.normal(0, 90, len(df))
    clamp_timestamps(ctx, df, pd.DatetimeIndex(df["timestamp"]) +
                     pd.to_timedelta(offset_s.astype("int64"), unit="s"))


def clamp_timestamps(ctx: AttackContext, df: pd.DataFrame, ts: pd.DatetimeIndex) -> None:
    """Clamp into the simulation window and refresh all derived time fields."""
    start = pd.Timestamp(ctx.cfg.benign.start_date)
    end = start + pd.Timedelta(days=ctx.cfg.benign.n_days) - pd.Timedelta(seconds=1)
    ts = pd.DatetimeIndex(np.clip(ts.values, start.to_datetime64(), end.to_datetime64()))
    df["timestamp"] = ts
    df["hour"] = ts.hour
    df["day_of_week"] = ts.dayofweek
    df["is_night"] = ((ts.hour < 6) | (ts.hour >= 23)).astype(int)
    df["is_salary_window"] = timing.is_salary_window(ts.day.to_numpy()).astype(int)


def u(low: float, high: float):
    """Uniform draw helper for ``dist(...)`` signal specs."""
    return lambda rng, k: rng.uniform(low, high, k)


def pois(lam: float):
    return lambda rng, k: rng.poisson(lam, k)


def lognorm(mu: float, sigma: float, lo: float, hi: float):
    return lambda rng, k: np.clip(rng.lognormal(mu, sigma, k), lo, hi)
