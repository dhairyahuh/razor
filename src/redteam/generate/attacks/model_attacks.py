"""Attacks against the detection model itself.

Two things live here. The first is a pair of attack generators for the vectors that target
the model directly. The second, and more important one, is :func:`apply_evasion` - a
reusable transform applied to a configurable share of *all* generated fraud.

That distinction matters. Treating evasion as a separate attack type quietly assumes the
attacker either evades or does not. In reality evasion is a modifier on everything: the
same voice-clone scam, run by an operator who has learned which amounts and hours draw a
step-up, becomes a materially harder detection problem without changing category. Reported
metrics that exclude this are optimistic by construction.

Only attacker-controllable features are perturbed. An attacker can choose the amount, the
hour, the merchant category and how long the session appears to last; they cannot choose
the victim's account tenure or how long their handset has been enrolled. Respecting that
constraint is what separates a plausible evasion model from an adversarial-example demo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    to_mule,
    to_real_merchant,
)
from .profiles import clamp_timestamps

# Hours with the densest legitimate volume: the cheapest place to hide.
BUSY_HOURS = np.array([10, 11, 12, 13, 18, 19, 20, 21])


def _move_amount(rng, df, idx, strength):
    """Pull the amount toward the dense middle of the legitimate distribution."""
    amt = df["amount"].to_numpy()[idx].astype(float)
    target = np.exp(rng.normal(np.log(2_600.0), 0.75, len(idx)))
    out = np.exp((1 - strength) * np.log(np.maximum(amt, 1)) + strength * np.log(target))
    # Break the round-number habit: round amounts are themselves a scam tell.
    return np.round(out * rng.uniform(0.985, 1.015, len(idx)), 2)


def _move_hour(rng, df, idx, strength):
    move = rng.random(len(idx)) < strength
    hour = df["hour"].to_numpy()[idx].copy()
    hour[move] = rng.choice(BUSY_HOURS, size=int(move.sum()))
    return hour


def _relax_session(rng, df, idx, strength):
    """Pace the session like a human instead of a script."""
    cur = df["form_fill_duration_s"].to_numpy()[idx].astype(float)
    target = rng.gamma(2.6, 9.0, len(idx))
    return np.round((1 - strength) * cur + strength * target, 2)


def _improve_name_match(rng, df, idx, strength):
    """Rent a better-named mule account so Confirmation of Payee stops helping."""
    cur = df["payee_name_match_score"].to_numpy()[idx].astype(float)
    target = rng.uniform(0.82, 0.98, len(idx))
    return np.round(np.maximum(cur, (1 - strength) * cur + strength * target), 4)


def _age_the_payee(rng, df, idx, strength):
    """Register the beneficiary days in advance instead of minutes."""
    cur = df["_payee_reg_lag_s"].to_numpy()[idx].astype(float)
    target = rng.uniform(2 * 86400, 20 * 86400, len(idx))
    return np.round((1 - strength) * cur + strength * target, 1)


def _suppress_flag(column: str, keep_p: float = 0.25):
    """Drop a coercion tell the attacker can simply choose not to leave behind."""
    def _fn(rng, df, idx, strength):
        cur = df[column].to_numpy()[idx].copy()
        drop = (rng.random(len(idx)) < strength) & (rng.random(len(idx)) > keep_p)
        cur[drop] = 0
        return cur
    return _fn


def _normalise_iso(rng, df, idx, strength):
    cur = df["iso20022_field_entropy"].to_numpy()[idx].astype(float)
    target = rng.normal(3.9, 0.2, len(idx))
    return np.round(np.where(cur > 0, (1 - strength) * cur + strength * target, cur), 4)


#: Ordered list of (column, transform). Each is a lever a real attacker controls.
EVASION_MOVES: List[Tuple[str, Callable]] = [
    ("amount", _move_amount),
    ("hour", _move_hour),
    ("form_fill_duration_s", _relax_session),
    ("payee_name_match_score", _improve_name_match),
    ("_payee_reg_lag_s", _age_the_payee),
    ("screen_share_active", _suppress_flag("screen_share_active")),
    ("remote_access_app_detected", _suppress_flag("remote_access_app_detected")),
    ("call_in_progress", _suppress_flag("call_in_progress", keep_p=0.45)),
    ("ip_asn_is_hosting", _suppress_flag("ip_asn_is_hosting", keep_p=0.3)),
    ("vpn_or_proxy", _suppress_flag("vpn_or_proxy", keep_p=0.4)),
    ("is_night", _suppress_flag("is_night", keep_p=0.2)),
    ("iso20022_field_entropy", _normalise_iso),
]


def apply_evasion(rng: np.random.Generator, df: pd.DataFrame, *, share: float,
                  strength: float = 0.65,
                  moves: Optional[Sequence[str]] = None,
                  restrict_to: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Apply evasion tuning to a share of the fraudulent rows, in place.

    ``strength`` in [0, 1] interpolates between the raw attack and a fully
    benign-mimicking version. ``moves`` optionally restricts which levers are used, which
    is what the closed loop searches over.
    """
    if len(df) == 0 or share <= 0:
        return df
    pool = np.where(df["is_fraud"].to_numpy() == 1)[0] if restrict_to is None else restrict_to
    if pool.size == 0:
        return df
    k = int(round(share * pool.size))
    if k <= 0:
        return df
    idx = rng.choice(pool, size=k, replace=False)

    active = EVASION_MOVES if moves is None else [m for m in EVASION_MOVES if m[0] in set(moves)]
    for column, fn in active:
        if column not in df.columns:
            continue
        values = fn(rng, df, idx, strength)
        col = df[column].to_numpy().copy()
        if col.dtype.kind in "iu" and np.asarray(values).dtype.kind == "f":
            col = col.astype(float)
        col[idx] = values
        df[column] = col

    # Keep the derived night flag consistent with any hour changes.
    if "hour" in df.columns:
        hours = df["hour"].to_numpy()
        df["is_night"] = np.where(
            np.isin(np.arange(len(df)), idx), ((hours < 6) | (hours >= 23)).astype(int),
            df["is_night"].to_numpy()
        )
    flags = df["evasion_applied"].to_numpy().copy()
    flags[idx] = 1
    df["evasion_applied"] = flags
    return df


# --------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------

class ModelAttackGenerator(AttackGenerator):
    name = "model_attacks"
    vector_ids = ("MODEL-ADVERSARIAL-TABULAR-EVASION", "MODEL-ORACLE-PROBING")

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        if n_events <= 0:
            return pd.DataFrame()
        if vector.id == "MODEL-ADVERSARIAL-TABULAR-EVASION":
            return self._adversarial(ctx, vector, n_events)
        return self._oracle_probing(ctx, vector, n_events)

    def _adversarial(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        """Fraudulent transfers designed from the outset to sit inside the benign envelope."""
        rng = ctx.rng
        counts = rng.integers(1, 4, n_events)
        total = int(counts.sum())
        victims = select_victims(ctx, n_events, by_vulnerability=0.4)
        episode_of = np.repeat(np.arange(n_events), counts)

        df = template_rows(ctx, victims[episode_of])
        df = reset_labels(df)
        set_rail(ctx, df, ["CARD_CNP", "UPI_P2P", "IMPS", "RTP_FEDNOW"])

        days = rng.uniform(0, max(ctx.cfg.benign.n_days - 0.5, 1), total)
        hours = rng.choice(BUSY_HOURS, size=total)
        ts = (pd.Timestamp(ctx.cfg.benign.start_date)
              + pd.to_timedelta((days.astype(int) * 86400 + hours * 3600
                                 + rng.integers(0, 3600, total)).astype("int64"), unit="s"))
        clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))
        to_mule(ctx, df, reuse=0.7, name_match_beta=(9.0, 1.4))

        # Amounts drawn directly from the dense core of legitimate spending.
        df["amount"] = np.round(np.exp(rng.normal(np.log(3_100), 0.7, total)), 2)
        balance = ctx.pop.customers["balance"].to_numpy()[df["_customer_idx"].to_numpy()]
        df["account_balance_ratio"] = np.round(
            np.clip(df["amount"].to_numpy() / np.maximum(balance, 100.0), 0, 50), 5)
        df["initiated_by_agent"] = 0
        df["mfa_passed"] = 1
        df["_payee_reg_lag_s"] = np.round(rng.uniform(3 * 86400, 21 * 86400, total), 1)

        campaign = np.array([ctx.campaign_id(vector.id) for _ in range(n_events)], dtype=object)
        df = finalise(df, vector, "", "card_fraud")
        df["campaign_id"] = campaign[episode_of]
        # Full-strength evasion on every row: this vector *is* the evasion.
        apply_evasion(rng, df, share=1.0, strength=0.85)
        return df

    def _oracle_probing(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        """Systematic amount ladders that map the issuer's decision boundary.

        The tell is not any single probe but the arithmetic regularity of the sequence:
        near-identical transactions walking an amount grid against one merchant.
        """
        rng = ctx.rng
        frames: List[pd.DataFrame] = []
        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            n = ctx.cap(int(rng.integers(8, 26)), minimum=5)
            payers = select_victims(ctx, n, by_vulnerability=0.0)
            df = template_rows(ctx, payers)
            df = reset_labels(df)
            set_rail(ctx, df, ["CARD_CNP"], keep_channel=True)
            df["channel"] = "agent_api"
            df["auth_method"] = "cvv_only"

            offsets = np.arange(n) * rng.uniform(20, 240) + rng.normal(0, 5, n)
            start_s = ctx.episode_start(float(offsets[-1]), night_bias=0.25)
            ts = (pd.Timestamp(ctx.cfg.benign.start_date)
                  + pd.to_timedelta((start_s + offsets).astype("int64"), unit="s"))
            clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))
            to_real_merchant(ctx, df, mccs=(5815, 5399, 4814), n_distinct=1)

            anchor = float(rng.choice([5_000, 10_000, 25_000, 50_000]))
            step = anchor * rng.uniform(0.01, 0.05)
            df["amount"] = np.round(anchor + (np.arange(n) - n / 2) * step, 2)
            df["auth_attempts"] = rng.integers(1, 4, n)
            df["mfa_passed"] = (rng.random(n) < 0.5).astype(int)
            df["step_up_triggered"] = (rng.random(n) < 0.4).astype(int)
            df["ip_asn_is_hosting"] = 1
            df["device_is_emulator"] = (rng.random(n) < 0.5).astype(int)
            df["keystroke_flight_cv"] = 0.0
            df["hesitation_events"] = 0
            df["form_fill_duration_s"] = np.round(rng.uniform(0.3, 2.0, n), 3)
            df["initiated_by_agent"] = 1
            df["agent_is_registered"] = 0
            df["agent_reputation"] = np.round(rng.uniform(0.0, 0.2, n), 4)
            df["_agent_payload"] = "oracle_probe"
            frames.append(finalise(df, vector, campaign, "card_fraud"))
        return pd.concat(frames, ignore_index=True)
