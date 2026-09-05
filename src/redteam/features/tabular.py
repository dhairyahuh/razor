"""Behavioural feature engineering for the defence.

The raw schema describes a payment. These features describe a payment *relative to the
history of the entities involved*, which is where almost all of the detectable signal in
payment fraud lives. A INR 40,000 transfer is meaningless on its own; a INR 40,000 transfer
from an account whose largest previous payment was INR 3,000, to a payee added eleven
minutes ago, as the third payment in a four-minute session, is not.

Four entity views are computed, because different attack families burst along different
axes and a defence that only profiles the payer is structurally blind to half the library:

* **payer** - takeovers, drains, instant-rail fan-out;
* **payee** - mule collection points, tampered QR codes, scam beneficiaries;
* **merchant** - card testing, oracle probing, transaction laundering;
* **device** - device farms and mass onboarding.

Every window is causal and excludes the row being scored.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..windows import (
    codes_of,
    epoch_seconds,
    expanding_mean_std,
    first_seen_and_rank,
    previous_value,
    shared_codes,
    window_sum_count,
    window_unique,
)

HOUR = 3_600
DAY = 86_400
WEEK = 7 * 86_400

#: Column prefix for everything this module produces, so features can be told apart from
#: raw schema fields when auditing what the model is allowed to see.
PREFIX = "f_"


def _base_amount(df: pd.DataFrame) -> np.ndarray:
    """Amount on one common scale, whatever currency the payment settled in."""
    col = "amount_inr" if "amount_inr" in df.columns else "amount"
    return df[col].to_numpy().astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with derived behavioural features appended.

    The builders write into a plain dict and the columns are attached in one concat.
    Appending sixty columns to a DataFrame one at a time copies the block manager on every
    assignment, and this function runs once per candidate inside the co-evolution search.
    """
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    out: dict = {}
    ts = epoch_seconds(df["timestamp"])
    # Base currency throughout. Summing a EUR-settled payment into the same velocity window as
    # an INR one understates it by roughly a hundredfold, which quietly hides exactly the
    # cross-border structuring these windows exist to catch.
    amount = _base_amount(df)
    log_amount = np.log10(np.maximum(amount, 1.0))

    (payer, payee), _ = shared_codes(
        df["payer_account_id"].to_numpy(), df["payee_account_id"].to_numpy()
    )
    merchant = codes_of(df["merchant_id"].to_numpy())
    device = codes_of(df["device_id"].to_numpy())

    _payer_velocity(out, payer, ts, amount, log_amount)
    _payer_context(out, df, payer, ts)
    _payee_velocity(out, payee, payer, ts, amount)
    _merchant_velocity(out, df, merchant, payer, ts, amount)
    _device_features(out, df, device, payer, ts)
    _auth_features(out, df, payer, ts)
    _instruction_features(out, df, amount)
    _agentic_features(out, df)

    features = pd.DataFrame(
        {k: np.where(np.isfinite(v), v, np.nan) if np.asarray(v).dtype.kind == "f" else v
         for k, v in out.items()},
        index=df.index,
    )
    return pd.concat([df, features], axis=1)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(PREFIX)]


# --------------------------------------------------------------------------------------
# Payer
# --------------------------------------------------------------------------------------

def _payer_velocity(out: dict, payer, ts, amount, log_amount) -> None:
    for label, window in (("1h", HOUR), ("24h", DAY), ("7d", WEEK)):
        s, c = window_sum_count(payer, ts, amount, window)
        out[f"{PREFIX}payer_txn_count_{label}"] = c
        out[f"{PREFIX}payer_amount_sum_{label}"] = np.round(s, 2)

    # Deviation from the payer's own spending baseline. Expressed in log space so it is
    # scale free: a 10x jump means the same thing for a student and for a corporate.
    mean, std, count = expanding_mean_std(payer, ts, log_amount)
    out[f"{PREFIX}payer_prior_txn_count"] = count
    out[f"{PREFIX}payer_log_amount_mean"] = np.round(mean, 4)
    z = (log_amount - mean) / np.maximum(std, 0.12)
    out[f"{PREFIX}payer_amount_zscore"] = np.round(np.where(count >= 3, z, np.nan), 4)
    out[f"{PREFIX}payer_amount_ratio_to_mean"] = np.round(
        np.where(count >= 3, log_amount - mean, np.nan), 4
    )

    prev_ts = previous_value(payer, ts, ts.astype(float))
    gap = ts - prev_ts
    out[f"{PREFIX}payer_seconds_since_prev"] = np.round(gap, 1)
    out[f"{PREFIX}payer_log_seconds_since_prev"] = np.round(np.log10(np.maximum(gap, 1.0)), 4)

    # Intra-session cumulative exposure: the only thing that can stop an instant-rail
    # fan-out, because each individual payment in the burst looks ordinary.
    burst_sum, burst_cnt = window_sum_count(payer, ts, amount, 900)
    out[f"{PREFIX}payer_amount_sum_15m"] = np.round(burst_sum, 2)
    out[f"{PREFIX}payer_txn_count_15m"] = burst_cnt


def _payer_context(out: dict, df: pd.DataFrame, payer, ts) -> None:
    payee_codes = codes_of(df["payee_account_id"].to_numpy())
    out[f"{PREFIX}payer_unique_payees_24h"] = window_unique(payer, ts, payee_codes, DAY)
    out[f"{PREFIX}payer_unique_payees_7d"] = window_unique(payer, ts, payee_codes, WEEK)

    first_payee = df["payee_is_first_time"].to_numpy().astype(float)
    novel_sum, _ = window_sum_count(payer, ts, first_payee, WEEK)
    out[f"{PREFIX}payer_new_payees_7d"] = novel_sum

    # Share of the payer's balance moving in this one instruction.
    out[f"{PREFIX}balance_drain_ratio"] = np.round(
        df["account_balance_ratio"].to_numpy().astype(float), 5
    )

    # Implied travel speed since the payer's previous payment. The raw geo_velocity field
    # is a vendor estimate; this is the arithmetic the bank can do itself.
    lat = df["_lat"].to_numpy().astype(float) if "_lat" in df.columns else np.zeros(len(df))
    lon = df["_lon"].to_numpy().astype(float) if "_lon" in df.columns else np.zeros(len(df))
    prev_lat = previous_value(payer, ts, lat)
    prev_lon = previous_value(payer, ts, lon)
    prev_ts = previous_value(payer, ts, ts.astype(float))
    from ..generate.entities import haversine_km

    with np.errstate(invalid="ignore"):
        dist = haversine_km(prev_lat, prev_lon, lat, lon)
        hours = np.maximum(ts - prev_ts, 60) / 3600.0
        implied = dist / hours
    out[f"{PREFIX}implied_speed_kmh"] = np.round(np.where(np.isnan(dist), np.nan, implied), 2)


# --------------------------------------------------------------------------------------
# Payee, merchant, device
# --------------------------------------------------------------------------------------

def _payee_velocity(out: dict, payee, payer, ts, amount) -> None:
    for label, window in (("1h", HOUR), ("24h", DAY), ("7d", WEEK)):
        s, c = window_sum_count(payee, ts, amount, window)
        out[f"{PREFIX}payee_txn_count_{label}"] = c
        out[f"{PREFIX}payee_amount_sum_{label}"] = np.round(s, 2)

    out[f"{PREFIX}payee_unique_payers_1h"] = window_unique(payee, ts, payer, HOUR)
    out[f"{PREFIX}payee_unique_payers_7d"] = window_unique(payee, ts, payer, WEEK)

    # Fan-in acceleration: a collection account's inbound rate spikes when a campaign runs.
    _, cnt_1h = window_sum_count(payee, ts, amount, HOUR)
    _, cnt_7d = window_sum_count(payee, ts, amount, WEEK)
    out[f"{PREFIX}payee_fanin_acceleration"] = np.round(
        cnt_1h / np.maximum(cnt_7d / (7 * 24), 0.02), 3
    )


def _merchant_velocity(out: dict, df: pd.DataFrame, merchant, payer, ts, amount) -> None:
    has_merchant = (df["merchant_id"].to_numpy() != "")
    for label, window in (("10m", 600), ("1h", HOUR), ("24h", DAY)):
        s, c = window_sum_count(merchant, ts, amount, window)
        out[f"{PREFIX}merchant_txn_count_{label}"] = np.where(has_merchant, c, np.nan)
        out[f"{PREFIX}merchant_amount_sum_{label}"] = np.where(has_merchant, np.round(s, 2), np.nan)

    uniq = window_unique(merchant, ts, payer, HOUR)
    out[f"{PREFIX}merchant_unique_payers_1h"] = np.where(has_merchant, uniq, np.nan)

    # Card testing shows up as a merchant taking an unusual number of tiny tickets.
    micro = (amount < 100).astype(float)
    micro_sum, micro_cnt = window_sum_count(merchant, ts, micro, HOUR)
    out[f"{PREFIX}merchant_micro_txn_share_1h"] = np.where(
        has_merchant & (micro_cnt > 0), np.round(micro_sum / np.maximum(micro_cnt, 1), 4), np.nan
    )

    # Amount relative to what this merchant normally charges.
    log_amount = np.log10(np.maximum(amount, 1.0))
    m_mean, m_std, m_cnt = expanding_mean_std(merchant, ts, log_amount)
    z = (log_amount - m_mean) / np.maximum(m_std, 0.15)
    out[f"{PREFIX}merchant_amount_zscore"] = np.round(
        np.where(has_merchant & (m_cnt >= 5), z, np.nan), 4
    )


def _device_features(out: dict, df: pd.DataFrame, device, payer, ts) -> None:
    has_device = (df["device_id"].to_numpy() != "")
    out[f"{PREFIX}device_unique_payers_7d"] = np.where(
        has_device, window_unique(device, ts, payer, WEEK), np.nan
    )
    _, cnt = window_sum_count(device, ts, np.ones(len(df)), DAY)
    out[f"{PREFIX}device_txn_count_24h"] = np.where(has_device, cnt, np.nan)

    # Did the payer switch device since their last payment?
    prev_device = previous_value(payer, ts, device.astype(float))
    out[f"{PREFIX}payer_device_changed"] = np.where(
        np.isnan(prev_device), 0, (prev_device != device).astype(int)
    )

    first_ts, _ = first_seen_and_rank(device, ts)
    out[f"{PREFIX}device_first_seen_days"] = np.round(np.maximum(ts - first_ts, 0) / DAY, 3)


# Authentication methods ordered by resistance to remote compromise. Phishing-resistant
# methods bound to the device sit at the top; a shared secret in transit sits near the
# bottom. The ordering is what makes a *downgrade* expressible: the raw method is a
# category, and a category cannot tell you that something got weaker.
AUTH_STRENGTH = {
    "passkey": 5, "biometric": 4, "3ds2": 4, "upi_pin": 3,
    "voice_print": 2, "otp_sms": 2, "cvv_only": 1, "none": 0,
}


def _auth_features(out: dict, df: pd.DataFrame, payer, ts) -> None:
    """Did this payment authenticate more weakly than the payer's own precedent?

    A downgrade attack does not present anomalous credentials - it presents perfectly valid
    ones from a weaker method the bank kept for recovery. The method alone therefore looks
    ordinary, because plenty of legitimate customers use SMS codes. What is not ordinary is
    an account that has been using a passkey for months suddenly falling back, and that is
    only visible by comparing against the payer's history rather than against the population.
    """
    strength = df["auth_method"].map(AUTH_STRENGTH).fillna(2).to_numpy().astype(float)
    out[f"{PREFIX}auth_strength"] = strength

    prev = previous_value(payer, ts, strength)
    mean, _, count = expanding_mean_std(payer, ts, strength)
    # Signed, so a step *up* to a stronger method is distinguishable from a step down. An
    # absolute difference would score a security improvement as suspicious.
    out[f"{PREFIX}auth_strength_delta"] = np.where(np.isnan(prev), 0.0, strength - prev)
    out[f"{PREFIX}auth_method_downgraded"] = np.where(
        np.isnan(prev), 0, (strength < prev).astype(int)
    )
    # Against the payer's whole history rather than only the previous payment, which catches
    # a downgrade that follows a single unrelated weak payment used to reset the comparison.
    out[f"{PREFIX}auth_strength_vs_baseline"] = np.round(
        np.where(count >= 3, strength - mean, 0.0), 4
    )


# --------------------------------------------------------------------------------------
# Instruction-level and agentic
# --------------------------------------------------------------------------------------

def _instruction_features(out: dict, df: pd.DataFrame, amount: np.ndarray) -> None:
    out[f"{PREFIX}log_amount"] = np.round(np.log10(np.maximum(amount, 1.0)), 4)
    # Round-number amounts are chosen by a human being told a number out loud.
    out[f"{PREFIX}amount_is_round_100"] = (np.isclose(amount % 100, 0)).astype(int)
    out[f"{PREFIX}amount_is_round_1000"] = (np.isclose(amount % 1000, 0)).astype(int)
    # Structuring: sitting just under a reporting threshold.
    near = np.zeros(len(df), dtype=int)
    for threshold in (50_000.0, 100_000.0, 200_000.0):
        near |= ((amount > 0.85 * threshold) & (amount < threshold)).astype(int)
    out[f"{PREFIX}amount_near_threshold"] = near

    out[f"{PREFIX}payee_added_recently"] = (
        df["payee_added_minutes_ago"].to_numpy() < 60
    ).astype(int)
    out[f"{PREFIX}log_payee_added_minutes"] = np.round(
        np.log10(np.maximum(df["payee_added_minutes_ago"].to_numpy(), 0.1)), 4
    )
    out[f"{PREFIX}cop_shortfall"] = np.round(1.0 - df["payee_name_match_score"].to_numpy(), 4)

    # Coercion telemetry, collapsed into one count. Individually each is rare and noisy;
    # together they are the strongest available proxy for "someone is on the phone".
    coercion = (
        df["screen_share_active"].to_numpy()
        + df["remote_access_app_detected"].to_numpy()
        + df["accessibility_service_active"].to_numpy()
        + df["call_in_progress"].to_numpy()
    )
    out[f"{PREFIX}coercion_signal_count"] = coercion

    # Behavioural regularity. Human input has a coefficient of variation around 0.3-0.55;
    # both scripted input (too low) and a different human (too high) fall outside it.
    cv = df["keystroke_flight_cv"].to_numpy().astype(float)
    typed = cv > 0
    out[f"{PREFIX}keystroke_cv_deviation"] = np.where(typed, np.abs(cv - 0.42), np.nan)
    out[f"{PREFIX}no_human_telemetry"] = ((~typed) & (df["channel"].to_numpy() != "pos")).astype(int)

    out[f"{PREFIX}auth_weak_for_amount"] = (
        df["auth_method"].isin(["cvv_only", "none", "otp_sms"]).to_numpy()
        & (amount > 25_000)
    ).astype(int)
    out[f"{PREFIX}irrevocable_and_new_payee"] = (
        (df["is_irrevocable_rail"].to_numpy() == 1) & (df["payee_is_first_time"].to_numpy() == 1)
    ).astype(int)


def _agentic_features(out: dict, df: pd.DataFrame) -> None:
    agent = df["initiated_by_agent"].to_numpy() == 1
    # Delegation integrity: an agent payment with no valid, matching intent artefact has
    # no verifiable mandate behind it regardless of how ordinary it looks.
    out[f"{PREFIX}intent_unverified"] = np.where(
        agent,
        ((df["intent_token_present"].to_numpy() == 0)
         | (df["intent_signature_valid"].to_numpy() == 0)
         | (df["intent_payee_match"].to_numpy() == 0)).astype(int),
        0,
    )
    out[f"{PREFIX}intent_amount_drift"] = np.where(
        agent, df["intent_amount_delta_ratio"].to_numpy(), 0.0
    )
    out[f"{PREFIX}agent_untrusted_per_tool_call"] = np.where(
        agent,
        df["agent_untrusted_content_tokens"].to_numpy()
        / np.maximum(df["agent_tool_calls"].to_numpy(), 1),
        0.0,
    )
    out[f"{PREFIX}agent_unregistered"] = np.where(
        agent, (df["agent_is_registered"].to_numpy() == 0).astype(int), 0
    )
    out[f"{PREFIX}agent_stale_intent"] = np.where(
        agent, (df["intent_age_s"].to_numpy() > 1800).astype(int), 0
    )
    out[f"{PREFIX}agent_token_reused"] = np.where(
        agent, df["spt_reuse_count"].to_numpy(), 0
    )
    out[f"{PREFIX}agent_new_merchant"] = np.where(
        agent, (df["merchant_domain_age_days"].to_numpy() < 90).astype(int), 0
    )
