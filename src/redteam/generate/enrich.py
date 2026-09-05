"""Counterparty enrichment: what the PSP knows about the payee at authorisation time.

These fields are part of the observable schema but are deliberately *not* written by the
attack generators. If a generator could set ``payee_inbound_unique_payers_24h`` directly it
would be asserting the answer rather than producing the behaviour, and the defence would be
learning the red team's opinion instead of the data's structure. They are derived here from
the merged event stream, exactly as a real counterparty-intelligence service would.

Everything is computed strictly causally - see :mod:`redteam.windows`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..windows import (
    cross_window_sum,
    epoch_seconds,
    first_seen_and_rank,
    shared_codes,
    window_sum_count,
    window_unique,
)
from .benign import convert_currency

DAY_S = 86_400.0
WINDOW_24H = 86_400


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add dynamic counterparty fields to a merged benign + attack event stream."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts = epoch_seconds(df["timestamp"])

    # Every generator draws amounts on the INR scale the population is built on, so a rail
    # that settles in another currency has to be converted here rather than merely relabelled.
    # ``amount`` becomes the settled figure and ``amount_inr`` keeps the common scale that
    # thresholds, velocity windows and cross-rail aggregates all need.
    amount = df["amount"].to_numpy().astype(float)
    df["amount_inr"] = np.round(amount, 2)
    df["amount"] = convert_currency(amount, df["currency"].to_numpy())

    # Payer and payee identifiers share one code space so an account's inbound and
    # outbound activity can be cross-referenced.
    (payer_codes, payee_codes), n_codes = shared_codes(
        df["payer_account_id"].to_numpy(), df["payee_account_id"].to_numpy()
    )

    # -- how well known is this payee, network-wide? --------------------------------------
    payee_first_ts, payee_prior_count = first_seen_and_rank(payee_codes, ts)
    df["payee_prior_txn_count"] = payee_prior_count
    df["payee_age_days"] = np.round(np.maximum(ts - payee_first_ts, 0) / DAY_S, 4)

    # -- has this payer used this payee before? ---------------------------------------------
    pair = pd.factorize(payer_codes * n_codes + payee_codes, sort=False)[0].astype(np.int64)
    pair_first_ts, pair_prior_count = first_seen_and_rank(pair, ts)
    first_time = (pair_prior_count == 0).astype(int)
    df["payee_is_first_time"] = first_time

    # Time since the beneficiary entered this payer's address book: the registration lag
    # for a brand new payee, elapsed relationship time afterwards.
    reg_lag_s = (df["_payee_reg_lag_s"].to_numpy().astype(float)
                 if "_payee_reg_lag_s" in df.columns else np.full(len(df), 3600.0))
    since_first = np.maximum(ts - pair_first_ts, 0).astype(float)
    added_minutes = np.where(first_time == 1, reg_lag_s, since_first + reg_lag_s) / 60.0
    df["payee_added_minutes_ago"] = np.round(np.minimum(added_minutes, 525_600.0), 3)

    # -- inbound pressure on the payee over the last 24h --------------------------------------
    inbound_amt, _ = window_sum_count(payee_codes, ts, amount, WINDOW_24H)
    df["payee_inbound_amount_24h"] = np.round(inbound_amt, 2)
    df["payee_inbound_unique_payers_24h"] = window_unique(payee_codes, ts, payer_codes, WINDOW_24H)

    # -- pass-through behaviour: does money leave as fast as it arrives? -----------------------
    # A ratio near or above 1.0 on a young account is the canonical mule signature.
    outbound = cross_window_sum(payer_codes, ts, amount, payee_codes, ts, WINDOW_24H)
    df["payee_outbound_ratio_24h"] = np.round(outbound / np.maximum(inbound_amt, 1.0), 4)

    return df
