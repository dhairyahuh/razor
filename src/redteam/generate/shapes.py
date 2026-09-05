"""Legitimate traffic that looks exactly like the fraud patterns the defence hunts.

Every strong mule signal is a *shape* rather than a value: money arriving from many unrelated
payers, money leaving as fast as it arrives, a burst of payments to people never paid before.
A generator that produces those shapes only under attack has not built a detection problem -
it has built a lookup table, and the resulting recall measures the generator.

Measured on this repo's own data, eight derived payee features sat between AUC 0.73 and 0.83
and stacked to 92% recall at a 0.5% review budget, while the raw schema alone reached 67%.
The entire gap was the payee-side feature block, because mules were the only accounts in the
dataset with mule-shaped inbound behaviour.

The four shapes here are the ones a real payments team argues about every week:

* **Collectors** - shopkeepers, tiffin services, tutors, landlords, chit-fund collectors, gig
  workers. High inbound fan-in from strangers, on ordinary consumer accounts. Handled at
  source in ``entities._collector_weights``, which biases who gets picked as a new payee.
* **Pass-through** - a small business sweeping its collection account to a current account at
  close of business, producing an outbound/inbound ratio near 1 over 24 hours. That ratio is
  the canonical mule tell, and it is also what every shopkeeper in the country does nightly.
* **Burst payers** - rent day, school-fee deadlines, festival season, a wedding. Several
  payments in minutes from one account.
* **Split bills and disbursement runs** - one payer, many *first-time* payees, one session, on
  an irrevocable rail. A person splitting a restaurant bill eight ways and a corporate paying
  forty new contractors are both indistinguishable from a fan-out drain on the numbers alone.

These are false-positive generators by design. They are what stops the defence from scoring
recall it has not earned, and they are the reason its precision figure means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import IRREVOCABLE_RAILS

#: Share of rows rewritten into each shape. Small on purpose: these are meant to be the hard
#: tail of legitimate behaviour, not a second mode of it. Collectively they touch a few percent
#: of traffic, which is roughly the share of real volume that trips velocity and fan-in rules.
SWEEP_SHARE = 0.010
BURST_SHARE = 0.014
SPLIT_SHARE = 0.010
DISBURSEMENT_SHARE = 0.006

PEER_RAILS_FAST = ("UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST")


def apply_legitimate_shapes(df: pd.DataFrame, rng: np.random.Generator,
                            pop, cfg) -> pd.DataFrame:
    """Rewrite a small share of benign rows into mule-shaped legitimate episodes."""
    df = df.copy()
    n = len(df)
    if n < 200:
        return df

    # Rows are claimed exclusively so the shapes cannot overwrite one another and leave a
    # half-built episode behind.
    order = rng.permutation(n)
    take = 0

    def claim(share: float) -> np.ndarray:
        nonlocal take
        k = int(n * share)
        idx = order[take:take + k]
        take += k
        return idx

    df = _sweeps(df, claim(SWEEP_SHARE), rng, pop)
    df = _bursts(df, claim(BURST_SHARE), rng, pop)
    df = _splits(df, claim(SPLIT_SHARE), rng, pop)
    df = _disbursements(df, claim(DISBURSEMENT_SHARE), rng, pop)
    return df


def _collectors(pop, rng: np.random.Generator, k: int) -> np.ndarray:
    """Indices of the busiest legitimate collection accounts."""
    w = pop.customers["collector_weight"].to_numpy()
    top = np.argsort(-w)[: max(int(len(w) * 0.02), 8)]
    return rng.choice(top, size=k, replace=True)


def _as_account(idx: np.ndarray) -> np.ndarray:
    return np.char.add("A", np.char.zfill(np.asarray(idx).astype(str), 7))


def _set_payer(df: pd.DataFrame, rows: np.ndarray, pop, idx: np.ndarray) -> None:
    df.loc[rows, "_customer_idx"] = idx
    df.loc[rows, "customer_id"] = pop.customers["customer_id"].to_numpy()[idx]
    df.loc[rows, "payer_account_id"] = _as_account(idx)
    df.loc[rows, "payer_psp"] = pop.customers["psp_id"].to_numpy()[idx]


def _set_peer_payee(df: pd.DataFrame, rows: np.ndarray, pop, idx: np.ndarray) -> None:
    df.loc[rows, "_peer_idx"] = idx
    df.loc[rows, "_merchant_idx"] = -1
    df.loc[rows, "payee_account_id"] = _as_account(idx)
    df.loc[rows, "merchant_id"] = ""
    df.loc[rows, "mcc"] = 0
    df.loc[rows, "payee_is_high_risk_category"] = 0
    df.loc[rows, "merchant_domain_age_days"] = 3650.0
    df.loc[rows, "payee_psp"] = pop.customers["psp_id"].to_numpy()[idx]
    df.loc[rows, "payee_account_age_days"] = np.round(
        pop.customers["tenure_days"].to_numpy()[idx].astype(float), 1
    )


def _retime(df: pd.DataFrame, rows: np.ndarray, ts: pd.DatetimeIndex, start, end) -> None:
    ts = pd.DatetimeIndex(np.clip(ts.values, start.to_datetime64(), end.to_datetime64()))
    df.loc[rows, "timestamp"] = ts
    df.loc[rows, "hour"] = ts.hour
    df.loc[rows, "day_of_week"] = ts.dayofweek
    df.loc[rows, "is_night"] = ((ts.hour < 6) | (ts.hour >= 23)).astype(int)


def _window(cfg):
    start = pd.Timestamp(cfg.benign.start_date)
    return start, start + pd.Timedelta(days=cfg.benign.n_days) - pd.Timedelta(seconds=1)


# --------------------------------------------------------------------------------------
# Shape 1: nightly sweeps out of a collection account
# --------------------------------------------------------------------------------------

def _sweeps(df: pd.DataFrame, rows: np.ndarray, rng, pop) -> pd.DataFrame:
    """A collector emptying its account into its own current account at close of business.

    This is what produces ``payee_outbound_ratio_24h`` near 1 - the signal the defence treats
    as the canonical layering tell - from a completely ordinary small business.
    """
    if not len(rows):
        return df
    payers = _collectors(pop, rng, len(rows))
    _set_payer(df, rows, pop, payers)

    # The destination is the same account every time: a business sweeping to its own current
    # account, not fanning out to strangers. That is the one thing that distinguishes it from
    # layering, and it is only learnable if the case exists in the data.
    dest = (payers + 1 + rng.integers(1, 40, len(rows))) % len(pop.customers)
    _set_peer_payee(df, rows, pop, dest)

    df.loc[rows, "rail"] = rng.choice(["IMPS", "NEFT", "UPI_P2P"], size=len(rows), p=[0.4, 0.35, 0.25])
    df.loc[rows, "channel"] = "mobile_app"
    df.loc[rows, "is_irrevocable_rail"] = np.isin(
        df.loc[rows, "rail"].to_numpy(), list(IRREVOCABLE_RAILS)
    ).astype(int)
    df.loc[rows, "currency"] = "INR"
    # A sweep moves the day's takings, so it is large relative to the account's usual payment
    # and it lands in the evening.
    df.loc[rows, "amount"] = np.round(
        np.exp(rng.normal(np.log(38_000), 0.85, len(rows))), 2
    )
    df.loc[rows, "account_balance_ratio"] = np.round(
        np.clip(rng.normal(0.93, 0.06, len(rows)), 0.4, 1.0), 4
    )
    ts = pd.DatetimeIndex(df.loc[rows, "timestamp"]).normalize() + pd.to_timedelta(
        rng.uniform(20.0, 23.4, len(rows)) * 3600, unit="s"
    )
    _retime(df, rows, ts, *_window_from(df))
    df.loc[rows, "_is_explore_payee"] = 0
    df.loc[rows, "_payee_reg_lag_s"] = np.round(rng.uniform(30 * 86400, 400 * 86400, len(rows)), 1)
    return df


def _window_from(df: pd.DataFrame):
    ts = pd.DatetimeIndex(df["timestamp"])
    return ts.min(), ts.max()


# --------------------------------------------------------------------------------------
# Shape 2: burst payers
# --------------------------------------------------------------------------------------

def _bursts(df: pd.DataFrame, rows: np.ndarray, rng, pop) -> pd.DataFrame:
    """Rent day, school fees, festival season: several payments from one account in minutes."""
    if len(rows) < 6:
        return df
    groups = _chunk(rows, rng, lo=3, hi=9)
    n_cust = len(pop.customers)
    for g in groups:
        payer = int(rng.integers(0, n_cust))
        _set_payer(df, g, pop, np.full(len(g), payer))
        anchor = pd.Timestamp(df.loc[g[0], "timestamp"])
        ts = pd.DatetimeIndex(
            [anchor + pd.Timedelta(seconds=float(s))
             for s in np.sort(rng.uniform(0, rng.uniform(180, 2_700), len(g)))]
        )
        _retime(df, g, ts, *_window_from(df))
    return df


# --------------------------------------------------------------------------------------
# Shape 3: split bills - many first-time payees, one session, irrevocable rail
# --------------------------------------------------------------------------------------

def _splits(df: pd.DataFrame, rows: np.ndarray, rng, pop) -> pd.DataFrame:
    """One person paying several people they have never paid before, in ninety seconds.

    This is the instant-rail fan-out signature - the shape the report calls out as the drain
    pattern - performed by a customer who must not be blocked. It is the single most useful
    hard negative in the whole file.
    """
    if len(rows) < 8:
        return df
    n_cust = len(pop.customers)
    for g in _chunk(rows, rng, lo=4, hi=9):
        payer = int(rng.integers(0, n_cust))
        _set_payer(df, g, pop, np.full(len(g), payer))
        # Distinct payees, none of them previously paid.
        peers = rng.choice(n_cust, size=len(g), replace=False)
        _set_peer_payee(df, g, pop, peers)
        df.loc[g, "rail"] = rng.choice(["UPI_P2P", "IMPS"], size=len(g), p=[0.88, 0.12])
        df.loc[g, "channel"] = "mobile_app"
        df.loc[g, "is_irrevocable_rail"] = 1
        df.loc[g, "currency"] = "INR"
        # A split bill is small and the shares are similar but not identical.
        share = float(np.exp(rng.normal(np.log(620), 0.7)))
        df.loc[g, "amount"] = np.round(share * rng.uniform(0.85, 1.15, len(g)), 2)
        df.loc[g, "_is_explore_payee"] = 1
        df.loc[g, "_payee_reg_lag_s"] = np.round(rng.uniform(20, 900, len(g)), 1)
        anchor = pd.Timestamp(df.loc[g[0], "timestamp"])
        ts = pd.DatetimeIndex(
            [anchor + pd.Timedelta(seconds=float(s))
             for s in np.sort(rng.uniform(0, rng.uniform(45, 240), len(g)))]
        )
        _retime(df, g, ts, *_window_from(df))
    return df


# --------------------------------------------------------------------------------------
# Shape 4: corporate disbursement runs
# --------------------------------------------------------------------------------------

def _disbursements(df: pd.DataFrame, rows: np.ndarray, rng, pop) -> pd.DataFrame:
    """A corporate paying many contractors, most of them new, in one session."""
    if len(rows) < 12:
        return df
    corp = np.where(pop.customers["is_corporate"].to_numpy() == 1)[0]
    if not corp.size:
        return df
    n_cust = len(pop.customers)
    for g in _chunk(rows, rng, lo=10, hi=45):
        payer = int(rng.choice(corp))
        _set_payer(df, g, pop, np.full(len(g), payer))
        peers = rng.choice(n_cust, size=len(g), replace=False)
        _set_peer_payee(df, g, pop, peers)
        df.loc[g, "rail"] = rng.choice(["NEFT", "IMPS", "RTP_FEDNOW"], size=len(g), p=[0.55, 0.35, 0.10])
        df.loc[g, "channel"] = "web"
        df.loc[g, "is_irrevocable_rail"] = np.isin(
            df.loc[g, "rail"].to_numpy(), list(IRREVOCABLE_RAILS)
        ).astype(int)
        df.loc[g, "customer_is_corporate"] = 1
        df.loc[g, "amount"] = np.round(np.exp(rng.normal(np.log(24_000), 0.6, len(g))), 2)
        # Most payees are new every cycle: contractors churn.
        fresh = rng.random(len(g)) < 0.7
        df.loc[g[fresh], "_is_explore_payee"] = 1
        anchor = pd.Timestamp(df.loc[g[0], "timestamp"]).normalize() + pd.Timedelta(
            seconds=float(rng.uniform(10, 18) * 3600)
        )
        ts = pd.DatetimeIndex(
            [anchor + pd.Timedelta(seconds=float(s))
             for s in np.sort(rng.uniform(0, rng.uniform(600, 5_400), len(g)))]
        )
        _retime(df, g, ts, *_window_from(df))
    return df


def _chunk(rows: np.ndarray, rng, *, lo: int, hi: int):
    """Split a flat index array into consecutive groups of random size in [lo, hi]."""
    out = []
    i = 0
    while i < len(rows):
        k = int(rng.integers(lo, hi + 1))
        if i + k > len(rows):
            if len(rows) - i >= lo:
                out.append(rows[i:])
            break
        out.append(rows[i:i + k])
        i += k
    return out
