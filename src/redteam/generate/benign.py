"""Legitimate payment traffic simulator.

Design notes that matter for fidelity:

* **Amounts** are merchant-anchored, not globally drawn. Ticket size is a property of the
  merchant category scaled by a persistent per-customer spending factor, so per-customer
  and per-MCC amount distributions are both realistic, and Benford's law emerges rather
  than being imposed.
* **Round-number mass.** Real UPI and P2P data has heavy spikes at 100/500/1000. A purely
  lognormal simulator has none, and any classifier trained on it will treat the round
  amounts common in scams as anomalous for the wrong reason.
* **Payee reuse** follows a rank-preference law over each customer's payee book, so
  "first time payee" is genuinely rare and genuinely informative.
* **Hard negatives** are legitimate payments deliberately given a suspicious surface:
  travel, a new device, a first-time high-value payee, a support call in progress. Without
  them a defence learns shortcuts and posts a fake 0.999 AUC.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..schema import DEFAULTS, IRREVOCABLE_RAILS
from . import timing
from .entities import Population, haversine_km
from .shapes import apply_legitimate_shapes

SECONDS_PER_DAY = 86_400

# Rails that move money to a merchant rather than a person.
MERCHANT_RAILS = {"UPI_P2M", "CARD_CNP", "CARD_CP", "WALLET"}
PEER_RAILS = {"UPI_P2P", "IMPS", "NEFT", "RTP_FEDNOW", "SEPA_INST"}

# Authentication method by rail, as a probability vector over AUTH_METHODS subsets.
RAIL_AUTH: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {
    "UPI_P2M": (("upi_pin", "biometric"), (0.72, 0.28)),
    "UPI_P2P": (("upi_pin", "biometric"), (0.80, 0.20)),
    "IMPS": (("otp_sms", "biometric", "passkey"), (0.55, 0.30, 0.15)),
    "NEFT": (("otp_sms", "passkey"), (0.70, 0.30)),
    "RTP_FEDNOW": (("passkey", "otp_sms", "biometric"), (0.45, 0.35, 0.20)),
    "SEPA_INST": (("passkey", "otp_sms", "biometric"), (0.50, 0.32, 0.18)),
    "CARD_CNP": (("3ds2", "cvv_only"), (0.74, 0.26)),
    "CARD_CP": (("none", "biometric"), (0.82, 0.18)),
    "WALLET": (("biometric", "upi_pin"), (0.60, 0.40)),
}

RAIL_CHANNEL: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {
    "UPI_P2M": (("mobile_app", "pos"), (0.86, 0.14)),
    "UPI_P2P": (("mobile_app",), (1.0,)),
    "IMPS": (("mobile_app", "web"), (0.72, 0.28)),
    "NEFT": (("web", "mobile_app", "branch_assisted"), (0.58, 0.36, 0.06)),
    "RTP_FEDNOW": (("web", "mobile_app"), (0.55, 0.45)),
    "SEPA_INST": (("web", "mobile_app"), (0.60, 0.40)),
    "CARD_CNP": (("web", "mobile_app", "recurring"), (0.52, 0.36, 0.12)),
    "CARD_CP": (("pos",), (1.0,)),
    "WALLET": (("mobile_app",), (1.0,)),
}

RAIL_CURRENCY = {"RTP_FEDNOW": "USD", "SEPA_INST": "EUR"}

#: Units of currency per rupee. Amounts are drawn on an INR scale throughout - the population,
#: the merchant ticket sizes and the reporting thresholds are all Indian - so a rail that
#: settles in another currency has to be converted rather than merely relabelled. Stamping
#: "USD" on an unconverted figure put the median FedNow payment near $9,800 and its tail in the
#: millions, which is the kind of thing a payments reviewer spots in one glance at a CSV.
FX_PER_INR = {"USD": 1.0 / 88.0, "EUR": 1.0 / 96.0, "INR": 1.0}


def convert_currency(amount: np.ndarray, currency: np.ndarray) -> np.ndarray:
    """Restate INR-scale amounts in the currency their rail actually settles in."""
    rate = pd.Series(currency).map(FX_PER_INR).fillna(1.0).to_numpy()
    return np.round(amount * rate, 2)

# Legitimate phone-banking rate on push rails, as a floor plus a term that rises as digital
# literacy falls. See ``_assisted_channel``.
IVR_BASE_RATE = 0.004
IVR_LITERACY_SLOPE = 0.055

#: Share of hard negatives that wear more than one suspicious costume at once. See
#: ``_apply_hard_negatives`` for why single-axis traps are not enough.
COMPOUND_HARD_NEGATIVE_SHARE = 0.35


def simulate_benign(cfg: Config, pop: Population, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the legitimate transaction stream for the whole simulation window."""
    n_cust = len(pop.customers)
    n_days = cfg.benign.n_days
    start = pd.Timestamp(cfg.benign.start_date)

    # ---- how many payments does each customer make over the window? -------------------
    lam = cfg.benign.base_txns_per_customer_per_day * pop.customers["activity"].to_numpy() * n_days
    counts = rng.poisson(lam)
    counts = np.maximum(counts, 1)
    total = int(counts.sum())
    cust_idx = np.repeat(np.arange(n_cust), counts)

    cust = pop.customers
    merch = pop.merchants

    # ---- timing -----------------------------------------------------------------------
    day = _draw_days(rng, total, n_days)
    hour, minute, second = _draw_time_of_day(rng, total, cust["night_owl"].to_numpy()[cust_idx])
    ts_offset = day * SECONDS_PER_DAY + hour * 3600 + minute * 60 + second
    timestamp = start + pd.to_timedelta(ts_offset, unit="s")

    # ---- rail -------------------------------------------------------------------------
    rail = _draw_rails(cfg, rng, total, cust_idx, cust)
    is_merchant_rail = np.isin(rail, list(MERCHANT_RAILS))

    # ---- counterparty ------------------------------------------------------------------
    merchant_idx, peer_idx, is_new_payee = _draw_counterparties(
        rng, cust_idx, is_merchant_rail, pop
    )

    # ---- amount -------------------------------------------------------------------------
    amount = _draw_amounts(rng, cust_idx, merchant_idx, peer_idx, is_merchant_rail, rail, cust, merch)

    # ---- assemble ------------------------------------------------------------------------
    df = pd.DataFrame({"_customer_idx": cust_idx})
    df["txn_id"] = [f"T{i:09d}" for i in range(total)]
    df["timestamp"] = timestamp
    df["customer_id"] = cust["customer_id"].to_numpy()[cust_idx]
    df["payer_account_id"] = np.char.add("A", np.char.zfill(cust_idx.astype(str), 7))
    df["_merchant_idx"] = merchant_idx
    df["_peer_idx"] = peer_idx

    payee = np.where(
        is_merchant_rail,
        merch["merchant_id"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)],
        np.char.add("A", np.char.zfill(np.clip(peer_idx, 0, n_cust - 1).astype(str), 7)),
    )
    df["payee_account_id"] = payee
    df["merchant_id"] = np.where(is_merchant_rail, merch["merchant_id"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)], "")

    df["rail"] = rail
    df["channel"] = _assisted_channel(rng, _draw_from_map(rng, rail, RAIL_CHANNEL), rail, cust, cust_idx)
    df["amount"] = amount
    df["currency"] = np.where(np.isin(rail, list(RAIL_CURRENCY)),
                              pd.Series(rail).map(RAIL_CURRENCY).fillna("INR").to_numpy(), "INR")
    df["is_irrevocable_rail"] = np.isin(rail, list(IRREVOCABLE_RAILS)).astype(int)
    df["mcc"] = np.where(is_merchant_rail, merch["mcc"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)], 0)
    df["is_cross_border"] = np.where(
        is_merchant_rail,
        merch["is_cross_border"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)],
        (rng.random(total) < 0.012).astype(int),
    )
    df["payer_psp"] = cust["psp_id"].to_numpy()[cust_idx]
    df["payee_psp"] = np.where(
        is_merchant_rail,
        merch["acquirer_psp"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)],
        cust["psp_id"].to_numpy()[np.clip(peer_idx, 0, n_cust - 1)],
    )

    _add_iso20022(df, rng, rail, total)
    _add_auth(df, rng, rail, total, cust, cust_idx)
    _add_device(df, rng, pop, cust_idx, day, total)
    _add_behaviour(df, rng, cust, cust_idx, total, amount)
    _add_payee_static(df, rng, pop, merchant_idx, peer_idx, is_merchant_rail, is_new_payee,
                      total, day)
    _add_agentic(df, rng, cust, cust_idx, merch, merchant_idx, is_merchant_rail, total)
    _add_customer_context(df, cust, cust_idx, amount, day)
    _add_time_features(df, timestamp)

    # Fill in every remaining schema field with its benign default. Built as one frame and
    # concatenated rather than assigned column by column: there are enough defaults that
    # repeated insertion fragments the block manager and pandas warns about it.
    missing = {col: default for col, default in DEFAULTS.items() if col not in df.columns}
    if missing:
        df = pd.concat(
            [df, pd.DataFrame(missing, index=df.index)],
            axis=1,
        )

    df = _apply_hard_negatives(cfg, df, rng, pop, cust_idx)
    # Legitimate traffic that carries the shapes the defence hunts for: collectors sweeping
    # their takings, split bills fanning out to strangers on an irrevocable rail, corporate
    # disbursement runs. Applied before the sort so the rewritten episodes are ordered with
    # everything else.
    df = apply_legitimate_shapes(df, rng, pop, cfg)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------

def _draw_days(rng: np.random.Generator, total: int, n_days: int) -> np.ndarray:
    return timing.draw_days(rng, total, n_days)


def _draw_time_of_day(rng: np.random.Generator, total: int, night_owl: np.ndarray):
    """Bimodal diurnal profile (late-morning and evening peaks) plus a night-owl tail."""
    hour = timing.draw_hours(rng, total, night_owl=night_owl)
    h, minute, second = timing.hours_to_hms(rng, hour)
    return h, minute, second


# --------------------------------------------------------------------------------------
# Rails, counterparties, amounts
# --------------------------------------------------------------------------------------

def _draw_rails(cfg: Config, rng, total: int, cust_idx: np.ndarray, cust: pd.DataFrame) -> np.ndarray:
    rails = list(cfg.benign.rail_mix.keys())
    base = np.array(list(cfg.benign.rail_mix.values()), dtype=float)
    base = base / base.sum()

    # Per-customer rail preference: a persistent Dirichlet tilt on the global mix, so
    # customers have habits rather than resampling the population mix every payment.
    n_cust = len(cust)
    tilt = rng.dirichlet(base * 26.0, size=n_cust)

    # Corporates barely touch consumer rails.
    corp = cust["is_corporate"].to_numpy() == 1
    corp_profile = np.array([0.06, 0.02, 0.10, 0.02, 0.24, 0.30, 0.14, 0.10, 0.02])
    corp_profile = corp_profile / corp_profile.sum()
    if len(corp_profile) == len(rails):
        tilt[corp] = corp_profile

    probs = tilt[cust_idx]
    # Vectorised categorical sampling via the inverse CDF.
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((total, 1))
    pick = (u > cdf).sum(axis=1)
    return np.array(rails, dtype=object)[np.clip(pick, 0, len(rails) - 1)]


def _rank_preference_positions(rng, lengths: np.ndarray) -> np.ndarray:
    """Pick a position inside each customer's payee book with rank preference.

    A geometric draw makes the top few payees dominate, matching the observed
    concentration of real payment relationships.
    """
    pos = rng.geometric(0.34, size=len(lengths)) - 1
    return np.minimum(pos, np.maximum(lengths - 1, 0))


def _flatten_book(book: Dict[int, np.ndarray], n: int):
    lens = np.array([len(book[i]) for i in range(n)])
    offsets = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate([book[i] for i in range(n)]) if lens.sum() else np.array([], dtype=int)
    return flat, offsets[:-1], lens


def _draw_counterparties(rng, cust_idx, is_merchant_rail, pop: Population):
    n_cust = len(pop.customers)
    total = len(cust_idx)

    m_flat, m_off, m_len = _flatten_book(pop.payee_book, n_cust)
    p_flat, p_off, p_len = _flatten_book(pop.peer_book, n_cust)

    merchant_idx = np.full(total, -1, dtype=int)
    peer_idx = np.full(total, -1, dtype=int)
    is_new_payee = np.zeros(total, dtype=int)

    # --- merchant rails ---
    mi = np.where(is_merchant_rail)[0]
    if mi.size:
        c = cust_idx[mi]
        # A meaningful share of merchant payments go somewhere new. Set this too low and
        # "first time payee" stops being a normal event and becomes a fraud giveaway.
        explore = rng.random(mi.size) < 0.13
        pos = _rank_preference_positions(rng, m_len[c])
        picked = m_flat[m_off[c] + pos]
        pop_p = pop.merchants["popularity"].to_numpy()
        pop_p = pop_p / pop_p.sum()
        novel = rng.choice(len(pop.merchants), size=int(explore.sum()), p=pop_p)
        picked[explore] = novel
        merchant_idx[mi] = picked
        is_new_payee[mi[explore]] = 1

    # --- peer rails ---
    pi = np.where(~is_merchant_rail)[0]
    if pi.size:
        c = cust_idx[pi]
        explore = rng.random(pi.size) < 0.17
        pos = _rank_preference_positions(rng, p_len[c])
        picked = p_flat[p_off[c] + pos]
        # First-time peer payees are drawn towards legitimate collection points rather than
        # uniformly across the population. A uniform draw spreads inbound novelty one payment
        # per account, so no legitimate account ever looks like a collector and payee fan-in
        # becomes a perfect mule detector - which is a property of the generator, not of fraud.
        w = pop.customers["collector_weight"].to_numpy()
        w = w / w.sum()
        picked[explore] = rng.choice(n_cust, size=int(explore.sum()), p=w)
        peer_idx[pi] = picked
        is_new_payee[pi[explore]] = 1

    return merchant_idx, peer_idx, is_new_payee


def _draw_amounts(rng, cust_idx, merchant_idx, peer_idx, is_merchant_rail, rail, cust, merch):
    total = len(cust_idx)
    spend = cust["spend_scale"].to_numpy()[cust_idx]
    amount = np.empty(total)

    mi = np.where(is_merchant_rail)[0]
    if mi.size:
        m = np.clip(merchant_idx[mi], 0, len(merch) - 1)
        centre = merch["median_ticket"].to_numpy()[m] * np.power(spend[mi], 0.75)
        sigma = merch["ticket_sigma"].to_numpy()[m]
        amount[mi] = centre * np.exp(rng.normal(0, sigma))

    pi = np.where(~is_merchant_rail)[0]
    if pi.size:
        # Person-to-person transfers: higher median, much fatter tail than retail.
        base = np.where(np.isin(rail[pi], ["NEFT", "RTP_FEDNOW", "SEPA_INST"]), 9800.0, 2400.0)
        amount[pi] = base * np.power(spend[pi], 0.85) * np.exp(rng.normal(0, 1.15, pi.size))

    amount = np.clip(amount, 5.0, 4_500_000.0)
    return _apply_round_number_mass(rng, amount)


def _apply_round_number_mass(rng, amount: np.ndarray) -> np.ndarray:
    """Snap a realistic share of payments to round values.

    Human-chosen amounts cluster hard on 100/500/1000 multiples. Omitting this is the
    single most common tell of synthetic payment data.
    """
    out = amount.copy()
    snap = rng.random(len(out)) < 0.27
    step = np.select(
        [out < 250, out < 2_000, out < 25_000, out < 200_000],
        [10.0, 50.0, 100.0, 1_000.0],
        default=5_000.0,
    )
    out[snap] = np.maximum(np.round(out[snap] / step[snap]) * step[snap], step[snap])
    return np.round(out, 2)


def _assisted_channel(rng, channel: np.ndarray, rail: np.ndarray,
                      cust: pd.DataFrame, cust_idx: np.ndarray) -> np.ndarray:
    """Move a slice of push payments onto phone banking.

    Without this, ``ivr`` appears only under attack and the channel alone convicts: a
    telephone-banking payment would be fraud with probability one, which is both false and
    the kind of artefact that flatters a detector. Phone banking is a small but real share
    of push volume and it concentrates in exactly the cohort attackers target, so the
    channel stays *correlated* with fraud without being decisive - the model has to weigh it
    against everything else, as it would in production.
    """
    push = np.isin(rail, list(PEER_RAILS))
    literacy = cust["digital_literacy"].to_numpy()[cust_idx]
    p = np.where(push, IVR_BASE_RATE + IVR_LITERACY_SLOPE * (1.0 - literacy), 0.0)
    out = channel.copy()
    out[rng.random(len(out)) < p] = "ivr"
    return out


def _draw_from_map(rng, rail: np.ndarray, mapping) -> np.ndarray:
    out = np.empty(len(rail), dtype=object)
    for r, (options, probs) in mapping.items():
        m = rail == r
        k = int(m.sum())
        if k:
            out[m] = rng.choice(np.array(options, dtype=object), size=k, p=np.array(probs))
    return out


# --------------------------------------------------------------------------------------
# Field groups
# --------------------------------------------------------------------------------------

def _add_iso20022(df, rng, rail, total):
    """Structured-messaging fields. Only ISO 20022 rails carry meaningful content."""
    iso_rail = np.isin(rail, ["NEFT", "RTP_FEDNOW", "SEPA_INST", "IMPS"])
    length = np.where(iso_rail, rng.gamma(2.6, 16.0, total), 0.0)
    df["iso20022_remittance_len"] = np.round(length, 1)
    # Shannon entropy of a natural remittance string sits around 3.6-4.2 bits/char.
    df["iso20022_field_entropy"] = np.where(iso_rail, np.clip(rng.normal(3.9, 0.28, total), 0, 8), 0.0)
    df["iso20022_unstructured_ratio"] = np.where(iso_rail, np.clip(rng.beta(2.0, 5.0, total), 0, 1), 0.0)


#: Share of legitimate phone-banking payments authenticated by voice biometrics. Banks that
#: run an IVR payment flow authenticate it with a voice print, so if only the voice-clone
#: attack ever produces one, the auth method *is* the label. The cloning attack has to hide
#: inside a population of genuine voice-authenticated calls to be a detection problem at all.
IVR_VOICE_PRINT_SHARE = 0.42


def _add_auth(df, rng, rail, total, cust, cust_idx):
    df["auth_method"] = _draw_from_map(rng, rail, RAIL_AUTH)
    on_phone = (df["channel"].to_numpy() == "ivr") & (rng.random(total) < IVR_VOICE_PRINT_SHARE)
    df.loc[on_phone, "auth_method"] = "voice_print"
    df["auth_attempts"] = np.where(rng.random(total) < 0.045, 2, 1)
    # Authentication latency: log-normal around ~4s, longer for OTP flows.
    is_otp = df["auth_method"].to_numpy() == "otp_sms"
    df["auth_latency_ms"] = np.round(
        np.exp(rng.normal(np.where(is_otp, 9.6, 8.3), 0.45, total)), 0
    )
    df["otp_delivery_delay_s"] = np.where(is_otp, np.clip(rng.gamma(2.0, 2.4, total), 0, 90), 0.0)
    df["step_up_triggered"] = (rng.random(total) < 0.06).astype(int)
    df["mfa_passed"] = 1
    # Time since the customer last changed a credential. Mostly "never recently".
    recent = rng.random(total) < 0.02
    df["recent_credential_change_h"] = np.where(recent, rng.uniform(1, 720, total), 9999.0)
    swap = rng.random(total) < 0.004
    df["sim_swap_recency_days"] = np.where(swap, rng.uniform(1, 120, total), 9999.0)


def _add_device(df, rng, pop: Population, cust_idx, day, total):
    devices = pop.devices
    dev_of_cust = pop.customer_devices
    lens = np.array([len(dev_of_cust[i]) for i in range(len(pop.customers))])
    flat = np.concatenate([dev_of_cust[i] for i in range(len(pop.customers))])
    off = np.concatenate([[0], np.cumsum(lens)])[:-1]

    # Primary device dominates; secondary device used ~14% of the time when present.
    use_secondary = (rng.random(total) < 0.14) & (lens[cust_idx] > 1)
    pos = np.where(use_secondary, 1, 0)
    dev_pos = flat[off[cust_idx] + pos]

    age_at_start = devices["age_days_at_start"].to_numpy()[dev_pos]
    device_age = age_at_start + day
    # A small share of payments come from a genuinely new device (upgrade, reinstall).
    fresh = rng.random(total) < 0.012
    device_age = np.where(fresh, rng.uniform(0, 3, total), device_age)

    df["device_id"] = devices["device_id"].to_numpy()[dev_pos]
    df["device_age_days"] = np.round(device_age, 2)
    df["device_is_new"] = (device_age < 7).astype(int)
    df["device_is_emulator"] = (rng.random(total) < 0.0015).astype(int)
    df["device_is_rooted"] = devices["is_rooted"].to_numpy()[dev_pos]

    # Legitimate baseline rates for the "coercion" signals. Non-zero on purpose: people do
    # share screens with real support agents and do pay while on a call.
    df["screen_share_active"] = (rng.random(total) < 0.0045).astype(int)
    df["remote_access_app_detected"] = (rng.random(total) < 0.0032).astype(int)
    df["accessibility_service_active"] = (rng.random(total) < 0.019).astype(int)
    df["call_in_progress"] = (rng.random(total) < 0.021).astype(int)
    df["vpn_or_proxy"] = (rng.random(total) < 0.036).astype(int)
    df["ip_asn_is_hosting"] = (rng.random(total) < 0.006).astype(int)
    df["ip_country_mismatch"] = (rng.random(total) < 0.009).astype(int)

    # Geography: usually near home, occasionally travelling.
    cust = pop.customers
    home_lat = cust["home_lat"].to_numpy()[cust_idx]
    home_lon = cust["home_lon"].to_numpy()[cust_idx]
    travelling = rng.random(total) < 0.035
    lat = home_lat + np.where(travelling, rng.normal(0, 4.0, total), rng.normal(0, 0.045, total))
    lon = home_lon + np.where(travelling, rng.normal(0, 4.0, total), rng.normal(0, 0.045, total))
    dist = haversine_km(home_lat, home_lon, lat, lon)
    df["distance_from_home_km"] = np.round(dist, 3)
    # Implied speed since the previous payment is computed later; the instantaneous proxy
    # here is what a device-intelligence vendor would return.
    df["geo_velocity_kmh"] = np.round(np.where(travelling, np.clip(rng.gamma(2.0, 90.0, total), 0, 900), rng.gamma(1.2, 6.0, total)), 2)
    df["session_duration_s"] = np.round(np.clip(rng.gamma(2.2, 34.0, total), 3, 3600), 1)
    df["app_backgrounded_count"] = rng.poisson(0.35, total)
    df["_lat"] = lat
    df["_lon"] = lon


def _add_behaviour(df, rng, cust, cust_idx, total, amount):
    """Passive behavioural biometrics, anchored on age band and digital literacy."""
    band = cust["age_band_ord"].to_numpy()[cust_idx]
    literacy = cust["digital_literacy"].to_numpy()[cust_idx]

    # Older / less fluent users type more slowly and hesitate more. Within-person variance
    # is small, between-person variance is large: that is what makes it a biometric.
    flight_centre = 118.0 + 46.0 * band + 40.0 * (1 - literacy)
    df["keystroke_flight_mean_ms"] = np.round(np.clip(rng.normal(flight_centre, 22.0, total), 45, 600), 2)
    # Coefficient of variation of flight times. Humans sit around 0.30-0.55; scripted
    # input is far more regular, which is exactly what mimicry attacks try to fake.
    df["keystroke_flight_cv"] = np.round(np.clip(rng.normal(0.42, 0.085, total), 0.05, 1.2), 4)
    df["typing_burstiness"] = np.round(np.clip(rng.normal(0.51, 0.13, total), 0, 1), 4)
    df["pointer_entropy"] = np.round(np.clip(rng.normal(4.35, 0.55, total), 0.5, 8), 4)

    log_amt = np.log10(np.maximum(amount, 1))
    # People spend longer on bigger payments and on unfamiliar screens.
    df["form_fill_duration_s"] = np.round(
        np.clip(rng.gamma(2.6, 4.0 + 1.9 * log_amt + 2.2 * band, total) / 2.6, 1.5, 900), 2
    )
    df["payee_field_pasted"] = (rng.random(total) < 0.17).astype(int)
    df["hesitation_events"] = rng.poisson(0.55 + 0.32 * band + 0.25 * log_amt.clip(0, 6))
    df["amount_field_corrections"] = rng.poisson(0.28 + 0.12 * band)

    # Biometric / liveness scores only exist where such a check actually ran.
    bio_used = df["auth_method"].isin(["biometric", "passkey"]).to_numpy()
    df["biometric_match_score"] = np.where(bio_used, np.clip(rng.beta(14, 1.5, total), 0, 1), 0.0)
    voice_used = df["channel"].to_numpy() == "ivr"
    df["voice_match_score"] = np.where(voice_used, np.clip(rng.beta(11, 2.0, total), 0, 1), 0.0)
    df["liveness_score"] = np.where(bio_used, np.clip(rng.beta(16, 1.6, total), 0, 1), 0.0)
    df["doc_ocr_confidence"] = 0.0


def _add_payee_static(df, rng, pop: Population, merchant_idx, peer_idx, is_merchant_rail,
                      is_new_payee, total, day):
    """Static counterparty attributes. Dynamic aggregates are computed in enrichment."""
    merch = pop.merchants
    m = np.clip(merchant_idx, 0, len(merch) - 1)
    p = np.clip(peer_idx, 0, len(pop.customers) - 1)

    # Ages advance with the simulation clock, exactly as ``customer_tenure_days`` does.
    # Freezing them at their day-zero value left a storefront that existed on day 1 still
    # twenty days old on day 45, and made counterparty age disagree with the account history
    # the enrichment step derives from the event stream immediately beside it.
    domain_age = merch["domain_age_days"].to_numpy()[m] + day
    peer_age = pop.customers["tenure_days"].to_numpy()[p].astype(float) + day

    # Payee account age: merchants inherit domain age, peers inherit customer tenure.
    df["payee_account_age_days"] = np.where(is_merchant_rail, domain_age, peer_age).round(1)
    df["merchant_domain_age_days"] = np.where(is_merchant_rail, domain_age, 3650.0).round(1)
    df["payee_is_high_risk_category"] = np.where(
        is_merchant_rail, merch["is_high_risk_category"].to_numpy()[m], 0
    )
    # Confirmation-of-Payee name match. Legitimate payments still miss sometimes:
    # nicknames, married names, initials, business trading names.
    match = np.clip(rng.beta(9.0, 1.1, total), 0, 1)
    df["payee_name_match_score"] = np.round(match, 4)
    df["payee_name_homoglyph_flag"] = 0
    # Registration lag: how long before the payment the payee entered the address book.
    df["_payee_reg_lag_s"] = np.round(np.clip(rng.lognormal(6.4, 1.5, total), 20, 3 * 86400), 1)
    df["_is_explore_payee"] = is_new_payee


def _add_agentic(df, rng, cust, cust_idx, merch, merchant_idx, is_merchant_rail, total):
    """Delegated-authority envelope for the share of traffic driven by AI agents."""
    adoption = cust["agent_adoption"].to_numpy()[cust_idx]
    by_agent = (rng.random(total) < adoption * 0.45) & is_merchant_rail
    n_agent = int(by_agent.sum())

    df["initiated_by_agent"] = by_agent.astype(int)
    registered = by_agent & (rng.random(total) < 0.94)
    df["agent_is_registered"] = registered.astype(int)

    # An enrolled agent asserts its identity and signs for it. The remaining 6% are legacy
    # integrations that never enrolled: they assert nothing, which costs them the trusted
    # lane but is not itself misconduct.
    #
    # The 1.8% of enrolled agents whose signature fails to verify are the reason this
    # control raises friction rather than declining. Key rotation that outpaces the
    # verifier's cache, a proxy that strips the Signature header, and clock skew past the
    # created-at window all produce a genuine verification failure on an honest request.
    # Without that tail the asserted-but-unsigned combination would be fraud-exclusive, the
    # model would learn it as an oracle, and the reported recall would be measuring the
    # generator rather than the defence.
    signed = registered & (rng.random(total) >= 0.018)
    df["agent_identity_asserted"] = registered.astype(int)
    df["agent_request_signature_valid"] = signed.astype(int)
    df["agent_reputation"] = np.where(by_agent, np.clip(rng.beta(8, 2, total), 0, 1), 0.0)
    # A well-behaved agent presents a signed intent artefact that matches what it does.
    df["intent_token_present"] = np.where(by_agent, (rng.random(total) < 0.97).astype(int), 0)
    df["intent_signature_valid"] = df["intent_token_present"].to_numpy()
    df["intent_payee_match"] = df["intent_token_present"].to_numpy()
    # Small legitimate drift between quoted and final amount: tax, shipping, FX.
    df["intent_amount_delta_ratio"] = np.where(
        by_agent, np.round(np.clip(np.abs(rng.normal(0.008, 0.012, total)), 0, 0.25), 5), 0.0
    )
    df["intent_age_s"] = np.where(by_agent, np.round(np.clip(rng.gamma(2.0, 22.0, total), 1, 900), 1), 0.0)
    df["agent_tool_calls"] = np.where(by_agent, rng.poisson(5.5, total) + 1, 0)
    df["agent_untrusted_content_tokens"] = np.where(
        by_agent, rng.poisson(1400, total), 0
    )
    # Baseline injection score from the content classifier: low but not identically zero.
    df["agent_injection_score"] = np.where(
        by_agent, np.round(np.clip(rng.beta(1.4, 22.0, total), 0, 1), 5), 0.0
    )
    df["spt_scope_violation"] = 0
    df["spt_reuse_count"] = np.where(by_agent, rng.binomial(1, 0.03, total), 0)
    # Links the transaction to its context bundle in the agentic corpus. Empty = clean.
    df["_agent_payload"] = ""
    df["merchant_is_agent_optimised"] = np.where(
        is_merchant_rail, merch["is_agent_optimised"].to_numpy()[np.clip(merchant_idx, 0, len(merch) - 1)], 0
    )
    # Agent-driven checkouts are machine-paced: no human typing telemetry at all.
    if n_agent:
        idx = np.where(by_agent)[0]
        df.loc[idx, "channel"] = "agent_api"
        df.loc[idx, "keystroke_flight_mean_ms"] = 0.0
        df.loc[idx, "keystroke_flight_cv"] = 0.0
        df.loc[idx, "typing_burstiness"] = 0.0
        df.loc[idx, "pointer_entropy"] = 0.0
        df.loc[idx, "hesitation_events"] = 0
        df.loc[idx, "amount_field_corrections"] = 0
        df.loc[idx, "form_fill_duration_s"] = np.round(rng.gamma(2.0, 0.9, n_agent), 3)


def _add_customer_context(df, cust, cust_idx, amount, day):
    df["customer_tenure_days"] = cust["tenure_days"].to_numpy()[cust_idx] + day
    df["customer_age_band_ord"] = cust["age_band_ord"].to_numpy()[cust_idx]
    df["customer_digital_literacy"] = np.round(cust["digital_literacy"].to_numpy()[cust_idx], 4)
    df["customer_is_corporate"] = cust["is_corporate"].to_numpy()[cust_idx]
    balance = cust["balance"].to_numpy()[cust_idx]
    df["account_balance_ratio"] = np.round(np.clip(amount / np.maximum(balance, 100.0), 0, 50), 5)
    df["kyc_level"] = cust["kyc_level"].to_numpy()[cust_idx]
    df["prior_fraud_reports"] = cust["prior_fraud_reports"].to_numpy()[cust_idx]


def _add_time_features(df, timestamp: pd.Series):
    ts = pd.DatetimeIndex(timestamp)
    df["hour"] = ts.hour
    df["day_of_week"] = ts.dayofweek
    df["is_night"] = ((ts.hour < 6) | (ts.hour >= 23)).astype(int)
    df["is_salary_window"] = timing.is_salary_window(ts.day.to_numpy()).astype(int)


# --------------------------------------------------------------------------------------
# Hard negatives
# --------------------------------------------------------------------------------------

def _apply_hard_negatives(cfg: Config, df: pd.DataFrame, rng, pop: Population, cust_idx) -> pd.DataFrame:
    """Legitimate payments that wear a suspicious costume.

    Each archetype mirrors a real customer-service scenario that fraud teams see every
    day and must not block. They are labelled so evaluation can report false positives on
    the hard slice separately from the easy slice.
    """
    n = len(df)
    chosen = rng.random(n) < cfg.benign.hard_negative_rate
    idx = np.where(chosen)[0]
    if idx.size == 0:
        return df

    kinds = ["travel", "new_device", "big_ticket_new_payee", "support_call", "elderly_first_upi",
             "salary_disbursement", "genuine_screen_share"]
    probs = [0.20, 0.18, 0.19, 0.14, 0.11, 0.10, 0.08]
    kind = rng.choice(kinds, size=idx.size, p=probs)

    # A costume on one axis is easy to see through: any model can learn to accept a single
    # anomaly and decline several. Real false positives are not like that. The customer who
    # lands abroad, reinstalls the app on a replacement phone, calls support because the
    # payment failed, and then sends a large sum to a payee added ten minutes ago is one
    # customer, and every one of those facts is true at once. Stacking archetypes is what
    # denies the model the "count the red flags" shortcut and makes its precision claim mean
    # something.
    stacked = idx[rng.random(idx.size) < COMPOUND_HARD_NEGATIVE_SHARE]
    extra_kind = rng.choice(kinds, size=(len(stacked), 2), p=probs)

    df["is_hard_negative"] = 0
    df.loc[idx, "is_hard_negative"] = 1

    def sel(name):
        base = idx[kind == name]
        if len(stacked) == 0:
            return base
        also = stacked[(extra_kind == name).any(axis=1)]
        return np.union1d(base, also)

    i = sel("travel")
    if i.size:
        df.loc[i, "distance_from_home_km"] = np.round(rng.uniform(400, 6500, i.size), 1)
        df.loc[i, "geo_velocity_kmh"] = np.round(rng.uniform(300, 850, i.size), 1)
        df.loc[i, "ip_country_mismatch"] = (rng.random(i.size) < 0.55).astype(int)
        df.loc[i, "vpn_or_proxy"] = (rng.random(i.size) < 0.30).astype(int)

    i = sel("new_device")
    if i.size:
        df.loc[i, "device_age_days"] = np.round(rng.uniform(0, 2, i.size), 3)
        df.loc[i, "device_is_new"] = 1
        df.loc[i, "recent_credential_change_h"] = np.round(rng.uniform(0.2, 48, i.size), 2)
        df.loc[i, "step_up_triggered"] = 1

    i = sel("big_ticket_new_payee")
    if i.size:
        df.loc[i, "amount"] = np.round(df.loc[i, "amount"].to_numpy() * rng.uniform(8, 45, i.size), 2)
        df.loc[i, "_payee_reg_lag_s"] = np.round(rng.uniform(60, 900, i.size), 1)
        df.loc[i, "_is_explore_payee"] = 1
        df.loc[i, "hesitation_events"] = rng.poisson(3.0, i.size)

    i = sel("support_call")
    if i.size:
        df.loc[i, "call_in_progress"] = 1
        df.loc[i, "session_duration_s"] = np.round(rng.uniform(240, 1800, i.size), 1)

    i = sel("elderly_first_upi")
    if i.size:
        df.loc[i, "customer_age_band_ord"] = 4
        df.loc[i, "keystroke_flight_mean_ms"] = np.round(rng.uniform(280, 520, i.size), 1)
        df.loc[i, "form_fill_duration_s"] = np.round(rng.uniform(120, 600, i.size), 1)
        df.loc[i, "hesitation_events"] = rng.poisson(5.0, i.size)
        df.loc[i, "_is_explore_payee"] = 1

    i = sel("salary_disbursement")
    if i.size:
        df.loc[i, "customer_is_corporate"] = 1
        df.loc[i, "amount"] = np.round(df.loc[i, "amount"].to_numpy() * rng.uniform(10, 60, i.size), 2)
        df.loc[i, "is_salary_window"] = 1

    i = sel("genuine_screen_share")
    if i.size:
        df.loc[i, "screen_share_active"] = 1
        df.loc[i, "call_in_progress"] = 1
        df.loc[i, "session_duration_s"] = np.round(rng.uniform(300, 2400, i.size), 1)

    return df
