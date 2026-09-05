"""Synthetic financial ecosystem: customers, merchants, devices, PSPs and payee graphs.

Fidelity of the *entities* is what makes the fidelity of the *transactions* possible.
A transaction simulator that draws amounts from a global lognormal produces data that
looks plausible in aggregate and falls apart under any per-customer analysis. Here each
customer carries a persistent spending scale, diurnal preference, rail preference, device
history and a preferential-attachment payee book, so their transaction stream is
self-consistent over the whole simulation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config

# Population centres with rough coordinates. India-weighted to match the rail mix, with
# a tail of international cities so cross-border and geo-velocity signals are meaningful.
CITIES: List[tuple] = [
    ("Mumbai", 19.076, 72.877, 0.155, "IN"),
    ("Delhi", 28.614, 77.209, 0.145, "IN"),
    ("Bengaluru", 12.972, 77.594, 0.130, "IN"),
    ("Hyderabad", 17.385, 78.487, 0.085, "IN"),
    ("Chennai", 13.083, 80.271, 0.080, "IN"),
    ("Kolkata", 22.573, 88.364, 0.070, "IN"),
    ("Pune", 18.520, 73.857, 0.065, "IN"),
    ("Ahmedabad", 23.023, 72.571, 0.050, "IN"),
    ("Jaipur", 26.912, 75.787, 0.040, "IN"),
    ("Lucknow", 26.847, 80.947, 0.035, "IN"),
    ("Kochi", 9.932, 76.267, 0.030, "IN"),
    ("Guwahati", 26.145, 91.736, 0.025, "IN"),
    ("London", 51.507, -0.128, 0.030, "GB"),
    ("Frankfurt", 50.110, 8.682, 0.020, "DE"),
    ("New York", 40.713, -74.006, 0.020, "US"),
    ("Singapore", 1.352, 103.820, 0.020, "SG"),
]

# Merchant categories: (label, mcc, median ticket, ticket sigma, risk tier, share)
MERCHANT_CATEGORIES: List[tuple] = [
    ("grocery", 5411, 640.0, 0.75, 0, 0.150),
    ("food_delivery", 5814, 380.0, 0.62, 0, 0.120),
    ("fuel", 5541, 1100.0, 0.55, 0, 0.070),
    ("apparel", 5651, 1450.0, 0.95, 0, 0.075),
    ("electronics", 5732, 6800.0, 1.15, 1, 0.055),
    ("pharmacy", 5912, 520.0, 0.80, 0, 0.060),
    ("utilities", 4900, 1350.0, 0.60, 0, 0.070),
    ("telecom_recharge", 4814, 299.0, 0.45, 0, 0.075),
    ("transport", 4121, 240.0, 0.70, 0, 0.070),
    ("travel_air", 4511, 9800.0, 0.90, 1, 0.030),
    ("hotel", 7011, 4200.0, 0.85, 1, 0.028),
    ("education", 8299, 12500.0, 1.00, 0, 0.025),
    ("healthcare", 8062, 3200.0, 1.10, 0, 0.030),
    ("digital_goods", 5815, 449.0, 0.85, 2, 0.045),
    ("gaming_iap", 7994, 799.0, 1.05, 2, 0.030),
    ("gift_cards", 5947, 2000.0, 0.70, 3, 0.018),
    ("crypto_exchange", 6051, 15000.0, 1.30, 3, 0.014),
    ("investment_platform", 6211, 22000.0, 1.25, 2, 0.016),
    ("marketplace_c2c", 5399, 1800.0, 1.20, 2, 0.019),
]

HIGH_RISK_MCCS = {5947, 6051, 6211, 7994, 5399}

#: Share of customer accounts opened recently enough to look "new" to a risk engine, and
#: share of merchants with a young storefront. Both exist to stop account age from being a
#: free separator; see the tenure and domain-age comments below for why that matters.
RECENT_ACCOUNT_SHARE = 0.14
RECENT_MERCHANT_SHARE = 0.18

PSP_NAMES = [
    "NEOBANK-01", "PSU-BANK-02", "PVT-BANK-03", "PVT-BANK-04", "SFB-05",
    "WALLET-PPI-06", "FINTECH-07", "COOP-BANK-08", "PVT-BANK-09", "NEOBANK-10",
    "EU-PSP-11", "US-PSP-12", "PAYFAC-13", "SFB-14",
]


@dataclass
class Population:
    """Every persistent entity in the simulated ecosystem."""

    customers: pd.DataFrame
    merchants: pd.DataFrame
    devices: pd.DataFrame
    psps: pd.DataFrame
    payee_book: Dict[int, np.ndarray]
    """customer index -> array of merchant indices they habitually pay."""

    peer_book: Dict[int, np.ndarray]
    """customer index -> array of customer indices they habitually pay (P2P)."""

    customer_devices: Dict[int, np.ndarray]

    def summary(self) -> Dict[str, object]:
        return {
            "customers": len(self.customers),
            "corporate_customers": int(self.customers["is_corporate"].sum()),
            "merchants": len(self.merchants),
            "devices": len(self.devices),
            "psps": len(self.psps),
            "mean_payee_book": round(float(np.mean([len(v) for v in self.payee_book.values()])), 2),
            "mean_peer_book": round(float(np.mean([len(v) for v in self.peer_book.values()])), 2),
        }


def build_population(cfg: Config, rng: np.random.Generator) -> Population:
    pop = cfg.population
    n = pop.n_customers

    psps = _build_psps(rng, pop.n_psps)
    customers = _build_customers(cfg, rng, psps)
    merchants = _build_merchants(cfg, rng, pop.n_merchants, psps)
    devices, customer_devices = _build_devices(cfg, rng, customers)
    payee_book = _build_payee_book(rng, customers, merchants)
    peer_book = _build_peer_book(rng, customers)

    return Population(
        customers=customers,
        merchants=merchants,
        devices=devices,
        psps=psps,
        payee_book=payee_book,
        peer_book=peer_book,
        customer_devices=customer_devices,
    )


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------

def _build_psps(rng: np.random.Generator, n_psps: int) -> pd.DataFrame:
    names = PSP_NAMES[:n_psps]
    # Market share follows a power law: a few large PSPs carry most of the volume.
    share = 1.0 / np.power(np.arange(1, len(names) + 1), 0.85)
    share = share / share.sum()
    return pd.DataFrame(
        {
            "psp_id": names,
            "market_share": share,
            # Inbound control maturity drives how attractive a PSP is as a mule host.
            "mule_control_maturity": np.clip(rng.beta(4, 2, len(names)), 0.05, 0.98),
            "country": ["IN"] * len(names),
        }
    ).assign(country=lambda d: np.where(d["psp_id"].str.startswith("EU"), "DE",
                                        np.where(d["psp_id"].str.startswith("US"), "US", "IN")))


def _build_customers(cfg: Config, rng: np.random.Generator, psps: pd.DataFrame) -> pd.DataFrame:
    pop = cfg.population
    n = pop.n_customers

    bands = list(pop.age_bands.keys())
    band_p = np.array(list(pop.age_bands.values()), dtype=float)
    band_p = band_p / band_p.sum()
    band_idx = rng.choice(len(bands), size=n, p=band_p)

    # Digital literacy declines with age band but with wide within-band variance, which is
    # what makes age a weak standalone predictor and forces the model to combine signals.
    literacy_centre = np.array([0.82, 0.86, 0.74, 0.58, 0.40])[band_idx]
    digital_literacy = np.clip(rng.normal(literacy_centre, 0.16), 0.02, 0.99)

    city_idx = rng.choice(len(CITIES), size=n, p=_normalise([c[3] for c in CITIES]))
    lat = np.array([CITIES[i][1] for i in city_idx]) + rng.normal(0, 0.09, n)
    lon = np.array([CITIES[i][2] for i in city_idx]) + rng.normal(0, 0.09, n)
    country = np.array([CITIES[i][4] for i in city_idx])

    is_corporate = (rng.random(n) < pop.corporate_share).astype(int)

    # Per-customer spending scale: lognormal, lifted for corporates.
    spend_scale = np.exp(rng.normal(0.0, 0.55, n)) * np.where(is_corporate == 1, 6.5, 1.0)
    # Per-customer activity multiplier: gamma gives the long tail of very active users.
    activity = rng.gamma(shape=2.4, scale=1 / 2.4, size=n)

    # Account vintage. A gamma alone gives an ecosystem where essentially nobody is new:
    # under 2% of accounts would be younger than a hundred days, so "the beneficiary's
    # account is young" becomes a fraud rule with no false positives, and the detection
    # problem collapses. Real payment ecosystems onboard continuously - new joiners, gig
    # workers, students, migrated customers - and those accounts receive ordinary payments
    # from day one. The recent cohort is what forces the defence to treat account age as
    # weak evidence rather than proof.
    tenure = np.clip(rng.gamma(shape=2.0, scale=520.0, size=n), 3, 6000)
    recent = rng.random(n) < RECENT_ACCOUNT_SHARE
    tenure[recent] = np.clip(rng.gamma(shape=1.4, scale=45.0, size=n)[recent], 1, 400)
    tenure = tenure.astype(int)
    kyc_level = np.where(is_corporate == 1, 3, rng.choice([1, 2, 3], size=n, p=[0.12, 0.53, 0.35]))

    psp_id = rng.choice(psps["psp_id"].to_numpy(), size=n, p=psps["market_share"].to_numpy())

    # Vulnerability to social engineering. Not observable by the model: used only to make
    # victim selection realistic (scams concentrate, they do not spray uniformly).
    vulnerability = np.clip(
        0.55 * (1 - digital_literacy) + 0.25 * (band_idx / 4.0) + rng.normal(0, 0.12, n), 0.01, 0.99
    )

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "age_band": [bands[i] for i in band_idx],
            "age_band_ord": band_idx,
            "digital_literacy": digital_literacy,
            "is_corporate": is_corporate,
            "home_lat": lat,
            "home_lon": lon,
            "home_country": country,
            "psp_id": psp_id,
            "spend_scale": spend_scale,
            "activity": activity,
            "tenure_days": tenure,
            "kyc_level": kyc_level,
            "balance": np.round(np.exp(rng.normal(10.4, 1.05, n)) * spend_scale, 2),
            "night_owl": np.clip(rng.beta(2, 6, n) + 0.35 * (band_idx == 0), 0, 1),
            "vulnerability": vulnerability,
            "prior_fraud_reports": rng.poisson(0.04, n),
            # Propensity to delegate purchases to an AI shopping agent. Concentrated in
            # digitally fluent cohorts, which is exactly where agentic fraud will land.
            "agent_adoption": np.clip(rng.beta(1.6, 7.0, n) * (0.4 + digital_literacy), 0, 1),
            "collector_weight": _collector_weights(rng, n, is_corporate),
        }
    )


#: Share of personal accounts that collect from many unrelated one-off payers. Shopkeepers,
#: tiffin services, tutors, landlords, chit-fund collectors, autorickshaw drivers and gig
#: workers all do this on ordinary consumer accounts.
COLLECTOR_SHARE = 0.14


def _collector_weights(rng: np.random.Generator, n: int, is_corporate: np.ndarray) -> np.ndarray:
    """Relative likelihood of being chosen as a brand-new peer payee.

    Without this, new peer payees are drawn uniformly, so no legitimate account ever
    accumulates inbound fan-in from strangers and *every* high-fan-in account in the dataset
    is a mule. Payee in-degree, distinct-payer counts and payee PageRank then separate fraud
    by construction - they are measuring "was this account minted by the attack generator",
    not "does this account behave like a collection point".

    Real ecosystems are full of legitimate collectors, and they are the reason a bank cannot
    simply block accounts that receive from many strangers. The weights are heavy-tailed so
    that a few accounts take a lot of inbound volume, which is the shape that actually occurs.
    """
    # Many moderate collectors rather than a handful of enormous ones. A very heavy tail
    # concentrates all inbound novelty on a few dozen accounts, which inverts the problem:
    # those accounts dwarf every mule, and "few inbound payments" becomes the fraud signal
    # instead of "many". What the defence has to face is a crowded middle - thousands of
    # ordinary accounts sitting in the same fan-in range a collection mule occupies.
    w = np.full(n, 1.0)
    picked = rng.random(n) < COLLECTOR_SHARE
    w[picked] = 1.0 + rng.gamma(1.6, 6.0, int(picked.sum()))
    # Corporates collect too - subscription businesses, schools, housing societies.
    corp = (is_corporate == 1) & ~picked
    w[corp] = 1.0 + rng.gamma(1.4, 4.0, int(corp.sum()))
    return w


def _build_merchants(cfg: Config, rng: np.random.Generator, n: int, psps: pd.DataFrame) -> pd.DataFrame:
    labels = [c[0] for c in MERCHANT_CATEGORIES]
    shares = _normalise([c[5] for c in MERCHANT_CATEGORIES])
    cat_idx = rng.choice(len(MERCHANT_CATEGORIES), size=n, p=shares)

    mcc = np.array([MERCHANT_CATEGORIES[i][1] for i in cat_idx])
    median_ticket = np.array([MERCHANT_CATEGORIES[i][2] for i in cat_idx])
    ticket_sigma = np.array([MERCHANT_CATEGORIES[i][3] for i in cat_idx])
    risk_tier = np.array([MERCHANT_CATEGORIES[i][4] for i in cat_idx])

    # Merchant popularity is heavy-tailed: a handful of merchants take most transactions.
    popularity = 1.0 / np.power(rng.permutation(np.arange(1, n + 1)), 0.9)
    popularity = popularity / popularity.sum()

    domain_age = np.clip(rng.gamma(2.2, 780.0, n), 20, 9000)
    # Established merchants skew older; risky categories skew newer.
    domain_age = domain_age * np.where(risk_tier >= 2, 0.35, 1.0)
    # New sellers list constantly, and a young storefront that is nonetheless genuine is
    # the single most common reason a counterfeit-merchant rule misfires in production.
    fresh = rng.random(n) < RECENT_MERCHANT_SHARE
    domain_age[fresh] = np.clip(rng.gamma(1.5, 40.0, n)[fresh], 5, 400)

    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:06d}" for i in range(n)],
            "category": [labels[i] for i in cat_idx],
            "mcc": mcc,
            "median_ticket": median_ticket * np.exp(rng.normal(0, 0.25, n)),
            "ticket_sigma": ticket_sigma,
            "risk_tier": risk_tier,
            "popularity": popularity,
            "domain_age_days": domain_age,
            "acquirer_psp": rng.choice(psps["psp_id"].to_numpy(), size=n, p=psps["market_share"].to_numpy()),
            "is_cross_border": (rng.random(n) < 0.06).astype(int),
            # Merchants that publish machine-readable catalogues for AI shopping agents.
            "is_agent_optimised": (rng.random(n) < 0.22).astype(int),
            "is_high_risk_category": np.isin(mcc, list(HIGH_RISK_MCCS)).astype(int),
            "is_counterfeit": np.zeros(n, dtype=int),
        }
    )


def _build_devices(cfg: Config, rng: np.random.Generator, customers: pd.DataFrame):
    n = len(customers)
    # Most people have one primary device; a minority carry two.
    n_devices = np.where(rng.random(n) < 0.28, 2, 1)
    rows = []
    customer_devices: Dict[int, np.ndarray] = {}
    did = 0
    for ci in range(n):
        ids = []
        for _ in range(int(n_devices[ci])):
            rows.append(
                {
                    "device_id": f"D{did:07d}",
                    "customer_idx": ci,
                    # Age at simulation start. Bounded by tenure: you cannot have used a
                    # device with the bank for longer than you have been a customer.
                    "age_days_at_start": float(
                        min(rng.gamma(2.0, 240.0), customers["tenure_days"].iat[ci])
                    ),
                    "os": rng.choice(["android", "ios"], p=[0.78, 0.22]),
                    "is_rooted": int(rng.random() < 0.018),
                }
            )
            ids.append(did)
            did += 1
        customer_devices[ci] = np.array(ids)
    devices = pd.DataFrame(rows)
    return devices, customer_devices


def _build_payee_book(rng: np.random.Generator, customers: pd.DataFrame, merchants: pd.DataFrame):
    """Assign each customer a habitual merchant set via preferential attachment."""
    n = len(customers)
    pop_p = merchants["popularity"].to_numpy()
    pop_p = pop_p / pop_p.sum()
    # Book size: most customers repeatedly use 8-25 merchants.
    sizes = np.clip(rng.negative_binomial(6, 0.30, n), 4, 60)
    m_idx = np.arange(len(merchants))
    book: Dict[int, np.ndarray] = {}
    # Sampling merchant sets one customer at a time is the bottleneck at scale, so draw
    # a single large pool and slice it; the popularity distribution is preserved.
    pool = rng.choice(m_idx, size=int(sizes.sum()), p=pop_p, replace=True)
    cursor = 0
    for ci in range(n):
        k = int(sizes[ci])
        book[ci] = np.unique(pool[cursor : cursor + k])
        cursor += k
    return book


def _build_peer_book(rng: np.random.Generator, customers: pd.DataFrame):
    """Assign each customer a small set of habitual peer-to-peer counterparties."""
    n = len(customers)
    sizes = np.clip(rng.negative_binomial(3, 0.42, n), 1, 22)
    pool = rng.integers(0, n, size=int(sizes.sum()))
    book: Dict[int, np.ndarray] = {}
    cursor = 0
    for ci in range(n):
        k = int(sizes[ci])
        peers = np.unique(pool[cursor : cursor + k])
        book[ci] = peers[peers != ci]
        if book[ci].size == 0:
            book[ci] = np.array([(ci + 1) % n])
        cursor += k
    return book


def _normalise(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr / arr.sum()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorised over numpy arrays."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
