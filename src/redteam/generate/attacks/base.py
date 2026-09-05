"""Shared machinery for attack generators.

Two ideas carry most of the fidelity here.

**Attacks are mutations of plausible payments.** Rather than synthesising a fraudulent
transaction from scratch (which produces a row that is unrealistic in a hundred fields
nobody thought about), every generator starts from a real benign transaction belonging to
a plausibly-selected victim and overwrites only the fields the attack actually touches.
Everything else — the victim's typing rhythm, their device history, their home city, their
spending scale — stays internally consistent.

**Signals are emitted probabilistically.** A generator that always sets
``screen_share_active = 1`` for the safe-account scam creates a single-feature giveaway and
a meaningless 0.999 AUC. ``Config.attacks.signal_emission`` controls how reliably each tell
actually fires, so the defence has to combine weak evidence the way a real system does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...config import Config
from ...identify.library import AttackLibrary, AttackVector
from .. import timing
from ..entities import Population

MULE_PSP_BIAS = 2.5
"""How strongly mule accounts concentrate at PSPs with weak inbound controls."""

#: Attacker-controlled accounts are numbered above the real population so that they share the
#: customer namespace without colliding with it. Two independent counters mint into that
#: space - purpose-opened mules and synthetic identities - so each gets a reserved block.
#: Sharing a base would let one account id mean two different things in the same run.
MULE_ID_BLOCK = 0
SYNTHETIC_ID_BLOCK = 5_000_000

#: Share of fraudulent payments whose beneficiary the victim has paid before. Repeat invoice
#: fraud, later top-ups in an investment scam, and the second and third payment of a drain all
#: reuse an account already on file. Setting this to zero - which is what "always mark the
#: payee as new" amounts to - hands the defence most of its recall for free.
KNOWN_PAYEE_SHARE = 0.32


# --------------------------------------------------------------------------------------
# Mule account registry
# --------------------------------------------------------------------------------------

@dataclass
class MuleRegistry:
    """Shared pool of receiving accounts used by every proceeds-taking attack.

    Keeping one registry across attack families is what makes the transaction graph
    realistic: an APP scam, a purchase scam and a QR-tampering campaign can all deposit
    into the same first-hop mule, which then fans out. That shared structure is the
    signal the graph features are designed to find, and it only exists if the generators
    share state.
    """

    accounts: List[str] = field(default_factory=list)
    ring_of: Dict[str, str] = field(default_factory=dict)
    opened_days_ago: Dict[str, float] = field(default_factory=dict)
    psp_of: Dict[str, str] = field(default_factory=dict)
    recruited: set = field(default_factory=set)
    """Real customer accounts turned into mules by recruitment or account renting.

    Modelling these is not a detail. If every receiving account is freshly minted, then
    "payee has no history" separates fraud from legitimate traffic almost perfectly and the
    whole detection problem collapses into one feature. Recruited accounts are genuine
    accounts with genuine tenure and genuine prior volume, which is both how a large share
    of real mule capacity is sourced and what forces the defence to reason about *change*
    in an account's behaviour rather than its age.
    """

    _seq: int = 0
    _ring_seq: int = 0

    def purpose_opened(self) -> set:
        """Accounts opened to be mules, as opposed to real customers who were recruited.

        The distinction drives seasoning: a recruited account already has a genuine history,
        a purpose-opened one has to be given a plausible cover story.
        """
        return set(self.opened_days_ago) - self.recruited

    def _psp_weights(self, pop: Population) -> np.ndarray:
        maturity = pop.psps["mule_control_maturity"].to_numpy()
        w = np.exp(-MULE_PSP_BIAS * maturity)
        return w / w.sum()

    def new_ring(self, pop: Population, rng: np.random.Generator, size: int) -> List[str]:
        """Provision a ring of accounts, but expose only the first as a collection point.

        Registering every ring member in the drawable pool spreads incoming payments one
        per account, so no mule ever accumulates the inbound fan-in that makes it
        detectable. Real rings have a small number of first-hop collection accounts feeding
        a much larger set of downstream hops, and the layering generator reaches the rest
        through ``ring_of``.
        """
        self._ring_seq += 1
        ring_id = f"RING{self._ring_seq:04d}"
        psps = pop.psps["psp_id"].to_numpy()
        p = self._psp_weights(pop)
        # Purpose-opened mules are minted into the *customer* namespace, above the real
        # population, rather than into a "MUL" range of their own. A separate prefix makes
        # every mule account identifiable from its id alone: the string is fraud with
        # probability 1, and every entity-keyed feature and graph node inherits that
        # separation. Which accounts are mules is tracked here, in the registry, where it
        # belongs - not encoded in a value the data carries around.
        base = len(pop.customers) + MULE_ID_BLOCK
        out = []
        for i in range(size):
            acct = f"A{base + self._seq:07d}"
            self._seq += 1
            if i == 0:
                self.accounts.append(acct)
            self.ring_of[acct] = ring_id
            # Mule accounts are young. Rented/recruited accounts are the older tail.
            self.opened_days_ago[acct] = float(
                rng.choice([rng.uniform(2, 45), rng.uniform(120, 900)], p=[0.78, 0.22])
            )
            self.psp_of[acct] = str(rng.choice(psps, p=p))
            out.append(acct)
        return out

    def recruit(self, pop: Population, rng: np.random.Generator) -> str:
        """Turn an existing customer account into a mule."""
        cust = pop.customers
        # Recruitment targets skew young, lower-balance and financially stretched.
        band = cust["age_band_ord"].to_numpy()
        w = np.where(band <= 1, 3.0, 1.0) / np.maximum(cust["balance"].to_numpy(), 1.0) ** 0.25
        ci = int(rng.choice(len(cust), p=w / w.sum()))
        acct = f"A{ci:07d}"
        if acct not in self.opened_days_ago:
            self._ring_seq += 1
            self.accounts.append(acct)
            self.recruited.add(acct)
            self.ring_of[acct] = f"RECRUIT{self._ring_seq:04d}"
            self.opened_days_ago[acct] = float(cust["tenure_days"].iat[ci])
            self.psp_of[acct] = str(cust["psp_id"].iat[ci])
        return acct

    def draw(self, pop: Population, rng: np.random.Generator, n: int, reuse: float = 0.62,
             recruit_share: float = 0.78) -> np.ndarray:
        """Draw n receiving accounts, reusing existing mules where possible.

        Most mule capacity is recruited rather than manufactured. Opening accounts at scale
        under modern KYC is expensive, while students, job-scam victims and willing renters
        are cheap and come with the one thing a fresh account cannot have: a genuine history.
        Weighting the draw the other way is not only less accurate, it hands the defence a
        free win - every beneficiary would be an account the network had never seen, and
        payee novelty alone would carry most of the detection.
        """
        picked: List[str] = []
        for _ in range(n):
            if self.accounts and rng.random() < reuse:
                picked.append(str(rng.choice(self.accounts)))
            elif rng.random() < recruit_share:
                picked.append(self.recruit(pop, rng))
            else:
                ring = self.new_ring(pop, rng, size=int(rng.integers(4, 26)))
                picked.append(ring[0])
        return np.array(picked, dtype=object)


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------

@dataclass
class CounterfeitRegistry:
    """Fake merchants: scam storefronts, front companies and agent-optimised decoys."""

    merchants: Dict[str, Dict[str, object]] = field(default_factory=dict)
    _seq: int = 0

    def draw(self, ctx: "AttackContext", n: int, *, mccs: Sequence[int],
             domain_age=(1.0, 90.0), agent_optimised_p: float = 0.2,
             reuse: float = 0.55) -> List[str]:
        picked: List[str] = []
        keys = list(self.merchants)
        for _ in range(n):
            if keys and ctx.rng.random() < reuse:
                picked.append(str(ctx.rng.choice(keys)))
                continue
            self._seq += 1
            # Counterfeit storefronts are minted into the ordinary merchant namespace, above
            # the real population. A dedicated "MX" prefix made every scam merchant - and so
            # every payment to one - identifiable from the id alone.
            mid = f"M{len(ctx.pop.merchants) + self._seq:06d}"
            self.merchants[mid] = {
                "mcc": int(ctx.rng.choice(list(mccs))),
                "domain_age_days": float(ctx.rng.uniform(*domain_age)),
                "is_agent_optimised": int(ctx.rng.random() < agent_optimised_p),
                "acquirer_psp": str(ctx.rng.choice(ctx.pop.psps["psp_id"].to_numpy())),
            }
            keys.append(mid)
            picked.append(mid)
        return picked


@dataclass
class AttackContext:
    cfg: Config
    pop: Population
    rng: np.random.Generator
    benign: pd.DataFrame
    library: AttackLibrary
    mules: MuleRegistry
    rows_by_customer: Dict[int, np.ndarray]
    counterfeits: CounterfeitRegistry = field(default_factory=CounterfeitRegistry)
    budget_hint: int = 10 ** 9
    """Fraud rows still wanted for the current vector.

    Burst-shaped attacks (card testing, QR clusters, mule fan-out) produce tens to
    hundreds of rows from a single episode. Without this hint a small run overshoots its
    fraud budget several-fold and the resulting prevalence stops matching the config.
    """

    _campaign_seq: int = 0
    _identity_seq: int = 0
    _device_seq: int = 0

    @property
    def emission(self) -> float:
        return float(self.cfg.attacks.signal_emission)

    def cap(self, n: int, minimum: int = 3) -> int:
        """Clamp a burst size to what the remaining fraud budget can absorb."""
        return int(max(minimum, min(n, self.budget_hint)))

    def campaign_id(self, vector_id: str) -> str:
        self._campaign_seq += 1
        return f"{vector_id}#{self._campaign_seq:05d}"

    def identity_seq(self) -> int:
        """Next synthetic-identity number.

        Counters live on the context rather than at module scope so that two runs in one
        process produce identical data. A module-level counter would carry over between
        them and silently break reproducibility for everything downstream of an id.
        """
        self._identity_seq += 1
        return self._identity_seq

    def episode_start(self, span_s: float = 0.0, night_bias: float = 0.0) -> float:
        """Seconds from the window start to an episode's first payment.

        Attack episodes are placed using the *benign* day and hour distributions rather than
        a flat draw over the window. A flat draw leaves day-of-week and hour-of-day carrying
        free signal: the defence separates fraud by noticing it is spread evenly across a
        week that real traffic is not spread evenly across. ``night_bias`` then applies the
        genuine attacker preference for the small hours on top, which keeps it as the weak
        evidence it is rather than the artefact it was.
        """
        return timing.episode_seconds(
            self.rng, self.cfg.benign.n_days, span_s,
            night_bias=night_bias, start_date=self.cfg.benign.start_date,
        )

    def device_ids(self, n: int, n_devices: int) -> np.ndarray:
        """Mint ``n`` attacker device ids drawn from a fresh pool of ``n_devices``.

        The counter is monotonic across the whole run for the same reason ``identity_seq``
        is: per-frame numbering restarts at zero, so unrelated takeovers in different
        vectors end up sharing a device id. Every device-keyed feature then sees one handset
        used by hundreds of unrelated victims - a phantom device farm spanning the dataset,
        which is both wrong and a strong fraud signal the defence gets for free.
        """
        n_devices = max(1, min(n_devices, n))
        pool = np.array([f"D{self._device_seq + i:08d}" for i in range(n_devices)], dtype=object)
        self._device_seq += n_devices
        return pool[np.arange(n) % n_devices]

    def tell(self, n: int, base_p: float = 1.0) -> np.ndarray:
        """Boolean mask deciding whether a given tell-tale signal fires."""
        return self.rng.random(n) < min(1.0, base_p * self.emission)


def build_context(cfg: Config, pop: Population, benign: pd.DataFrame,
                  library: AttackLibrary, rng: np.random.Generator) -> AttackContext:
    order = np.argsort(benign["_customer_idx"].to_numpy(), kind="stable")
    sorted_idx = benign["_customer_idx"].to_numpy()[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(len(pop.customers) + 1))
    rows_by_customer = {
        ci: order[boundaries[ci]: boundaries[ci + 1]] for ci in range(len(pop.customers))
    }
    return AttackContext(
        cfg=cfg, pop=pop, rng=rng, benign=benign, library=library,
        mules=MuleRegistry(), rows_by_customer=rows_by_customer,
    )


# --------------------------------------------------------------------------------------
# Victim selection and template sampling
# --------------------------------------------------------------------------------------

def select_victims(ctx: AttackContext, n: int, *, by_vulnerability: float = 1.0,
                   corporate: Optional[bool] = None, agentic: bool = False,
                   min_age_band: Optional[int] = None,
                   max_age_band: Optional[int] = None) -> np.ndarray:
    """Choose victims non-uniformly. Real scams concentrate; uniform sampling does not."""
    cust = ctx.pop.customers
    w = np.ones(len(cust))

    if by_vulnerability:
        w *= np.power(cust["vulnerability"].to_numpy(), 1.6 * by_vulnerability)
    if corporate is True:
        w *= np.where(cust["is_corporate"].to_numpy() == 1, 40.0, 0.02)
    elif corporate is False:
        w *= np.where(cust["is_corporate"].to_numpy() == 1, 0.05, 1.0)
    if agentic:
        w *= 0.02 + cust["agent_adoption"].to_numpy()
    band = cust["age_band_ord"].to_numpy()
    if min_age_band is not None:
        w *= np.where(band >= min_age_band, 1.0, 0.05)
    if max_age_band is not None:
        w *= np.where(band <= max_age_band, 1.0, 0.05)

    w = np.maximum(w, 1e-9)
    return ctx.rng.choice(len(cust), size=n, p=w / w.sum(), replace=True)


def template_rows(ctx: AttackContext, victims: np.ndarray) -> pd.DataFrame:
    """One plausible benign transaction per victim, used as the mutation base."""
    picks = np.empty(len(victims), dtype=int)
    for k, ci in enumerate(victims):
        rows = ctx.rows_by_customer.get(int(ci))
        if rows is None or rows.size == 0:  # pragma: no cover - population guarantees >=1
            picks[k] = int(ctx.rng.integers(0, len(ctx.benign)))
        else:
            picks[k] = int(rows[ctx.rng.integers(0, rows.size)])
    out = ctx.benign.iloc[picks].copy().reset_index(drop=True)
    out["_customer_idx"] = victims
    return out


def reset_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["is_hard_negative"] = 0
    df["evasion_applied"] = 0
    df["mule_ring_id"] = ""
    return df


def set_rail(ctx: AttackContext, df: pd.DataFrame, allowed: Sequence[str],
             *, keep_channel: bool = False) -> None:
    """Force the transaction onto a rail the vector actually uses.

    The channel is redrawn to match, because rail and channel are not independent in a real
    payment system: UPI person-to-person does not happen at a point-of-sale terminal, and a
    card-present payment does not happen in a banking app. Leaving the template's channel in
    place produces combinations that occur only under attack, and a rail/channel pair that
    legitimate traffic never occupies is a perfect detector of nothing but the simulator.

    Vectors whose channel *is* the attack - a vishing call driving an IVR payment, an agent
    settling over an API - pass ``keep_channel`` and set it themselves.
    """
    from ...schema import IRREVOCABLE_RAILS
    from ..benign import RAIL_CHANNEL, _draw_from_map

    n = len(df)
    rail = ctx.rng.choice(np.array(list(allowed), dtype=object), size=n)
    df["rail"] = rail
    df["is_irrevocable_rail"] = np.isin(rail, list(IRREVOCABLE_RAILS)).astype(int)
    currency = np.where(rail == "RTP_FEDNOW", "USD", np.where(rail == "SEPA_INST", "EUR", "INR"))
    df["currency"] = currency
    if not keep_channel:
        df["channel"] = _draw_from_map(ctx.rng, rail, RAIL_CHANNEL)


def retime(ctx: AttackContext, df: pd.DataFrame, *, night_bias: float = 0.0,
           jitter_days: float = 1.0, salary_bias: float = 0.0) -> None:
    """Move the template timestamp and refresh every derived time field."""
    n = len(df)
    ts = pd.DatetimeIndex(df["timestamp"])
    shift_s = ctx.rng.normal(0, jitter_days * 86400, n).astype("int64")
    ts = ts + pd.to_timedelta(shift_s, unit="s")

    start = pd.Timestamp(ctx.cfg.benign.start_date)
    end = start + pd.Timedelta(days=ctx.cfg.benign.n_days) - pd.Timedelta(seconds=1)
    ts = pd.DatetimeIndex(np.clip(ts.values, start.to_datetime64(), end.to_datetime64()))

    if salary_bias > 0:
        # Some campaigns are scheduled around payday, because a demand for money converts far
        # better when the money is actually in the account. Benign traffic already peaks in
        # the same window, so this makes the two *more* confusable rather than less - the
        # attack is hiding inside the busiest days of the month, not standing apart from them.
        move = ctx.rng.random(n) < salary_bias
        k = int(move.sum())
        if k:
            values = ts.values.copy()
            moved = pd.DatetimeIndex(ts[move])
            target_day = ctx.rng.choice(np.array([1, 2, 3, 28, 29, 30]), size=k)
            shifted = moved + pd.to_timedelta(
                (target_day - moved.day.to_numpy()).astype("int64"), unit="D"
            )
            values[move] = np.clip(shifted.values, start.to_datetime64(), end.to_datetime64())
            ts = pd.DatetimeIndex(values)

    if night_bias > 0:
        move = ctx.rng.random(n) < night_bias
        k = int(move.sum())
        if k:
            # Land on the same night window legitimate night-owl traffic occupies, rather than
            # on a hand-picked set of whole hours. Choosing from {0,1,2,3,4,23} gave the small
            # hours a shape no real traffic has, so the hour column separated fraud on the
            # distribution's edges instead of on the attacker's genuine preference for night.
            hours = timing.draw_hours(ctx.rng, k, night_bias=1.0)
            shifted = ts[move].normalize() + pd.to_timedelta(
                (hours * 3600).astype("int64"), unit="s"
            )
            values = ts.values.copy()
            values[move] = shifted.values
            ts = pd.DatetimeIndex(values)

    df["timestamp"] = ts
    df["hour"] = ts.hour
    df["day_of_week"] = ts.dayofweek
    df["is_night"] = ((ts.hour < 6) | (ts.hour >= 23)).astype(int)
    df["is_salary_window"] = timing.is_salary_window(ts.day.to_numpy()).astype(int)


def to_mule(ctx: AttackContext, df: pd.DataFrame, *, reuse: float = 0.62,
            recruit_share: float = 0.78, name_match_beta=(3.0, 3.0)) -> np.ndarray:
    """Point the payment at a mule account and set the counterparty fields accordingly."""
    n = len(df)
    accounts = ctx.mules.draw(ctx.pop, ctx.rng, n, reuse=reuse, recruit_share=recruit_share)
    df["payee_account_id"] = accounts
    df["merchant_id"] = ""
    df["_merchant_idx"] = -1
    df["_peer_idx"] = -1
    df["payee_account_age_days"] = np.round(
        [ctx.mules.opened_days_ago[a] for a in accounts], 1
    )
    df["payee_psp"] = [ctx.mules.psp_of[a] for a in accounts]
    df["mule_ring_id"] = [ctx.mules.ring_of[a] for a in accounts]
    df["merchant_domain_age_days"] = 3650.0
    df["mcc"] = 0
    # Confirmation-of-Payee against a mule usually mismatches, but attackers work hard on
    # this: recruited accounts carry a real name and clear the check outright.
    is_recruited = np.array([a in ctx.mules.recruited for a in accounts])
    match = np.clip(ctx.rng.beta(*name_match_beta, size=n), 0, 1)
    match[is_recruited] = np.clip(ctx.rng.beta(8.0, 1.5, int(is_recruited.sum())), 0, 1)
    df["payee_name_match_score"] = np.round(match, 4)

    # Beneficiary added minutes before the payment: the scam's most durable tell - but only
    # for the *first* payment to that beneficiary. Marking every fraudulent payment as
    # first-time-payee is what made payee novelty a near-perfect discriminator, and it is not
    # what real scams look like. Repeat invoice fraud bills the same account monthly; the
    # second and third pig-butchering top-up go where the first one went; the later payments
    # of a safe-account drain reuse the beneficiary the victim was walked through adding an
    # hour ago. In all of those the account is already on file, and the defence has to find
    # the fraud without the novelty flag.
    known = ctx.rng.random(n) < KNOWN_PAYEE_SHARE
    lag = np.clip(ctx.rng.lognormal(4.6, 0.9, n), 15, 7200)
    aged = np.clip(ctx.rng.lognormal(11.6, 1.4, n), 2 * 86_400, 400 * 86_400)
    df["_payee_reg_lag_s"] = np.round(np.where(known, aged, lag), 1)
    df["_is_explore_payee"] = (~known).astype(int)
    return accounts


def to_real_merchant(ctx: AttackContext, df: pd.DataFrame, *, mccs: Sequence[int],
                     n_distinct: Optional[int] = None) -> None:
    """Route the payment to a genuine, established merchant in a given category.

    Plenty of fraud settles at entirely legitimate businesses: card testing probes real
    low-friction merchants, a takeover buys real electronics, a groomed child buys real
    in-game currency. Sending every fraudulent payment to a freshly minted counterparty
    would make payee history a near-perfect discriminator and quietly delete the hardest
    part of the problem.
    """
    n = len(df)
    merch = ctx.pop.merchants
    pool = np.where(np.isin(merch["mcc"].to_numpy(), list(mccs)))[0]
    if pool.size == 0:
        pool = np.arange(len(merch))
    weights = merch["popularity"].to_numpy()[pool]
    weights = weights / weights.sum()
    if n_distinct is None:
        picks = ctx.rng.choice(pool, size=n, p=weights)
    else:
        # Concentrate the whole burst on a handful of merchants, which is what makes
        # merchant-side velocity the detectable signal rather than payer-side velocity.
        chosen = ctx.rng.choice(pool, size=max(1, min(n_distinct, pool.size)), p=weights,
                                replace=False)
        picks = ctx.rng.choice(chosen, size=n)

    df["_merchant_idx"] = picks
    df["_peer_idx"] = -1
    df["payee_account_id"] = merch["merchant_id"].to_numpy()[picks]
    df["merchant_id"] = merch["merchant_id"].to_numpy()[picks]
    df["mcc"] = merch["mcc"].to_numpy()[picks]
    df["payee_psp"] = merch["acquirer_psp"].to_numpy()[picks]
    df["merchant_domain_age_days"] = merch["domain_age_days"].to_numpy()[picks].round(1)
    df["payee_account_age_days"] = df["merchant_domain_age_days"].to_numpy()
    df["payee_is_high_risk_category"] = merch["is_high_risk_category"].to_numpy()[picks]
    df["merchant_is_agent_optimised"] = merch["is_agent_optimised"].to_numpy()[picks]
    df["payee_name_match_score"] = np.round(np.clip(ctx.rng.beta(9.0, 1.1, n), 0, 1), 4)
    df["mule_ring_id"] = ""
    df["_payee_reg_lag_s"] = np.round(np.clip(ctx.rng.lognormal(6.4, 1.5, n), 20, 3 * 86400), 1)


def to_counterfeit_merchant(ctx: AttackContext, df: pd.DataFrame, *, mccs: Sequence[int],
                            domain_age=(1.0, 90.0), agent_optimised_p: float = 0.2,
                            reuse: float = 0.55) -> List[str]:
    """Point the payment at a generated storefront rather than a real merchant."""
    n = len(df)
    ids = ctx.counterfeits.draw(ctx, n, mccs=mccs, domain_age=domain_age,
                                agent_optimised_p=agent_optimised_p, reuse=reuse)
    info = ctx.counterfeits.merchants
    df["payee_account_id"] = ids
    df["merchant_id"] = ids
    df["_merchant_idx"] = -1
    df["_peer_idx"] = -1
    df["mcc"] = [info[m]["mcc"] for m in ids]
    df["merchant_domain_age_days"] = np.round([info[m]["domain_age_days"] for m in ids], 1)
    df["payee_account_age_days"] = df["merchant_domain_age_days"].to_numpy()
    df["payee_psp"] = [info[m]["acquirer_psp"] for m in ids]
    df["merchant_is_agent_optimised"] = [info[m]["is_agent_optimised"] for m in ids]
    df["payee_is_high_risk_category"] = 1
    df["payee_name_match_score"] = np.round(np.clip(ctx.rng.beta(6.0, 2.0, n), 0, 1), 4)
    df["_payee_reg_lag_s"] = np.round(np.clip(ctx.rng.lognormal(6.0, 1.1, n), 20, 86400), 1)
    df["_is_explore_payee"] = 1
    return ids


def apply_signals(ctx: AttackContext, df: pd.DataFrame,
                  signals: Dict[str, "SignalSpec"]) -> None:
    """Apply a declarative bundle of tell-tale signals with probabilistic emission."""
    n = len(df)
    for column, spec in signals.items():
        mask = ctx.tell(n, spec.probability)
        if not mask.any():
            continue
        values = spec.draw(ctx, int(mask.sum()))
        col = df[column].to_numpy().copy()
        if col.dtype.kind in "iu" and np.asarray(values).dtype.kind == "f":
            col = col.astype(float)
        col[mask] = values
        df[column] = col


@dataclass
class SignalSpec:
    """A signal an attack emits, and how often it actually fires."""

    value: object
    probability: float = 1.0

    def draw(self, ctx: AttackContext, k: int):
        if callable(self.value):
            return self.value(ctx.rng, k)
        return np.full(k, self.value)


def const(value, probability: float = 1.0) -> SignalSpec:
    return SignalSpec(value=value, probability=probability)


def dist(fn: Callable[[np.random.Generator, int], np.ndarray], probability: float = 1.0) -> SignalSpec:
    return SignalSpec(value=fn, probability=probability)


def finalise(df: pd.DataFrame, vector: AttackVector, campaign_id: str,
             fraud_type: str) -> pd.DataFrame:
    df["is_fraud"] = 1
    df["fraud_type"] = fraud_type
    df["attack_vector_id"] = vector.id
    df["campaign_id"] = campaign_id
    df["is_hard_negative"] = 0
    return df


def scale_amount(ctx: AttackContext, df: pd.DataFrame, low: float, high: float,
                 *, round_bias: float = 0.45, floor: float = 50.0,
                 cap: Optional[float] = None) -> None:
    """Rescale the template amount into the range this attack actually extracts."""
    n = len(df)
    mult = np.exp(ctx.rng.uniform(np.log(low), np.log(high), n))
    amount = df["amount"].to_numpy() * mult
    # Socially-engineered amounts are round far more often than organic spending: the
    # attacker names a number out loud.
    snap = ctx.rng.random(n) < round_bias
    step = np.select([amount < 5_000, amount < 50_000, amount < 500_000],
                     [500.0, 1_000.0, 5_000.0], default=25_000.0)
    amount[snap] = np.maximum(np.round(amount[snap] / step[snap]) * step[snap], step[snap])
    amount = np.maximum(amount, floor)
    if cap is not None:
        amount = np.minimum(amount, cap)
    df["amount"] = np.round(amount, 2)
    balance = ctx.pop.customers["balance"].to_numpy()[df["_customer_idx"].to_numpy()]
    df["account_balance_ratio"] = np.round(np.clip(df["amount"].to_numpy() / np.maximum(balance, 100.0), 0, 50), 5)


class AttackGenerator:
    """Base class. Subclasses declare which vector ids they can produce."""

    name: str = "base"
    vector_ids: Sequence[str] = ()

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        raise NotImplementedError
