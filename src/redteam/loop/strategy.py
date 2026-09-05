"""Structural genes: the attacker changes plan, not just appearance.

The twelve evasion levers in :mod:`redteam.generate.attacks.model_attacks` all repaint
attributes of a payment that was going to happen anyway - a different amount, a quieter
hour, a coercion flag left unset. They are worth having, but an attacker whose only moves
are cosmetic is a strawman. Real operations respond to a detector by changing the shape of
the operation: moving to a rail with different limits, splitting one demand into six,
buying more drop accounts so no single one accumulates a suspicious fan-in, seasoning those
accounts for a fortnight first, or simply picking different victims.

Those are the genes in this module. Each one is deliberately something a criminal
organisation can actually buy or decide, and each one has a price in
:mod:`redteam.loop.economics`, so the search has to trade them off rather than switch all of
them on. The distinction that matters:

* a **cosmetic** lever changes a column, and the defence can respond by weighting that
  column less;
* a **structural** gene changes which rows exist, who they are between, and when - so it
  moves the derived counterparty, velocity and graph features that the defence leans on
  hardest, and there is no single column to down-weight in response.

``aged_payee_share`` was the only structural gene the loop had, and it was carrying the
entire search on its own. These are its siblings.

Genes that add rows
-------------------
``structuring`` and ``seasoning_depth`` both create transactions that did not exist before.
This is why the functions here return a frame instead of mutating in place: splitting one
₹90,000 demand into six ₹15,000 payments is not a column edit, it is five new rows, and
pretending otherwise would let the search claim a tactic it cannot actually execute. New
rows get fresh identifiers and are inserted into the timeline in the right order, so every
downstream causal window sees them exactly as it would see real traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schema import IRREVOCABLE_RAILS

#: Rails an attacker can realistically move a push-payment campaign onto. Card rails are
#: excluded: they are a different fraud economy with chargeback rights attached, and letting
#: the search move an APP scam onto CARD_CP would be modelling a physical terminal the
#: operator does not have.
SWITCHABLE_RAILS: List[str] = ["UPI_P2P", "IMPS", "NEFT", "RTP_FEDNOW", "WALLET"]

#: Channel is not free to choose either - it has to be one the victim could plausibly be
#: driven to. A branch-assisted payment requires the victim to walk into a branch, which is
#: where scams go to die, so it is not in the attacker's move set.
SWITCHABLE_CHANNELS: List[str] = ["mobile_app", "web", "ivr"]

#: Victim cohorts, expressed as a weighting over the population rather than a hard filter.
#: A campaign that targeted *only* the over-65s would be trivially detectable as a cohort
#: anomaly; real campaigns skew.
COHORTS: List[str] = ["elderly", "affluent", "corporate", "low_literacy", "young_digital"]


@dataclass(frozen=True)
class StrategyGenes:
    """The structural half of a genome.

    Every field is optional and ``None`` means "leave the campaign's own choice alone",
    which keeps a genome that pulls one structural lever from silently rewriting the other
    six to defaults.
    """

    rail: Optional[str] = None
    channel: Optional[str] = None
    cohort: Optional[str] = None
    structuring: int = 1
    """Split each payment into this many smaller ones."""

    spacing_s: Optional[float] = None
    """Seconds between the split payments. Short spacing is a burst the velocity features
    catch; long spacing is slow but gives the victim time to reconsider."""

    n_mules: Optional[int] = None
    """Cap the number of distinct beneficiary accounts the campaign uses."""

    mule_reuse: Optional[float] = None
    """Share of payments routed to already-used accounts rather than fresh ones."""

    payee_psp: Optional[str] = None
    """Concentrate the beneficiary accounts at one payment service provider."""

    seasoning_depth: int = 0
    """Benign inbound payments injected into each drop account before the campaign."""

    def active(self) -> Dict[str, object]:
        """Only the genes that are actually doing something, for labelling and costing."""
        out: Dict[str, object] = {}
        if self.rail:
            out["rail"] = self.rail
        if self.channel:
            out["channel"] = self.channel
        if self.cohort:
            out["cohort"] = self.cohort
        if self.structuring > 1:
            out["structuring"] = self.structuring
        if self.spacing_s is not None and self.structuring > 1:
            out["spacing_s"] = round(self.spacing_s, 1)
        if self.n_mules is not None:
            out["n_mules"] = self.n_mules
        if self.mule_reuse is not None:
            out["mule_reuse"] = round(self.mule_reuse, 2)
        if self.payee_psp:
            out["payee_psp"] = self.payee_psp
        if self.seasoning_depth > 0:
            out["seasoning_depth"] = self.seasoning_depth
        return out

    def label(self) -> str:
        active = self.active()
        return ",".join(f"{k}={v}" for k, v in active.items())

    def describe(self) -> Dict[str, object]:
        return {
            "gene_rail": self.rail or "",
            "gene_channel": self.channel or "",
            "gene_cohort": self.cohort or "",
            "gene_structuring": self.structuring,
            "gene_spacing_s": round(self.spacing_s, 1) if self.spacing_s is not None else "",
            "gene_n_mules": self.n_mules if self.n_mules is not None else "",
            "gene_mule_reuse": round(self.mule_reuse, 3) if self.mule_reuse is not None else "",
            "gene_payee_psp": self.payee_psp or "",
            "gene_seasoning_depth": self.seasoning_depth,
        }


def random_genes(rng: np.random.Generator, psps: Sequence[str],
                 parent: Optional[StrategyGenes] = None,
                 scale: float = 0.35) -> StrategyGenes:
    """Draw a structural genotype, or mutate one gene of an existing one.

    Mutation touches a single gene per step. Structural changes are large - moving rail
    changes the settlement window, the limits and the channel mix all at once - so a genome
    that resampled all nine every generation would be a random restart wearing the costume
    of a hill climb, and the fitness landscape would carry no gradient at all.
    """
    if parent is None:
        # Most genomes should be structurally quiet. A population where every candidate
        # rewires the whole campaign never discovers that one well-chosen change is enough.
        return StrategyGenes(
            rail=str(rng.choice(SWITCHABLE_RAILS)) if rng.random() < 0.30 else None,
            channel=str(rng.choice(SWITCHABLE_CHANNELS)) if rng.random() < 0.20 else None,
            cohort=str(rng.choice(COHORTS)) if rng.random() < 0.25 else None,
            structuring=int(rng.choice([1, 1, 1, 2, 3, 5])),
            spacing_s=float(rng.choice([120.0, 900.0, 3_600.0, 21_600.0])),
            n_mules=int(rng.integers(1, 12)) if rng.random() < 0.30 else None,
            mule_reuse=float(rng.uniform(0.2, 0.95)) if rng.random() < 0.25 else None,
            payee_psp=str(rng.choice(psps)) if psps is not None and len(psps)
            and rng.random() < 0.15 else None,
            seasoning_depth=int(rng.choice([0, 0, 0, 3, 8, 20])),
        )

    gene = str(rng.choice(["rail", "channel", "cohort", "structuring", "spacing",
                           "n_mules", "mule_reuse", "payee_psp", "seasoning"]))
    kwargs = {
        "rail": parent.rail, "channel": parent.channel, "cohort": parent.cohort,
        "structuring": parent.structuring, "spacing_s": parent.spacing_s,
        "n_mules": parent.n_mules, "mule_reuse": parent.mule_reuse,
        "payee_psp": parent.payee_psp, "seasoning_depth": parent.seasoning_depth,
    }
    if gene == "rail":
        kwargs["rail"] = None if parent.rail and rng.random() < 0.3 else str(
            rng.choice(SWITCHABLE_RAILS))
    elif gene == "channel":
        kwargs["channel"] = None if parent.channel and rng.random() < 0.3 else str(
            rng.choice(SWITCHABLE_CHANNELS))
    elif gene == "cohort":
        kwargs["cohort"] = None if parent.cohort and rng.random() < 0.3 else str(
            rng.choice(COHORTS))
    elif gene == "structuring":
        kwargs["structuring"] = int(np.clip(parent.structuring + rng.choice([-2, -1, 1, 2]), 1, 8))
    elif gene == "spacing":
        base = parent.spacing_s if parent.spacing_s is not None else 900.0
        kwargs["spacing_s"] = float(np.clip(base * np.exp(rng.normal(0, scale * 2)), 30, 86_400))
    elif gene == "n_mules":
        base = parent.n_mules if parent.n_mules is not None else 4
        kwargs["n_mules"] = int(np.clip(base + rng.choice([-3, -1, 1, 3]), 1, 40))
    elif gene == "mule_reuse":
        base = parent.mule_reuse if parent.mule_reuse is not None else 0.5
        kwargs["mule_reuse"] = float(np.clip(base + rng.normal(0, scale), 0.0, 1.0))
    elif gene == "payee_psp":
        kwargs["payee_psp"] = None if parent.payee_psp else (
            str(rng.choice(psps)) if psps is not None and len(psps) else None)
    else:
        kwargs["seasoning_depth"] = int(np.clip(
            parent.seasoning_depth + rng.choice([-8, -3, 3, 8]), 0, 40))
    return StrategyGenes(**kwargs)


# --------------------------------------------------------------------------------------
# Applying the genes
# --------------------------------------------------------------------------------------

def apply_strategy(df: pd.DataFrame, idx: np.ndarray, genes: StrategyGenes,
                   rng: np.random.Generator, *,
                   customers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Apply the structural genotype to the campaign rows at ``idx``.

    Returns a frame, because two of the genes create rows. Ordering matters: the genes that
    rewrite *which entities are involved* run before the ones that create rows, so seasoning
    is injected into the accounts the campaign will actually end up using rather than the
    ones it started with.
    """
    if idx.size == 0:
        return df

    if genes.rail:
        _set_rail(df, idx, genes.rail, rng)
    if genes.channel:
        _set_channel(df, idx, genes.channel)
    if genes.cohort and customers is not None:
        _retarget_cohort(df, idx, genes.cohort, customers, rng)
    if genes.n_mules is not None or genes.mule_reuse is not None:
        _reshape_mules(df, idx, genes.n_mules, genes.mule_reuse, rng)
    if genes.payee_psp:
        _concentrate_psp(df, idx, genes.payee_psp)

    # Row-creating genes last, and seasoning before structuring so the seasoning count is
    # per drop account rather than per split payment.
    if genes.seasoning_depth > 0:
        df = _season_accounts(df, idx, genes.seasoning_depth, rng)
        idx = _reindex(df, idx)
    if genes.structuring > 1:
        df = _structure_payments(df, idx, genes.structuring,
                                 genes.spacing_s if genes.spacing_s is not None else 900.0,
                                 rng)
    return df


def _reindex(df: pd.DataFrame, previous: np.ndarray) -> np.ndarray:
    """Recover positional indices after a gene inserted rows and reset the index."""
    marker = df["_loop_target"].to_numpy() if "_loop_target" in df.columns else None
    if marker is None:
        return previous
    return np.where(marker == 1)[0]


def _set_rail(df: pd.DataFrame, idx: np.ndarray, rail: str,
              rng: np.random.Generator) -> None:
    """Move the campaign onto a different rail, with its currency and revocability.

    The channel is left alone unless it is impossible on the new rail. Rail and channel are
    correlated in real traffic but not determined by each other, and forcing a redraw here
    would make every rail-switching genome also a channel-switching one, so the search could
    never tell which of the two did the work.
    """
    n = idx.size
    values = df["rail"].to_numpy().astype(object).copy()
    values[idx] = rail
    df["rail"] = values

    irrevocable = df["is_irrevocable_rail"].to_numpy().copy()
    irrevocable[idx] = int(rail in IRREVOCABLE_RAILS)
    df["is_irrevocable_rail"] = irrevocable

    currency = df["currency"].to_numpy().astype(object).copy()
    currency[idx] = {"RTP_FEDNOW": "USD", "SEPA_INST": "EUR"}.get(rail, "INR")
    df["currency"] = currency

    # A point-of-sale channel cannot carry a person-to-person transfer.
    channels = df["channel"].to_numpy().astype(object).copy()
    impossible = np.isin(channels[idx], ["pos", "recurring"])
    if impossible.any():
        channels[idx[impossible]] = rng.choice(
            np.array(["mobile_app", "web"], dtype=object), size=int(impossible.sum()))
        df["channel"] = channels


def _set_channel(df: pd.DataFrame, idx: np.ndarray, channel: str) -> None:
    values = df["channel"].to_numpy().astype(object).copy()
    values[idx] = channel
    df["channel"] = values


#: Weighting functions per cohort, over the population table. Each returns a positive
#: weight per customer; the campaign's victims are then resampled from that distribution.
_COHORT_WEIGHTS = {
    "elderly": lambda c: 0.05 + (c["age_band_ord"].to_numpy() >= 3) * 1.0,
    "affluent": lambda c: 0.05 + (c["balance"].to_numpy()
                                  > np.quantile(c["balance"].to_numpy(), 0.8)) * 1.0,
    "corporate": lambda c: 0.02 + (c["is_corporate"].to_numpy() == 1) * 1.0,
    "low_literacy": lambda c: 0.05 + (c["digital_literacy"].to_numpy() < 0.45) * 1.0,
    "young_digital": lambda c: 0.05 + ((c["age_band_ord"].to_numpy() <= 1)
                                       & (c["digital_literacy"].to_numpy() > 0.7)) * 1.0,
}


def _retarget_cohort(df: pd.DataFrame, idx: np.ndarray, cohort: str,
                     customers: pd.DataFrame, rng: np.random.Generator) -> None:
    """Point the campaign at a different kind of victim.

    Every customer-derived column on the row is rewritten from the population table, because
    a campaign that changed who it targeted but kept the old victims' account tenure and
    balance ratios would be handing the defence a contradiction that does not exist in real
    data - and the model would learn to detect the contradiction rather than the attack.

    Victims are chosen per *episode*, not per payment: a scam targets a person, and that
    person makes several payments. Resampling row by row would scatter one campaign across
    hundreds of unrelated accounts and destroy the payer-velocity signal that makes the
    episode structure realistic in the first place.
    """
    weight = _COHORT_WEIGHTS.get(cohort)
    if weight is None:
        return
    w = np.maximum(weight(customers).astype(float), 1e-9)
    w = w / w.sum()

    campaigns = df["campaign_id"].to_numpy()[idx]
    unique_campaigns, inverse = np.unique(campaigns, return_inverse=True)
    chosen = rng.choice(len(customers), size=len(unique_campaigns), p=w, replace=True)
    victims = chosen[inverse]

    cust = customers.iloc[victims]
    account_ids = np.array([f"A{int(v):07d}" for v in victims], dtype=object)

    assignments = {
        "_customer_idx": victims,
        "customer_id": cust["customer_id"].to_numpy(),
        "payer_account_id": account_ids,
        "customer_tenure_days": cust["tenure_days"].to_numpy(),
        "customer_age_band_ord": cust["age_band_ord"].to_numpy(),
        "customer_digital_literacy": np.round(cust["digital_literacy"].to_numpy(), 4),
        "customer_is_corporate": cust["is_corporate"].to_numpy(),
        "kyc_level": cust["kyc_level"].to_numpy(),
        "prior_fraud_reports": cust["prior_fraud_reports"].to_numpy(),
        "payer_psp": cust["psp_id"].to_numpy(),
        "_lat": cust["home_lat"].to_numpy(),
        "_lon": cust["home_lon"].to_numpy(),
    }
    for column, values in assignments.items():
        if column not in df.columns:
            continue
        current = df[column].to_numpy().copy()
        if current.dtype.kind in "iu" and np.asarray(values).dtype.kind == "f":
            current = current.astype(float)
        current[idx] = values
        df[column] = current

    # The share of balance moving has to follow the new victim's balance or an affluent
    # cohort would inherit a student's drain ratio.
    if "account_balance_ratio" in df.columns:
        balance = np.maximum(cust["balance"].to_numpy(), 100.0)
        amount = df["amount_inr"].to_numpy()[idx] if "amount_inr" in df.columns \
            else df["amount"].to_numpy()[idx]
        ratio = df["account_balance_ratio"].to_numpy().astype(float).copy()
        ratio[idx] = np.round(np.clip(amount.astype(float) / balance, 0, 50), 5)
        df["account_balance_ratio"] = ratio


def _reshape_mules(df: pd.DataFrame, idx: np.ndarray, n_mules: Optional[int],
                   reuse: Optional[float], rng: np.random.Generator) -> None:
    """Change how many drop accounts the campaign collects into, and how hard it reuses them.

    This is the gene that trades two detection surfaces against each other, which is why it
    belongs in the search rather than in a constant. Few accounts means high fan-in per
    account, and fan-in acceleration is one of the strongest payee features the defence has.
    Many accounts means every payment lands on a counterparty with no history, and
    counterparty novelty is the other one. There is no setting that avoids both, and finding
    where the trough sits is a genuinely non-obvious result.
    """
    payees = df["payee_account_id"].to_numpy().astype(object).copy()
    current = pd.unique(payees[idx])
    if current.size == 0:
        return

    target = int(np.clip(n_mules, 1, max(idx.size, 1))) if n_mules is not None else current.size
    if target < current.size:
        # Consolidate onto a subset of the accounts already in use. Buying fewer accounts is
        # the cheap direction, so no new identities are needed.
        accounts = rng.choice(current, size=target, replace=False)
    elif target > current.size:
        # Expanding needs accounts that do not exist yet. They are minted here rather than
        # drawn from the population because a fresh drop account is exactly that: an account
        # with no history, opened for this campaign.
        extra = np.array([f"MULE-LOOP-{rng.integers(0, 2**40):011x}"
                          for _ in range(target - current.size)], dtype=object)
        accounts = np.concatenate([current, extra])
    else:
        accounts = current

    if reuse is None:
        assigned = rng.choice(accounts, size=idx.size)
    else:
        # A high reuse rate concentrates the campaign on the first few accounts; a low one
        # spreads it evenly. Modelled as a power-law preference over the account list.
        concentration = np.power(np.arange(1, accounts.size + 1, dtype=float),
                                 -3.0 * float(reuse))
        assigned = rng.choice(accounts, size=idx.size, p=concentration / concentration.sum())

    payees[idx] = assigned
    df["payee_account_id"] = payees


def _concentrate_psp(df: pd.DataFrame, idx: np.ndarray, psp: str) -> None:
    values = df["payee_psp"].to_numpy().astype(object).copy()
    values[idx] = psp
    df["payee_psp"] = values


def _season_accounts(df: pd.DataFrame, idx: np.ndarray, depth: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Give each drop account a benign inbound history before the campaign uses it.

    Seasoning is the counter to every "this account is new and has never received anything"
    feature at once, and unlike buying an aged account it can be done with accounts the
    operator already controls. The cost is time and float: the money has to come from
    somewhere and it sits idle until the campaign runs.

    The injected payments are labelled ``is_fraud = 0`` deliberately. They are not fraud -
    that is the entire point of them - and labelling them as fraud would hand the defence a
    training signal that seasoning traffic is detectable, which would be the simulator
    marking its own homework.
    """
    if depth <= 0:
        return df

    targets = pd.unique(df["payee_account_id"].to_numpy()[idx])
    if targets.size == 0:
        return df

    # First fraudulent use per account: the seasoning has to land before it, or it is not
    # seasoning, it is just more traffic.
    ts = pd.DatetimeIndex(df["timestamp"])
    first_use = (
        pd.Series(ts[idx].to_numpy(), index=df["payee_account_id"].to_numpy()[idx])
        .groupby(level=0).min()
    )
    horizon = pd.Timestamp(ts.min())

    # Seasoning payers are ordinary accounts drawn from the legitimate traffic, which is what
    # a rented-account or money-mule-recruitment scheme produces in practice.
    legit = df[df["is_fraud"] == 0]
    if legit.empty:
        return df
    donor_positions = rng.integers(0, len(legit), size=targets.size * depth)
    donors = legit.iloc[donor_positions].copy().reset_index(drop=True)

    account_column = np.repeat(targets, depth)
    starts = first_use.reindex(account_column).to_numpy()
    # Spread the seasoning back over the weeks before first use, but never before the
    # simulation begins - a payment outside the observation window is not observable.
    lead_s = rng.uniform(2 * 86_400, 45 * 86_400, size=account_column.size)
    new_ts = pd.DatetimeIndex(starts) - pd.to_timedelta(lead_s.astype("int64"), unit="s")
    new_ts = pd.DatetimeIndex(np.maximum(new_ts.values, horizon.to_datetime64()))

    donors["timestamp"] = new_ts
    donors["payee_account_id"] = account_column
    donors["txn_id"] = [f"TSEED{rng.integers(0, 2**44):012x}" for _ in range(len(donors))]
    # Modest, unremarkable amounts. Seasoning with large sums would draw exactly the
    # attention it is meant to avoid, and would cost the operator far more float.
    amounts = np.round(np.exp(rng.normal(np.log(1_800.0), 0.6, len(donors))), 2)
    donors["amount"] = amounts
    if "amount_inr" in donors.columns:
        donors["amount_inr"] = amounts
    for column, value in (("is_fraud", 0), ("fraud_type", "none"), ("attack_vector_id", ""),
                          ("campaign_id", ""), ("mule_ring_id", ""), ("is_hard_negative", 0),
                          ("evasion_applied", 0)):
        if column in donors.columns:
            donors[column] = value
    if "_loop_target" in donors.columns:
        donors["_loop_target"] = 0
    if "hour" in donors.columns:
        donors["hour"] = new_ts.hour
    if "day_of_week" in donors.columns:
        donors["day_of_week"] = new_ts.dayofweek
    if "is_night" in donors.columns:
        donors["is_night"] = ((new_ts.hour < 6) | (new_ts.hour >= 23)).astype(int)

    return pd.concat([df, donors], ignore_index=True)


def _structure_payments(df: pd.DataFrame, idx: np.ndarray, k: int, spacing_s: float,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Split each targeted payment into ``k`` smaller ones spaced ``spacing_s`` apart.

    Smurfing, and the oldest structural evasion there is. It attacks amount-based features
    directly: every resulting payment is well under whatever threshold the original tripped,
    and each one on its own is unremarkable. What it cannot hide is the payer's cumulative
    exposure over the session, which is exactly why ``f_payer_amount_sum_15m`` exists - so
    this gene should be *findable*, and if the search discovers it beats the detector then
    the detector has a real gap.

    The split is uneven. Six identical payments are a signature in themselves; a human being
    coached through a sequence of transfers produces amounts that vary.
    """
    if k <= 1 or idx.size == 0:
        return df

    original = df.iloc[idx]
    amounts = original["amount"].to_numpy().astype(float)
    base_ts = pd.DatetimeIndex(original["timestamp"])

    # Dirichlet shares: sum to one exactly, so the campaign moves the same money in total.
    shares = rng.dirichlet(np.full(k, 6.0), size=idx.size)

    clones: List[pd.DataFrame] = []
    for part in range(k):
        piece = original.copy().reset_index(drop=True)
        split_amount = np.round(amounts * shares[:, part], 2)
        piece["amount"] = split_amount
        if "amount_inr" in piece.columns:
            # Preserve whatever FX ratio the row already carried rather than assuming parity.
            ratio = np.where(amounts > 0,
                             original["amount_inr"].to_numpy().astype(float) / np.maximum(amounts, 1e-9),
                             1.0)
            piece["amount_inr"] = np.round(split_amount * ratio, 2)
        if part > 0:
            offset = spacing_s * part * rng.uniform(0.7, 1.3, idx.size)
            shifted = base_ts + pd.to_timedelta(offset.astype("int64"), unit="s")
            piece["timestamp"] = shifted
            piece["txn_id"] = [f"TSPLIT{rng.integers(0, 2**44):012x}" for _ in range(idx.size)]
            if "hour" in piece.columns:
                piece["hour"] = shifted.hour
            if "day_of_week" in piece.columns:
                piece["day_of_week"] = shifted.dayofweek
            if "is_night" in piece.columns:
                piece["is_night"] = ((shifted.hour < 6) | (shifted.hour >= 23)).astype(int)
        if "account_balance_ratio" in piece.columns:
            scale = np.where(amounts > 0, split_amount / np.maximum(amounts, 1e-9), 1.0)
            piece["account_balance_ratio"] = np.round(
                original["account_balance_ratio"].to_numpy().astype(float) * scale, 5)
        clones.append(piece)

    keep = df.drop(df.index[idx])
    return pd.concat([keep] + clones, ignore_index=True)
