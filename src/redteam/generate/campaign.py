"""Red-team campaign orchestrator.

Turns an attack mix into a labelled dataset:

1. build the benign stream,
2. run every simulated vector until its share of the fraud budget is met,
3. run the mule family last, so it can launder proceeds that earlier families deposited,
4. apply evasion tuning to a configurable share of all fraud,
5. merge, enrich counterparty intelligence, and hand back one event stream.

Step 3 is not cosmetic. Running the mule generator after the predicate offences is what
connects the laundering subgraph to the scams that fed it, which is the difference between
graph features that mean something and graph features that describe the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..identify.library import AttackLibrary, AttackVector
from ..schema import DEFAULTS, IRREVOCABLE_RAILS, all_columns
from .agentic_corpus import build_corpus
from .attacks import apply_evasion, build_context, generator_for
from .attacks.base import AttackContext
from .benign import simulate_benign
from .enrich import enrich
from .entities import Population, build_population
from .transcripts import build_transcripts

#: Families run last because they depend on state earlier families create.
DEFERRED_GENERATORS = ("mule",)

#: Defaults for the generator's internal (underscore-prefixed) scratch columns.
INTERNAL_DEFAULTS = {
    "_customer_idx": -1,
    "_merchant_idx": -1,
    "_peer_idx": -1,
    "_payee_reg_lag_s": 3600.0,
    "_is_explore_payee": 0,
    "_lat": 0.0,
    "_lon": 0.0,
    "_agent_payload": "",
}


@dataclass
class GenerationResult:
    transactions: pd.DataFrame
    population: Population
    context: AttackContext
    per_vector: pd.DataFrame
    corpus: pd.DataFrame
    transcripts: pd.DataFrame = field(default_factory=pd.DataFrame)


    @property
    def fraud_rate(self) -> float:
        return float(self.transactions["is_fraud"].mean())


def generate_dataset(cfg: Config, library: AttackLibrary,
                     rng: Optional[np.random.Generator] = None,
                     mix_override: Optional[Dict[str, float]] = None,
                     payload_bank=None, transcript_bank=None) -> GenerationResult:
    rng = rng or np.random.default_rng(cfg.seed)

    population = build_population(cfg, rng)
    benign = simulate_benign(cfg, population, rng)
    ctx = build_context(cfg, population, benign, library, rng)

    mix = library.resolve_mix(mix_override or cfg.attacks.mix or None)
    n_benign = len(benign)
    # target_fraud_rate is a share of the *combined* stream.
    rate = float(np.clip(cfg.attacks.target_fraud_rate, 1e-5, 0.4))
    fraud_budget = int(round(n_benign * rate / (1.0 - rate)))

    ordered = _order_vectors(library, mix)
    frames: List[pd.DataFrame] = []
    stats: List[Dict[str, object]] = []

    for vector in ordered:
        target = int(round(mix[vector.id] * fraud_budget))
        if target <= 0:
            continue
        df = _generate_to_target(ctx, vector, target)
        if df.empty:
            continue
        frames.append(df)
        stats.append(
            {
                "attack_vector_id": vector.id,
                "family": vector.family,
                "generator": vector.generator,
                "target_fraud_rows": target,
                "fraud_rows": int((df["is_fraud"] == 1).sum()),
                "tagged_benign_rows": int((df["is_fraud"] == 0).sum()),
                "episodes": int(df["campaign_id"].nunique()),
                "median_amount": float(df.loc[df["is_fraud"] == 1, "amount"].median()) if (df["is_fraud"] == 1).any() else 0.0,
            }
        )

    attacks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=benign.columns)
    if not attacks.empty:
        apply_evasion(rng, attacks, share=cfg.attacks.evasion_share, strength=0.6)

    seasoning = _season_mule_accounts(ctx, benign, attacks, rng)
    combined = _merge(benign, attacks if seasoning.empty
                      else pd.concat([attacks, seasoning], ignore_index=True))
    combined = enrich(combined)
    combined = _finalise_ids(combined)

    # The agent's context window is the second data plane; it has to be built after ids are
    # final so every bundle keys back to a settled transaction.
    corpus = build_corpus(combined, rng, payload_bank=payload_bank)
    # The third plane: what was said to the victim. Also built after ids are final, and for
    # the same reason - a transcript that keys to a transaction that was later renumbered
    # would silently detach from the payment it explains.
    transcripts = build_transcripts(combined, rng, library=library, bank=transcript_bank)

    per_vector = pd.DataFrame(stats)
    return GenerationResult(transactions=combined, population=population, context=ctx,
                            per_vector=per_vector, corpus=corpus, transcripts=transcripts)


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

#: Fraction of collection accounts that get seasoned before use. Not every ring bothers -
#: the impatient ones are the easy catches.
SEASONED_MULE_SHARE = 0.62

#: Seasoning credits per account per day of lead time, and the number of days a ring is
#: willing to wait. Expressed as a rate rather than a fixed count because the alternative
#: silently changes meaning with the simulation window: over a long run legitimate payees
#: keep accumulating history while a fixed burst does not, and the gap reopens the very
#: separability the seasoning exists to close.
SEASONING_PER_DAY = 0.55
SEASONING_LEAD_DAYS = 21.0


def _season_mule_accounts(ctx, benign: pd.DataFrame, attacks: pd.DataFrame,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Give collection accounts an ordinary inbound history before the fraud lands.

    Without this the beneficiary of a scam is an account the network has never seen, and
    "payee has no history" separates fraud from legitimate traffic almost perfectly. That is
    an artefact of generating mules on demand, not a property of fraud. Real rings season
    accounts precisely because novelty rules exist: small credits from unrelated parties,
    spread over days, so that by the time the account collects a scam payment it has a
    plausible age, a nonzero inbound count and several distinct counterparties.

    The seasoning payments are labelled legitimate because they are - no victim is defrauded
    by them. They are the ring's cost of doing business, and they are what forces the
    defence to reason about a *change* in an account's inbound pattern rather than about the
    absence of one.
    """
    if attacks.empty or "payee_account_id" not in attacks.columns:
        return pd.DataFrame(columns=benign.columns)

    fraud = attacks[attacks["is_fraud"] == 1]
    if fraud.empty:
        return pd.DataFrame(columns=benign.columns)

    first_use = fraud.groupby("payee_account_id")["timestamp"].min()
    # Recruited mules are real customers and already carry their own benign history. Ask the
    # registry which accounts were purpose-opened rather than inferring it from the id: mule
    # ids deliberately share the customer namespace, precisely so that nothing downstream can
    # tell the two apart by looking at a string.
    opened = ctx.mules.purpose_opened()
    synthetic = first_use[[a in opened for a in first_use.index]]
    if synthetic.empty:
        return pd.DataFrame(columns=benign.columns)

    chosen = synthetic[rng.random(len(synthetic)) < SEASONED_MULE_SHARE]
    if chosen.empty:
        return pd.DataFrame(columns=benign.columns)

    # Each ring picks how long it is prepared to wait; patient rings look the most ordinary.
    lead_days = rng.uniform(0.25, 1.0, len(chosen)) * SEASONING_LEAD_DAYS
    counts = rng.poisson(lead_days * SEASONING_PER_DAY) + 1
    total = int(counts.sum())
    payees = np.repeat(chosen.index.to_numpy(), counts)
    first_ts = np.repeat(chosen.to_numpy(), counts)

    # Credits land before first use, never after it: a seasoning payment that arrived later
    # would be leakage from the campaign into its own cover.
    lead_s = rng.uniform(0.02, 1.0, total) * np.repeat(lead_days, counts) * 86_400.0
    ts = pd.to_datetime(first_ts) - pd.to_timedelta(lead_s, unit="s")
    window_start = benign["timestamp"].min()
    keep = ts >= window_start
    if not keep.any():
        return pd.DataFrame(columns=benign.columns)

    # Index positionally from the same Generator rather than handing a draw to pandas' legacy
    # RandomState. Both are reproducible, but the project documents one Generator and one seed
    # as its determinism contract, and a second RNG stream is how that quietly stops being true.
    picks = rng.integers(0, len(benign), int(keep.sum()))
    df = benign.iloc[picks].reset_index(drop=True).copy()
    df["timestamp"] = pd.Series(ts[keep]).reset_index(drop=True)
    df["payee_account_id"] = payees[keep]
    df["merchant_id"] = ""
    df["mcc"] = 0
    df["rail"] = rng.choice(["UPI_P2P", "IMPS", "NEFT"], size=len(df), p=[0.62, 0.26, 0.12])
    df["channel"] = "mobile_app"
    df["is_irrevocable_rail"] = np.isin(df["rail"].to_numpy(), list(IRREVOCABLE_RAILS)).astype(int)
    df["currency"] = "INR"
    # Small and unremarkable: the point of seasoning is to be uninteresting.
    df["amount"] = np.round(np.exp(rng.normal(np.log(900), 0.8, len(df))), 2)
    df["payee_psp"] = [ctx.mules.psp_of.get(a, p) for a, p in
                       zip(df["payee_account_id"], df["payee_psp"])]
    df["payee_account_age_days"] = np.round(
        [max(ctx.mules.opened_days_ago.get(a, 30.0) - l / 86_400.0, 0.5)
         for a, l in zip(df["payee_account_id"], lead_s[keep])], 1)
    df["_payee_reg_lag_s"] = np.round(np.clip(rng.lognormal(6.2, 1.4, len(df)), 20, 3 * 86_400), 1)
    df["is_fraud"] = 0
    df["fraud_type"] = "none"
    df["attack_vector_id"] = ""
    df["campaign_id"] = ""
    return df


def _order_vectors(library: AttackLibrary, mix: Dict[str, float]) -> List[AttackVector]:
    vectors = [library.get(vid) for vid in mix]
    primary = [v for v in vectors if v.generator not in DEFERRED_GENERATORS]
    deferred = [v for v in vectors if v.generator in DEFERRED_GENERATORS]
    return primary + deferred


def _generate_to_target(ctx: AttackContext, vector: AttackVector, target_rows: int,
                        max_rounds: int = 8) -> pd.DataFrame:
    """Generate whole episodes until the vector's fraud-row budget is met.

    Episodes are never split: a three-payment grooming sequence with its middle payment
    removed is not a smaller version of the attack, it is a different and less realistic
    one. The budget is therefore met to within one episode.
    """
    generator = generator_for(vector)
    frames: List[pd.DataFrame] = []
    produced = 0
    rows_per_event = 2.0

    for _ in range(max_rounds):
        if produced >= target_rows:
            break
        need = target_rows - produced
        ctx.budget_hint = need
        n_events = int(max(1, np.ceil(need / max(rows_per_event, 0.25))))
        n_events = int(min(n_events, 20_000))
        df = generator.generate(ctx, vector, n_events)
        if df is None or df.empty:
            break
        fraud_rows = int((df["is_fraud"] == 1).sum())
        rows_per_event = max(fraud_rows / n_events, 0.25)
        frames.append(df)
        produced += fraud_rows

    ctx.budget_hint = 10 ** 9
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return _trim_to_budget(out, target_rows)


def _trim_to_budget(df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    """Drop whole trailing episodes once the budget is exceeded."""
    fraud_per_campaign = df.groupby("campaign_id", sort=False)["is_fraud"].sum()
    if fraud_per_campaign.empty:
        # A generator can legitimately return only non-fraud rows - the bust-out cultivation
        # path does exactly that - and indexing position zero to find the first episode's size
        # then raised IndexError and took the whole run down.
        return df.reset_index(drop=True)
    running = fraud_per_campaign.cumsum()
    keep = running[running <= max(target_rows, int(fraud_per_campaign.iloc[0]))].index
    if len(keep) == 0:
        keep = fraud_per_campaign.index[:1]
    return df[df["campaign_id"].isin(set(keep))].reset_index(drop=True)


def _merge(benign: pd.DataFrame, attacks: pd.DataFrame) -> pd.DataFrame:
    if attacks.empty:
        return benign.copy()

    def _default(col):
        return DEFAULTS.get(col, INTERNAL_DEFAULTS.get(col, 0))

    # Copy before filling. Aligning the two schemas in place mutated the caller's frames, so
    # a second call - or a caller that reused the benign frame - saw columns it never created.
    benign, attacks = benign.copy(), attacks.copy()
    for col in benign.columns:
        if col not in attacks.columns:
            attacks[col] = _default(col)
    for col in attacks.columns:
        if col not in benign.columns:
            benign[col] = _default(col)
    return pd.concat([benign, attacks[benign.columns]], ignore_index=True)


def _finalise_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign transaction ids in time order and drop the generator's scratch columns."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["txn_id"] = [f"T{i:09d}" for i in range(len(df))]
    keep = [c for c in all_columns() if c in df.columns]
    internal = [c for c in df.columns if c.startswith("_")]
    return df[keep + internal]
