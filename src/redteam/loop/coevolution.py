"""The closed loop: two economically motivated adversaries, taking turns.

What a round is
---------------
1. **Blue freezes.** The deployed detector and its threshold are held fixed.
2. **Red steals a copy.** The attacker probes the frozen model with traffic it controls,
   records approve/decline, and fits a surrogate (:mod:`redteam.loop.surrogate`). Everything
   after this point is the attacker optimising against their own copy first and spending
   real attempts only on what already looks promising - which is both how adaptive fraud
   actually works and what makes a twelve-round search affordable.
3. **Red targets weakness.** Vectors are sampled in proportion to how badly they already
   evade, so search effort goes where the defence is thin.
4. **Red mutates.** A genome has a cosmetic half (twelve evasion levers, a strength and a
   coverage share) and a structural half (rail, channel, victim cohort, structuring depth
   and spacing, mule count and reuse, PSP concentration, account seasoning, aged-account
   purchase). The structural genes are the ones that matter: they change which rows exist
   and who they are between, so they move the derived counterparty and velocity features
   the defence leans on hardest.
5. **Red is scored on profit, not evasion.** Fitness is what the tactic nets after the cost
   of the accounts, proxies and operator time it needed, plus a novelty bonus. This is the
   single most important design decision in the module and the reason for
   :mod:`redteam.loop.economics`: a search rewarded for evasion alone converges on tiny
   harmless payments, declares victory, and teaches nobody anything.
6. **Blue answers, and picks the cheapest answer.** Not just a retrain - hold, re-threshold,
   add a rule, add friction, or retrain, each priced against fraud losses, review cost and
   customer friction (:mod:`redteam.loop.blue`).

What comes out
--------------
Per round: how much recall the attacker took, how much the defender got back, what the
attacker's return on spend was, and which response blue chose and why. Across rounds: a
transfer matrix showing whether round 2's tactics still work against round 9's detector, a
convergence verdict, a ranked list of library controls the search walked through, and a
``discovered_vectors.yaml`` of surviving tactics validated against the taxonomy's own schema
gate. Those last two are the return path that makes this a loop rather than a pipeline.

A defence that recovers fully every round is memorising; one that never recovers is broken.
The interesting result is in between, and it is reported as-is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..defend.dataset import EXTRA_INPUTS, temporal_split
from ..defend.model import FraudDetector
from ..generate.attacks.model_attacks import EVASION_MOVES, apply_evasion
from ..identify.library import AttackLibrary
from .blue import BlueMove, choose_move
from .discovery import (
    DiscoveredVector,
    control_gap_summary,
    discovered_vector,
    transfer_matrix,
    validate_and_emit,
)
from .economics import AttackerCosts, DefenderCosts, score_campaign
from .incremental import RebuildCache, dirty_entities, full_rebuild, rebuild_neighbourhood
from .strategy import StrategyGenes, apply_strategy, random_genes
from .surrogate import SurrogateOracle

MOVE_NAMES = [name for name, _ in EVASION_MOVES]

#: Marker column used to follow the targeted rows through genes that insert or delete rows.
#: Underscore-prefixed, so :func:`redteam.defend.dataset.input_columns` drops it and the
#: model can never see which rows the red team touched.
TARGET_MARKER = "_loop_target"


@dataclass(frozen=True)
class AttackGenome:
    """One red-team tactic: what to change, how hard, on how much of the campaign.

    The genome is deliberately in two halves. ``moves`` repaint attributes of payments that
    were going to happen anyway; ``genes`` change the plan. Only the second half can reach
    the counterparty-novelty and velocity features that carry most of the defence's signal,
    which is why a loop with only the first half spent four rounds discovering that its
    attacker could not do very much.
    """

    vector_id: str
    moves: Tuple[str, ...]
    strength: float
    share: float
    aged_payee_share: float = 0.0
    genes: StrategyGenes = field(default_factory=StrategyGenes)

    def levers(self) -> Tuple[str, ...]:
        """Every lever the tactic pulls, cosmetic and structural, for costing and control
        gap analysis."""
        structural = tuple(self.genes.active().keys())
        aged = ("aged_payee_share",) if self.aged_payee_share > 0.01 else ()
        return tuple(self.moves) + structural + aged

    def label(self) -> str:
        parts = ["+".join(sorted(self.moves))] if self.moves else []
        if self.aged_payee_share > 0.01:
            parts.append(f"aged{self.aged_payee_share:.2f}")
        structural = self.genes.label()
        if structural:
            parts.append(structural)
        return f"{self.vector_id}[{' '.join(parts)}]@{self.strength:.2f}"

    def describe(self) -> Dict[str, object]:
        return {
            "vector_id": self.vector_id,
            "moves": ", ".join(sorted(self.moves)),
            "n_moves": len(self.moves),
            "strength": round(self.strength, 3),
            "share": round(self.share, 3),
            "aged_payee_share": round(self.aged_payee_share, 3),
            **self.genes.describe(),
        }


@dataclass
class RoundResult:
    """One round, measured on the same held-out window three times over.

    The targeted figures repeat every measurement over only the vectors the surviving
    tactics actually touched. An attacker who breaks one vector barely moves an average
    taken over fifty of them, and that average is what a careless report would quote.
    """

    round_index: int
    rows_targeted: int
    eligible_vectors: int
    searched_vectors: int
    baseline_recall: float
    baseline_recall_targeted: float
    post_attack_recall: float
    post_attack_recall_targeted: float
    post_retrain_recall: float
    post_retrain_recall_targeted: float
    post_retrain_fpr: float
    attacker_value_extracted: float = 0.0
    attacker_cost: float = 0.0
    attacker_roi: float = float("nan")
    blue_move: str = "retrain"
    blue_move_detail: str = ""
    blue_cost: float = float("nan")
    surrogate_agreement: float = float("nan")
    survivors: List[AttackGenome] = field(default_factory=list)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    blue_options: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def recall_lost_to_red(self) -> float:
        return self.baseline_recall_targeted - self.post_attack_recall_targeted

    @property
    def recall_recovered_by_blue(self) -> float:
        return self.post_retrain_recall_targeted - self.post_attack_recall_targeted

    def to_dict(self) -> Dict[str, object]:
        return {
            "round": self.round_index,
            "vectors_eligible_to_search": self.eligible_vectors,
            "vectors_searched": self.searched_vectors,
            "targeted_fraud_rows": self.rows_targeted,
            "recall_overall_before": round(self.baseline_recall, 4),
            "recall_overall_under_attack": round(self.post_attack_recall, 4),
            "recall_overall_after_retrain": round(self.post_retrain_recall, 4),
            "recall_targeted_before": round(self.baseline_recall_targeted, 4),
            "recall_targeted_under_attack": round(self.post_attack_recall_targeted, 4),
            "recall_targeted_after_retrain": round(self.post_retrain_recall_targeted, 4),
            "recall_lost_to_red": round(self.recall_lost_to_red, 4),
            "recall_recovered_by_blue": round(self.recall_recovered_by_blue, 4),
            "false_positive_rate_after_retrain": round(self.post_retrain_fpr, 5),
            "attacker_value_extracted_inr": round(self.attacker_value_extracted, 2),
            "attacker_cost_inr": round(self.attacker_cost, 2),
            "attacker_roi": round(self.attacker_roi, 4),
            "blue_move": self.blue_move,
            "blue_move_detail": self.blue_move_detail,
            "blue_total_cost_inr": round(self.blue_cost, 2),
            "surrogate_agreement": round(self.surrogate_agreement, 4),
            "surviving_tactics": "; ".join(g.label() for g in self.survivors),
        }


@dataclass
class LoopResult:
    rounds: List[RoundResult]
    history: pd.DataFrame
    final_detector: FraudDetector
    tactics: pd.DataFrame
    transfer: pd.DataFrame = field(default_factory=pd.DataFrame)
    control_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    blue_options: pd.DataFrame = field(default_factory=pd.DataFrame)
    discovered: List[DiscoveredVector] = field(default_factory=list)
    discovered_yaml: str = ""
    discovery_rejections: List[str] = field(default_factory=list)
    convergence: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.rounds])


# --------------------------------------------------------------------------------------
# Red team: the genome
# --------------------------------------------------------------------------------------

def _random_genome(rng: np.random.Generator, vector_id: str,
                   psps: Sequence[str],
                   parent: Optional[AttackGenome] = None,
                   mutation_scale: float = 0.35) -> AttackGenome:
    if parent is None:
        k = int(rng.integers(2, 6))
        # Coerced to plain str: numpy hands back `np.str_`, which compares equal to a string
        # but is not one, and `yaml.safe_dump` refuses to serialise it when a surviving
        # tactic is written back into the taxonomy.
        moves = tuple(str(m) for m in rng.choice(MOVE_NAMES, size=k, replace=False))
        return AttackGenome(
            vector_id, moves, float(rng.uniform(0.35, 0.9)), float(rng.uniform(0.5, 1.0)),
            float(rng.choice([0.0, 0.0, 0.3, 0.6, 0.9])),
            random_genes(rng, psps),
        )

    # Mutate either the cosmetic half or the structural half, not both. Structural changes
    # are large, and a step that moved a lever and rewired the campaign at the same time
    # would leave the search unable to attribute the fitness change to either.
    if rng.random() < 0.45:
        return replace(parent, genes=random_genes(rng, psps, parent.genes, mutation_scale))

    moves = set(parent.moves)
    roll = rng.random()
    if roll < 0.4 and len(moves) < len(MOVE_NAMES):
        moves.add(str(rng.choice([m for m in MOVE_NAMES if m not in moves])))
    elif roll < 0.7 and len(moves) > 2:
        moves.discard(str(rng.choice(sorted(moves))))
    else:
        if len(moves) > 1:
            moves.discard(str(rng.choice(sorted(moves))))
        available = [m for m in MOVE_NAMES if m not in moves]
        if available:
            moves.add(str(rng.choice(available)))

    return replace(
        parent,
        moves=tuple(sorted(moves)),
        strength=float(np.clip(parent.strength + rng.normal(0, mutation_scale * 0.4), 0.2, 0.98)),
        share=float(np.clip(parent.share + rng.normal(0, mutation_scale * 0.3), 0.3, 1.0)),
        aged_payee_share=float(np.clip(
            parent.aged_payee_share + rng.normal(0, mutation_scale * 0.5), 0.0, 1.0)),
    )


def _novelty(genome: AttackGenome, incumbents: Sequence[AttackGenome]) -> float:
    """Distance from the current survivors, over both halves of the genome.

    Without this the search collapses onto whichever lever worked first and stops exploring,
    which would make the loop a hill climb dressed up as co-evolution.
    """
    if not incumbents:
        return 1.0
    mine = set(genome.levers())
    distances = []
    for other in incumbents:
        theirs = set(other.levers())
        union = len(mine | theirs)
        jaccard = 1.0 - (len(mine & theirs) / union if union else 0.0)
        structural = abs(genome.aged_payee_share - other.aged_payee_share)
        distances.append(0.7 * jaccard + 0.3 * structural)
    return float(np.mean(distances))


def _target_vectors(per_vector: pd.DataFrame, campaign_rows: pd.Series,
                    rng: np.random.Generator, k: int, min_rows: int) -> List[str]:
    """Sample vectors to attack, weighted toward the ones already getting through.

    Eligibility comes from the size of the campaign in the full stream rather than from the
    evaluation table, which only counts the held-out window. A vector with eight rows in the
    test slice can still have sixty over the timeline, and the timeline is what fitness is
    measured on - filtering on the test count would discard targets the search can score.
    """
    eligible_ids = campaign_rows[campaign_rows >= min_rows].index
    if len(eligible_ids) == 0:
        return []
    recall = (per_vector.set_index("attack_vector_id")["recall"]
              if not per_vector.empty else pd.Series(dtype=float))
    # A vector absent from the evaluation table was never scored, so treat it as an unknown
    # rather than as a success and let the search look at it.
    scores = recall.reindex(eligible_ids).fillna(0.5).to_numpy()
    weights = (1.05 - scores) ** 2
    weights = weights / weights.sum()
    size = min(k, len(eligible_ids))
    picks = rng.choice(eligible_ids.to_numpy(), size=size, replace=False, p=weights)
    return [str(p) for p in picks]


# --------------------------------------------------------------------------------------
# Applying a tactic to the stream
# --------------------------------------------------------------------------------------

def aged_account_pool(raw: pd.DataFrame, min_inbound: int = 10) -> np.ndarray:
    """Accounts with a real inbound history, as a mule broker's aged-account inventory.

    Drawn from accounts that only ever receive legitimate traffic, which is what a
    compromised or rented long-standing account looks like on the day it turns.
    """
    legit = raw[raw["is_fraud"] == 0]
    counts = legit["payee_account_id"].value_counts()
    eligible = counts[counts >= min_inbound].index.to_numpy()
    fraud_payees = set(raw.loc[raw["is_fraud"] == 1, "payee_account_id"].unique())
    return np.array([a for a in eligible if a not in fraud_payees], dtype=object)


#: Column stamping each row with the campaign whose realisation created it, or NA for rows
#: the generator produced. Only rows the loop minted carry a value.
LOOP_ORIGIN = "_loop_origin"


def reset_vector(frame: pd.DataFrame, pristine: pd.DataFrame, vector_id: str) -> pd.DataFrame:
    """Undo any previous realisation of one campaign, returning it to how it was generated.

    Committing a tactic has to *replace* how a campaign runs, not layer on top of how it ran
    last round. Layering is what the loop used to do, and with a row-creating gene it
    compounds: ``structuring=4`` splits each payment into four, and the next round splits
    those four into sixteen. Over a full-profile run the fraud population went 1,978 → 3,022
    → 7,198 → 23,902 → 90,718 across four commits, quadrupling every round.

    That is not only a runtime problem, though it is that - every phase of a round scaled with
    it, and round two took five times round one. It destroys the experiment. The book starts
    at a realistic 0.70% fraud and is 23% fraud by the fourth round, so precision, the
    false-positive rate, the defender's cost model and the attacker's economics are all being
    computed on a portfolio that no longer resembles a payment portfolio. Every loop number
    after about round two was measuring that drift.

    Rows the loop minted are dropped, and rows it edited in place are restored from the
    pristine stream, since ``structuring`` rewrites the original payment's amount as well as
    adding siblings.
    """
    original = pristine[(pristine["attack_vector_id"] == vector_id)
                        & (pristine["is_fraud"] == 1)]
    if original.empty and LOOP_ORIGIN not in frame.columns:
        return frame

    keep = frame
    if LOOP_ORIGIN in frame.columns:
        keep = keep[keep[LOOP_ORIGIN].ne(vector_id) | keep[LOOP_ORIGIN].isna()]
    if not original.empty:
        keep = keep[~keep["txn_id"].isin(set(original["txn_id"].to_numpy().tolist()))]
        keep = pd.concat([keep, original], ignore_index=True, sort=False)
    return keep.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


@dataclass
class Applied:
    """The result of running one genome over the stream."""

    frame: pd.DataFrame
    target_index: np.ndarray
    changed_txn_ids: Set[str]
    n_aged_accounts: int
    n_distinct_payees: int
    seasoning_payments: int


def apply_genome(raw: pd.DataFrame, genome: AttackGenome, rng: np.random.Generator,
                 aged_pool: Optional[np.ndarray] = None,
                 customers: Optional[pd.DataFrame] = None) -> Applied:
    """Re-run one campaign with the mutated settings, structural genes first.

    Mutations are applied across the whole timeline, not only the test window. That is what
    makes blue's answer meaningful: the defender's training data contains the new tactic too,
    so a recovery measures genuine adaptation rather than the tactic simply being absent from
    the training window.
    """
    out = raw.copy()
    marker = ((out["attack_vector_id"].to_numpy() == genome.vector_id)
              & (out["is_fraud"].to_numpy() == 1)).astype(int)
    out[TARGET_MARKER] = marker
    idx = np.where(marker == 1)[0]
    if idx.size == 0:
        return Applied(out, idx, set(), 0, 0, 0)

    before_ids = set(out["txn_id"].to_numpy()[idx].tolist())

    out = apply_strategy(out, idx, genome.genes, rng, customers=customers)
    idx = np.where(out[TARGET_MARKER].to_numpy() == 1)[0]

    # Record which campaign minted each new row, so a later realisation of the same campaign
    # can take them back out again. Without this, ``structuring`` splits payments that a
    # previous round already split - see :func:`reset_vector`.
    minted = ~out["txn_id"].isin(set(raw["txn_id"].to_numpy().tolist())).to_numpy()
    if minted.any():
        # Created lazily. A genome with no row-creating gene leaves the frame's shape exactly
        # as it found it, which several callers and tests reasonably expect.
        if LOOP_ORIGIN not in out.columns:
            out[LOOP_ORIGIN] = pd.Series([pd.NA] * len(out), dtype="object", index=out.index)
        out.loc[minted, LOOP_ORIGIN] = genome.vector_id

    n_aged = 0
    if genome.aged_payee_share > 0.01 and aged_pool is not None and aged_pool.size:
        n_aged = _route_to_aged_accounts(out, idx, genome.aged_payee_share, aged_pool, rng)

    apply_evasion(rng, out, share=genome.share, strength=genome.strength,
                  moves=genome.moves, restrict_to=idx)

    after_ids = set(out["txn_id"].to_numpy()[idx].tolist())
    payees = pd.unique(out["payee_account_id"].to_numpy()[idx])
    return Applied(
        frame=out,
        target_index=idx,
        changed_txn_ids=before_ids | after_ids,
        n_aged_accounts=n_aged,
        n_distinct_payees=int(payees.size),
        seasoning_payments=int(genome.genes.seasoning_depth * payees.size),
    )


def _route_to_aged_accounts(df: pd.DataFrame, idx: np.ndarray, share: float,
                            pool: np.ndarray, rng: np.random.Generator) -> int:
    """Repoint a share of a campaign's payments at accounts that already have history.

    Only the counterparty identity changes. The derived counterparty features are left alone
    deliberately: they are recomputed downstream from the event stream, so the aged account's
    genuine history flows through on its own. Writing them here would be the red team forging
    the defender's own feature store.
    """
    k = int(round(share * idx.size))
    if k <= 0:
        return 0
    chosen = rng.choice(idx, size=k, replace=False)
    # A handful of accounts, reused: brokers sell from a finite pool, and buying one aged
    # account per victim would cost more than the fraud earns.
    n_accounts = max(1, int(np.ceil(k / rng.uniform(3, 9))))
    accounts = rng.choice(pool, size=min(n_accounts, pool.size), replace=False)

    payees = df["payee_account_id"].to_numpy().copy()
    payees[chosen] = rng.choice(accounts, size=k)
    df["payee_account_id"] = payees

    # Confirmation of Payee now matches, because the account name is genuine. This is the
    # part that makes aged accounts worth paying for.
    match = df["payee_name_match_score"].to_numpy().astype(float).copy()
    match[chosen] = np.round(rng.uniform(0.88, 1.0, k), 4)
    df["payee_name_match_score"] = match

    lag = df["_payee_reg_lag_s"].to_numpy().astype(float).copy()
    lag[chosen] = np.round(rng.uniform(5 * 86400, 120 * 86400, k), 1)
    df["_payee_reg_lag_s"] = lag
    return int(accounts.size)


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------

def run_loop(raw: pd.DataFrame, featured: pd.DataFrame, per_vector: pd.DataFrame,
             detector: FraudDetector, cfg: Config, *, verbose: bool = True,
             library: Optional[AttackLibrary] = None,
             customers: Optional[pd.DataFrame] = None,
             attacker_costs: Optional[AttackerCosts] = None,
             defender_costs: Optional[DefenderCosts] = None) -> LoopResult:
    """Co-evolve the red and blue teams for ``cfg.loop.rounds`` rounds.

    ``raw`` is the generated stream before feature engineering (the generator's scratch
    columns are needed to re-apply evasion); ``featured`` is the same stream after features
    and guards, which is what the incoming ``detector`` was fitted on.
    """
    rng = np.random.default_rng(cfg.seed + 991)
    attacker_costs = attacker_costs or AttackerCosts()
    defender_costs = defender_costs or DefenderCosts()

    guard_columns = [c for c in EXTRA_INPUTS if c in featured.columns]
    guards = featured[["txn_id"] + guard_columns].copy()
    graph_columns = [c for c in featured.columns if c.startswith("g_")]
    psps = sorted(pd.unique(featured["payee_psp"].dropna().to_numpy()).tolist())

    aged_pool = aged_account_pool(raw)
    campaign_rows = raw.loc[raw["is_fraud"] == 1, "attack_vector_id"].value_counts()
    # A vector also has to appear in the held-out window, or the round's before/after recall
    # is measured on an empty set and reports as nan. Fitness still uses the whole timeline;
    # this only keeps the *reported* effect of the round meaningful.
    in_test = _test_slice(featured, cfg)
    testable = set(in_test.loc[in_test["is_fraud"] == 1, "attack_vector_id"].unique())
    campaign_rows = campaign_rows[campaign_rows.index.isin(testable)]
    eligible = int((campaign_rows >= cfg.loop.min_fraud_rows).sum())
    if verbose:
        print(f"  red team inventory: {aged_pool.size} aged accounts available for purchase")
        print(f"  {eligible} of {len(testable)} test-window campaigns are large enough to "
              f"search (>= {cfg.loop.min_fraud_rows} fraud rows)")
    if eligible == 0 and verbose:
        print("  no campaign is large enough to search: the loop will not run. Increase the "
              "run size or lower loop.min_fraud_rows.")

    # The stream as the generator produced it, never mutated. Each round's commit realises a
    # campaign from here rather than from last round's realisation of it, which is what stops
    # the row-creating genes compounding - see :func:`reset_vector`.
    pristine = raw.copy()
    current_raw = raw.copy()
    current_featured = featured
    current_detector = detector
    current_per_vector = per_vector.copy()

    survivors: List[AttackGenome] = []
    rounds: List[RoundResult] = []
    tactic_rows: List[Dict[str, object]] = []
    blue_rows: List[Dict[str, object]] = []
    transfer_rows: List[Dict[str, object]] = []
    detectors: List[Tuple[int, FraudDetector]] = [(0, detector)]
    tactic_frames: Dict[int, pd.DataFrame] = {}
    discoveries: List[DiscoveredVector] = []

    for round_index in range(1, cfg.loop.rounds + 1):
        if verbose:
            print(f"  round {round_index}/{cfg.loop.rounds}: red team searching")

        # Per-phase wall clock. A full-profile run had a round that outran the two before it
        # by a factor of seven with nothing in the log between "searching" and "answering" to
        # say where the time went, and no way to find out without attaching a profiler to a
        # process already an hour in. One dictionary is cheap enough to always pay for.
        phase: Dict[str, float] = {}
        clock = time.perf_counter()

        def _mark(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            phase[name] = phase.get(name, 0.0) + (now - clock)
            clock = now

        targets = _target_vectors(current_per_vector, campaign_rows, rng,
                                  k=max(3, cfg.loop.population // 3),
                                  min_rows=cfg.loop.min_fraud_rows)
        if not targets:
            break
        _mark("targets")

        cache = RebuildCache(featured=current_featured, guards=guards,
                             graph_columns=graph_columns)
        _mark("cache")

        # Red's surrogate, refitted each round because the model it is copying has changed.
        surrogate = SurrogateOracle(seed=cfg.seed + round_index).probe(
            current_detector, current_featured, budget=cfg.loop.oracle_probe_budget, rng=rng)
        _mark("surrogate")

        population = _propose(rng, cfg, targets, survivors, psps)
        _mark("propose")
        population = _screen(population, surrogate, current_featured, rng, cfg)
        _mark("screen")

        base_test = _test_slice(current_featured, cfg)
        scored: List[Dict[str, object]] = []
        for genome in population:
            evaluated = _evaluate(genome, current_raw, cache, current_detector, cfg,
                                  rng, aged_pool, customers, attacker_costs, pristine)
            if evaluated is None:
                continue
            evaluated["novelty"] = _novelty(genome, survivors)
            # Profit as a share of the money the campaign was trying to move, so the number
            # is comparable across campaigns of very different sizes, and clipped because a
            # tiny campaign can spend many times its own value on tooling.
            economics = evaluated["economics"]
            normalised = float(np.clip(
                economics.profit / max(economics.value_at_risk, 1.0), -1.0, 1.0))
            evaluated["fitness"] = normalised + cfg.loop.novelty_bonus * evaluated["novelty"]
            scored.append(evaluated)

        _mark("evaluate")

        if not scored:
            break
        scored.sort(key=lambda r: r["fitness"], reverse=True)
        survivors = [r["genome"] for r in scored[: cfg.loop.survivors]]
        targeted = {g.vector_id for g in survivors}

        for entry in scored:
            tactic_rows.append({
                "round": round_index,
                **entry["genome"].describe(),
                "fraud_rows_scored": entry["rows"],
                "recall_vs_frozen_model": round(entry["recall"], 4),
                "evasion": round(1.0 - entry["recall"], 4),
                **entry["economics"].describe(),
                "novelty": round(entry["novelty"], 4),
                "fitness": round(entry["fitness"], 4),
                "survived": entry["genome"] in survivors,
            })

        # Commit the surviving tactics, then let blue answer.
        if verbose:
            print(f"  round {round_index}: blue team answering "
                  f"{len(survivors)} surviving tactics")
        clock = time.perf_counter()
        for genome in survivors:
            current_raw = apply_genome(reset_vector(current_raw, pristine, genome.vector_id),
                                       genome, rng, aged_pool, customers).frame
        current_raw = current_raw.drop(columns=[TARGET_MARKER], errors="ignore")
        _mark("commit")

        # Full, exact recomputation at commit time. Everything reported below comes from
        # this frame, never from the incremental approximation the candidate search used.
        frame = full_rebuild(current_raw, guards)
        current_featured = frame
        graph_columns = [c for c in frame.columns if c.startswith("g_")]
        split = temporal_split(frame, test_days=cfg.defence.test_days,
                               calibration_days=cfg.defence.calibration_days)
        _mark("full_rebuild")

        retrained = FraudDetector(
            seed=cfg.seed + round_index,
            target_fpr=cfg.defence.target_fpr,
            max_iter=cfg.defence.max_iter,
            n_estimators=cfg.defence.forest_n_estimators,
        ).fit(split.train, split.calibration)
        _, post_fpr = _recall_and_fpr(retrained, split.test)
        _mark("retrain")

        move, options = choose_move(split=split, frozen=current_detector,
                                    retrained=retrained, targeted=sorted(targeted),
                                    costs=defender_costs, seed=cfg.seed + round_index)
        _mark("blue_moves")
        blue_rows.extend({"round": round_index, **row}
                         for row in options.to_dict("records"))

        extracted = float(sum(e["economics"].value_extracted
                              for e in scored if e["genome"] in survivors))
        spent = float(sum(e["economics"].cost for e in scored if e["genome"] in survivors))

        rounds.append(RoundResult(
            round_index=round_index,
            rows_targeted=int(((split.test["attack_vector_id"].isin(targeted))
                               & (split.test["is_fraud"] == 1)).sum()),
            eligible_vectors=eligible,
            searched_vectors=len(targets),
            baseline_recall=_recall(current_detector, base_test),
            baseline_recall_targeted=_recall(current_detector, base_test, targeted),
            post_attack_recall=_recall(current_detector, split.test),
            post_attack_recall_targeted=_recall(current_detector, split.test, targeted),
            post_retrain_recall=_recall(retrained, split.test),
            post_retrain_recall_targeted=_recall(retrained, split.test, targeted),
            post_retrain_fpr=post_fpr,
            attacker_value_extracted=extracted,
            attacker_cost=spent,
            attacker_roi=float((extracted - spent) / spent) if spent > 0 else float("nan"),
            blue_move=move.name,
            blue_move_detail=move.description,
            blue_cost=move.total_cost,
            surrogate_agreement=(surrogate.report.agreement if surrogate.report
                                 else float("nan")),
            survivors=list(survivors),
            blue_options=options,
        ))

        # Keep this round's attack traffic and this round's detector so the transfer matrix
        # can replay history rather than recompute it.
        tactic_frames[round_index] = split.test[
            (split.test["attack_vector_id"].isin(targeted))
            & (split.test["is_fraud"] == 1)
        ].copy()
        detectors.append((round_index, retrained))

        for entry in scored:
            if entry["genome"] not in survivors:
                continue
            discovery = discovered_vector(
                parent=library.by_id.get(entry["genome"].vector_id) if library else None,
                parent_id=entry["genome"].vector_id,
                label=", ".join(entry["genome"].levers()),
                levers=entry["genome"].levers(),
                round_index=round_index,
                recall=entry["recall"],
                roi=entry["economics"].roi,
                cost=entry["economics"].cost,
                index=len(discoveries) + 1,
            )
            if discovery is not None:
                discoveries.append(discovery)

        current_detector = move.detector or retrained
        if move.threshold is not None:
            current_detector = _with_threshold(current_detector, move.threshold)
        current_per_vector = _per_vector_recall(current_detector, split.test)
        _mark("bookkeeping")
        if verbose:
            spend = ", ".join(f"{k} {v:.0f}s" for k, v in
                              sorted(phase.items(), key=lambda kv: -kv[1]) if v >= 1.0)
            print(f"  round {round_index}: {sum(phase.values()):.0f}s ({spend})")

    # --- cross-round analysis -----------------------------------------------------------
    for tactic_round, frame in tactic_frames.items():
        if frame.empty:
            continue
        for detector_round, model in detectors:
            transfer_rows.append({
                "tactic_round": tactic_round,
                "detector_round": detector_round,
                "recall": float((model.predict_proba(frame) >= model.threshold).mean()),
                "rows": len(frame),
            })

    gaps = control_gap_summary([
        (g.label(), g.levers(), library.by_id.get(g.vector_id) if library else None)
        for r in rounds for g in r.survivors
    ])

    accepted, rejections, document = ([], [], "")
    if library is not None and discoveries:
        accepted, rejections, document = validate_and_emit(discoveries, library)

    history = pd.DataFrame([r.to_dict() for r in rounds])
    return LoopResult(
        rounds=rounds,
        history=history,
        final_detector=current_detector,
        tactics=pd.DataFrame(tactic_rows),
        transfer=transfer_matrix(transfer_rows),
        control_gaps=gaps,
        blue_options=pd.DataFrame(blue_rows),
        discovered=accepted,
        discovered_yaml=document,
        discovery_rejections=rejections,
        convergence=convergence(history),
    )


# --------------------------------------------------------------------------------------
# Round internals
# --------------------------------------------------------------------------------------

def _propose(rng: np.random.Generator, cfg: Config, targets: Sequence[str],
             survivors: Sequence[AttackGenome], psps: Sequence[str]) -> List[AttackGenome]:
    """Generate the round's candidate population, oversampled for surrogate screening."""
    size = cfg.loop.population * max(cfg.loop.screen_multiplier, 1)
    out: List[AttackGenome] = []
    for _ in range(size):
        if survivors and rng.random() < 0.6:
            parent = survivors[int(rng.integers(0, len(survivors)))]
            out.append(_random_genome(rng, parent.vector_id, psps, parent,
                                      cfg.loop.mutation_scale))
        else:
            out.append(_random_genome(rng, str(rng.choice(targets)), psps))
    return out


def _screen(population: List[AttackGenome], surrogate: SurrogateOracle,
            featured: pd.DataFrame, rng: np.random.Generator,
            cfg: Config) -> List[AttackGenome]:
    """Cut the population down to what the attacker's own model rates most promising.

    The preview is deliberately crude: the cosmetic levers are applied to a copy of the
    target rows and the surrogate is asked what it thinks, with the *derived* features left
    stale. That understates the structural genes, which is a real limitation of an attacker
    reasoning from a stolen copy rather than from the bank's feature store, and it is the
    honest version of the capability. Everything that survives screening is then measured
    properly against the real pipeline.
    """
    if not surrogate.usable or len(population) <= cfg.loop.population:
        return population[: cfg.loop.population]

    ranked: List[Tuple[float, int, AttackGenome]] = []
    for position, genome in enumerate(population):
        rows = featured[(featured["attack_vector_id"] == genome.vector_id)
                        & (featured["is_fraud"] == 1)]
        if rows.empty:
            ranked.append((0.5, position, genome))
            continue
        preview = rows.copy()
        apply_evasion(rng, preview, share=genome.share, strength=genome.strength,
                      moves=genome.moves)
        # Lower predicted decline probability is better for the attacker.
        ranked.append((float(surrogate.decline_probability(preview).mean()), position, genome))

    ranked.sort(key=lambda item: item[0])
    return [genome for _, _, genome in ranked[: cfg.loop.population]]


def _evaluate(genome: AttackGenome, raw: pd.DataFrame, cache: RebuildCache,
              detector: FraudDetector, cfg: Config, rng: np.random.Generator,
              aged_pool: np.ndarray, customers: Optional[pd.DataFrame],
              costs: AttackerCosts,
              pristine: Optional[pd.DataFrame] = None) -> Optional[Dict[str, object]]:
    """Score one genome against the frozen model and price what it cost to run.

    Recall is measured over the whole timeline rather than the test window. An attacker
    running a campaign has no train/test split; they observe whether their traffic gets
    through the deployed model, which is all of it. Restricting fitness to an eight-day slice
    leaves a handful of rows per vector, and a search whose signal is one transaction wide is
    a random walk. The measurement is biased toward the defender, since the model was fitted
    on part of this span, so a tactic that scores well here has cleared a conservative bar.
    """
    # The candidate is a different way of running this campaign, not an addition to the way
    # it is already being run, so its own vector goes back to how the generator produced it
    # before the genome is applied. Other campaigns keep their committed realisation.
    base = reset_vector(raw, pristine, genome.vector_id) if pristine is not None else raw
    applied = apply_genome(base, genome, rng, aged_pool, customers)
    if applied.target_index.size < cfg.loop.min_fraud_rows:
        return None

    dirty = dirty_entities(base, applied.frame, applied.changed_txn_ids)
    neighbourhood = rebuild_neighbourhood(applied.frame, cache, dirty,
                                          applied.changed_txn_ids)
    if neighbourhood.empty:
        return None

    rows = neighbourhood[(neighbourhood["attack_vector_id"] == genome.vector_id)
                         & (neighbourhood["is_fraud"] == 1)]
    if len(rows) < cfg.loop.min_fraud_rows:
        return None

    detected = detector.predict_proba(rows) >= detector.threshold
    amounts = rows["amount_inr" if "amount_inr" in rows.columns
                   else "amount"].to_numpy().astype(float)

    economics = score_campaign(
        amounts=amounts, detected=detected, moves=genome.levers(),
        structuring=genome.genes.structuring,
        n_distinct_payees=applied.n_distinct_payees,
        n_aged_accounts=applied.n_aged_accounts,
        seasoning_payments=applied.seasoning_payments,
        costs=costs,
    )
    return {
        "genome": genome,
        "recall": float(detected.mean()),
        "rows": int(len(rows)),
        "economics": economics,
    }


def _with_threshold(detector: FraudDetector, threshold: float) -> FraudDetector:
    """A shallow view of the detector cut at a different point.

    Copied rather than mutated because the original is kept in the transfer matrix, and
    silently re-cutting a model that history is being replayed against would rewrite the
    past every time blue chose to re-threshold.
    """
    import copy

    clone = copy.copy(detector)
    clone.threshold = float(threshold)
    return clone


# --------------------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------------------

def _test_slice(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return temporal_split(frame, test_days=cfg.defence.test_days,
                          calibration_days=cfg.defence.calibration_days).test


def _recall(detector: FraudDetector, test: pd.DataFrame,
            vector_ids: Optional[set] = None) -> float:
    fraud = test[test["is_fraud"] == 1]
    if vector_ids is not None:
        fraud = fraud[fraud["attack_vector_id"].isin(vector_ids)]
    if fraud.empty:
        return float("nan")
    return float((detector.predict_proba(fraud) >= detector.threshold).mean())


def _recall_and_fpr(detector: FraudDetector, test: pd.DataFrame) -> Tuple[float, float]:
    scores = detector.predict_proba(test)
    alert = scores >= detector.threshold
    y = test["is_fraud"].to_numpy() == 1
    return float(alert[y].mean()), float(alert[~y].mean())


def _per_vector_recall(detector: FraudDetector, test: pd.DataFrame) -> pd.DataFrame:
    fraud = test[test["is_fraud"] == 1].copy()
    if fraud.empty:
        return pd.DataFrame(columns=["attack_vector_id", "rows", "recall"])
    fraud["alert"] = detector.predict_proba(fraud) >= detector.threshold
    return (
        fraud.groupby("attack_vector_id")["alert"]
        .agg(rows="size", recall="mean")
        .reset_index()
    )


def convergence(history: pd.DataFrame) -> Dict[str, object]:
    """Does the arms race settle, oscillate, or run away?

    Answered from the targeted recall after each round's response. A least-squares slope
    says which way it is going; the sign changes in the round-to-round differences say
    whether it is going there smoothly or bouncing. Both are reported with the round count,
    because a slope fitted to three points is not evidence of anything and the reader should
    be able to see that immediately rather than infer it.
    """
    if history.empty or "recall_targeted_after_retrain" not in history.columns:
        return {"rounds": 0, "verdict": "not run"}

    series = history["recall_targeted_after_retrain"].astype(float).to_numpy()
    series = series[np.isfinite(series)]
    n = int(series.size)
    if n < 2:
        return {"rounds": n, "verdict": "too few rounds to characterise"}

    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, series, 1)[0])
    deltas = np.diff(series)
    sign_changes = int((np.diff(np.sign(deltas[deltas != 0])) != 0).sum()) if deltas.any() else 0
    oscillation = sign_changes / max(len(deltas) - 1, 1)

    if n < 5:
        verdict = "too few rounds to characterise"
    elif abs(slope) < 0.005 and oscillation < 0.5:
        verdict = "converged: the defence holds its ground round on round"
    elif slope < -0.005:
        verdict = "diverging: the attacker is gaining faster than the defence recovers"
    elif oscillation >= 0.5:
        verdict = "oscillating: each side takes turns, with no stable advantage"
    else:
        verdict = "converging: the defence is gaining round on round"

    return {
        "rounds": n,
        "recall_slope_per_round": round(slope, 5),
        "oscillation_index": round(float(oscillation), 3),
        "final_targeted_recall": round(float(series[-1]), 4),
        "verdict": verdict,
    }
