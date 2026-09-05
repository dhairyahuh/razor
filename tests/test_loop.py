"""Closed-loop tests.

The loop is the part of the system most able to produce a flattering result by accident, so
what is tested is the honesty of the search rather than whether red or blue wins: the
attacker may only pull levers a real attacker controls, mutations have to survive into the
data the defender is measured on, the fast path has to agree with the slow one, and neither
side may be rewarded for something it did not achieve.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from redteam.generate.attacks.model_attacks import EVASION_MOVES
from redteam.loop.coevolution import (
    TARGET_MARKER,
    AttackGenome,
    _novelty,
    _random_genome,
    _target_vectors,
    aged_account_pool,
    apply_genome,
    convergence,
)
from redteam.loop.economics import AttackerCosts, conversion_multiplier, score_campaign
from redteam.loop.strategy import StrategyGenes, apply_strategy, random_genes

MOVE_NAMES = [name for name, _ in EVASION_MOVES]
PSPS = ["PSP01", "PSP02", "PSP03"]


def _mutate(transactions, genome, seed=0, pool=None, customers=None):
    """Apply one genome and hand back a frame comparable with the input."""
    applied = apply_genome(transactions, genome, np.random.default_rng(seed), pool, customers)
    return applied.frame.drop(columns=[TARGET_MARKER], errors="ignore"), applied


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------

def test_the_loop_carries_every_guard_column_the_model_trained_on():
    """A duplicated literal here silently breaks every round of the loop.

    The loop rebuilds features after each mutation and re-attaches guard outputs, because no
    lever in the search space writes to them. That list used to be restated as a tuple of
    string literals, so adding a guard column to the model's inputs left the rebuilt frames
    missing it and the detector raised KeyError on a column it had been fitted on - during
    the loop stage, after generate and defend had already succeeded.
    """
    from redteam.defend.dataset import EXTRA_INPUTS
    from redteam.loop import coevolution

    source = inspect.getsource(coevolution.run_loop)
    assert "EXTRA_INPUTS" in source, "guard columns must be derived, not restated"
    for column in EXTRA_INPUTS:
        assert f'"{column}"' not in source, (
            f"{column} is hard-coded in run_loop; derive it from EXTRA_INPUTS instead"
        )


def test_the_target_marker_is_invisible_to_the_model():
    """The column that tracks which rows red touched must never reach the feature matrix."""
    from redteam.defend.dataset import input_columns

    frame = pd.DataFrame({"amount": [1.0], TARGET_MARKER: [1], "is_fraud": [0]})
    assert TARGET_MARKER not in input_columns(frame)


# --------------------------------------------------------------------------------------
# The genome
# --------------------------------------------------------------------------------------

def test_genomes_stay_inside_their_bounds():
    rng = np.random.default_rng(0)
    genome = _random_genome(rng, "APP-BEC-INVOICE-REDIRECT", PSPS)
    for _ in range(300):
        genome = _random_genome(rng, genome.vector_id, PSPS, genome, mutation_scale=0.9)
        assert 0.2 <= genome.strength <= 0.98
        assert 0.3 <= genome.share <= 1.0
        assert 0.0 <= genome.aged_payee_share <= 1.0
        assert 1 <= len(genome.moves) <= len(MOVE_NAMES)
        assert len(set(genome.moves)) == len(genome.moves)
        assert set(genome.moves) <= set(MOVE_NAMES)
        assert 1 <= genome.genes.structuring <= 8
        assert genome.genes.n_mules is None or 1 <= genome.genes.n_mules <= 40
        assert genome.genes.mule_reuse is None or 0.0 <= genome.genes.mule_reuse <= 1.0
        assert 0 <= genome.genes.seasoning_depth <= 40


def test_mutation_keeps_the_vector_it_inherited():
    """A tactic that drifts onto another vector would make per-vector fitness meaningless."""
    rng = np.random.default_rng(1)
    parent = _random_genome(rng, "ATO-SIM-SWAP-OTP", PSPS)
    for _ in range(50):
        child = _random_genome(rng, "ignored", PSPS, parent)
        assert child.vector_id == "ATO-SIM-SWAP-OTP"
        parent = child


def test_structural_mutation_changes_one_gene_at_a_time():
    """A step that rewires everything is a random restart, and carries no gradient."""
    rng = np.random.default_rng(3)
    parent = random_genes(rng, PSPS)
    for _ in range(100):
        child = random_genes(rng, PSPS, parent)
        differences = sum(
            getattr(parent, f) != getattr(child, f)
            for f in ("rail", "channel", "cohort", "structuring", "spacing_s",
                      "n_mules", "mule_reuse", "payee_psp", "seasoning_depth")
        )
        assert differences <= 1
        parent = child


def test_novelty_rewards_unlike_tactics():
    a = AttackGenome("V", ("amount", "hour"), 0.5, 0.5, 0.0)
    same = AttackGenome("V", ("amount", "hour"), 0.5, 0.5, 0.0)
    different = AttackGenome("V", ("vpn_or_proxy", "is_night"), 0.5, 0.5, 0.9)
    assert _novelty(a, []) == 1.0
    assert _novelty(a, [same]) == pytest.approx(0.0)
    assert _novelty(different, [same]) > 0.5


def test_novelty_sees_structural_genes_too():
    """Two tactics with identical levers but different plans are not the same tactic."""
    plain = AttackGenome("V", ("amount",), 0.5, 0.5, 0.0)
    structured = AttackGenome("V", ("amount",), 0.5, 0.5, 0.0,
                              StrategyGenes(structuring=5, seasoning_depth=20))
    assert _novelty(structured, [plain]) > 0.0


# --------------------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------------------

def test_target_selection_prefers_vectors_that_are_getting_through():
    per_vector = pd.DataFrame(
        {"attack_vector_id": ["EVADES", "CAUGHT"], "recall": [0.05, 1.0]}
    )
    campaign_rows = pd.Series({"EVADES": 100, "CAUGHT": 100})
    rng = np.random.default_rng(0)
    picks = [_target_vectors(per_vector, campaign_rows, rng, k=1, min_rows=15)[0]
             for _ in range(50)]
    assert picks.count("EVADES") > picks.count("CAUGHT")


def test_target_selection_skips_campaigns_too_small_to_score():
    per_vector = pd.DataFrame({"attack_vector_id": ["TINY"], "recall": [0.0]})
    campaign_rows = pd.Series({"TINY": 3})
    assert _target_vectors(per_vector, campaign_rows, np.random.default_rng(0),
                           k=3, min_rows=15) == []


def test_unscored_vectors_are_still_reachable():
    """A vector missing from the evaluation table is unknown, not known-safe."""
    per_vector = pd.DataFrame({"attack_vector_id": ["SCORED"], "recall": [1.0]})
    campaign_rows = pd.Series({"SCORED": 100, "UNSCORED": 100})
    picks = _target_vectors(per_vector, campaign_rows, np.random.default_rng(0),
                            k=2, min_rows=15)
    assert set(picks) == {"SCORED", "UNSCORED"}


# --------------------------------------------------------------------------------------
# Applying a tactic
# --------------------------------------------------------------------------------------

def test_applying_a_genome_only_touches_its_own_vector(transactions):
    vector = transactions.loc[transactions["is_fraud"] == 1,
                              "attack_vector_id"].value_counts().index[0]
    genome = AttackGenome(vector, ("amount", "hour"), 0.8, 1.0, 0.0)
    mutated, _ = _mutate(transactions, genome)

    untouched = transactions["attack_vector_id"] != vector
    pd.testing.assert_frame_equal(transactions[untouched], mutated[untouched])
    assert not transactions.loc[~untouched, "amount"].equals(mutated.loc[~untouched, "amount"])


def test_applying_a_genome_never_touches_legitimate_traffic(transactions):
    vector = transactions.loc[transactions["is_fraud"] == 1,
                              "attack_vector_id"].value_counts().index[0]
    genome = AttackGenome(vector, tuple(MOVE_NAMES), 0.9, 1.0, 0.9)
    pool = aged_account_pool(transactions)
    mutated, _ = _mutate(transactions, genome, pool=pool)

    legit = transactions["is_fraud"] == 0
    pd.testing.assert_frame_equal(transactions[legit], mutated[legit].iloc[: int(legit.sum())])


def test_aged_account_pool_excludes_known_mules(transactions):
    pool = set(aged_account_pool(transactions).tolist())
    assert pool
    fraud_payees = set(transactions.loc[transactions["is_fraud"] == 1, "payee_account_id"])
    assert pool.isdisjoint(fraud_payees)


def test_buying_aged_accounts_reroutes_the_money(transactions):
    """The one lever that changes where funds land rather than how the payment looks."""
    vector = transactions.loc[transactions["is_fraud"] == 1,
                              "attack_vector_id"].value_counts().index[0]
    pool = aged_account_pool(transactions)
    genome = AttackGenome(vector, ("amount",), 0.5, 1.0, 0.8)
    mutated, _ = _mutate(transactions, genome, pool=pool)

    # Fraud rows only. A campaign's vector id also lands on its mule hops and its seasoning
    # traffic, which are not fraud and which the red team has no reason to reroute.
    target = (transactions["attack_vector_id"] == vector) & (transactions["is_fraud"] == 1)
    changed = (transactions.loc[target, "payee_account_id"].to_numpy()
               != mutated.loc[target, "payee_account_id"].to_numpy())
    assert changed.mean() > 0.5
    # Bought accounts are reused: a broker sells from a finite pool.
    new_payees = set(mutated.loc[target, "payee_account_id"]) & set(pool.tolist())
    assert 0 < len(new_payees) < int(target.sum())
    # Confirmation of Payee stops helping, which is what makes them worth buying.
    assert (mutated.loc[target, "payee_name_match_score"].mean()
            > transactions.loc[target, "payee_name_match_score"].mean())


def test_derived_counterparty_features_are_not_forged(transactions):
    """The red team may move the money; it may not write the defender's feature store."""
    vector = transactions.loc[transactions["is_fraud"] == 1,
                              "attack_vector_id"].value_counts().index[0]
    genome = AttackGenome(vector, (), 0.5, 1.0, 0.9)
    mutated, _ = _mutate(transactions, genome, pool=aged_account_pool(transactions))
    for column in ("payee_prior_txn_count", "payee_account_age_days",
                   "payee_inbound_unique_payers_24h"):
        pd.testing.assert_series_equal(transactions[column], mutated[column])


# --------------------------------------------------------------------------------------
# Structural genes
# --------------------------------------------------------------------------------------

def _target_of(transactions):
    vector = transactions.loc[transactions["is_fraud"] == 1,
                              "attack_vector_id"].value_counts().index[0]
    idx = np.where((transactions["attack_vector_id"].to_numpy() == vector)
                   & (transactions["is_fraud"].to_numpy() == 1))[0]
    return vector, idx


def _marked(transactions, idx):
    """A copy carrying the marker `apply_genome` sets, which row-creating genes follow."""
    frame = transactions.copy()
    marker = np.zeros(len(frame), dtype=int)
    marker[idx] = 1
    frame[TARGET_MARKER] = marker
    return frame


def test_structuring_conserves_the_money_and_adds_rows(transactions):
    """Splitting a demand into six moves the same total, or the attacker is losing funds."""
    vector, idx = _target_of(transactions)
    frame = _marked(transactions, idx)
    before = frame.iloc[idx]["amount"].sum()

    out = apply_strategy(frame, idx, StrategyGenes(structuring=4, spacing_s=600.0),
                         np.random.default_rng(0))
    after_rows = out[out[TARGET_MARKER] == 1]
    assert len(after_rows) == 4 * idx.size
    assert after_rows["amount"].sum() == pytest.approx(before, rel=1e-3)
    assert after_rows["txn_id"].nunique() == len(after_rows)
    # Each piece is smaller than the demand it came from - that is the point of the gene.
    assert after_rows["amount"].max() < frame.iloc[idx]["amount"].max()


def test_seasoning_adds_benign_history_to_the_drop_accounts(transactions):
    """Seasoning traffic is not fraud, and labelling it as fraud would be marking our own
    homework."""
    vector, idx = _target_of(transactions)
    frame = _marked(transactions, idx)
    accounts = set(pd.unique(frame["payee_account_id"].to_numpy()[idx]))

    out = apply_strategy(frame, idx, StrategyGenes(seasoning_depth=5),
                         np.random.default_rng(0))
    added = out[out["txn_id"].str.startswith("TSEED")]
    assert len(added) == 5 * len(accounts)
    assert (added["is_fraud"] == 0).all()
    assert set(added["payee_account_id"]) <= accounts

    # Every seeded payment lands before its own account's first fraudulent use, or it is not
    # seasoning, it is just more traffic. Compared per account: the accounts in one campaign
    # are used at different times, so a campaign-wide minimum would be the wrong bar.
    first_use = (frame.iloc[idx].groupby("payee_account_id")["timestamp"].min())
    latest_seed = added.groupby("payee_account_id")["timestamp"].max()
    horizon = pd.Timestamp(frame["timestamp"].min())
    for account, seeded in latest_seed.items():
        assert seeded <= max(first_use[account], horizon)


def test_capping_the_mule_count_concentrates_the_fan_in(transactions):
    vector, idx = _target_of(transactions)
    frame = _marked(transactions, idx)
    apply_strategy(frame, idx, StrategyGenes(n_mules=2), np.random.default_rng(0))
    assert pd.unique(frame["payee_account_id"].to_numpy()[idx]).size <= 2


def test_switching_rail_keeps_revocability_and_currency_consistent(transactions):
    """A rail switch that left `is_irrevocable_rail` stale would be a simulator artefact the
    model could detect instead of the attack."""
    from redteam.schema import IRREVOCABLE_RAILS

    vector, idx = _target_of(transactions)
    frame = _marked(transactions, idx)
    apply_strategy(frame, idx, StrategyGenes(rail="NEFT"), np.random.default_rng(0))

    assert (frame["rail"].to_numpy()[idx] == "NEFT").all()
    assert (frame["is_irrevocable_rail"].to_numpy()[idx] == int("NEFT" in IRREVOCABLE_RAILS)).all()
    assert (frame["currency"].to_numpy()[idx] == "INR").all()
    assert not np.isin(frame["channel"].to_numpy()[idx], ["pos", "recurring"]).any()


def test_retargeting_a_cohort_rewrites_the_customer_context(transactions, population):
    """Changing who is attacked without changing their tenure would hand the model a
    contradiction that does not occur in real data."""
    vector, idx = _target_of(transactions)
    frame = _marked(transactions, idx)
    before = frame["customer_tenure_days"].to_numpy()[idx].copy()

    apply_strategy(frame, idx, StrategyGenes(cohort="corporate"), np.random.default_rng(0),
                   customers=population.customers)

    assert not np.array_equal(before, frame["customer_tenure_days"].to_numpy()[idx])
    # Corporate customers, so the corporate flag should be well above the population rate.
    assert frame["customer_is_corporate"].to_numpy()[idx].mean() > 0.5


# --------------------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------------------

def test_evading_by_becoming_harmless_is_not_rewarded():
    """The degenerate strategy the cost model exists to price out.

    A tactic that evades perfectly on trivial amounts must score worse than one that is
    caught half the time on real money. Under a `1 - recall` fitness the first one wins,
    which is why the search used to converge on nothing.
    """
    costs = AttackerCosts()
    harmless = score_campaign(
        amounts=np.full(40, 800.0), detected=np.zeros(40, dtype=bool),
        moves=("amount",), structuring=1, n_distinct_payees=4, n_aged_accounts=0,
        seasoning_payments=0, costs=costs,
    )
    lucrative = score_campaign(
        amounts=np.full(40, 60_000.0), detected=np.tile([True, False], 20),
        moves=("amount",), structuring=1, n_distinct_payees=4, n_aged_accounts=0,
        seasoning_payments=0, costs=costs,
    )
    assert harmless.value_extracted > 0
    assert lucrative.profit > harmless.profit


def test_aged_accounts_are_the_expensive_line_item():
    """The gene that defeats counterparty novelty has to cost what it costs, or the search
    will buy it every time and learn nothing about the trade-off."""
    costs = AttackerCosts()
    common = dict(amounts=np.full(30, 50_000.0), detected=np.zeros(30, dtype=bool),
                  structuring=1, n_distinct_payees=6, seasoning_payments=0, costs=costs)
    fresh = score_campaign(moves=(), n_aged_accounts=0, **common)
    aged = score_campaign(moves=("aged_payee_share",), n_aged_accounts=6, **common)
    assert aged.cost > fresh.cost * 3
    assert aged.cost_breakdown["aged_accounts"] > sum(
        v for k, v in aged.cost_breakdown.items() if k != "aged_accounts")


def test_suppressing_the_coercion_tells_costs_conversion():
    """Hiding the tells is free in cash and expensive in outcomes: without an operator on
    the phone, fewer victims complete."""
    costs = AttackerCosts()
    assert conversion_multiplier((), 1, costs) == 1.0
    assert conversion_multiplier(("call_in_progress",), 1, costs) < 0.7
    # Screen share and remote access are the same operational loss, charged once.
    both = conversion_multiplier(("screen_share_active", "remote_access_app_detected"), 1, costs)
    one = conversion_multiplier(("screen_share_active",), 1, costs)
    assert both == pytest.approx(one)


def test_structuring_costs_conversion_per_extra_payment():
    costs = AttackerCosts()
    assert conversion_multiplier((), 6, costs) < conversion_multiplier((), 2, costs) < 1.0


# --------------------------------------------------------------------------------------
# The incremental rebuild
# --------------------------------------------------------------------------------------

def test_the_fast_rebuild_agrees_with_the_slow_one(transactions):
    """The whole twelve-round budget rests on this equality, so it is asserted rather than
    argued.

    Features for the rows the search scores must come out bit-for-bit identical whether the
    neighbourhood was recomputed or the entire stream was. Anything less and the fitness
    function is measuring a different model than the one blue is graded on.
    """
    from redteam.defend.dataset import EXTRA_INPUTS
    from redteam.features import build_features, build_graph_features
    from redteam.generate.enrich import enrich
    from redteam.loop.incremental import (
        RebuildCache,
        dirty_entities,
        rebuild_neighbourhood,
    )

    base = build_graph_features(build_features(enrich(transactions)))
    guards = pd.DataFrame({"txn_id": base["txn_id"]})
    for column in EXTRA_INPUTS:
        guards[column] = base[column] if column in base.columns else 0.0
    cache = RebuildCache(featured=base, guards=guards,
                         graph_columns=[c for c in base.columns if c.startswith("g_")])

    vector, _ = _target_of(transactions)
    genome = AttackGenome(vector, ("amount", "form_fill_duration_s"), 0.7, 1.0, 0.0)
    applied = apply_genome(transactions, genome, np.random.default_rng(7))

    # Exercised the way the search calls it, horizon and all. Asserting the unbounded path
    # instead would leave the one that actually runs untested.
    fast = rebuild_neighbourhood(
        applied.frame, cache,
        dirty_entities(transactions, applied.frame, applied.changed_txn_ids),
        applied.changed_txn_ids)
    slow = build_graph_features(build_features(enrich(applied.frame)))

    # Every column the model reads has to survive the rebuild under its own name. Both
    # `agent_injection_score` and the other guard outputs exist in the raw schema *and* in
    # the guard layer, so a merge that does not drop the raw copies first silently renames
    # them to `_x`/`_y` and the detector raises KeyError minutes into a round.
    from redteam.defend.dataset import input_columns

    for column in input_columns(base):
        assert column in fast.columns, f"{column} lost by the incremental rebuild"
    assert not any(c.endswith(("_x", "_y")) for c in fast.columns)

    scored = (fast["attack_vector_id"] == vector) & (fast["is_fraud"] == 1)
    fast_rows = fast[scored].set_index("txn_id").sort_index()
    slow_rows = slow[(slow["attack_vector_id"] == vector)
                     & (slow["is_fraud"] == 1)].set_index("txn_id").sort_index()
    assert len(fast_rows) > 10

    columns = [c for c in fast_rows.columns
               if c.startswith("f_") and pd.api.types.is_numeric_dtype(fast_rows[c])]
    assert len(columns) > 30
    for column in columns:
        np.testing.assert_allclose(
            fast_rows[column].to_numpy(dtype=float),
            slow_rows[column].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9, equal_nan=True,
            err_msg=f"{column} differs between the incremental and full rebuild",
        )


def test_committing_a_tactic_replaces_it_rather_than_stacking_on_it(transactions):
    """Row-creating genes must not compound across rounds.

    ``structuring=4`` splits each of a campaign's payments into four. Applied again next round
    to the frame it already produced, it splits those four into sixteen. On a full-profile run
    the fraud population went 1,978 -> 3,022 -> 7,198 -> 23,902 -> 90,718 over four commits,
    and the book went from a realistic 0.70% fraud to 23%. Runtime was the visible symptom -
    every phase of a round scaled with the frame, so round two took five times round one - but
    the real damage is that precision, false-positive rate and both cost models were being
    computed on a portfolio that no longer resembled one.
    """
    from redteam.loop.coevolution import reset_vector
    from redteam.loop.strategy import StrategyGenes

    vector, _ = _target_of(transactions)
    genome = AttackGenome(vector, ("amount",), 0.7, 1.0, 0.0,
                          StrategyGenes(structuring=4, seasoning_depth=3))
    rng = np.random.default_rng(7)

    pristine = transactions.copy()
    once = apply_genome(pristine, genome, rng).frame
    grew = int(once["is_fraud"].sum())
    assert grew > int(pristine["is_fraud"].sum()), "the gene should create rows at all"

    # Re-realising the same campaign four more times must land on the same size every time,
    # because each commit replaces the last rather than building on it.
    frame = once
    for _ in range(4):
        frame = apply_genome(reset_vector(frame, pristine, vector), genome, rng).frame
        assert int(frame["is_fraud"].sum()) == grew, "campaign realisation compounded"

    # And the campaign has to survive the round trip intact, not merely stay the same size.
    assert set(pristine.loc[(pristine["attack_vector_id"] == vector)
                            & (pristine["is_fraud"] == 1), "txn_id"]).issubset(
        set(frame["txn_id"]))
    back = reset_vector(frame, pristine, vector)
    assert len(back) == len(pristine)


def test_the_window_horizon_actually_removes_rows(transactions):
    """The bound has to bind, or it is a comment rather than an optimisation.

    A dirty merchant is only ever read through a trailing window, so its rows outside that
    horizon cannot affect anything the search scores. Leaving them in is what made round three
    of a full-profile run outrun rounds one and two by a factor of seven: the largest merchant
    in that book owns 5% of every row in it, and each of twelve candidates per round rebuilt
    the lot. The equality above proves the bound is safe; this proves it is doing something.
    """
    from redteam.loop.incremental import dirty_entities, neighbourhood

    vector, _ = _target_of(transactions)
    genome = AttackGenome(vector, ("amount", "form_fill_duration_s"), 0.7, 1.0, 0.0)
    applied = apply_genome(transactions, genome, np.random.default_rng(7))
    dirty = dirty_entities(transactions, applied.frame, applied.changed_txn_ids)

    unbounded = neighbourhood(applied.frame, dirty).sum()
    bounded = neighbourhood(applied.frame, dirty, applied.changed_txn_ids).sum()

    assert bounded <= unbounded
    # Every row the bound drops has to be one no dirty payer or payee owned, since those keep
    # their full history for the cumulative columns in `enrich`.
    keep = np.zeros(len(applied.frame), dtype=bool)
    for key in ("payer_account_id", "payee_account_id"):
        keep |= applied.frame[key].isin(dirty[key]).to_numpy()
    dropped = neighbourhood(applied.frame, dirty) & ~neighbourhood(
        applied.frame, dirty, applied.changed_txn_ids)
    assert not (dropped & keep).any()


# --------------------------------------------------------------------------------------
# Blue's move set
# --------------------------------------------------------------------------------------

def test_blue_prices_every_move_and_picks_the_cheapest(tiny_config, transactions):
    from redteam.defend.dataset import temporal_split
    from redteam.defend.model import FraudDetector
    from redteam.features import build_features, build_graph_features
    from redteam.generate.enrich import enrich
    from redteam.loop.blue import choose_move
    from redteam.loop.economics import DefenderCosts

    frame = build_graph_features(build_features(enrich(transactions)))
    split = temporal_split(frame, test_days=tiny_config.defence.test_days,
                           calibration_days=tiny_config.defence.calibration_days)
    detector = FraudDetector(seed=0, target_fpr=tiny_config.defence.target_fpr,
                             max_iter=40, n_estimators=20).fit(split.train, split.calibration)

    targeted = sorted(split.test.loc[split.test["is_fraud"] == 1,
                                     "attack_vector_id"].unique())[:3]
    move, options = choose_move(split=split, frozen=detector, retrained=detector,
                                targeted=targeted, costs=DefenderCosts())

    assert {"hold", "rethreshold", "retrain"} <= set(options["move"])
    assert move.total_cost == pytest.approx(options["total_cost"].min())
    # Every option has to be costed; an unpriced move would always look free and always win.
    assert options["total_cost"].notna().all()


def test_holding_is_a_legitimate_move():
    """A defence that must always act will always act, and over-fit to whatever it saw last."""
    from redteam.loop.blue import choose_move

    source = inspect.getsource(choose_move)
    assert "_hold" in source


# --------------------------------------------------------------------------------------
# The return path
# --------------------------------------------------------------------------------------

def test_control_gap_never_claims_a_control_it_did_not_defeat():
    from redteam.identify.library import load_library
    from redteam.loop.discovery import control_gap

    library = load_library()
    vector = next(v for v in library.vectors if "confirmation_of_payee" in v.controls)

    defeated = control_gap(("payee_name_match_score",), vector)
    assert defeated.loc[defeated["control"] == "confirmation_of_payee", "status"].iloc[0] \
        == "defeated"

    # A tactic that pulled an unrelated lever leaves it intact.
    intact = control_gap(("hour",), vector)
    assert intact.loc[intact["control"] == "confirmation_of_payee", "status"].iloc[0] == "intact"


def test_discovered_vectors_pass_the_same_schema_gate_as_hand_authored_ones():
    from redteam.identify.library import load_library
    from redteam.loop.discovery import discovered_vector, validate_and_emit

    library = load_library()
    parent = library.simulated()[0]
    discovery = discovered_vector(
        parent=parent, parent_id=parent.id, label="aged accounts + structuring",
        levers=("aged_payee_share", "structuring", "payee_name_match_score"),
        round_index=3, recall=0.2, roi=1.4, cost=42_000.0, index=1,
    )
    assert discovery is not None

    accepted, rejections, document = validate_and_emit([discovery], library)
    assert accepted and not rejections
    assert "discovered_by" in document
    assert accepted[0].provenance["parent_vector"] == parent.id
    # Every declared signal has to be a real observable column, which is the gate itself.
    from redteam.schema import observable_columns

    assert set(accepted[0].vector.signals) <= set(observable_columns())


def test_a_tactic_that_was_caught_is_not_written_into_the_taxonomy():
    """The taxonomy is the Identify pillar's foundation; a search artefact must not dilute it."""
    from redteam.identify.library import load_library
    from redteam.loop.discovery import discovered_vector

    parent = load_library().simulated()[0]
    assert discovered_vector(parent=parent, parent_id=parent.id, label="x",
                             levers=("hour",), round_index=1, recall=0.95,
                             roi=0.1, cost=1.0, index=1) is None


def test_convergence_refuses_to_characterise_too_few_rounds():
    """Four points of a stochastic search is not a curve, and the verdict should say so."""
    short = pd.DataFrame({"recall_targeted_after_retrain": [0.8, 0.7, 0.75]})
    assert "too few rounds" in convergence(short)["verdict"]

    steady = pd.DataFrame({"recall_targeted_after_retrain": [0.80] * 10})
    assert convergence(steady)["verdict"].startswith("converged")

    losing = pd.DataFrame({"recall_targeted_after_retrain":
                           [0.9, 0.85, 0.8, 0.72, 0.66, 0.6, 0.51, 0.45]})
    assert convergence(losing)["verdict"].startswith("diverging")


def test_transfer_matrix_is_square_over_the_rounds_it_saw():
    from redteam.loop.discovery import transfer_matrix

    rows = [{"tactic_round": t, "detector_round": d, "recall": 0.5, "rows": 10}
            for t in (1, 2) for d in (0, 1, 2)]
    matrix = transfer_matrix(rows)
    assert list(matrix.index) == [1, 2]
    assert list(matrix.columns) == [0, 1, 2]
