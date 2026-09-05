"""Properties the generated stream has to hold before any detection number means anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redteam.generate.agentic_corpus import CORPUS_COLUMNS
from redteam.generate.artefacts import find_artefacts
from redteam.generate.attacks.model_attacks import EVASION_MOVES, apply_evasion
from redteam.generate.fidelity import (RECALL_AT_BUDGET_CEILING, FidelityReport, assess)
from redteam.schema import CATEGORICAL_COLUMNS, DEFAULTS, LABEL_COLUMNS, all_columns


def test_stream_conforms_to_the_schema(transactions):
    expected = [c for c in all_columns()]
    assert [c for c in transactions.columns if not c.startswith("_")] == expected


def test_no_missing_values_in_observable_columns(transactions):
    observable = [c for c in transactions.columns if not c.startswith("_")]
    missing = transactions[observable].isna().sum()
    assert missing[missing > 0].empty, missing[missing > 0].to_dict()


def test_stream_is_time_ordered_with_unique_ids(transactions):
    assert transactions["timestamp"].is_monotonic_increasing
    assert transactions["txn_id"].is_unique


def test_fraud_rate_lands_near_the_configured_target(transactions, tiny_config):
    got = float(transactions["is_fraud"].mean())
    target = tiny_config.attacks.target_fraud_rate
    # Episodes are never split to hit a budget, so the realised rate lands within an
    # episode's worth of the target rather than exactly on it.
    assert 0.8 * target <= got <= 1.25 * target, got


def test_labels_are_internally_consistent(transactions):
    fraud = transactions[transactions["is_fraud"] == 1]
    legit = transactions[transactions["is_fraud"] == 0]
    assert (fraud["attack_vector_id"] != "").all()
    assert (fraud["fraud_type"] != "none").all()
    # Campaigns contain payments that are not themselves fraudulent - the small cultivation
    # payments a synthetic identity makes to build a credit file, for instance. They keep
    # the campaign id but must never be labelled fraud, or recall would be measured against
    # a population that includes the attacker's own cover traffic.
    assert (legit["fraud_type"] == "none").all()
    tagged = legit[legit["attack_vector_id"] != ""]
    assert len(tagged) < len(fraud), "cover traffic should not outweigh the attack itself"

    # A hard negative is a legitimate payment that looks alarming; if any were labelled
    # fraud the false-positive analysis would be measuring the wrong population.
    assert int((transactions["is_hard_negative"] == 1).sum()) > 0
    assert int(((transactions["is_hard_negative"] == 1) & (transactions["is_fraud"] == 1)).sum()) == 0


def test_attack_diversity(transactions, library):
    vectors = set(transactions.loc[transactions["is_fraud"] == 1, "attack_vector_id"].unique())
    simulated = {v.id for v in library.simulated()}
    assert vectors <= simulated
    # Every simulated vector is allocated budget, so all of them should appear.
    assert len(vectors) >= len(simulated) - 2, sorted(simulated - vectors)
    families = transactions.loc[transactions["is_fraud"] == 1, "fraud_type"].nunique()
    assert families >= 6


def test_categoricals_stay_inside_their_enumerations(transactions):
    from redteam.schema import AUTH_METHODS, CHANNELS, RAILS

    assert set(transactions["rail"].unique()) <= set(RAILS)
    assert set(transactions["channel"].unique()) <= set(CHANNELS)
    assert set(transactions["auth_method"].unique()) <= set(AUTH_METHODS)
    for col in CATEGORICAL_COLUMNS:
        assert transactions[col].dtype == object or str(transactions[col].dtype) == "category"


def test_amounts_are_positive_and_heavy_tailed(transactions):
    amounts = transactions["amount"].to_numpy()
    assert (amounts > 0).all()
    # A lognormal-ish spend distribution: the mean sits well above the median.
    assert amounts.mean() > np.median(amounts)


def test_agent_corpus_keys_back_to_agentic_transactions(transactions, corpus):
    assert list(corpus.columns) == CORPUS_COLUMNS
    linked = corpus[corpus["linked"] == 1]
    unlinked = corpus[corpus["linked"] == 0]
    assert not linked.empty and not unlinked.empty

    agentic = set(transactions.loc[transactions["initiated_by_agent"] == 1, "txn_id"])
    assert set(linked["txn_id"]) <= agentic
    assert (unlinked["txn_id"] == "").all()

    injected = corpus[corpus["has_injection"] == 1]
    assert not injected.empty
    assert (injected["payload_family"] != "none").all()
    # Poisoned content the agent read but did not buy from has to exist, or the guard
    # trains on a survivorship-biased sample of only the payloads that worked.
    assert int(((corpus["linked"] == 0) & (corpus["has_injection"] == 1)).sum()) > 0


def test_text_free_agentic_vectors_carry_no_payload(corpus):
    """A counterfeit storefront or a spoofed crawler leaves no adversarial text.

    These bundles look completely ordinary, which is the point: they are the agentic
    vectors a content classifier structurally cannot catch, and the intent guard and the
    behavioural model have to carry them instead.
    """
    text_free = corpus[corpus["payload_family"].isin(["aeo_storefront", "crawler_spoof"])]
    assert not text_free.empty
    assert (text_free["has_injection"] == 0).all()
    assert (text_free["obfuscation"] == "none").all()


def test_injected_payloads_are_not_a_handful_of_fixed_strings(corpus):
    injected = corpus[corpus["has_injection"] == 1]
    assert injected["payload_template"].nunique() >= 8
    assert injected["obfuscation"].nunique() >= 4
    assert injected["text"].nunique() / len(injected) > 0.95


def test_the_payload_span_points_at_the_payload(corpus):
    """Offsets recorded at insertion, not recovered by searching afterwards.

    An interface that highlights an injected span has to be right about where it is: a
    near-miss is indistinguishable from a fabrication to anyone reading the screen. Searching
    for the payload cannot deliver that, because several obfuscations rewrite it after
    composition - ``zero_width`` interleaves U+200B between every character and ``spaced``
    doubles the spaces and upper-cases - so the composed string is not present verbatim in
    the bundle it was inserted into.
    """
    injected = corpus[corpus["has_injection"] == 1]
    assert not injected.empty
    assert (injected["payload_start"] >= 0).all()
    assert (injected["payload_end"] > injected["payload_start"]).all()

    for _, row in injected.head(300).iterrows():
        span = row["text"][row["payload_start"]:row["payload_end"]]
        # The payload is inserted as a whole line, so the span must be exactly one.
        assert span in row["text"].split("\n"), (row["obfuscation"], span[:60])
        assert len(span) == row["payload_end"] - row["payload_start"]

    clean = corpus[corpus["has_injection"] == 0]
    assert (clean["payload_start"] == -1).all(), "a clean bundle must not claim a span"
    assert (clean["payload_end"] == -1).all()


def test_no_single_feature_gives_the_game_away(transactions):
    """The realism check that matters most: fraud must not be separable by one column."""
    report = assess(transactions)
    worst_column, worst_auc = report.details["top_single_feature_auc"][0]
    assert worst_auc < 0.9, (worst_column, worst_auc)
    assert report.overall > 0.6
    assert report.flags == [], report.flags


def test_fraud_is_hard_at_a_realistic_review_budget(transactions):
    """The check the per-feature one cannot make.

    Fraud that no single column reveals can still be trivially separable once the columns
    are combined, and that is the failure mode that quietly turns a detection result into a
    measurement of the generator. The yardstick is recall at a review budget a fraud team
    could actually fund, not AUC, which heavy class imbalance flatters into meaninglessness.
    """
    report = assess(transactions)
    joint = report.details["joint_separability"]
    assert joint["recall_at_review_budget"] <= RECALL_AT_BUDGET_CEILING, joint
    # The opposite failure is data so noisy that no defence could work on it, which would
    # make the whole exercise unfalsifiable.
    assert joint["recall_at_review_budget"] > 0.35, joint


def test_the_separability_probe_reports_its_own_uncertainty(transactions):
    """The probe must not be more confident than its sample size allows.

    Its held-out slice carries a few dozen fraud rows on a run this size, where recall moves
    in steps of 1/n and the interval spans ten points or more. Flagging on the point estimate
    there would fail a build over noise - which is the same error the probe exists to catch,
    committed by the check itself.
    """
    report = assess(transactions)
    for key in ("joint_separability", "derived_separability"):
        detail = report.details[key]
        lo, point, hi = (detail["recall_lo95"], detail["recall_at_review_budget"],
                         detail["recall_hi95"])
        assert lo <= point <= hi, detail
        assert detail["test_fraud_rows"] > 0

        # No flag may be raised unless the lower bound clears the ceiling, whatever the
        # point estimate says. But that state must not read as clean either, so it has to
        # produce a warning - see the next test.
        if point > RECALL_AT_BUDGET_CEILING >= lo:
            assert not any(key.split("_")[0] in f for f in report.flags), report.flags
            assert report.warnings, (key, detail)


def test_the_separability_probe_can_reproduce_its_own_answer(transactions):
    """A probe noisier than the effect it polices is not evidence of anything.

    The original configuration - a depth-6 booster run for a fixed 150 rounds against a sub-1%
    positive rate - overfit by an amount that depended on the draw. Holding the test slice and
    the model seed identical and varying only which rows trained it, it returned 92.3%, 76.9%,
    93.4% and 82.3% recall, around a ceiling set at 92%. Every separability number published
    before that was one sample from that spread.
    """
    report = assess(transactions)
    for key in ("joint_separability", "derived_separability"):
        detail = report.details[key]
        assert detail["refits"] > 1, detail
        # Generous next to the 19 points the old configuration produced, and deliberately
        # looser than the ~7 seen on a run this small, so this fails on a return of the
        # pathology rather than on ordinary small-sample noise.
        assert detail["refit_spread"] < 0.15, (key, detail)


def test_a_check_that_scores_zero_is_never_silent():
    """A zero score with an empty flags list is the report's most misleading state.

    It is reachable and it happened: on a full-size run the derived matrix recovered 93.7% of
    fraud against a 92% ceiling, scoring zero, while the lower bound sat at 91.5% so no flag
    fired. The summary line printed "no flags" and the detection numbers underneath it were
    substantially the generator's. Not flagging was correct - the bar for failing a build is
    "too separable at any reading of the evidence" - but saying nothing was not.
    """
    report = FidelityReport()
    report.scores["derived_separability"] = 0.0
    report.scores["benford_first_digit"] = 0.98

    assert report.zeroed == ["derived_separability"]
    assert not report.flags
    # Whatever the presentation layer does with it, the fact has to survive serialisation.
    assert report.to_dict()["zeroed"] == ["derived_separability"]


def test_the_leakage_detector_catches_a_planted_leak(transactions):
    """A canary for the guard rail itself.

    Every other fidelity assertion here trusts `assess` to notice a leak. That trust is worth
    nothing unless it is tested: a probe that silently stopped working - a mis-wired column
    list, a model that halts before it learns anything, an exception swallowed into a default
    score - would report a clean bill of health on data that was trivially separable, and
    every test above it would keep passing. This plants an obvious leak and demands the
    machinery notice.
    """
    planted = transactions.copy()
    # A column that is the label with a little noise. Nothing subtle: if this gets through,
    # the detector cannot be relied on for anything subtle either.
    rng = np.random.default_rng(11)
    planted["account_balance_ratio"] = np.where(
        planted["is_fraud"] == 1,
        rng.uniform(0.90, 0.99, len(planted)),
        rng.uniform(0.01, 0.10, len(planted)),
    )

    report = assess(planted)
    worst_column, worst_auc = report.details["top_single_feature_auc"][0]
    assert worst_column == "account_balance_ratio", report.details["top_single_feature_auc"][:3]
    assert worst_auc > 0.9, worst_auc
    assert report.flags, "a near-perfect single-feature separator raised no flag"

    # And the joint probe must see it too, since a leak can arrive in a combination that no
    # single-column AUC would rank highly.
    assert report.details["joint_separability"]["recall_at_review_budget"] > 0.9


def test_legitimate_traffic_reaches_young_accounts(transactions):
    """Account age must not be a free separator.

    An ecosystem where every legitimate beneficiary is well established makes "the payee is
    new" a fraud rule with no false positives. Real ecosystems onboard continuously, and
    those accounts get paid from day one.
    """
    legit = transactions[transactions["is_fraud"] == 0]
    young = legit[legit["payee_account_age_days"] < 30]
    assert len(young) / len(legit) > 0.01, "no legitimate payments to young accounts"


def test_every_channel_carries_legitimate_volume(transactions):
    """A channel that only ever appears under attack convicts on its own."""
    legit = transactions[transactions["is_fraud"] == 0]
    for channel in transactions.loc[transactions["is_fraud"] == 1, "channel"].unique():
        assert (legit["channel"] == channel).any(), f"{channel} is fraud-only"


def test_rail_and_channel_stay_coherent(transactions):
    """UPI person-to-person does not happen at a card terminal, under attack or otherwise.

    Forcing a vector onto a rail without redrawing the channel invents pairs that only
    fraud ever produces, handing the defence recall it has not earned.
    """
    pairs = transactions.groupby(["rail", "channel"])["is_fraud"].agg(["sum", "count"])
    fraud_only = pairs[(pairs["sum"] == pairs["count"]) & (pairs["sum"] > 0)]
    leaked = int(fraud_only["sum"].sum())
    assert leaked / int(transactions["is_fraud"].sum()) < 0.05, fraud_only


def test_no_value_or_namespace_convicts_on_its_own(transactions):
    """The generator must not stamp fraud with a value legitimate traffic never carries.

    Three of these shipped at once - a "MULEOP-" customer prefix, an "MX" merchant prefix and
    colliding "DATT" device ids - and each was found by reading code, which does not scale.
    The search runs here instead, so the next one is caught by CI rather than by a judge.

    Only absolute separations fail the test. A tail band that fraud occupies more densely
    than legitimate traffic is a real signal and is expected; a *categorical* value or id
    namespace that fraud alone ever carries is an artefact with no counterpart in reality.
    """
    found = find_artefacts(transactions, include_numeric=False)
    absolute = [a for a in found if a.legit_rows == 0]
    assert not absolute, "fraud-only values: " + "; ".join(a.describe() for a in absolute)


def test_attack_timing_matches_the_benign_calendar(transactions):
    """Day-of-week must not separate fraud, because no attacker chooses Tuesday.

    Benign days are drawn from a weekly x salary-window x drift weighting. Bespoke attack
    generators used to place episodes with a flat draw over the window, so the two
    populations differed in a way nobody designed and day-of-week carried free signal.
    """
    fraud = transactions[transactions["is_fraud"] == 1]
    legit = transactions[transactions["is_fraud"] == 0]
    f = fraud["day_of_week"].value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    l = legit["day_of_week"].value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    # Total variation distance. Some separation is expected from small samples and from the
    # night bias pushing episodes over midnight; a flat-versus-weighted mismatch is far larger.
    tvd = 0.5 * float((f - l).abs().sum())
    assert tvd < 0.16, dict(fraud=f.round(3).to_dict(), legit=l.round(3).to_dict(), tvd=tvd)


def test_mule_accounts_are_seasoned_before_they_collect(transactions):
    """Collection accounts need a history, or "payee has never been seen" ends the game.

    Seasoning is real tradecraft precisely because novelty rules exist, and the credits must
    land strictly before the fraud they are covering for.
    """
    fraud = transactions[transactions["is_fraud"] == 1]
    legit = transactions[transactions["is_fraud"] == 0]
    # Only purpose-opened collection accounts are seasoned. Recruited mules are ordinary
    # customers whose own traffic runs either side of the fraud, which is the point of them.
    # The two are told apart by the ring label - a purpose-opened ring against a recruited
    # one - and deliberately not by the account id, which shares the customer namespace so
    # that no mule is identifiable from its id alone.
    purpose_opened = fraud[fraud["mule_ring_id"].astype(str).str.startswith("RING")]
    opened = purpose_opened.groupby("payee_account_id")["timestamp"].min()
    assert not opened.empty

    seasoning = legit[legit["payee_account_id"].isin(opened.index)]
    assert not seasoning.empty, "no legitimate inbound to any collection account"
    assert seasoning["payee_account_id"].nunique() / len(opened) > 0.3

    lands_before = seasoning["timestamp"] < seasoning["payee_account_id"].map(opened)
    assert lands_before.all(), "seasoning must precede the fraud it covers"


def test_hard_negatives_sometimes_stack(transactions):
    """Single-axis traps let the model win by counting red flags.

    The customer who is abroad, on a new phone, on a support call and paying a fresh payee
    is one person and every fact is true at once; that is the false positive that matters.
    """
    hard = transactions[(transactions["is_fraud"] == 0)
                        & (transactions["is_hard_negative"] == 1)]
    axes = (hard["device_is_new"].astype(int)
            + (hard["distance_from_home_km"] > 400).astype(int)
            + hard["call_in_progress"].astype(int)
            + hard["screen_share_active"].astype(int))
    assert (axes >= 2).mean() > 0.10, "hard negatives are all single-axis"


def test_evasion_only_touches_attacker_controllable_columns(transactions):
    df = transactions.copy()
    before = df.copy()
    rng = np.random.default_rng(0)
    apply_evasion(rng, df, share=1.0, strength=0.9)

    allowed = {name for name, _ in EVASION_MOVES} | {"is_night", "evasion_applied"}
    changed = {c for c in before.columns
               if not before[c].equals(df[c])}
    assert changed <= allowed, changed - allowed

    # The victim's own history is not a lever an attacker has.
    for column in ("customer_tenure_days", "device_age_days", "payee_prior_txn_count"):
        assert before[column].equals(df[column])


def test_evasion_moves_fraud_toward_the_benign_amount_distribution(transactions):
    df = transactions.copy()
    fraud = df["is_fraud"] == 1
    legit_median = float(df.loc[~fraud, "amount"].median())
    before_gap = abs(float(df.loc[fraud, "amount"].median()) - legit_median)

    apply_evasion(np.random.default_rng(1), df, share=1.0, strength=0.95)
    after_gap = abs(float(df.loc[fraud, "amount"].median()) - legit_median)
    assert after_gap < before_gap


def test_evasion_flags_exactly_the_rows_it_touched():
    df = pd.DataFrame(
        {
            "is_fraud": [1] * 10 + [0] * 10,
            "evasion_applied": [0] * 20,
            "amount": np.linspace(100, 2000, 20),
            "hour": [3] * 20,
            "is_night": [1] * 20,
        }
    )
    apply_evasion(np.random.default_rng(2), df, share=0.5, strength=0.5)
    flagged = df["evasion_applied"].to_numpy()
    assert flagged.sum() == 5
    assert flagged[10:].sum() == 0, "evasion must never be applied to legitimate rows"


def test_generation_is_reproducible(tiny_config, library):
    from redteam.generate.campaign import generate_dataset

    a = generate_dataset(tiny_config, library, rng=np.random.default_rng(99)).transactions
    b = generate_dataset(tiny_config, library, rng=np.random.default_rng(99)).transactions
    assert a.shape == b.shape
    pd.testing.assert_frame_equal(a, b)


def test_schema_defaults_cover_every_label_column():
    assert set(LABEL_COLUMNS) <= set(DEFAULTS)


@pytest.mark.parametrize("column", ["payee_prior_txn_count", "payee_inbound_unique_payers_24h"])
def test_counterparty_enrichment_is_causal(transactions, column):
    """A payee's first ever appearance cannot have prior history attached to it."""
    first = transactions.drop_duplicates("payee_account_id", keep="first")
    assert (first[column] == 0).all()
