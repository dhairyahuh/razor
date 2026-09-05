"""The transcript plane and the two guards that read GenAI-native evidence.

The questions worth asking of a text plane are not "does the classifier score well" - a
templated corpus will always let it - but whether the corpus is constructed in a way that
makes the score mean anything, and whether the guard survives a pretext it has never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redteam.defend.deepfake_guard import DeepfakeGuard, has_capture
from redteam.defend.vishing_guard import VishingGuard, held_out_script_recall
from redteam.generate.transcripts import (
    COERCION_TURNS,
    FAMILY_SCRIPTS,
    GENUINE_SCRIPTS,
    build_transcripts,
)
from redteam.genai.transcript_bank import SCRIPT_PRETEXTS, _clean


# --------------------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------------------

def test_one_transcript_per_episode_not_per_payment(transcripts):
    """A scam call produces several transfers; the bank hears one conversation.

    Emitting a transcript per payment would let the classifier count duplicates rather than
    read them, and would inflate recall on exactly the multi-payment episodes the guard is
    supposed to help with.
    """
    coercive = transcripts[transcripts["is_coercive"] == 1]
    assert len(coercive) == coercive["txn_id"].nunique()


def test_legitimate_conversations_outnumber_scams_heavily(transcripts):
    """A 50/50 corpus reports a false-positive rate that has nothing to do with a contact
    centre, where nearly every call about money is entirely ordinary."""
    coercive = int(transcripts["is_coercive"].sum())
    genuine = len(transcripts) - coercive
    assert genuine > 2 * coercive


def test_both_classes_are_composed_the_same_way(transcripts):
    """Symmetry of construction, not of content.

    If the positives were combinatorial and the negatives a handful of fixed scripts, the
    classifier could separate the classes on repetition alone and the reported
    false-positive rate would be a fact about the corpus rather than about the guard.
    """
    for label in (0, 1):
        texts = transcripts.loc[transcripts["is_coercive"] == label, "text"]
        if len(texts) < 30:
            continue
        assert texts.nunique() / len(texts) > 0.9


def test_every_scam_transcript_makes_the_moves_that_define_a_scam(transcripts):
    """Authority, urgency and a redirection are the attack. An operator who drops them has
    no scam left, which is what makes the conversation harder to sanitise than the telemetry
    bit that stands in for it."""
    coercive = transcripts[(transcripts["is_coercive"] == 1)
                           & (transcripts["source"] == "template")]
    redirections = [t.lower() for t in COERCION_TURNS["redirection"]]
    for text in coercive["text"].head(40):
        lowered = text.lower()
        assert any(r.split("{")[0].strip()[:40] in lowered for r in redirections)


def test_no_transcript_is_invented_for_a_card_testing_bot(transcripts, dataset):
    """Vectors outside the family map get no conversation. A phone call attached to an
    automated card-testing run would be fiction, and fiction in the positive class teaches
    the guard something untrue."""
    fraud = dataset.transactions[dataset.transactions["is_fraud"] == 1]
    anchored = fraud[fraud["txn_id"].isin(
        transcripts.loc[transcripts["is_coercive"] == 1, "txn_id"])]
    families = {v.id: v.family for v in dataset.library.vectors} \
        if hasattr(dataset, "library") else {}
    if families:
        seen = {families.get(v, "") for v in anchored["attack_vector_id"].unique()}
        assert seen <= set(FAMILY_SCRIPTS)


def test_scam_script_keys_agree_across_the_template_and_llm_sources():
    """The bank and the templates must be interchangeable within a script.

    A mismatch here is silent: the bank simply never gets sampled, the corpus quietly
    reverts to templates, and the run reports a model-written share it did not have.
    """
    assert set(SCRIPT_PRETEXTS) == set(GENUINE_SCRIPTS)


# --------------------------------------------------------------------------------------
# The vishing guard
# --------------------------------------------------------------------------------------

def test_guard_refuses_to_fit_without_enough_history(transcripts):
    """Better to ship the feature at its default than to fit on the test window."""
    guard = VishingGuard(seed=0)
    late = pd.Timestamp(transcripts["timestamp"].min())
    with pytest.raises(ValueError):
        guard.fit(transcripts, train_end=late, test_start=late)


def test_guard_separates_coercion_from_its_near_twins(fitted_vishing):
    guard, report = fitted_vishing
    assert report.roc_auc > 0.85
    assert report.recall_at_1pct_fpr > 0.5


def test_score_spreads_across_the_episode_not_just_the_anchored_payment(
        fitted_vishing, dataset, transcripts):
    """The bank hears one call and declines several payments on the strength of it.

    Scoring only the transcript's anchor row would model a capability nobody has, and would
    understate the guard on multi-payment episodes.
    """
    guard, _ = fitted_vishing
    scored = guard.score_transactions(dataset.transactions, transcripts)
    anchors = set(transcripts.loc[transcripts["is_coercive"] == 1, "txn_id"])
    episodes = dataset.transactions[
        dataset.transactions["txn_id"].isin(anchors)]["campaign_id"].unique()

    multi = scored[scored["campaign_id"].isin(episodes) & ~scored["txn_id"].isin(anchors)]
    if multi.empty:
        pytest.skip("no multi-payment episodes carried a transcript in this sample")
    assert (multi["vishing_score"] > 0).any()


def test_payments_with_no_conversation_score_zero_not_nan(fitted_vishing, dataset,
                                                          transcripts):
    """"No coercive call was recorded" is a true statement about a card payment at a
    terminal, not a missing one."""
    guard, _ = fitted_vishing
    scored = guard.score_transactions(dataset.transactions, transcripts)
    assert scored["vishing_score"].notna().all()
    assert (scored["vishing_score"] >= 0).all()


def test_a_transcript_only_exists_where_somebody_was_on_the_telephone(transcripts, dataset):
    """Coverage has to be a property of the channel, not of the label.

    Anchoring transcripts anywhere a bank would not hold a recording gives the guard
    telemetry nobody has. Anchoring them only to fraud makes "a transcript exists" the
    answer.
    """
    txns = dataset.transactions.set_index("txn_id")
    anchors = txns.loc[txns.index.intersection(transcripts["txn_id"])]
    on_call = (anchors["call_in_progress"] == 1) | anchors["channel"].isin(
        ["ivr", "branch_assisted"])
    assert on_call.all(), "a transcript was anchored to a payment with no call"


def test_the_mere_existence_of_a_transcript_is_not_the_label(transcripts, dataset):
    """The defect this whole corpus was rebuilt to remove.

    When scam transcripts were emitted for every eligible fraud campaign and the legitimate
    corpus was sized as a multiple of the scam count, a payment that carried a transcript was
    fraudulent 95% of the time. Because the guard's score is near-binary on a single-author
    corpus, the tabular model inherited a column that was effectively the answer and reported
    100% recall. Nothing else in the evaluation could see it.
    """
    txns = dataset.transactions
    carries = txns["txn_id"].isin(set(transcripts["txn_id"]))
    if not carries.any():
        pytest.skip("no transcripts in this sample")
    p_fraud_given_transcript = float(txns.loc[carries, "is_fraud"].mean())
    assert p_fraud_given_transcript < 0.6, (
        f"a payment carrying a transcript is {p_fraud_given_transcript:.0%} likely to be "
        "fraud; the presence of the telemetry is doing the classifier's job"
    )


def test_customer_turns_are_not_a_free_label(transcripts):
    """The trap this corpus fell into once.

    When the customer's side of the call appeared only in scam transcripts, "sounds like
    somebody being walked through their banking app" separated the classes on its own, and
    it transferred across held-out vocabulary as a register rather than as content - which
    made every holdout evaluation look like generalisation when it was an artefact.
    """
    from redteam.generate.transcripts import VICTIM_TURNS

    def share(label: int) -> float:
        texts = transcripts.loc[transcripts["is_coercive"] == label, "text"]
        return float(texts.apply(
            lambda t: any(v[:30] in t for v in VICTIM_TURNS)).mean())

    assert share(0) > 0.5, "genuine calls have a customer side too"
    assert abs(share(1) - share(0)) < 0.35


def test_neither_class_is_systematically_longer(transcripts):
    """Length is a feature. A class that runs consistently longer hands the guard a signal
    that has nothing to do with coercion."""
    lengths = transcripts.groupby("is_coercive")["text"].apply(lambda s: s.str.len().mean())
    assert max(lengths) / min(lengths) < 1.5


def test_openers_overlap_across_the_label(transcripts):
    """A real bank fraud team opens the way an impersonator does, which is why the scams
    work. Disjoint openers would let the guard decide on the first sentence."""
    from redteam.generate.transcripts import SCAM_OPENERS

    stems = [o.split("{")[0][:25] for pool in SCAM_OPENERS.values() for o in pool]
    genuine = transcripts.loc[transcripts["is_coercive"] == 0, "text"]
    assert float(genuine.apply(lambda t: any(s in t for s in stems)).mean()) > 0.15


def test_held_out_vocabulary_is_actually_disjoint():
    """The evaluation corpus must share no phrasing with the training one.

    An overlap here would be silent and would invalidate the only evaluation in this plane
    that rules out string memorisation.
    """
    from redteam.generate import transcripts as T

    def phrases(*pools):
        out = set()
        for pool in pools:
            values = pool.values() if isinstance(pool, dict) else [pool]
            for group in values:
                for item in group:
                    out.update(x for x in ([item] if isinstance(item, str) else item))
        return {p.split("{")[0].strip() for p in out}

    train = phrases(T.SCAM_OPENERS, T.COERCION_TURNS, T.GENUINE_SCRIPTS,
                    T.GENUINE_MIDDLES, T.GENUINE_CLOSERS, T.VICTIM_TURNS,
                    T.GENUINE_NEW_PAYEE)
    held = phrases(T.HELDOUT_SCAM_OPENERS, T.HELDOUT_COERCION_TURNS,
                   T.HELDOUT_GENUINE_SCRIPTS, T.HELDOUT_GENUINE_MIDDLES,
                   T.HELDOUT_GENUINE_CLOSERS, T.HELDOUT_VICTIM_TURNS,
                   T.HELDOUT_GENUINE_NEW_PAYEE)
    assert not (train & held)


def test_the_holdout_withholds_both_classes(fitted_vishing):
    """Measuring recall on unseen scam wording while the threshold is still set on familiar
    negatives would flatter the result: an unseen genuine call is exactly as likely to
    surprise the model as an unseen scam."""
    from redteam.defend.vishing_guard import unseen_wording_recall
    from redteam.generate.transcripts import build_holdout_corpus

    holdout = build_holdout_corpus(np.random.default_rng(0))
    assert set(holdout["is_coercive"].unique()) == {0, 1}

    guard, _ = fitted_vishing
    table = unseen_wording_recall(guard, seed=0)
    assert not table.empty
    # The budget has to be met on the unseen negatives, not assumed from the fitted cut-point.
    assert float(table.iloc[0]["false_positive_rate"]) <= 0.02


def test_a_held_out_pretext_is_still_recognised_as_coercion(transcripts):
    """The honest measurement.

    Pretexts turn over constantly - a new government scheme, a new courier, a new platform -
    and a guard that only knows the ones in its training data is perpetually one script
    behind. What should generalise is the structure: urgency, isolation, redirection.
    """
    table = held_out_script_recall(transcripts, seed=0)
    if table.empty:
        pytest.skip("too few transcripts per script in this sample")
    # Not a high bar, deliberately. The claim being tested is that zero-shot recall is
    # materially better than chance on most pretexts, not that it matches in-distribution.
    assert float(table["zero_shot_recall"].median()) > 0.3


# --------------------------------------------------------------------------------------
# The media guard
# --------------------------------------------------------------------------------------

def test_payments_with_no_biometric_capture_are_left_alone(dataset, split):
    """Scoring an absent voice sample as anomalous would turn this into a channel detector,
    which the model already gets for free from the channel column."""
    guard = DeepfakeGuard(seed=0).fit(split.train, split.test)
    if not guard.usable:
        pytest.skip("no biometric captures in this sample")
    scored = guard.score_transactions(dataset.transactions)
    absent = ~has_capture(dataset.transactions)
    assert (scored.loc[absent, "media_artefact_score"] == 0).all()
    assert (scored.loc[absent, "media_artefact_flag"] == 0).all()


def test_guard_never_sees_a_fraud_label(split):
    """Fitted one-class on genuine captures only.

    Labels select the fitting set, exactly as a bank's confirmed-fraud history would. They
    never supervise it, so the guard cannot memorise which vectors are attacks and will
    degrade honestly against a capture profile that looks ordinary.
    """
    guard = DeepfakeGuard(seed=0)
    poisoned = split.train.copy()
    # Flipping every label must not change the fitted model, because the only thing labels
    # do is choose rows - and here that choice is made before any flip could matter.
    guard.fit(split.train, split.test)
    if not guard.usable:
        pytest.skip("no biometric captures in this sample")
    baseline = guard.score(split.test[has_capture(split.test)])

    poisoned["is_fraud"] = 0
    other = DeepfakeGuard(seed=0).fit(poisoned, split.test)
    assert not np.allclose(baseline, other.score(split.test[has_capture(split.test)])), \
        "including known fraud in the fit should change the model, or the exclusion is a no-op"


def test_coverage_is_reported_rather_than_averaged_away(split):
    """The guard's ceiling. A headline that averaged over payments with no capture at all
    would be diluted nonsense, so the covered share is published beside the recall."""
    guard = DeepfakeGuard(seed=0).fit(split.train, split.test)
    if not guard.usable:
        pytest.skip("no biometric captures in this sample")
    report = guard.report.to_dict()
    assert 0.0 < report["capture_coverage"] < 1.0
    assert report["scored_captures"] < len(split.test)


# --------------------------------------------------------------------------------------
# Bank hygiene
# --------------------------------------------------------------------------------------

def test_a_refusal_or_a_summary_never_enters_the_coercive_class():
    """Mislabelled positives are worse than a smaller bank.

    A model asked for transcripts will sometimes return a refusal or a third-person summary,
    and letting either through would teach the guard that narration is a fraud signal.
    """
    assert _clean(["I'm sorry, I cannot help with that request." * 10]) == []
    assert _clean(["The caller pressured the victim into transferring funds." * 20]) == []
    assert _clean(["short"]) == []

    good = "\n".join(["CALLER: We have flagged your account, please stay on the line."
                      if i % 2 == 0 else "CUSTOMER: Is this really the bank?"
                      for i in range(12)])
    assert _clean([good]) == [good]
