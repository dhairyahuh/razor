"""Defence tests, weighted toward the failure modes that produce flattering nonsense.

Most of these check a boundary rather than a score: what the model is allowed to read,
where the split falls, whether a threshold honours its budget. A model that quietly reads
``attack_vector_id`` posts a perfect AUC, and no accuracy assertion would catch it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from sklearn.metrics import roc_auc_score

from redteam.defend.dataset import (
    MatrixBuilder,
    column_layers,
    input_columns,
    temporal_split,
)
from redteam.defend.agent_controls import (
    apply_controls,
    ceiling_loss_bound,
    control_coverage,
    demo_stack,
    sign_request,
)
from redteam.defend.intent_guard import (
    IntentGuard,
    _b64u,
    _b64u_decode,
    apply_guard,
    generate_keypair,
    issue_intent,
)
from redteam.defend.stats import bootstrap_metric, partial_auc, wilson_interval
from redteam.defend.thresholds import threshold_for_budget
from redteam.features import build_features, build_graph_features
from redteam.schema import LABEL_COLUMNS, META_COLUMNS


# --------------------------------------------------------------------------------------
# Splitting and matrix construction
# --------------------------------------------------------------------------------------

def test_temporal_split_is_ordered_and_disjoint(transactions, tiny_config):
    split = temporal_split(transactions, test_days=tiny_config.defence.test_days,
                           calibration_days=tiny_config.defence.calibration_days)
    assert split.train["timestamp"].max() < split.calibration["timestamp"].min()
    assert split.calibration["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.train) + len(split.calibration) + len(split.test) == len(transactions)
    assert set(split.train["txn_id"]).isdisjoint(split.test["txn_id"])
    for part in (split.train, split.calibration, split.test):
        assert int(part["is_fraud"].sum()) > 0


def test_model_inputs_exclude_labels_meta_and_scratch(transactions):
    featured = build_features(transactions)
    cols = set(input_columns(featured))
    assert cols.isdisjoint(LABEL_COLUMNS)
    assert cols.isdisjoint(META_COLUMNS)
    assert not any(c.startswith("_") for c in cols)
    # The guards' outputs are inputs; their free-text explanation is not.
    assert "intent_guard_reasons" not in cols
    assert any(c.startswith("f_") for c in cols)


def test_matrix_builder_maps_unseen_categories_to_missing():
    train = pd.DataFrame({"rail": ["UPI_P2P", "CARD_CNP"], "amount": [10.0, 20.0]})
    builder = MatrixBuilder(["rail", "amount"]).fit(train)

    out = builder.transform(pd.DataFrame({"rail": ["UPI_P2P", "NEW_RAIL"], "amount": [1.0, 2.0]}))
    assert not np.isnan(out[0, 0])
    # An unseen PSP is genuinely unknown, not category zero.
    assert np.isnan(out[1, 0])
    assert out[:, 1].tolist() == [1.0, 2.0]


def test_matrix_builder_column_order_is_stable():
    cols = ["amount", "rail", "hour"]
    builder = MatrixBuilder(cols).fit(pd.DataFrame({"amount": [1.0], "rail": ["UPI_P2P"], "hour": [3]}))
    frame = pd.DataFrame({"hour": [7], "rail": ["UPI_P2P"], "amount": [9.0]})
    out = builder.transform(frame)
    assert out[0, 0] == 9.0 and out[0, 2] == 7.0


# --------------------------------------------------------------------------------------
# Feature causality
# --------------------------------------------------------------------------------------

def test_ablation_layers_are_nested_and_complete(transactions):
    """The ablation grid is only interpretable if its rows are strictly cumulative.

    If a layer dropped a column an earlier layer had, `delta_recall` would no longer be the
    value that layer added, and the most explanatory table in the report would be quietly
    wrong. Also asserts the final layer is the model's real input set, so the bottom row of
    the grid is the headline model rather than a near-miss.
    """
    frame = build_graph_features(build_features(transactions))
    layers = column_layers(frame)

    names = list(layers)
    assert names == ["raw_schema", "+ tabular", "+ graph", "+ guards (full)"]
    for earlier, later in zip(names, names[1:]):
        assert set(layers[earlier]) < set(layers[later]), (earlier, later)

    assert set(layers["+ guards (full)"]) == set(input_columns(frame))
    # Each layer must contribute something, or the grid has a row that says nothing.
    assert layers["raw_schema"]
    assert any(c.startswith("f_") for c in layers["+ tabular"])
    assert any(c.startswith("g_") for c in layers["+ graph"])


def test_no_guard_output_is_the_label_in_disguise(tiny_config, transactions, corpus,
                                                  transcripts):
    """The check that would have caught a 100% headline before it was published.

    A guard column earns its place by classifying content. It stops being a feature and
    becomes a label in two ways, and both have happened here: the score separates perfectly
    because one author wrote both classes of the corpus behind it, or the telemetry it reads
    is only ever attached to fraud. The second is invisible to every fidelity probe, because
    they zero-fill guard outputs by design.

    The deterministic controls are exempt and the exemption is principled: an intent block
    requires a presented artefact to contradict the settlement request, which legitimate
    traffic cannot do, so `presence_lift` is supposed to be maximal there. The bound applies
    to the content guards, whose job is to be informative rather than decisive.
    """
    from redteam.defend.evaluate import guard_leakage
    from redteam.defend.pipeline import prepare

    prepared = prepare(transactions, corpus, tiny_config, transcripts=transcripts)
    table = guard_leakage(prepared.featured)
    if table.empty:
        pytest.skip("no guard columns varied in this sample")

    deterministic = {"intent_guard_blocked", "intent_guard_stepup",
                     "intent_guard_violation_count", "agent_control_blocked",
                     "agent_control_stepup"}
    content = table[~table["guard_column"].isin(deterministic)]
    for row in content.to_dict("records"):
        assert row["single_feature_auc"] < 0.95, (
            f"{row['guard_column']} separates fraud on its own at AUC "
            f"{row['single_feature_auc']}; it is a label, not a feature"
        )
        assert row["p_fraud_given_present"] < 0.75, (
            f"{row['guard_column']} is populated on fraud {row['p_fraud_given_present']:.0%} "
            f"of the time it is populated at all (base rate {row['base_fraud_rate']}); the "
            "model can read the presence of the telemetry instead of its content"
        )


def test_features_do_not_depend_on_the_future(transactions):
    """Truncating the stream must not change the features of the rows that remain.

    This is the property that makes an offline number transferable to production, where
    the future has not happened yet.
    """
    cutoff = int(len(transactions) * 0.7)
    full = build_features(transactions)
    truncated = build_features(transactions.iloc[:cutoff].copy())

    feature_cols = [c for c in full.columns if c.startswith("f_")]
    assert feature_cols
    a = full.iloc[:cutoff][feature_cols].reset_index(drop=True)
    b = truncated[feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-9)


def test_graph_features_do_not_depend_on_the_future(transactions):
    cutoff = int(len(transactions) * 0.7)
    full = build_graph_features(build_features(transactions))
    truncated = build_graph_features(build_features(transactions.iloc[:cutoff].copy()))

    graph_cols = [c for c in full.columns if c.startswith("g_")]
    assert graph_cols
    a = full.iloc[:cutoff][graph_cols].reset_index(drop=True)
    b = truncated[graph_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-9)


# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------

def test_threshold_honours_the_false_positive_budget():
    negatives = np.linspace(0, 1, 1000)
    thr = threshold_for_budget(negatives, target_fpr=0.01)
    assert (negatives >= thr).mean() <= 0.01


def test_threshold_survives_scores_piled_up_at_zero():
    """Guard scores are mostly exact zeros; a naive quantile would return zero itself."""
    negatives = np.concatenate([np.zeros(900), np.linspace(0.5, 1.0, 100)])
    thr = threshold_for_budget(negatives, target_fpr=0.01)
    assert thr > 0.0
    assert (negatives >= thr).mean() <= 0.01


def test_threshold_on_empty_input_is_defined():
    assert threshold_for_budget(np.array([]), target_fpr=0.01) == 0.5


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def test_wilson_interval_stays_inside_the_unit_range_on_thin_cells():
    """The reason for Wilson over the normal approximation.

    Three successes out of three is exactly the cell the per-vector table keeps producing,
    and the textbook interval collapses to zero width there - reporting perfect certainty
    from three observations. Wilson stays wide and stays in bounds.
    """
    lo, hi = wilson_interval(3, 3)
    assert 0.0 <= lo < hi <= 1.0
    assert lo < 0.45, "an interval this narrow on three rows would be a false assurance"

    for successes, trials in ((0, 1), (0, 7), (1, 1), (5, 5), (0, 400), (400, 400)):
        lo, hi = wilson_interval(successes, trials)
        assert 0.0 <= lo <= hi <= 1.0, (successes, trials, lo, hi)

    # More evidence must narrow the interval.
    assert (wilson_interval(400, 800)[1] - wilson_interval(400, 800)[0]) < (
        wilson_interval(5, 10)[1] - wilson_interval(5, 10)[0]
    )


def test_wilson_interval_covers_the_true_rate():
    """Empirical coverage near the nominal 95%, at a sample size the report actually uses."""
    rng = np.random.default_rng(0)
    truth, trials = 0.8, 25
    draws = rng.binomial(trials, truth, 4000)
    covered = [lo <= truth <= hi for lo, hi in (wilson_interval(int(k), trials) for k in draws)]
    assert 0.92 < float(np.mean(covered)) < 0.99, float(np.mean(covered))


def test_partial_auc_ignores_the_region_nobody_deploys():
    """Two models identical where it matters, different where it does not, must tie.

    This is the whole argument for reporting partial AUC: the full ROC AUC rewards ranking
    that only pays off at false-positive rates an operations team would never fund.
    """
    rng = np.random.default_rng(1)
    y = np.zeros(4000, dtype=int)
    y[:80] = 1
    # Model A separates the top of the ranking cleanly; both then rank the remaining
    # negatives differently, which only affects the far end of the ROC curve.
    scores_a = np.concatenate([rng.uniform(0.9, 1.0, 80), rng.uniform(0.0, 0.5, 3920)])
    scores_b = scores_a.copy()
    tail = np.arange(80, 4000)
    scores_b[tail] = rng.uniform(0.0, 0.5, len(tail))

    assert partial_auc(y, scores_a, max_fpr=0.02) == pytest.approx(
        partial_auc(y, scores_b, max_fpr=0.02), abs=0.02
    )
    # A random ranking must read at the 0.5 floor, not shrink toward zero with max_fpr.
    assert partial_auc(y, rng.random(4000), max_fpr=0.02) == pytest.approx(0.5, abs=0.1)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.02).astype(int)
    scores = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 3000), 0, 1)
    out = bootstrap_metric(y, scores, lambda yy, ss: float(roc_auc_score(yy, ss)),
                           n_resamples=200, seed=0)
    assert out["lo95"] <= out["point"] <= out["hi95"]
    assert out["resamples"] > 0


# --------------------------------------------------------------------------------------
# Intent guard
# --------------------------------------------------------------------------------------

WALLET_KEY, WALLET_PUB = generate_keypair(seed=b"test-wallet-seed")
OTHER_KEY, OTHER_PUB = generate_keypair(seed=b"a-different-wallet")


def _intent(**kwargs):
    defaults = dict(user_id="U1", payee_id="PAYEE-A", max_amount=1000.0, now=1_000.0)
    defaults.update(kwargs)
    return issue_intent(WALLET_KEY, **defaults)


def test_intent_guard_allows_a_faithful_settlement():
    artifact, sig = _intent()
    decision = IntentGuard(WALLET_PUB).verify(artifact, sig, settle_payee="PAYEE-A",
                                              settle_amount=1000.0, settle_currency="INR",
                                              now=1_100.0)
    assert decision.allowed
    assert decision.reason_string == "ok"


def test_intent_guard_catches_a_hijacked_payee():
    """The signature is valid and the payee changed: the fingerprint of prompt injection."""
    artifact, sig = _intent()
    decision = IntentGuard(WALLET_PUB).verify(artifact, sig, settle_payee="MULE-9",
                                              settle_amount=1000.0, settle_currency="INR",
                                              now=1_100.0)
    assert not decision.allowed
    assert "PAYEE_MISMATCH" in decision.reasons


def test_intent_guard_rejects_replay_of_a_spent_nonce():
    artifact, sig = _intent()
    guard = IntentGuard(WALLET_PUB)
    first = guard.verify(artifact, sig, settle_payee="PAYEE-A", settle_amount=100.0,
                         settle_currency="INR", now=1_100.0)
    second = guard.verify(artifact, sig, settle_payee="PAYEE-A", settle_amount=100.0,
                          settle_currency="INR", now=1_200.0)
    assert first.allowed
    assert not second.allowed and "INTENT_REPLAYED" in second.reasons


def test_intent_guard_rejects_a_forged_signature():
    artifact, _ = _intent()
    decision = IntentGuard(WALLET_PUB).verify(artifact, "deadbeef", settle_payee="PAYEE-A",
                                              settle_amount=100.0, settle_currency="INR",
                                              now=1_100.0)
    assert decision.reasons == ["INTENT_SIGNATURE_INVALID"]


def test_intent_guard_rejects_a_key_it_did_not_issue():
    artifact, sig = _intent()
    decision = IntentGuard(OTHER_PUB).verify(artifact, sig, settle_payee="PAYEE-A",
                                             settle_amount=100.0, settle_currency="INR",
                                             now=1_100.0)
    assert decision.reasons == ["INTENT_SIGNATURE_INVALID"]


def test_an_unknown_signer_fails_closed():
    """A key id the registry has never seen is a failed signature, not an unchecked one.

    Resolving a missing key to "skip verification" is the standard way a signature check
    stops running in production without anyone noticing, because every legitimate payment
    keeps working.
    """
    artifact, sig = _intent()
    guard = IntentGuard({"somebody-else": OTHER_PUB})
    decision = guard.verify(artifact, sig, settle_payee="PAYEE-A", settle_amount=100.0,
                            settle_currency="INR", now=1_100.0)
    assert decision.reasons == ["INTENT_SIGNATURE_INVALID"]


def test_the_verifier_cannot_mint_an_artefact():
    """The reason for asymmetric signing, asserted rather than described in a docstring.

    Under a liability-splitting reimbursement regime both PSPs have an interest in what the
    user is held to have instructed. With a shared secret, a verifier could produce an
    artefact indistinguishable from the wallet's, so the artefact would prove nothing in the
    dispute it exists to settle. Here the guard holds public keys only, and there is no
    method on it that returns a signature.
    """
    guard = IntentGuard({"U1": WALLET_PUB})
    assert not hasattr(guard, "sign")
    assert all(isinstance(k, Ed25519PublicKey) for k in guard.public_keys.values())

    # And the public key genuinely cannot sign: no private-key material reachable from it.
    pub = guard.public_keys["U1"]
    assert not hasattr(pub, "sign")


def test_the_signed_header_pins_the_algorithm():
    """Algorithm confusion is how JWT implementations are usually broken.

    The protected header is inside the signing input, so an attacker who rewrites ``alg``
    to something forgeable changes the bytes that were signed and the signature fails.
    """
    artifact, sig = _intent()
    signing_input = artifact.signing_input().decode("ascii")
    header_b64 = signing_input.split(".")[0]
    header = json.loads(_b64u_decode(header_b64))
    assert header["alg"] == "EdDSA"
    assert header["kid"] == "U1"

    # Swapping the header for an unsigned-JWT style "alg": "none" invalidates the signature.
    forged = _b64u(json.dumps({"alg": "none", "typ": "intent+jws", "kid": "U1"},
                              sort_keys=True, separators=(",", ":")).encode())
    tampered = f"{forged}.{signing_input.split('.')[1]}".encode("ascii")
    with pytest.raises(Exception):
        WALLET_PUB.verify(_b64u_decode(sig), tampered)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(now=1_000_000.0), "INTENT_EXPIRED"),
        (dict(settle_amount=5_000.0), "AMOUNT_EXCEEDS_INTENT"),
        (dict(settle_currency="USD"), "CURRENCY_MISMATCH"),
        (dict(requested_scope=("purchase", "transfer")), "SCOPE_VIOLATION"),
    ],
)
def test_intent_guard_reason_codes(kwargs, expected):
    artifact, sig = _intent()
    call = dict(settle_payee="PAYEE-A", settle_amount=100.0, settle_currency="INR", now=1_100.0)
    call.update(kwargs)
    decision = IntentGuard(WALLET_PUB).verify(artifact, sig, **call)
    assert not decision.allowed
    assert expected in decision.reasons


def test_intent_guard_tolerates_ordinary_price_drift():
    """Tax, shipping and FX move the settled amount; a guard that blocks them is unusable."""
    artifact, sig = _intent(max_amount=1000.0)
    decision = IntentGuard(WALLET_PUB).verify(artifact, sig, settle_payee="PAYEE-A",
                                              settle_amount=1_040.0, settle_currency="INR",
                                              now=1_100.0)
    assert decision.allowed


def test_intent_guard_never_blocks_a_non_agentic_payment(transactions):
    guarded = apply_guard(transactions)
    human = guarded[guarded["initiated_by_agent"] == 0]
    assert int(human["intent_guard_blocked"].sum()) == 0
    assert (human["intent_guard_reasons"] == "").all()


def test_intent_guard_never_declines_a_legitimate_agent_payment(transactions):
    """A hard block is a provable contradiction, so it has no false positives at all.

    Every declining reason code compares a presented artefact against the settlement
    request. Honest agent traffic cannot trip one, and if it ever does the control has
    stopped being defensible in a dispute.
    """
    guarded = apply_guard(transactions)
    agentic = guarded[guarded["initiated_by_agent"] == 1]
    legit = agentic[agentic["is_fraud"] == 0]
    assert int(legit["intent_guard_blocked"].sum()) == 0

    # The guard also has to actually fire on the attacks it exists for, or the zero above is
    # satisfied by a control that never does anything. Asserted on the upper confidence bound
    # rather than the point estimate: the fixture carries a few dozen agentic fraud rows, so a
    # bare "> 0.3" fails whenever the attack mix shifts and a couple of blocks move - which is
    # a property of the sample, not of the guard.
    blocked = agentic.loc[agentic["is_fraud"] == 1, "intent_guard_blocked"]
    assert len(blocked) > 0
    _, upper = wilson_interval(int(blocked.sum()), len(blocked))
    assert upper > 0.3, (int(blocked.sum()), len(blocked), upper)


def test_no_hard_block_survives_a_synthetic_benign_tail():
    """The zero above has to hold at scale, not just on a fixture too small to reach the tail.

    Asserting an exact zero on a few thousand rows is a false negative waiting to happen: the
    benign amount-drift distribution has a tail, and it only crosses a fixed tolerance a
    couple of times in ten thousand agent payments. The fixture never got there, so the test
    passed while the committed run's own coverage CSV showed legitimate rows being declined.
    This drives the tail directly.
    """
    n = 200_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "initiated_by_agent": np.ones(n, dtype=int),
        "intent_token_present": np.ones(n, dtype=int),
        "intent_signature_valid": np.ones(n, dtype=int),
        "intent_payee_match": np.ones(n, dtype=int),
        "intent_age_s": rng.gamma(2.0, 22.0, n),
        # The real benign draw, which is what produced the false declines.
        "intent_amount_delta_ratio": np.abs(rng.normal(0.008, 0.012, n)),
        "spt_reuse_count": np.ones(n, dtype=int),
        "spt_scope_violation": np.zeros(n, dtype=int),
    })
    guarded = apply_guard(df)
    assert int(guarded["intent_guard_blocked"].sum()) == 0
    # The tail still has to be *seen* - it becomes friction rather than vanishing.
    assert int(guarded["intent_guard_stepup"].sum()) > 0


def test_a_missing_artefact_cannot_be_hard_blocked_for_violating_it():
    """You cannot prove a settlement contradicts an intent that was never presented.

    Replay, amount and scope checks were evaluated without the "artefact was presented" gate
    that the signature, expiry and payee checks had, so a row carrying nothing at all could be
    declined outright - the exact outcome the two-verdict split exists to prevent.
    """
    df = pd.DataFrame({
        "initiated_by_agent": [1, 1, 1],
        "intent_token_present": [0, 0, 0],
        "intent_signature_valid": [0, 0, 0],
        "intent_payee_match": [0, 0, 0],
        "intent_age_s": [10_000.0, 1.0, 1.0],
        "intent_amount_delta_ratio": [0.0, 0.95, 0.0],
        "spt_reuse_count": [1, 1, 7],
        "spt_scope_violation": [0, 0, 1],
    })
    guarded = apply_guard(df)
    assert int(guarded["intent_guard_blocked"].sum()) == 0
    assert (guarded["intent_guard_reasons"] == "INTENT_ABSENT").all()


def test_missing_intent_raises_friction_rather_than_declining(transactions):
    """Agents predating the protocol present no artefact; declining them all is not viable."""
    guarded = apply_guard(transactions)
    legit_agentic = guarded[(guarded["initiated_by_agent"] == 1) & (guarded["is_fraud"] == 0)]
    stepped_up = legit_agentic[legit_agentic["intent_guard_stepup"] == 1]
    assert not stepped_up.empty
    assert (stepped_up["intent_guard_reasons"] == "INTENT_ABSENT").all()
    assert int(stepped_up["intent_guard_blocked"].sum()) == 0


# --------------------------------------------------------------------------------------
# Agent identity, token binding and spend ceilings
# --------------------------------------------------------------------------------------

def test_an_enrolled_agent_can_prove_its_identity():
    registry, _, private, agent = demo_stack()
    request = sign_request(private, agent_id=agent.agent_id, token_thumbprint="abc",
                          created=1_000.0)
    assert registry.verify(request, now=1_010.0) == []


def test_an_impersonator_cannot_sign_for_a_name_it_copied():
    """The whole of Web Bot Auth in one assertion.

    Spoofing a trusted crawler means sending its user-agent string, which is free. The
    request signature is not free: it needs the private key the identity was enrolled with,
    and copying a name does not copy a key.
    """
    registry, _, _, agent = demo_stack()
    impostor, _ = generate_keypair(seed=b"impostor")
    forged = sign_request(impostor, agent_id=agent.agent_id, token_thumbprint="abc",
                          created=1_000.0)
    assert registry.verify(forged, now=1_010.0) == ["AGENT_SIGNATURE_INVALID"]


def test_an_unenrolled_name_is_reported_separately_from_a_bad_signature():
    """Two different facts, and conflating them is what made the control unusable.

    An agent that never enrolled is a legacy integration and gets no trusted lane. An agent
    that asserts an enrolled identity and fails to sign for it is an impostor. The first is
    six percent of honest agent traffic in the generator, so treating them alike would
    decline one legitimate agent payment in twenty.
    """
    registry, _, private, _ = demo_stack()
    request = sign_request(private, agent_id="never-enrolled", token_thumbprint="abc",
                           created=1_000.0)
    assert registry.verify(request, now=1_010.0) == ["AGENT_NOT_ENROLLED"]


def test_a_stale_signature_is_refused():
    registry, _, private, agent = demo_stack()
    request = sign_request(private, agent_id=agent.agent_id, created=1_000.0)
    assert registry.verify(request, now=1_000.0 + 3_600) == ["AGENT_SIGNATURE_STALE"]


def test_replaying_a_signed_request_fails_the_second_time():
    registry, _, private, agent = demo_stack()
    request = sign_request(private, agent_id=agent.agent_id, created=1_000.0)
    assert registry.verify(request, now=1_010.0) == []
    assert registry.verify(request, now=1_020.0) == ["AGENT_REQUEST_REPLAYED"]


def test_the_covered_components_are_inside_the_signature():
    """A signature over a request whose fields can still be edited is decoration.

    Rewriting the agent id after signing has to invalidate the signature, or an attacker
    could take any valid signature and reattach it to a request naming a trusted agent.
    """
    registry, _, private, agent = demo_stack()
    request = sign_request(private, agent_id=agent.agent_id, token_thumbprint="abc",
                           created=1_000.0)
    tampered = replace(request, path="/settle-to-mule")
    assert registry.verify(tampered, now=1_010.0) == ["AGENT_SIGNATURE_INVALID"]


def test_a_bound_token_is_worthless_to_a_thief():
    """Proof of possession, which is the counter to a stolen or replayed payment token.

    The token is a bearer credential right up until it carries the thumbprint of the key it
    was issued to. After that, stealing it is not enough - the thief needs the private key
    it was bound to, which never left the holder's device.
    """
    _, vault, private, _ = demo_stack()
    holder = private.public_key()
    thief, thief_pub = generate_keypair(seed=b"thief")

    token = vault.mint(user_id="U1", holder=holder, max_amount=5_000.0, now=1_000.0)
    assert vault.redeem(token, presenter=thief_pub, amount=100.0, now=1_010.0) == [
        "TOKEN_NOT_BOUND_TO_PRESENTER"
    ]
    assert vault.redeem(token, presenter=holder, amount=100.0, now=1_010.0) == []


def test_a_single_use_token_cannot_be_spent_twice():
    _, vault, private, _ = demo_stack()
    holder = private.public_key()
    token = vault.mint(user_id="U1", holder=holder, max_amount=5_000.0, now=1_000.0)
    assert vault.redeem(token, presenter=holder, amount=100.0, now=1_010.0) == []
    assert vault.redeem(token, presenter=holder, amount=100.0, now=1_020.0) == [
        "TOKEN_ALREADY_SPENT"
    ]


def test_the_ceiling_bounds_the_loss_when_everything_else_has_failed():
    """The control that assumes the other two are already defeated.

    A per-transaction ceiling does not detect anything. It decides what a compromised
    credential is worth, which is the only question left once the agent is hijacked and
    the token is genuinely held.
    """
    _, vault, private, _ = demo_stack()
    holder = private.public_key()
    token = vault.mint(user_id="U1", holder=holder, max_amount=5_000.0, now=1_000.0)
    assert vault.redeem(token, presenter=holder, amount=250_000.0, now=1_010.0) == [
        "TOKEN_CEILING_EXCEEDED"
    ]


def test_the_deterministic_controls_have_no_false_declines(transactions):
    """Blocking is reserved for contradictions of a credential the request itself presented.

    A population-quantile amount cap was in this set and had to come out: it declined one
    legitimate agent payment in a hundred, by construction, and a large payment contradicts
    nothing. What remains - a single-use token spent twice, a token spent outside its scope -
    cannot occur on honest traffic at all.
    """
    controlled = apply_controls(transactions)
    legit_agentic = controlled[(controlled["initiated_by_agent"] == 1)
                               & (controlled["is_fraud"] == 0)]
    assert not legit_agentic.empty
    assert int(legit_agentic["agent_control_blocked"].sum()) == 0


def test_the_ceiling_is_reported_as_exposure_not_as_detection(transactions):
    """It prevents no fraud, so it must not be counted as though it did.

    The honest claim for a spend cap is that it bounds the loss per compromised credential.
    That is a value figure, and reporting it as a detection rate would be claiming the
    attack was stopped when it was only made cheaper.
    """
    controlled = apply_controls(transactions)
    bound = ceiling_loss_bound(controlled)
    if bound.empty:
        pytest.skip("fixture has too little agentic traffic to size a ceiling")

    row = bound.iloc[0]
    assert row["loss_scoped_token"] <= row["loss_standing_delegation"]
    assert row["loss_reduction"] >= 0
    # And the cost side is published beside the benefit, rather than left implicit.
    assert 0.0 <= row["legit_payments_capped_pct"] <= 2.0


def test_controls_do_not_step_up_honest_enrolled_agents(transactions):
    """The false-positive side, which is where a friction control actually gets judged.

    A signed request from an enrolled agent must pass all three controls untouched. The
    stepped-up remainder is the legacy tail and the verification failures, both of which
    are supposed to be there.
    """
    controlled = apply_controls(transactions)
    agentic = controlled[controlled["initiated_by_agent"] == 1]
    proved = agentic[(agentic["agent_identity_asserted"] == 1)
                     & (agentic["agent_request_signature_valid"] == 1)]
    assert not proved.empty
    assert not proved["agent_control_reasons"].str.contains("AGENT_IDENTITY_UNPROVEN").any()
    assert not proved["agent_control_reasons"].str.contains("AGENT_UNTRUSTED_LANE").any()


def test_identity_verification_catches_the_impersonated_crawler(transactions):
    """The vector the intent guard reports at exactly 0%, which is why this control exists.

    An impersonator asserts an enrolled identity and cannot sign for it. Nothing about its
    intent chain is wrong, because it never presented one, so intent verification has
    nothing to compare and the vector walks straight past it.
    """
    controlled = apply_controls(transactions)
    coverage = control_coverage(controlled).set_index("attack_vector_id")
    if "AGENTIC-IMPERSONATED-CRAWLER" not in coverage.index:
        pytest.skip("fixture drew no impersonated-crawler rows in this window")

    row = coverage.loc["AGENTIC-IMPERSONATED-CRAWLER"]
    assert row["identity_unproven_rate"] > 0.5, row.to_dict()
    legit = coverage.loc["(legitimate)"]
    assert legit["identity_unproven_rate"] < 0.05, legit.to_dict()
