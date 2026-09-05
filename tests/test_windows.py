"""Window aggregates are checked against brute force, because leakage hides in off-by-ones.

Each vectorised routine is compared to the naive definition on random input. A trailing
window that accidentally includes the current row leaks the label for every burst-shaped
attack in the library, and the resulting AUC would look excellent.
"""

from __future__ import annotations

import numpy as np
import pytest

from redteam.windows import (
    cross_window_sum,
    expanding_mean_std,
    first_seen_and_rank,
    previous_value,
    window_sum_count,
    window_unique,
)

WINDOW = 100


EPOCH = 1_777_000_000  # a plausible unix second, as the production callers pass


@pytest.fixture
def stream():
    rng = np.random.default_rng(3)
    n = 400
    codes = rng.integers(0, 7, n).astype(np.int64)
    ts = (EPOCH + np.sort(rng.integers(0, 1_000, n))).astype(np.int64)
    values = rng.uniform(1, 500, n)
    return codes, ts, values


def _prior_mask(codes, ts, window, i):
    """Same key, arrived before row ``i``, inside the window.

    The stream is in time order, so "arrived before" is position order. Events sharing a
    timestamp are ordered by arrival, which is what a real feature store sees; the only
    hard requirement is that a row never counts itself.
    """
    return (codes == codes[i]) & (np.arange(len(codes)) < i) & (ts >= ts[i] - window)


def test_window_sum_count_excludes_the_current_row(stream):
    codes, ts, values = stream
    got_sum, got_cnt = window_sum_count(codes, ts, values, WINDOW)
    for i in range(len(codes)):
        prior = _prior_mask(codes, ts, WINDOW, i)
        assert got_sum[i] == pytest.approx(values[prior].sum(), abs=1e-6), i
        assert got_cnt[i] == int(prior.sum()), i


def test_simultaneous_events_never_see_themselves():
    """A burst landing in the same second is the shape most likely to leak a label."""
    codes = np.array([0, 0, 0], dtype=np.int64)
    ts = np.array([EPOCH, EPOCH, EPOCH], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0])
    sums, counts = window_sum_count(codes, ts, values, WINDOW)
    assert counts.tolist() == [0.0, 1.0, 2.0]
    assert sums.tolist() == [0.0, 1.0, 3.0]


def test_window_reaching_past_the_epoch_does_not_cross_entities():
    """The lower bound is packed as ``t - window``; a negative there would borrow bits."""
    codes = np.array([0, 1, 1], dtype=np.int64)
    ts = np.array([5, 5, 6], dtype=np.int64)
    values = np.array([100.0, 2.0, 3.0])
    sums, counts = window_sum_count(codes, ts, values, window_s=86_400)
    assert sums.tolist() == [0.0, 0.0, 2.0]
    assert counts.tolist() == [0.0, 0.0, 1.0]


def test_expanding_mean_uses_only_prior_events(stream):
    codes, ts, values = stream
    mean, std, count = expanding_mean_std(codes, ts, values)
    for i in range(len(codes)):
        prior = values[(codes == codes[i]) & (np.arange(len(codes)) < i)]
        assert count[i] == len(prior)
        if len(prior) > 1:
            assert mean[i] == pytest.approx(prior.mean(), rel=1e-6)
            assert std[i] == pytest.approx(prior.std(), rel=1e-5, abs=1e-6)
        elif len(prior) == 1:
            assert mean[i] == pytest.approx(prior[0], rel=1e-6)
        else:
            assert np.isnan(mean[i])


def test_first_seen_and_rank(stream):
    codes, ts, _ = stream
    first_ts, rank = first_seen_and_rank(codes, ts)
    for i in range(len(codes)):
        same = codes == codes[i]
        assert first_ts[i] == ts[same].min()
        assert rank[i] == int(((codes == codes[i]) & (np.arange(len(codes)) < i)).sum())


def test_previous_value(stream):
    codes, ts, values = stream
    prev = previous_value(codes, ts, values, fill=-1.0)
    for i in range(len(codes)):
        earlier = np.where((codes == codes[i]) & (np.arange(len(codes)) < i))[0]
        assert prev[i] == (values[earlier[-1]] if earlier.size else -1.0)


def test_window_unique_matches_brute_force(stream):
    codes, ts, _ = stream
    rng = np.random.default_rng(4)
    secondary = rng.integers(0, 11, len(codes)).astype(np.int64)
    got = window_unique(codes, ts, secondary, WINDOW)
    for i in range(len(codes)):
        prior = _prior_mask(codes, ts, WINDOW, i)
        assert got[i] == len(set(secondary[prior].tolist())), i


def test_cross_window_sum_is_strictly_causal():
    """A source event at exactly the query time must not be counted."""
    src_codes = np.array([1, 1, 1], dtype=np.int64)
    src_ts = EPOCH + np.array([10, 20, 30], dtype=np.int64)
    src_values = np.array([1.0, 2.0, 4.0])
    query_codes = np.array([1, 1], dtype=np.int64)
    query_ts = EPOCH + np.array([20, 31], dtype=np.int64)

    got = cross_window_sum(src_codes, src_ts, src_values, query_codes, query_ts, window_s=100)
    assert got.tolist() == [1.0, 7.0]


def test_empty_input_is_handled():
    empty = np.array([], dtype=np.int64)
    assert window_sum_count(empty, empty, np.array([]), WINDOW)[0].size == 0
    assert window_unique(empty, empty, empty, WINDOW).size == 0
    assert cross_window_sum(empty, empty, np.array([]),
                            np.array([1], dtype=np.int64),
                            np.array([5], dtype=np.int64), WINDOW).tolist() == [0.0]
