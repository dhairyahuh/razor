"""Causal trailing-window aggregates over an event stream.

Every aggregate here answers "what had this entity done *before* this event", never
"including". That is not pedantry: a velocity feature that includes the row being scored
leaks the label for any burst-shaped attack, and it is the single most common way a fraud
model posts an impressive offline AUC and then fails in production.

The implementation packs ``(entity_code, timestamp)`` into one sorted int64 key. Because
all events for one entity form a contiguous ordered block, a global ``searchsorted`` for
``(code, t - window)`` cannot cross into a neighbouring entity, which turns per-group
Python loops into two vectorised binary searches.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TS_BITS = 34  # unix seconds (~1.8e9) sit well below 2**34


def codes_of(values: np.ndarray) -> np.ndarray:
    return pd.factorize(values, sort=False)[0].astype(np.int64)


def shared_codes(*arrays: np.ndarray) -> Tuple[Tuple[np.ndarray, ...], int]:
    """Factorise several id arrays into one shared code space."""
    lengths = [len(a) for a in arrays]
    combined = np.concatenate([np.asarray(a).astype(str) for a in arrays])
    codes = pd.factorize(combined, sort=False)[0].astype(np.int64)
    out, start = [], 0
    for n in lengths:
        out.append(codes[start: start + n])
        start += n
    return tuple(out), int(codes.max()) + 1 if len(codes) else 0


def pack(codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Pack ``(code, timestamp)`` into one sortable int64.

    Timestamps are floored at zero. Callers pack ``t - window`` to find a window's lower
    bound, and a negative timestamp would set the sign bit and borrow into the code field,
    silently returning another entity's events. Unix seconds never go negative, so the
    floor only ever fires on a window that reaches past the epoch, where "everything so
    far" is the right answer anyway.
    """
    return (codes.astype(np.int64) << TS_BITS) | np.maximum(ts.astype(np.int64), 0)


def first_seen_and_rank(codes: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First-seen timestamp and count of strictly prior occurrences, per key."""
    n = len(codes)
    if n == 0:
        return np.array([]), np.array([], dtype=np.int64)
    order = np.lexsort((ts, codes))
    sc, st = codes[order], ts[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sc[1:] != sc[:-1]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))

    first_ts = np.empty(n, dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    first_ts[order] = st[group_start]
    rank[order] = np.arange(n) - group_start
    return first_ts, rank


def window_sum_count(codes: np.ndarray, ts: np.ndarray, values: np.ndarray,
                     window_s: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-key sum and count over the trailing window, excluding the row itself."""
    n = len(codes)
    if n == 0:
        return np.array([]), np.array([])
    order = np.lexsort((ts, codes))
    key = pack(codes[order], ts[order])
    v = values[order]

    cumsum = np.concatenate([[0.0], np.cumsum(v)])
    pos = np.arange(n)
    lo = np.searchsorted(key, pack(codes[order], ts[order] - window_s), side="left")

    out_sum = np.empty(n)
    out_cnt = np.empty(n)
    out_sum[order] = cumsum[pos] - cumsum[lo]
    out_cnt[order] = (pos - lo).astype(float)
    return out_sum, out_cnt


def expanding_mean_std(codes: np.ndarray, ts: np.ndarray,
                       values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-key running mean, standard deviation and count over all strictly prior events."""
    n = len(codes)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.lexsort((ts, codes))
    c, v = codes[order], values[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = c[1:] != c[:-1]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))

    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    pos = np.arange(n)
    count = (pos - group_start).astype(float)
    total = cs[pos] - cs[group_start]
    total2 = cs2[pos] - cs2[group_start]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        var = np.where(count > 1, (total2 / np.maximum(count, 1)) - mean ** 2, np.nan)
        std = np.sqrt(np.maximum(var, 0))

    out_mean = np.empty(n)
    out_std = np.empty(n)
    out_cnt = np.empty(n)
    out_mean[order] = mean
    out_std[order] = std
    out_cnt[order] = count
    return out_mean, out_std, out_cnt


def previous_value(codes: np.ndarray, ts: np.ndarray, values: np.ndarray,
                   fill: float = np.nan) -> np.ndarray:
    """The key's immediately preceding value, or ``fill`` for its first event."""
    n = len(codes)
    if n == 0:
        return np.array([])
    order = np.lexsort((ts, codes))
    c, v = codes[order], values[order]
    prev = np.empty(n)
    prev[0] = fill
    prev[1:] = np.where(c[1:] == c[:-1], v[:-1], fill)
    out = np.empty(n)
    out[order] = prev
    return out


def cross_window_sum(src_codes: np.ndarray, src_ts: np.ndarray, src_values: np.ndarray,
                     query_codes: np.ndarray, query_ts: np.ndarray,
                     window_s: int) -> np.ndarray:
    """Sum of source events for ``query_codes`` in the window ending strictly before ``query_ts``."""
    if len(src_codes) == 0:
        return np.zeros(len(query_codes))
    order = np.argsort(pack(src_codes, src_ts), kind="stable")
    key = pack(src_codes[order], src_ts[order])
    cumsum = np.concatenate([[0.0], np.cumsum(src_values[order])])

    hi = np.searchsorted(key, pack(query_codes, query_ts), side="left")
    lo = np.searchsorted(key, pack(query_codes, query_ts - window_s), side="left")
    return cumsum[hi] - cumsum[lo]


def window_unique(codes: np.ndarray, ts: np.ndarray, secondary: np.ndarray,
                  window_s: int) -> np.ndarray:
    """Distinct secondary values per key inside the trailing window, excluding self.

    Sliding-window distinct counts have no prefix-sum form, so this is a two-pointer with a
    live multiset. Sorting by key first keeps it a single linear pass over the stream.
    """
    n = len(codes)
    if n == 0:
        return np.array([], dtype=np.int32)
    order = np.lexsort((ts, codes))
    c, t, s = codes[order], ts[order], secondary[order]

    out = np.zeros(n, dtype=np.int32)
    counter: Dict[int, int] = {}
    left = 0
    for i in range(n):
        if i == 0 or c[i] != c[i - 1]:
            counter = {}
            left = i
        cutoff = t[i] - window_s
        while left < i and t[left] < cutoff:
            k = int(s[left])
            counter[k] -= 1
            if counter[k] == 0:
                del counter[k]
            left += 1
        out[i] = len(counter)
        k = int(s[i])
        counter[k] = counter.get(k, 0) + 1

    result = np.empty(n, dtype=np.int32)
    result[order] = out
    return result


def epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    return (timestamps.astype("int64").to_numpy() // 1_000_000_000).astype(np.int64)
