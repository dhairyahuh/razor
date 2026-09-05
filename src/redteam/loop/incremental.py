"""Recompute features for the rows a mutation touched, instead of all 280,000 of them.

Why this exists
---------------
The naive loop calls ``enrich`` and ``build_features`` over the entire stream once per
candidate genome. With a population of twelve that is twelve full feature builds per round,
and a full build is the most expensive thing in the pipeline. Four rounds was all the
compute budget allowed, and four rounds of a stochastic search is not a curve - it is four
points of noise with a line drawn through them. Everything interesting about co-evolution
(does it converge? do tactics transfer? does the attacker's ROI trend?) needs ten times that.

What makes it safe
------------------
Every feature in :mod:`redteam.features.tabular` and every column in
:mod:`redteam.generate.enrich` is a causal window keyed on exactly one of four entities:
the payer account, the payee account, the merchant, or the device. No feature mixes two
entities' histories, and none is global. That gives an exact locality argument:

    A row's features for entity view *V* can only change if the entity it has in view *V*
    owns at least one row that changed.

So mark every entity that owns a changed row as **dirty**, take every row belonging to any
dirty entity, and recompute over that subset. For a row whose *V*-entity is dirty, the whole
of that entity's history is inside the subset, so the recomputed value is not an
approximation of the full-frame value - it is bit-for-bit the same number.

The rows the search scores are the mutated campaign's own rows, and their payer, payee,
merchant and device are all dirty by construction. Their features are therefore exact, which
is the only guarantee the fitness function needs. ``test_loop.py`` asserts this equality
against a full rebuild rather than taking the argument on trust.

The one carried-over column
---------------------------
``f_payer_new_payees_7d`` is the exception that proves the rule: it is a payer-keyed window
over ``payee_is_first_time``, which is itself a payee-keyed output of ``enrich``. A dirty
payer's rows can point at clean payees whose history is only partly inside the subset, and
those payees would be recomputed as first-time when they are nothing of the sort. So the
enrich outputs are restored from cache for every row with a clean payee, in between the
enrich step and the feature step. Clean payees did not change, so the cached value is by
definition correct.

Graph features
--------------
The nightly graph snapshot is global, so it does not decompose this way. It is served from a
``(day, account) -> metrics`` lookup built from the cached full-frame build, which is exact
whenever a genome leaves the topology alone, and for a genome that reroutes money it is
exact for accounts whose own prior edges are unchanged and reports the correct zero for
accounts that have none. The residual inexactness is confined to the red team's internal
ranking of candidates within a round. Nothing reported ever depends on it: when the surviving
tactics are committed and blue retrains, the graph is rebuilt in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from ..features import build_features, build_graph_features
from ..features.graph import PREFIX as GRAPH_PREFIX
from ..generate.enrich import enrich

#: Entity columns each feature view is keyed on.
ENTITY_KEYS = ("payer_account_id", "payee_account_id", "merchant_id", "device_id")

#: Entities whose every feature is a bounded trailing window, so their history is only needed
#: as far back as :data:`MAX_FEATURE_WINDOW_S`. The payee is deliberately absent: ``enrich``
#: counts a payee's prior payments from the beginning of the stream, and the payer appears in
#: that count through the (payer, payee) pair, so truncating either would corrupt a cumulative
#: column rather than a windowed one.
WINDOWED_ENTITY_KEYS = ("merchant_id", "device_id")

#: The longest trailing window any tabular feature reads, in seconds. Merchant views top out
#: at 24 hours and device views at seven days; taking the maximum keeps this correct if a
#: shorter view later grows. A change at time T cannot alter a windowed feature of a row
#: earlier than T, nor of a row more than this long after T.
MAX_FEATURE_WINDOW_S = 7 * 24 * 3600

#: ``enrich`` outputs keyed on the payee. These are the ones restored from cache for rows
#: whose payee is clean, because a partially present payee would be recomputed wrongly.
PAYEE_ENRICH_COLUMNS = (
    "payee_prior_txn_count",
    "payee_age_days",
    "payee_is_first_time",
    "payee_added_minutes_ago",
    "payee_inbound_amount_24h",
    "payee_inbound_unique_payers_24h",
    "payee_outbound_ratio_24h",
)


@dataclass
class RebuildCache:
    """Everything needed to rebuild a mutation's neighbourhood without touching the rest.

    Built once per round from the committed stream and reused across every candidate, which
    is the memoisation that makes a twelve-strong population affordable.
    """

    featured: pd.DataFrame
    """The full featured frame for the current committed stream."""

    guards: pd.DataFrame
    """Guard outputs keyed by ``txn_id``. Neither guard reads a field any gene writes."""

    graph_columns: List[str]

    def __post_init__(self) -> None:
        self._enrich_cache = self.featured.set_index("txn_id")[
            [c for c in PAYEE_ENRICH_COLUMNS if c in self.featured.columns]
        ]
        self._graph_lookup = _build_graph_lookup(self.featured, self.graph_columns)
        self._origin = pd.Timestamp(self.featured["timestamp"].min())


def dirty_entities(before: pd.DataFrame, after: pd.DataFrame,
                   changed_txn_ids: Iterable[str]) -> Dict[str, Set]:
    """Entities owning at least one row that the mutation created, deleted or edited.

    Both frames are consulted. A genome that reroutes a payment to a different account
    makes *two* accounts dirty: the one that lost the payment and the one that gained it.
    Taking only the "after" side would leave the abandoned account's fan-in features stale.
    """
    changed = set(changed_txn_ids)
    before_rows = before[before["txn_id"].isin(changed)]
    after_rows = after[after["txn_id"].isin(changed)]

    # Rows that exist in exactly one frame were created or deleted, so their entities are
    # dirty too even if no txn_id was explicitly flagged.
    before_ids = set(before["txn_id"])
    after_ids = set(after["txn_id"])
    created = after[after["txn_id"].isin(after_ids - before_ids)]
    deleted = before[before["txn_id"].isin(before_ids - after_ids)]

    out: Dict[str, Set] = {}
    for key in ENTITY_KEYS:
        values: Set = set()
        for frame in (before_rows, after_rows, created, deleted):
            if key in frame.columns and len(frame):
                values.update(frame[key].to_numpy().tolist())
        values.discard("")
        out[key] = values
    return out


def neighbourhood(frame: pd.DataFrame, dirty: Dict[str, Set],
                  changed_txn_ids: Optional[Iterable[str]] = None) -> np.ndarray:
    """Boolean mask over ``frame`` selecting every row owned by any dirty entity.

    A merchant's history is truncated to the window horizon around the change, because it can
    only ever be read through a bounded trailing window. Without that, one dirty merchant drags
    in every row it owns, and merchants are not small: the largest in a full run owns 14,123
    rows, 5% of the entire book. Once a surviving tactic involved such a merchant, every genome
    mutated from it inherited the merchant, and each of the twelve candidates in a round paid
    for a near-complete feature build. Rounds one and two of a full-profile run took seven
    minutes between them; round three had not finished fifty minutes later.

    The truncation is not an approximation. ``enrich``'s cumulative columns are keyed on the
    payee and the (payer, payee) pair, and both of those entities keep their whole history
    here; only the merchant and device views are bounded, and every feature reading them is a
    window of at most :data:`MAX_FEATURE_WINDOW_S`. Rows outside the horizon cannot contribute
    to any of them. ``test_loop.py`` asserts equality against the unbounded rebuild rather than
    resting on that argument.
    """
    mask = np.zeros(len(frame), dtype=bool)
    unbounded = np.zeros(len(frame), dtype=bool)
    bounded = np.zeros(len(frame), dtype=bool)
    for key, values in dirty.items():
        if not values or key not in frame.columns:
            continue
        owned = frame[key].isin(values).to_numpy()
        if key in WINDOWED_ENTITY_KEYS:
            bounded |= owned
        else:
            unbounded |= owned

    changed = set(changed_txn_ids or ())
    if not changed or not bounded.any():
        return unbounded | bounded

    # The horizon is anchored on the rows that moved, since those are the only rows whose
    # windows the change can enter, and it extends backwards because the windows are trailing.
    moved = frame["txn_id"].isin(changed).to_numpy()
    if not moved.any():
        return unbounded | bounded
    stamps = frame["timestamp"].to_numpy()
    lo = stamps[moved].min() - np.timedelta64(MAX_FEATURE_WINDOW_S, "s")
    hi = stamps[moved].max() + np.timedelta64(MAX_FEATURE_WINDOW_S, "s")
    mask = unbounded | (bounded & (stamps >= lo) & (stamps <= hi))
    return mask


def rebuild_neighbourhood(mutated: pd.DataFrame, cache: RebuildCache,
                          dirty: Dict[str, Set],
                          changed_txn_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Features for the mutation's neighbourhood, exact for every dirty entity's rows.

    Returns only the neighbourhood, not the whole stream. Callers score a subset of it and
    have no use for the rows that did not move; assembling a full frame would put the
    concatenation cost straight back in.

    ``changed_txn_ids`` anchors the window horizon that bounds merchant and device history.
    Omitting it keeps every dirty entity's full history, which is correct but, on a large
    merchant, ruinously slow - see :func:`neighbourhood`.
    """
    subset = mutated[neighbourhood(mutated, dirty, changed_txn_ids)].copy()
    if subset.empty:
        return subset

    frame = enrich(subset)
    _restore_clean_payee_enrichment(frame, cache, dirty["payee_account_id"])
    frame = build_features(frame)
    frame = _attach_graph(frame, cache)
    return _merge_guards(frame, cache.guards)


def full_rebuild(raw: pd.DataFrame, guards: pd.DataFrame) -> pd.DataFrame:
    """The exact, expensive path: enrich, features and a full graph rebuild.

    Used when the round's surviving tactics are committed and blue retrains, so every
    reported number comes from a complete recomputation rather than from the incremental
    approximation the candidate search runs on.
    """
    frame = enrich(raw)
    frame = build_features(frame)
    frame = build_graph_features(frame)
    return _merge_guards(frame, guards)


def _merge_guards(frame: pd.DataFrame, guards: pd.DataFrame) -> pd.DataFrame:
    """Attach guard outputs, dropping the raw frame's own copies first.

    ``agent_injection_score`` lives in both the raw schema and the guard layer - the raw one
    is what the generator wrote, the guard one is what the classifier scored. Merging without
    dropping the first produces ``_x`` and ``_y`` suffixes, so the column the model was
    fitted on stops existing under the name it was fitted under, and the detector raises a
    ``KeyError`` several minutes into the loop.
    """
    stale = [c for c in guards.columns if c != "txn_id" and c in frame.columns]
    return frame.drop(columns=stale).merge(guards, on="txn_id", how="left")


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

def _restore_clean_payee_enrichment(frame: pd.DataFrame, cache: RebuildCache,
                                    dirty_payees: Set) -> None:
    """Put back the cached payee enrichment for rows whose payee did not change.

    Modifies ``frame`` in place. Rows with a dirty payee keep their freshly computed values,
    which are exact because a dirty payee's entire history is present.
    """
    columns = [c for c in PAYEE_ENRICH_COLUMNS if c in frame.columns
               and c in cache._enrich_cache.columns]
    if not columns:
        return
    clean = ~frame["payee_account_id"].isin(dirty_payees).to_numpy()
    if not clean.any():
        return

    cached = cache._enrich_cache.reindex(frame["txn_id"].to_numpy())
    for column in columns:
        values = frame[column].to_numpy().copy()
        replacement = cached[column].to_numpy()
        # A clean-payee row absent from the cache cannot exist - the genes only create rows
        # on the campaign's own accounts - but guard anyway rather than write a NaN into a
        # feature the model will read.
        usable = clean & ~pd.isna(replacement)
        if values.dtype.kind in "iu" and replacement.dtype.kind == "f":
            values = values.astype(float)
        values[usable] = replacement[usable]
        frame[column] = values


def _build_graph_lookup(featured: pd.DataFrame, graph_columns: List[str]) -> Dict:
    """``(day, account) -> metrics`` tables, one for the payee side and one for the payer.

    The snapshot metrics are a per-account lookup by construction, so this reproduces the
    same table the builder used rather than approximating it.
    """
    if not graph_columns or featured.empty:
        return {}
    day = _day_index(featured["timestamp"], pd.Timestamp(featured["timestamp"].min()))
    payee_cols = [c for c in graph_columns if c.startswith(f"{GRAPH_PREFIX}payee_")]
    payer_cols = [c for c in graph_columns if c.startswith(f"{GRAPH_PREFIX}payer_")]

    lookup: Dict[str, object] = {}
    for side, columns, key in (("payee", payee_cols, "payee_account_id"),
                               ("payer", payer_cols, "payer_account_id")):
        if not columns:
            continue
        table = featured[[key] + columns].copy()
        table["_day"] = day
        lookup[side] = (
            table.groupby(["_day", key], sort=False)[columns].first(),
            columns,
            key,
        )
    # Days on which the builder had too little history to take a snapshot at all, so the
    # correct value is nan rather than zero.
    snapshot_days = set()
    if payee_cols:
        has = featured[payee_cols[0]].notna().to_numpy()
        snapshot_days = set(np.unique(day[has]).tolist())
    lookup["_snapshot_days"] = snapshot_days
    return lookup


def _attach_graph(frame: pd.DataFrame, cache: RebuildCache) -> pd.DataFrame:
    """Serve graph features from the cached snapshot rather than rebuilding it."""
    if not cache._graph_lookup:
        return frame
    day = _day_index(frame["timestamp"], cache._origin)
    snapshot_days = cache._graph_lookup.get("_snapshot_days", set())
    on_snapshot_day = np.isin(day, list(snapshot_days)) if snapshot_days else np.zeros(
        len(frame), dtype=bool)

    attached: Dict[str, np.ndarray] = {}
    for side in ("payee", "payer"):
        entry = cache._graph_lookup.get(side)
        if entry is None:
            continue
        table, columns, key = entry
        index = pd.MultiIndex.from_arrays([day, frame[key].to_numpy()])
        values = table.reindex(index)
        for column in columns:
            series = values[column].to_numpy()
            # An account missing from the snapshot has no prior edges, which the builder
            # scores as zero - but only on days it took a snapshot at all.
            attached[column] = np.where(pd.isna(series),
                                        np.where(on_snapshot_day, 0.0, np.nan),
                                        series)
    if not attached:
        return frame
    return pd.concat([frame.reset_index(drop=True),
                      pd.DataFrame(attached, index=frame.reset_index(drop=True).index)], axis=1)


def _day_index(timestamps: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((pd.DatetimeIndex(timestamps) - origin).total_seconds() // 86_400).astype(int)
