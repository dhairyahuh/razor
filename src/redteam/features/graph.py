"""Account-graph features from nightly snapshots.

Mule networks are a property of the payment graph, not of any single payment. Fan-in,
pass-through and ring membership only exist once you look at an account's neighbourhood.

Computing graph metrics per transaction would be both intractable and dishonest - it would
use edges that had not happened yet. Instead this mirrors what banks actually do: every
night, build the graph of the trailing window, compute per-account metrics, and serve them
as features to the *next* day's authorisations. Snapshot ``d`` uses edges strictly before
day ``d``, so a transaction on day ``d`` is scored against a graph that existed before it.

The cost is deliberate staleness of up to 24 hours, which is exactly the operational
trade-off a real deployment makes.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

PREFIX = "g_"
WINDOW_DAYS = 7

#: Metrics computed for the payee (the receiving side, where mules live).
PAYEE_METRICS = [
    "in_degree", "out_degree", "unique_in", "unique_out",
    "passthrough_ratio", "pagerank", "component_size", "neighbour_mean_in_degree",
]
#: A narrower set for the payer, which matters mainly for layering hops.
PAYER_METRICS = ["out_degree", "unique_out", "passthrough_ratio"]


def build_graph_features(df: pd.DataFrame, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Attach nightly-snapshot graph metrics for both counterparties."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts = df["timestamp"]
    day_index = ((ts - ts.min()).dt.total_seconds() // 86_400).astype(int).to_numpy()
    payer = df["payer_account_id"].to_numpy()
    payee = df["payee_account_id"].to_numpy()
    # Base currency: edge weights are summed across rails that settle in different currencies.
    amount = df["amount_inr" if "amount_inr" in df.columns else "amount"].to_numpy().astype(float)

    n_days = int(day_index.max()) + 1
    out_cols: Dict[str, np.ndarray] = {
        f"{PREFIX}payee_{m}": np.full(len(df), np.nan) for m in PAYEE_METRICS
    }
    out_cols.update({f"{PREFIX}payer_{m}": np.full(len(df), np.nan) for m in PAYER_METRICS})

    for day in range(n_days):
        rows_today = np.where(day_index == day)[0]
        if rows_today.size == 0:
            continue
        # Strictly prior edges only.
        hist = np.where((day_index < day) & (day_index >= day - window_days))[0]
        if hist.size < 10:
            continue

        metrics = _snapshot_metrics(payer[hist], payee[hist], amount[hist])
        for m in PAYEE_METRICS:
            table = metrics[m]
            out_cols[f"{PREFIX}payee_{m}"][rows_today] = [
                table.get(a, 0.0) for a in payee[rows_today]
            ]
        for m in PAYER_METRICS:
            table = metrics[m]
            out_cols[f"{PREFIX}payer_{m}"][rows_today] = [
                table.get(a, 0.0) for a in payer[rows_today]
            ]

    graph = pd.DataFrame({name: np.round(v, 5) for name, v in out_cols.items()}, index=df.index)
    return pd.concat([df, graph], axis=1)


def graph_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(PREFIX)]


def _snapshot_metrics(src: np.ndarray, dst: np.ndarray, amount: np.ndarray) -> Dict[str, Dict[str, float]]:
    g = nx.DiGraph()
    frame = pd.DataFrame({"src": src, "dst": dst, "amt": amount})
    agg = frame.groupby(["src", "dst"], sort=False)["amt"].agg(["sum", "count"]).reset_index()
    for s, d, total, cnt in agg.itertuples(index=False):
        g.add_edge(s, d, weight=float(total), count=int(cnt))

    in_amount: Dict[str, float] = {}
    out_amount: Dict[str, float] = {}
    for s, d, data in g.edges(data=True):
        out_amount[s] = out_amount.get(s, 0.0) + data["weight"]
        in_amount[d] = in_amount.get(d, 0.0) + data["weight"]

    in_degree = {n: float(sum(g[u][n]["count"] for u in g.predecessors(n))) for n in g.nodes}
    out_degree = {n: float(sum(g[n][v]["count"] for v in g.successors(n))) for n in g.nodes}
    unique_in = {n: float(g.in_degree(n)) for n in g.nodes}
    unique_out = {n: float(g.out_degree(n)) for n in g.nodes}

    # Money in versus money straight back out: the defining mule behaviour.
    passthrough = {
        n: float(out_amount.get(n, 0.0) / max(in_amount.get(n, 0.0), 1.0)) for n in g.nodes
    }

    try:
        pagerank = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=60, tol=1e-5)
    except (nx.PowerIterationFailedConvergence, ZeroDivisionError):  # pragma: no cover
        pagerank = {n: 0.0 for n in g.nodes}

    undirected = g.to_undirected(as_view=True)
    component_size: Dict[str, float] = {}
    for component in nx.connected_components(undirected):
        size = float(len(component))
        for n in component:
            component_size[n] = size

    # How concentrated is the neighbourhood feeding this account? Collection points sit
    # next to many low-degree one-shot senders.
    neighbour_mean_in = {}
    for n in g.nodes:
        preds = list(g.predecessors(n))
        neighbour_mean_in[n] = (
            float(np.mean([in_degree.get(p, 0.0) for p in preds])) if preds else 0.0
        )

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "unique_in": unique_in,
        "unique_out": unique_out,
        "passthrough_ratio": passthrough,
        "pagerank": {k: float(v) for k, v in pagerank.items()},
        "component_size": component_size,
        "neighbour_mean_in_degree": neighbour_mean_in,
    }


def detect_mule_communities(df: pd.DataFrame, min_size: int = 4) -> pd.DataFrame:
    """Unsupervised sweep for laundering-shaped communities in the whole graph.

    This is an investigator's tool rather than a model feature: it produces ranked
    candidate rings for review, which is how mule intelligence is actually consumed. It is
    also the only part of the defence that can surface a ring nobody has labelled yet.
    """
    col = "amount_inr" if "amount_inr" in df.columns else "amount"
    frame = df[["payer_account_id", "payee_account_id", col]].copy()
    agg = frame.groupby(["payer_account_id", "payee_account_id"], sort=False)[col].sum()
    g = nx.DiGraph()
    for (s, d), total in agg.items():
        g.add_edge(s, d, weight=float(total))

    undirected = nx.Graph()
    undirected.add_weighted_edges_from((u, v, d["weight"]) for u, v, d in g.edges(data=True))
    if undirected.number_of_nodes() == 0:
        return pd.DataFrame()

    communities = nx.community.greedy_modularity_communities(undirected, weight="weight")
    rows = []
    for i, members in enumerate(communities):
        if len(members) < min_size:
            continue
        sub = g.subgraph(members)
        in_amt = sum(d["weight"] for _, _, d in sub.in_edges(data=True))
        out_amt = sum(d["weight"] for _, _, d in sub.out_edges(data=True))
        internal = sub.number_of_edges()
        rows.append(
            {
                "community_id": i,
                "size": len(members),
                "internal_edges": internal,
                "density": round(internal / max(len(members) * (len(members) - 1), 1), 4),
                "internal_value": round(in_amt, 2),
                "passthrough_ratio": round(out_amt / max(in_amt, 1.0), 4),
                "members_sample": sorted(list(members))[:8],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Rank by a laundering-shaped score: dense, high-value, high pass-through.
    out["ring_score"] = (
        out["density"].rank(pct=True) * 0.35
        + out["passthrough_ratio"].clip(0, 2).rank(pct=True) * 0.35
        + out["internal_value"].rank(pct=True) * 0.30
    ).round(4)
    return out.sort_values("ring_score", ascending=False).reset_index(drop=True)
