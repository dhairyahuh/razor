"""Run the co-evolution loop against an existing dataset and print per-phase timings.

The loop is the last stage of a full run, so reproducing a problem in it normally means
waiting out generate and defend first - twenty minutes before the thing under investigation
even starts. This trains the minimum detector the loop needs and goes straight there.

It exists because a full-profile run had a round that took seven times the two before it,
with nothing in the log between "red team searching" and "blue team answering" to say where
the time went.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from redteam.config import Config
from redteam.defend.dataset import EXTRA_INPUTS, temporal_split
from redteam.defend.model import FraudDetector
from redteam.features import build_features, build_graph_features
from redteam.generate.enrich import enrich
from redteam.identify.library import load_library
from redteam.loop.coevolution import run_loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/default_run/transactions.parquet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int, default=None,
                        help="override loop.rounds, so a scaling problem can be seen in three "
                             "rounds rather than twelve")
    parser.add_argument("--no-customers", action="store_true",
                        help="disable the victim-cohort gene, to isolate its cost")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.rounds:
        cfg.loop.rounds = args.rounds

    raw = pd.read_parquet(args.data)
    print(f"{len(raw):,} rows", flush=True)

    t = time.perf_counter()
    featured = build_graph_features(build_features(enrich(raw)))
    print(f"features {time.perf_counter() - t:.0f}s", flush=True)

    # The guard columns the loop carries through unchanged. Their real values come from the
    # defence stage; the loop never recomputes them, so schema defaults are enough to
    # reproduce its timing behaviour.
    for column in EXTRA_INPUTS:
        if column not in featured.columns:
            featured[column] = 0.0

    split = temporal_split(featured, test_days=cfg.defence.test_days,
                           calibration_days=cfg.defence.calibration_days)
    t = time.perf_counter()
    detector = FraudDetector(seed=cfg.seed, target_fpr=cfg.defence.target_fpr,
                             max_iter=cfg.defence.max_iter,
                             n_estimators=cfg.defence.forest_n_estimators
                             ).fit(split.train, split.calibration)
    print(f"detector {time.perf_counter() - t:.0f}s", flush=True)

    scores = detector.predict_proba(split.test) >= detector.threshold
    per_vector = (pd.DataFrame({"attack_vector_id": split.test["attack_vector_id"],
                                "is_fraud": split.test["is_fraud"], "hit": scores})
                  .query("is_fraud == 1").groupby("attack_vector_id")["hit"].mean()
                  .rename("recall").reset_index())

    # The victim-cohort gene silently does nothing without this, and it is the gene that
    # rewrites who a campaign's payers are - which is exactly the kind of change that makes a
    # lot of accounts dirty at once. Omitting it is how a first attempt at this script failed
    # to reproduce the problem it was written to find.
    import numpy as np
    from redteam.generate.entities import build_population

    customers = (None if args.no_customers
                 else build_population(cfg, np.random.default_rng(cfg.seed)).customers)

    t = time.perf_counter()
    run_loop(raw, featured, per_vector, detector, cfg, verbose=True,
             library=load_library(), customers=customers)
    print(f"\nloop total {time.perf_counter() - t:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
