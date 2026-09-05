"""Is the full profile more separable because of its calendar, or its sample size?

The full-profile run recovered 93.7% of fraud from the derived matrix against a 92% ceiling,
where the quick profile recovered 83.5%. Two explanations fit that, and they call for opposite
fixes:

* **Calendar.** Forty-five days give legitimate payees history that the seasoning model never
  gives mule accounts, so the payee block drifts apart as a run lengthens. The fix is in the
  generator: season mules harder.
* **Sample size.** The generator is unchanged and defines fraud as a probabilistic combination
  of a few dozen tells. With a couple of hundred training fraud rows most of those combinations
  are unlearnable noise; with a couple of thousand they are reliable rules. The fix is not in
  the generator at all - it is to stop quoting separability without saying how many fraud rows
  bought it.

Holding the generator fixed and varying only the learner's data budget separates them. This
subsamples one dataset, so every row comes from the same 45-day timeline with the same payee
histories: if recall still climbs with sample size, the calendar cannot be what is doing it.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from redteam.generate.fidelity import FidelityReport, _probe_separability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/default_run/transactions.parquet")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[20_000, 40_000, 80_000, 160_000, 400_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--derived", action="store_true",
                        help="probe the derived matrix rather than the raw schema")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.derived:
        from redteam.features.graph import build_graph_features
        from redteam.features.tabular import build_features
        df = build_graph_features(build_features(df))
        for c in ("agent_injection_score", "agent_injection_flag", "intent_guard_blocked",
                  "intent_guard_stepup", "intent_guard_violation_count"):
            if c not in df.columns:
                df[c] = 0.0
        from redteam.defend.dataset import input_columns
        cols = input_columns(df)
    else:
        # Exactly the set the shipped probe uses. Rolling my own "everything but the labels"
        # list quietly admitted columns the real check excludes and produced AUCs of 0.99 -
        # a measurement of my column selection rather than of the generator.
        from redteam.schema import observable_columns
        cols = [c for c in observable_columns() if c in df.columns]

    rows = []
    for size in args.sizes:
        # Several seeds per size, because the probe is known to swing by tens of points
        # between subsample draws at small sizes. One seed per point would produce a curve
        # made mostly of sampling noise, which is the error this script exists to rule out.
        for seed in args.seeds:
            report = FidelityReport()
            _probe_separability(df, report, cols, key="probe", label="probe",
                                max_rows=size, seed=seed)
            detail = report.details.get("probe")
            if not detail:
                continue
            rows.append({"rows": min(size, len(df)), "seed": seed,
                         "test_fraud": detail["test_fraud_rows"],
                         "recall": detail["recall_at_review_budget"],
                         "lo95": detail["recall_lo95"],
                         "hi95": detail["recall_hi95"],
                         "auc": detail["held_out_auc"]})
            print(json.dumps(rows[-1]), flush=True)
            if min(size, len(df)) == len(df):
                break  # no subsampling happens, so further seeds repeat the same fit

    frame = pd.DataFrame(rows)
    print()
    print(frame.to_string(index=False))
    print()
    print(frame.groupby("rows").agg(n=("recall", "size"), mean_recall=("recall", "mean"),
                                    min_recall=("recall", "min"),
                                    max_recall=("recall", "max"),
                                    mean_auc=("auc", "mean")).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
