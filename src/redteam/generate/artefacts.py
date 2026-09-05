"""Automated search for generator artefacts - values that betray fraud by construction.

Every synthetic-fraud generator leaks, and it leaks in the same three shapes:

1. **A fraud-only namespace.** An attack path mints ids from its own prefix, so
   ``MULEOP-`` or ``DATT`` is fraud with probability 1. Nothing downstream can un-learn
   that, and it silently contaminates any feature keyed on the entity - a colliding
   attacker device id becomes a phantom device farm spanning the whole dataset.

2. **A fraud-only categorical level.** A rail, channel or auth method that legitimate
   traffic never uses, so the level alone decides the label.

3. **A constant or near-constant block.** A bespoke generator writes one scalar across an
   entire episode, producing a numeric band fraud occupies and legitimate traffic does
   not. Individually each looks like a plausible attacker choice; together they hand the
   model a lookup table.

The three defects this module was written to find - a fraud-only mule-operator namespace,
colliding attacker device ids, and constant-blocked hop rows - were all found by reading
code, which does not scale and does not catch the next one. So the search runs as a test.

It is deliberately blunt: it makes no judgement about whether a separation is *legitimate*.
Some are - ``fraud_type`` is a label, and an attack-only channel may be genuinely
attack-only in the real world. Those go in :data:`EXPECTED` with a reason, which turns the
suppression list itself into documentation of every place the generator is knowingly
unrealistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schema import LABEL_COLUMNS, META_COLUMNS

#: A categorical level or id prefix must appear on at least this many fraud rows before the
#: absence of legitimate examples means anything. Below it, absence is small-sample noise.
MIN_SUPPORT = 25

#: Legitimate share of a level below which it counts as fraud-only. Not zero: one stray
#: legitimate row should not launder an otherwise perfect tell.
MAX_LEGIT_SHARE = 0.02

#: Fraud share of a numeric band above which the band is a fraud tell, given support.
BAND_FRAUD_SHARE = 0.90

#: Columns whose separation is intended, each with the reason it is not a defect.
EXPECTED: Dict[str, str] = {
    "fraud_type": "label column - describes the fraud, does not predict it",
    "attack_vector_id": "label column",
    "campaign_id": "label column",
    "mule_ring_id": "label column - null for legitimate traffic by definition",
    "is_hard_negative": "generator bookkeeping, never a model input",
    "evasion_applied": "generator bookkeeping, never a model input",
}


@dataclass
class Artefact:
    """One value, prefix or band that separates fraud far better than it should."""

    column: str
    kind: str
    value: str
    fraud_rows: int
    legit_rows: int
    fraud_share: float

    def describe(self) -> str:
        return (
            f"{self.column} {self.kind} {self.value!r}: {self.fraud_rows} fraud rows vs "
            f"{self.legit_rows} legitimate ({self.fraud_share:.1%} fraud)"
        )


def _candidate_columns(df: pd.DataFrame, extra_skip: Iterable[str]) -> List[str]:
    skip = set(LABEL_COLUMNS) | set(extra_skip) | set(EXPECTED)
    return [c for c in df.columns if c not in skip and not c.startswith("_")]


def _scan_levels(fraud: pd.Series, legit: pd.Series, column: str, kind: str) -> List[Artefact]:
    fc = fraud.value_counts()
    lc = legit.value_counts()
    out: List[Artefact] = []
    for value, n_fraud in fc.items():
        if n_fraud < MIN_SUPPORT:
            continue
        n_legit = int(lc.get(value, 0))
        share = n_fraud / (n_fraud + n_legit)
        # Compare against the legitimate *rate* of the level, not its raw count: legitimate
        # traffic outnumbers fraud ~140:1, so a handful of legitimate rows on a level with
        # thousands of fraud rows is still effectively a fraud-only level.
        if n_legit / max(len(legit), 1) <= MAX_LEGIT_SHARE * (n_fraud / max(len(fraud), 1)):
            out.append(Artefact(column, kind, str(value), int(n_fraud), n_legit, share))
    return out


def _id_prefix(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"^([A-Za-z]+)", expand=False).fillna("")


def _scan_numeric(fraud: pd.Series, legit: pd.Series, column: str) -> List[Artefact]:
    """Look for a narrow band of the range that fraud occupies and legitimate traffic does not.

    Quantile bins of the *fraud* distribution are the right unit here. Binning the pooled
    range would spread fraud thinly across bins dominated by legitimate volume and hide
    exactly the concentration we are looking for.
    """
    f = pd.to_numeric(fraud, errors="coerce").dropna()
    l = pd.to_numeric(legit, errors="coerce").dropna()
    if len(f) < MIN_SUPPORT * 4 or len(l) < MIN_SUPPORT or f.nunique() < 4:
        return []

    edges = np.unique(np.quantile(f, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return []
    edges[0], edges[-1] = -np.inf, np.inf

    fh, _ = np.histogram(f, bins=edges)
    lh, _ = np.histogram(l, bins=edges)
    out: List[Artefact] = []
    for i, n_fraud in enumerate(fh):
        if n_fraud < MIN_SUPPORT:
            continue
        n_legit = int(lh[i])
        # Rate-normalised: what share of this band would be fraud if the two classes were
        # equally prevalent. Raw share would never exceed 50% at a 0.7% base rate.
        rate_f = n_fraud / len(f)
        rate_l = n_legit / len(l)
        share = rate_f / (rate_f + rate_l) if (rate_f + rate_l) else 0.0
        if share >= BAND_FRAUD_SHARE:
            lo, hi = edges[i], edges[i + 1]
            out.append(
                Artefact(column, "band", f"[{lo:.4g}, {hi:.4g})",
                         int(n_fraud), n_legit, float(share))
            )
    return out


def find_artefacts(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    skip: Iterable[str] = (),
    include_numeric: bool = True,
) -> List[Artefact]:
    """Search ``df`` for values that separate fraud from legitimate traffic too cleanly.

    Meta columns are scanned by *id prefix* rather than by value: a mule account id is
    unique by design, and the defect is the namespace it was minted into.
    """
    if "is_fraud" not in df.columns:
        raise ValueError("frame has no is_fraud column")
    fraud_mask = df["is_fraud"].to_numpy().astype(bool)
    if not fraud_mask.any() or fraud_mask.all():
        return []

    cols = list(columns) if columns is not None else _candidate_columns(df, skip)
    found: List[Artefact] = []
    for c in cols:
        col = df[c]
        fraud, legit = col[fraud_mask], col[~fraud_mask]
        if c in META_COLUMNS:
            if c == "timestamp":
                continue
            found += _scan_levels(_id_prefix(fraud), _id_prefix(legit), c, "id prefix")
        elif col.dtype == object or isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == bool:
            found += _scan_levels(fraud.astype(str), legit.astype(str), c, "level")
        elif include_numeric and pd.api.types.is_numeric_dtype(col):
            found += _scan_numeric(fraud, legit, c)

    found.sort(key=lambda a: (-a.fraud_rows, a.column))
    return found


def artefact_frame(found: Sequence[Artefact]) -> pd.DataFrame:
    if not found:
        return pd.DataFrame(columns=["column", "kind", "value", "fraud_rows",
                                     "legit_rows", "fraud_share"])
    return pd.DataFrame([vars(a) for a in found])
