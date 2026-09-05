"""Loader, validator and reporting layer for the attack library.

The library is treated as code, not prose: it is parsed into typed objects, validated
against the transaction schema, and used to drive both the attack mix and the defence
coverage matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from ..schema import observable_columns

LIBRARY_PATH = Path(__file__).with_name("attack_library.yaml")


@dataclass(frozen=True)
class AttackVector:
    id: str
    name: str
    family: str
    description: str
    genai_enablers: List[str] = field(default_factory=list)
    kill_chain: List[str] = field(default_factory=list)
    rails: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    victim: str = "consumer"
    signals: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    simulated: bool = False
    generator: Optional[str] = None
    default_weight: float = 0.0
    severity: int = 1
    prevalence: int = 1
    detection_difficulty: int = 1
    liability_note: str = ""

    @property
    def risk_score(self) -> float:
        """Composite prioritisation score in [0, 1].

        Severity and prevalence set the expected loss; detection difficulty is what makes
        a vector worth spending red-team budget on, so it carries equal weight to severity.
        """
        return round((0.4 * self.severity + 0.25 * self.prevalence + 0.35 * self.detection_difficulty) / 5.0, 4)


@dataclass
class AttackLibrary:
    version: str
    updated: str
    kill_chain: List[str]
    families: Dict[str, Dict[str, str]]
    vectors: List[AttackVector]

    # -- lookups -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def by_id(self) -> Dict[str, AttackVector]:
        return {v.id: v for v in self.vectors}

    def get(self, vector_id: str) -> AttackVector:
        try:
            return self.by_id[vector_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown attack vector id: {vector_id!r}") from exc

    def simulated(self) -> List[AttackVector]:
        return [v for v in self.vectors if v.simulated]

    def by_family(self, family: str) -> List[AttackVector]:
        return [v for v in self.vectors if v.family == family]

    def generators(self) -> Dict[str, List[AttackVector]]:
        out: Dict[str, List[AttackVector]] = {}
        for v in self.simulated():
            out.setdefault(v.generator or "", []).append(v)
        return out

    # -- weighting ---------------------------------------------------------
    def default_mix(self) -> Dict[str, float]:
        """Normalised attack mix over simulated vectors."""
        sim = self.simulated()
        total = sum(v.default_weight for v in sim)
        if total <= 0:  # pragma: no cover - defensive
            return {v.id: 1.0 / len(sim) for v in sim}
        return {v.id: v.default_weight / total for v in sim}

    def resolve_mix(self, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Apply a config override on top of the library defaults and renormalise."""
        if not override:
            return self.default_mix()
        known = self.by_id
        unknown = sorted(set(override) - set(known))
        if unknown:
            raise KeyError(f"config attack mix references unknown vectors: {unknown}")
        not_simulated = sorted(k for k in override if not known[k].simulated)
        if not_simulated:
            raise ValueError(f"config attack mix references non-simulated vectors: {not_simulated}")
        total = sum(override.values())
        return {k: v / total for k, v in override.items() if v > 0}

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of consistency problems; empty means the library is sound."""
        problems: List[str] = []
        observable = set(observable_columns())
        seen: set = set()
        valid_stages = set(self.kill_chain)

        for v in self.vectors:
            if v.id in seen:
                problems.append(f"{v.id}: duplicate vector id")
            seen.add(v.id)
            if v.family not in self.families:
                problems.append(f"{v.id}: unknown family {v.family!r}")
            missing = [s for s in v.signals if s not in observable]
            if missing:
                problems.append(f"{v.id}: signals not present in schema: {missing}")
            bad_stages = [s for s in v.kill_chain if s not in valid_stages]
            if bad_stages:
                problems.append(f"{v.id}: unknown kill-chain stages: {bad_stages}")
            if v.simulated and not v.generator:
                problems.append(f"{v.id}: marked simulated but has no generator")
            if v.simulated and v.default_weight <= 0:
                problems.append(f"{v.id}: marked simulated but has zero default_weight")
            if not v.simulated and v.default_weight > 0:
                problems.append(f"{v.id}: not simulated but carries a non-zero weight")
            if not v.signals:
                problems.append(f"{v.id}: no observable signals declared")
        return problems

    # -- reporting ---------------------------------------------------------
    def coverage_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for v in self.vectors:
            rows.append(
                {
                    "id": v.id,
                    "name": v.name,
                    "family": v.family,
                    "victim": v.victim,
                    "rails": "|".join(v.rails),
                    "genai_enablers": "|".join(v.genai_enablers),
                    "n_signals": len(v.signals),
                    "signals": "|".join(v.signals),
                    "controls": "|".join(v.controls),
                    "severity": v.severity,
                    "prevalence": v.prevalence,
                    "detection_difficulty": v.detection_difficulty,
                    "risk_score": v.risk_score,
                    "simulated": v.simulated,
                    "generator": v.generator or "",
                    "default_weight": v.default_weight,
                }
            )
        return rows

    def signal_usage(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in self.vectors:
            for s in v.signals:
                counts[s] = counts.get(s, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def load_library(path: str | Path | None = None) -> AttackLibrary:
    path = Path(path) if path else LIBRARY_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    vectors = [AttackVector(**_clean(v)) for v in raw.get("vectors", [])]
    return AttackLibrary(
        version=raw.get("version", "0"),
        updated=str(raw.get("updated", "")),
        kill_chain=raw.get("kill_chain", []),
        families=raw.get("families", {}),
        vectors=vectors,
    )


def _clean(raw: Dict[str, object]) -> Dict[str, object]:
    """Normalise a raw YAML vector into AttackVector kwargs."""
    allowed = set(AttackVector.__dataclass_fields__)  # type: ignore[attr-defined]
    out = {k: v for k, v in raw.items() if k in allowed}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"vector {raw.get('id')!r} has unknown keys: {sorted(unknown)}")
    if isinstance(out.get("description"), str):
        out["description"] = " ".join(out["description"].split())
    if isinstance(out.get("liability_note"), str):
        out["liability_note"] = " ".join(out["liability_note"].split())
    return out


def summarise(library: AttackLibrary) -> Dict[str, object]:
    sim = library.simulated()
    families = {}
    for fam in library.families:
        vs = library.by_family(fam)
        families[fam] = {
            "total": len(vs),
            "simulated": sum(1 for v in vs if v.simulated),
            "mean_risk": round(sum(v.risk_score for v in vs) / max(len(vs), 1), 3),
        }
    return {
        "library_version": library.version,
        "total_vectors": len(library),
        "simulated_vectors": len(sim),
        "families": families,
        "distinct_genai_enablers": len({e for v in library.vectors for e in v.genai_enablers}),
        "distinct_controls": len({c for v in library.vectors for c in v.controls}),
        "top_risk": [v.id for v in sorted(library.vectors, key=lambda x: -x.risk_score)[:10]],
    }


def iter_signals(vectors: Iterable[AttackVector]) -> List[str]:
    seen: List[str] = []
    for v in vectors:
        for s in v.signals:
            if s not in seen:
                seen.append(s)
    return seen
