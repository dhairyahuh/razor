"""Attack generator registry.

Every ``simulated: true`` vector in ``attack_library.yaml`` names a generator module here.
The registry is validated against the library at import time by the test suite, so the
taxonomy and the code cannot silently drift apart.
"""

from __future__ import annotations

from typing import Dict

from .agentic import AgenticGenerator
from .app_scam import AppScamGenerator
from .ato import AtoGenerator
from .base import AttackContext, AttackGenerator, MuleRegistry, build_context
from .card import CardGenerator
from .credential import CredentialGenerator
from .identity import SyntheticIdentityGenerator
from .model_attacks import ModelAttackGenerator, apply_evasion
from .mule import MuleGenerator
from .rail import RailGenerator

GENERATORS: Dict[str, AttackGenerator] = {
    g.name: g
    for g in (
        AppScamGenerator(),
        AtoGenerator(),
        SyntheticIdentityGenerator(),
        CardGenerator(),
        CredentialGenerator(),
        AgenticGenerator(),
        RailGenerator(),
        MuleGenerator(),
        ModelAttackGenerator(),
    )
}


def generator_for(vector) -> AttackGenerator:
    try:
        return GENERATORS[vector.generator]
    except KeyError as exc:  # pragma: no cover - guarded by tests
        raise KeyError(
            f"vector {vector.id!r} declares generator {vector.generator!r}, "
            f"which is not registered. Known: {sorted(GENERATORS)}"
        ) from exc


__all__ = [
    "GENERATORS",
    "generator_for",
    "AttackContext",
    "AttackGenerator",
    "MuleRegistry",
    "build_context",
    "apply_evasion",
]
