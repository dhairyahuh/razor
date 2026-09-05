"""Configuration objects for the red-team / blue-team payment fraud pipeline.

A single :class:`Config` instance is threaded through every stage so that one YAML
file fully determines a run. Every stochastic component derives its randomness from
``Config.seed``, which makes the whole pipeline bit-for-bit reproducible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclass
class PopulationConfig:
    """Size and shape of the synthetic financial ecosystem."""

    n_customers: int = 5000
    n_merchants: int = 900
    n_psps: int = 14
    # Share of customers in each digital-literacy band. Low-literacy and elderly
    # cohorts are over-represented among APP scam victims, so the split matters.
    age_bands: Dict[str, float] = field(
        default_factory=lambda: {"18-24": 0.16, "25-34": 0.28, "35-49": 0.27, "50-64": 0.18, "65+": 0.11}
    )
    corporate_share: float = 0.04
    """Fraction of accounts that are business accounts (BEC / invoice-redirect targets)."""


@dataclass
class BenignConfig:
    """Parameters of the legitimate payment stream."""

    start_date: str = "2026-05-01"
    n_days: int = 45
    base_txns_per_customer_per_day: float = 1.25
    # Rail mix. Deliberately India-forward (UPI dominant) with card and instant-credit
    # transfer rails present so rail-specific attacks have somewhere to live.
    rail_mix: Dict[str, float] = field(
        default_factory=lambda: {
            "UPI_P2M": 0.44,
            "UPI_P2P": 0.21,
            "CARD_CNP": 0.13,
            "CARD_CP": 0.07,
            "IMPS": 0.05,
            "NEFT": 0.03,
            "RTP_FEDNOW": 0.03,
            "SEPA_INST": 0.02,
            "WALLET": 0.02,
        }
    )
    # Probability that an otherwise-legitimate payment looks superficially suspicious.
    # These are the false-positive traps the defence has to survive.
    hard_negative_rate: float = 0.055


@dataclass
class AttackConfig:
    """Volume and composition of the red-team campaigns."""

    target_fraud_rate: float = 0.0075
    """Share of all transactions that carry ``is_fraud == 1``."""

    mix: Dict[str, float] = field(default_factory=dict)
    """Optional override of the attack mix, keyed by vector id.

    Left empty, the mix comes from ``default_weight`` in ``attack_library.yaml`` so the
    taxonomy stays the single source of truth. Any override is validated against the
    library and renormalised.
    """

    # How reliably an attack emits its tell-tale signals. 1.0 makes fraud trivially
    # separable; real fraud is noisy, so the default deliberately blurs the signature.
    signal_emission: float = 0.72
    # Fraction of fraud transactions that receive gradient-free evasion tuning.
    evasion_share: float = 0.18


@dataclass
class DefenceConfig:
    """Blue-team model and evaluation settings."""

    test_days: int = 12
    """Temporal holdout: the last N days are the test set (no random splitting)."""

    calibration_days: int = 6
    """Slice between train and test, used to fit the stack and set the threshold."""

    max_iter: int = 400
    forest_n_estimators: int = 300
    # Operating point: the alert budget a real fraud-ops team can absorb, expressed
    # as the share of legitimate payments we are willing to interrupt.
    target_fpr: float = 0.005
    zero_day_vectors: int = 10
    """How many vectors to hold out one-at-a-time in the leave-one-vector-out stress test."""


@dataclass
class LoopConfig:
    """Closed-loop red-vs-blue co-evolution settings."""

    rounds: int = 12
    """Rounds of co-evolution.

    Four was the old default and it was too few to say anything: an arms race characterised
    from four noisy points is a line drawn through nothing. Twelve is affordable because the
    candidate search rebuilds only the neighbourhood a mutation touched rather than the
    whole stream, and because the attacker's surrogate screens most candidates out before
    they ever reach the real pipeline.
    """

    population: int = 12
    """Attack variants measured against the real pipeline per round."""

    screen_multiplier: int = 4
    """How many candidates the attacker's surrogate sifts for each one measured properly.

    The surrogate is cheap and approximate, the real evaluation is expensive and exact, so
    generating four times the population and screening it costs little and searches wider.
    """

    oracle_probe_budget: int = 4_000
    """Payments the attacker submits to observe approve/decline before fitting the
    surrogate. A cap, because an attacker who could probe without limit would not need a
    surrogate at all."""

    survivors: int = 4
    mutation_scale: float = 0.35
    novelty_bonus: float = 0.25
    min_fraud_rows: int = 15
    """Smallest campaign the red team will bother to search against.

    Fitness is the profit a tactic nets on a vector's traffic, so a vector with a handful of
    rows gives a fitness that moves in large steps and the search becomes a random walk.
    Small runs need a lower floor to have any eligible vectors at all, at the cost of
    noisier fitness.
    """


@dataclass
class Config:
    seed: int = 20260909
    run_name: str = "default_run"
    population: PopulationConfig = field(default_factory=PopulationConfig)
    benign: BenignConfig = field(default_factory=BenignConfig)
    attacks: AttackConfig = field(default_factory=AttackConfig)
    defence: DefenceConfig = field(default_factory=DefenceConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # -- paths -------------------------------------------------------------
    @property
    def data_path(self) -> Path:
        p = ROOT / self.data_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def artifacts_path(self) -> Path:
        p = ROOT / self.artifacts_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, source: str | None = None) -> "Config":
        """Build a config, naming the offending key when something does not fit.

        A mistyped key inside a nested block otherwise surfaces as a bare
        ``TypeError: __init__() got an unexpected keyword argument``, with no indication of
        which file or which block it came from. The attack library already reports its schema
        errors properly; the config should too, since it is the file people actually edit.
        """
        raw = dict(raw or {})
        nested = {
            "population": PopulationConfig,
            "benign": BenignConfig,
            "attacks": AttackConfig,
            "defence": DefenceConfig,
            "loop": LoopConfig,
        }
        where = f" in {source}" if source else ""
        kwargs: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in nested:
                try:
                    kwargs[key] = nested[key](**(value or {}))
                except TypeError as exc:
                    known = sorted(f.name for f in fields(nested[key]))
                    raise ValueError(
                        f"bad key in the '{key}' block{where}: {exc}. "
                        f"Valid keys are: {', '.join(known)}"
                    ) from exc
            else:
                kwargs[key] = value
        try:
            return cls(**kwargs)
        except TypeError as exc:
            known = sorted(f.name for f in fields(cls))
            raise ValueError(
                f"bad top-level key{where}: {exc}. Valid keys are: {', '.join(known)}"
            ) from exc

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh), source=str(path))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
