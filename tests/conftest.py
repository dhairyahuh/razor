"""Shared fixtures.

The generated dataset is session-scoped and deliberately tiny. Every test that needs
realistic data shares one campaign, because generating a stream is the expensive part and
the properties under test (schema conformance, causality, splitting) do not need volume.
"""

from __future__ import annotations

import numpy as np
import pytest

from redteam.config import BenignConfig, Config, DefenceConfig, PopulationConfig
from redteam.generate.campaign import generate_dataset
from redteam.identify.library import load_library


@pytest.fixture(scope="session")
def library():
    return load_library()


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    cfg = Config(seed=11, run_name="pytest")
    # Sized at the smallest run where the fraud budget still spans several episodes per
    # vector. Below this, whole-episode granularity dominates and the realised fraud rate
    # overshoots its target by half again, which would make prevalence untestable.
    #
    # This has to scale with the size of the attack library, which is the part that is easy
    # to forget. The fixture was tuned when 39 vectors were simulated; wiring 14 more split
    # the same fraud budget 53 ways and left roughly one row per vector in the test window.
    # Nothing was wrong with the pipeline, but every downstream assertion about recall or
    # false-positive segments became a coin flip, and the failures pointed at the detector
    # rather than at the fixture. Keeping the fraud count per vector roughly constant costs
    # about ten seconds of ensemble training and buys back a suite that fails only when
    # something is actually broken.
    cfg.population = PopulationConfig(n_customers=1400, n_merchants=120, n_psps=8)
    cfg.benign = BenignConfig(n_days=24, base_txns_per_customer_per_day=1.0)
    cfg.attacks.target_fraud_rate = 0.01
    cfg.defence = DefenceConfig(test_days=6, calibration_days=4, max_iter=60,
                                forest_n_estimators=40, zero_day_vectors=1)
    return cfg


@pytest.fixture(scope="session")
def dataset(tiny_config, library):
    return generate_dataset(tiny_config, library, rng=np.random.default_rng(tiny_config.seed))


@pytest.fixture(scope="session")
def transactions(dataset):
    return dataset.transactions


@pytest.fixture(scope="session")
def corpus(dataset):
    return dataset.corpus


@pytest.fixture(scope="session")
def transcripts(dataset):
    return dataset.transcripts


@pytest.fixture(scope="session")
def split(tiny_config, dataset):
    """A temporally split, feature-engineered frame, shared by the guard tests.

    Features first, then the split, matching the pipeline: the derived layer is causal by
    construction, and recomputing it per split would make a transaction's features depend on
    which split it landed in.
    """
    from redteam.defend.dataset import temporal_split
    from redteam.features import build_features

    featured = build_features(dataset.transactions)
    return temporal_split(featured, test_days=tiny_config.defence.test_days,
                          calibration_days=tiny_config.defence.calibration_days)


@pytest.fixture(scope="session")
def fitted_vishing(transcripts, split):
    from redteam.defend.vishing_guard import VishingGuard

    guard = VishingGuard(seed=0)
    report = guard.fit(transcripts,
                       train_end=split.calibration["timestamp"].min(),
                       test_start=split.test["timestamp"].min())
    return guard, report


@pytest.fixture(scope="session")
def population(tiny_config):
    """The same population the dataset was generated against.

    Rebuilt from the seed rather than taken off the dataset, because the population is a
    pure function of the config and the closed loop's victim-cohort gene reconstructs it the
    same way at run time.
    """
    from redteam.generate.entities import build_population

    return build_population(tiny_config, np.random.default_rng(tiny_config.seed))
