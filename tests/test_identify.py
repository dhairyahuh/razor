"""The taxonomy is treated as code, so it gets the same tests code gets."""

from __future__ import annotations

import pytest

from redteam.generate.attacks import GENERATORS, generator_for
from redteam.identify.library import load_library, summarise
from redteam.schema import observable_columns


def test_library_validates(library):
    assert library.validate() == []


def test_every_simulated_vector_has_a_registered_generator(library):
    for vector in library.simulated():
        assert vector.generator in GENERATORS, vector.id
        assert vector.id in generator_for(vector).vector_ids, vector.id


def test_declared_signals_exist_in_the_schema(library):
    """A vector cannot claim evidence the pipeline is incapable of producing."""
    observable = set(observable_columns())
    for vector in library.vectors:
        assert set(vector.signals) <= observable, vector.id


def test_breadth_across_families_and_rails(library):
    summary = summarise(library)
    assert summary["total_vectors"] >= 60
    assert len(summary["families"]) >= 8
    # Every family has at least one vector wired to a generator, or the family is
    # documentation rather than a testable claim.
    unsimulated = [f for f, s in summary["families"].items() if s["simulated"] == 0]
    assert unsimulated == ["CRYPTO"], unsimulated


def test_default_mix_is_a_distribution_over_simulated_vectors(library):
    mix = library.default_mix()
    assert set(mix) == {v.id for v in library.simulated()}
    assert sum(mix.values()) == pytest.approx(1.0)
    assert all(w > 0 for w in mix.values())


def test_mix_override_rejects_unknown_and_unsimulated_vectors(library):
    with pytest.raises(KeyError):
        library.resolve_mix({"NOT-A-VECTOR": 1.0})

    not_simulated = next(v for v in library.vectors if not v.simulated)
    with pytest.raises(ValueError):
        library.resolve_mix({not_simulated.id: 1.0})


def test_mix_override_renormalises(library):
    picked = [v.id for v in library.simulated()[:3]]
    mix = library.resolve_mix({picked[0]: 2.0, picked[1]: 1.0, picked[2]: 1.0})
    assert sum(mix.values()) == pytest.approx(1.0)
    assert mix[picked[0]] == pytest.approx(0.5)


def test_unknown_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "library.yaml"
    path.write_text(
        "version: '1'\nfamilies: {APP: {}}\nkill_chain: [target]\n"
        "vectors:\n  - id: X\n    name: x\n    family: APP\n"
        "    description: x\n    typo_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_library(path)


def test_risk_score_is_bounded(library):
    assert all(0.0 <= v.risk_score <= 1.0 for v in library.vectors)
