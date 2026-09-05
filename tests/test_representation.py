import numpy as np
import pytest

from sonara import SonaraRuntime
from sonara.neural import NeuronPopulation, SparseRepresentationField, SynapseProjection


STIMULUS = np.array([1.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.1, 0.0])
SIMILAR = np.array([0.60, 0.23, -0.03, -0.386, 0.524, -0.02, 0.212, -0.337])
DIFFERENT = np.array([0.0, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0, 0.4])


def make_field() -> SparseRepresentationField:
    return SparseRepresentationField(
        "sensory",
        "R",
        population_size=96,
        feature_size=8,
        winner_count=8,
        seed=11,
        learning_rate=0.08,
    )


def train(field: SparseRepresentationField, repeats: int = 6) -> None:
    for _ in range(repeats):
        field.encode(STIMULUS, learn=True)


def test_sparse_assemblies_preserve_identity_and_similarity_structure():
    field = make_field()
    train(field)
    first = field.encode(STIMULUS)
    repeated = field.encode(STIMULUS)
    similar = field.encode(SIMILAR)
    different = field.encode(DIFFERENT)

    assert first.overlap(repeated) == 1.0
    assert 0.50 <= first.overlap(similar) < 1.0
    assert first.overlap(different) <= 0.25
    assert first.overlap(similar) > first.overlap(different)
    assert first.winner_indices.size == 8
    assert first.sparsity == pytest.approx(8 / 96)


def test_learning_changes_future_neural_drive():
    field = make_field()
    before = field.encode(STIMULUS)
    before_weights = field.weights
    train(field)
    after = field.encode(STIMULUS)

    assert after.mean_winner_score > before.mean_winner_score
    assert after.mean_winner_current > before.mean_winner_current
    assert not np.allclose(field.weights, before_weights)


def test_live_stimulus_assembly_drives_existing_projection():
    runtime = SonaraRuntime()
    runtime.add_population(NeuronPopulation("R", 96))
    runtime.add_population(NeuronPopulation("D", 96))
    field = make_field()
    train(field)
    runtime.add_representation_field(field)
    runtime.add_projection(SynapseProjection.one_to_one("R_D", "R", "D", 96, 16.0, delay_ms=1.0))

    t1 = runtime.present_stimulus("sensory", STIMULUS, learn=False)
    t2 = runtime.tick(1.0)

    assert t1.activation.winner_indices.size == 8
    assert t1.tick.spike_counts()["R"] == 8
    assert t1.tick.spike_counts()["D"] == 0
    assert t2.spike_counts()["D"] == 8


def test_representation_does_not_mutate_unrelated_population():
    runtime = SonaraRuntime()
    runtime.add_population(NeuronPopulation("R", 96))
    unrelated = NeuronPopulation("U", 16)
    runtime.add_population(unrelated)
    runtime.add_representation_field(make_field())
    before = unrelated.membrane_mv.copy()

    runtime.present_stimulus("sensory", STIMULUS, learn=True)
    runtime.tick(1.0)

    assert np.all(unrelated.spike_count == 0)
    assert np.allclose(unrelated.membrane_mv, before)


def test_malformed_stimulus_and_field_registration_fail_safely():
    field = make_field()
    with pytest.raises(ValueError, match="features must have shape"):
        field.encode(np.ones(7))
    with pytest.raises(ValueError, match="zero-length stimulus"):
        field.encode(np.zeros(8))

    runtime = SonaraRuntime()
    with pytest.raises(ValueError, match="unknown population"):
        runtime.add_representation_field(field)
