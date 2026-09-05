import numpy as np
import pytest

from sonara import SonaraRuntime
from sonara.neural import NeuronPopulation, SynapseProjection
from sonara.neural.projection import ProjectionLearning


def build_cascade() -> SonaraRuntime:
    runtime = SonaraRuntime()
    for population_id in ("A", "B", "C", "D"):
        runtime.add_population(NeuronPopulation(population_id, 8))
    runtime.add_projection(SynapseProjection.one_to_one("A_B", "A", "B", 8, 16.0, delay_ms=1.0))
    runtime.add_projection(SynapseProjection.one_to_one("A_C", "A", "C", 8, 7.0, delay_ms=1.0))
    runtime.add_projection(SynapseProjection.one_to_one("B_D", "B", "D", 8, 16.0, delay_ms=1.0))
    return runtime


def test_live_graph_drives_recursive_cascade_without_runtime_route_logic():
    runtime = build_cascade()

    t1 = runtime.tick(1.0, {"A": 16.0})
    t2 = runtime.tick(1.0)
    t3 = runtime.tick(1.0)

    assert t1.spike_counts() == {"A": 8, "B": 0, "C": 0, "D": 0}
    assert t2.spike_counts()["B"] == 8
    assert t2.spike_counts()["C"] == 0
    assert t3.spike_counts()["D"] == 8


def test_weak_competing_projection_receives_signal_but_fails_threshold():
    runtime = build_cascade()
    runtime.tick(1.0, {"A": 16.0})
    t2 = runtime.tick(1.0)

    assert t2.delivered_currents["B"] > t2.delivered_currents["C"] > 0.0
    assert t2.spike_counts()["B"] == 8
    assert t2.spike_counts()["C"] == 0


def test_local_causal_learning_strengthens_existing_projection():
    runtime = SonaraRuntime()
    runtime.add_population(NeuronPopulation("A", 4))
    runtime.add_population(NeuronPopulation("B", 4))
    projection = SynapseProjection.one_to_one(
        "A_B",
        "A",
        "B",
        4,
        14.0,
        delay_ms=1.0,
        learning=ProjectionLearning(learning_rate=0.20, max_weight=20.0),
    )
    runtime.add_projection(projection)
    before = projection.weights.copy()

    runtime.tick(1.0, {"A": 16.0})
    t2 = runtime.tick(1.0, {"B": 2.0})

    assert t2.spike_counts()["B"] == 4
    assert t2.plastic_synapses_updated == 4
    assert np.all(projection.weights > before)


def test_unrelated_population_is_not_mutated_by_other_route():
    runtime = build_cascade()
    e = NeuronPopulation("E", 8)
    runtime.add_population(e)
    before = e.membrane_mv.copy()

    runtime.tick(1.0, {"A": 16.0})
    runtime.tick(1.0)
    runtime.tick(1.0)

    assert np.all(e.spike_count == 0)
    assert np.allclose(e.membrane_mv, before)


def test_malformed_projection_fails_before_runtime_execution():
    runtime = SonaraRuntime()
    runtime.add_population(NeuronPopulation("A", 2))
    runtime.add_population(NeuronPopulation("B", 2))
    malformed = SynapseProjection(
        "bad",
        "A",
        "B",
        np.array([0, 3]),
        np.array([0, 1]),
        1.0,
    )

    with pytest.raises(ValueError, match="source index outside"):
        runtime.add_projection(malformed)
