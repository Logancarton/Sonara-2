import numpy as np

from sonara.experiments.cortical_sheet import SignalBundle
from sonara.experiments.parallel_bundle_memory import ParallelBundleMemoryNetwork


def _bundle(indices, emitted_ms):
    indices = np.asarray(indices, dtype=np.int64)
    return SignalBundle(
        indices,
        np.ones(indices.size, dtype=np.float32),
        emitted_ms,
    )


def _memory(*, eligibility_decay, cortical_trace_decay=0.0):
    return ParallelBundleMemoryNetwork(
        cortical_size=8,
        cortical_winner_count=2,
        dg_size=16,
        dg_winner_count=16,
        dg_fan_out=8,
        cortical_trace_decay=cortical_trace_decay,
        eligibility_decay=eligibility_decay,
        dg_homeostatic_pressure=0.0,
        ca3_size=32,
        ca3_winner_count=4,
        ca3_direct_seed_count=2,
        ca3_direct_fan_out=4,
        ca3_dg_fan_in=8,
        ca3_recurrent_fan_out=8,
        seed=1401,
    )


def test_temporally_adjacent_bundles_strengthen_an_existing_route_through_eligibility_overlap():
    persistent = _memory(eligibility_decay=0.8)
    instantaneous = _memory(eligibility_decay=0.0)

    source_a = 0
    source_b = 1
    targets_a = persistent.dg_target_edges[
        persistent.dg_source_edges == source_a
    ]
    targets_b = persistent.dg_target_edges[
        persistent.dg_source_edges == source_b
    ]
    shared = np.intersect1d(targets_a, targets_b)
    assert shared.size > 0
    target = int(shared[0])
    edge = int(
        np.flatnonzero(
            (persistent.dg_source_edges == source_a)
            & (persistent.dg_target_edges == target)
        )[0]
    )

    persistent.advance(_bundle([source_a], 0.0), learn=True)
    instantaneous.advance(_bundle([source_a], 0.0), learn=True)
    after_first = float(persistent.dg_edge_weights[edge])
    assert np.isclose(after_first, instantaneous.dg_edge_weights[edge])

    # Source A is gone from the present cue. Only its eligibility trace remains.
    persistent.advance(_bundle([source_b], 1.0), learn=True)
    instantaneous.advance(_bundle([source_b], 1.0), learn=True)

    assert persistent.cortical_eligibility[source_a] > 0.0
    assert instantaneous.cortical_eligibility[source_a] == 0.0
    assert persistent.dg_edge_weights[edge] > instantaneous.dg_edge_weights[edge]


def test_unsupported_cascade_activity_really_fades_and_reset_clears_transient_state():
    memory = _memory(eligibility_decay=0.5, cortical_trace_decay=0.5)
    cue = _bundle([0], 0.0)
    empty = _bundle([], 1.0)

    memory.advance(cue, learn=False)
    first_trace = float(memory.cortical_trace[0])
    first_eligibility = float(memory.cortical_eligibility[0])
    memory.advance(empty, learn=False)
    second_trace = float(memory.cortical_trace[0])
    second_eligibility = float(memory.cortical_eligibility[0])

    assert np.isclose(first_trace, 1.0)
    assert np.isclose(first_eligibility, 1.0)
    assert np.isclose(second_trace, 0.5)
    assert np.isclose(second_eligibility, 0.5)

    learned_weights = memory.dg_edge_weights.copy()
    memory.reset_dynamic()
    assert not np.any(memory.cortical_trace)
    assert not np.any(memory.cortical_eligibility)
    assert not np.any(memory.dentate_eligibility)
    assert not np.any(memory.ca3_eligibility)
    assert np.array_equal(memory.dg_edge_weights, learned_weights)
