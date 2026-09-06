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


def test_ca3_recurrent_learning_strengthens_only_existing_excitatory_routes_with_bounded_output():
    memory = ParallelBundleMemoryNetwork(
        cortical_size=32,
        cortical_winner_count=8,
        dg_size=256,
        dg_winner_count=24,
        dg_fan_out=24,
        ca3_size=64,
        ca3_winner_count=8,
        ca3_direct_seed_count=2,
        ca3_direct_fan_out=4,
        ca3_dg_fan_in=12,
        ca3_recurrent_fan_out=8,
        ca3_recurrent_output_budget=0.5,
        seed=917,
    )
    cue = _bundle(np.arange(0, 8), 0.0)

    for _ in range(6):
        memory.advance(cue, learn=True)

    weights = memory.ca3_recurrent_weights
    rows = weights.reshape(memory.ca3_size, memory.ca3_recurrent_fan_out)

    assert np.any(weights > 0.0)
    assert np.all(weights >= 0.0)
    assert np.all(np.sum(rows, axis=1) <= 0.500001)
    assert weights.size == memory.ca3_size * memory.ca3_recurrent_fan_out
