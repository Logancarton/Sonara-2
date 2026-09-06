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


def test_same_current_cortical_bundle_reaches_dg_differently_after_different_recent_history():
    kwargs = dict(
        cortical_size=32,
        cortical_winner_count=8,
        dg_size=256,
        dg_winner_count=24,
        dg_fan_out=24,
        cortical_trace_decay=0.75,
        ca3_size=64,
        ca3_winner_count=8,
        ca3_direct_seed_count=2,
        ca3_direct_fan_out=4,
        ca3_dg_fan_in=12,
        seed=811,
    )
    with_history = ParallelBundleMemoryNetwork(**kwargs)
    current_only = ParallelBundleMemoryNetwork(**kwargs)

    prior = _bundle(np.arange(0, 8), 0.0)
    current = _bundle(np.arange(16, 24), 1.0)

    with_history.advance(prior)
    history_response = with_history.advance(current).dentate
    current_response = current_only.advance(current).dentate

    assert history_response.width == current_response.width == 24
    assert not np.array_equal(history_response.indices, current_response.indices)

    with_history.reset_dynamic()
    reset_response = with_history.advance(current).dentate
    assert np.array_equal(reset_response.indices, current_response.indices)
