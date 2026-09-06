import numpy as np

from sonara.experiments.cortical_sheet import FastCorticalSheet, SignalBundle, StreamSpec
from sonara.experiments.cortical_sheet_benchmark import completion_trial
from sonara.experiments.parallel_bundle_benchmark import parallel_bundle_completion_trial
from sonara.experiments.parallel_bundle_memory import ParallelBundleMemoryNetwork


def test_cortex_keeps_propagating_while_dg_and_ca3_advance_on_delayed_bundle_ticks():
    sheet = FastCorticalSheet(
        16,
        16,
        (StreamSpec("signal", 4, (0.2, 0.5)),),
        seed=401,
        local_degree=8,
        long_range_degree=2,
    )
    memory = ParallelBundleMemoryNetwork(
        sheet.size,
        cortical_winner_count=16,
        cortical_projector=sheet.project_bundle,
        dg_size=512,
        dg_winner_count=32,
        ca3_size=128,
        ca3_winner_count=16,
        seed=402,
    )
    external = SignalBundle(
        np.arange(40, 56, dtype=np.int64),
        np.linspace(0.6, 1.0, 16, dtype=np.float32),
        0.0,
    )

    first = memory.advance(external)
    second = memory.advance(external)
    third = memory.advance(external)

    assert first.cortical.width > 1
    assert first.dentate.width == 0
    assert first.ca3.width == 0

    assert second.cortical.width > 1
    assert second.dentate.width > 1
    assert second.ca3.width == 0

    assert third.cortical.width > 1
    assert third.dentate.width > 1
    assert third.ca3.width > 1


def test_parallel_bundle_memory_beats_serial_handoff_on_partial_cue_reconstruction():
    parallel = np.asarray(
        [parallel_bundle_completion_trial(seed) for seed in range(4)]
    )
    serial = np.asarray([completion_trial(seed, True) for seed in range(4)])

    parallel_cortical_accuracy = float(np.mean(parallel[:, 0]))
    parallel_cortical_margin = float(np.mean(parallel[:, 1]))
    parallel_ca3_accuracy = float(np.mean(parallel[:, 2]))
    parallel_ca3_margin = float(np.mean(parallel[:, 3]))
    serial_accuracy = float(np.mean(serial[:, 0]))
    serial_margin = float(np.mean(serial[:, 1]))

    assert parallel_cortical_accuracy >= 0.50, parallel
    assert parallel_cortical_accuracy >= serial_accuracy + 0.15, (
        parallel_cortical_accuracy,
        serial_accuracy,
    )
    assert parallel_cortical_margin >= serial_margin + 0.05, (
        parallel_cortical_margin,
        serial_margin,
    )
    assert parallel_ca3_accuracy >= 0.50, parallel
    assert parallel_ca3_margin > 0.0, parallel
