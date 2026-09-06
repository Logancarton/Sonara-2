import numpy as np
import pytest

from sonara.experiments.cortical_sheet_benchmark import completion_trial, normalize
from sonara.experiments.feedback_cortical_sheet import FeedbackCorticalSheet
from sonara.experiments.cortical_sheet import StreamSpec
from sonara.experiments.parallel_bundle_benchmark import live_parallel_bundle_trial
from sonara.experiments.parallel_bundle_memory import ParallelBundleMemoryNetwork


def test_live_cortex_and_hippocampal_branch_advance_as_delayed_broad_bundles():
    rng = np.random.default_rng(401)
    sheet = FeedbackCorticalSheet(
        16,
        16,
        (StreamSpec("signal", 8, (0.2, 0.5), sigma=0.30, gain=6.0),),
        sparsity=0.06,
        seed=402,
        local_degree=8,
        long_range_degree=2,
    )
    memory = ParallelBundleMemoryNetwork(
        sheet.size,
        cortical_winner_count=sheet.winner_budget,
        dg_size=512,
        dg_winner_count=32,
        ca3_size=128,
        ca3_winner_count=16,
        seed=403,
    )
    cue = normalize(rng.normal(size=8))

    feedback = None
    history = []
    for _ in range(5):
        cortical = sheet.step(
            {"signal": cue},
            feedback_bundle=feedback,
            feedback_gain=2.0,
            learn=True,
        )
        hippocampal = memory.advance(cortical.bundle, learn=True)
        feedback = hippocampal.cortical_return
        history.append((cortical, hippocampal))

    assert history[0][0].bundle.width > 1
    assert history[0][1].dentate.width > 1
    assert history[0][1].ca3.width == 0

    assert history[1][0].bundle.width > 1
    assert history[1][1].dentate.width > 1
    assert history[1][1].ca3.width > 1

    assert history[-1][0].bundle.width > 1
    assert history[-1][1].cortical_return.width > 1


def test_live_parallel_bundle_cascade_recovers_identity_better_with_memory_return():
    rows = np.asarray([live_parallel_bundle_trial(seed) for seed in range(4)])
    serial = np.asarray([completion_trial(seed, True) for seed in range(4)])

    cortical_accuracy = float(np.mean(rows[:, 0]))
    cortical_margin = float(np.mean(rows[:, 1]))
    no_return_accuracy = float(np.mean(rows[:, 2]))
    no_return_margin = float(np.mean(rows[:, 3]))
    dg_accuracy = float(np.mean(rows[:, 4]))
    ca3_accuracy = float(np.mean(rows[:, 5]))
    ca3_margin = float(np.mean(rows[:, 6]))
    serial_accuracy = float(np.mean(serial[:, 0]))

    assert cortical_accuracy >= 0.50, rows
    assert cortical_accuracy >= no_return_accuracy + 0.15, (
        cortical_accuracy,
        no_return_accuracy,
    )
    assert cortical_accuracy >= serial_accuracy + 0.15, (
        cortical_accuracy,
        serial_accuracy,
    )
    assert cortical_margin >= no_return_margin + 0.05, (
        cortical_margin,
        no_return_margin,
    )
    assert dg_accuracy >= 0.50, rows
    assert ca3_accuracy >= 0.50, rows
    assert ca3_margin > 0.0, rows


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A moving broad-bundle memory cascade is not required to end in a "
        "family-specific static final frame. This diagnostic remains visible "
        "until static final-state identity is independently demonstrated."
    ),
)
def test_static_final_frame_identity_is_not_yet_claimed():
    rows = np.asarray([live_parallel_bundle_trial(seed) for seed in range(2)])
    assert float(np.mean(rows[:, 7])) >= 0.50, rows
