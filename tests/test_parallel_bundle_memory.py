import copy

import numpy as np
import pytest

from sonara.experiments.cortical_sheet_benchmark import (
    completion_trial,
    noisy_experience,
    normalize,
)
from sonara.experiments.feedback_cortical_sheet import FeedbackCorticalSheet
from sonara.experiments.cortical_sheet import StreamSpec
from sonara.experiments.parallel_bundle_benchmark import (
    _identity_metrics,
    _run_coupled,
    _trained_coupled_system,
    live_parallel_bundle_trial,
)
from sonara.experiments.parallel_bundle_memory import ParallelBundleMemoryNetwork


def _normalized_frame(bundle, size):
    dense = bundle.as_dense(size).astype(np.float32)
    norm = float(np.linalg.norm(dense))
    if norm > 1e-12:
        dense /= norm
    return dense


def _first_tick_identity(seed):
    feature_size = 12
    families = 6
    rng, trained_sheet, trained_memory, sensory, other = _trained_coupled_system(seed)

    full_cortical = []
    full_dg = []
    full_ca3 = []
    partial_cortical = []
    partial_dg = []
    partial_ca3 = []

    for family in range(families):
        full = noisy_experience(
            rng,
            sensory,
            other,
            family,
            feature_size,
            ("sensory", "context", "body", "time"),
            noise=0.02,
        )
        full_step = _run_coupled(
            copy.deepcopy(trained_sheet),
            copy.deepcopy(trained_memory),
            full,
            total_steps=1,
            cue_steps=1,
            learn=False,
            memory_return=True,
        )[0]
        full_cortical.append(_normalized_frame(full_step.cortical, trained_sheet.size))
        full_dg.append(_normalized_frame(full_step.dentate, trained_memory.dg_size))
        full_ca3.append(_normalized_frame(full_step.ca3, trained_memory.ca3_size))

    for family in range(families):
        partial = noisy_experience(
            rng,
            sensory,
            other,
            family,
            feature_size,
            ("sensory", "context"),
            noise=0.08,
        )
        partial_step = _run_coupled(
            copy.deepcopy(trained_sheet),
            copy.deepcopy(trained_memory),
            partial,
            total_steps=1,
            cue_steps=1,
            learn=False,
            memory_return=True,
        )[0]
        partial_cortical.append(
            _normalized_frame(partial_step.cortical, trained_sheet.size)
        )
        partial_dg.append(_normalized_frame(partial_step.dentate, trained_memory.dg_size))
        partial_ca3.append(_normalized_frame(partial_step.ca3, trained_memory.ca3_size))

    cortical_accuracy, _ = _identity_metrics(full_cortical, partial_cortical)
    dg_accuracy, _ = _identity_metrics(full_dg, partial_dg)
    ca3_accuracy, _ = _identity_metrics(full_ca3, partial_ca3)
    return cortical_accuracy, dg_accuracy, ca3_accuracy


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
    assert history[0][1].ca3.width == memory.ca3_winner_count

    assert history[1][0].bundle.width > 1
    assert history[1][1].dentate.width > 1
    assert history[1][1].ca3.width == memory.ca3_winner_count

    assert history[-1][0].bundle.width > 1
    assert history[-1][1].cortical_return.width > 1


def test_live_parallel_bundle_cascade_recovers_identity_better_with_memory_return():
    rows = np.asarray([live_parallel_bundle_trial(seed) for seed in range(4)])
    first_tick = np.asarray([_first_tick_identity(seed) for seed in range(4)])
    serial = np.asarray([completion_trial(seed, True) for seed in range(4)])

    cortical_accuracy = float(np.mean(rows[:, 0]))
    cortical_margin = float(np.mean(rows[:, 1]))
    no_return_accuracy = float(np.mean(rows[:, 2]))
    no_return_margin = float(np.mean(rows[:, 3]))
    dg_accuracy = float(np.mean(rows[:, 4]))
    ca3_accuracy = float(np.mean(rows[:, 5]))
    ca3_margin = float(np.mean(rows[:, 6]))
    serial_accuracy = float(np.mean(serial[:, 0]))
    diagnostic = {
        "first_cortical_accuracy": float(np.mean(first_tick[:, 0])),
        "first_dg_accuracy": float(np.mean(first_tick[:, 1])),
        "first_ca3_accuracy": float(np.mean(first_tick[:, 2])),
        "trajectory_cortical_accuracy": cortical_accuracy,
        "trajectory_dg_accuracy": dg_accuracy,
        "trajectory_ca3_accuracy": ca3_accuracy,
        "no_return_accuracy": no_return_accuracy,
    }

    assert cortical_accuracy >= 0.50, diagnostic
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
