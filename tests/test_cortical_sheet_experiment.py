import numpy as np
import pytest

from sonara.experiments.cortical_sheet import FastCorticalSheet, StreamSpec
from sonara.experiments.cortical_sheet_benchmark import (
    ca3_identity_trial,
    completion_trial,
    default_streams,
    internal_ca3_completion_trial,
    normalize,
    separation_trial,
)


def test_independent_converging_streams_increase_ambiguous_state_separation():
    stream_sets = (
        ("sensory",),
        ("sensory", "context"),
        ("sensory", "context", "body"),
        ("sensory", "context", "body", "time"),
    )
    aggregated = [
        float(np.mean([separation_trial(seed, present) for seed in range(6)]))
        for present in stream_sets
    ]

    assert aggregated[0] < aggregated[1] < aggregated[2] < aggregated[3], aggregated
    assert aggregated[0] < 0.08
    assert aggregated[-1] > 0.40


def test_stream_anchor_moves_where_the_same_information_first_wins():
    rng = np.random.default_rng(5)
    cue = normalize(rng.normal(size=10))
    left = FastCorticalSheet(
        32,
        32,
        (StreamSpec("signal", 10, (0.10, 0.50), sigma=0.18, gain=5.0),),
        sparsity=0.04,
        seed=77,
    )
    right = FastCorticalSheet(
        32,
        32,
        (StreamSpec("signal", 10, (0.90, 0.50), sigma=0.18, gain=5.0),),
        sparsity=0.04,
        seed=77,
    )

    left_winners = left.settle({"signal": cue}, cue_steps=4, recurrent=False)
    right_winners = right.settle({"signal": cue}, cue_steps=4, recurrent=False)
    left_x, _ = left.response_centroid(left_winners)
    right_x, _ = right.response_centroid(right_winners)

    assert left_x < 0.35
    assert right_x > 0.65
    assert right_x - left_x > 0.45


def test_multistream_compartments_create_nonlinear_coincidence_drive():
    rng = np.random.default_rng(8)
    sheet = FastCorticalSheet(24, 24, default_streams(8), seed=10)
    sensory = normalize(rng.normal(size=8))
    context = normalize(rng.normal(size=8))

    sensory_drive, _, _ = sheet.convergent_drive({"sensory": sensory})
    context_drive, _, _ = sheet.convergent_drive({"context": context})
    combined_drive, _, _ = sheet.convergent_drive(
        {"sensory": sensory, "context": context}
    )

    synergy = combined_drive - sensory_drive - context_drive
    assert float(np.max(synergy)) > 0.05
    assert int(np.count_nonzero(synergy > 0.0)) > 0


def test_units_farther_from_anchored_inputs_have_longer_intrinsic_persistence():
    sheet = FastCorticalSheet(32, 32, default_streams(6), seed=13)
    hot = int(np.argmin(sheet.coldness))
    cold = int(np.argmax(sheet.coldness))
    sheet.state[hot] = 1.0
    sheet.state[cold] = 1.0

    for _ in range(20):
        sheet.step({}, recurrent=False)

    assert sheet.tau_ms[cold] > sheet.tau_ms[hot]
    assert sheet.state[cold] > sheet.state[hot] * 5.0


def test_ca3_recurrence_expands_partial_seed_toward_full_cue_state():
    recurrent = np.asarray(
        [internal_ca3_completion_trial(seed, True) for seed in range(6)]
    )
    feed_forward = np.asarray(
        [internal_ca3_completion_trial(seed, False) for seed in range(6)]
    )

    assert float(np.mean(recurrent)) >= 0.75
    assert float(np.mean(recurrent)) >= float(np.mean(feed_forward)) + 0.40


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Recurrence can expand a seed, but it must also recover the correct "
        "distinct episode rather than the same or wrong attractor."
    ),
)
def test_ca3_recurrence_must_preserve_distinct_episode_identity():
    recurrent = np.asarray([ca3_identity_trial(seed, True) for seed in range(6)])
    feed_forward = np.asarray([ca3_identity_trial(seed, False) for seed in range(6)])

    recurrent_accuracy = float(np.mean(recurrent[:, 0]))
    feed_forward_accuracy = float(np.mean(feed_forward[:, 0]))
    recurrent_margin = float(np.mean(recurrent[:, 1]))
    between_overlap = float(np.mean(recurrent[:, 2]))

    assert recurrent_accuracy >= 0.75
    assert recurrent_accuracy >= feed_forward_accuracy + 0.10
    assert recurrent_margin >= 0.10
    assert between_overlap <= 0.25


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The strongest gate requires the correct completed CA3 episode to "
        "reactivate the corresponding distributed cortical state."
    ),
)
def test_recurrence_must_eventually_complete_the_correct_partial_cue():
    recurrent_rows = np.asarray([completion_trial(seed, True) for seed in range(6)])
    frozen_rows = np.asarray([completion_trial(seed, False) for seed in range(6)])
    recurrent_accuracy = float(np.mean(recurrent_rows[:, 0]))
    frozen_accuracy = float(np.mean(frozen_rows[:, 0]))
    recurrent_margin = float(np.mean(recurrent_rows[:, 1]))
    frozen_margin = float(np.mean(frozen_rows[:, 1]))

    assert recurrent_accuracy >= 0.75
    assert recurrent_accuracy >= frozen_accuracy + 0.10
    assert recurrent_margin >= frozen_margin + 0.05
