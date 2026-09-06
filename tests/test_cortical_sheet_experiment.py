import numpy as np
import pytest

from sonara.experiments.cortical_sheet import (
    FastCorticalSheet,
    SignalBundle,
    StreamSpec,
    assembly_overlap,
)
from sonara.experiments.cortical_sheet_benchmark import (
    _trained_hippocampal_system,
    ca3_identity_trial,
    completion_trial,
    default_streams,
    internal_ca3_completion_trial,
    noisy_experience,
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


def test_live_sheet_emits_a_broad_population_bundle_not_a_single_cell_trail():
    rng = np.random.default_rng(41)
    sheet = FastCorticalSheet(
        24,
        24,
        (StreamSpec("signal", 10, (0.15, 0.50), sigma=0.30, gain=6.0),),
        sparsity=0.05,
        seed=42,
        local_degree=12,
        long_range_degree=4,
    )
    cue = normalize(rng.normal(size=10))

    step = sheet.step({"signal": cue}, recurrent=False)

    assert step.bundle.width == step.winner_indices.size
    assert step.bundle.width > 1
    np.testing.assert_array_equal(
        np.sort(step.bundle.indices),
        np.sort(step.winner_indices),
    )
    np.testing.assert_allclose(step.bundle.as_dense(sheet.size), sheet.activity)

    projected = sheet.project_bundle(step.bundle)
    assert projected.width > step.bundle.width * 2
    assert projected.total_activity > 0.0


def test_separate_broad_bundles_reconverge_by_summing_on_shared_targets():
    sheet = FastCorticalSheet(
        24,
        24,
        (StreamSpec("signal", 4, (0.50, 0.50)),),
        seed=53,
        local_degree=16,
        long_range_degree=4,
    )
    left = SignalBundle(
        np.arange(200, 212, dtype=np.int64),
        np.linspace(0.5, 1.0, 12, dtype=np.float32),
        1.0,
    )
    right = SignalBundle(
        np.arange(212, 224, dtype=np.int64),
        np.linspace(1.0, 0.5, 12, dtype=np.float32),
        1.0,
    )

    left_out = sheet.project_bundle(left)
    right_out = sheet.project_bundle(right)
    merged = SignalBundle.merge((left_out, right_out), size=sheet.size, emitted_ms=2.0)

    left_dense = left_out.as_dense(sheet.size)
    right_dense = right_out.as_dense(sheet.size)
    merged_dense = merged.as_dense(sheet.size)
    shared = np.flatnonzero((left_dense > 0.0) & (right_dense > 0.0))

    assert left_out.width > left.width
    assert right_out.width > right.width
    assert shared.size > 0
    np.testing.assert_allclose(merged_dense, left_dense + right_dense, rtol=1e-6, atol=1e-7)
    assert np.all(merged_dense[shared] > left_dense[shared])
    assert np.all(merged_dense[shared] > right_dense[shared])


def test_broad_partial_bundle_preserves_more_ca3_trace_than_one_cell_trail():
    broad_overlaps: list[float] = []
    narrow_overlaps: list[float] = []

    for seed in range(4):
        feature_size = 12
        families = 6
        rng, sheet, hippocampus, sensory, other = _trained_hippocampal_system(seed)

        full_ca3: list[np.ndarray] = []
        for family in range(families):
            sheet.reset_state()
            sheet.settle(
                noisy_experience(
                    rng,
                    sensory,
                    other,
                    family,
                    feature_size,
                    ("sensory", "context", "body", "time"),
                    noise=0.02,
                ),
                cue_steps=5,
                recurrent=True,
            )
            full_ca3.append(
                hippocampus.recall(
                    sheet.state.copy(), recurrent=True, settle_steps=5
                ).ca3_winner_indices
            )

        for family in range(families):
            sheet.reset_state()
            sheet.settle(
                noisy_experience(
                    rng,
                    sensory,
                    other,
                    family,
                    feature_size,
                    ("sensory", "context"),
                    noise=0.08,
                ),
                cue_steps=2,
                recurrent=True,
            )

            active = np.flatnonzero(sheet.activity > 0.0)
            broad = SignalBundle(
                active,
                sheet.activity[active],
                sheet.time_ms,
            )
            strongest = int(active[np.argmax(sheet.activity[active])])
            narrow = SignalBundle(
                np.asarray([strongest], dtype=np.int64),
                np.asarray([sheet.activity[strongest]], dtype=np.float32),
                sheet.time_ms,
            )

            broad_recall = hippocampus.recall(
                broad.as_dense(sheet.size), recurrent=True, settle_steps=5
            )
            narrow_recall = hippocampus.recall(
                narrow.as_dense(sheet.size), recurrent=True, settle_steps=5
            )
            broad_overlaps.append(
                assembly_overlap(
                    full_ca3[family],
                    broad_recall.ca3_winner_indices,
                    hippocampus.ca3_assembly_size,
                )
            )
            narrow_overlaps.append(
                assembly_overlap(
                    full_ca3[family],
                    narrow_recall.ca3_winner_indices,
                    hippocampus.ca3_assembly_size,
                )
            )

    broad_mean = float(np.mean(broad_overlaps))
    narrow_mean = float(np.mean(narrow_overlaps))
    assert broad_mean > narrow_mean + 0.10, (broad_mean, narrow_mean)


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