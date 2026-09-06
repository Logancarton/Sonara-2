import numpy as np

from sonara.experiments.cortical_sheet import StreamSpec, assembly_overlap
from sonara.experiments.feedback_cortical_sheet import FeedbackCorticalSheet


def _sheet(decay):
    return FeedbackCorticalSheet(
        8,
        8,
        (StreamSpec("signal", 8, (0.5, 0.5), sigma=0.45, gain=8.0),),
        sparsity=0.125,
        seed=1901,
        local_degree=8,
        long_range_degree=4,
        hot_tau_ms=4.0,
        cold_tau_ms=12.0,
        firing_threshold=0.03,
        afferent_learning_rate=0.0,
        recurrent_learning_rate=0.10,
        homeostatic_pressure=0.0,
        recurrent_eligibility_decay=decay,
    )


def _response(sheet, vector):
    sheet.reset_state(reset_usage=True)
    return sheet.step({"signal": vector}, recurrent=False).winner_indices.copy()


def test_repeated_a_then_b_cascade_makes_a_recruit_b_more_with_temporal_plasticity():
    persistent = _sheet(0.82)
    instantaneous = _sheet(0.0)

    basis = np.eye(8, dtype=np.float32)
    a = basis[0]
    a_winners = _response(persistent, a)

    candidates = []
    for index in range(1, 8):
        winners = _response(persistent, basis[index])
        overlap = assembly_overlap(a_winners, winners, persistent.winner_budget)
        candidates.append((overlap, index, winners))
    _, b_index, b_winners = min(candidates, key=lambda item: item[0])
    b = basis[b_index]

    for _ in range(40):
        for sheet in (persistent, instantaneous):
            sheet.reset_state(reset_usage=True)
            sheet.step({"signal": a}, learn=True, recurrent=False)
            sheet.step({"signal": b}, learn=True, recurrent=False)

    persistent.reset_state(reset_usage=True)
    instantaneous.reset_state(reset_usage=True)
    persistent_a = persistent.step({"signal": a}, recurrent=False).bundle
    instantaneous_a = instantaneous.step({"signal": a}, recurrent=False).bundle

    persistent_projection = persistent.project_bundle(persistent_a).as_dense(persistent.size)
    instantaneous_projection = instantaneous.project_bundle(instantaneous_a).as_dense(
        instantaneous.size
    )
    persistent_b_drive = float(np.mean(persistent_projection[b_winners]))
    instantaneous_b_drive = float(np.mean(instantaneous_projection[b_winners]))

    diagnostic = (
        f"persistent_b_drive={persistent_b_drive:.6f} "
        f"instantaneous_b_drive={instantaneous_b_drive:.6f} "
        f"a_b_overlap={assembly_overlap(a_winners, b_winners, persistent.winner_budget):.3f}"
    )
    assert persistent_b_drive >= instantaneous_b_drive + 0.01, diagnostic
