import numpy as np

from sonara.experiments.cortical_sheet import StreamSpec
from sonara.experiments.feedback_cortical_sheet import FeedbackCorticalSheet


def _sheet(decay):
    return FeedbackCorticalSheet(
        4,
        4,
        (StreamSpec("signal", 4, (0.2, 0.5), gain=4.0),),
        sparsity=0.25,
        seed=1701,
        local_degree=3,
        long_range_degree=1,
        recurrent_learning_rate=0.20,
        recurrent_eligibility_decay=decay,
    )


def test_prior_cortical_source_can_strengthen_route_when_target_joins_next_bundle():
    persistent = _sheet(0.8)
    instantaneous = _sheet(0.0)

    edge = 0
    source = int(persistent.source_edges[edge])
    target = int(persistent.target_edges[edge])
    assert source != target
    initial = float(persistent.recurrent_weights[edge])
    assert np.isclose(initial, instantaneous.recurrent_weights[edge])

    for sheet in (persistent, instantaneous):
        sheet.activity.fill(0.0)
        sheet.activity[source] = 1.0
        sheet._update_recurrent_eligibility()

        sheet.activity.fill(0.0)
        sheet.activity[target] = 1.0
        sheet._update_recurrent_eligibility()
        sheet._learn_temporal_recurrent_edges()

    assert persistent.recurrent_eligibility[source] > 0.0
    assert instantaneous.recurrent_eligibility[source] == 0.0
    assert persistent.recurrent_weights[edge] > initial
    assert np.isclose(instantaneous.recurrent_weights[edge], initial)


def test_cortical_cascade_eligibility_is_transient_not_stored_memory():
    sheet = _sheet(0.5)
    source = int(sheet.source_edges[0])
    weights_before = sheet.recurrent_weights.copy()

    sheet.activity[source] = 1.0
    sheet._update_recurrent_eligibility()
    assert np.isclose(sheet.recurrent_eligibility[source], 1.0)

    sheet.activity.fill(0.0)
    sheet._update_recurrent_eligibility()
    assert np.isclose(sheet.recurrent_eligibility[source], 0.5)

    sheet.reset_state()
    assert not np.any(sheet.recurrent_eligibility)
    assert np.array_equal(sheet.recurrent_weights, weights_before)
