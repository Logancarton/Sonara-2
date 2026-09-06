from __future__ import annotations

import numpy as np

from .cortical_sheet import FastCorticalSheet, SignalBundle, assembly_overlap
from .cortical_sheet_benchmark import (
    default_streams,
    experience_prototypes,
    noisy_experience,
)
from .parallel_bundle_memory import ParallelBundleMemoryNetwork


def _trained_parallel_system(
    seed: int,
) -> tuple[
    np.random.Generator,
    FastCorticalSheet,
    ParallelBundleMemoryNetwork,
    list[np.ndarray],
    dict[str, list[np.ndarray]],
]:
    rng = np.random.default_rng(seed)
    feature_size = 12
    families = 6
    sheet = FastCorticalSheet(
        32,
        32,
        default_streams(feature_size),
        sparsity=0.05,
        seed=200 + seed,
        recurrent_learning_rate=0.20,
        recurrence_cold_gain=2.0,
    )
    memory = ParallelBundleMemoryNetwork(
        sheet.size,
        cortical_winner_count=sheet.winner_budget,
        cortical_projector=sheet.project_bundle,
        seed=1200 + seed,
    )
    sensory, other = experience_prototypes(rng, families, feature_size)

    order = np.tile(np.arange(families), 20)
    rng.shuffle(order)
    for family in order:
        sheet.reset_state()
        sheet.settle(
            noisy_experience(
                rng,
                sensory,
                other,
                int(family),
                feature_size,
                ("sensory", "context", "body", "time"),
                noise=0.10,
            ),
            cue_steps=5,
            learn=True,
            recurrent=True,
        )
        memory.learn_experience(sheet.current_bundle(), steps=8, driven_steps=5)

    return rng, sheet, memory, sensory, other


def parallel_bundle_completion_trial(seed: int) -> tuple[float, float, float, float]:
    """
    Evaluate distributed cortical and CA3 identity recovery from a partial cue.

    Ground-truth family labels are used only by the benchmark to score outputs;
    the memory network never receives them.
    """
    feature_size = 12
    families = 6
    rng, sheet, memory, sensory, other = _trained_parallel_system(seed)

    cortical_references: list[np.ndarray] = []
    ca3_references: list[np.ndarray] = []
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
        full_bundle = sheet.current_bundle()
        cortical_references.append(full_bundle.indices.copy())
        ca3_references.append(
            memory.recall(full_bundle, steps=10, driven_steps=3).ca3.indices.copy()
        )

    cortical_correct = 0
    ca3_correct = 0
    cortical_margins: list[float] = []
    ca3_margins: list[float] = []

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
        partial = sheet.current_bundle()
        recalled = memory.recall(partial, steps=10, driven_steps=2)

        cortical_overlaps = np.asarray(
            [
                assembly_overlap(
                    reference,
                    recalled.cortical.indices,
                    sheet.winner_budget,
                )
                for reference in cortical_references
            ]
        )
        cortical_correct += int(np.argmax(cortical_overlaps) == family)
        cortical_margins.append(
            cortical_overlaps[family]
            - np.max(np.delete(cortical_overlaps, family))
        )

        ca3_overlaps = np.asarray(
            [
                assembly_overlap(
                    reference,
                    recalled.ca3.indices,
                    memory.ca3_winner_count,
                )
                for reference in ca3_references
            ]
        )
        ca3_correct += int(np.argmax(ca3_overlaps) == family)
        ca3_margins.append(
            ca3_overlaps[family] - np.max(np.delete(ca3_overlaps, family))
        )

    return (
        cortical_correct / families,
        float(np.mean(cortical_margins)),
        ca3_correct / families,
        float(np.mean(ca3_margins)),
    )
