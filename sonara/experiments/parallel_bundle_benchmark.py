from __future__ import annotations

import numpy as np

from .cortical_sheet import FastCorticalSheet, SignalBundle, assembly_overlap
from .cortical_sheet_benchmark import (
    default_streams,
    experience_prototypes,
    noisy_experience,
)
from .parallel_bundle_memory import ParallelBundleMemoryNetwork, ParallelBundleStep


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


def _mean_between(assemblies: list[np.ndarray], budget: int) -> float:
    values: list[float] = []
    for left in range(len(assemblies)):
        for right in range(left + 1, len(assemblies)):
            values.append(assembly_overlap(assemblies[left], assemblies[right], budget))
    return float(np.mean(values)) if values else 0.0


def _trajectory_signature(
    trajectory: tuple[ParallelBundleStep, ...],
    *,
    population: str,
    size: int,
    start_tick: int = 2,
) -> np.ndarray:
    """Flatten post-cue bundle activity so identity can live across time, not one frame."""
    if population not in {"cortical", "ca3"}:
        raise ValueError("population must be 'cortical' or 'ca3'")
    if not 0 <= start_tick < len(trajectory):
        raise ValueError("start_tick outside trajectory")

    frames: list[np.ndarray] = []
    for step in trajectory[start_tick:]:
        bundle = getattr(step, population)
        dense = bundle.as_dense(size).astype(np.float32)
        norm = float(np.linalg.norm(dense))
        if norm > 1e-12:
            dense /= norm
        frames.append(dense)
    signature = np.concatenate(frames)
    norm = float(np.linalg.norm(signature))
    if norm > 1e-12:
        signature /= norm
    return signature


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def parallel_bundle_separation_diagnostic(seed: int) -> tuple[float, float, float]:
    """Measure where distinct full experiences collapse along the parallel path."""
    feature_size = 12
    families = 6
    rng, sheet, memory, sensory, other = _trained_parallel_system(seed)

    dg_references: list[np.ndarray] = []
    initial_ca3_references: list[np.ndarray] = []
    settled_ca3_references: list[np.ndarray] = []

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

        memory.reset_dynamic()
        memory.advance(full_bundle)
        second = memory.advance(full_bundle)
        third = memory.advance(full_bundle)
        dg_references.append(second.dentate.indices.copy())
        initial_ca3_references.append(third.ca3.indices.copy())
        settled_ca3_references.append(
            memory.recall(full_bundle, steps=10, driven_steps=3).ca3.indices.copy()
        )

    return (
        _mean_between(dg_references, memory.dg_winner_count),
        _mean_between(initial_ca3_references, memory.ca3_winner_count),
        _mean_between(settled_ca3_references, memory.ca3_winner_count),
    )


def parallel_bundle_trajectory_trial(
    seed: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Score the whole post-cue cascade rather than forcing memory into a final frame.

    Full and degraded cues use the same two driven ticks. The hippocampal return
    is ablated on a second pass through the exact same learned network so any
    improvement is attributable to the learned return branch, not the cue.
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
        trajectory = memory.recall_trajectory(
            full_bundle,
            steps=8,
            driven_steps=2,
            memory_return=True,
        )
        cortical_references.append(
            _trajectory_signature(
                trajectory,
                population="cortical",
                size=memory.cortical_size,
            )
        )
        ca3_references.append(
            _trajectory_signature(
                trajectory,
                population="ca3",
                size=memory.ca3_size,
            )
        )

    memory_correct = 0
    no_return_correct = 0
    ca3_correct = 0
    memory_margins: list[float] = []
    no_return_margins: list[float] = []
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

        memory_trajectory = memory.recall_trajectory(
            partial,
            steps=8,
            driven_steps=2,
            memory_return=True,
        )
        no_return_trajectory = memory.recall_trajectory(
            partial,
            steps=8,
            driven_steps=2,
            memory_return=False,
        )

        cortical_signature = _trajectory_signature(
            memory_trajectory,
            population="cortical",
            size=memory.cortical_size,
        )
        no_return_signature = _trajectory_signature(
            no_return_trajectory,
            population="cortical",
            size=memory.cortical_size,
        )
        ca3_signature = _trajectory_signature(
            memory_trajectory,
            population="ca3",
            size=memory.ca3_size,
        )

        memory_scores = np.asarray(
            [_cosine(reference, cortical_signature) for reference in cortical_references]
        )
        no_return_scores = np.asarray(
            [_cosine(reference, no_return_signature) for reference in cortical_references]
        )
        ca3_scores = np.asarray(
            [_cosine(reference, ca3_signature) for reference in ca3_references]
        )

        memory_correct += int(np.argmax(memory_scores) == family)
        no_return_correct += int(np.argmax(no_return_scores) == family)
        ca3_correct += int(np.argmax(ca3_scores) == family)

        memory_margins.append(
            memory_scores[family] - np.max(np.delete(memory_scores, family))
        )
        no_return_margins.append(
            no_return_scores[family] - np.max(np.delete(no_return_scores, family))
        )
        ca3_margins.append(
            ca3_scores[family] - np.max(np.delete(ca3_scores, family))
        )

    return (
        memory_correct / families,
        float(np.mean(memory_margins)),
        no_return_correct / families,
        float(np.mean(no_return_margins)),
        ca3_correct / families,
        float(np.mean(ca3_margins)),
    )


def parallel_bundle_completion_trial(seed: int) -> tuple[float, float, float, float]:
    """Legacy final-frame diagnostic retained to expose attractor collapse."""
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
