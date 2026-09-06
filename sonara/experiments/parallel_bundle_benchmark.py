from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .cortical_sheet import SignalBundle, assembly_overlap
from .cortical_sheet_benchmark import (
    default_streams,
    experience_prototypes,
    noisy_experience,
)
from .feedback_cortical_sheet import FeedbackCorticalSheet
from .parallel_bundle_memory import (
    ParallelBundleMemoryNetwork,
    ParallelBundleMemoryStep,
)


@dataclass(frozen=True)
class CoupledBundleStep:
    """One neural tick across the live cortex and hippocampal bundle branch."""

    cortical: SignalBundle
    dentate: SignalBundle
    ca3: SignalBundle
    cortical_return: SignalBundle


def _run_coupled(
    sheet: FeedbackCorticalSheet,
    memory: ParallelBundleMemoryNetwork,
    inputs: dict[str, np.ndarray],
    *,
    total_steps: int = 8,
    cue_steps: int = 5,
    learn: bool = False,
    memory_return: bool = True,
    feedback_gain: float = 2.0,
) -> tuple[CoupledBundleStep, ...]:
    if total_steps <= 0 or not 0 < cue_steps <= total_steps:
        raise ValueError("require total_steps > 0 and 0 < cue_steps <= total_steps")

    sheet.reset_state()
    memory.reset_dynamic()
    feedback: SignalBundle | None = None
    trajectory: list[CoupledBundleStep] = []

    for tick in range(total_steps):
        cortical_step = sheet.step(
            inputs if tick < cue_steps else {},
            feedback_bundle=feedback if memory_return else None,
            feedback_gain=feedback_gain,
            learn=learn,
            recurrent=True,
        )
        memory_step: ParallelBundleMemoryStep = memory.advance(
            cortical_step.bundle,
            learn=learn,
        )
        feedback = memory_step.cortical_return
        trajectory.append(
            CoupledBundleStep(
                cortical=cortical_step.bundle,
                dentate=memory_step.dentate,
                ca3=memory_step.ca3,
                cortical_return=memory_step.cortical_return,
            )
        )

    return tuple(trajectory)


def _trained_coupled_system(
    seed: int,
) -> tuple[
    np.random.Generator,
    FeedbackCorticalSheet,
    ParallelBundleMemoryNetwork,
    list[np.ndarray],
    dict[str, list[np.ndarray]],
]:
    rng = np.random.default_rng(seed)
    feature_size = 12
    families = 6
    sheet = FeedbackCorticalSheet(
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
        seed=1200 + seed,
    )
    sensory, other = experience_prototypes(rng, families, feature_size)

    order = np.tile(np.arange(families), 20)
    rng.shuffle(order)
    for family in order:
        experience = noisy_experience(
            rng,
            sensory,
            other,
            int(family),
            feature_size,
            ("sensory", "context", "body", "time"),
            noise=0.10,
        )
        _run_coupled(
            sheet,
            memory,
            experience,
            total_steps=8,
            cue_steps=5,
            learn=True,
            memory_return=True,
        )

    return rng, sheet, memory, sensory, other


def _trajectory_signature(
    trajectory: tuple[CoupledBundleStep, ...],
    *,
    population: str,
    size: int,
    start_tick: int = 2,
) -> np.ndarray:
    if population not in {"cortical", "dentate", "ca3"}:
        raise ValueError("unsupported population")
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


def _identity_metrics(
    references: list[np.ndarray],
    candidates: list[np.ndarray],
) -> tuple[float, float]:
    correct = 0
    margins: list[float] = []
    for family, candidate in enumerate(candidates):
        scores = np.asarray([_cosine(reference, candidate) for reference in references])
        correct += int(np.argmax(scores) == family)
        margins.append(scores[family] - np.max(np.delete(scores, family)))
    return correct / len(candidates), float(np.mean(margins))


def live_parallel_bundle_trial(
    seed: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Test partial-cue recovery in the actual evolving cortex↔hippocampus loop.

    The memory-return ablation uses deep copies of the same learned organism so
    both conditions begin from identical learned weights and homeostatic state.
    Family IDs exist only in this scoring function and never enter the organism.
    """
    feature_size = 12
    families = 6
    rng, trained_sheet, trained_memory, sensory, other = _trained_coupled_system(seed)

    cortical_references: list[np.ndarray] = []
    dg_references: list[np.ndarray] = []
    ca3_references: list[np.ndarray] = []
    final_cortical_references: list[np.ndarray] = []

    for family in range(families):
        sheet = copy.deepcopy(trained_sheet)
        memory = copy.deepcopy(trained_memory)
        full = noisy_experience(
            rng,
            sensory,
            other,
            family,
            feature_size,
            ("sensory", "context", "body", "time"),
            noise=0.02,
        )
        trajectory = _run_coupled(
            sheet,
            memory,
            full,
            total_steps=8,
            cue_steps=2,
            learn=False,
            memory_return=True,
        )
        cortical_references.append(
            _trajectory_signature(
                trajectory,
                population="cortical",
                size=sheet.size,
            )
        )
        dg_references.append(
            _trajectory_signature(
                trajectory,
                population="dentate",
                size=memory.dg_size,
            )
        )
        ca3_references.append(
            _trajectory_signature(
                trajectory,
                population="ca3",
                size=memory.ca3_size,
            )
        )
        final_cortical_references.append(trajectory[-1].cortical.indices.copy())

    memory_cortical: list[np.ndarray] = []
    no_return_cortical: list[np.ndarray] = []
    memory_dg: list[np.ndarray] = []
    memory_ca3: list[np.ndarray] = []
    final_memory_cortical: list[np.ndarray] = []

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

        memory_sheet = copy.deepcopy(trained_sheet)
        memory_branch = copy.deepcopy(trained_memory)
        memory_trajectory = _run_coupled(
            memory_sheet,
            memory_branch,
            partial,
            total_steps=8,
            cue_steps=2,
            learn=False,
            memory_return=True,
        )
        memory_cortical.append(
            _trajectory_signature(
                memory_trajectory,
                population="cortical",
                size=memory_sheet.size,
            )
        )
        memory_dg.append(
            _trajectory_signature(
                memory_trajectory,
                population="dentate",
                size=memory_branch.dg_size,
            )
        )
        memory_ca3.append(
            _trajectory_signature(
                memory_trajectory,
                population="ca3",
                size=memory_branch.ca3_size,
            )
        )
        final_memory_cortical.append(memory_trajectory[-1].cortical.indices.copy())

        no_return_sheet = copy.deepcopy(trained_sheet)
        no_return_branch = copy.deepcopy(trained_memory)
        no_return_trajectory = _run_coupled(
            no_return_sheet,
            no_return_branch,
            partial,
            total_steps=8,
            cue_steps=2,
            learn=False,
            memory_return=False,
        )
        no_return_cortical.append(
            _trajectory_signature(
                no_return_trajectory,
                population="cortical",
                size=no_return_sheet.size,
            )
        )

    cortical_accuracy, cortical_margin = _identity_metrics(
        cortical_references,
        memory_cortical,
    )
    no_return_accuracy, no_return_margin = _identity_metrics(
        cortical_references,
        no_return_cortical,
    )
    dg_accuracy, _ = _identity_metrics(dg_references, memory_dg)
    ca3_accuracy, ca3_margin = _identity_metrics(ca3_references, memory_ca3)

    final_correct = 0
    for family, candidate in enumerate(final_memory_cortical):
        scores = np.asarray(
            [
                assembly_overlap(
                    reference,
                    candidate,
                    trained_sheet.winner_budget,
                )
                for reference in final_cortical_references
            ]
        )
        final_correct += int(np.argmax(scores) == family)

    return (
        cortical_accuracy,
        cortical_margin,
        no_return_accuracy,
        no_return_margin,
        dg_accuracy,
        ca3_accuracy,
        ca3_margin,
        final_correct / families,
    )
