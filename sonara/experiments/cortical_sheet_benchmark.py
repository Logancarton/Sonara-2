from __future__ import annotations

import numpy as np

from .cortical_sheet import (
    FastCorticalSheet,
    StreamSpec,
    assembly_overlap,
    separation_metrics,
)


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / np.linalg.norm(value)


def default_streams(feature_size: int = 12) -> tuple[StreamSpec, ...]:
    return (
        StreamSpec("sensory", feature_size, (0.05, 0.85), sigma=0.32, gain=4.0),
        StreamSpec("context", feature_size, (0.95, 0.85), sigma=0.32, gain=3.0),
        StreamSpec("body", feature_size, (0.05, 0.15), sigma=0.32, gain=3.0),
        StreamSpec("time", feature_size, (0.95, 0.15), sigma=0.32, gain=3.0),
    )


def experience_prototypes(
    rng: np.random.Generator,
    families: int,
    feature_size: int,
) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """
    Build deliberately ambiguous experiences.

    Sensory prototypes are almost identical across families. Context, body,
    and temporal streams contain independent family-specific evidence.
    """
    common_sensory = normalize(rng.normal(size=feature_size))
    sensory = [
        normalize(common_sensory + 0.05 * rng.normal(size=feature_size))
        for _ in range(families)
    ]
    other = {
        name: [normalize(rng.normal(size=feature_size)) for _ in range(families)]
        for name in ("context", "body", "time")
    }
    return sensory, other


def noisy_experience(
    rng: np.random.Generator,
    sensory: list[np.ndarray],
    other: dict[str, list[np.ndarray]],
    family: int,
    feature_size: int,
    present: tuple[str, ...],
    *,
    noise: float,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if "sensory" in present:
        result["sensory"] = normalize(
            sensory[family]
            + noise * rng.normal(size=feature_size) / np.sqrt(feature_size)
        )
    for name in ("context", "body", "time"):
        if name in present:
            result[name] = normalize(
                other[name][family]
                + noise * rng.normal(size=feature_size) / np.sqrt(feature_size)
            )
    return result


def separation_trial(seed: int, present: tuple[str, ...]) -> float:
    rng = np.random.default_rng(seed)
    feature_size = 12
    families = 6
    sheet = FastCorticalSheet(
        32,
        32,
        default_streams(feature_size),
        sparsity=0.05,
        seed=100 + seed,
    )
    sensory, other = experience_prototypes(rng, families, feature_size)

    assemblies: list[list[np.ndarray]] = []
    for family in range(families):
        family_assemblies: list[np.ndarray] = []
        for _ in range(5):
            sheet.reset_state()
            family_assemblies.append(
                sheet.settle(
                    noisy_experience(
                        rng,
                        sensory,
                        other,
                        family,
                        feature_size,
                        present,
                        noise=0.18,
                    ),
                    cue_steps=4,
                    learn=False,
                )
            )
        assemblies.append(family_assemblies)
    return separation_metrics(assemblies, sheet.winner_budget).separation


def completion_trial(seed: int, recurrent: bool) -> tuple[float, float]:
    """
    Diagnostic for a future hippocampal-like completion mechanism.

    The current recurrent Hebbian rule is intentionally evaluated rather than
    assumed to be pattern completion.
    """
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

    references: list[np.ndarray] = []
    for family in range(families):
        sheet.reset_state()
        references.append(
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
        )

    correct = 0
    margins: list[float] = []
    for family in range(families):
        sheet.reset_state()
        retrieved = sheet.settle(
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
            blank_steps=10,
            recurrent=recurrent,
        )
        overlaps = np.asarray(
            [
                assembly_overlap(reference, retrieved, sheet.winner_budget)
                for reference in references
            ]
        )
        correct += int(np.argmax(overlaps) == family)
        margins.append(overlaps[family] - np.max(np.delete(overlaps, family)))
    return correct / families, float(np.mean(margins))
