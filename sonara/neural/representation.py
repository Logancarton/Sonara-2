from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssemblyActivation:
    """One sparse response of a representation field to a numeric stimulus."""

    field_id: str
    population_id: str
    winner_indices: np.ndarray
    winner_scores: np.ndarray
    current: np.ndarray
    sparsity: float

    @property
    def mean_winner_score(self) -> float:
        return float(np.mean(self.winner_scores)) if self.winner_scores.size else 0.0

    @property
    def mean_winner_current(self) -> float:
        if self.winner_indices.size == 0:
            return 0.0
        return float(np.mean(self.current[self.winner_indices]))

    def overlap(self, other: "AssemblyActivation") -> float:
        if self.winner_indices.size == 0:
            return 0.0
        shared = np.intersect1d(self.winner_indices, other.winner_indices).size
        return float(shared / self.winner_indices.size)


class SparseRepresentationField:
    """
    Deterministic sparse competitive encoder with local Hebbian adaptation.

    Numeric features drive a distributed candidate population through a learned
    feature-to-neuron weight matrix. k-winner competition selects the sparse
    assembly. Repeated winners move their afferent weights toward the current
    input, so future neural drive changes without storing stimulus→neuron IDs.
    """

    def __init__(
        self,
        field_id: str,
        population_id: str,
        population_size: int,
        feature_size: int,
        *,
        winner_count: int,
        seed: int = 0,
        learning_rate: float = 0.08,
        base_drive: float = 10.0,
        score_drive: float = 10.0,
    ) -> None:
        if not field_id.strip():
            raise ValueError("field_id must be non-empty")
        if not population_id.strip():
            raise ValueError("population_id must be non-empty")
        if int(population_size) <= 0 or int(feature_size) <= 0:
            raise ValueError("population_size and feature_size must be > 0")
        if not 0 < int(winner_count) <= int(population_size):
            raise ValueError("winner_count must be in [1, population_size]")
        if not 0.0 <= float(learning_rate) <= 1.0:
            raise ValueError("learning_rate must be in [0, 1]")

        self.field_id = field_id
        self.population_id = population_id
        self.population_size = int(population_size)
        self.feature_size = int(feature_size)
        self.winner_count = int(winner_count)
        self.learning_rate = float(learning_rate)
        self.base_drive = float(base_drive)
        self.score_drive = float(score_drive)

        rng = np.random.default_rng(int(seed))
        weights = rng.normal(size=(self.population_size, self.feature_size)).astype(np.float64)
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        self._weights = weights / np.maximum(norms, 1e-12)

    @property
    def weights(self) -> np.ndarray:
        """Read-only copy for diagnostics/tests; learning authority stays here."""
        return self._weights.copy()

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (self.feature_size,):
            raise ValueError(
                f"{self.field_id}: features must have shape {(self.feature_size,)}, got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{self.field_id}: features must be finite")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError(f"{self.field_id}: zero-length stimulus has no direction")
        return vector / norm

    def _winner_indices(self, scores: np.ndarray) -> np.ndarray:
        neuron_ids = np.arange(self.population_size, dtype=np.int64)
        order = np.lexsort((neuron_ids, -scores))
        return order[: self.winner_count].astype(np.int64, copy=False)

    def encode(self, features: np.ndarray, *, learn: bool = False) -> AssemblyActivation:
        stimulus = self._normalize_features(features)
        scores = self._weights @ stimulus
        winners = self._winner_indices(scores)
        winner_scores = scores[winners].copy()

        # Cosine score [-1, 1] -> [0, 1], then into actual neural input current.
        normalized_scores = np.clip((winner_scores + 1.0) * 0.5, 0.0, 1.0)
        current = np.zeros(self.population_size, dtype=np.float32)
        current[winners] = (self.base_drive + self.score_drive * normalized_scores).astype(np.float32)

        activation = AssemblyActivation(
            field_id=self.field_id,
            population_id=self.population_id,
            winner_indices=winners.copy(),
            winner_scores=winner_scores,
            current=current,
            sparsity=float(self.winner_count / self.population_size),
        )

        if learn:
            lr = self.learning_rate
            updated = (1.0 - lr) * self._weights[winners] + lr * stimulus
            norms = np.linalg.norm(updated, axis=1, keepdims=True)
            self._weights[winners] = updated / np.maximum(norms, 1e-12)

        return activation
