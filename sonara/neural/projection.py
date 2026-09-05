from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionLearning:
    enabled: bool = True
    learning_rate: float = 0.08
    stdp_window_ms: float = 25.0
    tau_ms: float = 12.0
    min_weight: float = 0.0
    max_weight: float = 30.0


class SynapseProjection:
    """
    Sparse directed synapses between two populations.

    Runtime routing is generic: a projection is data (source, target, sparse
    edges, weights, delay), not a hardcoded cognitive instruction.
    """

    def __init__(
        self,
        projection_id: str,
        source_id: str,
        target_id: str,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        weights: float | np.ndarray,
        *,
        delay_ms: float = 1.0,
        learning: ProjectionLearning | None = None,
    ) -> None:
        if not projection_id.strip():
            raise ValueError("projection_id must be non-empty")
        if delay_ms < 0.0:
            raise ValueError("delay_ms must be >= 0")

        src = np.asarray(source_indices, dtype=np.int64)
        dst = np.asarray(target_indices, dtype=np.int64)
        if src.ndim != 1 or dst.ndim != 1 or src.shape != dst.shape:
            raise ValueError("source_indices and target_indices must be same-length 1D arrays")
        if src.size == 0:
            raise ValueError("projection must contain at least one synapse")

        if np.isscalar(weights):
            w = np.full(src.size, float(weights), dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32)
            if w.shape != src.shape:
                raise ValueError("weights must be scalar or match synapse count")

        self.projection_id = projection_id
        self.source_id = source_id
        self.target_id = target_id
        self.source_indices = src
        self.target_indices = dst
        self.weights = w
        self.delay_ms = float(delay_ms)
        self.learning = learning or ProjectionLearning()
        self.last_pre_ms = np.full(src.size, -np.inf, dtype=np.float64)

    @classmethod
    def one_to_one(
        cls,
        projection_id: str,
        source_id: str,
        target_id: str,
        size: int,
        weight: float,
        *,
        delay_ms: float = 1.0,
        learning: ProjectionLearning | None = None,
    ) -> "SynapseProjection":
        idx = np.arange(int(size), dtype=np.int64)
        return cls(
            projection_id,
            source_id,
            target_id,
            idx,
            idx.copy(),
            weight,
            delay_ms=delay_ms,
            learning=learning,
        )

    @classmethod
    def random_sparse(
        cls,
        projection_id: str,
        source_id: str,
        target_id: str,
        source_size: int,
        target_size: int,
        *,
        density: float,
        weight: float,
        delay_ms: float = 1.0,
        seed: int = 0,
        learning: ProjectionLearning | None = None,
    ) -> "SynapseProjection":
        if source_size <= 0 or target_size <= 0:
            raise ValueError("source_size and target_size must be > 0")
        if not 0.0 < float(density) <= 1.0:
            raise ValueError("density must be in (0, 1]")

        rng = np.random.default_rng(seed)
        total_possible = int(source_size) * int(target_size)
        edge_count = max(1, int(round(total_possible * float(density))))
        flat = rng.choice(total_possible, size=edge_count, replace=False)
        src = flat // int(target_size)
        dst = flat % int(target_size)
        return cls(
            projection_id,
            source_id,
            target_id,
            src,
            dst,
            weight,
            delay_ms=delay_ms,
            learning=learning,
        )

    def validate_sizes(self, source_size: int, target_size: int) -> None:
        if np.any(self.source_indices < 0) or np.any(self.source_indices >= int(source_size)):
            raise ValueError(f"{self.projection_id}: source index outside {self.source_id}")
        if np.any(self.target_indices < 0) or np.any(self.target_indices >= int(target_size)):
            raise ValueError(f"{self.projection_id}: target index outside {self.target_id}")

    def current_from_spikes(self, source_spikes: np.ndarray, target_size: int) -> np.ndarray:
        """Convert source spikes into target current using sparse synapses."""
        fired = np.asarray(source_spikes, dtype=np.int64)
        target_current = np.zeros(int(target_size), dtype=np.float32)
        if fired.size == 0:
            return target_current

        active_edges = np.isin(self.source_indices, fired, assume_unique=False)
        if not np.any(active_edges):
            return target_current

        np.add.at(
            target_current,
            self.target_indices[active_edges],
            self.weights[active_edges],
        )
        return target_current

    def record_pre_spikes(self, source_spikes: np.ndarray, now_ms: float) -> None:
        fired = np.asarray(source_spikes, dtype=np.int64)
        if fired.size == 0:
            return
        active_edges = np.isin(self.source_indices, fired, assume_unique=False)
        self.last_pre_ms[active_edges] = float(now_ms)

    def learn_from_post_spikes(self, target_spikes: np.ndarray, now_ms: float) -> int:
        """
        Local causal STDP-like potentiation.

        Existing synapses strengthen when their presynaptic neuron fired shortly
        before their postsynaptic target. No global task label is required.
        """
        if not self.learning.enabled:
            return 0
        fired = np.asarray(target_spikes, dtype=np.int64)
        if fired.size == 0:
            return 0

        post_edges = np.isin(self.target_indices, fired, assume_unique=False)
        dt = float(now_ms) - self.last_pre_ms
        eligible = post_edges & (dt >= 0.0) & (dt <= self.learning.stdp_window_ms)
        if not np.any(eligible):
            return 0

        lr = max(0.0, float(self.learning.learning_rate))
        tau = max(float(self.learning.tau_ms), 1e-9)
        growth = lr * np.exp(-dt[eligible] / tau)
        headroom = self.learning.max_weight - self.weights[eligible]
        self.weights[eligible] += growth * headroom
        np.clip(
            self.weights,
            self.learning.min_weight,
            self.learning.max_weight,
            out=self.weights,
        )
        return int(np.count_nonzero(eligible))
