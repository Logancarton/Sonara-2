from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HippocampalRecall:
    """One experimental hippocampal recall pass."""

    dg_indices: np.ndarray
    ca3_seed_indices: np.ndarray
    ca3_winner_indices: np.ndarray
    cortical_winner_indices: np.ndarray
    cortical_reactivation: np.ndarray


class HippocampalLoop:
    """
    Fast experimental DG -> CA3 -> cortical-reactivation loop.

    This is intentionally not a literal hippocampus. It tests the mechanism we
    care about:

        distributed cortical state
          -> sparse DG expansion/separation
          -> sparse CA3 seed
          -> recurrent CA3 completion
          -> sparse learned divergence back toward cortex

    No semantic labels are accepted. Repeated inputs are grouped only by their
    learned similarity. Runtime recall is vectorized; the only Python loops are
    over a handful of recurrent settling steps.
    """

    def __init__(
        self,
        input_size: int,
        *,
        dg_size: int = 4096,
        dg_winner_count: int = 64,
        dg_fan_in: int = 32,
        ca3_size: int = 512,
        ca3_assembly_size: int = 32,
        ca3_seed_size: int = 8,
        cortical_winner_count: int = 51,
        cortical_fan_out: int = 4,
        novelty_threshold: float = 0.32,
        learning_rate: float = 0.15,
        recurrent_gain: float = 2.0,
        cue_gain: float = 0.25,
        seed: int = 0,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if dg_size <= 0 or ca3_size <= 0:
            raise ValueError("dg_size and ca3_size must be > 0")
        if not 0 < dg_winner_count <= dg_size:
            raise ValueError("dg_winner_count must be in (0, dg_size]")
        if not 0 < ca3_seed_size <= ca3_assembly_size <= ca3_size:
            raise ValueError(
                "require 0 < ca3_seed_size <= ca3_assembly_size <= ca3_size"
            )
        if not 0 < cortical_winner_count <= input_size:
            raise ValueError("cortical_winner_count must be in (0, input_size]")
        if dg_fan_in <= 0 or cortical_fan_out <= 0:
            raise ValueError("fan-in and fan-out must be > 0")
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in [0, 1]")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if recurrent_gain < 0.0 or cue_gain < 0.0:
            raise ValueError("recurrent_gain and cue_gain must be >= 0")

        self.input_size = int(input_size)
        self.dg_size = int(dg_size)
        self.dg_winner_count = int(dg_winner_count)
        self.ca3_size = int(ca3_size)
        self.ca3_assembly_size = int(ca3_assembly_size)
        self.ca3_seed_size = int(ca3_seed_size)
        self.cortical_winner_count = int(cortical_winner_count)
        self.cortical_fan_out = int(cortical_fan_out)
        self.novelty_threshold = float(novelty_threshold)
        self.learning_rate = float(learning_rate)
        self.recurrent_gain = float(recurrent_gain)
        self.cue_gain = float(cue_gain)
        self.rng = np.random.default_rng(int(seed))

        self._dg_source_indices = self.rng.integers(
            0,
            self.input_size,
            size=(self.dg_size, dg_fan_in),
            dtype=np.int32,
        )
        self._dg_weights = self.rng.normal(
            size=(self.dg_size, dg_fan_in)
        ).astype(np.float32)
        self._dg_weights /= np.maximum(
            np.linalg.norm(self._dg_weights, axis=1, keepdims=True),
            1e-12,
        )

        self.ca3_afferent_weights = np.zeros(
            (self.ca3_size, self.dg_size), dtype=np.float32
        )
        self.ca3_recurrent_weights = np.zeros(
            (self.ca3_size, self.ca3_size), dtype=np.float32
        )

        # CA3 -> cortical routes are intentionally sparse. Targets are not fixed
        # on the first exposure. They are consolidated from repeated cortical
        # co-activity when learning is finalized, so the outbound map reflects a
        # stable learned trace rather than an early transient winner set.
        self._cortical_targets = np.full(
            (self.ca3_size, self.cortical_fan_out),
            -1,
            dtype=np.int32,
        )
        self._cortical_output_weights = np.zeros(
            (self.ca3_size, self.cortical_fan_out), dtype=np.float32
        )

        self._trace_prototypes: list[np.ndarray] = []
        self._trace_assemblies: list[np.ndarray] = []
        self._trace_counts: list[int] = []
        self._trace_cortical_mean: list[np.ndarray] = []
        self._trace_cortical_winner_counts: list[np.ndarray] = []
        self._allocated_ca3 = np.zeros(self.ca3_size, dtype=bool)
        self._finalized = False

    @property
    def trace_count(self) -> int:
        return len(self._trace_prototypes)

    @staticmethod
    def _normalize(vector: np.ndarray, expected_size: int) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (expected_size,):
            raise ValueError(
                f"expected cortical state shape {(expected_size,)}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("cortical state must be finite")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            return np.zeros_like(value)
        return value / norm

    @staticmethod
    def _top_k(scores: np.ndarray, count: int) -> np.ndarray:
        indices = np.argpartition(scores, -count)[-count:]
        return indices[np.argsort(-scores[indices], kind="stable")]

    def _dentate_input(self, cortical_state: np.ndarray) -> np.ndarray:
        """
        Preserve distributed cortical state while emphasizing its sparse coalition.

        The cortical sheet already proved that multiple converging streams create
        more distinct winner coalitions. DG should not let the large shared
        sensory/background component drown those episode-specific differences.
        The transformation therefore mixes the normalized continuous state with
        an unlabeled sparse signal made from the currently strongest cortical
        cells. Both components have unit norm before mixing.
        """
        cortical = self._normalize(cortical_state, self.input_size)
        cortical_winners = self._top_k(cortical, self.cortical_winner_count)
        sparse_signal = np.zeros(self.input_size, dtype=np.float32)
        sparse_signal[cortical_winners] = 1.0 / np.sqrt(self.cortical_winner_count)
        mixed = 0.35 * cortical + 0.65 * sparse_signal
        return self._normalize(mixed, self.input_size)

    def dentate_code(self, cortical_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand a cortical state into a much larger, very sparse DG code."""
        cortical = self._dentate_input(cortical_state)
        scores = np.sum(
            self._dg_weights * cortical[self._dg_source_indices],
            axis=1,
        )
        scores = np.maximum(scores, 0.0)
        winners = self._top_k(scores, self.dg_winner_count)
        code = np.zeros(self.dg_size, dtype=np.float32)
        code[winners] = 1.0
        return code, winners

    def _prototype_similarity(self, dg_code: np.ndarray) -> np.ndarray:
        if not self._trace_prototypes:
            return np.empty(0, dtype=np.float32)
        prototypes = np.stack(self._trace_prototypes)
        numerator = prototypes @ dg_code
        denominator = (
            np.linalg.norm(prototypes, axis=1) * max(float(np.linalg.norm(dg_code)), 1e-12)
        )
        return (numerator / np.maximum(denominator, 1e-12)).astype(np.float32)

    def _allocate_ca3_assembly(self) -> np.ndarray:
        available = np.flatnonzero(~self._allocated_ca3)
        if available.size < self.ca3_assembly_size:
            raise RuntimeError(
                "experimental CA3 capacity exhausted; increase ca3_size or reduce assembly size"
            )
        assembly = self.rng.choice(
            available,
            size=self.ca3_assembly_size,
            replace=False,
        ).astype(np.int64)
        self._allocated_ca3[assembly] = True
        return assembly

    def _validated_cortical_indices(
        self,
        cortical: np.ndarray,
        cortical_active_indices: np.ndarray | None,
    ) -> np.ndarray:
        if cortical_active_indices is None:
            return self._top_k(cortical, self.cortical_winner_count)

        active = np.asarray(cortical_active_indices, dtype=np.int64)
        if active.ndim != 1 or active.size == 0:
            raise ValueError("cortical_active_indices must be a non-empty 1-D array")
        if np.any(active < 0) or np.any(active >= self.input_size):
            raise IndexError("cortical active index outside input range")
        return np.unique(active)

    def _record_cortical_trace(
        self,
        trace_index: int,
        cortical: np.ndarray,
        active_cortex: np.ndarray,
    ) -> None:
        count = self._trace_counts[trace_index]
        self._trace_cortical_mean[trace_index] = (
            (self._trace_cortical_mean[trace_index] * (count - 1) + cortical) / count
        ).astype(np.float32)
        self._trace_cortical_winner_counts[trace_index][active_cortex] += 1.0

    def _consolidate_cortical_routes(self, trace_index: int) -> None:
        assembly = self._trace_assemblies[trace_index]
        mean_state = self._trace_cortical_mean[trace_index]
        winner_counts = self._trace_cortical_winner_counts[trace_index]

        max_count = max(float(np.max(winner_counts)), 1.0)
        stable_score = winner_counts / max_count + 1e-3 * np.maximum(mean_state, 0.0)
        stable_cortex = self._top_k(stable_score, self.cortical_winner_count)

        total_routes = self.ca3_assembly_size * self.cortical_fan_out
        ordered = stable_cortex[np.argsort(-stable_score[stable_cortex], kind="stable")]
        route_pool = np.resize(ordered, total_routes)
        self._cortical_targets[assembly] = route_pool.reshape(
            self.ca3_assembly_size,
            self.cortical_fan_out,
        )

        targets = self._cortical_targets[assembly]
        frequency = winner_counts[targets] / max(
            float(self._trace_counts[trace_index]), 1.0
        )
        state_strength = np.maximum(mean_state[targets], 0.0)
        self._cortical_output_weights[assembly] = (
            frequency + state_strength
        ).astype(np.float32)

    def learn(
        self,
        cortical_state: np.ndarray,
        *,
        cortical_active_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Learn one experience without labels.

        Similar DG states reinforce an existing CA3 attractor. A sufficiently
        novel DG state allocates another sparse CA3 assembly. Repeated exposures
        accumulate the cortical co-activity statistics used later to construct
        a stable sparse outbound reactivation map.
        """
        if self._finalized:
            raise RuntimeError("cannot learn after finalize_learning()")

        cortical = self._normalize(cortical_state, self.input_size)
        active_cortex = self._validated_cortical_indices(
            cortical,
            cortical_active_indices,
        )
        dg_code, _ = self.dentate_code(cortical)
        similarities = self._prototype_similarity(dg_code)
        new_trace = not (
            similarities.size
            and float(np.max(similarities)) >= self.novelty_threshold
        )

        if new_trace:
            assembly = self._allocate_ca3_assembly()
            trace_index = len(self._trace_assemblies)
            self._trace_assemblies.append(assembly)
            self._trace_prototypes.append(dg_code.copy())
            self._trace_counts.append(1)
            self._trace_cortical_mean.append(cortical.copy())
            winner_counts = np.zeros(self.input_size, dtype=np.float32)
            winner_counts[active_cortex] = 1.0
            self._trace_cortical_winner_counts.append(winner_counts)
        else:
            trace_index = int(np.argmax(similarities))
            assembly = self._trace_assemblies[trace_index]
            count = self._trace_counts[trace_index]
            self._trace_prototypes[trace_index] = (
                (self._trace_prototypes[trace_index] * count + dg_code) / (count + 1)
            ).astype(np.float32)
            self._trace_counts[trace_index] = count + 1
            self._record_cortical_trace(trace_index, cortical, active_cortex)

        eta = self.learning_rate
        dg_target = self._normalize(dg_code, self.dg_size)
        updated_afferents = (
            (1.0 - eta) * self.ca3_afferent_weights[assembly]
            + eta * dg_target
        )
        updated_afferents /= np.maximum(
            np.linalg.norm(updated_afferents, axis=1, keepdims=True),
            1e-12,
        )
        self.ca3_afferent_weights[assembly] = updated_afferents.astype(np.float32)

        self.ca3_recurrent_weights[np.ix_(assembly, assembly)] += 0.10
        np.fill_diagonal(self.ca3_recurrent_weights, 0.0)
        return assembly.copy()

    def finalize_learning(self) -> None:
        """Consolidate recurrent and sparse cortical reactivation routes."""
        row_sum = np.sum(self.ca3_recurrent_weights, axis=1, keepdims=True)
        active_rows = row_sum[:, 0] > 1e-12
        self.ca3_recurrent_weights[active_rows] /= row_sum[active_rows]
        for trace_index in range(self.trace_count):
            self._consolidate_cortical_routes(trace_index)
        self._finalized = True

    def ca3_seed(self, cortical_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a small feed-forward CA3 cue from a degraded cortical state."""
        dg_code, dg_indices = self.dentate_code(cortical_state)
        scores = self.ca3_afferent_weights @ self._normalize(dg_code, self.dg_size)
        seed_indices = self._top_k(scores, self.ca3_seed_size)
        seed = np.zeros(self.ca3_size, dtype=np.float32)
        seed[seed_indices] = 1.0
        return seed, seed_indices, dg_indices

    def recall(
        self,
        cortical_state: np.ndarray,
        *,
        recurrent: bool = True,
        settle_steps: int = 5,
    ) -> HippocampalRecall:
        if not self._finalized:
            raise RuntimeError("call finalize_learning() before recall")
        if settle_steps < 0:
            raise ValueError("settle_steps must be >= 0")

        cue, seed_indices, dg_indices = self.ca3_seed(cortical_state)
        activity = cue.copy()
        winner_indices = seed_indices.copy()

        if recurrent:
            for _ in range(settle_steps):
                scores = (
                    self.cue_gain * cue
                    + self.recurrent_gain * (self.ca3_recurrent_weights @ activity)
                )
                winner_indices = self._top_k(scores, self.ca3_assembly_size)
                activity.fill(0.0)
                activity[winner_indices] = 1.0

        active_ca3 = np.flatnonzero(activity > 0.0)
        active_targets = self._cortical_targets[active_ca3]
        valid_routes = active_targets >= 0
        cortical_reactivation = np.bincount(
            active_targets[valid_routes],
            weights=self._cortical_output_weights[active_ca3][valid_routes],
            minlength=self.input_size,
        ).astype(np.float32)
        cortical_winners = self._top_k(
            cortical_reactivation,
            self.cortical_winner_count,
        )

        return HippocampalRecall(
            dg_indices=dg_indices.copy(),
            ca3_seed_indices=seed_indices.copy(),
            ca3_winner_indices=winner_indices.copy(),
            cortical_winner_indices=cortical_winners.copy(),
            cortical_reactivation=cortical_reactivation,
        )
