from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cortical_sheet import SignalBundle


@dataclass(frozen=True)
class ParallelBundleMemoryStep:
    """One synchronous hippocampal-branch update from a live cortical bundle."""

    time_ms: float
    dentate: SignalBundle
    ca3: SignalBundle
    cortical_return: SignalBundle


class ParallelBundleMemoryNetwork:
    """
    Experimental DG/CA3 branch for a live broad-bundle cortical substrate.

    Cortex remains the sole cortical owner. Cortical bundles diverge through
    registered excitatory edges into DG, DG expands/separates them, sparse
    powerful DG->CA3 routes converge with a weaker direct cortical seed and CA3
    recurrence, and CA3 emits a learned broad return toward cortex.

    Long-range cortical/DG afferents are excitatory. Separation and suppression
    are owned by sparse expansion, convergence, competition, and homeostatic
    pressure rather than arbitrary signed feed-forward weights.
    """

    def __init__(
        self,
        cortical_size: int,
        *,
        cortical_winner_count: int,
        dg_size: int = 2048,
        dg_winner_count: int = 64,
        dg_fan_out: int = 32,
        ca3_size: int = 512,
        ca3_winner_count: int = 16,
        ca3_direct_seed_count: int = 4,
        ca3_direct_fan_out: int = 8,
        ca3_dg_fan_in: int = 24,
        cortical_return_width: int | None = None,
        dentate_gain: float = 2.0,
        direct_cortical_gain: float = 0.20,
        recurrent_gain: float = 1.6,
        ca3_homeostatic_pressure: float = 6.0,
        afferent_learning_rate: float = 0.08,
        recurrent_learning_rate: float = 0.12,
        return_learning_rate: float = 0.12,
        seed: int = 0,
    ) -> None:
        if cortical_size <= 0 or dg_size <= 0 or ca3_size <= 0:
            raise ValueError("population sizes must be > 0")
        if not 0 < cortical_winner_count <= cortical_size:
            raise ValueError("invalid cortical_winner_count")
        if not 0 < dg_winner_count <= dg_size:
            raise ValueError("invalid dg_winner_count")
        if not 0 < ca3_direct_seed_count <= ca3_winner_count <= ca3_size:
            raise ValueError("invalid CA3 winner/seed counts")
        if dg_fan_out <= 0 or ca3_dg_fan_in <= 0 or ca3_direct_fan_out <= 0:
            raise ValueError("fan-in/fan-out values must be > 0")
        if dg_fan_out > dg_size:
            raise ValueError("dg_fan_out cannot exceed dg_size")
        if ca3_direct_fan_out > ca3_size:
            raise ValueError("ca3_direct_fan_out cannot exceed ca3_size")
        if dentate_gain < 0.0 or direct_cortical_gain < 0.0 or recurrent_gain < 0.0:
            raise ValueError("CA3 pathway gains must be >= 0")

        self.cortical_size = int(cortical_size)
        self.cortical_winner_count = int(cortical_winner_count)
        self.dg_size = int(dg_size)
        self.dg_winner_count = int(dg_winner_count)
        self.dg_fan_out = int(dg_fan_out)
        self.ca3_size = int(ca3_size)
        self.ca3_winner_count = int(ca3_winner_count)
        self.ca3_direct_seed_count = int(ca3_direct_seed_count)
        self.ca3_direct_fan_out = int(ca3_direct_fan_out)
        self.cortical_return_width = int(
            cortical_return_width
            if cortical_return_width is not None
            else min(self.cortical_size, self.cortical_winner_count * 4)
        )
        self.dentate_gain = float(dentate_gain)
        self.direct_cortical_gain = float(direct_cortical_gain)
        self.recurrent_gain = float(recurrent_gain)
        self.ca3_homeostatic_pressure = float(ca3_homeostatic_pressure)
        self.afferent_learning_rate = float(afferent_learning_rate)
        self.recurrent_learning_rate = float(recurrent_learning_rate)
        self.return_learning_rate = float(return_learning_rate)
        self.rng = np.random.default_rng(int(seed))

        # Cortex -> DG is a true sparse projection graph. Every active cortical
        # source diverges to several DG targets, and DG cells sum converging
        # currents. Target-wise L2 normalization prevents accidental high-degree
        # DG hubs from winning simply because of construction variance.
        self.dg_source_edges = np.repeat(
            np.arange(self.cortical_size, dtype=np.int32),
            self.dg_fan_out,
        )
        dg_targets = np.empty(
            (self.cortical_size, self.dg_fan_out), dtype=np.int32
        )
        for source_id in range(self.cortical_size):
            dg_targets[source_id] = self.rng.choice(
                self.dg_size,
                size=self.dg_fan_out,
                replace=False,
            )
        self.dg_target_edges = dg_targets.reshape(-1)
        dg_weights = self.rng.uniform(
            0.5,
            1.0,
            size=self.dg_source_edges.size,
        ).astype(np.float32)
        target_norm = np.sqrt(
            np.bincount(
                self.dg_target_edges,
                weights=np.square(dg_weights),
                minlength=self.dg_size,
            )
        ).astype(np.float32)
        self.dg_edge_weights = dg_weights / np.maximum(
            target_norm[self.dg_target_edges], 1e-12
        )

        # Sparse direct cortical/entorhinal fan-out preserves source identity:
        # shared active cortical cells send current down the same registered
        # routes instead of being remixed through a dense random matrix.
        self.ca3_direct_source_edges = np.repeat(
            np.arange(self.cortical_size, dtype=np.int32),
            self.ca3_direct_fan_out,
        )
        direct_targets = np.empty(
            (self.cortical_size, self.ca3_direct_fan_out), dtype=np.int32
        )
        for source_id in range(self.cortical_size):
            direct_targets[source_id] = self.rng.choice(
                self.ca3_size,
                size=self.ca3_direct_fan_out,
                replace=False,
            )
        self.ca3_direct_target_edges = direct_targets.reshape(-1)
        direct_weights = self.rng.uniform(
            0.5,
            1.0,
            size=(self.cortical_size, self.ca3_direct_fan_out),
        ).astype(np.float32)
        direct_weights /= np.maximum(
            np.linalg.norm(direct_weights, axis=1, keepdims=True),
            1e-12,
        )
        self.ca3_direct_weights = direct_weights.reshape(-1)

        # Mossy-fiber-like path: each CA3 unit samples only a small subset of
        # the large sparse DG population.
        self.ca3_dg_sources = self.rng.integers(
            0,
            self.dg_size,
            size=(self.ca3_size, ca3_dg_fan_in),
            dtype=np.int32,
        )
        self.ca3_dg_weights = self.rng.uniform(
            0.5,
            1.0,
            size=(self.ca3_size, ca3_dg_fan_in),
        ).astype(np.float32)
        self.ca3_dg_weights /= np.maximum(
            np.linalg.norm(self.ca3_dg_weights, axis=1, keepdims=True),
            1e-12,
        )

        self.ca3_recurrent_weights = np.zeros(
            (self.ca3_size, self.ca3_size), dtype=np.float32
        )
        self.cortical_return_weights = np.zeros(
            (self.cortical_size, self.ca3_size), dtype=np.float32
        )
        self.ca3_usage = np.zeros(self.ca3_size, dtype=np.float32)

        self.time_ms = 0.0
        self.dentate = self._empty_bundle()
        self.ca3 = self._empty_bundle()

    def _empty_bundle(self) -> SignalBundle:
        return SignalBundle(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            self.time_ms,
        )

    @staticmethod
    def _top_bundle(
        scores: np.ndarray,
        count: int,
        emitted_ms: float,
    ) -> SignalBundle:
        value = np.asarray(scores, dtype=np.float32)
        positive = np.flatnonzero(value > 1e-12)
        if positive.size == 0:
            return SignalBundle(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                emitted_ms,
            )
        if positive.size > count:
            chosen = positive[
                np.argpartition(value[positive], -count)[-count:]
            ]
        else:
            chosen = positive
        chosen = chosen[np.argsort(-value[chosen], kind="stable")]
        peak = max(float(np.max(value[chosen])), 1e-12)
        return SignalBundle(chosen, value[chosen] / peak, emitted_ms)

    def reset_dynamic(self, *, reset_usage: bool = False) -> None:
        self.time_ms = 0.0
        self.dentate = self._empty_bundle()
        self.ca3 = self._empty_bundle()
        if reset_usage:
            self.ca3_usage.fill(0.0)

    def _dentate_from_cortex(
        self,
        cortical_bundle: SignalBundle,
        emitted_ms: float,
    ) -> SignalBundle:
        cortical = cortical_bundle.as_dense(self.cortical_size)
        edge_signal = self.dg_edge_weights * cortical[self.dg_source_edges]
        scores = np.bincount(
            self.dg_target_edges,
            weights=edge_signal,
            minlength=self.dg_size,
        ).astype(np.float32)
        return self._top_bundle(scores, self.dg_winner_count, emitted_ms)

    def _direct_cortical_scores(self, cortical: np.ndarray) -> np.ndarray:
        edge_signal = (
            self.ca3_direct_weights * cortical[self.ca3_direct_source_edges]
        )
        return np.bincount(
            self.ca3_direct_target_edges,
            weights=edge_signal,
            minlength=self.ca3_size,
        ).astype(np.float32)

    def _ca3_from_inputs(
        self,
        previous_dentate: SignalBundle,
        cortical_bundle: SignalBundle,
        previous_ca3: SignalBundle,
        emitted_ms: float,
    ) -> SignalBundle:
        dg = previous_dentate.as_dense(self.dg_size)
        cortical = cortical_bundle.as_dense(self.cortical_size)
        ca3 = previous_ca3.as_dense(self.ca3_size)

        mossy_scores = np.sum(
            self.ca3_dg_weights * dg[self.ca3_dg_sources],
            axis=1,
        )
        scores = self.dentate_gain * mossy_scores
        scores += self.direct_cortical_gain * self._direct_cortical_scores(cortical)
        if previous_ca3.width:
            scores += self.recurrent_gain * (self.ca3_recurrent_weights @ ca3)
        scores /= 1.0 + self.ca3_homeostatic_pressure * self.ca3_usage

        count = (
            self.ca3_winner_count
            if previous_dentate.width or previous_ca3.width
            else self.ca3_direct_seed_count
        )
        return self._top_bundle(scores, count, emitted_ms)

    def _cortical_return(
        self,
        ca3_bundle: SignalBundle,
        emitted_ms: float,
    ) -> SignalBundle:
        if ca3_bundle.width == 0:
            return SignalBundle(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                emitted_ms,
            )
        scores = self.cortical_return_weights @ ca3_bundle.as_dense(self.ca3_size)
        return self._top_bundle(scores, self.cortical_return_width, emitted_ms)

    def _learn_direct_cortical_afferents(
        self,
        cortical_bundle: SignalBundle,
        next_ca3: SignalBundle,
    ) -> None:
        if cortical_bundle.width == 0 or next_ca3.width == 0:
            return
        cortical = cortical_bundle.as_dense(self.cortical_size)
        ca3 = next_ca3.as_dense(self.ca3_size)
        source_strength = cortical[self.ca3_direct_source_edges]
        target_strength = ca3[self.ca3_direct_target_edges]
        eligible = (source_strength > 0.0) & (target_strength > 0.0)
        if not np.any(eligible):
            return
        coactivity = source_strength[eligible] * target_strength[eligible]
        self.ca3_direct_weights[eligible] += (
            self.afferent_learning_rate
            * 0.5
            * coactivity
            * (1.0 - self.ca3_direct_weights[eligible])
        )
        rows = self.ca3_direct_weights.reshape(
            self.cortical_size, self.ca3_direct_fan_out
        )
        rows /= np.maximum(
            np.linalg.norm(rows, axis=1, keepdims=True),
            1e-12,
        )

    def _learn_mossy_afferents(
        self,
        previous_dentate: SignalBundle,
        next_ca3: SignalBundle,
    ) -> None:
        if previous_dentate.width == 0 or next_ca3.width == 0:
            return
        dg = previous_dentate.as_dense(self.dg_size)
        active_strength = dg[self.ca3_dg_sources[next_ca3.indices]]
        rows = self.ca3_dg_weights[next_ca3.indices]
        eta = (
            self.afferent_learning_rate * next_ca3.amplitudes.astype(np.float32)
        )[:, None]
        target = active_strength.astype(np.float32)
        target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
        updated = (1.0 - eta) * rows + eta * target
        updated /= np.maximum(
            np.linalg.norm(updated, axis=1, keepdims=True),
            1e-12,
        )
        self.ca3_dg_weights[next_ca3.indices] = updated.astype(np.float32)

    def _learn_ca3_afferents(
        self,
        previous_dentate: SignalBundle,
        cortical_bundle: SignalBundle,
        next_ca3: SignalBundle,
    ) -> None:
        self._learn_mossy_afferents(previous_dentate, next_ca3)
        self._learn_direct_cortical_afferents(cortical_bundle, next_ca3)

    def _learn_competitive_recurrence(
        self,
        previous_ca3: SignalBundle,
        next_ca3: SignalBundle,
    ) -> None:
        if previous_ca3.width == 0 or next_ca3.width == 0:
            return
        source_amplitudes = previous_ca3.amplitudes.astype(np.float32)
        target_activity = next_ca3.as_dense(self.ca3_size)
        expected_activity = float(next_ca3.width / self.ca3_size)
        centered_target = target_activity - expected_activity
        delta = self.recurrent_learning_rate * np.outer(
            centered_target,
            source_amplitudes,
        ).astype(np.float32)
        self.ca3_recurrent_weights[:, previous_ca3.indices] += delta
        np.clip(
            self.ca3_recurrent_weights,
            -1.0,
            1.0,
            out=self.ca3_recurrent_weights,
        )
        np.fill_diagonal(self.ca3_recurrent_weights, 0.0)

    def _learn_cortical_return(
        self,
        previous_ca3: SignalBundle,
        cortical_bundle: SignalBundle,
    ) -> None:
        if previous_ca3.width == 0 or cortical_bundle.width == 0:
            return
        block = self.cortical_return_weights[
            np.ix_(cortical_bundle.indices, previous_ca3.indices)
        ]
        coactivity = np.outer(
            cortical_bundle.amplitudes,
            previous_ca3.amplitudes,
        ).astype(np.float32)
        block += self.return_learning_rate * coactivity * (1.0 - block)
        self.cortical_return_weights[
            np.ix_(cortical_bundle.indices, previous_ca3.indices)
        ] = np.clip(block, 0.0, 1.0)

    def advance(
        self,
        cortical_bundle: SignalBundle,
        *,
        learn: bool = False,
        dt_ms: float = 1.0,
    ) -> ParallelBundleMemoryStep:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        if cortical_bundle.width and int(np.max(cortical_bundle.indices)) >= self.cortical_size:
            raise IndexError("cortical bundle outside cortical population")

        self.time_ms += float(dt_ms)
        now = self.time_ms
        previous_dentate = self.dentate
        previous_ca3 = self.ca3

        cortical_return = self._cortical_return(previous_ca3, now)
        next_dentate = self._dentate_from_cortex(cortical_bundle, now)
        next_ca3 = self._ca3_from_inputs(
            previous_dentate,
            cortical_bundle,
            previous_ca3,
            now,
        )

        if learn:
            self._learn_ca3_afferents(previous_dentate, cortical_bundle, next_ca3)
            self._learn_competitive_recurrence(previous_ca3, next_ca3)
            self._learn_cortical_return(previous_ca3, cortical_bundle)

        self.ca3_usage = (
            0.995 * self.ca3_usage
            + 0.005
            * np.isin(np.arange(self.ca3_size), next_ca3.indices).astype(np.float32)
        )
        self.dentate = next_dentate
        self.ca3 = next_ca3
        return ParallelBundleMemoryStep(
            time_ms=now,
            dentate=next_dentate,
            ca3=next_ca3,
            cortical_return=cortical_return,
        )
