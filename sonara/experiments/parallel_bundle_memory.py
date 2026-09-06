from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cortical_sheet import SignalBundle


@dataclass(frozen=True)
class ParallelBundleStep:
    """One synchronous update of cortex, DG-like, and CA3-like bundle states."""

    time_ms: float
    cortical: SignalBundle
    dentate: SignalBundle
    ca3: SignalBundle


class ParallelBundleMemoryNetwork:
    """
    Experimental parallel bundle memory substrate.

    Cortex, DG-like expansion, CA3-like recurrence, and cortical return all
    advance from the previous tick's bundle state. The hippocampal branch never
    blocks cortical propagation and there is no episode/trace chooser.
    Learning is local Hebbian reinforcement of co-active bundle routes.
    """

    def __init__(
        self,
        cortical_size: int,
        *,
        cortical_winner_count: int,
        cortical_projector: Callable[[SignalBundle], SignalBundle] | None = None,
        dg_size: int = 2048,
        dg_winner_count: int = 64,
        dg_fan_in: int = 32,
        ca3_size: int = 384,
        ca3_winner_count: int = 32,
        cortical_return_width: int | None = None,
        recurrent_gain: float = 1.6,
        cortical_recurrence_gain: float = 0.75,
        cortical_return_gain: float = 1.25,
        cortical_persistence: float = 0.35,
        ca3_homeostatic_pressure: float = 2.0,
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
        if not 0 < ca3_winner_count <= ca3_size:
            raise ValueError("invalid ca3_winner_count")
        if dg_fan_in <= 0:
            raise ValueError("dg_fan_in must be > 0")

        self.cortical_size = int(cortical_size)
        self.cortical_winner_count = int(cortical_winner_count)
        self.cortical_projector = cortical_projector
        self.dg_size = int(dg_size)
        self.dg_winner_count = int(dg_winner_count)
        self.ca3_size = int(ca3_size)
        self.ca3_winner_count = int(ca3_winner_count)
        self.cortical_return_width = int(
            cortical_return_width
            if cortical_return_width is not None
            else min(self.cortical_size, self.cortical_winner_count * 4)
        )
        self.recurrent_gain = float(recurrent_gain)
        self.cortical_recurrence_gain = float(cortical_recurrence_gain)
        self.cortical_return_gain = float(cortical_return_gain)
        self.cortical_persistence = float(cortical_persistence)
        self.ca3_homeostatic_pressure = float(ca3_homeostatic_pressure)
        self.afferent_learning_rate = float(afferent_learning_rate)
        self.recurrent_learning_rate = float(recurrent_learning_rate)
        self.return_learning_rate = float(return_learning_rate)
        self.rng = np.random.default_rng(int(seed))

        self._dg_sources = self.rng.integers(
            0,
            self.cortical_size,
            size=(self.dg_size, dg_fan_in),
            dtype=np.int32,
        )
        self._dg_weights = self.rng.uniform(
            0.5,
            1.0,
            size=(self.dg_size, dg_fan_in),
        ).astype(np.float32)
        self._dg_weights /= np.maximum(
            np.linalg.norm(self._dg_weights, axis=1, keepdims=True),
            1e-12,
        )

        self.ca3_afferent_weights = self.rng.uniform(
            0.0,
            0.02,
            size=(self.ca3_size, self.dg_size),
        ).astype(np.float32)
        self.ca3_recurrent_weights = np.zeros(
            (self.ca3_size, self.ca3_size), dtype=np.float32
        )
        self.cortical_return_weights = np.zeros(
            (self.cortical_size, self.ca3_size), dtype=np.float32
        )
        self.ca3_usage = np.zeros(self.ca3_size, dtype=np.float32)

        self.time_ms = 0.0
        self.cortical = self._empty_bundle(self.cortical_size)
        self.dentate = self._empty_bundle(self.dg_size)
        self.ca3 = self._empty_bundle(self.ca3_size)

    def _empty_bundle(self, _size: int) -> SignalBundle:
        return SignalBundle(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            self.time_ms,
        )

    @staticmethod
    def _scale(bundle: SignalBundle, gain: float, emitted_ms: float) -> SignalBundle:
        if bundle.width == 0 or gain <= 0.0:
            return SignalBundle(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                emitted_ms,
            )
        return SignalBundle(
            bundle.indices,
            bundle.amplitudes * float(gain),
            emitted_ms,
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
        self.cortical = self._empty_bundle(self.cortical_size)
        self.dentate = self._empty_bundle(self.dg_size)
        self.ca3 = self._empty_bundle(self.ca3_size)
        if reset_usage:
            self.ca3_usage.fill(0.0)

    def _dentate_from_cortex(self, bundle: SignalBundle, emitted_ms: float) -> SignalBundle:
        cortical = bundle.as_dense(self.cortical_size)
        scores = np.sum(
            self._dg_weights * cortical[self._dg_sources],
            axis=1,
        )
        return self._top_bundle(scores, self.dg_winner_count, emitted_ms)

    def _ca3_from_previous(
        self,
        dentate: SignalBundle,
        previous_ca3: SignalBundle,
        emitted_ms: float,
    ) -> SignalBundle:
        dg = dentate.as_dense(self.dg_size)
        ca3 = previous_ca3.as_dense(self.ca3_size)
        scores = self.ca3_afferent_weights @ dg
        if previous_ca3.width:
            scores += self.recurrent_gain * (self.ca3_recurrent_weights @ ca3)
        scores /= 1.0 + self.ca3_homeostatic_pressure * self.ca3_usage
        return self._top_bundle(scores, self.ca3_winner_count, emitted_ms)

    def _cortical_return(self, ca3: SignalBundle, emitted_ms: float) -> SignalBundle:
        if ca3.width == 0:
            return self._empty_bundle(self.cortical_size)
        scores = self.cortical_return_weights @ ca3.as_dense(self.ca3_size)
        return self._top_bundle(scores, self.cortical_return_width, emitted_ms)

    @staticmethod
    def _hebbian_saturating_update(
        weights: np.ndarray,
        target_indices: np.ndarray,
        source_indices: np.ndarray,
        target_amplitudes: np.ndarray,
        source_amplitudes: np.ndarray,
        learning_rate: float,
    ) -> None:
        if target_indices.size == 0 or source_indices.size == 0:
            return
        block = weights[np.ix_(target_indices, source_indices)]
        coactivity = np.outer(target_amplitudes, source_amplitudes).astype(np.float32)
        block += float(learning_rate) * coactivity * (1.0 - block)
        weights[np.ix_(target_indices, source_indices)] = np.clip(block, 0.0, 1.0)

    def _learn_local_routes(
        self,
        previous_dentate: SignalBundle,
        previous_ca3: SignalBundle,
        next_ca3: SignalBundle,
        next_cortical: SignalBundle,
    ) -> None:
        self._hebbian_saturating_update(
            self.ca3_afferent_weights,
            next_ca3.indices,
            previous_dentate.indices,
            next_ca3.amplitudes,
            previous_dentate.amplitudes,
            self.afferent_learning_rate,
        )
        self._hebbian_saturating_update(
            self.ca3_recurrent_weights,
            next_ca3.indices,
            previous_ca3.indices,
            next_ca3.amplitudes,
            previous_ca3.amplitudes,
            self.recurrent_learning_rate,
        )
        if self.ca3_recurrent_weights.size:
            np.fill_diagonal(self.ca3_recurrent_weights, 0.0)
        self._hebbian_saturating_update(
            self.cortical_return_weights,
            next_cortical.indices,
            previous_ca3.indices,
            next_cortical.amplitudes,
            previous_ca3.amplitudes,
            self.return_learning_rate,
        )

    def advance(
        self,
        external_cortical: SignalBundle | None = None,
        *,
        learn: bool = False,
        dt_ms: float = 1.0,
    ) -> ParallelBundleStep:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        self.time_ms += float(dt_ms)
        now = self.time_ms

        previous_cortical = self.cortical
        previous_dentate = self.dentate
        previous_ca3 = self.ca3

        cortical_components: list[SignalBundle] = []
        if external_cortical is not None:
            if external_cortical.width and int(np.max(external_cortical.indices)) >= self.cortical_size:
                raise IndexError("external cortical bundle outside cortical population")
            cortical_components.append(self._scale(external_cortical, 1.0, now))
        if previous_cortical.width:
            cortical_components.append(
                self._scale(previous_cortical, self.cortical_persistence, now)
            )
            if self.cortical_projector is not None:
                cortical_components.append(
                    self._scale(
                        self.cortical_projector(previous_cortical),
                        self.cortical_recurrence_gain,
                        now,
                    )
                )
        if previous_ca3.width:
            cortical_components.append(
                self._scale(
                    self._cortical_return(previous_ca3, now),
                    self.cortical_return_gain,
                    now,
                )
            )

        if cortical_components:
            merged = SignalBundle.merge(
                cortical_components,
                size=self.cortical_size,
                emitted_ms=now,
            )
            next_cortical = self._top_bundle(
                merged.as_dense(self.cortical_size),
                self.cortical_winner_count,
                now,
            )
        else:
            next_cortical = self._empty_bundle(self.cortical_size)

        next_dentate = self._dentate_from_cortex(previous_cortical, now)
        next_ca3 = self._ca3_from_previous(previous_dentate, previous_ca3, now)

        if learn:
            self._learn_local_routes(
                previous_dentate,
                previous_ca3,
                next_ca3,
                next_cortical,
            )

        self.ca3_usage = (
            0.995 * self.ca3_usage
            + 0.005 * np.isin(
                np.arange(self.ca3_size), next_ca3.indices
            ).astype(np.float32)
        )
        self.cortical = next_cortical
        self.dentate = next_dentate
        self.ca3 = next_ca3
        return ParallelBundleStep(now, self.cortical, self.dentate, self.ca3)

    def learn_experience(
        self,
        cortical_bundle: SignalBundle,
        *,
        steps: int = 8,
        driven_steps: int = 5,
    ) -> None:
        if steps <= 0 or not 0 < driven_steps <= steps:
            raise ValueError("require steps > 0 and 0 < driven_steps <= steps")
        self.reset_dynamic()
        for tick in range(steps):
            self.advance(
                cortical_bundle if tick < driven_steps else None,
                learn=True,
            )

    def recall(
        self,
        cortical_bundle: SignalBundle,
        *,
        steps: int = 10,
        driven_steps: int = 2,
    ) -> ParallelBundleStep:
        if steps <= 0 or not 0 < driven_steps <= steps:
            raise ValueError("require steps > 0 and 0 < driven_steps <= steps")
        self.reset_dynamic()
        result: ParallelBundleStep | None = None
        for tick in range(steps):
            result = self.advance(
                cortical_bundle if tick < driven_steps else None,
                learn=False,
            )
        assert result is not None
        return result
