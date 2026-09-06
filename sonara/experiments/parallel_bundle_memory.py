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

    Cortex remains the sole cortical owner. Broad cortical bundles fan through
    existing excitatory routes into DG and CA3 while short activity traces keep
    recent parts of the moving cascade causally available. Plasticity is driven
    by overlap between decaying pre/post activity traces, so adjacent moments of
    one cascade can strengthen the routes they actually used without an episode
    lookup or semantic label.

    Long-term memory remains in synaptic weights. Eligibility/activity traces
    are transient and reset between experiences.
    """

    def __init__(
        self,
        cortical_size: int,
        *,
        cortical_winner_count: int,
        dg_size: int = 2048,
        dg_winner_count: int = 64,
        dg_fan_out: int = 32,
        cortical_trace_decay: float = 0.75,
        eligibility_decay: float = 0.82,
        dg_homeostatic_pressure: float = 2.0,
        ca3_size: int = 512,
        ca3_winner_count: int = 16,
        ca3_direct_seed_count: int = 4,
        ca3_direct_fan_out: int = 8,
        ca3_dg_fan_in: int = 24,
        ca3_recurrent_fan_out: int = 24,
        ca3_recurrent_output_budget: float = 1.0,
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
        if cortical_size <= 0 or dg_size <= 0 or ca3_size <= 1:
            raise ValueError("population sizes must be > 0 and ca3_size must be > 1")
        if not 0 < cortical_winner_count <= cortical_size:
            raise ValueError("invalid cortical_winner_count")
        if not 0 < dg_winner_count <= dg_size:
            raise ValueError("invalid dg_winner_count")
        if not 0 <= cortical_trace_decay < 1.0:
            raise ValueError("cortical_trace_decay must be in [0, 1)")
        if not 0 <= eligibility_decay < 1.0:
            raise ValueError("eligibility_decay must be in [0, 1)")
        if dg_homeostatic_pressure < 0.0:
            raise ValueError("dg_homeostatic_pressure must be >= 0")
        if not 0 < ca3_direct_seed_count <= ca3_winner_count <= ca3_size:
            raise ValueError("invalid CA3 winner/seed counts")
        if dg_fan_out <= 0 or ca3_dg_fan_in <= 0 or ca3_direct_fan_out <= 0:
            raise ValueError("fan-in/fan-out values must be > 0")
        if not 0 < ca3_recurrent_fan_out < ca3_size:
            raise ValueError("ca3_recurrent_fan_out must be in [1, ca3_size)")
        if ca3_recurrent_output_budget <= 0.0:
            raise ValueError("ca3_recurrent_output_budget must be > 0")
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
        self.cortical_trace_decay = float(cortical_trace_decay)
        self.eligibility_decay = float(eligibility_decay)
        self.dg_homeostatic_pressure = float(dg_homeostatic_pressure)
        self.ca3_size = int(ca3_size)
        self.ca3_winner_count = int(ca3_winner_count)
        self.ca3_direct_seed_count = int(ca3_direct_seed_count)
        self.ca3_direct_fan_out = int(ca3_direct_fan_out)
        self.ca3_recurrent_fan_out = int(ca3_recurrent_fan_out)
        self.ca3_recurrent_output_budget = float(ca3_recurrent_output_budget)
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

        # Cortex -> DG is a sparse excitatory graph. Every active cortical
        # source fans out; DG cells sum whatever routes converge on them.
        self.dg_source_edges = np.repeat(
            np.arange(self.cortical_size, dtype=np.int32),
            self.dg_fan_out,
        )
        dg_targets = np.empty((self.cortical_size, self.dg_fan_out), dtype=np.int32)
        for source_id in range(self.cortical_size):
            dg_targets[source_id] = self.rng.choice(
                self.dg_size,
                size=self.dg_fan_out,
                replace=False,
            )
        self.dg_target_edges = dg_targets.reshape(-1)
        dg_weights = self.rng.uniform(
            0.5, 1.0, size=self.dg_source_edges.size
        ).astype(np.float32)
        self.dg_edge_weights = dg_weights
        self._scale_dg_incoming()

        # Direct cortical/entorhinal seed into CA3 is also sparse and excitatory.
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
            np.linalg.norm(direct_weights, axis=1, keepdims=True), 1e-12
        )
        self.ca3_direct_weights = direct_weights.reshape(-1)

        # Mossy-fiber-like DG -> CA3 path: sparse, strong, excitatory fan-in.
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
            np.linalg.norm(self.ca3_dg_weights, axis=1, keepdims=True), 1e-12
        )

        # CA3 recurrence is a sparse structural graph. Only existing routes can
        # potentiate, and each source has a finite recurrent output budget.
        self.ca3_recurrent_source_edges = np.repeat(
            np.arange(self.ca3_size, dtype=np.int32),
            self.ca3_recurrent_fan_out,
        )
        recurrent_targets = np.empty(
            (self.ca3_size, self.ca3_recurrent_fan_out), dtype=np.int32
        )
        for source_id in range(self.ca3_size):
            choices = self.rng.choice(
                self.ca3_size - 1,
                size=self.ca3_recurrent_fan_out,
                replace=False,
            ).astype(np.int32)
            choices[choices >= source_id] += 1
            recurrent_targets[source_id] = choices
        self.ca3_recurrent_target_edges = recurrent_targets.reshape(-1)
        self.ca3_recurrent_weights = np.zeros(
            self.ca3_recurrent_source_edges.size, dtype=np.float32
        )

        self.cortical_return_weights = np.zeros(
            (self.cortical_size, self.ca3_size), dtype=np.float32
        )

        # Usage is slow competitive state. Eligibility traces are fast transient
        # state and are never a second long-term memory store.
        self.dg_usage = np.zeros(self.dg_size, dtype=np.float32)
        self.ca3_usage = np.zeros(self.ca3_size, dtype=np.float32)
        self.cortical_trace = np.zeros(self.cortical_size, dtype=np.float32)
        self.cortical_eligibility = np.zeros(self.cortical_size, dtype=np.float32)
        self.dentate_eligibility = np.zeros(self.dg_size, dtype=np.float32)
        self.ca3_eligibility = np.zeros(self.ca3_size, dtype=np.float32)

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
    def _top_bundle(scores: np.ndarray, count: int, emitted_ms: float) -> SignalBundle:
        value = np.asarray(scores, dtype=np.float32)
        positive = np.flatnonzero(value > 1e-12)
        if positive.size == 0:
            return SignalBundle(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                emitted_ms,
            )
        if positive.size > count:
            chosen = positive[np.argpartition(value[positive], -count)[-count:]]
        else:
            chosen = positive
        chosen = chosen[np.argsort(-value[chosen], kind="stable")]
        peak = max(float(np.max(value[chosen])), 1e-12)
        return SignalBundle(chosen, value[chosen] / peak, emitted_ms)

    @staticmethod
    def _decay_and_refresh(
        trace: np.ndarray,
        current: np.ndarray,
        decay: float,
    ) -> np.ndarray:
        """Keep recent activity alive while letting unsupported activity fade."""
        trace *= decay
        np.maximum(trace, np.asarray(current, dtype=np.float32), out=trace)
        return trace

    def reset_dynamic(self, *, reset_usage: bool = False) -> None:
        self.time_ms = 0.0
        self.dentate = self._empty_bundle()
        self.ca3 = self._empty_bundle()
        self.cortical_trace.fill(0.0)
        self.cortical_eligibility.fill(0.0)
        self.dentate_eligibility.fill(0.0)
        self.ca3_eligibility.fill(0.0)
        if reset_usage:
            self.dg_usage.fill(0.0)
            self.ca3_usage.fill(0.0)

    def _update_cortical_trace(self, cortical_bundle: SignalBundle) -> np.ndarray:
        current = cortical_bundle.as_dense(self.cortical_size)
        return self._decay_and_refresh(
            self.cortical_trace,
            current,
            self.cortical_trace_decay,
        )

    def _scale_dg_incoming(self) -> None:
        target_norm = np.sqrt(
            np.bincount(
                self.dg_target_edges,
                weights=np.square(self.dg_edge_weights),
                minlength=self.dg_size,
            )
        ).astype(np.float32)
        self.dg_edge_weights /= np.maximum(
            target_norm[self.dg_target_edges], 1e-12
        )

    def _dentate_from_cortical_state(
        self,
        cortical_state: np.ndarray,
        emitted_ms: float,
    ) -> SignalBundle:
        edge_signal = self.dg_edge_weights * cortical_state[self.dg_source_edges]
        scores = np.bincount(
            self.dg_target_edges,
            weights=edge_signal,
            minlength=self.dg_size,
        ).astype(np.float32)
        scores /= 1.0 + self.dg_homeostatic_pressure * self.dg_usage
        return self._top_bundle(scores, self.dg_winner_count, emitted_ms)

    def _direct_cortical_scores(self, cortical: np.ndarray) -> np.ndarray:
        edge_signal = self.ca3_direct_weights * cortical[self.ca3_direct_source_edges]
        return np.bincount(
            self.ca3_direct_target_edges,
            weights=edge_signal,
            minlength=self.ca3_size,
        ).astype(np.float32)

    def _recurrent_ca3_scores(self, ca3: np.ndarray) -> np.ndarray:
        edge_signal = self.ca3_recurrent_weights * ca3[self.ca3_recurrent_source_edges]
        return np.bincount(
            self.ca3_recurrent_target_edges,
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
            scores += self.recurrent_gain * self._recurrent_ca3_scores(ca3)
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

    def _learn_dg_routes(
        self,
        cortical_pre: np.ndarray,
        dentate_post: np.ndarray,
    ) -> None:
        pre = cortical_pre[self.dg_source_edges]
        post = dentate_post[self.dg_target_edges]
        eligible = (pre > 0.0) & (post > 0.0)
        if not np.any(eligible):
            return
        coactivity = pre[eligible] * post[eligible]
        self.dg_edge_weights[eligible] += (
            self.afferent_learning_rate
            * coactivity
            * (1.0 - self.dg_edge_weights[eligible])
        )
        np.clip(self.dg_edge_weights, 0.0, 1.0, out=self.dg_edge_weights)
        self._scale_dg_incoming()

    def _learn_direct_cortical_afferents(
        self,
        cortical_pre: np.ndarray,
        ca3_post: np.ndarray,
    ) -> None:
        pre = cortical_pre[self.ca3_direct_source_edges]
        post = ca3_post[self.ca3_direct_target_edges]
        eligible = (pre > 0.0) & (post > 0.0)
        if not np.any(eligible):
            return
        coactivity = pre[eligible] * post[eligible]
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
            np.linalg.norm(rows, axis=1, keepdims=True), 1e-12
        )

    def _learn_mossy_afferents(
        self,
        dentate_pre: np.ndarray,
        ca3_post: np.ndarray,
    ) -> None:
        source_strength = dentate_pre[self.ca3_dg_sources]
        target_strength = ca3_post[:, None]
        coactivity = source_strength * target_strength
        if not np.any(coactivity > 0.0):
            return
        self.ca3_dg_weights += (
            self.afferent_learning_rate
            * coactivity
            * (1.0 - self.ca3_dg_weights)
        )
        np.maximum(self.ca3_dg_weights, 0.0, out=self.ca3_dg_weights)
        self.ca3_dg_weights /= np.maximum(
            np.linalg.norm(self.ca3_dg_weights, axis=1, keepdims=True), 1e-12
        )

    def _learn_recurrent_routes(
        self,
        ca3_pre: np.ndarray,
        ca3_post: np.ndarray,
    ) -> None:
        pre = ca3_pre[self.ca3_recurrent_source_edges]
        post = ca3_post[self.ca3_recurrent_target_edges]
        eligible = (pre > 0.0) & (post > 0.0)
        if np.any(eligible):
            coactivity = pre[eligible] * post[eligible]
            self.ca3_recurrent_weights[eligible] += (
                self.recurrent_learning_rate
                * coactivity
                * (1.0 - self.ca3_recurrent_weights[eligible])
            )

        rows = self.ca3_recurrent_weights.reshape(
            self.ca3_size, self.ca3_recurrent_fan_out
        )
        totals = np.sum(rows, axis=1, keepdims=True)
        scale = np.ones_like(totals)
        over_budget = totals[:, 0] > self.ca3_recurrent_output_budget
        scale[over_budget, 0] = (
            self.ca3_recurrent_output_budget / totals[over_budget, 0]
        )
        rows *= scale
        np.clip(rows, 0.0, 1.0, out=rows)

    def _learn_cortical_return(
        self,
        ca3_pre: np.ndarray,
        cortical_post: np.ndarray,
    ) -> None:
        if not np.any(ca3_pre > 0.0) or not np.any(cortical_post > 0.0):
            return
        coactivity = np.outer(cortical_post, ca3_pre).astype(np.float32)
        self.cortical_return_weights += (
            self.return_learning_rate
            * coactivity
            * (1.0 - self.cortical_return_weights)
        )
        np.clip(
            self.cortical_return_weights,
            0.0,
            1.0,
            out=self.cortical_return_weights,
        )

    def advance(
        self,
        cortical_bundle: SignalBundle,
        *,
        learn: bool = False,
        dt_ms: float = 1.0,
    ) -> ParallelBundleMemoryStep:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        if (
            cortical_bundle.width
            and int(np.max(cortical_bundle.indices)) >= self.cortical_size
        ):
            raise IndexError("cortical bundle outside cortical population")

        self.time_ms += float(dt_ms)
        now = self.time_ms
        previous_dentate = self.dentate
        previous_ca3 = self.ca3

        cortical_return = self._cortical_return(previous_ca3, now)
        cortical_state = self._update_cortical_trace(cortical_bundle)
        next_dentate = self._dentate_from_cortical_state(cortical_state, now)
        next_ca3 = self._ca3_from_inputs(
            previous_dentate,
            cortical_bundle,
            previous_ca3,
            now,
        )

        # Snapshot pre traces before the new postsynaptic bundles are merged in.
        # This keeps the causal order: lingering earlier activity can modify a
        # route when a later bundle arrives.
        cortical_current = cortical_bundle.as_dense(self.cortical_size)
        next_dg_current = next_dentate.as_dense(self.dg_size)
        next_ca3_current = next_ca3.as_dense(self.ca3_size)

        self._decay_and_refresh(
            self.cortical_eligibility,
            cortical_current,
            self.eligibility_decay,
        )
        self.dentate_eligibility *= self.eligibility_decay
        self.ca3_eligibility *= self.eligibility_decay

        cortical_pre = self.cortical_eligibility.copy()
        dentate_pre = self.dentate_eligibility.copy()
        ca3_pre = self.ca3_eligibility.copy()

        dentate_post = np.maximum(
            dentate_pre,
            next_dg_current,
        ).astype(np.float32)
        ca3_post = np.maximum(
            ca3_pre,
            next_ca3_current,
        ).astype(np.float32)

        if learn:
            self._learn_dg_routes(cortical_pre, dentate_post)
            self._learn_mossy_afferents(dentate_pre, ca3_post)
            self._learn_direct_cortical_afferents(cortical_pre, ca3_post)
            self._learn_recurrent_routes(ca3_pre, ca3_post)
            self._learn_cortical_return(ca3_pre, cortical_pre)

        # Carry the new postsynaptic activity forward into the next millisecond.
        self.dentate_eligibility[:] = dentate_post
        self.ca3_eligibility[:] = ca3_post

        self.dg_usage = (
            0.995 * self.dg_usage
            + 0.005
            * np.isin(
                np.arange(self.dg_size), next_dentate.indices
            ).astype(np.float32)
        )
        self.ca3_usage = (
            0.995 * self.ca3_usage
            + 0.005
            * np.isin(
                np.arange(self.ca3_size), next_ca3.indices
            ).astype(np.float32)
        )

        self.dentate = next_dentate
        self.ca3 = next_ca3
        return ParallelBundleMemoryStep(
            time_ms=now,
            dentate=next_dentate,
            ca3=next_ca3,
            cortical_return=cortical_return,
        )
