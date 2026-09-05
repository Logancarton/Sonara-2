from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    """One anchored information stream entering the generic neural sheet."""

    name: str
    feature_size: int
    anchor: tuple[float, float]
    sigma: float = 0.28
    gain: float = 3.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stream name must be non-empty")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be > 0")
        if not (0.0 <= self.anchor[0] <= 1.0 and 0.0 <= self.anchor[1] <= 1.0):
            raise ValueError("stream anchor coordinates must be in [0, 1]")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        if self.gain <= 0.0:
            raise ValueError("gain must be > 0")


@dataclass(frozen=True)
class SheetStep:
    time_ms: float
    winner_indices: np.ndarray
    winner_activity: np.ndarray
    active_fraction: float
    mean_winner_coldness: float


@dataclass(frozen=True)
class SeparationMetrics:
    within_family_overlap: float
    between_family_overlap: float

    @property
    def separation(self) -> float:
        return self.within_family_overlap - self.between_family_overlap


class FastCorticalSheet:
    """
    Generic 2-D neural sheet for testing Sonara's signal-flow hypothesis.

    The experiment deliberately assigns no semantic function to intermediate
    coordinates. Function can only arise from stream anchors, connection
    geometry, recurrent state, sparse competition, and local plasticity.

    Runtime math is vectorized:
    - one dense matrix-vector product per present input stream,
    - edge-array gather + bincount for sparse recurrent propagation,
    - one top-k competition over the sheet.

    There are no Python loops over neurons during a step.
    """

    def __init__(
        self,
        width: int,
        height: int,
        streams: Sequence[StreamSpec],
        *,
        sparsity: float = 0.04,
        seed: int = 0,
        local_degree: int = 24,
        long_range_degree: int = 6,
        local_radius: float = 0.07,
        hot_tau_ms: float = 6.0,
        cold_tau_ms: float = 90.0,
        recurrence_base_gain: float = 0.20,
        recurrence_cold_gain: float = 1.20,
        coincidence_gain: float = 0.75,
        firing_threshold: float = 0.10,
        afferent_learning_rate: float = 0.04,
        recurrent_learning_rate: float = 0.03,
        homeostatic_pressure: float = 0.60,
    ) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("width and height must both be > 1")
        if not streams:
            raise ValueError("at least one stream is required")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("sparsity must be in (0, 1]")
        if local_degree < 0 or long_range_degree < 0:
            raise ValueError("connection degrees must be >= 0")
        if local_degree + long_range_degree == 0:
            raise ValueError("at least one recurrent edge per source is required")
        if local_radius <= 0.0:
            raise ValueError("local_radius must be > 0")
        if hot_tau_ms <= 0.0 or cold_tau_ms < hot_tau_ms:
            raise ValueError("require 0 < hot_tau_ms <= cold_tau_ms")

        stream_names = [stream.name for stream in streams]
        if len(stream_names) != len(set(stream_names)):
            raise ValueError("stream names must be unique")

        self.width = int(width)
        self.height = int(height)
        self.size = self.width * self.height
        self.streams = tuple(streams)
        self.sparsity = float(sparsity)
        self.winner_budget = max(1, int(round(self.size * self.sparsity)))
        self.rng = np.random.default_rng(int(seed))
        self.coincidence_gain = float(coincidence_gain)
        self.firing_threshold = float(firing_threshold)
        self.afferent_learning_rate = float(afferent_learning_rate)
        self.recurrent_learning_rate = float(recurrent_learning_rate)
        self.homeostatic_pressure = float(homeostatic_pressure)
        self.time_ms = 0.0

        xs = np.linspace(0.0, 1.0, self.width)
        ys = np.linspace(0.0, 1.0, self.height)
        xx, yy = np.meshgrid(xs, ys)
        self.coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)

        self._afferent_weights: list[np.ndarray] = []
        self._stream_envelopes: list[np.ndarray] = []
        for stream in self.streams:
            weights = self.rng.normal(size=(self.size, stream.feature_size)).astype(np.float32)
            weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
            self._afferent_weights.append(weights)

            anchor = np.asarray(stream.anchor, dtype=np.float32)
            distance = np.linalg.norm(self.coordinates - anchor, axis=1)
            envelope = np.exp(-(distance**2) / (2.0 * stream.sigma**2)).astype(np.float32)
            self._stream_envelopes.append(envelope)

        distance_to_anchor = np.stack(
            [
                np.linalg.norm(
                    self.coordinates - np.asarray(stream.anchor, dtype=np.float32),
                    axis=1,
                )
                for stream in self.streams
            ]
        ).min(axis=0)
        distance_span = float(np.ptp(distance_to_anchor))
        if distance_span <= 1e-12:
            coldness = np.zeros(self.size, dtype=np.float32)
        else:
            coldness = (
                (distance_to_anchor - float(np.min(distance_to_anchor))) / distance_span
            ).astype(np.float32)
        self.coldness = coldness
        self.tau_ms = (
            hot_tau_ms + (cold_tau_ms - hot_tau_ms) * self.coldness
        ).astype(np.float32)
        self.recurrence_gain = (
            recurrence_base_gain + recurrence_cold_gain * self.coldness
        ).astype(np.float32)

        self.source_edges, self.target_edges = self._build_recurrent_edges(
            local_degree=local_degree,
            long_range_degree=long_range_degree,
            local_radius=local_radius,
        )
        self.recurrent_weights = self.rng.uniform(
            0.025,
            0.075,
            size=self.source_edges.size,
        ).astype(np.float32)

        self.state = np.zeros(self.size, dtype=np.float32)
        self.activity = np.zeros(self.size, dtype=np.float32)
        self.usage = np.zeros(self.size, dtype=np.float32)

    @property
    def recurrent_edge_count(self) -> int:
        return int(self.source_edges.size)

    def _build_recurrent_edges(
        self,
        *,
        local_degree: int,
        long_range_degree: int,
        local_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        source: list[int] = []
        target: list[int] = []

        # Construction is one-time. Runtime propagation is fully vectorized.
        for source_id, (x_coord, y_coord) in enumerate(self.coordinates):
            if local_degree:
                local_x = np.clip(
                    x_coord + self.rng.normal(scale=local_radius, size=local_degree),
                    0.0,
                    1.0,
                )
                local_y = np.clip(
                    y_coord + self.rng.normal(scale=local_radius, size=local_degree),
                    0.0,
                    1.0,
                )
                x_index = np.rint(local_x * (self.width - 1)).astype(np.int64)
                y_index = np.rint(local_y * (self.height - 1)).astype(np.int64)
                local_targets = y_index * self.width + x_index
                local_targets[local_targets == source_id] = (source_id + 1) % self.size
                source.extend([source_id] * local_degree)
                target.extend(local_targets.tolist())

            if long_range_degree:
                long_targets = self.rng.integers(
                    0,
                    self.size - 1,
                    size=long_range_degree,
                    dtype=np.int64,
                )
                long_targets[long_targets >= source_id] += 1
                source.extend([source_id] * long_range_degree)
                target.extend(long_targets.tolist())

        return np.asarray(source, dtype=np.int32), np.asarray(target, dtype=np.int32)

    @staticmethod
    def _normalize(vector: np.ndarray, expected_size: int, stream_name: str) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (expected_size,):
            raise ValueError(
                f"stream {stream_name!r} expects shape {(expected_size,)}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"stream {stream_name!r} must be finite")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            return np.zeros_like(value)
        return value / norm

    def compartment_activity(
        self,
        inputs: Mapping[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Return one nonlinear dendritic-like branch vector per information stream."""
        inputs = inputs or {}
        known = {stream.name for stream in self.streams}
        unknown = set(inputs) - known
        if unknown:
            raise ValueError(f"unknown streams: {sorted(unknown)}")

        branches: list[np.ndarray] = []
        normalized_inputs: list[np.ndarray] = []
        for index, stream in enumerate(self.streams):
            raw = inputs.get(stream.name)
            if raw is None:
                normalized = np.zeros(stream.feature_size, dtype=np.float32)
                branch = np.zeros(self.size, dtype=np.float32)
            else:
                normalized = self._normalize(raw, stream.feature_size, stream.name)
                alignment = self._afferent_weights[index] @ normalized
                branch = (
                    np.maximum(alignment, 0.0)
                    * self._stream_envelopes[index]
                    * stream.gain
                ).astype(np.float32)
            normalized_inputs.append(normalized)
            branches.append(branch)

        return np.stack(branches), tuple(normalized_inputs)

    def convergent_drive(
        self,
        inputs: Mapping[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        """
        Combine separate branches with a cheap nonlinear coincidence term.

        The pair term is the sum of all pairwise branch products, computed
        without explicitly constructing every pair:

            0.5 * ((sum branches)^2 - sum(branch^2))
        """
        branches, normalized = self.compartment_activity(inputs)
        branch_sum = np.sum(branches, axis=0)
        pair_coincidence = 0.5 * np.maximum(
            branch_sum**2 - np.sum(branches**2, axis=0),
            0.0,
        )
        drive = branch_sum + self.coincidence_gain * pair_coincidence
        return drive.astype(np.float32), branches, normalized

    def recurrent_current(self) -> np.ndarray:
        edge_signal = self.recurrent_weights * self.activity[self.source_edges]
        current = np.bincount(
            self.target_edges,
            weights=edge_signal,
            minlength=self.size,
        ).astype(np.float32)
        return current * self.recurrence_gain

    def step(
        self,
        inputs: Mapping[str, np.ndarray] | None = None,
        *,
        dt_ms: float = 1.0,
        learn: bool = False,
        recurrent: bool = True,
    ) -> SheetStep:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        self.time_ms += float(dt_ms)

        direct_drive, branches, normalized_inputs = self.convergent_drive(inputs)
        recurrent_drive = self.recurrent_current() if recurrent else 0.0
        target_state = direct_drive + recurrent_drive

        alpha = 1.0 - np.exp(-float(dt_ms) / np.maximum(self.tau_ms, 1e-6))
        self.state += alpha.astype(np.float32) * (target_state - self.state)

        competition_score = np.maximum(self.state - self.firing_threshold, 0.0)
        competition_score /= 1.0 + self.homeostatic_pressure * self.usage
        positive = np.flatnonzero(competition_score > 0.0)
        if positive.size > self.winner_budget:
            selected = positive[
                np.argpartition(
                    competition_score[positive],
                    -self.winner_budget,
                )[-self.winner_budget :]
            ]
            selected = selected[
                np.argsort(-competition_score[selected], kind="stable")
            ]
        else:
            selected = positive[
                np.argsort(-competition_score[positive], kind="stable")
            ]

        self.activity.fill(0.0)
        if selected.size:
            selected_scores = competition_score[selected]
            peak = max(float(np.max(selected_scores)), 1e-12)
            self.activity[selected] = selected_scores / peak

        self.usage = (
            0.995 * self.usage + 0.005 * (self.activity > 0.0).astype(np.float32)
        )

        if learn and selected.size:
            self._learn_afferents(selected, branches, normalized_inputs)
            self._learn_recurrent_edges()

        return SheetStep(
            time_ms=self.time_ms,
            winner_indices=selected.copy(),
            winner_activity=self.activity[selected].copy(),
            active_fraction=float(selected.size / self.size),
            mean_winner_coldness=(
                float(np.mean(self.coldness[selected])) if selected.size else 0.0
            ),
        )

    def _learn_afferents(
        self,
        selected: np.ndarray,
        branches: np.ndarray,
        normalized_inputs: tuple[np.ndarray, ...],
    ) -> None:
        for stream_index, normalized in enumerate(normalized_inputs):
            if not np.any(normalized):
                continue
            branch_strength = branches[stream_index, selected]
            peak = float(np.max(branch_strength))
            if peak <= 1e-12:
                continue
            eta = (
                self.afferent_learning_rate * (branch_strength / peak)
            )[:, None]
            updated = (
                (1.0 - eta) * self._afferent_weights[stream_index][selected]
                + eta * normalized
            )
            updated /= np.maximum(
                np.linalg.norm(updated, axis=1, keepdims=True),
                1e-12,
            )
            self._afferent_weights[stream_index][selected] = updated.astype(np.float32)

    def _learn_recurrent_edges(self) -> None:
        active = self.activity > 0.0
        eligible = active[self.source_edges] & active[self.target_edges]
        if not np.any(eligible):
            return
        self.recurrent_weights[eligible] += self.recurrent_learning_rate * (
            1.0 - self.recurrent_weights[eligible]
        )
        np.clip(self.recurrent_weights, 0.0, 1.0, out=self.recurrent_weights)

    def reset_state(self, *, reset_usage: bool = False) -> None:
        self.state.fill(0.0)
        self.activity.fill(0.0)
        self.time_ms = 0.0
        if reset_usage:
            self.usage.fill(0.0)

    def settle(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        cue_steps: int = 4,
        blank_steps: int = 0,
        learn: bool = False,
        recurrent: bool = True,
    ) -> np.ndarray:
        if cue_steps <= 0 or blank_steps < 0:
            raise ValueError("cue_steps must be > 0 and blank_steps must be >= 0")
        winners = np.empty(0, dtype=np.int64)
        for _ in range(cue_steps):
            winners = self.step(inputs, learn=learn, recurrent=recurrent).winner_indices
        for _ in range(blank_steps):
            winners = self.step({}, learn=False, recurrent=recurrent).winner_indices
        return winners

    def response_centroid(self, winner_indices: np.ndarray) -> tuple[float, float]:
        winners = np.asarray(winner_indices, dtype=np.int64)
        if winners.size == 0:
            raise ValueError("cannot compute centroid of an empty response")
        if np.any(winners < 0) or np.any(winners >= self.size):
            raise IndexError("winner index outside sheet")
        centroid = np.mean(self.coordinates[winners], axis=0)
        return float(centroid[0]), float(centroid[1])


def assembly_overlap(left: np.ndarray, right: np.ndarray, winner_budget: int) -> float:
    if winner_budget <= 0:
        raise ValueError("winner_budget must be > 0")
    return float(np.intersect1d(left, right).size / winner_budget)


def separation_metrics(
    assemblies: Sequence[Sequence[np.ndarray]],
    winner_budget: int,
) -> SeparationMetrics:
    within: list[float] = []
    between: list[float] = []
    for family_index, family in enumerate(assemblies):
        for left_index in range(len(family)):
            for right_index in range(left_index + 1, len(family)):
                within.append(
                    assembly_overlap(
                        family[left_index],
                        family[right_index],
                        winner_budget,
                    )
                )
        for other_index in range(family_index + 1, len(assemblies)):
            other = assemblies[other_index]
            for example_index in range(min(len(family), len(other))):
                between.append(
                    assembly_overlap(
                        family[example_index],
                        other[example_index],
                        winner_budget,
                    )
                )
    return SeparationMetrics(
        within_family_overlap=float(np.mean(within)) if within else 0.0,
        between_family_overlap=float(np.mean(between)) if between else 0.0,
    )
