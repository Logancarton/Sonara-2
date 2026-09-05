from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PopulationParams:
    """Minimal LIF dynamics retained from the old Sonara donor."""

    resting_mv: float = -70.0
    threshold_mv: float = -55.0
    reset_mv: float = -75.0
    refractory_ms: float = 2.0
    tau_membrane_ms: float = 20.0


class NeuronPopulation:
    """
    Vectorized leaky integrate-and-fire population.

    The population owns neuron state. It does not know where spikes go; that
    authority belongs to SynapseProjection and PropagationNetwork.
    """

    def __init__(
        self,
        population_id: str,
        size: int,
        params: PopulationParams | None = None,
    ) -> None:
        if not population_id.strip():
            raise ValueError("population_id must be non-empty")
        if int(size) <= 0:
            raise ValueError("size must be > 0")

        self.population_id = population_id
        self.size = int(size)
        self.params = params or PopulationParams()

        self.membrane_mv = np.full(self.size, self.params.resting_mv, dtype=np.float32)
        self.refractory_remaining_ms = np.zeros(self.size, dtype=np.float32)
        self.spike_count = np.zeros(self.size, dtype=np.int64)
        self.last_spike_ms = np.full(self.size, -np.inf, dtype=np.float64)
        self.last_spikes = np.zeros(self.size, dtype=bool)
        self.time_ms = 0.0

    def reset(self) -> None:
        self.membrane_mv.fill(self.params.resting_mv)
        self.refractory_remaining_ms.fill(0.0)
        self.spike_count.fill(0)
        self.last_spike_ms.fill(-np.inf)
        self.last_spikes.fill(False)
        self.time_ms = 0.0

    def integrate(
        self,
        dt_ms: float,
        input_current: float | np.ndarray | None = None,
        *,
        now_ms: float | None = None,
    ) -> np.ndarray:
        """Advance the population once and return firing neuron indices."""
        dt = float(dt_ms)
        if dt <= 0.0:
            raise ValueError("dt_ms must be > 0")

        if now_ms is None:
            self.time_ms += dt
        else:
            self.time_ms = float(now_ms)

        if input_current is None:
            current = np.zeros(self.size, dtype=np.float32)
        elif np.isscalar(input_current):
            current = np.full(self.size, float(input_current), dtype=np.float32)
        else:
            current = np.asarray(input_current, dtype=np.float32)
            if current.shape != (self.size,):
                raise ValueError(
                    f"{self.population_id}: input_current must have shape {(self.size,)}, "
                    f"got {current.shape}"
                )

        tau = max(float(self.params.tau_membrane_ms), 1e-9)
        alpha = min(1.0, dt / tau)
        self.membrane_mv += (self.params.resting_mv - self.membrane_mv) * alpha

        self.refractory_remaining_ms -= dt
        np.maximum(self.refractory_remaining_ms, 0.0, out=self.refractory_remaining_ms)

        available = self.refractory_remaining_ms <= 0.0
        self.membrane_mv[available] += current[available]

        self.last_spikes = available & (self.membrane_mv >= self.params.threshold_mv)
        fired = np.flatnonzero(self.last_spikes)

        if fired.size:
            self.membrane_mv[fired] = self.params.reset_mv
            self.refractory_remaining_ms[fired] = self.params.refractory_ms
            self.spike_count[fired] += 1
            self.last_spike_ms[fired] = self.time_ms

        return fired.astype(np.int64, copy=False)

    def force_current(self, indices: Iterable[int], amplitude: float) -> np.ndarray:
        """Convenience helper for building an external-current vector."""
        current = np.zeros(self.size, dtype=np.float32)
        idx = np.fromiter((int(i) for i in indices), dtype=np.int64)
        if idx.size:
            if np.any(idx < 0) or np.any(idx >= self.size):
                raise IndexError(f"index outside population {self.population_id}")
            current[idx] = float(amplitude)
        return current
