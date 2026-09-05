from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .population import NeuronPopulation
from .projection import SynapseProjection


@dataclass(frozen=True)
class ScheduledDelivery:
    projection_id: str
    target_id: str
    due_ms: float
    current: np.ndarray


@dataclass
class TickResult:
    time_ms: float
    spikes: Dict[str, np.ndarray] = field(default_factory=dict)
    scheduled_deliveries: int = 0
    delivered_currents: Dict[str, float] = field(default_factory=dict)
    plastic_synapses_updated: int = 0

    def spike_counts(self) -> Dict[str, int]:
        return {key: int(value.size) for key, value in self.spikes.items()}


class PropagationNetwork:
    """
    Event-driven sparse neural graph.

    The runtime never contains population-specific routing. Spikes consult
    registered outgoing projections, deliveries arrive after their delays, and
    newly produced spikes recursively participate on later ticks.
    """

    def __init__(self) -> None:
        self.populations: Dict[str, NeuronPopulation] = {}
        self.projections: Dict[str, SynapseProjection] = {}
        self._outgoing: Dict[str, list[str]] = {}
        self._incoming: Dict[str, list[str]] = {}
        self._deliveries: list[ScheduledDelivery] = []
        self.time_ms = 0.0

    def add_population(self, population: NeuronPopulation) -> None:
        if population.population_id in self.populations:
            raise ValueError(f"duplicate population: {population.population_id}")
        self.populations[population.population_id] = population
        self._outgoing.setdefault(population.population_id, [])
        self._incoming.setdefault(population.population_id, [])

    def add_projection(self, projection: SynapseProjection) -> None:
        if projection.projection_id in self.projections:
            raise ValueError(f"duplicate projection: {projection.projection_id}")
        if projection.source_id not in self.populations:
            raise ValueError(f"unknown source population: {projection.source_id}")
        if projection.target_id not in self.populations:
            raise ValueError(f"unknown target population: {projection.target_id}")

        projection.validate_sizes(
            self.populations[projection.source_id].size,
            self.populations[projection.target_id].size,
        )
        self.projections[projection.projection_id] = projection
        self._outgoing[projection.source_id].append(projection.projection_id)
        self._incoming[projection.target_id].append(projection.projection_id)

    def outgoing(self, population_id: str) -> tuple[SynapseProjection, ...]:
        return tuple(self.projections[pid] for pid in self._outgoing.get(population_id, ()))

    def _collect_due_currents(self, now_ms: float) -> Dict[str, np.ndarray]:
        currents = {
            pid: np.zeros(pop.size, dtype=np.float32)
            for pid, pop in self.populations.items()
        }
        remaining: list[ScheduledDelivery] = []
        epsilon = 1e-9
        for delivery in self._deliveries:
            if delivery.due_ms <= now_ms + epsilon:
                currents[delivery.target_id] += delivery.current
            else:
                remaining.append(delivery)
        self._deliveries = remaining
        return currents

    def tick(
        self,
        dt_ms: float,
        external_currents: Dict[str, float | np.ndarray] | None = None,
    ) -> TickResult:
        dt = float(dt_ms)
        if dt <= 0.0:
            raise ValueError("dt_ms must be > 0")
        self.time_ms += dt

        currents = self._collect_due_currents(self.time_ms)
        external = external_currents or {}
        for population_id, value in external.items():
            if population_id not in self.populations:
                raise ValueError(f"external current targets unknown population: {population_id}")
            if np.isscalar(value):
                currents[population_id] += float(value)
            else:
                arr = np.asarray(value, dtype=np.float32)
                if arr.shape != currents[population_id].shape:
                    raise ValueError(
                        f"external current for {population_id} must have shape "
                        f"{currents[population_id].shape}, got {arr.shape}"
                    )
                currents[population_id] += arr

        spikes: Dict[str, np.ndarray] = {}
        delivered_currents: Dict[str, float] = {}
        for population_id, population in self.populations.items():
            delivered_currents[population_id] = float(np.sum(currents[population_id]))
            spikes[population_id] = population.integrate(
                dt,
                currents[population_id],
                now_ms=self.time_ms,
            )

        plastic_updates = 0
        for projection in self.projections.values():
            plastic_updates += projection.learn_from_post_spikes(
                spikes[projection.target_id],
                self.time_ms,
            )

        scheduled = 0
        for source_id, fired in spikes.items():
            if fired.size == 0:
                continue
            for projection in self.outgoing(source_id):
                projection.record_pre_spikes(fired, self.time_ms)
                current = projection.current_from_spikes(
                    fired,
                    self.populations[projection.target_id].size,
                )
                if not np.any(current):
                    continue
                self._deliveries.append(
                    ScheduledDelivery(
                        projection_id=projection.projection_id,
                        target_id=projection.target_id,
                        due_ms=self.time_ms + projection.delay_ms,
                        current=current,
                    )
                )
                scheduled += 1

        return TickResult(
            time_ms=self.time_ms,
            spikes=spikes,
            scheduled_deliveries=scheduled,
            delivered_currents=delivered_currents,
            plastic_synapses_updated=plastic_updates,
        )

    def run(
        self,
        ticks: int,
        dt_ms: float,
        stimulus_schedule: Dict[int, Dict[str, float | np.ndarray]] | None = None,
    ) -> list[TickResult]:
        if int(ticks) < 0:
            raise ValueError("ticks must be >= 0")
        schedule = stimulus_schedule or {}
        results: list[TickResult] = []
        for tick_index in range(int(ticks)):
            results.append(self.tick(dt_ms, schedule.get(tick_index)))
        return results

    def pending_delivery_count(self) -> int:
        return len(self._deliveries)
