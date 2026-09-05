from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .neural.network import PropagationNetwork, TickResult
from .neural.population import NeuronPopulation
from .neural.projection import SynapseProjection
from .neural.representation import AssemblyActivation, SparseRepresentationField


@dataclass(frozen=True)
class StimulusTickResult:
    activation: AssemblyActivation
    tick: TickResult


class SonaraRuntime:
    """Thin integration layer around the live propagation organism."""

    def __init__(self, network: PropagationNetwork | None = None) -> None:
        self.network = network or PropagationNetwork()
        self._representation_fields: Dict[str, SparseRepresentationField] = {}

    def add_population(self, population: NeuronPopulation) -> None:
        self.network.add_population(population)

    def add_projection(self, projection: SynapseProjection) -> None:
        self.network.add_projection(projection)

    def add_representation_field(self, field: SparseRepresentationField) -> None:
        if field.field_id in self._representation_fields:
            raise ValueError(f"duplicate representation field: {field.field_id}")
        population = self.network.populations.get(field.population_id)
        if population is None:
            raise ValueError(f"representation field targets unknown population: {field.population_id}")
        if population.size != field.population_size:
            raise ValueError(
                f"representation field {field.field_id} expects population size "
                f"{field.population_size}, got {population.size}"
            )
        self._representation_fields[field.field_id] = field

    def present_stimulus(
        self,
        field_id: str,
        features: np.ndarray,
        *,
        dt_ms: float = 1.0,
        learn: bool = True,
    ) -> StimulusTickResult:
        try:
            field = self._representation_fields[field_id]
        except KeyError as exc:
            raise ValueError(f"unknown representation field: {field_id}") from exc
        activation = field.encode(features, learn=learn)
        tick = self.network.tick(dt_ms, {field.population_id: activation.current})
        return StimulusTickResult(activation=activation, tick=tick)

    def tick(
        self,
        dt_ms: float = 1.0,
        external_currents: Dict[str, float | np.ndarray] | None = None,
    ) -> TickResult:
        return self.network.tick(dt_ms, external_currents)
