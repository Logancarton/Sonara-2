from __future__ import annotations

from typing import Dict

import numpy as np

from .neural.network import PropagationNetwork, TickResult
from .neural.population import NeuronPopulation
from .neural.projection import SynapseProjection


class SonaraRuntime:
    """Thin integration layer around the live propagation organism."""

    def __init__(self, network: PropagationNetwork | None = None) -> None:
        self.network = network or PropagationNetwork()

    def add_population(self, population: NeuronPopulation) -> None:
        self.network.add_population(population)

    def add_projection(self, projection: SynapseProjection) -> None:
        self.network.add_projection(projection)

    def tick(
        self,
        dt_ms: float = 1.0,
        external_currents: Dict[str, float | np.ndarray] | None = None,
    ) -> TickResult:
        return self.network.tick(dt_ms, external_currents)
