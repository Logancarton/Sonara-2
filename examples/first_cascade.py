"""First Sonara-2 proof: propagation is selected by the graph, not runtime code."""

from sonara import SonaraRuntime
from sonara.neural import NeuronPopulation, SynapseProjection


def build_runtime() -> SonaraRuntime:
    runtime = SonaraRuntime()
    for population_id in ("A", "B", "C", "D"):
        runtime.add_population(NeuronPopulation(population_id, 8))

    runtime.add_projection(SynapseProjection.one_to_one("A_B", "A", "B", 8, 16.0, delay_ms=1.0))
    runtime.add_projection(SynapseProjection.one_to_one("A_C", "A", "C", 8, 7.0, delay_ms=1.0))
    runtime.add_projection(SynapseProjection.one_to_one("B_D", "B", "D", 8, 16.0, delay_ms=1.0))
    return runtime


def main() -> None:
    runtime = build_runtime()
    for tick in range(5):
        stimulus = {"A": 16.0} if tick == 0 else None
        result = runtime.tick(1.0, stimulus)
        print(f"t={result.time_ms:>3.0f}ms spikes={result.spike_counts()}")


if __name__ == "__main__":
    main()
