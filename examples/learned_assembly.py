import numpy as np

from sonara import SonaraRuntime
from sonara.neural import NeuronPopulation, SparseRepresentationField, SynapseProjection

STIMULUS = np.array([1.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.1, 0.0])
SIMILAR = np.array([0.60, 0.23, -0.03, -0.386, 0.524, -0.02, 0.212, -0.337])
DIFFERENT = np.array([0.0, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0, 0.4])

runtime = SonaraRuntime()
runtime.add_population(NeuronPopulation("R", 96))
runtime.add_population(NeuronPopulation("D", 96))
field = SparseRepresentationField(
    "sensory",
    "R",
    population_size=96,
    feature_size=8,
    winner_count=8,
    seed=11,
    learning_rate=0.08,
)
runtime.add_representation_field(field)
runtime.add_projection(SynapseProjection.one_to_one("R_D", "R", "D", 96, 16.0, delay_ms=1.0))

before = field.encode(STIMULUS)
for _ in range(6):
    field.encode(STIMULUS, learn=True)
after = field.encode(STIMULUS)
similar = field.encode(SIMILAR)
different = field.encode(DIFFERENT)

print(f"sparsity={after.sparsity:.4f} winners={after.winner_indices.tolist()}")
print(f"identical_overlap={after.overlap(field.encode(STIMULUS)):.3f}")
print(f"similar_overlap={after.overlap(similar):.3f}")
print(f"different_overlap={after.overlap(different):.3f}")
print(f"mean_winner_drive_before={before.mean_winner_current:.3f}")
print(f"mean_winner_drive_after={after.mean_winner_current:.3f}")

first_tick = runtime.present_stimulus("sensory", STIMULUS, learn=False)
second_tick = runtime.tick(1.0)
print(f"t=1ms spikes={first_tick.tick.spike_counts()}")
print(f"t=2ms spikes={second_tick.spike_counts()}")
