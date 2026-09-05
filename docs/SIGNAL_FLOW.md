# Signal Flow

## Current live path

```text
numeric feature vector
  → SparseRepresentationField.encode()
  → deterministic feature-to-neuron scores
  → k-winner competition
  → sparse current vector
  → NeuronPopulation.integrate()
  → firing neuron indices
  → PropagationNetwork.outgoing(source)
  → SynapseProjection.current_from_spikes()
  → ScheduledDelivery(delay)
  → target current on a later tick
  → target NeuronPopulation.integrate()
  → target spikes
  → SynapseProjection.learn_from_post_spikes()
```

Repeated stimulus exposure also updates only the winning feature-to-neuron afferent rows inside `SparseRepresentationField`, increasing future neural drive for those recurring patterns.

Ownership is intentional:

- `SparseRepresentationField` owns numeric encoding, k-winner competition, and local feature-to-neuron adaptation.
- `NeuronPopulation` owns membrane, threshold, refractory, and spike state.
- `SynapseProjection` owns sparse neural-to-neural connectivity, weights, delays, and local synaptic learning.
- `PropagationNetwork` owns graph registration and event delivery.
- `SonaraRuntime` registers representation fields and sequences stimulus → current → network tick. It does not choose winners or routes.

No cognitive meaning is assigned to assemblies yet. Stage 1 proves stable sparse identity and causal propagation, not semantic understanding.
