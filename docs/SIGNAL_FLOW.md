# Signal Flow

## Current live path

```text
external current
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

Ownership is intentional:

- `NeuronPopulation` owns membrane, threshold, refractory, and spike state.
- `SynapseProjection` owns sparse connectivity, weights, delays, and local synaptic learning.
- `PropagationNetwork` owns graph registration and event delivery.
- `SonaraRuntime` only sequences the organism.

No cognitive meaning is assigned to population IDs yet.
