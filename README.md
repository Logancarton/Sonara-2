# Sonara 2

Sonara 2 is a clean rebuild of the useful mechanisms from the earlier Sonara experiments.

The first architectural goal is deliberately narrow: **a signal must be able to enter one neural population, propagate through a sparse graph of weighted/delayed synapses, cause downstream populations to fire, and alter future propagation through local learning — without population-specific routing code in the runtime.**

## Current status

**Built and integrated:**

- Vectorized leaky integrate-and-fire neural populations.
- Sparse directed synapse projections with weights and delays.
- Generic outgoing-projection lookup from firing populations.
- Event-driven delayed current delivery.
- Recursive multi-population cascades across ticks.
- Local causal STDP-like potentiation of existing synapses.
- Behavioral tests for cascade propagation, weak-route suppression, plasticity, isolation, and malformed graph input.

**Not built yet:** semantic representations, memory, prediction, action selection, structural synapse growth/pruning, neuromodulation, or imported whole-brain anatomy. Those are future mechanisms and should not be claimed as cognition yet.

## Signal flow

```text
external stimulus
      ↓
NeuronPopulation
      ↓ spikes
outgoing SynapseProjection lookup
      ↓ weight + delay
scheduled target current
      ↓
NeuronPopulation
      ↓ threshold / refractory dynamics
new spikes
      ↓
recursive propagation
      ↓
local pre→post plasticity
```

The runtime only sequences this flow. It does not decide that A should activate B or that a particular population "means" a concept.

## Run

```powershell
python -m pip install -e .[dev]
python -m examples.first_cascade
pytest -q
```

Expected cascade:

```text
t=  1ms spikes={'A': 8, 'B': 0, 'C': 0, 'D': 0}
t=  2ms spikes={'A': 0, 'B': 8, 'C': 0, 'D': 0}
t=  3ms spikes={'A': 0, 'B': 0, 'C': 0, 'D': 8}
```

A weak A→C route receives current but does not reach firing threshold.
