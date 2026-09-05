# Sonara 2

Sonara 2 is a clean rebuild of useful mechanisms from the earlier Sonara experiments.

The current architectural goal is deliberately narrow: **numeric experience must form a sparse internal neural assembly, that assembly must remain recognizable across repeated/related input, and it must drive the same generic weighted/delayed propagation graph without semantic routing code in runtime.**

## Current status

**Built and integrated:**

- Vectorized leaky integrate-and-fire neural populations.
- Sparse directed synapse projections with weights and delays.
- Generic outgoing-projection lookup from firing populations.
- Event-driven delayed current delivery and recursive cascades.
- Local causal STDP-like potentiation of existing neural-to-neural synapses.
- Deterministic numeric feature encoding into distributed candidate neurons.
- k-winner sparse competition with a fixed activity budget.
- Local competitive Hebbian adaptation of feature-to-neuron weights.
- Live learned assemblies that enter the existing propagation graph as real spikes.
- Behavioral tests for identity overlap, similarity structure, sparsity, learning effect, propagation, isolation, and malformed input.

**Not built yet:** semantic labels, associative memory, prediction, action selection, structural synapse growth/pruning, neuromodulation, or imported whole-brain anatomy. A stable assembly is not yet a concept or memory.

## Live signal flow

```text
numeric stimulus
      ↓
SparseRepresentationField
      ↓ deterministic feature drive + k-winner competition
sparse neural current
      ↓
NeuronPopulation
      ↓ spikes
outgoing SynapseProjection lookup
      ↓ weight + delay
scheduled target current
      ↓
downstream NeuronPopulation
      ↓
recursive propagation + local pre→post plasticity
```

The runtime only sequences typed handoffs. It does not choose winner neuron IDs, decide which downstream population should activate, or assign semantic meaning.

## Run

```powershell
python -m pip install -e .[dev]
python -m examples.first_cascade
python -m examples.learned_assembly
pytest -q
```

Current Stage 1 example (deterministic seed):

```text
sparsity=0.0833
identical_overlap=1.000
similar_overlap=0.875
different_overlap=0.000
mean_winner_drive_before=17.602
mean_winner_drive_after=18.969
t=1ms spikes={'R': 8, 'D': 0}
t=2ms spikes={'R': 0, 'D': 8}
```
