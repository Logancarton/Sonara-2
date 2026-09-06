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

No cognitive meaning is assigned to assemblies yet. Stage 1 proves sparse identity and causal propagation, not semantic understanding.

## Experimental broad-bundle substrate

Status: **Built and experimentally integrated on `experiment-fast-cortical-sheet-final`; not integrated into the live Sonara runtime.**

The cortical-sheet experiment now treats one neural moment as a broad sparse population packet rather than a selected single-neuron trail:

```text
anchored input streams
        ↓
nonlinear multi-stream convergence
        ↓
local sparse competition
        ↓
SignalBundle
[many co-active cells + graded amplitudes]
        ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
ALL outgoing recurrent edges at once
   ↙      ↓       ↓      ↘
local   local   long     long
bundle  bundle  range    range
   ↘      ↓       ↓      ↙
       target currents
            ↓
several bundles can overlap
and SUM on shared targets
            ↓
next broad sparse population packet
            ↺
```

`FastCorticalSheet.activity` remains a dense numeric backing vector for speed. `SignalBundle` is the sparse externally visible packet for that same activity. `project_bundle()` fans every active bundle member through every registered outgoing recurrent edge simultaneously; no one-cell route is selected. `SignalBundle.merge()` sums separately propagated packets on shared targets, making convergence explicit.

Current branch tests prove:

- a live sheet step emits a bundle containing multiple simultaneously active cells;
- that bundle fans out to substantially more downstream targets than its source width;
- two separate projected bundles share downstream targets and their amplitudes add at those intersections;
- the 10,000-unit / 300,000-edge scale probe remains fast after the bundle abstraction;
- the prior 1 → 2 → 3 → 4 independent-stream separation result remains unchanged.

The experimental hippocampal path is still unresolved. CA3 recurrence can expand partial activity, but distinct episode identity and correct cortical reactivation remain failed gates. Those failures must not be interpreted as failures of broad bundle propagation itself.
