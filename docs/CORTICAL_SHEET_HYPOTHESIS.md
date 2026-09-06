# Fast cortical-sheet hypothesis experiment

Status: **experimental; not production Sonara cognition**.

This experiment tests a specific architectural hypothesis:

> Functional meaning can emerge partly from where independent signal streams enter a generic neural substrate, how they diverge and converge, recurrent state, physical connection costs, temporal integration gradients, and experience-dependent plasticity. Intermediate coordinates are not assigned semantic functions.

## Mechanisms intentionally represented

```text
anchored external/internal streams
        ↓
separate dendritic-like compartments
        ↓
nonlinear coincidence / mixed selectivity
        ↓
hot → cold temporal integration gradient
        ↓
sparse winner competition
        ↓
local-biased + sparse long-range divergence
        ↓
recurrent activity
        ↺
local afferent + recurrent Hebbian adaptation
```

The experiment uses vectorized NumPy math. A runtime step has no Python loop over neurons. Recurrent propagation uses edge-array gather plus `numpy.bincount`, allowing large sparse graphs without a dense N×N recurrent matrix.

## What the current probe can establish

- Input location can move the first functional response territory because intermediate coordinates have no assigned semantic identity.
- Multiple independent streams can converge onto the same generic sheet.
- Nonlinear branch coincidence creates mixed-selective drive.
- Increasing independent convergent evidence can improve discrimination when sensory evidence alone is intentionally ambiguous.
- Units farther in connectional/spatial distance from anchored inputs can be assigned progressively longer integration constants, creating a measurable hot→cold persistence gradient.
- Sparse local and long-range recurrent edges can carry activity away from its entry location.

## What is not yet proved

The first recurrent Hebbian rule does **not** reliably implement hippocampal-like pattern completion. It can maintain/reinforce activity, but a partial cue does not yet consistently recover the correct previously learned full experience. The test suite keeps this as a strict expected failure so it cannot be silently reported as solved.

The experiment therefore distinguishes:

```text
multi-stream convergence / separation      testable now
recurrent maintenance                       present
correct pattern completion                  not yet proved
semantic cognition                           not claimed
```

## Why this is separate from production

The experiment should not become Sonara's production substrate merely because its construction tests pass. It should earn integration by demonstrating behavior that the simpler Stage 1 substrate cannot produce, especially robust discrimination under ambiguity and correct recurrent recovery from degraded cues.

## Interpretation discipline

The convergence benchmark intentionally runs with learning disabled. Its claim is narrower than "Sonara learned meaning": it tests whether the proposed hardware-flow geometry makes independent evidence computationally useful. A monotonic separation gain therefore supports the convergence hypothesis, not plasticity or semantic learning.

Plasticity remains present in the sheet for later experiments, but the current completion diagnostic shows that the simple recurrent Hebbian rule is insufficient: recurrence can reinforce state without recovering the correct prior experience. That failure must be solved before any hippocampal-like claim is made.
