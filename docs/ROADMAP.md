# Roadmap

## Stage 0 — Causal propagation organism

Status: **Integrated**

Gate: one external stimulus causes graph-selected, delayed, recursive neural activity; a weaker competing route fails threshold; causal firing changes synaptic weight.

## Stage 1 — Sparse learned representations

Status: **Integrated, robustness not fully proved**

Gate: numeric stimuli form sparse assemblies without stimulus-ID lookup tables; identical input reproduces the assembly, related input overlaps it more than unrelated input, repeated exposure strengthens future neural drive, and the assembly propagates through the existing live graph.

Current deterministic proof: 8/96 winners (8.33%), identical overlap 1.000, related overlap 0.875, different overlap 0.000, mean winner current 17.602 → 18.969 after six learning exposures, then 8 downstream spikes after the configured 1 ms projection delay.

Broader adversarial testing previously showed that the current representation learner can collapse moderately correlated families, so Stage 1 should not be treated as a robust learned semantic representation system yet.

## Experimental direction — Broad bundle cortical substrate

Status: **Built and experimentally integrated on `experiment-fast-cortical-sheet-final`; not integrated into live runtime.**

The experiment tests a different organism-level primitive: broad sparse `SignalBundle` packets rather than one-neuron trails. A bundle contains many co-active cells and graded amplitudes; every bundle member fans through all of its outgoing recurrent edges simultaneously, and independently propagated bundles can reconverge by summing on shared targets.

Current experimental evidence:

- anchored input location changes the first winning territory;
- nonlinear multi-stream convergence is active;
- ambiguous-state separation improves monotonically as independent streams are added: approximately 0.027 → 0.291 → 0.411 → 0.468;
- hot/input-near versus cold/input-distant persistence remains active;
- live steps emit broad multi-cell bundles;
- broad bundles fan out to many more targets and reconverge additively;
- 10,000 units / 300,000 recurrent edges / 2% active sparsity remains fast in the live probe.

The hippocampal experiment remains incomplete. CA3 recurrence can expand a partial seed, but distinct episode identity and correct cortical reactivation are still failed gates. Do not tune thresholds around those failures or call memory integrated.

## Stage 2 — Associative memory

Future. Retrieval must reactivate learned distributed representations through the same propagation substrate. The current experimental hippocampal branch does not meet this gate.

## Stage 3 — Prediction and outcome learning

Future. Predictions and actual outcomes must produce local/neuromodulated credit signals that change future propagation.

## Stage 4 — Action selection

Future. Competing action populations should win through network dynamics and receive execution feedback.

## Stage 5 — Donor anatomy and modulatory systems

Future. Import only donor anatomy/gating mechanisms that improve the live organism without introducing parallel authority.
