# Donor Map

The old Sonara material is donor evidence, not the receiving architecture.

| Donor mechanism | Decision | Sonara 2 receiving owner |
| --- | --- | --- |
| Vectorized LIF membrane / threshold / refractory state from `core/sonara_state.py` | MINE + REWRITE | `sonara/neural/population.py` |
| Old runtime synthetic sinusoidal activity selection | DELETE | none |
| Old fixed semantic/anatomical activation assumptions | DELETE | none; Stage 1 uses numeric distributed encoding |
| Tier-3 gate concepts | DEFER / MINE LATER | future state-dependent projection modulation |
| Anatomy tracts + channels | DEFER / MINE LATER | future graph loader into `PropagationNetwork` |
| Consolidation/replay concepts | DEFER | future memory owner |
| Emotion/NT constants | DEFER; treat as hypotheses, not truth | future modulatory owner |
| Runtime bundle generator | KEEP AS TOOLING CANDIDATE | future `tools/` |

Stage 0 mined the useful neuron-state idea and rebuilt the missing causal propagation bridge. Stage 1 is a clean receiving-side mechanism: deterministic numeric encoding, k-winner sparsity, and local competitive Hebbian adaptation. It does not copy donor semantic mappings or anatomy.
