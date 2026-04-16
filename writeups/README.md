# anima-bridge write-ups

Three tiers of the same research story, written for different audiences.

| File | Audience | Length | Status |
|---|---|---|---|
| [`01-layman.md`](01-layman.md) | Curious non-specialist reader (general blog / popular audience) | ~2,300 words | Draft — ready for read-through |
| [`02-blog.md`](02-blog.md) | ML / cogsci practitioners (technical blog post) | ~7,500 words | Draft — ready for read-through |
| [`03-paper.md`](03-paper.md) | Academic venue (formal paper, targeting NeurIPS/ICML 2026) | ~11,000 words | Draft — needs causal interventions and multi-seed unifier replication before submission |

All three cover the full arc: counting manifold → binary counting machine → FP Unifier → mechanism of integration (accuracy-geometry dissociation, late-onset VICReg, cooperative residual plasticity).

## Fact-check status

An automated audit of numerical claims was run against the raw artifact files in `../artifacts/`:

**Confirmed** against source data:
- Counting manifold 5-seed replication: mean GHE 0.329 ± 0.027, unanimous topology (`replication_summary.json`)
- Ablation cascade GHE values (0.336, 0.344, 0.367) and std (`ablation_multiseed_results.json`)
- Untrained baseline RSA 0.982 → 0.591 and PC1 73% → 23% (`ablation_multiseed_results.json`)
- Intrinsic dimensionality TwoNN 5.5–6.1, MLE 7.6–9.3 (`lid_results.json`)
- Imagination rollout: posterior span 10.02, imagination span 10.31 for 7→8 (`imagination_rollout_colstate.json`)

**Needs verification** (artifact file not found in expected path by the audit — may live elsewhere):
- LSTM/MLP/MLP-nocount architecture table (Table 4 in original paper draft) — numbers sourced from `counting_from_observation.md`, backing file `battery/lstm_mlp/summary.json` not located
- Binary random baseline table (trained 100% vs random 58.7%) — numbers sourced from `UNIFIER.md`, backing file `battery/binary_random_baseline.json` not located
- Unifier contrastive sweep (7 λ conditions) — numbers sourced from `UNIFIER.md`, backing file `sweep_results/results.json` not located

These three claims are load-bearing and should be traced to their actual artifact files before the paper is submitted. The numbers may well be correct; the audit simply didn't find the files at the paths named in the source docs.

## Known gaps for publication readiness

Before any of these are "submission-ready" (especially `03-paper.md`):

1. **Causal intervention on the counting manifold.** Still missing. Correlational evidence only. Othello-GPT standard demands hidden-state editing along the probe direction. This is the single most important missing experiment.
2. **Multi-seed unifier ablations.** Contrastive sweep, VICReg timing, and GRU freeze are single seeds. Directional replication exists for VICReg timing only.
3. **Artifact path verification** for the three claims above.
4. **Polish pass on prose.** Drafts are substantively complete but could benefit from one editorial read-through (sentence flow, heading consistency, caption standardization).

## Source material

All three write-ups are synthesized from:
- `../README.md` — counting manifold (grid world) in repo-README style, 453 lines
- `../UNIFIER.md` — binary world + FP Unifier in blog style, 748 lines
- `../artifacts/paper/counting_from_observation.md` — formal paper draft covering the counting half only, 629 lines

The three new write-ups *extend* this material: the layman version is wholly new, the blog and paper versions unify the counting and binary/unifier halves (which previously lived in separate documents) into single narratives appropriate for each tier.
