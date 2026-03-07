# Report: DreamerV3 Trained on 0-25 Blobs

**Date:** 2026-02-26
**Job ID:** `train_25blobs_v3` (train) / `eval_new_25blobs_sweep` (eval)
**Status:** Complete — all analyses run, visualizations generated
**Dashboard:** http://localhost:8080 (from CCC)

---

## 1. What We Did

Trained DreamerV3 from scratch on the full 0-25 blob range to test whether expanding the count range (from the frozen baseline's 0-8) would produce a model with an operational successor function — i.e., a consistent +1 displacement vector in latent space.

### Pipeline
1. **Training:** 312K steps, 6.3 hours on MPS, 113 episodes, `counting_continuous` preset
2. **Eval sweep:** 80 episodes across 8 blob counts (3/5/8/10/12/15/20/25), 191,503 timesteps
3. **Conversion:** Raw .npy tensors → trajectories.npz (h_t: 191503×512, counts, predictions, etc.)
4. **Successor battery:** 5-analysis measurement suite (RSA, geodesic, homomorphism, anisotropy, topology)
5. **Visualization:** PCA (full 191K pts) + 3 UMAP configs (20K subsample) + inter-cluster distances

---

## 2. Key Results

### Head-to-head: Frozen Baseline (0-8) vs New Model (0-25)

| Metric | Old (0-8) | New (0-25) | Direction |
|--------|-----------|------------|-----------|
| RSA Spearman ρ | 0.951 | **0.990** | better |
| RSA Pearson r | 0.956 | **0.988** | better |
| HE Total | 2.83 | **7.41** | worse |
| Mean Pairwise Cosine | 0.202 | 0.167 | worse |
| Displacement Sim Mean | 0.204 | 0.161 | worse |
| Corrected Disp Mean | 0.385 | 0.179 | worse |

**Verdict (both models):** ORDINAL WITHOUT OPERATIONAL SUCCESSOR

### What this means
- **Ordinal structure is excellent** — the model perfectly orders all 26 count values along a smooth manifold (ρ = 0.990, near ceiling)
- **Successor structure is absent** — the displacement vectors between consecutive counts are inconsistent in both direction and magnitude (HE = 7.41, threshold is < 0.5)
- **More range made it worse** — the 25-blob model has *less* successor consistency than the 8-blob model, not more

---

## 3. The Boundary Compression Pattern

The most striking finding is systematic HE spikes at "round number" transitions:

| Transition | HE | Displacement Norm |
|------------|-----|-------------------|
| 0→1 | **32.5** | 3.43 |
| 4→5 | **11.2** | 2.12 |
| 11→12 | **11.2** | 2.10 |
| 14→15 | **21.1** | 2.76 |
| 19→20 | **37.9** | 3.75 |

Between these spikes, HE is typically 1.5-4.0 — still above threshold but an order of magnitude lower. The pattern:

- **Within a "decade"** (e.g., 5→6→7→8→9): relatively smooth, displacement norms ~1.0, HE ~2-3
- **At decade boundaries** (0→1, 4→5, 14→15, 19→20): massive jumps, displacement norms 2-4x larger, HE 10-40x larger

**Interpretation:** The visual layout of blobs changes qualitatively at these boundaries (new rows, new spatial configurations). The model's latent space has a piecewise-smooth manifold with discontinuities at observation-geometry boundaries. It builds a **lookup table** indexed by visual pattern, not an **algorithm** that iterates +1.

---

## 4. PCA Geometry

- PC1 explains 24.0% of variance, PC2 explains 11.0% — the number line is the dominant axis but not overwhelmingly so
- The manifold curves in a characteristic J-shape: low counts (0-8) spread wide and flat along PC1, then the path bends upward and compresses for high counts (15-25)
- This curvature means PCA sees counting as a nonlinear manifold, not a line — consistent with the piecewise structure

### UMAP findings
- All three configs (default/local/global) show fragmented cluster distributions
- Centroids trace a clean ordinal path in all configs
- The fragmentation suggests that within each count, there's substantial variance from different spatial configurations of blobs — the model encodes position/arrangement, not just count

---

## 5. Artifacts

All artifacts are in `/workspace/bridge/artifacts/`:

```
tools/data/
  trajectories.npz                    -- 191K timesteps, h_t/counts/predictions/etc.
  trajectories_frozen_baseline.npz    -- backup of old 0-8 model data
  successor_battery_results.json      -- full battery output (new model)
  successor_battery_results_frozen_baseline.json  -- battery output (old model)
  umap_embeddings.npz                 -- PCA + 3 UMAP embeddings + centroids

tools/figures/
  brain_map_pca.png                   -- PCA full dataset (191K pts)
  brain_map_default.png               -- UMAP n=15, d=0.1 (20K pts)
  brain_map_local.png                 -- UMAP n=5, d=0.05
  brain_map_global.png                -- UMAP n=30, d=0.3
  inter_cluster_distances.png         -- 4-panel distance comparison
  successor_rsa.png                   -- RSA matrix visualization
  successor_homomorphism.png          -- Per-transition HE bar chart
  successor_anisotropy.png            -- Anisotropy analysis
  successor_geodesic.png              -- Geodesic linearity
  index.html                          -- Interactive dashboard (port 8080)

tensors/eval_new_25blobs_sweep/       -- Raw eval dump tensors (.npy)
checkpoints/train_25blobs_v3/         -- Trained model checkpoint
```

---

## 6. Implications for Next Experiments

The core finding is clear: **scaling the count range does not produce successor structure.** The model gets better at ordinal discrimination but worse at learning a uniform +1 operation. This makes sense — more range means more visual-layout boundaries to memorize, and the reward signal (prediction error) doesn't specifically incentivize transition consistency.

### What would break through

Three experimental directions, in order of expected impact:

**A. Transition-focused training (highest priority)**
Instead of training on static scenes ("here are N blobs, predict N"), train on *transitions*: the agent watches a blob appear (N→N+1) or disappear (N→N-1) and must predict the new count from the old count plus the observed change. This directly incentivizes learning the successor operation as a latent-space displacement.

**B. Curriculum with explicit +1/-1 reward shaping**
Add an auxiliary reward that measures whether the model's internal state shifted by a consistent vector when count changed by 1. This is a soft constraint that nudges the latent space toward homomorphic structure without forcing it.

**C. Symbolic grounding via count tokens**
Replace or augment the continuous prediction head with a discrete token prediction (0/1/.../25). This forces the model to form categorical boundaries that might better align with successor structure, since token 5 is always "one after" token 4 in the vocabulary.

### What probably won't help
- More training steps (the ordinal structure converged; more steps won't fix the geometry)
- Different RSSM sizes (the 512-dim space has plenty of capacity; it's using it for configuration encoding)
- Different UMAP hyperparameters (the fragmentation is real, not a visualization artifact)

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Does the 25-blob model know the number line? | **Yes** — ρ = 0.990, near perfect ordinal structure |
| Does it have a successor function? | **No** — HE = 7.41, displacement vectors are inconsistent |
| Did more range help? | **Mixed** — better ordinal, worse successor |
| Why? | Piecewise manifold with discontinuities at visual-layout boundaries |
| What's the pattern? | "Lookup table" numeracy — recognizes configurations, doesn't iterate +1 |
| What would help? | Training on transitions, not just states |
