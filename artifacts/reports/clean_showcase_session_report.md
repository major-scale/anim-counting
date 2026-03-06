# Clean Showcase Session Report

**Date:** 2026-03-04/05
**Author:** Claude Code (Builder)
**Audience:** Planner (Gemini), major-scale

## Summary

Retrained two DreamerV3 models with cleaner environment design, deployed to Python GUI visualization, and debugged multiple issues along the way. The visualization now works correctly for both baseline and random-projection models. This was a full-day session with significant debugging.

## What We Built

### 1. Count-at-Placement Environment Change
**Problem:** The old environment incremented `grid_filled_count` when the bot *touched* a blob, but the blob was still flying through the air to the grid. This created ~125 frames of conflicting signal per count transition — the obs said "count=N" while visually only N-1 blobs were on the grid.

**Fix:** Deferred `grid.filled_count` increment until the blob animation completes (lands on its grid slot). Added `pending_grid_placement` flag to the Blob dataclass. Required three sub-fixes:
- Forward path: defer increment, track pending state
- Uncount path: don't decrement `filled_count` for pending blobs (it was never incremented)
- Phase transition: force-finalize all in-flight blobs when counting phase ends (otherwise last blobs never land)

**Validation:** 100% frame consistency — `filled_count` matches physically-landed blobs at every frame. 500/1208 frames had deferred counts (conflicting-signal frames eliminated).

### 2. Zero-Action RSSM Training
**Problem:** The bot follows hardcoded gathering logic — the action channel is pure noise. Training the RSSM on random actions pollutes the world model.

**Fix:** Added `DREAMER_ZERO_ACTION` env var flag to `models.py` shadow. When set, `action_input = torch.zeros_like(data["action"])` in both `_train()` and `video_pred()`. Backward compatible — flag off = original behavior.

### 3. Cloud Training (RunPod)
Trained two models at 200K steps each on RTX 4090 ($0.59/hr):
- `clean_baseline_s0`: grid arrangement, zero action, count-at-placement
- `clean_randproj_s0`: same + 82x82 orthogonal random projection on obs

Both completed successfully (~50 min each). Full battery (GHE, PaCMAP, TriMap, etc.) ran on RunPod before downloading.

### 4. Python GUI Deployment
Deployed both models to the pygame `visualize_counting.py` for Mac.

## Troubles Encountered (Chronological)

### Trouble 1: 82-dim vs 80-dim Model Mismatch
**Symptom:** Export scripts crashed with reshape errors.
**Cause:** New models were trained on 82-dim obs (including `grid_filled_norm` and `grid_filled_raw` at indices 80-81). Old export scripts hardcoded `OBS_SIZE=80`.
**Fix:** Made `OBS_SIZE` dynamic — derived from `enc_linear0_w.shape[1]` at weight-load time. Updated all export scripts and the visualizer.

### Trouble 2: Read-Only dreamerv3-torch Directory
**Symptom:** Couldn't edit model files directly.
**Fix:** Used the existing `bridge/scripts/models.py` shadow (PYTHONPATH priority). All changes go through bridge shadows.

### Trouble 3: User Reported "Predictions Jump to 23"
**Symptom:** User ran the randproj visualization on Mac and saw predictions of ~23-25 when ground truth was 0.
**Initial debugging:** Ran step-by-step RSSM diagnostic locally — predictions started correctly at 0. Rendered headless frames — looked perfect. Couldn't reproduce the issue.
**Red herring:** Investigated probe compression at counts 24-25 (centroid 25 maps to probe value 23.2). Tried nearest-centroid prediction — it was WORSE (47.6% vs 59.9% exact) and predicted count=11 at step 0 because the zero deter vector is closest to the count=11 centroid in Euclidean distance. Reverted immediately.

### Trouble 4: The ACTUAL Bug — Missing Random Projection in Export (ROOT CAUSE)
**Symptom:** Randproj model predictions were complete garbage on Mac.
**Cause:** `export_deter_centroids.py` didn't apply the random projection matrix when collecting deter states for PCA/probe training. The RSSM was trained on projected (scrambled) obs, but the export fed it normal (unscrambled) obs. The resulting PCA centroids and probe weights were trained on nonsense deter states.
**Evidence:**
- Export WITHOUT projection: R²=0.965, Exact=31%, Within ±1=73.7%
- Export WITH projection: R²=0.9996, **Exact=99.0%**, **Within ±1=100%**

**Fix:** Added `--randproj` flag to `export_deter_centroids.py` that applies the same 82x82 orthogonal matrix (seed 42000) to obs before feeding to RSSM during data collection. Also added `--weights-dir` and `--output-dir` flags for flexibility.

### Trouble 5: Orphaned Processes Starving CPU
**Symptom:** Export scripts taking 150+ minutes instead of ~15 minutes.
**Cause:** A killed background agent left its Python subprocess running (40-episode probe retrain at 471% CPU). Three processes competing for CPU, each running at 1/3 speed.
**Fix:** Manually killed the orphaned process. Exports completed normally after that.

## Battery Results: Clean Baseline vs Clean Randproj

| Metric | Clean Baseline | Clean Randproj | Old Baseline (reference) |
|---|---|---|---|
| GHE | **0.394** | 0.411 | 0.329 |
| Arc R² | 0.995 | **0.997** | 0.998 |
| Topology | β₀=1, β₁=0 | β₀=1, β₁=0 | β₀=1, β₁=0 |
| RSA | 0.972 | **0.978** | 0.982 |
| PCA PC1 | 62.7% | **69.2%** | 73% |
| NN Accuracy | 99.99% | **100%** | 96% |
| Linear Probe R² | 0.870 | **0.883** | ~0.98 |
| PaCMAP R² | **0.975** | 0.897 | 0.651 |
| TriMap R² | **0.955** | 0.843 | 0.716 |

### Is Randproj Significantly Better?

**Short answer: No. In the clean environment, randproj no longer provides a clear advantage.**

The old finding was that randproj dramatically improved meso-scale geometry (PaCMAP R² 0.651→0.976, TriMap R² 0.716→0.916). But the clean environment changes (count-at-placement + zero action) achieved a similar improvement for the BASELINE:
- Old baseline PaCMAP: 0.651 → Clean baseline PaCMAP: **0.975** (+50%)
- Old baseline TriMap: 0.716 → Clean baseline TriMap: **0.955** (+33%)

The clean randproj actually has WORSE projection metrics than the clean baseline (PaCMAP 0.897 vs 0.975, TriMap 0.843 vs 0.955). The random projection no longer helps because:
1. Count-at-placement eliminated the conflicting-signal frames that created noisy deter states
2. Zero action removed the noise channel the RSSM was trying to model
3. These two fixes cleaned up the representation enough that the RSSM no longer benefits from having spatial semantics disrupted

**However**, the randproj model does have slightly better ordinal structure (RSA 0.978 vs 0.972) and perfect NN accuracy (100% vs 99.99%). The probe is also better calibrated across the full range.

### Probe Calibration Comparison

| Count | Baseline Probe | Randproj Probe | Ideal |
|---|---|---|---|
| 0 | 0.00 | 0.02 | 0 |
| 10 | 10.03 | 10.00 | 10 |
| 20 | 20.00 | 19.99 | 20 |
| 23 | 22.94 | 22.99 | 23 |
| 24 | 23.42 | **23.87** | 24 |
| 25 | 23.24 | **23.67** | 25 |

Both probes compress counts 24-25, but randproj is less severe (23.87 vs 23.42 for count 24). This is because count 25 has very few training samples (~12-14 per export run) — the episode ends almost immediately after all 25 blobs are gathered.

## Key Insight for Planner

**The clean environment design (count-at-placement + zero action) is the real win, not the random projection.** The old randproj advantage was an artifact of a noisy baseline — once the noise is removed, the baseline achieves equivalent or better geometry.

This reinforces the lesson from the displacement loss experiments: **environment design >> training tricks**. Every time.

## Visualization Status

| Model | Command | Status |
|---|---|---|
| Clean baseline | `python3 visualize_counting.py` | Working (probe lags ±1 at transitions, ~56% exact) |
| Clean randproj | `python3 visualize_counting.py --randproj --models-dir ~/anim-bridge/models/randproj_clean` | Working (99% probe exact, user confirmed "so much better") |

The randproj viz looks dramatically better than the baseline viz because the probe accuracy is 99% vs ~56%. But this is a probe/export quality difference, not a model quality difference — the baseline probe will improve once the 30-episode re-export completes (still running).

## Post-Session Analysis: Linear Decodability Dissociation

Planner identified an apparent contradiction: the battery says baseline is as good or better (PaCMAP R² 0.975 vs 0.897), but live probe accuracy says randproj is dramatically better (91% vs 56% exact). Both are true because they measure different things.

**Empirical analysis of within-count scatter:**

| Count | Model | Total Variance | Probe-Direction Std (counts) | Effective Dim |
|---|---|---|---|---|
| 5 | baseline | 45.1 | **0.87** | 3.6 |
| 5 | randproj | 34.2 | 0.31 | 2.3 |
| 10 | baseline | 57.1 | **0.70** | 4.1 |
| 10 | randproj | 37.0 | 0.27 | 2.7 |
| 15 | baseline | 57.1 | **0.60** | 4.0 |
| 15 | randproj | 41.1 | 0.32 | 2.7 |
| 20 | baseline | 31.9 | **0.89** | 4.3 |
| 20 | randproj | 21.3 | 0.27 | 2.5 |

**Key finding — corrects Planner's isotropic scatter hypothesis:**

The randproj scatter is NOT more isotropic. It's actually MORE anisotropic (lower effective dimensionality, higher condition number). But the scatter dimensions are **orthogonal to the count axis**. The probe-direction std is 0.27-0.32 counts for randproj vs 0.60-0.89 for baseline.

**Mechanism:** Random projection doesn't make scatter isotropic — it makes scatter **factorized**. The RSSM, denied access to spatial coordinates, learns to separate count information from nuisance variation along orthogonal subspaces. The baseline entangles them because spatial features (blob positions, bot location) are correlated with count in ways that bleed into the probe direction.

**Practical implication:** A representation can look equivalent under nonlinear analysis (PaCMAP) while performing dramatically differently under linear readout. For any system that needs to READ the count from the representation (which is what a downstream decision-maker would need), the randproj representation is far superior despite having "worse" manifold metrics.

## Files Changed

- `bridge/scripts/counting_env_pure.py` — count-at-placement implementation
- `bridge/scripts/models.py` — DREAMER_ZERO_ACTION flag
- `bridge/scripts/train.py` — zero_action config support
- `bridge/scripts/export_dreamer_weights.py` — dynamic encoder input shape
- `bridge/scripts/export_deter_centroids.py` — `--randproj`, `--weights-dir`, `--output-dir` flags; projection support in data collection
- `bridge/scripts/export_embed_centroids.py` — OBS_SIZE 80→82
- `bridge/scripts/visualize_counting.py` — dynamic OBS_SIZE, centroid matrix loading
- `bridge/cloud/cloud_run.sh` — zero_action parameter
- `bridge/cloud/launch_clean_showcase.py` — new launch script
- `bridge/models/` — fresh baseline checkpoint + PCA/probe
- `bridge/models/randproj_clean/` — fresh randproj checkpoint + PCA/probe (with projection fix)
