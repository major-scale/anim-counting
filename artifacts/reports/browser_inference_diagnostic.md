# Browser DreamerV3 Inference — Diagnostic Report

## Status: Resolved (2026-03-04)

### Three Issues Found and Fixed

#### 1. Missing Symlog Transform (Root Cause)

**The browser encoder was missing the DreamerV3 symlog preprocessing.**

DreamerV3 applies `symlog(x) = sign(x) * ln(|x| + 1)` to all observation dimensions before the encoder MLP. Confirmed by:
- Config analysis: `configs.yaml` defaults have `symlog_inputs: True`, inherited by `counting_continuous`
- Model verification: `encoder._mlp._symlog_inputs = True` in checkpoint
- Numerical parity: max diff 0.000004 with symlog vs 1.459 without

**Fix**: Added symlog to `dreamerInference.ts` `_encode()` method.

#### 2. Blob Animation Timing Mismatch

`buildObsVector.ts` was reading `blob.position` during animation, but the Python env snaps positions instantly.

When `countBlobToGrid()` fires:
- `grid.filledCount` increments immediately
- `blob.gridSlot` is set
- `blob.position` starts a ~125-frame animation from field to grid

The browser obs at count=N had the Nth blob at an intermediate position, while the Python env always shows it at its final grid position.

**Fix**: `buildObsVector.ts` now uses `blob.animatingTo ?? blob.position` — reports the target position as soon as animation starts.

#### 3. Training Distribution Mismatch

Original probe was trained on mixed blob counts (3-25 blobs per episode), but the browser always uses 25 blobs. This created a slope mismatch where the probe over/under-predicted by ~1 at mid-high counts.

**Fix**: Retrained probe on 25-blob-only data from pure Python env (173K samples, 50 episodes).

### Final Results

| Metric | Before (all bugs) | After (all fixes) |
|--------|-------------------|-------------------|
| Jitter events | Multiple | **0** |
| Exact match | ~31% | **65-70%** |
| Within ±1 | ~60% | **100%** |
| Systematic bias | -1.36 | ~+0.25 |

### Remaining Limitation: Per-Arrangement Variance

Each browser episode has a unique random blob spatial arrangement. The encoder embedding at count=N depends on WHERE the blobs are, not just HOW MANY are on the grid. This creates ±0.5 variance in the probe output:

- When offset < 0.5: probe rounds correctly → exact match
- When offset > 0.5: probe rounds to wrong integer → off by 1

This varies across episodes: some get 68% exact (good arrangement), others 50% (harder arrangement). Always 100% within ±1.

This is intrinsic to the encoder-only approach — the encoder is a pure feedforward function of a single obs frame. The RSSM accumulates temporal context that would disambiguate, but doesn't work for cold-start inference.

### Files Modified

1. **`packages/signal-app/src/counting-world/dreamerInference.ts`** — Added symlog transform
2. **`packages/signal-app/src/counting-world/buildObsVector.ts`** — Use animation target position
3. **`packages/signal-app/src/counting-world/manifoldProjection.ts`** — Added 512-dim centroid support
4. **`packages/signal-app/public/models/embed_pca.json`** — Recomputed with 25-blob centroids + 512-dim centroids
5. **`packages/signal-app/public/models/embed_probe.json`** — Retrained on 25-blob data, R²=0.9993

### Technical Details

- **Probe**: Ridge(alpha=10) on 173K samples, 25-blob-only, symlog encoder, R²=0.9993, 98% exact on training data
- **512-dim centroids**: Mean embeddings per count (0-25), for nearest-centroid prediction and PCA visualization
- **PCA**: PC1=87.1%, PC2=9.5% on centroids (total 96.6%)
- **Training script**: `bridge/scripts/export_embed_centroids.py`

### Architecture Summary

```
obs[80] → symlog → Linear(512)→LN→SiLU × 3 → embed[512]
                                                    ↓
                                              probe: dot(w, embed) + b → count_pred
                                              pca: (embed - mean) · PCs → [x, y] 2D
```

### Checkpoint Info
- **Checkpoint**: `train_25blobs_v3/latest.pt`
- **Config**: `counting_continuous` (inherits `symlog_inputs: True` from defaults)
- **Encoder input**: 80-dim (first 80 of 82-dim obs vector)
- **The exported weights (`dreamer_weights.bin`) do NOT include symlog** — applied to input before first linear layer
