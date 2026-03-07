# Training Report: train_grid_displacement_v2 (FINAL)

**Date**: 2026-03-01 07:45 UTC
**Status**: Complete — 305,000 steps, eval done, battery done
**Duration**: ~8.5 hours training + 43 min eval
**Verdict**: Displacement loss achieved its training objective perfectly and destroyed the representation.

## Configuration

- Environment: Grid counting (82-dim obs, blobs physically move to/from 5x5 grid)
- Displacement loss: lambda=0.5, ema_decay=0.99, warmup=50
- Detection: FIXED — uses obs[81] (grid_filled_raw) directly
- Blob range: 3-25, continuous actions, bidirectional episodes, conservation enabled
- 100 training episodes, 304K dataset, 80 eval episodes across 8 blob counts

## Three-Way Comparison

| Metric | Mark-in-place (bidir v1) | Grid baseline (v1) | Grid + displacement (v2) |
|--------|-------------------------|--------------------|-----------------------|
| RSA Spearman | — | **0.990** | 0.921 |
| HE | 7.41 | **1.32** | 18.27 |
| Displacement cos | +0.78 (wrong) | N/A (detection bug) | -0.979 (perfect) |
| Eval return | ~2500 | ~3000 | ~3000 |

## What Happened

### The displacement loss converged beautifully...

Three-phase training trajectory:
- **0-120K**: Reconstruction dominance, cos oscillated 0 to +0.55
- **120K-175K**: Phase transition once model_loss < 3.4, cos dropped to -0.87
- **175K-300K**: Smooth convergence to **cos = -0.979**

The +1 and -1 count transition vectors became nearly perfectly antiparallel in 512-d RSSM space. Loss_bi dropped from 0.95 to 0.17. The auxiliary loss achieved exactly what it was designed to do.

### ...and it destroyed the representation geometry

The grid baseline's natural representation was a **curved manifold** — an arc through 512-d space where each count occupies its own neighborhood with smooth, natural spacing. The displacement loss forced this curve into a straight line by insisting all +1 transitions must be identical vectors.

Per-step HE breakdown shows the damage:

| Transition | HE (v2) | ||d|| | Problem |
|-----------|---------|-------|---------|
| 0→1 | 68.53 | 1.90 | Extreme — pushed far from manifold |
| 19→20 | 55.10 | 1.73 | Boundary effect |
| 24→25 | 235.46 | 3.66 | Only 9 samples, wildly distorted |
| 12→13 | 0.79 | 0.32 | Mid-range was fine |
| 5→6 | 0.89 | 0.29 | Mid-range was fine |

The extremes (0→1, 19→20, 24→25) were catastrophically distorted. The displacement loss forced transitions at the edges of the count range to align with transitions in the middle, but they live in fundamentally different regions of latent space. Forcing alignment crushed the natural curvature.

## Key Insight: Goodhart's Law in 512 Dimensions

The displacement loss optimized exactly the metric it was given (cosine alignment of +1/-1 transition vectors) and achieved near-perfect results (cos = -0.979). But the downstream metric we actually care about (HE — centroid displacement consistency) got 14x worse.

**Achieving a training objective perfectly can destroy the thing you care about.**

The natural grid manifold was already better organized than any linear projection could be. The curvature wasn't noise — it was structure. Non-uniform spacing carried information about the different phenomenology of counting at different scales (counting 3 blobs vs 25 blobs are genuinely different tasks).

## Implicit Warmup Finding (Still Valid)

The phase transition at 120K steps remains an interesting finding independent of the HE result. The reconstruction loss needed to stabilize (model_loss 11→3.4) before the displacement loss could take effect. This is a general principle for auxiliary losses: **they need organic convergence of the primary objective before they can reshape geometry.**

The problem is that reshaping geometry was the wrong goal.

## Displacement Loss — Final Verdict

Three attempts across two environments:

| Run | Lambda | Environment | cos | HE | Outcome |
|-----|--------|-------------|-----|----|---------|
| bidir v1 | 0.1 | Mark-in-place | +0.78 | — | Failed: wrong direction |
| grid v1 | 0.5 | Grid | N/A | 1.32 | Bug: never fired (was baseline) |
| **grid v2** | **0.5** | **Grid** | **-0.979** | **18.27** | **Succeeded at objective, destroyed representation** |

**Conclusion: Stop. No more auxiliary losses.** The environment is the answer. The grid baseline's HE of 1.32 is the real result. The displacement loss is retired.

## What the HE Metric Means

The grid baseline's HE of 1.32 may actually represent an **operational successor on a curved manifold**. The original threshold of 0.5 assumed flat translation geometry. The grid's natural representation found something more nuanced — a curved path that the HE metric penalizes for non-uniformity, but which is the correct geometry for this environment.

Future metric refinement should consider:
- Geodesic HE (measuring along the manifold, not in Euclidean space)
- Scale-aware HE (allowing displacement magnitude to vary with count)
- Curvature-aware thresholds

## Next Steps (No More Auxiliary Losses)

1. **Replication seeds** — Run grid baseline 3x with different seeds to confirm HE=1.32 is real
2. **Variable initial states** — Break always-start-at-zero pattern, force agent to experience +1 from every starting count
3. **Metric refinement** — Consider geodesic or curvature-aware HE variants

The environment carries the structure. The physics is the curriculum. Stop whispering and keep redesigning the world.
