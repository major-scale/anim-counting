# Training Report: train_grid_displacement_v2 (In Progress)

**Date**: 2026-03-01 02:30 UTC
**Status**: Running — 165,000 / 300,000 steps (55%)
**ETA**: ~4 hours remaining

## Configuration

- Environment: Grid counting (82-dim obs, blobs physically move to/from 5x5 grid)
- Displacement loss: lambda=0.5, ema_decay=0.99, warmup=50
- Detection: FIXED — uses obs[81] (grid_filled_raw) directly instead of summing mark flags
- Blob range: 3-25, continuous actions, bidirectional episodes, conservation enabled

## Key Finding: Displacement Loss Phase Transition

The displacement cosine similarity (d_plus_d_minus_cos) showed a clear phase transition after 130K steps:

| Step | d_plus_d_minus_cos | loss_bi | model_loss | Interpretation |
|------|-------------------|---------|------------|----------------|
| 5K   | -0.008 | 0.949 | 11.13 | Random (clean slate) |
| 25K  | +0.553 | 0.969 | 5.43  | Wrong direction (reconstruction dominates) |
| 45K  | +0.035 | 0.973 | 4.38  | Correction |
| 85K  | +0.501 | 0.999 | 3.75  | Back to wrong direction |
| 105K | +0.401 | 0.991 | 3.57  | Slow decline |
| 135K | -0.066 | 0.944 | 3.38  | Crosses zero |
| 145K | -0.353 | 0.821 | 3.28  | Accelerating negative |
| **155K** | **-0.615** | **0.638** | **3.07** | **Strong separation emerging** |

### Interpretation

For the first 120K steps, the cosine oscillated between 0 and +0.55 — the same failure mode as v1 (mark-in-place), where reconstruction loss pushed +1 and -1 representations together. The critical difference: **in the grid environment, it reversed**.

The planner's hypothesis appears correct: in the grid env, reconstruction loss and displacement loss are not adversarial. The reconstruction loss needed ~120K steps to learn the grid spatial structure well enough (model_loss dropped from 11 to 3.4) that it stopped fighting the displacement signal. Once reconstruction "got out of the way," the displacement loss rapidly organized the vectors.

The trajectory from 135K to 155K is steep: -0.066 → -0.353 → -0.615 in 20K steps. If this rate continues, we could see cos < -0.9 by 200K.

This would be the **first time displacement loss has successfully separated +1/-1 count transitions** in any training run.

## Comparison to Prior Runs

| Run | Environment | Lambda | Final cos | HE |
|-----|-------------|--------|-----------|-----|
| train_25blobs_displacement_bidir_v1 | Mark-in-place | 0.1 | +0.78 (wrong) | ~7.41* |
| train_grid_displacement_v1 | Grid | 0.5 | never fired (detection bug) | 1.32 |
| **train_grid_displacement_v2** | **Grid** | **0.5** | **-0.615 at 155K (trending)** | **TBD** |

*HE 7.41 is from grid baseline with lambda=0, not from the mark-in-place displacement run directly.

## Task Performance

Eval returns stabilized around 2500-3500 by step 65K. The agent counts effectively. Bidirectional episodes are working (n+ and n- balanced at ~4.1 per batch).

## Remaining Questions

1. **Does cos reaching -1.0 translate to lower HE?** The grid baseline (lambda=0) got HE=1.32 without any displacement pressure. If v2 reaches cos=-0.9+ and HE drops below 1.0, it proves displacement loss adds value on top of good environment design.

2. **Is the phase transition reproducible?** Variable: could be noise. Need replication seeds (planned next).

3. **Convergence speed**: At current rate, cos could reach -0.9 by ~200K. But it may plateau — cosine losses often have diminishing returns as vectors become more aligned.

## Next Steps (After Completion)

1. Run eval dump (80 episodes, 8 blob counts)
2. Run successor battery → get HE number
3. Three-way comparison: mark-in-place (7.41) vs grid baseline (1.32) vs grid+displacement (v2)
4. If HE < 1.0: displacement loss is validated as additive to environment design
5. If HE ~ 1.3: environment alone is sufficient, pivot to variable starts + replication seeds
