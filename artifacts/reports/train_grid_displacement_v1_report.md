# Training Report: train_grid_displacement_v1

## Run Config
- **Environment**: Grid-based counting (82-dim obs vector)
- **Displacement loss**: lambda=0.5 (BUT never fired — see below)
- **Blobs**: 3-25, bidirectional episodes
- **Steps**: 300K (completed 313K)
- **Duration**: ~9 hours on MPS
- **Episodes**: 101 total

## Key Finding: Displacement Loss Never Fired

`disp/loss_bi = 0.0` at every checkpoint. `disp/n_plus = 0`, `disp/n_minus = 0` throughout.

This means the +1/-1 event detection in the displacement loss code isn't recognizing grid counting events. The old detection likely looked for mark flag flips (0→1 or 1→0 in obs slots 53-77), but the grid env now puts **normalized slot indices** there instead of binary marks. The detection code needs updating to recognize grid slot assignment changes.

**Consequence**: This run is effectively a grid baseline with lambda=0.0. It and `train_grid_baseline_v1` will produce near-identical results.

## World Model Performance

| Metric | 5K | 50K | 100K | 200K | 300K |
|--------|-----|------|-------|-------|-------|
| vector_loss | 6.09 | 0.62 | 0.33 | 0.21 | 0.16 |
| reward_loss | 3.86 | 1.31 | 0.84 | 0.70 | 0.64 |
| model_loss | 11.19 | 2.72 | 2.09 | 1.87 | 1.76 |

Vector loss converged well: 6.09 → 0.16 (38x reduction). The world model is learning the 82-dim observation space effectively. The dynamic blob positions (sliding to/from grid) don't seem to cause reconstruction difficulty.

## Eval Returns

| Step | Eval Return | Eval Length |
|------|------------|-------------|
| 5K | -4521 | 3344 |
| 25K | 1061 | 3427 |
| 50K | 2601 | 3179 |
| 100K | 3140 | 3548 |
| 200K | 3402 | 3778 |
| 300K | 3526 | 3849 |

Returns stabilize around 2800-3500 after 50K steps. Variance is high (2157-4030 range) due to variable blob counts per episode (3-25 blobs = very different episode lengths).

## Comparison with Prior Mark-in-Place Run

Previous run (`train_25blobs_displacement_bidir_v1`, mark-in-place, lambda=0.1):
- That run's displacement loss DID fire but converged wrong (cos=0.78, wanted -1.0)
- Direct return comparison is not meaningful because the obs format and reward shaping changed

## What This Tells Us

1. **Grid env is trainable** — DreamerV3 learns the 82-dim obs with dynamic positions
2. **Displacement loss detection is broken for grid obs** — needs fix before lambda>0 adds value
3. **Baseline comparison will be uninformative** — both runs are effectively lambda=0
4. **To get the three-way comparison we wanted**, we need to fix the +1/-1 detection to use `obs[81]` (grid_filled_raw) changes instead of mark flag flips

## Next Steps

1. Fix displacement loss +1/-1 detection in `dreamer.py` to detect `grid_filled_raw` deltas
2. Re-run `train_grid_displacement_v1` with working loss
3. Then the three-way comparison: old mark-in-place vs grid baseline vs grid+displacement
4. Consider: the grid env's natural spatial signal may already encode successor structure through position changes alone — the HE battery on this checkpoint will tell us
