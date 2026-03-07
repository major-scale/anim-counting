# Planner Update: Grid Counting Training + Eval Status

## What Happened

### train_grid_displacement_v1 — DONE (300K steps, ~9hrs)
- Grid-based 82-dim obs, lambda=0.5, bidirectional
- **Displacement loss never fired.** `disp/loss_bi = 0.0` at every checkpoint.
  - Root cause: the +1/-1 detection still looks for binary mark flag flips in obs[53:77], but the grid env puts normalized slot indices there now. Detection needs to diff `obs[81]` (grid_filled_raw) between consecutive timesteps instead.
  - Consequence: this run is effectively a grid baseline (lambda=0.0)
- World model learned well: vector_loss 6.09 → 0.16, eval return stabilized ~3000
- 101 episodes, no crashes, MPS stable

### train_grid_baseline_v1 — CANCELLED
- Would have produced identical results (both are lambda=0 effectively)
- Killed at ~5K steps, no data lost

### eval_grid_displacement_v1 — RUNNING NOW
- Tensor dump: 8 blob counts × 10 episodes = 80 episodes
- Checkpoint: train_grid_displacement_v1/latest.pt
- Updated eval_dump.py to handle 82-dim grid obs (copied to bridge/scripts/, server.py updated to run from there)
- Should complete in ~30 min

## What We'll Have When Eval Finishes

Tensor arrays (h_t, z_t, counts, positions, predictions) across [3, 5, 8, 10, 12, 15, 20, 25] blobs. Ready for the successor battery:
- Hausdorff Embedding dimension on h_t grouped by count
- Compare against the mark-in-place baseline's HE = 7.41

## Key Question the Battery Answers

**Does the grid's spatial signal alone (blobs physically moving to/from grid positions) produce better successor structure than mark-in-place?**

- If HE improves: environment physics was the real lever. The displacement loss is refinement.
- If HE ≈ 7.41: spatial signal is cosmetic. The displacement loss (once fixed) is essential.

## Next Steps After Battery

1. **Fix displacement detection**: one-line change in dreamer.py — diff `obs[81]` between timesteps instead of summing mark flags
2. **Re-run train_grid_displacement_v1** with working loss
3. **Three-way comparison**: mark-in-place (old) vs grid baseline (this run) vs grid+displacement (re-run)

## Infrastructure Improvements Made

- Bridge conventions documented — no more sync confusion
- eval_dump.py now runs from bridge/scripts/ (CCC-controlled, no read-only issues)
- server.py updated to use bridge/scripts/ for eval_dump
- All future changes follow: edit in bridge → restart server → done

## Files Changed This Session

| File | Change |
|------|--------|
| `packages/training/src/counting-world/blob.ts` | Added grid/anim fields |
| `packages/training/src/counting-world/gridCounting.ts` | New — grid counting logic |
| `packages/training/src/headlessEnv.ts` | Rewritten for grid counting, 82-dim obs |
| `bridge/dist/*.js` | Compiled JS for Mac runtime |
| `bridge/scripts/counting_env.py` | OBS_SIZE=82 |
| `bridge/scripts/envs/counting.py` | Reward shaping uses grid_filled_raw |
| `bridge/scripts/eval_dump.py` | New — updated for 82-dim grid obs |
| `bridge/server.py` | eval_dump runs from bridge/scripts/ |
