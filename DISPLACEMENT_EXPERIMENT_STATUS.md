# Contrastive Displacement Loss Experiment — Status Report

**Date:** 2026-02-26
**Goal:** Add auxiliary displacement loss to DreamerV3 to pressure RSSM into encoding +1/-1 transitions as consistent displacement vectors in latent space. Baseline achieves RSA rho=0.990 but HE=7.41 (no operational successor). This experiment tests whether geometric pressure produces successor structure.

## What Was Built

### Strategy: PYTHONPATH Override
- `/workspace/dreamerv3-torch/` is read-only
- Copied `models.py` and `dreamer.py` to `/workspace/bridge/scripts/`
- `train.py` sets `PYTHONPATH` so Python imports our modified versions first
- Everything else (networks.py, tools.py, envs/) loads from original location unchanged

### Files Modified

1. **`/workspace/bridge/scripts/models.py`** — copied + patched
   - Added `DisplacementLoss` class (nn.Module) after `RewardEMA`
   - Extracts mark counts from obs indices [53:78], computes +1/-1 transitions
   - EMA tracking of D_plus/D_minus displacement directions (momentum 0.99)
   - L_plus/L_minus: cosine alignment to detached EMA targets
   - L_bi: bidirectional repulsion using gradient-carrying batch means
   - Linear warmup ramp over configurable batches
   - Episode boundary masking via is_first
   - Integrated into `WorldModel.__init__` and `WorldModel._train()`
   - 7 logged metrics: disp/{n_plus, n_minus, loss_plus, loss_minus, loss_bi, d_plus_d_minus_cos, warmup_scale}

2. **`/workspace/bridge/scripts/dreamer.py`** — copied + patched
   - Fixed configs.yaml path resolution (local dir -> DREAMER_DIR env -> ~/dreamerv3-torch)
   - Added 3 config defaults: disp_lambda=0.0, disp_ema_decay=0.99, disp_warmup=50
   - strict=False on checkpoint load for backward compatibility

3. **`/workspace/bridge/scripts/train.py`** — modified in place
   - Added BRIDGE_SCRIPTS path constant
   - Points dreamer_script at local patched copy
   - Sets PYTHONPATH to shadow originals
   - Sets DREAMER_DIR env var for configs.yaml discovery
   - Passes displacement_loss params from job YAML to CLI args

## Experiment Status

### Smoke Test: PASSED
- Job: `train_displacement_smoke` (5K steps)
- All 7 disp/ metrics logged correctly
- disp/n_plus=6.2 (transitions detected), disp/loss_plus=1.0, warmup_scale=0.8
- No NaN/Inf, clean exit code 0
- Confirmed PYTHONPATH shadowing works (Mac loaded our models.py)

### Full Training Run: IN PROGRESS
- Job: `train_25blobs_displacement_v1` (300K steps)
- Config: disp_lambda=0.1, ema_decay=0.99, warmup=50
- Status: Running on GPU Mac, currently in prefill phase
- Expected duration: ~6 hours
- Queue file: `/workspace/bridge/queue/running/train_25blobs_displacement_v1.yaml`

### Known Issue: Sync Lag
- Files on Mac aren't syncing back to this container in real-time
- Progress JSON and live log may be stale
- Check directly on Mac: `tail ~/anim-bridge/artifacts/logs/train_25blobs_displacement_v1_live.log`

## Next Steps After Training Completes

1. **Verify completion:** Check `queue/done/` for the job file
2. **Check disp/ metrics in final log:** loss_plus should decrease over training, d_plus_d_minus_cos should trend toward -1
3. **Submit eval sweep:** Same as baseline eval but pointing at new checkpoint
4. **Run successor battery:** Compare against baseline

### Expected Comparison Table

| Metric | Baseline (train_25blobs_v3) | Treatment (displacement_v1) |
|--------|---------------------------|---------------------------|
| RSA rho | 0.990 | ? (should stay >0.95) |
| HE | 7.41 | ? (hoping <3.0) |
| Mean disp cosine | 0.161 | ? (hoping >0.5) |

## Key Design Decisions
- Gradients flow through post["deter"] -> GRU weights (shapes hidden state geometry)
- EMA target is detached (prevents mode collapse)
- Warmup 50 batches (lets RSSM form basic representations first)
- Lambda 0.1 (conservative — main loss ~5-15, displacement ~0.1-0.4, ~5% contribution)
- Bidirectional term uses batch means (not EMA) for gradient-carrying repulsion

## Warning Signs During Training
- RSA rho drops below 0.9 -> lambda too high
- HE doesn't budge -> lambda too low or RSSM can't maintain consistent geometry
- Loss explodes/NaN -> numerical issue
- disp/n_plus consistently 0 -> mark detection broken
