# RunPod Cloud Validation — Status Report

## Summary

First cloud training run completed successfully on RunPod Secure Cloud. Training was 10-12x faster than Mac MPS. GHE analysis is the bottleneck — it's environment-bound, not GPU-bound, and takes the same ~45-70 minutes regardless of hardware.

## Cloud Setup

| Parameter | Value |
|-----------|-------|
| Provider | RunPod Secure Cloud (SOC2 Type 2) |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Cost | $0.34/hr |
| Image | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Container disk | 50 GB |
| Volume | 20 GB |
| SSH access | Yes (ed25519 key auth) |

## Training Performance

| Metric | Mac MPS (M-series) | RunPod RTX 4090 |
|--------|-------------------|-----------------|
| Steps/minute | ~300-400 | ~4,300 |
| FPS (reported) | ~5 | ~48 |
| 50K steps wall time | ~2.5 hours | **21 minutes** |
| 200K steps (projected) | ~6-7 hours | **~50 minutes** |
| Speedup | 1x | **10-12x** |

The validation run (seed 5, 50K steps, bidirectional, 3-25 blobs) completed in **1,258 seconds (21 minutes)**. Model loss converged to 2.6 by step 55K. The auto-device detection correctly selected `cuda:0`.

## GHE Analysis Bottleneck

**Problem:** The `quick_ghe.py` evaluation script runs 40 episodes (5 per blob count × 8 blob counts) through the counting environment. Each episode step requires:

1. Python → Node.js bridge (JSON over stdin/stdout subprocess)
2. Node.js headless environment physics tick
3. Node.js → Python response
4. DreamerV3 forward pass (trivially fast on GPU)

The bottleneck is the Node.js bridge subprocess round-trip, not the GPU inference. This means GHE analysis takes **45-70 minutes regardless of GPU speed** — the same on a $0.34/hr RTX 4090 as on the container CPU.

**Cost implication:** For a 200K step training run:
- Training: ~50 min ($0.28)
- GHE eval: ~60 min ($0.34)
- **Total: ~110 min ($0.62 per seed)**

The GHE eval is now *more expensive than training*. For the ablation program (25 conditions × 5 seeds = 125 runs), GHE analysis alone would cost ~$42 at current timing.

## Optimization Options for GHE Eval

Three potential approaches to cut GHE eval time:

### Option A: Reduce episodes (quick win)
Currently 5 episodes × 8 blob counts = 40 episodes. Could reduce to 3 episodes × 8 counts = 24 episodes. Would cut eval time by ~40%. Minimal statistical impact — centroids from 3 episodes vs 5 are nearly identical.

### Option B: Batch inference (medium effort)
Currently runs episodes sequentially. Could run all 8 blob-count environments in parallel Node.js processes, batch their observations through the model in a single forward pass. Would require rewriting `run_eval()` but could achieve ~4-6x speedup.

### Option C: Pure Python environment (high effort, high reward)
Rewrite the headless counting environment in Python, eliminating the Node.js bridge entirely. The physics is simple (2D steering, grid slot assignment, collision detection). Would reduce per-step latency from ~10ms (subprocess IPC) to ~0.1ms (pure Python). GHE eval would drop from 60 minutes to ~5 minutes.

**Recommendation for ablation program:** Implement Option A immediately (trivial change), consider Option C before launching the full 125-run program.

## Pipeline Packaging

Created a self-contained cloud training package:

```
/workspace/bridge/cloud/
├── cloud_run.sh                    # End-to-end: install deps → train → GHE → output JSON
├── anim-training-package.tar.gz   # 5.5 MB (DreamerV3 + bridge scripts + compiled bridge.js)
└── active_pod.txt                  # Current pod ID tracking
```

The `cloud_run.sh` script handles everything: Node.js installation, Python deps, directory setup, training with auto-detected CUDA device, and GHE analysis. A single command runs the full pipeline:

```bash
bash cloud_run.sh <seed> <total_steps> [job_id]
```

## Code Changes

- **`train.py`**: Added `_detect_device()` for auto-detection (cuda > mps > cpu). `DREAMER_DIR` now reads from env var with fallback to `~/dreamerv3-torch`.
- **`cloud_run.sh`**: New self-contained launcher. Writes flat params YAML (not the server.py wrapper format).

## Cost Projections

| Scenario | Runs | Time/run | Cost/run | Total |
|----------|------|----------|----------|-------|
| Seeds 3+4 (parallel) | 2 | ~110 min | $0.62 | **$1.24** |
| Full ablation (25 conditions × 5 seeds) | 125 | ~110 min | $0.62 | **$77.50** |
| Full ablation with Option A (3 eps) | 125 | ~80 min | $0.45 | **$56.25** |
| Full ablation with Option C (Python env) | 125 | ~55 min | $0.31 | **$38.75** |

With $38 in credits: Options A or C make the full program feasible. Without optimization, we'd need ~$78 — over budget. Parallelizing across 5 pods doesn't change total cost, just wall-clock time.

**Recommendation:** Run seeds 3+4 now ($1.24). Implement Option A before the ablation program. Evaluate whether $38 covers the priority conditions (maybe 15 conditions × 5 seeds = 75 runs at $0.45 = $34).

## Validation Result

Awaiting GHE analysis completion. Expected: GHE ~0.35-0.45 at 50K steps (consistent with seed 1's 0.399 at 51K). If it matches, the cloud pipeline is validated and seeds 3+4 can launch immediately.

## Current Status

- **Cloud pod**: Running, GHE eval in progress (~40 min in, ~20-30 min remaining)
- **Mac local**: Seed 2 training at ~5K/200K steps
- **Pod runtime**: ~75 min → ~$0.43 so far
