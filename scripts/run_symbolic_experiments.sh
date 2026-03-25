#!/usr/bin/env bash
# ============================================================================
# Symbolic Binary Specialist — RunPod Training + Evaluation
# ============================================================================
# Runs both Variant A (clean) and Variant B (rich) as separate processes,
# plus random baselines and full analysis battery with cross-modal comparison.
#
# Usage:
#   bash run_symbolic_experiments.sh [--steps 500000] [--physical_battery /path/to/battery.npz]
#
# Expects:
#   - PyTorch with CUDA
#   - bridge/scripts/ in working directory (or BRIDGE_SCRIPTS set)
#   - Physical specialist battery.npz for cross-modal comparison
#
# Estimated time: ~1-2 hours total on RTX 3090/4090 (both variants sequential)
# Estimated cost: <$2 on RunPod
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS="${1:-500000}"
PHYSICAL_BATTERY="${2:-/workspace/bridge/artifacts/battery/binary_baseline_s0/battery.npz}"

# Auto-detect alternate physical battery location
if [ ! -f "$PHYSICAL_BATTERY" ]; then
    ALT="/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz"
    if [ -f "$ALT" ]; then
        PHYSICAL_BATTERY="$ALT"
    else
        echo "WARNING: Physical battery not found. Cross-modal comparison will be skipped."
        PHYSICAL_BATTERY=""
    fi
fi

echo "============================================================"
echo "Symbolic Binary Specialist Experiments"
echo "============================================================"
echo "  Steps:            $STEPS"
echo "  Physical battery: ${PHYSICAL_BATTERY:-NOT FOUND}"
echo "  Device:           auto (will use CUDA if available)"
echo "  Script dir:       $SCRIPT_DIR"
echo ""

# ---- Random baselines ----
echo "============================================================"
echo "Phase 0: Random baselines"
echo "============================================================"
python3 train_symbolic.py --random-baseline --env clean 2>&1 | tee /tmp/random_baseline_clean.log
python3 train_symbolic.py --random-baseline --env rich 2>&1 | tee /tmp/random_baseline_rich.log

# ---- Training Variant A (clean) ----
echo ""
echo "============================================================"
echo "Phase 1: Training Variant A (clean symbolic)"
echo "============================================================"
python3 train_symbolic.py \
    --env clean \
    --steps "$STEPS" \
    --device auto \
    --seed 0 \
    2>&1 | tee /tmp/train_symbolic_A.log

# ---- Training Variant B (rich) ----
echo ""
echo "============================================================"
echo "Phase 2: Training Variant B (rich symbolic with carry dynamics)"
echo "============================================================"
python3 train_symbolic.py \
    --env rich \
    --steps "$STEPS" \
    --device auto \
    --seed 0 \
    2>&1 | tee /tmp/train_symbolic_B.log

# ---- Evaluation ----
CKPT_A="../artifacts/checkpoints/symbolic_clean_s0"
CKPT_B="../artifacts/checkpoints/symbolic_rich_s0"

EVAL_ARGS="--n_episodes 10 --n_cycles 2 --device auto"

if [ -n "$PHYSICAL_BATTERY" ]; then
    EVAL_ARGS="$EVAL_ARGS --physical_battery $PHYSICAL_BATTERY"
fi

echo ""
echo "============================================================"
echo "Phase 3: Full battery — Variant A"
echo "============================================================"
python3 eval_symbolic_battery.py "$CKPT_A" --env clean $EVAL_ARGS \
    2>&1 | tee /tmp/eval_symbolic_A.log

echo ""
echo "============================================================"
echo "Phase 4: Full battery — Variant B"
echo "============================================================"
python3 eval_symbolic_battery.py "$CKPT_B" --env rich $EVAL_ARGS \
    2>&1 | tee /tmp/eval_symbolic_B.log

# ---- Three-way comparison ----
if [ -n "$PHYSICAL_BATTERY" ]; then
    echo ""
    echo "============================================================"
    echo "Phase 5: Three-way comparison (Physical × A × B)"
    echo "============================================================"
    python3 eval_symbolic_battery.py --three-way \
        --ckpt_a "$CKPT_A" \
        --ckpt_b "$CKPT_B" \
        --physical_battery "$PHYSICAL_BATTERY" \
        --device auto \
        2>&1 | tee /tmp/three_way_comparison.log
fi

echo ""
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
echo "Outputs:"
echo "  Variant A: $CKPT_A/"
echo "  Variant B: $CKPT_B/"
echo "  Logs:      /tmp/train_symbolic_*.log"
echo ""
echo "To download results:"
echo "  scp -r runpod:bridge/artifacts/checkpoints/symbolic_* ."
