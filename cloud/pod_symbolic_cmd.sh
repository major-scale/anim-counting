#!/bin/bash
# Auto-generated: symbolic binary specialist experiments
# Run this on RunPod after uploading training package
set -e

echo "============================================"
echo "Symbolic Binary Specialist Experiments"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# --- 1. Install dependencies ---
echo "Installing dependencies..."
pip install -q \
    ruamel.yaml==0.17.4 \
    gym==0.22.0 \
    scikit-learn \
    scipy \
    matplotlib \
    2>&1 | tail -1

# Optional: ripser for topology
pip install -q ripser 2>/dev/null || echo "ripser not available, topology metrics will be skipped"

# --- 2. Unpack training package ---
cd /workspace
mkdir -p bridge
tar xzf /workspace/anim-symbolic-package.tar.gz -C /workspace/bridge/

cd /workspace/bridge/scripts

# --- 3. Random baselines ---
echo ""
echo "=== Random Baselines ==="
python3 train_symbolic.py --random-baseline --env clean
python3 train_symbolic.py --random-baseline --env rich

# --- 4. Train Variant A (clean) ---
echo ""
echo "=== Training Variant A (clean, 500000 steps) ==="
python3 train_symbolic.py \
    --env clean \
    --steps 500000 \
    --device auto \
    --seed 0

# --- 5. Train Variant B (rich) ---
echo ""
echo "=== Training Variant B (rich, 500000 steps) ==="
python3 train_symbolic.py \
    --env rich \
    --steps 500000 \
    --device auto \
    --seed 0

# --- 6. Full battery + cross-modal ---
CKPT_A="../artifacts/checkpoints/symbolic_clean_s0"
CKPT_B="../artifacts/checkpoints/symbolic_rich_s0"
PHYS="../artifacts/battery/binary_baseline_s0/battery.npz"

EVAL_ARGS="--n_episodes 10 --n_cycles 2 --device auto"
if [ -f "$PHYS" ]; then
    EVAL_ARGS="$EVAL_ARGS --physical_battery $PHYS"
fi

echo ""
echo "=== Full Battery: Variant A ==="
python3 eval_symbolic_battery.py "$CKPT_A" --env clean $EVAL_ARGS

echo ""
echo "=== Full Battery: Variant B ==="
python3 eval_symbolic_battery.py "$CKPT_B" --env rich $EVAL_ARGS

# --- 7. Three-way comparison ---
if [ -f "$PHYS" ]; then
    echo ""
    echo "=== Three-Way Comparison ==="
    python3 eval_symbolic_battery.py --three-way \
        --ckpt_a "$CKPT_A" \
        --ckpt_b "$CKPT_B" \
        --physical_battery "$PHYS" \
        --device auto
fi

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Results:"
ls -la "$CKPT_A"/battery_results.json "$CKPT_B"/battery_results.json 2>/dev/null
ls -la "$CKPT_A"/cross_modal.json "$CKPT_B"/cross_modal.json 2>/dev/null
echo ""
echo "Download:"
echo "  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_clean_s0 ."
echo "  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_rich_s0 ."
