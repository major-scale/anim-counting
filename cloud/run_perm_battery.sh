#!/bin/bash
# Run full measurement battery on all permutation checkpoints
# Execute after all 3 training seeds complete

set -e

export PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH"
export COUNTING_RANDOM_PERMUTE=true

for SEED in 0 7 8; do
    JOB_ID="ablation_randperm_s${SEED}"
    CKPT_DIR="/workspace/bridge/artifacts/checkpoints/${JOB_ID}"
    OUTPUT_DIR="/workspace/results/battery_${JOB_ID}"

    if [ ! -f "$CKPT_DIR/latest.pt" ]; then
        echo "SKIP: No checkpoint at $CKPT_DIR/latest.pt"
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Full battery: ${JOB_ID}"
    echo "=================================================="

    python3 /workspace/bridge/scripts/full_battery.py \
        "$CKPT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --episodes_per 5 \
        --label "randperm_s${SEED}"

    echo "Results: $OUTPUT_DIR"
    cat "${OUTPUT_DIR}/randperm_s${SEED}_metrics.json" 2>/dev/null || echo "No metrics file"
done

echo ""
echo "=================================================="
echo "ALL BATTERIES COMPLETE"
echo "=================================================="
echo ""

# Print PaCMAP R² comparison
echo "=== PaCMAP R² Results ==="
for SEED in 0 7 8; do
    JOB_ID="ablation_randperm_s${SEED}"
    OUTPUT_DIR="/workspace/results/battery_${JOB_ID}"
    python3 -c "
import json
with open('${OUTPUT_DIR}/randperm_s${SEED}_metrics.json') as f:
    m = json.load(f)
print(f'Seed ${SEED}: GHE={m[\"ghe\"]:.3f}  PaCMAP_R2={m[\"proj_r2_pacmap\"]:.3f}  TriMap_R2={m[\"proj_r2_trimap\"]:.3f}  NN_acc={m[\"nn_accuracy\"]:.3f}')
" 2>/dev/null || echo "Seed ${SEED}: metrics not available"
done
