#!/bin/bash
# Self-contained cloud training script for multi-dimensional counting world v2 on RunPod
# v2: 13 blobs, D=2-5, faster bot, smaller world, probe transfer eval
#
# Usage: cloud_run_multidim.sh <seed> <total_steps> [job_id] [proj_dim]
#
# Expects the anim training package to be unpacked at /workspace/
# Outputs: /workspace/results/ with GHE + probe transfer metrics

set -e

SEED=${1:-0}
TOTAL_STEPS=${2:-200000}
JOB_ID=${3:-"multidim_v2_s${SEED}"}
PROJ_DIM=${4:-128}

echo "=================================================="
echo "Multi-Dimensional Counting World v2 Training"
echo "=================================================="
echo "  Seed:        $SEED"
echo "  Steps:       $TOTAL_STEPS"
echo "  Job ID:      $JOB_ID"
echo "  Proj dim:    $PROJ_DIM"
echo "  Blob range:  3-13 (v2)"
echo "  Dim range:   2-5 (v2)"
echo "  Max steps:   2000 (v2)"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

# --- 1. Install Python dependencies ---
echo "Installing Python dependencies..."
pip install -q \
    ruamel.yaml==0.17.4 \
    gym==0.22.0 \
    einops==0.3.0 \
    tensorboard==2.17.1 \
    scikit-learn \
    ripser \
    scipy \
    2>&1 | tail -1

# --- 2. Set up directory structure ---
mkdir -p /workspace/bridge/artifacts/checkpoints/$JOB_ID
mkdir -p /workspace/bridge/artifacts/status
mkdir -p /workspace/results

# --- 3. Create params YAML ---
PARAMS_FILE=/workspace/bridge/cloud/${JOB_ID}_params.yaml
cat > "$PARAMS_FILE" << EOF
env_config:
  n_blobs_min: 3
  n_blobs_max: 13
  max_steps: 2000
  proj_dim: ${PROJ_DIM}
  fixed_dim: null
  apply_symlog: true
training:
  total_steps: ${TOTAL_STEPS}
  seed: ${SEED}
EOF

# --- 4. Set environment and run training ---
export DREAMER_DIR=/workspace/dreamerv3-torch
export ANIM_BRIDGE=/workspace/bridge
export ANIM_ARTIFACTS=/workspace/bridge/artifacts
export JOB_ID=$JOB_ID
export JOB_PARAMS=$PARAMS_FILE
export PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH"

echo ""
echo "Starting multi-dim v2 training..."
python3 /workspace/bridge/scripts/train_multidim.py

# --- 5. Run GHE analysis per dimensionality ---
CKPT_DIR=/workspace/bridge/artifacts/checkpoints/$JOB_ID

if [ -f "$CKPT_DIR/latest.pt" ]; then
    echo ""
    echo "Running GHE analysis per dimensionality..."
    cd /workspace

    # Evaluate at each training dimensionality (v2: D=2-5)
    for D in 2 3 4 5; do
        echo "  Evaluating D=$D..."
        MULTIDIM_FIXED_DIM="$D" \
        MULTIDIM_PROJ_DIM="$PROJ_DIM" \
        MULTIDIM_BLOB_MIN="13" \
        MULTIDIM_BLOB_MAX="13" \
        PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH" \
            python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
                --episodes_per 5 --fixed_dim $D --proj_dim $PROJ_DIM \
            2>&1 || echo "  WARNING: D=$D GHE eval failed"
    done

    # --- 6. Cross-dimensional probe transfer (PRIMARY metric) ---
    echo ""
    echo "Running cross-dimensional probe transfer evaluation..."
    PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH" \
        python3 /workspace/bridge/scripts/probe_transfer_eval.py \
            --checkpoint-dir "$CKPT_DIR" \
            --episodes 10 --blob-count 13 \
            --dims 2 3 4 5 --train-dim 2 \
            --proj-dim $PROJ_DIM \
        2>&1 || echo "  WARNING: Probe transfer eval failed"

    # --- 7. Collect results ---
    echo ""
    echo "Results:"
    for f in "$CKPT_DIR"/quick_ghe_D*.json; do
        if [ -f "$f" ]; then
            echo "  $(basename $f):"
            cat "$f"
            echo ""
        fi
    done

    if [ -f "$CKPT_DIR/probe_transfer.json" ]; then
        echo "  probe_transfer.json:"
        cat "$CKPT_DIR/probe_transfer.json"
        echo ""
    fi

    # Copy to results dir
    cp "$CKPT_DIR"/quick_ghe_D*.json /workspace/results/ 2>/dev/null || true
    cp "$CKPT_DIR"/probe_transfer.json /workspace/results/ 2>/dev/null || true
else
    echo "ERROR: No checkpoint found at $CKPT_DIR/latest.pt"
    exit 1
fi

echo ""
echo "=================================================="
echo "Multi-dim v2 cloud run complete!"
echo "=================================================="
