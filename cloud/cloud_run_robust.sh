#!/bin/bash
# Self-contained cloud training script for robustness experiments on RunPod
# Usage: cloud_run_robust.sh <seed> <total_steps> <job_id> <varied_arr> <random_start> <random_project>
#
# Expects the anim training package to be unpacked at /workspace/
# Outputs: /workspace/results/summary.json with GHE metrics

set -e

SEED=${1:-0}
TOTAL_STEPS=${2:-200000}
JOB_ID=${3:-"robust_combined_s${SEED}"}
VARIED_ARR=${4:-"false"}
RANDOM_START=${5:-"false"}
RANDOM_PROJECT=${6:-"true"}

echo "=================================================="
echo "Robustness Experiment Cloud Training"
echo "=================================================="
echo "  Seed:           $SEED"
echo "  Steps:          $TOTAL_STEPS"
echo "  Job ID:         $JOB_ID"
echo "  Varied arr:     $VARIED_ARR"
echo "  Random start:   $RANDOM_START"
echo "  Random project: $RANDOM_PROJECT"
echo "  GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
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
mkdir -p /workspace/bridge/queue/pending
mkdir -p /workspace/results

# --- 3. Create params YAML ---
PARAMS_FILE=/workspace/bridge/cloud/${JOB_ID}_params.yaml
cat > "$PARAMS_FILE" << EOF
env_config:
  n_blobs_min: 3
  n_blobs_max: 25
  max_steps: 6000
  arrangement: grid
  random_project: ${RANDOM_PROJECT}
  varied_arrangements: ${VARIED_ARR}
  random_start_count: ${RANDOM_START}
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
echo "Starting robustness training..."
python3 /workspace/bridge/scripts/train.py

# --- 5. Run GHE analysis ---
CKPT_DIR=/workspace/bridge/artifacts/checkpoints/$JOB_ID

if [ -f "$CKPT_DIR/latest.pt" ]; then
    echo ""
    echo "Running GHE analysis..."
    cd /workspace

    RANDPROJ_FLAG=""
    if [ "$RANDOM_PROJECT" = "true" ]; then
        RANDPROJ_FLAG="--random_project"
    fi
    VARIED_FLAG=""
    if [ "$VARIED_ARR" = "true" ]; then
        VARIED_FLAG="--varied_arrangements"
    fi
    RANDSTART_FLAG=""
    if [ "$RANDOM_START" = "true" ]; then
        RANDSTART_FLAG="--random_start_count"
    fi
    PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH" \
        python3 /workspace/bridge/scripts/quick_ghe_robust.py "$CKPT_DIR" \
            --episodes_per 5 $RANDPROJ_FLAG $VARIED_FLAG $RANDSTART_FLAG

    # Copy results
    cp "$CKPT_DIR/quick_ghe.json" /workspace/results/summary.json
    echo ""
    echo "Results saved to /workspace/results/summary.json"
    cat /workspace/results/summary.json
else
    echo "ERROR: No checkpoint found at $CKPT_DIR/latest.pt"
    exit 1
fi

echo ""
echo "=================================================="
echo "Robustness cloud run complete!"
echo "=================================================="
