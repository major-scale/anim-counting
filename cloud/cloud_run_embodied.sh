#!/bin/bash
# Self-contained cloud training script for embodied counting world on RunPod
# Usage: cloud_run_embodied.sh <seed> <total_steps> [job_id] [arrangement] [random_project] [random_permute]
#
# Expects the anim training package to be unpacked at /workspace/
# Outputs: /workspace/results/summary.json with GHE metrics

set -e

SEED=${1:-0}
TOTAL_STEPS=${2:-200000}
JOB_ID=${3:-"embodied_baseline_s${SEED}"}
ARRANGEMENT=${4:-"grid"}
RANDOM_PROJECT=${5:-"false"}
RANDOM_PERMUTE=${6:-"false"}

echo "=================================================="
echo "Embodied Agent Cloud Training"
echo "=================================================="
echo "  Seed:        $SEED"
echo "  Steps:       $TOTAL_STEPS"
echo "  Job ID:      $JOB_ID"
echo "  Arrangement: $ARRANGEMENT"
echo "  Random proj: $RANDOM_PROJECT"
echo "  Random perm: $RANDOM_PERMUTE"
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
mkdir -p /workspace/bridge/queue/pending
mkdir -p /workspace/results

# --- 3. Create params YAML ---
PARAMS_FILE=/workspace/bridge/cloud/${JOB_ID}_params.yaml
cat > "$PARAMS_FILE" << EOF
env_config:
  n_blobs_min: 3
  n_blobs_max: 25
  max_steps: 8000
  arrangement: ${ARRANGEMENT}
  random_project: ${RANDOM_PROJECT}
  random_permute: ${RANDOM_PERMUTE}
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
echo "Starting embodied training..."
python3 /workspace/bridge/scripts/train_embodied.py

# --- 5. Run GHE analysis ---
CKPT_DIR=/workspace/bridge/artifacts/checkpoints/$JOB_ID

if [ -f "$CKPT_DIR/latest.pt" ]; then
    echo ""
    echo "Running GHE analysis..."
    cd /workspace

    # Use embodied-specific GHE script (num_actions=2, embodied env)
    RANDPROJ_FLAG=""
    if [ "$RANDOM_PROJECT" = "true" ]; then
        RANDPROJ_FLAG="--random_project"
    fi
    RANDPERM_FLAG=""
    if [ "$RANDOM_PERMUTE" = "true" ]; then
        RANDPERM_FLAG="--random_permute"
    fi
    PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH" \
        python3 /workspace/bridge/scripts/quick_ghe_embodied.py "$CKPT_DIR" \
            --episodes_per 5 --arrangement "$ARRANGEMENT" $RANDPROJ_FLAG $RANDPERM_FLAG

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
echo "Embodied cloud run complete!"
echo "=================================================="
