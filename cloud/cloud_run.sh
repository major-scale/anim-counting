#!/bin/bash
# Self-contained cloud training script for RunPod
# Usage: cloud_run.sh <seed> <total_steps> [job_id] [arrangement] [mask_count] [mask_slots] [shuffle_blobs] [random_project] [random_permute]
#
# Expects the anim training package to be unpacked at /workspace/
# Outputs: /workspace/results/summary.json with GHE metrics

set -e

SEED=${1:-5}
TOTAL_STEPS=${2:-50000}
JOB_ID=${3:-"cloud_seed${SEED}"}
ARRANGEMENT=${4:-"grid"}
MASK_COUNT=${5:-"false"}
MASK_SLOTS=${6:-"false"}
SHUFFLE_BLOBS=${7:-"false"}
RANDOM_PROJECT=${8:-"false"}
RANDOM_PERMUTE=${9:-"false"}
ZERO_ACTION=${10:-"false"}

echo "=================================================="
echo "Anim Cloud Training"
echo "=================================================="
echo "  Seed:       $SEED"
echo "  Steps:      $TOTAL_STEPS"
echo "  Job ID:     $JOB_ID"
echo "  Arrangement: $ARRANGEMENT"
echo "  Mask count: $MASK_COUNT"
echo "  Mask slots: $MASK_SLOTS"
echo "  Shuffle:    $SHUFFLE_BLOBS"
echo "  Random proj: $RANDOM_PROJECT"
echo "  Random perm: $RANDOM_PERMUTE"
echo "  Zero action: $ZERO_ACTION"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
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

# --- 3. Set up directory structure ---
mkdir -p /workspace/bridge/artifacts/checkpoints/$JOB_ID
mkdir -p /workspace/bridge/artifacts/status
mkdir -p /workspace/bridge/queue/pending
mkdir -p /workspace/results

# --- 4. Create params YAML (train.py reads flat params, not the full job wrapper) ---
PARAMS_FILE=/workspace/bridge/cloud/${JOB_ID}_params.yaml
cat > "$PARAMS_FILE" << EOF
env_config:
  n_blobs_min: 3
  n_blobs_max: 25
  action_space: continuous
  bidirectional: true
  arrangement: ${ARRANGEMENT}
  mask_count: ${MASK_COUNT}
  mask_slots: ${MASK_SLOTS}
  shuffle_blobs: ${SHUFFLE_BLOBS}
  random_project: ${RANDOM_PROJECT}
  random_permute: ${RANDOM_PERMUTE}
  zero_action: ${ZERO_ACTION}
training:
  total_steps: ${TOTAL_STEPS}
  checkpoint_interval: 50000
  seed: ${SEED}
EOF

# --- 5. Set environment and run training ---
export DREAMER_DIR=/workspace/dreamerv3-torch
export ANIM_BRIDGE=/workspace/bridge
export ANIM_ARTIFACTS=/workspace/bridge/artifacts
export JOB_ID=$JOB_ID
export JOB_PARAMS=$PARAMS_FILE
export PYTHONPATH="/workspace/bridge/scripts:$PYTHONPATH"

echo ""
echo "Starting training..."
python3 /workspace/bridge/scripts/train.py

# --- 6. Run GHE analysis ---
CKPT_DIR=/workspace/bridge/artifacts/checkpoints/$JOB_ID

if [ -f "$CKPT_DIR/latest.pt" ]; then
    echo ""
    echo "Running GHE analysis..."
    cd /workspace
    COUNTING_MASK_COUNT="$MASK_COUNT" \
    COUNTING_MASK_SLOTS="$MASK_SLOTS" \
    COUNTING_SHUFFLE_BLOBS="$SHUFFLE_BLOBS" \
    COUNTING_RANDOM_PROJECT="$RANDOM_PROJECT" \
    COUNTING_RANDOM_PERMUTE="$RANDOM_PERMUTE" \
    PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH" \
        python3 /workspace/bridge/scripts/quick_ghe.py "$CKPT_DIR" --episodes_per 5 --arrangement "$ARRANGEMENT"

    # Copy results to /workspace/results/
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
echo "Cloud run complete!"
echo "=================================================="
