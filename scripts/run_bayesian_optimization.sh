#!/usr/bin/env bash
#
# Run Bayesian probe optimization: analyze battery data, export params, launch GUI.
#
# Usage:
#   cd ~/anim-bridge/scripts && bash run_bayesian_optimization.sh
#
#   # Skip GUI launch (analysis only):
#   bash run_bayesian_optimization.sh --no-gui
#
#   # Use specific battery data:
#   bash run_bayesian_optimization.sh --data ~/path/to/randproj_battery.npz
#
#   # Record fresh episode first (if no battery data exists):
#   bash run_bayesian_optimization.sh --record
#
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
BRIDGE_DIR="$(dirname "$SCRIPTS_DIR")"
MODELS_DIR="${BRIDGE_DIR}/models/randproj_clean"
PROBE_JSON="${MODELS_DIR}/embed_probe.json"
OUTPUT_DIR="${BRIDGE_DIR}/artifacts/bayesian"
EPISODE_DIR="${BRIDGE_DIR}/artifacts/episodes"
DATA=""
NO_GUI=false
RECORD=false
PREDICTION="bayes+monotonic"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA="$2"; shift 2 ;;
        --no-gui) NO_GUI=true; shift ;;
        --record) RECORD=true; shift ;;
        --prediction) PREDICTION="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; PROBE_JSON="${MODELS_DIR}/embed_probe.json"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--data <npz>] [--record] [--no-gui] [--prediction round|bayes|bayes+monotonic]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Step 0: Record fresh episode if requested or no data found
EPISODE_FILE="${EPISODE_DIR}/randproj_episode.npz"

if [[ "$RECORD" == true ]] || [[ -z "$DATA" ]]; then
    # Auto-detect existing data.
    # IMPORTANT: Prefer educational episode (action=0, matching GUI's FastRSSM)
    # over battery data (DreamerV3 policy actions, different state distribution).
    if [[ -z "$DATA" ]]; then
        candidates=(
            "$EPISODE_FILE"
            "${BRIDGE_DIR}/artifacts/battery/clean/clean_randproj_s0/clean_randproj_s0.npz"
            "${BRIDGE_DIR}/artifacts/battery/randproj_s0/randproj_s0.npz"
        )
        for c in "${candidates[@]}"; do
            if [[ -f "$c" ]]; then
                DATA="$c"
                echo "  Found existing data: $DATA"
                break
            fi
        done
    fi

    # If --record or still no data, record a fresh episode
    if [[ "$RECORD" == true ]] || [[ -z "$DATA" ]]; then
        echo "=== Recording educational episode with randproj model ==="
        mkdir -p "$EPISODE_DIR"
        python3 "${SCRIPTS_DIR}/record_educational_episode.py" \
            --weights-dir "$MODELS_DIR" \
            --out "$EPISODE_FILE" \
            --seed 100 \
            --n-blobs 25
        DATA="$EPISODE_FILE"
    fi
fi

if [[ -z "$DATA" ]]; then
    echo "ERROR: No data available. Use --data <path> or --record to generate."
    exit 1
fi

echo ""
echo "=== Bayesian Probe Optimization ==="
echo "  Data:       $DATA"
echo "  Probe:      $PROBE_JSON"
echo "  Output:     $OUTPUT_DIR"
echo "  Prediction: $PREDICTION"
echo ""

# Step 1: Back up probe JSON before modifying
if [[ -f "$PROBE_JSON" ]]; then
    cp "$PROBE_JSON" "${PROBE_JSON}.bak"
fi

# Step 2: Run optimizer
python3 "${SCRIPTS_DIR}/bayesian_probe_optimizer.py" \
    --data "$DATA" \
    --use-existing-probe "$PROBE_JSON" \
    --export-probe "$PROBE_JSON" \
    --output "$OUTPUT_DIR"

echo ""
echo "=== Bayesian parameters exported to ${PROBE_JSON} ==="

# Step 3: Launch GUI
if [[ "$NO_GUI" == false ]]; then
    echo ""
    echo "=== Launching GUI with --prediction ${PREDICTION} ==="
    python3 "${SCRIPTS_DIR}/visualize_counting.py" \
        --models-dir "$MODELS_DIR" \
        --prediction "$PREDICTION"
fi
