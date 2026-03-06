#!/bin/bash
# Monitor and stop multi-dim training at 100K steps, run per-dim GHE eval
CKPT_DIR="/workspace/bridge/artifacts/checkpoints/multidim_mixed_s0"
export PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH"

while true; do
    STEP=$(tail -1 "$CKPT_DIR/metrics.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step', 0))" 2>/dev/null || echo 0)
    echo "$(date): step=$STEP"

    if [ "$STEP" -ge 100000 ]; then
        echo "Reached 100K! Killing training..."
        kill $(pgrep -f dreamer_multidim) 2>/dev/null
        sleep 10

        cd /workspace

        for DIM in 2 3 5 10; do
            EPS=5
            if [ "$DIM" -gt 3 ]; then EPS=3; fi
            echo "Running GHE at D=$DIM ($EPS episodes)..."
            python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
                --fixed_dim $DIM --proj_dim 128 --episodes_per $EPS 2>&1
            echo "D=$DIM done."
        done

        echo "ALL EVAL DONE"
        touch /workspace/EVAL_DONE_100K
        exit 0
    fi

    sleep 300
done
