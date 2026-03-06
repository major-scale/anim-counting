#!/bin/bash
# Monitor and stop multi-dim training at 50K steps, run per-dim GHE eval
CKPT_DIR="/workspace/bridge/artifacts/checkpoints/multidim_mixed_s0"

while true; do
    STEP=$(tail -1 "$CKPT_DIR/metrics.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step', 0))" 2>/dev/null || echo 0)
    echo "$(date): step=$STEP"

    if [ "$STEP" -ge 50000 ]; then
        echo "Reached 50K! Killing training..."
        kill $(pgrep -f dreamer_multidim) 2>/dev/null
        kill $(pgrep -f train_multidim) 2>/dev/null
        sleep 10

        echo "Running GHE at D=2..."
        cd /workspace
        export PYTHONPATH="/workspace/bridge/scripts:/workspace/dreamerv3-torch:$PYTHONPATH"

        python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
            --fixed_dim 2 --proj_dim 128 --episodes_per 5

        echo "Running GHE at D=3..."
        python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
            --fixed_dim 3 --proj_dim 128 --episodes_per 3

        echo "Running GHE at D=5..."
        python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
            --fixed_dim 5 --proj_dim 128 --episodes_per 3

        echo "Running GHE at D=10..."
        python3 /workspace/bridge/scripts/quick_ghe_multidim.py "$CKPT_DIR" \
            --fixed_dim 10 --proj_dim 128 --episodes_per 3

        echo "ALL EVAL DONE"
        touch /workspace/EVAL_DONE
        exit 0
    fi

    sleep 300
done
