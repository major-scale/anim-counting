#!/bin/bash
# Auto-generated: clean showcase models
# Run this on RunPod after uploading training package

cd /workspace
tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/


# Job: clean_baseline_s0 (seed=0, steps=200000)
bash /workspace/bridge/cloud/cloud_run.sh 0 200000 clean_baseline_s0 grid false false false false false true
cp /workspace/bridge/artifacts/checkpoints/clean_baseline_s0/quick_ghe.json /workspace/results/clean_baseline_s0_ghe.json 2>/dev/null || true

# Job: clean_randproj_s0 (seed=0, steps=200000)
bash /workspace/bridge/cloud/cloud_run.sh 0 200000 clean_randproj_s0 grid false false false true false true
cp /workspace/bridge/artifacts/checkpoints/clean_randproj_s0/quick_ghe.json /workspace/results/clean_randproj_s0_ghe.json 2>/dev/null || true

echo '=== ALL JOBS COMPLETE ==='
ls -la /workspace/results/
cat /workspace/results/*_ghe.json
