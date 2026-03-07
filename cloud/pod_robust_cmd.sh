#!/bin/bash
# Auto-generated: robustness experiment training
# Run this on RunPod after uploading training package

cd /workspace
tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/


# Job: robust_varied_s0 (seed=0)
bash /workspace/bridge/cloud/cloud_run_robust.sh 0 200000 robust_varied_s0 true false true
cp /workspace/bridge/artifacts/checkpoints/robust_varied_s0/quick_ghe.json /workspace/results/robust_varied_s0_ghe.json 2>/dev/null || true

# Job: robust_randstart_s0 (seed=0)
bash /workspace/bridge/cloud/cloud_run_robust.sh 0 200000 robust_randstart_s0 false true true
cp /workspace/bridge/artifacts/checkpoints/robust_randstart_s0/quick_ghe.json /workspace/results/robust_randstart_s0_ghe.json 2>/dev/null || true

# Job: robust_combined_s0 (seed=0)
bash /workspace/bridge/cloud/cloud_run_robust.sh 0 200000 robust_combined_s0 true true true
cp /workspace/bridge/artifacts/checkpoints/robust_combined_s0/quick_ghe.json /workspace/results/robust_combined_s0_ghe.json 2>/dev/null || true

# Job: robust_combined_s1 (seed=1)
bash /workspace/bridge/cloud/cloud_run_robust.sh 1 200000 robust_combined_s1 true true true
cp /workspace/bridge/artifacts/checkpoints/robust_combined_s1/quick_ghe.json /workspace/results/robust_combined_s1_ghe.json 2>/dev/null || true

echo '=== ALL JOBS COMPLETE ==='
ls -la /workspace/results/
cat /workspace/results/*_ghe.json
