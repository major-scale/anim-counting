#!/bin/bash
# Auto-generated: embodied agent training
# Run this on RunPod after uploading training package

cd /workspace
tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/


# Job: embodied_baseline_s0 (seed=0)
bash /workspace/bridge/cloud/cloud_run_embodied.sh 0 200000 embodied_baseline_s0 grid false false

# Job: embodied_randproj_s0 (seed=0)
bash /workspace/bridge/cloud/cloud_run_embodied.sh 0 200000 embodied_randproj_s0 grid true false
