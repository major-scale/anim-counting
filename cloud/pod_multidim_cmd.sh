#!/bin/bash
# Auto-generated: multi-dimensional counting world v2 training
# Run this on RunPod after uploading training package

cd /workspace
tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/

# Job 1: Mixed dimensionality (2-5), 200K steps, seed 0
bash /workspace/bridge/cloud/cloud_run_multidim.sh 0 200000 multidim_v2_s0 128
