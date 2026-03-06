#!/bin/bash
# Auto-generated: random permutation ablation
# Run this on RunPod after uploading training package


# Job: ablation_randperm_s0 (seed=0, steps=200000)
bash /workspace/bridge/cloud/cloud_run.sh 0 200000 ablation_randperm_s0 grid false false false false true

# Job: ablation_randperm_s7 (seed=7, steps=200000)
bash /workspace/bridge/cloud/cloud_run.sh 7 200000 ablation_randperm_s7 grid false false false false true

# Job: ablation_randperm_s8 (seed=8, steps=200000)
bash /workspace/bridge/cloud/cloud_run.sh 8 200000 ablation_randperm_s8 grid false false false false true
