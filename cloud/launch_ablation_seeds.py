#!/usr/bin/env python3
"""
Launch 6 ablation seed runs on RunPod.
3 pods x 2 sequential jobs each.

Pod 1: no-count seed 7, no-count seed 8 (100K each, ~50 min total)
Pod 2: no-slots+no-count seed 7, no-slots+no-count seed 8 (200K each, ~100 min total)
Pod 3: shuffle+no-slots+no-count seed 7, shuffle+no-slots+no-count seed 8 (200K each, ~100 min total)
"""

import runpod
import time
import json
import os

runpod.api_key = "REDACTED_RUNPOD_API_KEY"

# Training package URL (uploaded from local)
PACKAGE_PATH = "/workspace/bridge/cloud/anim-training-package.tar.gz"

# Pod configuration
GPU_TYPE = "NVIDIA GeForce RTX 4090"
CLOUD_TYPE = "SECURE"  # COMMUNITY for cheaper, SECURE for reliability
TEMPLATE_ID = "runpod-torch-v21"  # Official PyTorch template

# Seeds: use 7 and 8 (different from any existing seeds 0-5)
SEED_A, SEED_B = 7, 8

# The 3 conditions with their cloud_run.sh arguments
# Format: (job_prefix, steps, arrangement, mask_count, mask_slots, shuffle_blobs)
CONDITIONS = {
    "pod1_nocount": [
        ("ablation_nocount_s7", 100000, "grid", "true", "false", "false", SEED_A),
        ("ablation_nocount_s8", 100000, "grid", "true", "false", "false", SEED_B),
    ],
    "pod2_noslots": [
        ("ablation_noslots_s7", 200000, "grid", "true", "true", "false", SEED_A),
        ("ablation_noslots_s8", 200000, "grid", "true", "true", "false", SEED_B),
    ],
    "pod3_shuffle": [
        ("ablation_shuffle_s7", 200000, "grid", "true", "true", "true", SEED_A),
        ("ablation_shuffle_s8", 200000, "grid", "true", "true", "true", SEED_B),
    ],
}


def make_start_command(jobs):
    """Create a bash command that runs cloud_run.sh for each job sequentially."""
    cmds = []
    cmds.append("cd /workspace")
    cmds.append("tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/")

    for job_id, steps, arrangement, mask_count, mask_slots, shuffle_blobs, seed in jobs:
        cmds.append(f"echo '=== Starting {job_id} ==='")
        cmds.append(
            f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
            f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs}"
        )
        cmds.append(f"echo '=== Finished {job_id} ==='")
        # Save results to a predictable location
        cmds.append(f"cp /workspace/bridge/artifacts/checkpoints/{job_id}/quick_ghe.json /workspace/results/{job_id}_ghe.json 2>/dev/null || true")

    # Final summary
    cmds.append("echo '=== ALL JOBS COMPLETE ==='")
    cmds.append("ls -la /workspace/results/")
    cmds.append("cat /workspace/results/*_ghe.json")

    return " && ".join(cmds)


def launch_pod(name, jobs):
    """Launch a RunPod pod with the given jobs."""
    start_cmd = make_start_command(jobs)

    pod = runpod.create_pod(
        name=name,
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        cloud_type=CLOUD_TYPE,
        volume_in_gb=50,
        container_disk_in_gb=20,
        min_vcpu_count=4,
        min_memory_in_gb=16,
        gpu_count=1,
        # Upload the package via volume mount
        docker_args=f"bash -c '{start_cmd}'",
    )

    return pod


def main():
    print("=" * 60)
    print("LAUNCHING ABLATION SEED RUNS ON RUNPOD")
    print("=" * 60)
    print(f"Seeds: {SEED_A}, {SEED_B}")
    print(f"GPU: {GPU_TYPE}")
    print()

    # First, we need to upload the training package.
    # The simplest approach: create pods with volume, upload via runpodctl
    # But that's complex. Instead, let's create pods that download the package.

    # For now, let's just create the pods and note we need to upload the package.
    # The cloud_run.sh expects the package to already be unpacked at /workspace/

    pods = {}
    for pod_name, jobs in CONDITIONS.items():
        total_steps = sum(j[1] for j in jobs)
        est_minutes = total_steps / 4300  # ~4300 steps/min on RTX 4090
        print(f"\nPod: {pod_name}")
        print(f"  Jobs: {[j[0] for j in jobs]}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Estimated time: {est_minutes:.0f} min")

    print("\n" + "=" * 60)
    print("Ready to launch. Use runpod CLI or web UI to create pods.")
    print("=" * 60)

    # Generate the commands for each pod
    for pod_name, jobs in CONDITIONS.items():
        print(f"\n### {pod_name} ###")
        cmd = make_start_command(jobs)
        print(f"Command length: {len(cmd)} chars")
        # Save command to file for reference
        with open(f"/workspace/bridge/cloud/{pod_name}_cmd.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated command for RunPod pod\n")
            for job_id, steps, arrangement, mask_count, mask_slots, shuffle_blobs, seed in jobs:
                f.write(f"\n# Job: {job_id} (seed={seed}, steps={steps})\n")
                f.write(f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
                        f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs}\n")
        print(f"  Saved to /workspace/bridge/cloud/{pod_name}_cmd.sh")


if __name__ == "__main__":
    main()
