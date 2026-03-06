#!/usr/bin/env python3
"""
Launch random projection ablation on RunPod.
1 pod x 3 sequential jobs (seeds 0, 7, 8 @ 200K steps each).

Estimated: ~140 min total (~$1.40 at $0.59/hr Secure Cloud).

This ablation multiplies the 82-dim observation by a fixed random orthogonal
matrix, preserving all pairwise distances while destroying spatial semantics.
If the counting manifold survives (GHE < 0.5), the RSSM extracts counting
from distance relationships, not spatial features.
"""

import runpod
import os
from pathlib import Path

# Load API key from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            runpod.api_key = line.split("=", 1)[1].strip()
            break

CLOUD_TYPE = "SECURE"
SEEDS = [0, 7, 8]
STEPS = 200000

# Job definitions: (job_id, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, seed)
JOBS = [
    (f"ablation_randproj_s{seed}", STEPS, "grid", "false", "false", "false", "true", seed)
    for seed in SEEDS
]


def make_start_command(jobs):
    """Create bash command that runs cloud_run.sh for each job sequentially."""
    cmds = [
        "cd /workspace",
        "tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/",
    ]

    for job_id, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, seed in jobs:
        cmds.append(f"echo '=== Starting {job_id} ==='")
        cmds.append(
            f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
            f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs} {random_project}"
        )
        cmds.append(f"echo '=== Finished {job_id} ==='")
        cmds.append(
            f"cp /workspace/bridge/artifacts/checkpoints/{job_id}/quick_ghe.json "
            f"/workspace/results/{job_id}_ghe.json 2>/dev/null || true"
        )

    cmds.append("echo '=== ALL JOBS COMPLETE ==='")
    cmds.append("ls -la /workspace/results/")
    cmds.append("cat /workspace/results/*_ghe.json")

    return " && ".join(cmds)


def main():
    print("=" * 60)
    print("RANDOM PROJECTION ABLATION — RunPod Launch")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"Steps per seed: {STEPS:,}")
    est_min = len(SEEDS) * STEPS / 4300
    print(f"Estimated total time: {est_min:.0f} min (~${est_min / 60 * 0.59:.2f})")
    print()

    for job_id, steps, *_, seed in JOBS:
        print(f"  {job_id}: seed={seed}, {steps:,} steps")
    print()

    # Generate command file
    cmd = make_start_command(JOBS)
    cmd_file = Path(__file__).parent / "pod_randproj_cmd.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated: random projection ablation\n")
        f.write("# Run this on RunPod after uploading training package\n\n")
        for job_id, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, seed in JOBS:
            f.write(f"\n# Job: {job_id} (seed={seed}, steps={steps})\n")
            f.write(
                f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
                f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs} {random_project}\n"
            )
    print(f"Command script saved to: {cmd_file}")
    print()

    # Try to launch pod
    if not runpod.api_key:
        print("No RunPod API key found. Set RUNPOD_API_KEY in bridge/.env")
        print("You can manually run pod_randproj_cmd.sh on a RunPod pod.")
        return

    print("Creating RunPod pod...")
    try:
        pod = runpod.create_pod(
            name="anim-randproj",
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id="NVIDIA GeForce RTX 4090",
            cloud_type=CLOUD_TYPE,
            volume_in_gb=50,
            container_disk_in_gb=20,
            min_vcpu_count=4,
            min_memory_in_gb=16,
            gpu_count=1,
            ports="22/tcp",
            env={
                "PUBLIC_KEY": Path(os.path.expanduser("~/.ssh/id_ed25519.pub")).read_text().strip()
                if Path(os.path.expanduser("~/.ssh/id_ed25519.pub")).exists()
                else ""
            },
        )
        print(f"Pod created: {pod}")
        print()
        print("Next steps:")
        print("  1. Wait for pod to start (check RunPod dashboard)")
        print("  2. SCP training package: scp -P <port> bridge/cloud/anim-training-package.tar.gz root@<ip>:/workspace/bridge/cloud/")
        print("  3. SSH in and run: bash /workspace/bridge/cloud/pod_randproj_cmd.sh")
        print("  4. Results will be in /workspace/results/*_ghe.json")
    except Exception as e:
        print(f"Failed to create pod: {e}")
        print("You can manually create a pod and run pod_randproj_cmd.sh")


if __name__ == "__main__":
    main()
