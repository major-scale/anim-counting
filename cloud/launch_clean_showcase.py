#!/usr/bin/env python3
"""
Launch two clean showcase models on RunPod.

Model 1: clean_baseline_s0 — count-at-placement + zero_action, seed 0
Model 2: clean_randproj_s0 — count-at-placement + zero_action + random_project, seed 0

Both run 200K steps. Sequential on a single RTX 4090 pod.
Estimated: ~95 min total (~$0.93 at $0.59/hr).

Changes from previous training:
  - count-at-placement: grid_filled_count defers until blob animation completes
    (eliminates ~125 frames of conflicting signal per count transition)
  - zero_action: RSSM receives zeros instead of action (bot is hardcoded,
    action channel was pure noise)
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
STEPS = 200000

# Job definitions:
# (job_id, seed, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, random_permute, zero_action)
JOBS = [
    ("clean_baseline_s0", 0, STEPS, "grid", "false", "false", "false", "false", "false", "true"),
    ("clean_randproj_s0", 0, STEPS, "grid", "false", "false", "false", "true", "false", "true"),
]


def make_start_command(jobs):
    """Create bash command that runs cloud_run.sh for each job sequentially."""
    cmds = [
        "cd /workspace",
        "tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/",
    ]

    for job_id, seed, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, random_permute, zero_action in jobs:
        cmds.append(f"echo '=== Starting {job_id} ==='")
        cmds.append(
            f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
            f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs} {random_project} "
            f"{random_permute} {zero_action}"
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
    print("CLEAN SHOWCASE MODELS — RunPod Launch")
    print("=" * 60)
    print(f"Steps per model: {STEPS:,}")
    est_min = len(JOBS) * STEPS / 4300
    print(f"Estimated total time: {est_min:.0f} min (~${est_min / 60 * 0.59:.2f})")
    print()
    print("Changes from previous training:")
    print("  - count-at-placement: grid_filled defers until blob lands")
    print("  - zero_action: RSSM receives zeros (bot is hardcoded)")
    print()

    for job_id, seed, steps, *rest in JOBS:
        random_project = rest[4]
        print(f"  {job_id}: seed={seed}, {steps:,} steps, random_project={random_project}")
    print()

    # Generate command file
    cmd_file = Path(__file__).parent / "pod_clean_showcase_cmd.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated: clean showcase models\n")
        f.write("# Run this on RunPod after uploading training package\n\n")
        f.write("cd /workspace\n")
        f.write("tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/\n\n")
        for job_id, seed, steps, arrangement, mask_count, mask_slots, shuffle_blobs, random_project, random_permute, zero_action in JOBS:
            f.write(f"\n# Job: {job_id} (seed={seed}, steps={steps})\n")
            f.write(
                f"bash /workspace/bridge/cloud/cloud_run.sh {seed} {steps} {job_id} "
                f"{arrangement} {mask_count} {mask_slots} {shuffle_blobs} {random_project} "
                f"{random_permute} {zero_action}\n"
            )
            f.write(
                f"cp /workspace/bridge/artifacts/checkpoints/{job_id}/quick_ghe.json "
                f"/workspace/results/{job_id}_ghe.json 2>/dev/null || true\n"
            )
        f.write("\necho '=== ALL JOBS COMPLETE ==='\n")
        f.write("ls -la /workspace/results/\n")
        f.write("cat /workspace/results/*_ghe.json\n")
    print(f"Command script saved to: {cmd_file}")
    print()

    # Try to launch pod
    if not runpod.api_key:
        print("No RunPod API key found. Set RUNPOD_API_KEY in bridge/.env")
        print("You can manually run pod_clean_showcase_cmd.sh on a RunPod pod.")
        return

    print("Creating RunPod pod...")
    try:
        pod = runpod.create_pod(
            name="anim-clean-showcase",
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
        pod_id = pod.get("id", "unknown")
        print()
        print("Next steps:")
        print(f"  1. Wait for pod to start: runpod pod list")
        print(f"  2. SCP package: scp -P <port> /workspace/bridge/cloud/anim-training-package.tar.gz root@<ip>:/workspace/bridge/cloud/")
        print(f"  3. SSH in and run: bash /workspace/bridge/cloud/pod_clean_showcase_cmd.sh")
        print(f"  4. When done, SCP checkpoints back:")
        print(f"     scp -rP <port> root@<ip>:/workspace/bridge/artifacts/checkpoints/clean_baseline_s0 /workspace/bridge/artifacts/checkpoints/")
        print(f"     scp -rP <port> root@<ip>:/workspace/bridge/artifacts/checkpoints/clean_randproj_s0 /workspace/bridge/artifacts/checkpoints/")
        print(f"  5. Terminate pod: python3 -c \"import runpod; runpod.api_key='...'; runpod.terminate_pod('{pod_id}')\"")

        # Save pod info
        with open(Path(__file__).parent / "active_pod.txt", "w") as f:
            f.write(f"{pod_id}\n")

    except Exception as e:
        print(f"Failed to create pod: {e}")
        print("You can manually create a pod and run pod_clean_showcase_cmd.sh")


if __name__ == "__main__":
    main()
