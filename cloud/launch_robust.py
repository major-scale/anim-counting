#!/usr/bin/env python3
"""
Launch robustness experiment training on RunPod.
4 jobs (sequential): varied-only, randstart-only, combined x2 seeds.
All with random_project=true (our best baseline).

Estimated: ~200 min total (~$2.00 at $0.59/hr Secure Cloud).
"""

import os
from pathlib import Path

try:
    import runpod
except ImportError:
    runpod = None

# Load API key from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists() and runpod is not None:
    for line in env_path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            runpod.api_key = line.split("=", 1)[1].strip()
            break

CLOUD_TYPE = "SECURE"
STEPS = 200000

# Jobs: (job_id, seed, varied_arr, random_start, random_project)
JOBS = [
    ("robust_varied_s0",    0, "true",  "false", "true"),
    ("robust_randstart_s0", 0, "false", "true",  "true"),
    ("robust_combined_s0",  0, "true",  "true",  "true"),
    ("robust_combined_s1",  1, "true",  "true",  "true"),
]


def main():
    print("=" * 60)
    print("ROBUSTNESS EXPERIMENTS -- RunPod Launch")
    print("=" * 60)
    print(f"Steps per job: {STEPS:,}")
    est_min = len(JOBS) * STEPS / 4300
    print(f"Estimated total time: {est_min:.0f} min (~${est_min / 60 * 0.59:.2f})")
    print()

    for job_id, seed, varied, randstart, randproj in JOBS:
        flags = []
        if varied == "true":
            flags.append("varied")
        if randstart == "true":
            flags.append("randstart")
        if randproj == "true":
            flags.append("randproj")
        print(f"  {job_id}: seed={seed}, {'+'.join(flags)}")
    print()

    # Generate command script
    cmd_file = Path(__file__).parent / "pod_robust_cmd.sh"
    with open(cmd_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated: robustness experiment training\n")
        f.write("# Run this on RunPod after uploading training package\n\n")
        f.write("cd /workspace\n")
        f.write("tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/\n\n")
        for job_id, seed, varied, randstart, randproj in JOBS:
            f.write(f"\n# Job: {job_id} (seed={seed})\n")
            f.write(
                f"bash /workspace/bridge/cloud/cloud_run_robust.sh "
                f"{seed} {STEPS} {job_id} {varied} {randstart} {randproj}\n"
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
    if runpod is None or not getattr(runpod, 'api_key', None):
        print("No RunPod API key found. Set RUNPOD_API_KEY in bridge/.env")
        print("You can manually run pod_robust_cmd.sh on a RunPod pod.")
        return

    print("Creating RunPod pod...")
    try:
        pod = runpod.create_pod(
            name="anim-robust",
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
        print("  3. SSH in and run: bash /workspace/bridge/cloud/pod_robust_cmd.sh")
        print("  4. Results will be in /workspace/results/*_ghe.json")
    except Exception as e:
        print(f"Failed to create pod: {e}")
        print("You can manually create a pod and run pod_robust_cmd.sh")


if __name__ == "__main__":
    main()
