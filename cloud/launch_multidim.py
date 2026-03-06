#!/usr/bin/env python3
"""
Launch multi-dimensional counting world v2 training on RunPod.

v2 changes:
  - 13 blobs max (was 25)
  - D=2-5 with weights 30/25/25/20 (was 2-10 with 50/20/declining)
  - Faster bot (8 units/step), smaller world (700), larger pickup (60)
  - 2000 step cap (was 6000)
  - Cross-dimensional probe transfer as PRIMARY eval metric

Jobs:
  1. multidim_v2_s0: Mixed D (2-5), 200K steps, seed 0

Usage:
    python launch_multidim.py [--dry-run]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BRIDGE = Path(os.environ.get("ANIM_BRIDGE", "/workspace/bridge"))


def build_command_script():
    """Generate the pod command script."""
    script = """#!/bin/bash
# Auto-generated: multi-dimensional counting world v2 training
# Run this on RunPod after uploading training package

cd /workspace
tar xzf /workspace/bridge/cloud/anim-training-package.tar.gz -C /workspace/

# Job 1: Mixed dimensionality (2-5), 200K steps, seed 0
bash /workspace/bridge/cloud/cloud_run_multidim.sh 0 200000 multidim_v2_s0 128
"""
    cmd_path = BRIDGE / "cloud" / "pod_multidim_cmd.sh"
    with open(cmd_path, "w") as f:
        f.write(script)
    print(f"Command script written to {cmd_path}")
    return cmd_path


def create_pod(dry_run=False):
    """Create a RunPod pod for multi-dim training."""
    try:
        import runpod
    except ImportError:
        print("ERROR: runpod package not installed. pip install runpod")
        sys.exit(1)

    # Load API key
    env_path = BRIDGE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                os.environ["RUNPOD_API_KEY"] = line.split("=", 1)[1].strip()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    runpod.api_key = api_key

    # Get SSH public key
    ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
    ssh_pub_key = ""
    if ssh_key_path.exists():
        ssh_pub_key = ssh_key_path.read_text().strip()

    pod_name = "anim-multidim-v2"
    gpu_type = "NVIDIA GeForce RTX 4090"

    print(f"\nCreating pod: {pod_name}")
    print(f"  GPU: {gpu_type}")
    print(f"  Template: runpod-torch-v21")

    if dry_run:
        print("\n[DRY RUN] Would create pod. Exiting.")
        return

    pod = runpod.create_pod(
        name=pod_name,
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=50,
        container_disk_in_gb=20,
        ports="22/tcp",
        env={
            "PUBLIC_KEY": ssh_pub_key,
        },
    )

    pod_id = pod["id"]
    print(f"\nPod created: {pod_id}")
    print(f"  Waiting for pod to be ready...")

    # Wait for ready
    for _ in range(60):
        time.sleep(10)
        status = runpod.get_pod(pod_id)
        runtime = status.get("runtime", {})
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            ports = runtime.get("ports", [])
            ssh_port = None
            for p in ports:
                if p.get("privatePort") == 22:
                    ssh_port = p.get("publicPort")
                    break
            ip = runtime.get("publicIp", "unknown")
            print(f"\n  Pod ready!")
            print(f"  IP: {ip}")
            print(f"  SSH port: {ssh_port}")
            print(f"\n  Upload package:")
            print(f"    scp -P {ssh_port} {BRIDGE}/cloud/anim-training-package.tar.gz root@{ip}:/workspace/bridge/cloud/")
            print(f"    scp -P {ssh_port} {BRIDGE}/cloud/pod_multidim_cmd.sh root@{ip}:/workspace/")
            print(f"\n  Then SSH in and run:")
            print(f"    ssh -p {ssh_port} root@{ip}")
            print(f"    bash /workspace/pod_multidim_cmd.sh")
            return
        print(".", end="", flush=True)

    print("\nWARNING: Pod didn't become ready in 10 minutes. Check RunPod dashboard.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_command_script()

    # Rebuild training package
    print("\nRebuilding training package...")
    subprocess.run(
        ["tar", "czf", str(BRIDGE / "cloud/anim-training-package.tar.gz"),
         "dreamerv3-torch/dreamer.py",
         "dreamerv3-torch/exploration.py",
         "dreamerv3-torch/models.py",
         "dreamerv3-torch/networks.py",
         "dreamerv3-torch/parallel.py",
         "dreamerv3-torch/tools.py",
         "dreamerv3-torch/configs.yaml",
         "bridge/scripts/counting_env_pure.py",
         "bridge/scripts/counting_env_multidim.py",
         "bridge/scripts/configs.yaml",
         "bridge/scripts/dreamer_multidim.py",
         "bridge/scripts/train_multidim.py",
         "bridge/scripts/quick_ghe_multidim.py",
         "bridge/scripts/probe_transfer_eval.py",
         "bridge/scripts/envs/__init__.py",
         "bridge/scripts/envs/counting.py",
         "bridge/scripts/envs/multidim.py",
         "bridge/scripts/envs/wrappers.py",
         "bridge/cloud/cloud_run_multidim.sh"],
        cwd="/workspace",
        check=True,
    )
    print("  Package rebuilt.")

    create_pod(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
