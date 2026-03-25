#!/usr/bin/env python3
"""
Launch symbolic binary specialist experiments on RunPod.
Two variants (clean + rich) trained sequentially, plus random baselines,
full battery, and three-way cross-modal comparison.

Estimated: ~2 hours total (~$1.20 at $0.59/hr Secure Cloud).
Bundle with weight decay sweep for ~6h total, ~$4.

Usage:
    python launch_symbolic.py
"""

import os
import subprocess
import textwrap
from pathlib import Path

try:
    import runpod
except ImportError:
    runpod = None

# Load API key
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists() and runpod is not None:
    for line in env_path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            runpod.api_key = line.split("=", 1)[1].strip()
            break

CLOUD_TYPE = "SECURE"
STEPS = 500_000
POD_NAME = "anim-symbolic-ab"


def build_training_package():
    """Create tar.gz with everything needed to run on the pod."""
    cloud_dir = Path(__file__).parent
    bridge_dir = cloud_dir.parent
    pkg_path = cloud_dir / "anim-symbolic-package.tar.gz"

    print("Building training package...")

    # Files to include
    includes = [
        "scripts/symbolic_binary_env.py",
        "scripts/symbolic_binary_env_rich.py",
        "scripts/symbolic_rssm.py",
        "scripts/train_symbolic.py",
        "scripts/eval_symbolic_battery.py",
        "scripts/run_symbolic_experiments.sh",
        # Physical specialist battery for cross-modal comparison
        "artifacts/battery/binary_baseline_s0/battery.npz",
    ]

    # Check jamstack-v1 fallback for battery
    battery_src = bridge_dir / "artifacts/battery/binary_baseline_s0/battery.npz"
    if not battery_src.exists():
        alt = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")
        if alt.exists():
            # Copy to expected location
            battery_src.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(alt), str(battery_src))
            print(f"  Copied physical battery from {alt}")

    # Build tar
    existing = [f for f in includes if (bridge_dir / f).exists()]
    missing = [f for f in includes if not (bridge_dir / f).exists()]

    if missing:
        print(f"  WARNING: Missing files: {missing}")

    cmd = ["tar", "czf", str(pkg_path)]
    cmd += ["-C", str(bridge_dir)]
    cmd += existing

    subprocess.run(cmd, check=True)
    size_mb = pkg_path.stat().st_size / 1024 / 1024
    print(f"  Package: {pkg_path} ({size_mb:.1f} MB, {len(existing)} files)")
    return pkg_path


def generate_pod_cmd():
    """Generate the command script to run on the pod."""
    cmd_file = Path(__file__).parent / "pod_symbolic_cmd.sh"
    content = textwrap.dedent(f"""\
        #!/bin/bash
        # Auto-generated: symbolic binary specialist experiments
        # Run this on RunPod after uploading training package
        set -e

        echo "============================================"
        echo "Symbolic Binary Specialist Experiments"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
        echo "============================================"

        # --- 1. Install dependencies ---
        echo "Installing dependencies..."
        pip install -q \\
            ruamel.yaml==0.17.4 \\
            gym==0.22.0 \\
            scikit-learn \\
            scipy \\
            matplotlib \\
            2>&1 | tail -1

        # Optional: ripser for topology
        pip install -q ripser 2>/dev/null || echo "ripser not available, topology metrics will be skipped"

        # --- 2. Unpack training package ---
        cd /workspace
        mkdir -p bridge
        tar xzf /workspace/anim-symbolic-package.tar.gz -C /workspace/bridge/

        cd /workspace/bridge/scripts

        # --- 3. Random baselines ---
        echo ""
        echo "=== Random Baselines ==="
        python3 train_symbolic.py --random-baseline --env clean
        python3 train_symbolic.py --random-baseline --env rich

        # --- 4. Train Variant A (clean) ---
        echo ""
        echo "=== Training Variant A (clean, {STEPS} steps) ==="
        python3 train_symbolic.py \\
            --env clean \\
            --steps {STEPS} \\
            --device auto \\
            --seed 0

        # --- 5. Train Variant B (rich) ---
        echo ""
        echo "=== Training Variant B (rich, {STEPS} steps) ==="
        python3 train_symbolic.py \\
            --env rich \\
            --steps {STEPS} \\
            --device auto \\
            --seed 0

        # --- 6. Full battery + cross-modal ---
        CKPT_A="../artifacts/checkpoints/symbolic_clean_s0"
        CKPT_B="../artifacts/checkpoints/symbolic_rich_s0"
        PHYS="../artifacts/battery/binary_baseline_s0/battery.npz"

        EVAL_ARGS="--n_episodes 10 --n_cycles 2 --device auto"
        if [ -f "$PHYS" ]; then
            EVAL_ARGS="$EVAL_ARGS --physical_battery $PHYS"
        fi

        echo ""
        echo "=== Full Battery: Variant A ==="
        python3 eval_symbolic_battery.py "$CKPT_A" --env clean $EVAL_ARGS

        echo ""
        echo "=== Full Battery: Variant B ==="
        python3 eval_symbolic_battery.py "$CKPT_B" --env rich $EVAL_ARGS

        # --- 7. Three-way comparison ---
        if [ -f "$PHYS" ]; then
            echo ""
            echo "=== Three-Way Comparison ==="
            python3 eval_symbolic_battery.py --three-way \\
                --ckpt_a "$CKPT_A" \\
                --ckpt_b "$CKPT_B" \\
                --physical_battery "$PHYS" \\
                --device auto
        fi

        echo ""
        echo "============================================"
        echo "ALL DONE"
        echo "============================================"
        echo ""
        echo "Results:"
        ls -la "$CKPT_A"/battery_results.json "$CKPT_B"/battery_results.json 2>/dev/null
        ls -la "$CKPT_A"/cross_modal.json "$CKPT_B"/cross_modal.json 2>/dev/null
        echo ""
        echo "Download:"
        echo "  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_clean_s0 ."
        echo "  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_rich_s0 ."
    """)
    cmd_file.write_text(content)
    print(f"  Pod command script: {cmd_file}")
    return cmd_file


def main():
    print("=" * 60)
    print("SYMBOLIC BINARY SPECIALIST — RunPod Launch")
    print("=" * 60)
    print(f"  Variant A: clean symbolic (no carry dynamics)")
    print(f"  Variant B: rich symbolic (carry dynamics, matched timing)")
    print(f"  Steps per variant: {STEPS:,}")
    est_h = 2.0
    print(f"  Estimated time: ~{est_h:.0f}h (~${est_h * 0.59:.2f})")
    print()

    pkg_path = build_training_package()
    cmd_file = generate_pod_cmd()
    print()

    if runpod is None or not getattr(runpod, "api_key", None):
        print("No RunPod API key found. Set RUNPOD_API_KEY in bridge/.env")
        print(f"Manual steps:")
        print(f"  1. Create pod on RunPod (RTX 4090, PyTorch 2.1)")
        print(f"  2. scp {pkg_path} root@<ip>:/workspace/anim-symbolic-package.tar.gz")
        print(f"  3. SSH in, run: bash /workspace/bridge/cloud/pod_symbolic_cmd.sh")
        return

    print("Creating RunPod pod...")
    try:
        ssh_pub = ""
        for key_name in ["id_ed25519.pub", "id_rsa.pub"]:
            key_path = Path(os.path.expanduser(f"~/.ssh/{key_name}"))
            if key_path.exists():
                ssh_pub = key_path.read_text().strip()
                break

        pod = runpod.create_pod(
            name=POD_NAME,
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id="NVIDIA GeForce RTX 4090",
            cloud_type=CLOUD_TYPE,
            volume_in_gb=50,
            container_disk_in_gb=20,
            min_vcpu_count=4,
            min_memory_in_gb=16,
            gpu_count=1,
            ports="22/tcp",
            env={"PUBLIC_KEY": ssh_pub} if ssh_pub else {},
        )
        pod_id = pod.get("id", "unknown")
        print(f"  Pod created: {pod_id}")

        # Save pod info
        info_file = Path(__file__).parent / "active_pod_symbolic.txt"
        info_file.write_text(
            f"Pod: {pod_id}\n"
            f"Experiment: symbolic binary A+B\n"
            f"Steps: {STEPS:,} per variant\n\n"
            f"Upload:\n"
            f"  scp -P PORT {pkg_path} root@IP:/workspace/anim-symbolic-package.tar.gz\n\n"
            f"Run:\n"
            f"  ssh -p PORT root@IP 'bash /workspace/bridge/cloud/pod_symbolic_cmd.sh'\n\n"
            f"Download:\n"
            f"  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_clean_s0 .\n"
            f"  scp -rP PORT root@IP:/workspace/bridge/artifacts/checkpoints/symbolic_rich_s0 .\n"
        )
        print(f"  Pod info saved to: {info_file}")
        print()
        print("Next steps:")
        print("  1. Wait for pod to start (check RunPod dashboard)")
        print("  2. Get SSH port from dashboard")
        print(f"  3. scp -P <port> {pkg_path} root@<ip>:/workspace/anim-symbolic-package.tar.gz")
        print("  4. ssh -p <port> root@<ip> 'nohup bash /workspace/bridge/cloud/pod_symbolic_cmd.sh > /workspace/symbolic.log 2>&1 &'")
        print("  5. Monitor: ssh -p <port> root@<ip> 'tail -f /workspace/symbolic.log'")

    except Exception as e:
        print(f"Failed to create pod: {e}")
        print("Create manually on RunPod dashboard and run pod_symbolic_cmd.sh")


if __name__ == "__main__":
    main()
