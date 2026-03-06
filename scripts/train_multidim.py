#!/usr/bin/env python3
"""
Training script for multi-dimensional counting world.
Like train.py but uses dreamer_multidim.py and MULTIDIM_* env vars.

Env vars set by server (or manually):
    JOB_ID          — unique job identifier
    JOB_PARAMS      — path to params YAML
    ANIM_BRIDGE    — bridge root directory
    ANIM_ARTIFACTS — artifacts directory
"""

import json
import os
import re
import subprocess
import sys
import time
import yaml
from pathlib import Path
from threading import Thread

# --- Paths ---
DREAMER_DIR = Path(os.environ.get("DREAMER_DIR", str(Path.home() / "dreamerv3-torch")))
BRIDGE = Path(os.environ.get("ANIM_BRIDGE", str(Path.home() / "anim-bridge")))
BRIDGE_SCRIPTS = BRIDGE / "scripts"
ARTIFACTS = Path(os.environ.get("ANIM_ARTIFACTS", str(BRIDGE / "artifacts")))
JOB_ID = os.environ.get("JOB_ID", "unnamed_multidim")
STATUS_DIR = ARTIFACTS / "status"


def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def read_params():
    params_path = os.environ.get("JOB_PARAMS")
    if params_path and Path(params_path).exists():
        with open(params_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def write_progress(data):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_DIR / f"{JOB_ID}_progress.json", "w") as f:
        json.dump(data, f)


def parse_dreamer_output(line, total_steps):
    m = re.search(r'\[(\d+)\]|[Ss]tep\s+(\d+)', line)
    if m:
        step = int(m.group(1) or m.group(2))
        pct = min(100.0, step / total_steps * 100)
        metrics = {}
        for pattern, key in [
            (r'loss[:\s]+([0-9.e+-]+)', 'loss'),
            (r'reward[:\s]+([0-9.e+-]+)', 'reward'),
        ]:
            mm = re.search(pattern, line, re.IGNORECASE)
            if mm:
                try:
                    metrics[key] = float(mm.group(1))
                except ValueError:
                    pass
        return step, pct, metrics
    return None, None, None


def monitor_logdir(logdir, total_steps, stop_flag):
    last_checkpoint = None
    while not stop_flag[0]:
        try:
            ckpts = sorted(logdir.glob("*.pt")) if logdir.exists() else []
            if ckpts and ckpts[-1] != last_checkpoint:
                last_checkpoint = ckpts[-1]
                latest = logdir / "latest.pt"
                if last_checkpoint.name != "latest.pt":
                    import shutil
                    shutil.copy2(str(last_checkpoint), str(latest))
        except Exception:
            pass
        time.sleep(30)


def main():
    params = read_params()
    env_config = params.get("env_config", {})
    training = params.get("training", {})

    total_steps = training.get("total_steps", 200000)

    blob_min = env_config.get("n_blobs_min", 3)
    blob_max = env_config.get("n_blobs_max", 13)
    max_steps = env_config.get("max_steps", 2000)
    proj_dim = env_config.get("proj_dim", 128)
    fixed_dim = env_config.get("fixed_dim", None)
    apply_symlog = env_config.get("apply_symlog", True)

    logdir = ARTIFACTS / "checkpoints" / JOB_ID
    logdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Multi-Dim Training: {JOB_ID}")
    print("=" * 60)
    print(f"  Blobs:        {blob_min}-{blob_max}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Proj dim:     {proj_dim}")
    print(f"  Fixed dim:    {fixed_dim or 'sampled'}")
    print(f"  Apply symlog: {apply_symlog}")
    print(f"  Steps:        {total_steps:,}")
    print(f"  Logdir:       {logdir}")
    print(f"  DreamerV3:    {DREAMER_DIR}")
    print()

    dreamer_script = BRIDGE_SCRIPTS / "dreamer_multidim.py"
    if not dreamer_script.exists():
        print(f"ERROR: dreamer_multidim.py not found at {dreamer_script}")
        sys.exit(1)

    # Set MULTIDIM_* env vars
    env = os.environ.copy()
    env["MULTIDIM_BLOB_MIN"] = str(blob_min)
    env["MULTIDIM_BLOB_MAX"] = str(blob_max)
    env["MULTIDIM_MAX_STEPS"] = str(max_steps)
    env["MULTIDIM_PROJ_DIM"] = str(proj_dim)
    if apply_symlog:
        env["MULTIDIM_APPLY_SYMLOG"] = "true"
    else:
        env["MULTIDIM_APPLY_SYMLOG"] = "false"
    if fixed_dim:
        env["MULTIDIM_FIXED_DIM"] = str(fixed_dim)
    # Zero action for passive observer
    env["DREAMER_ZERO_ACTION"] = "true"

    env["PYTHONPATH"] = str(BRIDGE_SCRIPTS) + ":" + str(DREAMER_DIR) + ":" + env.get("PYTHONPATH", "")
    env["DREAMER_DIR"] = str(DREAMER_DIR)

    cmd = [
        sys.executable, str(dreamer_script),
        "--configs", "multidim_counting",
        "--logdir", str(logdir),
        "--steps", str(total_steps),
        "--compile", "False",
        "--device", _detect_device(),
    ]

    seed = training.get("seed", 0)
    cmd.extend(["--seed", str(seed)])

    print(f"Command: {' '.join(cmd)}")
    print(f"Env: MULTIDIM_BLOB_MIN={blob_min} MULTIDIM_BLOB_MAX={blob_max}")
    print()
    sys.stdout.flush()

    start_time = time.time()
    write_progress({
        "step": 0, "total_steps": total_steps,
        "percent": 0.0, "elapsed_seconds": 0, "status": "starting",
    })

    stop_flag = [False]
    monitor = Thread(target=monitor_logdir, args=(logdir, total_steps, stop_flag), daemon=True)
    monitor.start()

    process = subprocess.Popen(
        cmd, cwd=str(DREAMER_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )

    last_step = 0
    last_progress_time = time.time()

    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            step, pct, metrics = parse_dreamer_output(line, total_steps)
            if step is not None and step > last_step:
                last_step = step
                elapsed = time.time() - start_time
                eta = (elapsed / step * (total_steps - step)) if step > 0 else 0
                progress = {
                    "step": step, "total_steps": total_steps,
                    "percent": round(pct, 1), "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1), "status": "training",
                }
                if metrics:
                    progress.update(metrics)
                write_progress(progress)
                last_progress_time = time.time()
            elif time.time() - last_progress_time > 60:
                elapsed = time.time() - start_time
                write_progress({
                    "step": last_step, "total_steps": total_steps,
                    "percent": round(last_step / total_steps * 100, 1) if total_steps > 0 else 0,
                    "elapsed_seconds": round(elapsed, 1), "status": "training (heartbeat)",
                })
                last_progress_time = time.time()
    except Exception as e:
        print(f"Error reading process output: {e}", flush=True)

    returncode = process.wait()
    stop_flag[0] = True

    elapsed = time.time() - start_time

    if returncode == 0:
        print()
        print("=" * 60)
        print(f"Multi-dim training complete!")
        print(f"  Steps: {last_step:,}")
        print(f"  Time:  {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Logdir: {logdir}")
        print("=" * 60)
        write_progress({
            "step": total_steps, "total_steps": total_steps,
            "percent": 100.0, "elapsed_seconds": round(elapsed, 1), "status": "done",
        })
    else:
        print(f"\nTraining failed with exit code {returncode}", flush=True)
        write_progress({
            "step": last_step, "total_steps": total_steps,
            "percent": round(last_step / total_steps * 100, 1),
            "elapsed_seconds": round(elapsed, 1), "status": f"failed (exit {returncode})",
        })
        sys.exit(returncode)


if __name__ == "__main__":
    main()
