#!/usr/bin/env python3
"""
Training script for Anim counting world agent.
Launched by the GPU server via server.py.

Reads JOB_PARAMS YAML for config, sets env vars for the counting world,
launches DreamerV3 training, and writes periodic progress JSON.

Env vars set by server:
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
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

# --- Paths ---
DREAMER_DIR = Path.home() / "dreamerv3-torch"
BRIDGE = Path(os.environ.get("ANIM_BRIDGE", str(Path.home() / "anim-bridge")))
BRIDGE_SCRIPTS = BRIDGE / "scripts"
ARTIFACTS = Path(os.environ.get("ANIM_ARTIFACTS", str(BRIDGE / "artifacts")))
JOB_ID = os.environ.get("JOB_ID", "unnamed_train")
STATUS_DIR = ARTIFACTS / "status"


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
    """Parse DreamerV3 stdout for step count and metrics.

    DreamerV3 typically logs lines like:
        [100000] train/loss: 1.234 train/reward: 0.56
    or:
        Step 100000, ...
    """
    # Pattern: [step_number] or Step step_number
    m = re.search(r'\[(\d+)\]|[Ss]tep\s+(\d+)', line)
    if m:
        step = int(m.group(1) or m.group(2))
        pct = min(100.0, step / total_steps * 100)

        # Try to extract loss/reward from the line
        metrics = {}
        for pattern, key in [
            (r'loss[:\s]+([0-9.e+-]+)', 'loss'),
            (r'reward[:\s]+([0-9.e+-]+)', 'reward'),
            (r'actor_loss[:\s]+([0-9.e+-]+)', 'actor_loss'),
            (r'value_loss[:\s]+([0-9.e+-]+)', 'value_loss'),
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
    """Background thread: poll logdir for checkpoints and update progress."""
    last_checkpoint = None
    while not stop_flag[0]:
        try:
            # Check for new checkpoint files
            ckpts = sorted(logdir.glob("*.pt")) if logdir.exists() else []
            if ckpts and ckpts[-1] != last_checkpoint:
                last_checkpoint = ckpts[-1]
                # Also copy checkpoint to a stable "latest" location
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

    total_steps = training.get("total_steps", 300000)
    checkpoint_interval = training.get("checkpoint_interval", 50000)
    eval_interval = training.get("eval_interval", 25000)

    n_blobs_min = env_config.get("n_blobs_min", 3)
    n_blobs_max = env_config.get("n_blobs_max", 25)
    stage = env_config.get("stage", 1)
    conservation = env_config.get("conservation", True)
    action_space = env_config.get("action_space", "continuous")

    # Logdir: save in artifacts/checkpoints/{job_id}/
    logdir = ARTIFACTS / "checkpoints" / JOB_ID
    logdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training: {JOB_ID}")
    print("=" * 60)
    print(f"  Blobs:      {n_blobs_min}-{n_blobs_max}")
    print(f"  Stage:      {stage}")
    print(f"  Action:     {action_space}")
    print(f"  Conservation: {conservation}")
    print(f"  Steps:      {total_steps:,}")
    print(f"  Checkpoint: every {checkpoint_interval:,}")
    print(f"  Eval:       every {eval_interval:,}")
    print(f"  Logdir:     {logdir}")
    print(f"  DreamerV3:  {DREAMER_DIR}")
    disp = params.get("displacement_loss", {})
    if disp.get("enabled", False):
        print(f"  Disp loss:  lambda={disp.get('lambda', 0.1)}, "
              f"ema_decay={disp.get('ema_decay', 0.99)}, "
              f"warmup={disp.get('warmup', 50)}")
    print()

    # Use local patched dreamer.py (with displacement loss support)
    dreamer_script = BRIDGE_SCRIPTS / "dreamer.py"
    if not dreamer_script.exists():
        # Fallback to original
        dreamer_script = DREAMER_DIR / "dreamer.py"
    if not dreamer_script.exists():
        print(f"ERROR: DreamerV3 not found at {dreamer_script}")
        sys.exit(1)

    # Set counting world env vars
    env = os.environ.copy()
    env["COUNTING_BLOB_MIN"] = str(n_blobs_min)
    env["COUNTING_BLOB_MAX"] = str(n_blobs_max)
    env["COUNTING_STAGE"] = str(stage)
    env["COUNTING_CONSERVATION"] = str(conservation).lower()
    env["COUNTING_ACTION_SPACE"] = action_space

    # PYTHONPATH: our scripts/ dir first so local models.py/dreamer.py shadow originals
    env["PYTHONPATH"] = str(BRIDGE_SCRIPTS) + ":" + env.get("PYTHONPATH", "")
    # DREAMER_DIR: so our patched dreamer.py can find configs.yaml
    env["DREAMER_DIR"] = str(DREAMER_DIR)

    # Build DreamerV3 command
    # NM512/dreamerv3-torch: --configs loads presets from configs.yaml (defaults always included),
    # then remaining --key value pairs override individual settings.
    cmd = [
        sys.executable, str(dreamer_script),
        "--configs", "counting_continuous",
        "--logdir", str(logdir),
        "--steps", str(total_steps),
        "--compile", "False",
        "--device", "mps",
    ]

    # --- Displacement loss params from job YAML ---
    disp = params.get("displacement_loss", {})
    if disp.get("enabled", False):
        cmd.extend(["--disp_lambda", str(disp.get("lambda", 0.1))])
        cmd.extend(["--disp_ema_decay", str(disp.get("ema_decay", 0.99))])
        cmd.extend(["--disp_warmup", str(disp.get("warmup", 50))])

    print(f"Command: {' '.join(cmd)}")
    print(f"Env: COUNTING_BLOB_MIN={n_blobs_min} COUNTING_BLOB_MAX={n_blobs_max}")
    print()
    sys.stdout.flush()

    # Write initial progress
    start_time = time.time()
    write_progress({
        "step": 0,
        "total_steps": total_steps,
        "percent": 0.0,
        "elapsed_seconds": 0,
        "status": "starting",
    })

    # Start logdir monitor thread
    stop_flag = [False]
    monitor = Thread(target=monitor_logdir, args=(logdir, total_steps, stop_flag), daemon=True)
    monitor.start()

    # Launch DreamerV3 training
    process = subprocess.Popen(
        cmd,
        cwd=str(DREAMER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    last_step = 0
    last_progress_time = time.time()

    try:
        for line in process.stdout:
            # Pass through to our stdout (captured by server.py)
            print(line, end="", flush=True)

            # Parse for progress
            step, pct, metrics = parse_dreamer_output(line, total_steps)
            if step is not None and step > last_step:
                last_step = step
                elapsed = time.time() - start_time
                eta = (elapsed / step * (total_steps - step)) if step > 0 else 0

                progress = {
                    "step": step,
                    "total_steps": total_steps,
                    "percent": round(pct, 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1),
                    "status": "training",
                }
                if metrics:
                    progress.update(metrics)

                write_progress(progress)
                last_progress_time = time.time()

            # Even if we can't parse the output, write a heartbeat every 60s
            elif time.time() - last_progress_time > 60:
                elapsed = time.time() - start_time
                write_progress({
                    "step": last_step,
                    "total_steps": total_steps,
                    "percent": round(last_step / total_steps * 100, 1) if total_steps > 0 else 0,
                    "elapsed_seconds": round(elapsed, 1),
                    "status": "training (heartbeat)",
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
        print(f"Training complete!")
        print(f"  Steps: {last_step:,}")
        print(f"  Time:  {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Logdir: {logdir}")
        print("=" * 60)

        write_progress({
            "step": total_steps,
            "total_steps": total_steps,
            "percent": 100.0,
            "elapsed_seconds": round(elapsed, 1),
            "status": "done",
        })
    else:
        print(f"\nTraining failed with exit code {returncode}", flush=True)
        write_progress({
            "step": last_step,
            "total_steps": total_steps,
            "percent": round(last_step / total_steps * 100, 1),
            "elapsed_seconds": round(elapsed, 1),
            "status": f"failed (exit {returncode})",
        })
        sys.exit(returncode)


if __name__ == "__main__":
    main()
