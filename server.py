#!/usr/bin/env python3
"""
Anim GPU server. Watches for job files, runs them, saves results.
All brains live in the container. This just runs whitelisted scripts.

Usage:
    source ~/anim-ml/bin/activate
    python ~/anim-bridge/server.py
"""

import json
import os
import shutil
import subprocess
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

BRIDGE = Path.home() / "anim-bridge"
PENDING = BRIDGE / "queue" / "pending"
RUNNING = BRIDGE / "queue" / "running"
DONE = BRIDGE / "queue" / "done"
FAILED = BRIDGE / "queue" / "failed"
STATUS_DIR = BRIDGE / "artifacts" / "status"
LOG_DIR = BRIDGE / "artifacts" / "logs"

SCRIPTS_DIR = BRIDGE / "scripts"
PROJECT_DIR = Path.home() / "anim-training"

ALLOWED_COMMANDS = {
    "train": {
        "script": "train.py",
        "cwd": str(SCRIPTS_DIR),
        "default_timeout": 86400,
    },
    "eval_dump": {
        "script": "eval_dump.py",
        "cwd": str(PROJECT_DIR),
        "default_timeout": 3600,
    },
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_status(job_id, data):
    """Write status JSON for a job and update latest.json."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATUS_DIR / f"{job_id}.json"
    with open(status_path, "w") as f:
        json.dump(data, f, indent=2)
    # Update latest.json
    shutil.copy2(str(status_path), str(STATUS_DIR / "latest.json"))


def run_job(job_path: Path):
    """Execute a single job file."""
    if job_path.suffix == ".tmp":
        return

    try:
        with open(job_path) as f:
            job = yaml.safe_load(f)
    except Exception as e:
        print(f"  Warning: couldn't parse {job_path.name}: {e}")
        return

    if not job:
        return

    job_id = job.get("id", job_path.stem)
    job_type = job.get("type")
    params = job.get("params", {}) or {}

    if job_type not in ALLOWED_COMMANDS:
        print(f"[{datetime.now():%H:%M:%S}] Rejected job {job_id}: unknown type '{job_type}'")
        # Move to failed without running
        running_path = RUNNING / job_path.name
        shutil.move(str(job_path), str(running_path))
        fail_job(running_path, job_id, job_type, f"Unknown job type: {job_type}")
        return

    spec = ALLOWED_COMMANDS[job_type]
    started_at = now_iso()

    print(f"[{datetime.now():%H:%M:%S}] Running job {job_id} (type: {job_type})")

    # Move to running
    running_path = RUNNING / job_path.name
    shutil.move(str(job_path), str(running_path))

    # Write running status
    write_status(job_id, {
        "id": job_id,
        "status": "running",
        "started_at": started_at,
        "type": job_type,
    })

    # Resolve script path
    cwd = Path(spec["cwd"])
    script_path = cwd / spec["script"]
    if not script_path.exists():
        fail_job(running_path, job_id, job_type,
                 f"Script not found: {script_path}", started_at=started_at)
        return

    # Environment
    env = os.environ.copy()
    env["ANIM_BRIDGE"] = str(BRIDGE)
    env["ANIM_ARTIFACTS"] = str(BRIDGE / "artifacts")
    env["JOB_ID"] = job_id

    if params:
        params_path = RUNNING / f"{job_id}_params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)
        env["JOB_PARAMS"] = str(params_path)

    timeout = params.get("timeout", spec["default_timeout"])

    # Stream stdout+stderr to a live log file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    live_log_path = LOG_DIR / f"{job_id}_live.log"

    try:
        with open(live_log_path, "w", buffering=1) as log_file:
            process = subprocess.Popen(
                ["python", str(script_path)],
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                fail_job(running_path, job_id, job_type,
                         f"Timeout after {timeout}s", started_at=started_at,
                         live_log_path=live_log_path)
                return

        finished_at = now_iso()

        if returncode == 0:
            # Collect output files
            outputs = collect_outputs(job_id)

            write_status(job_id, {
                "id": job_id,
                "status": "done",
                "started_at": started_at,
                "finished_at": finished_at,
                "type": job_type,
                "outputs": outputs,
            })
            print(f"[{datetime.now():%H:%M:%S}] Job {job_id} completed successfully")
            shutil.move(str(running_path), str(DONE / job_path.name))
        else:
            # Read tail of log for error context
            stderr_tail = read_log_tail(live_log_path, 10)
            error_msg = f"Exit code {returncode}"

            for line in stderr_tail[-5:]:
                print(f"  | {line}")

            fail_job(running_path, job_id, job_type, error_msg,
                     started_at=started_at, stderr_tail=stderr_tail)

    except Exception as e:
        fail_job(running_path, job_id, job_type, str(e), started_at=started_at)

    # Clean up params file
    params_path = RUNNING / f"{job_id}_params.yaml"
    if params_path.exists():
        params_path.unlink()


def collect_outputs(job_id):
    """Find any output files this job created in artifacts/."""
    outputs = []
    artifacts = BRIDGE / "artifacts"
    for subdir in ["tensors", "checkpoints", "plots"]:
        job_dir = artifacts / subdir / job_id
        if job_dir.exists():
            for f in sorted(job_dir.rglob("*")):
                if f.is_file():
                    outputs.append(str(f.relative_to(BRIDGE)))
    return outputs


def read_log_tail(log_path, n_lines):
    """Read the last n lines of a log file."""
    try:
        with open(log_path) as f:
            lines = f.read().strip().splitlines()
        return lines[-n_lines:]
    except Exception:
        return []


def fail_job(job_path, job_id, job_type, reason, started_at=None,
             stderr_tail=None, live_log_path=None):
    """Move job to failed and write structured status."""
    print(f"  FAILED: {reason}")

    if stderr_tail is None and live_log_path:
        stderr_tail = read_log_tail(live_log_path, 10)

    status = {
        "id": job_id,
        "status": "failed",
        "type": job_type,
        "finished_at": now_iso(),
        "error": reason,
    }
    if started_at:
        status["started_at"] = started_at
    if stderr_tail:
        status["stderr_tail"] = stderr_tail

    write_status(job_id, status)

    if job_path.exists():
        shutil.move(str(job_path), str(FAILED / job_path.name))


def main():
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Anim GPU Server")
    print("=" * 50)
    print(f"  Watching:   {PENDING}")
    print(f"  Artifacts:  {BRIDGE / 'artifacts'}")
    print(f"  Status:     {STATUS_DIR}")
    print(f"  Logs:       {LOG_DIR}")
    print(f"  Job types:  {', '.join(ALLOWED_COMMANDS.keys())}")
    print(f"\nPress Ctrl+C to stop\n")

    while True:
        pending = sorted(PENDING.glob("*.yaml"))
        for job_path in pending:
            run_job(job_path)
        time.sleep(3)


if __name__ == "__main__":
    main()
