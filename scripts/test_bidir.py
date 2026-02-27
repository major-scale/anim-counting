#!/usr/bin/env python3
"""Quick test: verify bidirectional episodes produce both +1 and -1 mark transitions."""

import json
import subprocess
import sys
from pathlib import Path

BRIDGE_SCRIPT = Path(__file__).parent.parent / "dist" / "bridge.js"

def run_episode():
    """Run one bidirectional episode and track mark deltas."""
    proc = subprocess.Popen(
        ["node", str(BRIDGE_SCRIPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    # Read ready
    ready = json.loads(proc.stdout.readline().strip())
    assert ready["status"] == "ready", f"Unexpected: {ready}"

    # Reset with bidirectional=true
    config = {
        "stage": 1, "conservation": True,
        "blobCountMin": 6, "blobCountMax": 10,
        "maxSteps": 5000, "bidirectional": True,
    }
    proc.stdin.write(json.dumps({"cmd": "reset", "config": config}) + "\n")
    proc.stdin.flush()
    result = json.loads(proc.stdout.readline().strip())
    obs = result["obs"]

    MARK_START = 53
    MARK_END = 78
    prev_marks = sum(obs[MARK_START:MARK_END])
    blob_count = int(obs[78])

    plus_transitions = 0
    minus_transitions = 0
    max_marks = 0
    steps = 0
    phases_seen = set()

    done = False
    while not done:
        proc.stdin.write(json.dumps({"cmd": "step", "action": 0}) + "\n")
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline().strip())

        obs = result["obs"]
        done = result.get("done", False)
        info = result.get("info", {})

        cur_marks = sum(obs[MARK_START:MARK_END])
        phase = obs[79]
        phases_seen.add(round(phase, 1))

        delta = cur_marks - prev_marks
        if delta == 1:
            plus_transitions += 1
        elif delta == -1:
            minus_transitions += 1

        max_marks = max(max_marks, cur_marks)
        prev_marks = cur_marks
        steps += 1

    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
    proc.stdin.flush()
    proc.terminate()

    return {
        "blob_count": blob_count,
        "plus_transitions": plus_transitions,
        "minus_transitions": minus_transitions,
        "max_marks": max_marks,
        "steps": steps,
        "phases_seen": phases_seen,
        "final_phase": info.get("phase"),
        "bidirectional": info.get("bidirectional"),
    }


if __name__ == "__main__":
    print("Running bidirectional episode test...")
    result = run_episode()

    print(f"\nResults:")
    print(f"  Blob count:        {result['blob_count']}")
    print(f"  +1 transitions:    {result['plus_transitions']}")
    print(f"  -1 transitions:    {result['minus_transitions']}")
    print(f"  Max marks:         {result['max_marks']}")
    print(f"  Episode steps:     {result['steps']}")
    print(f"  Phases seen:       {result['phases_seen']}")
    print(f"  Final phase:       {result['final_phase']}")
    print(f"  Bidirectional:     {result['bidirectional']}")

    # Assertions
    ok = True
    if result["plus_transitions"] == 0:
        print("\nFAIL: No +1 transitions detected!")
        ok = False
    if result["minus_transitions"] == 0:
        print("\nFAIL: No -1 transitions detected! Unmark phase not working.")
        ok = False
    if 0.5 not in result["phases_seen"]:
        print("\nFAIL: Unmarking phase (0.5) never seen in observation vector!")
        ok = False
    if result["plus_transitions"] != result["minus_transitions"]:
        print(f"\nWARN: Asymmetric transitions: +1={result['plus_transitions']}, -1={result['minus_transitions']}")
        print("  (Expected roughly equal for bidirectional episodes)")

    if ok:
        print("\nPASS: Bidirectional episodes produce both +1 and -1 transitions!")
    else:
        sys.exit(1)
