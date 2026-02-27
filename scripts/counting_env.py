"""
Counting World Gymnasium Environment
=====================================
A gym.Env wrapper around the headless TypeScript counting world simulation.
Communicates with a Node.js subprocess via stdin/stdout JSON lines.

Observation: Box(80,) float32 — bot position, blob positions (25 slots), mark status, count, phase
Action: Discrete(25) — predicted tally (0-24), only meaningful when phase=1

Compatible with gym 0.26.x (old API) as required by NM512/dreamerv3-torch.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

import gym
import gym.spaces
import numpy as np


# Locate the compiled bridge script
# Check env var first (set by train.py), then fall back to original location
_PKG_DIR = Path(__file__).resolve().parent
_BRIDGE_SCRIPT = Path(os.environ.get(
    "COUNTING_BRIDGE_SCRIPT",
    str(Path.home() / "anim-training" / "env" / "dist" / "bridge.js"),
))

OBS_SIZE = 80
MAX_TALLY = 25  # Discrete action space: predict 0..24


class CountingWorldEnv(gym.Env):
    """
    Counting World RL Environment.

    The agent observes a bot counting blobs and predicts the final tally.

    Config options:
        stage (int): 1=marking, 2=confused, 3=random(marking/confused), 4=+organizing/grid, 5=all
        conservation (bool): True=blob count fixed, False=blobs may appear/vanish
        blob_count_min (int): Minimum blobs per episode (default 6)
        blob_count_max (int): Maximum blobs per episode (default 20)
        max_steps (int): Safety truncation limit (default 2000)
    """

    metadata = {"render.modes": []}

    def __init__(self, **kwargs):
        super().__init__()

        self.stage = kwargs.get("stage", 1)
        self.conservation = kwargs.get("conservation", True)
        self.blob_count_min = kwargs.get("blob_count_min", 6)
        self.blob_count_max = kwargs.get("blob_count_max", 25)
        self.max_steps = kwargs.get("max_steps", 5000)
        self.bidirectional = kwargs.get("bidirectional", False)

        # Observation: 80 float values
        # Positions normalized to [0,1], marks are 0/1, count is raw int, phase 0/1
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float(self.blob_count_max),  # blob count can be up to max
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        # Action: predict tally 0..24
        self.action_space = gym.spaces.Discrete(MAX_TALLY)

        # State
        self._proc = None
        self._current_obs = None
        self._episode_info = {}
        self._start_bridge()

    def _start_bridge(self):
        """Launch the Node.js bridge subprocess."""
        self._proc = subprocess.Popen(
            ["node", str(_BRIDGE_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Read the ready message
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Bridge failed to start: {stderr}")
        ready = json.loads(ready_line.strip())
        if ready.get("status") != "ready":
            raise RuntimeError(f"Bridge returned unexpected ready message: {ready}")

    def _send(self, cmd: dict) -> dict:
        """Send a command to the bridge and return the response."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Bridge process is not running")
        line = json.dumps(cmd) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Bridge returned no response. stderr: {stderr}")
        return json.loads(response_line.strip())

    def reset(self):
        """Reset the environment. Returns initial observation (gym 0.26 API)."""
        config = {
            "stage": self.stage,
            "conservation": self.conservation,
            "blobCountMin": self.blob_count_min,
            "blobCountMax": self.blob_count_max,
            "maxSteps": self.max_steps,
            "bidirectional": self.bidirectional,
        }
        result = self._send({"cmd": "reset", "config": config})
        if "error" in result:
            raise RuntimeError(f"Reset failed: {result['error']}")
        self._current_obs = np.array(result["obs"], dtype=np.float32)
        return self._current_obs

    def step(self, action):
        """
        Step the environment.

        During counting phase (phase=0): action is ignored, simulation advances.
        During predict phase (phase=1): action is the predicted tally, reward computed.

        Returns: (obs, reward, done, info) — gym 0.26 API (4 values).
        """
        action_val = int(action) if action is not None else None

        # If currently in predict phase, send the action as prediction
        # Otherwise send null (action ignored during counting)
        phase_indicator = self._current_obs[OBS_SIZE - 1] if self._current_obs is not None else 0
        send_action = action_val if phase_indicator == 1.0 else None

        result = self._send({"cmd": "step", "action": send_action})
        if "error" in result:
            raise RuntimeError(f"Step failed: {result['error']}")

        obs = np.array(result["obs"], dtype=np.float32)
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        info = result.get("info", {})

        self._current_obs = obs
        self._episode_info = info

        return obs, reward, done, info

    def close(self):
        """Shut down the bridge subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None

    def __del__(self):
        self.close()


# Allow running directly for quick smoke test
if __name__ == "__main__":
    env = CountingWorldEnv(stage=1, conservation=True)
    obs = env.reset()
    print(f"Reset: obs shape={obs.shape}, blob_count={obs[OBS_SIZE - 2]:.0f}")

    total_steps = 0
    done = False
    while not done:
        obs, reward, done, info = env.step(0)
        total_steps += 1

    print(f"Episode done after {total_steps} steps")
    print(f"  Bot tally: {info.get('bot_tally')}")
    print(f"  Blob count: {info.get('blob_count_start')}")
    print(f"  Bot type: {info.get('bot_type')}")
    print(f"  Reward: {reward}")

    env.close()
    print("Smoke test passed!")
