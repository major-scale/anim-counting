"""
Multi-Dimensional Counting World wrapper for DreamerV3.
Passive observer variant — bot follows hardcoded navigation, agent only predicts count.

Action space: Box(-1, 1, shape=(1,)) — count prediction mapped to [0, blob_max]
Observation: Dict with vector(128,), is_first, is_last, is_terminal

Env vars (MULTIDIM_* namespace):
  MULTIDIM_BLOB_MIN, MULTIDIM_BLOB_MAX (default 3, 25)
  MULTIDIM_MAX_STEPS (default 6000)
  MULTIDIM_PROJ_DIM (default 128)
  MULTIDIM_FIXED_DIM (default: None = sample from distribution)
  MULTIDIM_APPLY_SYMLOG (default true)
"""

import math
import os

import gym
import gym.spaces
import numpy as np

from counting_env_multidim import (
    MultiDimCountingWorldEnv, PROJ_DIM, DEFAULT_MAX_STEPS,
)

_DEFAULT_BLOB_MIN = 3
_DEFAULT_BLOB_MAX = 13


def _env_bool(key, default="true"):
    return os.environ.get(key, default).lower() in ("true", "1", "yes")


class MultiDimCountingWorld:
    """DreamerV3-compatible wrapper around MultiDimCountingWorldEnv."""

    metadata = {}

    def __init__(self, task, seed=0, **kwargs):
        self._blob_min = int(os.environ.get("MULTIDIM_BLOB_MIN", _DEFAULT_BLOB_MIN))
        self._blob_max = int(os.environ.get("MULTIDIM_BLOB_MAX", _DEFAULT_BLOB_MAX))
        max_steps = int(os.environ.get("MULTIDIM_MAX_STEPS", DEFAULT_MAX_STEPS))
        proj_dim = int(os.environ.get("MULTIDIM_PROJ_DIM", PROJ_DIM))
        apply_symlog = _env_bool("MULTIDIM_APPLY_SYMLOG", "true")

        # Fixed dim for evaluation (None = sample from distribution)
        fixed_dim_str = os.environ.get("MULTIDIM_FIXED_DIM", "")
        fixed_dim = int(fixed_dim_str) if fixed_dim_str else None

        self._env = MultiDimCountingWorldEnv(
            blob_count_min=self._blob_min,
            blob_count_max=self._blob_max,
            max_steps_base=max_steps,
            proj_dim=proj_dim,
            fixed_dim=fixed_dim,
            apply_symlog=apply_symlog,
        )
        self._proj_dim = proj_dim
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "vector": gym.spaces.Box(
                low=-50.0, high=50.0, shape=(self._proj_dim,), dtype=np.float32,
            ),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        # Map [-1,1] → [0, blob_max]
        raw = float(action[0])
        half = self._blob_max / 2.0
        count_prediction = np.clip((raw + 1.0) * half, 0.0, float(self._blob_max))

        # The inner env expects a discrete tally for the predict phase
        # During counting phase, action is ignored
        tally = int(round(count_prediction))

        obs_flat, reward, done, info = self._env.step(tally)

        # Zero action to RSSM — bot is hardcoded, action is noise
        # (Same as DREAMER_ZERO_ACTION in passive observer)

        # Count prediction shaping reward (same as passive observer)
        grid_filled = info.get("grid_filled", 0)
        error = abs(count_prediction - grid_filled)
        # Small shaping during counting, full at predict
        if info.get("phase") == "predict":
            reward = 1.0 - 0.25 * error
        else:
            reward = (1.0 - 0.25 * error) * 0.05

        obs = {
            "vector": obs_flat,
            "is_first": False,
            "is_last": done,
            "is_terminal": info.get("discount", 1.0 - float(done)) == 0,
        }

        info["count_prediction"] = count_prediction
        info["count_error"] = error

        return obs, reward, done, info

    def reset(self):
        obs_flat = self._env.reset()
        return {
            "vector": obs_flat,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def close(self):
        self._env.close()

    def render(self):
        return None
