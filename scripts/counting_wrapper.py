"""
Counting World environment wrapper for DreamerV3.
Adapts the CountingWorldEnv (gym 0.26 flat-obs) to DreamerV3's Dict observation format.

Action mode controlled by COUNTING_ACTION_SPACE env var:
  "discrete" (default): Discrete(9) where action i -> tally i (0-8)
  "continuous": Box(-1, 1, (1,)) — DreamerV3 convention, rescaled to [0, 8]
"""
import os
import sys
import gym
import gym.spaces
import numpy as np

sys.path.insert(0, "~/anim-training/env")
from counting_env import CountingWorldEnv, OBS_SIZE

_DEFAULT_BLOB_MIN = 3
_DEFAULT_BLOB_MAX = 25

# Observation layout for shaping reward (25-blob layout, OBS_SIZE=80)
_MARK_START = 53
_MARK_END = 78


class CountingWorld:
    """DreamerV3-compatible wrapper around CountingWorldEnv."""

    metadata = {}

    def __init__(self, task, seed=0, **kwargs):
        conservation_env = os.environ.get("COUNTING_CONSERVATION", "true")
        conservation = conservation_env.lower() not in ("false", "0", "no")
        self._continuous = os.environ.get("COUNTING_ACTION_SPACE", "discrete") == "continuous"

        self._blob_min = int(os.environ.get("COUNTING_BLOB_MIN", _DEFAULT_BLOB_MIN))
        self._blob_max = int(os.environ.get("COUNTING_BLOB_MAX", _DEFAULT_BLOB_MAX))

        self._env = CountingWorldEnv(
            stage=int(os.environ.get("COUNTING_STAGE", "1")),
            conservation=conservation,
            blob_count_min=self._blob_min,
            blob_count_max=self._blob_max,
            max_steps=kwargs.get("max_steps", 5000),
        )
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "vector": gym.spaces.Box(
                low=0.0, high=float(self._blob_max), shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        })

    @property
    def action_space(self):
        if self._continuous:
            # DreamerV3 continuous actions live in [-1, 1]; we rescale in step()
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            num_actions = self._blob_max + 1
            space = gym.spaces.Discrete(num_actions)
            space.discrete = True
            return space

    def step(self, action):
        if self._continuous:
            # Rescale from [-1, 1] -> [0, blob_max]
            half = self._blob_max / 2.0
            raw_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            raw_prediction = np.clip((raw_val + 1.0) * half, 0.0, float(self._blob_max))
            tally = int(round(np.clip(raw_prediction, 0, self._blob_max)))
        else:
            raw_prediction = None
            tally = int(action)

        obs_flat, reward, done, info = self._env.step(tally)

        # Distance-based shaping reward using raw continuous prediction
        # (or discrete tally if in discrete mode)
        marked_so_far = int(np.sum(obs_flat[_MARK_START:_MARK_END]))
        pred_for_reward = raw_prediction if raw_prediction is not None else float(tally)
        error = abs(pred_for_reward - marked_so_far)
        reward += 1.0 - 0.25 * error

        # Store extra info for logging
        if self._continuous:
            info["raw_prediction"] = raw_prediction
            info["rounded_prediction"] = tally
            info["raw_error"] = error

        obs = {
            "vector": obs_flat,
            "is_first": False,
            "is_last": done,
            "is_terminal": info.get("discount", 1.0 - float(done)) == 0,
        }
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
