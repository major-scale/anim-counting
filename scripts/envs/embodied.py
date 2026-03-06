"""
Embodied Counting World wrapper for DreamerV3.
Agent controls steering (dim 0) AND predicts count (dim 1).

Action space: Box(-1, 1, shape=(2,))
  dim 0: steering angle, mapped [-1, 1] → [-π, π]
  dim 1: count prediction, mapped [-1, 1] → [0, blob_max]

Reward (balanced by phase):
  Gathering phase:
    - Approach shaping: (prev_dist - curr_dist) * 0.1  (~0.8/step when heading at blob)
    - Placement bonus: +2.0 per blob placed on grid
    - Completion bonus: +5.0 when all blobs gathered
    - Count tracking: (1.0 - 0.25 * |pred - filled|) * 0.05  (small signal, ~0.05/step)
  Predict phase:
    - Count prediction: 1.0 - 0.25 * |pred - filled|  (full weight, terminal step)

Env vars (EMBODIED_* namespace — won't collide with COUNTING_*):
  EMBODIED_BLOB_MIN, EMBODIED_BLOB_MAX (default 3, 25)
  EMBODIED_MAX_STEPS (default 8000)
  EMBODIED_RANDOM_PROJECT (default false)
  EMBODIED_RANDOM_PERMUTE (default false)
  EMBODIED_ARRANGEMENT (default grid)
"""

import math
import os

import gym
import gym.spaces
import numpy as np

from counting_env_embodied import EmbodiedCountingWorldEnv, OBS_SIZE

_DEFAULT_BLOB_MIN = 3
_DEFAULT_BLOB_MAX = 25
_DEFAULT_MAX_STEPS = 8000


def _env_bool(key, default="false"):
    return os.environ.get(key, default).lower() in ("true", "1", "yes")


class EmbodiedCountingWorld:
    """DreamerV3-compatible wrapper around EmbodiedCountingWorldEnv."""

    metadata = {}

    def __init__(self, task, seed=0, **kwargs):
        self._blob_min = int(os.environ.get("EMBODIED_BLOB_MIN", _DEFAULT_BLOB_MIN))
        self._blob_max = int(os.environ.get("EMBODIED_BLOB_MAX", _DEFAULT_BLOB_MAX))
        max_steps = int(os.environ.get("EMBODIED_MAX_STEPS", _DEFAULT_MAX_STEPS))
        arrangement = os.environ.get("EMBODIED_ARRANGEMENT", "grid")
        random_project = _env_bool("EMBODIED_RANDOM_PROJECT")
        random_permute = _env_bool("EMBODIED_RANDOM_PERMUTE")

        self._env = EmbodiedCountingWorldEnv(
            blob_count_min=self._blob_min,
            blob_count_max=self._blob_max,
            max_steps=max_steps,
            target_arrangement=arrangement,
            random_project=random_project,
            random_permute=random_permute,
        )
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

        # Track previous nearest distance for potential-based shaping
        self._prev_nearest_dist = 0.0
        self._prev_grid_filled = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "vector": gym.spaces.Box(
                low=-50.0, high=50.0, shape=(OBS_SIZE,), dtype=np.float32,
            ),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        })

    @property
    def action_space(self):
        # dim 0: steering, dim 1: count prediction — both in [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _nearest_uncounted_dist(self):
        """Distance from bot to nearest uncounted blob."""
        state = self._env._state
        if state is None:
            return 0.0
        bot = state.bot
        min_d = float("inf")
        for blob in state.blobs:
            if blob.grid_slot is not None or blob.pending_grid_placement:
                continue
            d = math.sqrt((bot.pos_x - blob.pos_x)**2 + (bot.pos_y - blob.pos_y)**2)
            if d < min_d:
                min_d = d
        return min_d if min_d < float("inf") else 0.0

    def step(self, action):
        # Decode action
        raw_steering = float(action[0])  # [-1, 1]
        raw_count = float(action[1])     # [-1, 1]

        # Map steering: [-1, 1] → [-π, π]
        steering_angle = raw_steering * math.pi

        # Map count prediction: [-1, 1] → [0, blob_max]
        half = self._blob_max / 2.0
        count_prediction = np.clip((raw_count + 1.0) * half, 0.0, float(self._blob_max))

        # Capture pre-step phase for transition detection
        pre_phase = self._env._state.phase if self._env._state else "gathering"

        # Step inner env with steering
        obs_flat, inner_reward, done, info = self._env.step([steering_angle])

        # --- Reward composition (balanced by phase) ---
        state = self._env._state
        grid_filled = state.grid.filled_count if state else 0

        r_approach = 0.0
        r_placement = 0.0
        r_completion = 0.0
        r_count = 0.0

        error = abs(count_prediction - grid_filled)

        if state and pre_phase == "gathering":
            picked_up = grid_filled > self._prev_grid_filled

            # 1. Approach shaping: scale 0.1 → ~0.8/step when heading at blob
            #    Skip on pickup steps — nearest blob jumps to a farther one,
            #    causing a large negative spike. Placement bonus covers those steps.
            nearest_dist = self._nearest_uncounted_dist()
            if not picked_up and self._prev_nearest_dist > 0 and nearest_dist > 0:
                delta = (self._prev_nearest_dist - nearest_dist) * 0.1
                r_approach = max(-1.0, min(1.0, delta))  # clamp for safety
            self._prev_nearest_dist = nearest_dist

            # 2. Placement bonus: +2.0 per blob on grid
            if picked_up:
                r_placement = 2.0

            # 3. All-gathered bonus (phase just transitioned to predict)
            if state.phase == "predict":
                r_completion = 5.0

            # 4. Count tracking: small signal during gathering (5% weight)
            r_count = (1.0 - 0.25 * error) * 0.05

        elif state and pre_phase == "predict":
            # Terminal count prediction: full weight
            r_count = 1.0 - 0.25 * error

        reward = r_approach + r_placement + r_completion + r_count
        self._prev_grid_filled = grid_filled

        # Log all components for reward balance monitoring
        info["raw_steering"] = raw_steering
        info["steering_angle"] = steering_angle
        info["count_prediction"] = count_prediction
        info["count_error"] = error
        info["r_approach"] = r_approach
        info["r_placement"] = r_placement
        info["r_completion"] = r_completion
        info["r_count"] = r_count
        info["r_total"] = reward

        obs = {
            "vector": obs_flat,
            "is_first": False,
            "is_last": done,
            "is_terminal": info.get("discount", 1.0 - float(done)) == 0,
        }
        return obs, reward, done, info

    def reset(self):
        obs_flat = self._env.reset()
        self._prev_nearest_dist = self._nearest_uncounted_dist()
        self._prev_grid_filled = 0
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
