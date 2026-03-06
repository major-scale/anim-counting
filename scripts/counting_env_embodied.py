"""
Embodied Counting World Environment
====================================
DreamerV3 controls the bot via a steering angle action. The agent must navigate
to blobs and pick them up (proximity-based), then predict the count.

This is a NEW experiment — no existing files are modified.

Observation: Box(82,) float32  (same layout as passive observer)
Action: steering angle in radians (from wrapper: [-1,1] → [-π, π])

Reuses data structures and helpers from counting_env_pure.py.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import gym
import gym.spaces
import numpy as np

# Reuse from passive observer env
from counting_env_pure import (
    WORLD_WIDTH, WORLD_HEIGHT, MARGIN, MIN_SEPARATION, MAX_BLOB_COUNT,
    BOUNDARY_MARGIN, OBS_SIZE, GRID_COLS, GRID_ROWS, ANIM_SPEED,
    Blob, GridState,
    _create_grid, _create_target_grid,
    _start_blob_transition, _update_blob_animations,
    _count_blob_to_grid,
    _scattered_positions, _find_scatter_position,
    _random_arrangement, _generate_arrangement,
    _dist, _len, _normalize,
    _get_projection_matrix, _get_permutation_order,
)


# =====================================================================
# Constants
# =====================================================================

BOT_SPEED = 8.0           # Fixed speed (units/step)
PICKUP_RADIUS = 60.0      # Auto-pickup range
DEFAULT_MAX_STEPS = 8000   # Longer episodes for exploration


# =====================================================================
# Embodied Bot (simple vehicle — no waypoints, no personality)
# =====================================================================

@dataclass
class EmbodiedBot:
    pos_x: float
    pos_y: float
    vel_x: float = 0.0
    vel_y: float = 0.0
    heading: float = 0.0   # Current heading in radians


# =====================================================================
# Episode State
# =====================================================================

@dataclass
class EmbodiedEpisodeState:
    bot: EmbodiedBot
    blobs: list             # list of Blob
    grid: GridState
    phase: str              # "gathering" or "predict"
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    bot_total_distance: float = 0.0
    prev_bot_x: float = 0.0
    prev_bot_y: float = 0.0
    blob_count_at_start: int = 0
    arrangement_type: str = ""
    reward: float = 0.0
    blobs_gathered: int = 0
    # Ablation flags
    random_project: bool = False
    random_permute: bool = False


# =====================================================================
# Observation vector (same 82-dim layout as passive observer)
# =====================================================================

def _get_observation_embodied(state: EmbodiedEpisodeState) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    bot = state.bot
    blobs = state.blobs
    grid = state.grid

    # [0-1]: bot position (normalized)
    obs[0] = bot.pos_x / WORLD_WIDTH
    obs[1] = bot.pos_y / WORLD_HEIGHT

    # [2]: bot state — 0=active gathering, 2=all gathered
    if state.phase == "predict":
        obs[2] = 2.0
    else:
        obs[2] = 0.0

    # [3-52]: blob positions (padded to MAX_BLOB_COUNT=25, 2 dims each)
    blob_offset = 3
    for i in range(min(len(blobs), MAX_BLOB_COUNT)):
        obs[blob_offset + i * 2] = blobs[i].pos_x / WORLD_WIDTH
        obs[blob_offset + i * 2 + 1] = blobs[i].pos_y / WORLD_HEIGHT

    # [53-77]: grid slot assignments
    slot_offset = blob_offset + MAX_BLOB_COUNT * 2  # 53
    max_slots = GRID_COLS * GRID_ROWS
    for i in range(min(len(blobs), MAX_BLOB_COUNT)):
        if blobs[i].grid_slot is not None:
            obs[slot_offset + i] = (blobs[i].grid_slot + 1) / max_slots
        else:
            obs[slot_offset + i] = 0.0

    # [78]: actual blob count
    obs[slot_offset + MAX_BLOB_COUNT] = len(blobs)

    # [79]: phase indicator — 0=gathering, 1=predict
    if state.phase == "predict":
        obs[slot_offset + MAX_BLOB_COUNT + 1] = 1.0
    else:
        obs[slot_offset + MAX_BLOB_COUNT + 1] = 0.0

    # [80]: grid filled normalized
    obs[slot_offset + MAX_BLOB_COUNT + 2] = grid.filled_count / MAX_BLOB_COUNT

    # [81]: grid filled raw
    obs[slot_offset + MAX_BLOB_COUNT + 3] = grid.filled_count

    # Ablation: random projection
    if state.random_project:
        obs = _get_projection_matrix() @ obs

    # Ablation: random permutation
    if state.random_permute:
        obs = obs[_get_permutation_order()]

    return obs


# =====================================================================
# Core: reset and step
# =====================================================================

def _reset_embodied(blob_count_min, blob_count_max, max_steps,
                    target_arrangement="grid",
                    random_project=False, random_permute=False) -> EmbodiedEpisodeState:
    blob_count = blob_count_min + int(random.random() * (blob_count_max - blob_count_min + 1))

    # Arrangement in field zone (left 55%)
    arrangement_type = _random_arrangement()
    field_width = int(WORLD_WIDTH * 0.55)
    positions = _generate_arrangement(
        blob_count, field_width, WORLD_HEIGHT, MARGIN, MIN_SEPARATION, arrangement_type
    )

    # Create blobs
    blobs = []
    for i, (px, py) in enumerate(positions):
        blobs.append(Blob(
            id=i,
            pos_x=px, pos_y=py,
            field_pos_x=px, field_pos_y=py,
        ))

    # Create target grid
    grid = _create_target_grid(target_arrangement, blob_count)

    # Random bot start in field area
    start_x = MARGIN + random.random() * (field_width - 2 * MARGIN)
    start_y = MARGIN + random.random() * (WORLD_HEIGHT - 2 * MARGIN)
    heading = random.random() * 2 * math.pi

    bot = EmbodiedBot(
        pos_x=start_x, pos_y=start_y,
        heading=heading,
    )

    state = EmbodiedEpisodeState(
        bot=bot,
        blobs=blobs,
        grid=grid,
        phase="gathering",
        blob_count_at_start=len(blobs),
        arrangement_type=arrangement_type,
        prev_bot_x=start_x,
        prev_bot_y=start_y,
        random_project=random_project,
        random_permute=random_permute,
    )
    return state


def _step_embodied(state: EmbodiedEpisodeState, steering_angle: float,
                   max_steps: int) -> dict:
    """Advance one step with agent-controlled steering.

    Args:
        steering_angle: direction in radians (agent's chosen heading)
        max_steps: episode truncation limit
    """
    # Predict phase: terminal — no more stepping
    if state.phase == "predict":
        state.done = True
        state.reward = 0.0
        return {
            "obs": _get_observation_embodied(state),
            "reward": 0.0,
            "done": True,
            "info": _build_embodied_info(state),
        }

    state.step_count += 1

    # Update blob animations (deferred count-at-placement)
    _update_blob_animations(state.blobs, state.grid)

    bot = state.bot

    # --- Move bot using agent's steering angle ---
    bot.heading = steering_angle
    bot.vel_x = BOT_SPEED * math.cos(steering_angle)
    bot.vel_y = BOT_SPEED * math.sin(steering_angle)
    bot.pos_x += bot.vel_x
    bot.pos_y += bot.vel_y

    # Boundary clamping (keep in world bounds)
    bot.pos_x = max(BOUNDARY_MARGIN, min(WORLD_WIDTH - BOUNDARY_MARGIN, bot.pos_x))
    bot.pos_y = max(BOUNDARY_MARGIN, min(WORLD_HEIGHT - BOUNDARY_MARGIN, bot.pos_y))

    # --- Proximity pickup: auto-collect nearest uncounted blob ---
    pickup_happened = False
    for i, blob in enumerate(state.blobs):
        if blob.grid_slot is not None:
            continue  # already on grid
        if blob.pending_grid_placement:
            continue  # already flying to grid
        d = _dist(bot.pos_x, bot.pos_y, blob.pos_x, blob.pos_y)
        if d < PICKUP_RADIUS:
            placed = _count_blob_to_grid(blob, i, state.grid, "embodied", "#34D399")
            if placed:
                pickup_happened = True
                state.blobs_gathered += 1
                break  # one pickup per step

    # Track distance
    dx = bot.pos_x - state.prev_bot_x
    dy = bot.pos_y - state.prev_bot_y
    state.bot_total_distance += math.sqrt(dx * dx + dy * dy)
    state.prev_bot_x = bot.pos_x
    state.prev_bot_y = bot.pos_y

    # Phase transition: all blobs placed → predict
    all_placed = all(
        blob.grid_slot is not None or blob.pending_grid_placement
        for blob in state.blobs
    )
    if all_placed and state.phase == "gathering":
        # Force-finalize any blobs still flying
        for blob in state.blobs:
            if blob.pending_grid_placement:
                blob.pos_x = blob.anim_to_x
                blob.pos_y = blob.anim_to_y
                blob.animating = False
                blob.anim_progress = 0.0
                blob.pending_grid_placement = False
                state.grid.filled_count += 1
        state.phase = "predict"

    # Truncation
    if state.step_count >= max_steps:
        state.truncated = True
        state.done = True
        return {
            "obs": _get_observation_embodied(state),
            "reward": 0.0,
            "done": True,
            "info": _build_embodied_info(state),
        }

    return {
        "obs": _get_observation_embodied(state),
        "reward": 0.0,  # reward shaping done in wrapper
        "done": False,
        "info": _build_embodied_info(state),
    }


def _build_embodied_info(state: EmbodiedEpisodeState) -> dict:
    return {
        "blob_count_start": state.blob_count_at_start,
        "blob_count_end": len(state.blobs),
        "episode_length": state.step_count,
        "bot_distance": state.bot_total_distance,
        "arrangement_type": state.arrangement_type,
        "phase": state.phase,
        "truncated": state.truncated,
        "grid_filled": state.grid.filled_count,
        "blobs_gathered": state.blobs_gathered,
    }


# =====================================================================
# Gym Environment
# =====================================================================

class EmbodiedCountingWorldEnv(gym.Env):
    """Embodied counting world — agent controls steering."""

    metadata = {"render.modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.blob_count_min = kwargs.get("blob_count_min", 3)
        self.blob_count_max = kwargs.get("blob_count_max", 25)
        self.max_steps = kwargs.get("max_steps", DEFAULT_MAX_STEPS)
        self.target_arrangement = kwargs.get("target_arrangement", "grid")
        self.random_project = kwargs.get("random_project", False)
        self.random_permute = kwargs.get("random_permute", False)

        self.observation_space = gym.spaces.Box(
            low=-50.0, high=50.0,
            shape=(OBS_SIZE,), dtype=np.float32,
        )
        # Steering angle: [-π, π]
        self.action_space = gym.spaces.Box(
            low=-math.pi, high=math.pi,
            shape=(1,), dtype=np.float32,
        )

        self._state: Optional[EmbodiedEpisodeState] = None
        self._current_obs: Optional[np.ndarray] = None

    def reset(self):
        self._state = _reset_embodied(
            self.blob_count_min, self.blob_count_max,
            self.max_steps,
            self.target_arrangement,
            self.random_project,
            self.random_permute,
        )
        self._current_obs = _get_observation_embodied(self._state)
        return self._current_obs

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        steering = float(action[0]) if hasattr(action, '__len__') else float(action)

        result = _step_embodied(self._state, steering, self.max_steps)
        self._current_obs = result["obs"]
        return result["obs"], result["reward"], result["done"], result["info"]

    def close(self):
        self._state = None


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    env = EmbodiedCountingWorldEnv(
        blob_count_min=10, blob_count_max=10,
        max_steps=5000,
    )

    import time
    t0 = time.time()
    n_episodes = 5
    total_steps = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_steps = 0
        while not done:
            # Random steering angle
            angle = random.uniform(-math.pi, math.pi)
            obs, reward, done, info = env.step([angle])
            ep_steps += 1
        total_steps += ep_steps
        print(f"  Episode {ep+1}: {ep_steps} steps, "
              f"grid_filled={info['grid_filled']}/{info['blob_count_start']}, "
              f"phase={info['phase']}, "
              f"distance={info['bot_distance']:.0f}, "
              f"gathered={info['blobs_gathered']}")

    elapsed = time.time() - t0
    print(f"\n{total_steps} steps in {elapsed:.2f}s = {total_steps/elapsed:.0f} steps/sec")

    # Verify obs shape
    obs = env.reset()
    assert obs.shape == (OBS_SIZE,), f"Expected ({OBS_SIZE},), got {obs.shape}"
    print(f"Obs shape: {obs.shape} ✓")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    env.close()
    print("Smoke test passed!")
