"""
Multi-Dimensional Counting World Environment
=============================================
Parameterized counting environment where spatial dimensionality is a constructor
argument. During training, dimensionality is randomly sampled each episode.
All observations are projected to a fixed output dimensionality via random
orthogonal projection before the RSSM sees them.

Hypothesis: The counting manifold is dimension-independent. An agent trained
across 2D-10D should develop the same or cleaner counting manifold because it
can't rely on dimension-specific shortcuts.

Observation: Box(proj_dim,) float32, where proj_dim=128 by default.
Action: same as counting_env_pure.py (predicted tally, only at predict phase).

Compatible with gym 0.26.x (old API) as required by NM512/dreamerv3-torch.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import gym
import gym.spaces
import numpy as np

# Reuse constants and shared logic from counting_env_pure
from counting_env_pure import (
    MARGIN, MIN_SEPARATION, MAX_BLOB_COUNT, DT,
    GRID_COLS, GRID_ROWS, GRID_CELL_SIZE,
    ANIM_SPEED,
    BotPersonality, MARKING_BOT, ALL_PERSONALITIES,
    _ease_out_cubic,
    _select_personality,
)


# =====================================================================
# Multi-dim constants
# =====================================================================

PROJ_DIM = 128                      # Fixed output dimensionality
DEFAULT_WORLD_SCALE = 700.0         # v2: halved from 1400 for faster episodes
DEFAULT_MAX_STEPS = 2000            # v2: capped — faster bot + smaller world
DEFAULT_BOT_SPEED = 8.0             # v2: 4x faster (was 2.0)
DEFAULT_BOT_FORCE = 0.4             # v2: scaled with speed for similar turn radius
DEFAULT_SLOW_RADIUS = 80.0          # v2: scaled with speed
PICKUP_RADIUS_2D = 60.0             # v2: larger pickup (was 12) — ~90 at D=5
PLACE_RADIUS_2D = 60.0
MULTIDIM_MAX_BLOBS = 13             # v2: max blobs (was MAX_BLOB_COUNT=25)

# v2: Faster bot personality for multi-dim (4x speed, proportional force)
MARKING_BOT_FAST = BotPersonality(
    name="marking_fast", color="#34D399",
    max_speed=DEFAULT_BOT_SPEED, max_force=DEFAULT_BOT_FORCE,
    slow_radius=DEFAULT_SLOW_RADIUS, arrive_easing=1.8,
    wander_weight=0, wander_radius=0, wander_distance=0, wander_jitter=0,
    target_switch_probability=0, revisit_probability=0,
    target_selection_order="nearest",
    speed_variance=0.05,
    pause_duration=15, pause_variance=3,  # v2: shorter pause (was 30±5)
    anticipation_strength=0.3, follow_through_strength=0.2,
    separation_weight=0.5, separation_radius=50,
    waypoint_arrival_radius=PICKUP_RADIUS_2D,
)

# Dimensionality sampling weights (v2: capped at D=5, rebalanced)
DEFAULT_DIM_WEIGHTS = {
    2: 0.30,
    3: 0.25,
    4: 0.25,
    5: 0.20,
}


# =====================================================================
# N-dimensional vector operations
# =====================================================================

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _dist_nd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize_nd(v: np.ndarray) -> np.ndarray:
    d = _norm(v)
    if d < 1e-10:
        return np.zeros_like(v)
    return v / d


def _limit_nd(v: np.ndarray, max_val: float) -> np.ndarray:
    d = _norm(v)
    if d <= max_val:
        return v
    return _normalize_nd(v) * max_val


# =====================================================================
# Multi-dim Blob
# =====================================================================

@dataclass
class BlobND:
    id: int
    pos: np.ndarray            # D-dimensional position
    field_pos: np.ndarray      # Original position in field zone
    grid_slot: Optional[int] = None
    counted_by: set = field(default_factory=set)
    marked_by: Optional[str] = None
    # Animation
    anim_from: Optional[np.ndarray] = None
    anim_to: Optional[np.ndarray] = None
    anim_progress: float = 0.0
    animating: bool = False
    pending_grid_placement: bool = False


def _start_blob_transition_nd(blob: BlobND, target: np.ndarray):
    blob.anim_from = blob.pos.copy()
    blob.anim_to = target.copy()
    blob.anim_progress = 0.0
    blob.animating = True


def _update_blob_animations_nd(blobs: list, grid=None):
    for blob in blobs:
        if blob.animating:
            blob.anim_progress = min(1.0, blob.anim_progress + ANIM_SPEED * DT)
            t = _ease_out_cubic(blob.anim_progress)
            blob.pos = blob.anim_from + (blob.anim_to - blob.anim_from) * t
            if blob.anim_progress >= 1.0:
                blob.pos = blob.anim_to.copy()
                blob.animating = False
                blob.anim_progress = 0.0
                if blob.pending_grid_placement and grid is not None:
                    grid.filled_count += 1
                    blob.pending_grid_placement = False


# =====================================================================
# Multi-dim Grid
# =====================================================================

@dataclass
class GridStateND:
    slots: list                 # list of D-dimensional np.ndarray positions
    occupancy: list             # slot -> blob_id (-1 = empty)
    placement_order: list       # FILO stack
    filled_count: int = 0


def _create_grid_nd(D: int, count: int, world_scale: float) -> GridStateND:
    """Create a 1D line of grid slots in D-dimensional space.

    Slots are arranged along the first axis (dim 0) in the "grid zone"
    (right side of space). All other coordinates are centered.
    """
    n_slots = MULTIDIM_MAX_BLOBS  # v2: 13 (was 25)

    # Grid zone: right side of first dimension
    grid_start = world_scale * 0.65
    grid_end = world_scale * 0.95
    spacing = (grid_end - grid_start) / max(n_slots - 1, 1)

    center = world_scale / 2.0

    slots = []
    for i in range(n_slots):
        pos = np.full(D, center, dtype=np.float64)
        pos[0] = grid_start + i * spacing
        slots.append(pos)

    return GridStateND(
        slots=slots,
        occupancy=[-1] * n_slots,
        placement_order=[],
        filled_count=0,
    )


def _count_blob_to_grid_nd(blob: BlobND, blob_index: int, grid: GridStateND) -> bool:
    """Place a blob on the next available grid slot."""
    if blob.grid_slot is not None:
        return False

    for i, occ in enumerate(grid.occupancy):
        if occ == -1:
            grid.occupancy[i] = blob_index
            blob.grid_slot = i
            grid.placement_order.append(blob_index)
            # Start animation to grid slot
            _start_blob_transition_nd(blob, grid.slots[i])
            blob.pending_grid_placement = True
            return True
    return False


# =====================================================================
# Multi-dim Bot (passive observer — hardcoded navigation)
# =====================================================================

@dataclass
class BotND:
    pos: np.ndarray            # D-dimensional
    vel: np.ndarray            # D-dimensional
    personality: BotPersonality
    waypoints: list = field(default_factory=list)  # list of np.ndarray
    current_target_index: int = 0
    visited_indices: set = field(default_factory=set)
    all_visited: bool = False
    pause_timer: int = 0
    count_tally: int = 0
    wander_angle: float = 0.0


def _select_next_target_nd(bot: BotND) -> int:
    """Select the nearest unvisited waypoint."""
    wps = bot.waypoints
    vis = bot.visited_indices

    unvisited = [i for i in range(len(wps)) if i not in vis]
    if not unvisited:
        return bot.current_target_index

    best_i, best_d = unvisited[0], float("inf")
    for i in unvisited:
        d = _dist_nd(bot.pos, wps[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _update_bot_nd(bot: BotND, D: int, world_scale: float):
    """Passive observer steering in D dimensions."""
    p = bot.personality

    if bot.all_visited or bot.pause_timer > 0:
        if bot.pause_timer > 0:
            bot.pause_timer -= 1
        bot.vel *= 0.95  # friction
        bot.pos += bot.vel
        return

    if not bot.waypoints:
        return

    target = bot.waypoints[bot.current_target_index]

    # Arrive steering in N-D
    offset = target - bot.pos
    dist = _norm(offset)

    if dist < 0.5:
        # Brake
        bot.vel *= 0.8
        bot.pos += bot.vel
        return

    direction = offset / dist

    if dist < p.slow_radius:
        t = dist / p.slow_radius
        speed = p.max_speed * (t ** p.arrive_easing)
    else:
        speed = p.max_speed

    desired = direction * speed
    steer = desired - bot.vel
    steer = _limit_nd(steer, p.max_force)

    bot.vel += steer
    bot.vel = _limit_nd(bot.vel, p.max_speed)
    bot.pos += bot.vel

    # Boundary clamping
    margin = MARGIN
    for d in range(D):
        bot.pos[d] = np.clip(bot.pos[d], margin, world_scale - margin)


def _check_arrival_nd(bot: BotND, arrival_radius: float) -> bool:
    """Check if bot arrived at current waypoint."""
    if not bot.waypoints:
        return False
    target = bot.waypoints[bot.current_target_index]
    return _dist_nd(bot.pos, target) < arrival_radius


# =====================================================================
# Blob position generation in D dimensions
# =====================================================================

def _scattered_positions_nd(count: int, D: int, world_scale: float,
                            margin: float, min_sep: float) -> list:
    """Generate scattered positions in the field zone (first-dim < 0.55 * scale).

    v2: Spawning region scales as world_scale / sqrt(D) per non-primary axis
    to maintain comparable blob density across dimensionalities. Without this,
    13 blobs in a 700^5 hypervolume are absurdly sparse.
    """
    field_max = world_scale * 0.55
    # v2: Scale spawning region so nearest-neighbor density stays ~constant
    # Primary axis (dim 0) uses field_max as before
    # Other axes shrink by 1/sqrt(D) so effective density is D-independent
    spawn_scale = world_scale / math.sqrt(D) if D > 1 else world_scale
    center = world_scale / 2.0

    positions = []
    max_attempts = count * 200
    attempts = 0

    while len(positions) < count and attempts < max_attempts:
        pos = np.zeros(D, dtype=np.float64)
        pos[0] = margin + random.random() * (field_max - 2 * margin)
        for d in range(1, D):
            # Center blobs in a smaller region for high D
            half_range = (spawn_scale - 2 * margin) / 2.0
            pos[d] = center + (random.random() - 0.5) * 2.0 * half_range

        # Check separation
        ok = True
        for existing in positions:
            if _dist_nd(pos, existing) < min_sep:
                ok = False
                break
        if ok:
            positions.append(pos)
        attempts += 1

    # Fallback: fill remaining with random positions (ignore separation)
    while len(positions) < count:
        pos = np.zeros(D, dtype=np.float64)
        pos[0] = margin + random.random() * (field_max - 2 * margin)
        for d in range(1, D):
            half_range = (spawn_scale - 2 * margin) / 2.0
            pos[d] = center + (random.random() - 0.5) * 2.0 * half_range
        positions.append(pos)

    return positions


# =====================================================================
# Random orthogonal projection matrices (per dimensionality)
# =====================================================================

_PROJECTION_MATRICES: Dict[int, np.ndarray] = {}


def _obs_dim(D: int) -> int:
    """Compute raw observation dimensionality for spatial dimension D.

    v2 layout: bot_pos(D) + bot_state(1) + blob_positions(13*D) +
               grid_slots(13) + blob_count(1) + phase(1) + grid_filled_norm(1) + grid_filled_raw(1)
    = 14*D + 18
    """
    return (MULTIDIM_MAX_BLOBS + 1) * D + MULTIDIM_MAX_BLOBS + 5


def _get_projection_matrix_nd(D: int, proj_dim: int = PROJ_DIM) -> np.ndarray:
    """Get or create a random orthogonal projection R ∈ ℝ^{proj_dim × obs_dim}.

    Uses QR decomposition with Haar correction for non-square matrices.
    Deterministic seed per dimensionality: 50000 + D.
    """
    key = (D, proj_dim)
    if key not in _PROJECTION_MATRICES:
        obs_d = _obs_dim(D)
        rng = np.random.RandomState(50_000 + D)

        if obs_d == proj_dim:
            # Square: use scipy for proper Haar distribution
            from scipy.stats import ortho_group
            mat = ortho_group.rvs(obs_d, random_state=rng).astype(np.float32)
        elif obs_d < proj_dim:
            # obs_dim < proj_dim: pad with zeros (rare — only for D=2 if proj_dim>82)
            # Actually project to obs_dim first, then zero-pad
            from scipy.stats import ortho_group
            mat_sq = ortho_group.rvs(obs_d, random_state=rng).astype(np.float32)
            mat = np.zeros((proj_dim, obs_d), dtype=np.float32)
            mat[:obs_d, :] = mat_sq
        else:
            # obs_dim > proj_dim: generate Gaussian, QR with sign correction
            G = rng.randn(proj_dim, obs_d).astype(np.float64)
            Q, R = np.linalg.qr(G.T)  # Q is (obs_dim, proj_dim)
            # Sign correction for Haar distribution
            signs = np.sign(np.diag(R))
            Q = Q * signs[np.newaxis, :]
            mat = Q.T.astype(np.float32)  # (proj_dim, obs_dim)

        _PROJECTION_MATRICES[key] = mat
    return _PROJECTION_MATRICES[key]


# =====================================================================
# Episode state
# =====================================================================

@dataclass
class MultiDimEpisodeState:
    D: int                     # Spatial dimensionality this episode
    bot: BotND
    blobs: list                # list of BlobND
    grid: GridStateND
    phase: str                 # "counting", "unmarking", "predict"
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    bot_total_distance: float = 0.0
    prev_bot_pos: Optional[np.ndarray] = None
    blob_count_at_start: int = 0
    bot_type: str = ""
    reward: float = 0.0
    bidirectional: bool = False
    uncount_pause_timer: int = 0
    world_scale: float = DEFAULT_WORLD_SCALE


# =====================================================================
# Observation vector (raw, before projection)
# =====================================================================

def _get_observation_nd(state: MultiDimEpisodeState) -> np.ndarray:
    """Build raw observation vector of dimension 26*D + 30."""
    D = state.D
    obs_d = _obs_dim(D)
    obs = np.zeros(obs_d, dtype=np.float32)
    bot = state.bot
    blobs = state.blobs
    grid = state.grid
    scale = state.world_scale

    idx = 0

    # Bot position (D dims)
    for d in range(D):
        obs[idx] = bot.pos[d] / scale
        idx += 1

    # Bot state: 0=seeking, 1=paused, 2=done
    if bot.all_visited:
        obs[idx] = 2.0
    elif bot.pause_timer > 0:
        obs[idx] = 1.0
    else:
        obs[idx] = 0.0
    idx += 1

    # Blob positions (padded to MULTIDIM_MAX_BLOBS, D dims each)
    for i in range(MULTIDIM_MAX_BLOBS):
        if i < len(blobs):
            for d in range(D):
                obs[idx] = blobs[i].pos[d] / scale
                idx += 1
        else:
            idx += D  # zeros

    # Grid slot assignments (MULTIDIM_MAX_BLOBS slots)
    max_slots = MULTIDIM_MAX_BLOBS
    for i in range(MULTIDIM_MAX_BLOBS):
        if i < len(blobs) and blobs[i].grid_slot is not None:
            obs[idx] = (blobs[i].grid_slot + 1) / max_slots
        idx += 1

    # Blob count
    obs[idx] = len(blobs)
    idx += 1

    # Phase: 0=counting, 0.5=unmarking, 1=predict
    if state.phase == "predict":
        obs[idx] = 1.0
    elif state.phase == "unmarking":
        obs[idx] = 0.5
    idx += 1

    # Grid filled normalized
    obs[idx] = grid.filled_count / max(len(blobs), 1)
    idx += 1

    # Grid filled raw
    obs[idx] = grid.filled_count
    idx += 1

    assert idx == obs_d, f"Observation indexing error: {idx} != {obs_d}"
    return obs


# =====================================================================
# Symlog (applied AFTER projection per spec)
# =====================================================================

def _symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# =====================================================================
# Reset and Step
# =====================================================================

def _reset_multidim(D: int, blob_count_min: int, blob_count_max: int,
                    max_steps: int, bidirectional: bool,
                    world_scale: float = DEFAULT_WORLD_SCALE) -> MultiDimEpisodeState:
    """Initialize a new episode in D dimensions."""
    blob_count = blob_count_min + int(random.random() * (blob_count_max - blob_count_min + 1))

    # Scale min_sep with sqrt(D/2)
    min_sep = MIN_SEPARATION * math.sqrt(D / 2.0)

    # Generate blob positions in field zone
    positions = _scattered_positions_nd(blob_count, D, world_scale, MARGIN, min_sep)

    blobs = []
    for i, pos in enumerate(positions):
        blobs.append(BlobND(
            id=i,
            pos=pos.copy(),
            field_pos=pos.copy(),
        ))

    # Create grid
    grid = _create_grid_nd(D, blob_count, world_scale)

    # v2: Fast bot personality for shorter episodes
    personality = MARKING_BOT_FAST

    # Random bot start in field area (within spawn region for high D)
    field_max = world_scale * 0.55
    spawn_scale = world_scale / math.sqrt(D) if D > 1 else world_scale
    center = world_scale / 2.0
    start = np.zeros(D, dtype=np.float64)
    start[0] = MARGIN + random.random() * (field_max - 2 * MARGIN)
    for d in range(1, D):
        half_range = (spawn_scale - 2 * MARGIN) / 2.0
        start[d] = center + (random.random() - 0.5) * 2.0 * half_range

    # Waypoints = blob positions
    waypoints = [b.pos.copy() for b in blobs]

    # Select initial target (nearest)
    best_i, best_d = 0, float("inf")
    for i, wp in enumerate(waypoints):
        d = _dist_nd(start, wp)
        if d < best_d:
            best_d = d
            best_i = i

    bot = BotND(
        pos=start,
        vel=np.zeros(D, dtype=np.float64),
        personality=personality,
        waypoints=waypoints,
        current_target_index=best_i,
    )

    state = MultiDimEpisodeState(
        D=D,
        bot=bot,
        blobs=blobs,
        grid=grid,
        phase="counting",
        blob_count_at_start=len(blobs),
        bot_type=personality.name,
        bidirectional=bidirectional,
        prev_bot_pos=start.copy(),
        world_scale=world_scale,
    )
    return state


def _step_multidim(state: MultiDimEpisodeState, max_steps: int, action) -> dict:
    """Step the multi-dimensional counting world."""
    D = state.D

    # Predict phase
    if state.phase == "predict":
        bot_tally = state.bot.count_tally
        reward = 0.0
        if action is not None:
            if action == bot_tally:
                reward = 1.0
            elif abs(action - bot_tally) <= 1:
                reward = 0.5
        state.done = True
        state.reward = reward
        return {"reward": reward, "done": True, "truncated": False, "discount": 1.0}

    # --- Counting phase ---
    bot = state.bot

    # Update bot
    _update_bot_nd(bot, D, state.world_scale)

    # Track distance
    if state.prev_bot_pos is not None:
        state.bot_total_distance += _dist_nd(bot.pos, state.prev_bot_pos)
    state.prev_bot_pos = bot.pos.copy()

    # Scale arrival radius with sqrt(D/2)
    arrival_radius = PICKUP_RADIUS_2D * math.sqrt(D / 2.0)

    # Check waypoint arrival
    if _check_arrival_nd(bot, arrival_radius):
        idx = bot.current_target_index
        if idx < len(state.blobs):
            blob = state.blobs[idx]
            if blob.grid_slot is None:
                placed = _count_blob_to_grid_nd(blob, idx, state.grid)
                if placed:
                    bot.count_tally += 1
                    bot.visited_indices.add(idx)
                    # Pause
                    bot.pause_timer = bot.personality.pause_duration
                    # Select next target
                    bot.current_target_index = _select_next_target_nd(bot)
            else:
                bot.visited_indices.add(idx)
                bot.current_target_index = _select_next_target_nd(bot)

        # Check all visited
        if len(bot.visited_indices) >= len(state.blobs):
            bot.all_visited = True

    # Update blob animations
    _update_blob_animations_nd(state.blobs, state.grid)

    # Phase transition: all blobs placed → predict
    all_placed = all(b.grid_slot is not None for b in state.blobs)
    no_animating = not any(b.animating for b in state.blobs)
    if all_placed and no_animating and state.phase == "counting":
        state.phase = "predict"

    state.step_count += 1

    # Truncation
    if state.step_count >= max_steps and not state.done:
        state.truncated = True
        state.done = True

    reward = 0.0
    state.reward = reward

    return {
        "reward": reward,
        "done": state.done,
        "truncated": state.truncated,
        "discount": 0.0 if state.truncated else 1.0,
    }


# =====================================================================
# Gym Environment
# =====================================================================

class MultiDimCountingWorldEnv(gym.Env):
    """Multi-dimensional counting world with random orthogonal projection.

    Each episode samples a spatial dimensionality D from a weighted distribution.
    The raw observation (26*D + 30 dims) is projected to a fixed proj_dim vector.

    Args:
        blob_count_min: Min blobs per episode
        blob_count_max: Max blobs per episode
        max_steps_base: Base max steps (scaled by sqrt(D) for high dims)
        bidirectional: Whether to include unmarking phase
        proj_dim: Output observation dimensionality (default 128)
        dim_weights: Dict[int, float] mapping D -> sampling probability
        fixed_dim: If set, always use this dimensionality (for evaluation)
        world_scale: Size of the world in each dimension
        apply_symlog: Apply symlog after projection (default True)
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 blob_count_min: int = 3,
                 blob_count_max: int = 13,
                 max_steps_base: int = DEFAULT_MAX_STEPS,
                 bidirectional: bool = False,
                 proj_dim: int = PROJ_DIM,
                 dim_weights: Optional[Dict[int, float]] = None,
                 fixed_dim: Optional[int] = None,
                 world_scale: float = DEFAULT_WORLD_SCALE,
                 apply_symlog: bool = True):
        super().__init__()

        self._blob_count_min = blob_count_min
        self._blob_count_max = blob_count_max
        self._max_steps_base = max_steps_base
        self._bidirectional = bidirectional
        self._proj_dim = proj_dim
        self._dim_weights = dim_weights or DEFAULT_DIM_WEIGHTS
        self._fixed_dim = fixed_dim
        self._world_scale = world_scale
        self._apply_symlog = apply_symlog

        # Normalize weights
        total = sum(self._dim_weights.values())
        self._dims = sorted(self._dim_weights.keys())
        self._probs = [self._dim_weights[d] / total for d in self._dims]

        # Action space: predicted tally (0-12)
        self.action_space = gym.spaces.Discrete(MULTIDIM_MAX_BLOBS)

        # Observation space: projected vector
        self.observation_space = gym.spaces.Box(
            low=-50.0, high=50.0, shape=(proj_dim,), dtype=np.float32
        )

        self._state: Optional[MultiDimEpisodeState] = None
        self._episode_count = 0
        self._current_D = 2

        # Pre-cache all projection matrices
        for D in self._dims:
            _get_projection_matrix_nd(D, proj_dim)
        if fixed_dim and fixed_dim not in self._dims:
            _get_projection_matrix_nd(fixed_dim, proj_dim)

    def _sample_dim(self) -> int:
        if self._fixed_dim is not None:
            return self._fixed_dim
        return int(np.random.choice(self._dims, p=self._probs))

    def _project_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Project raw observation to fixed dimensionality."""
        proj_mat = _get_projection_matrix_nd(self._current_D, self._proj_dim)
        projected = proj_mat @ raw_obs
        if self._apply_symlog:
            projected = _symlog(projected)
        return projected.astype(np.float32)

    def _max_steps(self) -> int:
        """v2: Fixed cap — faster bot + smaller world + density scaling make
        episodes completable within the base limit at all trained dimensions."""
        return self._max_steps_base

    def reset(self) -> np.ndarray:
        self._current_D = self._sample_dim()
        self._state = _reset_multidim(
            D=self._current_D,
            blob_count_min=self._blob_count_min,
            blob_count_max=self._blob_count_max,
            max_steps=self._max_steps(),
            bidirectional=self._bidirectional,
            world_scale=self._world_scale,
        )
        self._episode_count += 1
        raw_obs = _get_observation_nd(self._state)
        return self._project_obs(raw_obs)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        result = _step_multidim(self._state, self._max_steps(), action)
        raw_obs = _get_observation_nd(self._state)
        projected_obs = self._project_obs(raw_obs)

        info = {
            "discount": result["discount"],
            "spatial_dim": self._current_D,
            "grid_filled": self._state.grid.filled_count,
            "blob_count": len(self._state.blobs),
            "phase": self._state.phase,
            "step_count": self._state.step_count,
        }

        return projected_obs, result["reward"], result["done"], info

    def close(self):
        pass

    def render(self, mode="human"):
        return None


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    print("Multi-Dimensional Counting World — Smoke Test")
    print("=" * 60)

    # Test each dimensionality
    for D in [2, 3, 4, 5]:
        print(f"\n--- D={D} ---")
        env = MultiDimCountingWorldEnv(
            blob_count_min=5, blob_count_max=13,
            fixed_dim=D,
        )

        obs = env.reset()
        print(f"  Obs shape: {obs.shape}")
        print(f"  Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Raw obs dim for D={D}: {_obs_dim(D)}")

        total_reward = 0
        steps = 0
        while True:
            action = random.randint(0, 24)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break

        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final phase: {info['phase']}")
        print(f"  Grid filled: {info['grid_filled']}/{info['blob_count']}")
        env.close()

    # Test mixed dimensionality (training mode)
    print(f"\n--- Mixed dimensionality (10 episodes) ---")
    env = MultiDimCountingWorldEnv(blob_count_min=3, blob_count_max=13)
    dim_counts = {}
    for ep in range(10):
        obs = env.reset()
        D = env._current_D
        dim_counts[D] = dim_counts.get(D, 0) + 1
        steps = 0
        while True:
            obs, reward, done, info = env.step(0)
            steps += 1
            if done:
                break
        print(f"  Episode {ep+1}: D={D}, steps={steps}, filled={info['grid_filled']}/{info['blob_count']}")

    print(f"\n  Dim distribution: {dict(sorted(dim_counts.items()))}")
    env.close()

    print("\nSmoke test passed!")
