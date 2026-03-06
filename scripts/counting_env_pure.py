"""
Pure Python Counting World Environment
=======================================
Drop-in replacement for counting_env.py — no Node.js subprocess, no JSON IPC.
Faithful 1:1 port of the TypeScript headlessEnv.ts + supporting modules.

Observation: Box(82,) float32
Action: Discrete(25) — predicted tally (0-24), only meaningful when phase=predict

Compatible with gym 0.26.x (old API) as required by NM512/dreamerv3-torch.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import gym
import gym.spaces
import numpy as np

# =====================================================================
# Constants (matching TypeScript exactly)
# =====================================================================

WORLD_WIDTH = 1400
WORLD_HEIGHT = 1000
MARGIN = 120
MIN_SEPARATION = 50
DT = 1
MAX_BLOB_COUNT = 25
BOUNDARY_MARGIN = 40
UNCOUNT_ARRIVAL_RADIUS = 30
UNCOUNT_PAUSE_FRAMES = 15
OBS_SIZE = 82

# Grid
GRID_COLS = 5
GRID_ROWS = 5
GRID_CELL_SIZE = 50
GRID_RIGHT_MARGIN = 60

# Blob animation
ANIM_SPEED = 0.008


# =====================================================================
# Vec2 operations
# =====================================================================

def _len(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return _len(ax - bx, ay - by)


def _normalize(x: float, y: float):
    d = _len(x, y)
    if d == 0:
        return 0.0, 0.0
    return x / d, y / d


def _limit(x: float, y: float, max_val: float):
    d = _len(x, y)
    if d <= max_val:
        return x, y
    nx, ny = _normalize(x, y)
    return nx * max_val, ny * max_val


def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


# =====================================================================
# Steering behaviors
# =====================================================================

def _arrive(pos_x, pos_y, vel_x, vel_y, max_speed, max_force,
            target_x, target_y, slow_radius, easing):
    """Arrive steering: seek with smooth deceleration."""
    ox = target_x - pos_x
    oy = target_y - pos_y
    d = _len(ox, oy)
    if d < 0.5:
        return -vel_x, -vel_y  # brake

    if d < slow_radius:
        t = d / slow_radius
        eased = t ** easing
        speed = max_speed * eased
    else:
        speed = max_speed

    nx, ny = _normalize(ox, oy)
    desired_x = nx * speed
    desired_y = ny * speed
    steer_x = desired_x - vel_x
    steer_y = desired_y - vel_y
    return _limit(steer_x, steer_y, max_force)


def _wander(pos_x, pos_y, vel_x, vel_y, max_speed, max_force,
            wander_angle, wander_radius, wander_distance, wander_jitter):
    """Wander steering: random circle ahead of agent."""
    new_angle = wander_angle + (random.random() - 0.5) * wander_jitter
    vl = _len(vel_x, vel_y)
    if vl > 0.01:
        vx, vy = vel_x / vl, vel_y / vl
    else:
        vx, vy = 1.0, 0.0
    # Circle center ahead
    cx = pos_x + vx * wander_distance
    cy = pos_y + vy * wander_distance
    # Target on circle
    tx = cx + math.cos(new_angle) * wander_radius
    ty = cy + math.sin(new_angle) * wander_radius
    # Desired
    dx, dy = tx - pos_x, ty - pos_y
    nx, ny = _normalize(dx, dy)
    desired_x = nx * max_speed
    desired_y = ny * max_speed
    fx = desired_x - vel_x
    fy = desired_y - vel_y
    fx, fy = _limit(fx, fy, max_force)
    return fx, fy, new_angle


# =====================================================================
# Bot Personalities (matching TypeScript BotPersonality)
# =====================================================================

@dataclass
class BotPersonality:
    name: str
    color: str
    max_speed: float
    max_force: float
    slow_radius: float
    arrive_easing: float
    wander_weight: float
    wander_radius: float
    wander_distance: float
    wander_jitter: float
    target_switch_probability: float
    revisit_probability: float
    target_selection_order: str  # nearest, sequential, random, farthest-first, center-out
    speed_variance: float
    pause_duration: int
    pause_variance: int
    anticipation_strength: float
    follow_through_strength: float
    separation_weight: float
    separation_radius: float
    waypoint_arrival_radius: float


MARKING_BOT = BotPersonality(
    name="marking", color="#34D399",
    max_speed=2.0, max_force=0.12,
    slow_radius=50, arrive_easing=1.8,
    wander_weight=0, wander_radius=0, wander_distance=0, wander_jitter=0,
    target_switch_probability=0, revisit_probability=0,
    target_selection_order="nearest",
    speed_variance=0.05,
    pause_duration=30, pause_variance=5,
    anticipation_strength=0.3, follow_through_strength=0.2,
    separation_weight=0.5, separation_radius=50,
    waypoint_arrival_radius=12,
)

CONFUSED_BOT = BotPersonality(
    name="confused", color="#F87171",
    max_speed=2.5, max_force=0.15,
    slow_radius=30, arrive_easing=0.5,
    wander_weight=0.3, wander_radius=30, wander_distance=50, wander_jitter=0.8,
    target_switch_probability=0.004, revisit_probability=0.12,
    target_selection_order="random",
    speed_variance=0.25,
    pause_duration=15, pause_variance=15,
    anticipation_strength=0.1, follow_through_strength=0.05,
    separation_weight=0.3, separation_radius=40,
    waypoint_arrival_radius=20,
)

ORGANIZING_BOT = BotPersonality(
    name="organizing", color="#60A5FA",
    max_speed=3.2, max_force=0.2,
    slow_radius=35, arrive_easing=1.2,
    wander_weight=0, wander_radius=0, wander_distance=0, wander_jitter=0,
    target_switch_probability=0, revisit_probability=0,
    target_selection_order="nearest",
    speed_variance=0.08,
    pause_duration=10, pause_variance=3,
    anticipation_strength=0.15, follow_through_strength=0.1,
    separation_weight=0.4, separation_radius=45,
    waypoint_arrival_radius=10,
)

GRID_BOT = BotPersonality(
    name="grid", color="#A78BFA",
    max_speed=2.0, max_force=0.12,
    slow_radius=60, arrive_easing=2.2,
    wander_weight=0, wander_radius=0, wander_distance=0, wander_jitter=0,
    target_switch_probability=0, revisit_probability=0,
    target_selection_order="nearest",
    speed_variance=0.03,
    pause_duration=25, pause_variance=5,
    anticipation_strength=0.5, follow_through_strength=0.4,
    separation_weight=0.5, separation_radius=55,
    waypoint_arrival_radius=10,
)

UNCONVENTIONAL_BOT = BotPersonality(
    name="unconventional", color="#FBBF24",
    max_speed=2.2, max_force=0.14,
    slow_radius=45, arrive_easing=1.6,
    wander_weight=0.05, wander_radius=15, wander_distance=40, wander_jitter=0.4,
    target_switch_probability=0, revisit_probability=0,
    target_selection_order="center-out",
    speed_variance=0.1,
    pause_duration=20, pause_variance=8,
    anticipation_strength=0.25, follow_through_strength=0.2,
    separation_weight=0.4, separation_radius=45,
    waypoint_arrival_radius=12,
)

ALL_PERSONALITIES = [CONFUSED_BOT, MARKING_BOT, ORGANIZING_BOT, GRID_BOT, UNCONVENTIONAL_BOT]

_PERSONALITY_BY_NAME = {p.name: p for p in ALL_PERSONALITIES}


# =====================================================================
# Bot
# =====================================================================

@dataclass
class Bot:
    personality: BotPersonality
    pos_x: float
    pos_y: float
    vel_x: float = 0.0
    vel_y: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    # Waypoints (blob positions)
    waypoints: list = field(default_factory=list)  # list of (x, y) tuples
    current_target_index: int = 0
    visited_indices: set = field(default_factory=set)
    all_visited: bool = False
    # Wander
    wander_angle: float = 0.0
    # Pause
    pause_timer: int = 0
    # Speed modulation
    current_speed_multiplier: float = 1.0
    speed_multiplier_target: float = 1.0
    # Anticipation
    prev_dir_x: float = 0.0
    prev_dir_y: float = 0.0
    # Counting
    count_tally: int = 0
    # Dynamic target (uncount phase)
    dynamic_target: Optional[tuple] = None
    # Confused bot: stop after N
    _confused_stop_after: int = 0


def _select_initial_target(personality: BotPersonality, waypoints: list) -> int:
    if not waypoints:
        return 0
    order = personality.target_selection_order
    if order == "center-out":
        cx = sum(w[0] for w in waypoints) / len(waypoints)
        cy = sum(w[1] for w in waypoints) / len(waypoints)
        best_i, best_d = 0, float("inf")
        for i, (wx, wy) in enumerate(waypoints):
            d = _dist(cx, cy, wx, wy)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i
    elif order == "farthest-first":
        return len(waypoints) - 1
    return 0


def _select_next_target(bot: Bot) -> int:
    p = bot.personality
    wps = bot.waypoints
    vis = bot.visited_indices
    cur = bot.current_target_index

    # Confused revisit
    if p.revisit_probability > 0 and random.random() < p.revisit_probability and vis:
        return random.choice(list(vis))

    unvisited = [i for i in range(len(wps)) if i not in vis]
    if not unvisited:
        return cur

    order = p.target_selection_order
    if order == "nearest":
        best_i, best_d = unvisited[0], _dist(bot.pos_x, bot.pos_y, wps[unvisited[0]][0], wps[unvisited[0]][1])
        for i in unvisited:
            d = _dist(bot.pos_x, bot.pos_y, wps[i][0], wps[i][1])
            if d < best_d:
                best_d = d
                best_i = i
        return best_i
    elif order == "sequential":
        for offset in range(1, len(wps) + 1):
            idx = (cur + offset) % len(wps)
            if idx not in vis:
                return idx
        return unvisited[0]
    elif order == "random":
        return random.choice(unvisited)
    elif order == "farthest-first":
        best_i, best_d = unvisited[0], 0.0
        for i in unvisited:
            d = _dist(bot.pos_x, bot.pos_y, wps[i][0], wps[i][1])
            if d > best_d:
                best_d = d
                best_i = i
        return best_i
    elif order == "center-out":
        cx = sum(w[0] for w in wps) / len(wps)
        cy = sum(w[1] for w in wps) / len(wps)
        unvisited.sort(key=lambda i: _dist(cx, cy, wps[i][0], wps[i][1]))
        return unvisited[0]
    return unvisited[0]


def _update_bot(bot: Bot):
    """Update bot for one frame. No other bots (separation array always empty)."""
    p = bot.personality

    # Handle pause
    if bot.pause_timer > 0:
        bot.pause_timer -= 1
        bot.vel_x *= 0.85
        bot.vel_y *= 0.85
        return

    # Check waypoint arrival (skip if using dynamic target)
    if bot.waypoints and bot.dynamic_target is None:
        tx, ty = bot.waypoints[bot.current_target_index]
        d = _dist(bot.pos_x, bot.pos_y, tx, ty)

        if d < p.waypoint_arrival_radius:
            # This is where the arrival callback fires — handled externally
            # Just return a flag; caller handles blob interaction
            return  # caller calls _handle_waypoint_arrival

        # Confused mid-path switch
        if p.target_switch_probability > 0 and random.random() < p.target_switch_probability:
            bot.current_target_index = _select_next_target(bot)

    # --- Compute steering forces ---
    max_speed = p.max_speed * bot.current_speed_multiplier
    forces = []

    if bot.dynamic_target is not None:
        fx, fy = _arrive(bot.pos_x, bot.pos_y, bot.vel_x, bot.vel_y,
                         max_speed, p.max_force,
                         bot.dynamic_target[0], bot.dynamic_target[1],
                         p.slow_radius, p.arrive_easing)
        forces.append((fx, fy, 1.0))
    elif bot.waypoints:
        tx, ty = bot.waypoints[bot.current_target_index]
        fx, fy = _arrive(bot.pos_x, bot.pos_y, bot.vel_x, bot.vel_y,
                         max_speed, p.max_force, tx, ty,
                         p.slow_radius, p.arrive_easing)
        forces.append((fx, fy, 1.0))

    # Wander
    if p.wander_weight > 0:
        wx, wy, new_angle = _wander(
            bot.pos_x, bot.pos_y, bot.vel_x, bot.vel_y,
            max_speed, p.max_force,
            bot.wander_angle, p.wander_radius, p.wander_distance, p.wander_jitter)
        forces.append((wx, wy, p.wander_weight))
        bot.wander_angle = new_angle

    # No separation (single bot)

    # Combine forces
    total_x, total_y = 0.0, 0.0
    for fx, fy, w in forces:
        total_x += fx * w
        total_y += fy * w
    bot.acc_x, bot.acc_y = _limit(total_x, total_y, p.max_force)

    # Anticipation
    if p.anticipation_strength > 0 and _len(bot.vel_x, bot.vel_y) > 0.5:
        cd_x, cd_y = _normalize(bot.vel_x, bot.vel_y)
        ad_x, ad_y = _normalize(bot.acc_x, bot.acc_y)
        dir_change = 1 - (cd_x * ad_x + cd_y * ad_y)
        if dir_change > 0.3:
            anti_scale = -p.anticipation_strength * dir_change * p.max_force
            bot.acc_x += cd_x * anti_scale
            bot.acc_y += cd_y * anti_scale

    # Update velocity
    bot.vel_x += bot.acc_x * DT
    bot.vel_y += bot.acc_y * DT
    bot.vel_x, bot.vel_y = _limit(bot.vel_x, bot.vel_y, max_speed)

    # Follow-through dampening
    dampening = 1 - (0.02 * (1 - p.follow_through_strength))
    bot.vel_x *= dampening
    bot.vel_y *= dampening

    # Speed variation
    if p.speed_variance > 0:
        if random.random() < 0.02:
            bot.speed_multiplier_target = 1 + (random.random() - 0.5) * 2 * p.speed_variance
        bot.current_speed_multiplier += (bot.speed_multiplier_target - bot.current_speed_multiplier) * 0.05

    # Update position
    bot.pos_x += bot.vel_x * DT
    bot.pos_y += bot.vel_y * DT

    # Store direction
    if _len(bot.vel_x, bot.vel_y) > 0.1:
        bot.prev_dir_x, bot.prev_dir_y = _normalize(bot.vel_x, bot.vel_y)


def _check_waypoint_arrival(bot: Bot) -> bool:
    """Check if bot is at current waypoint. Returns True if arrived."""
    if not bot.waypoints or bot.dynamic_target is not None:
        return False
    if bot.pause_timer > 0:
        return False
    tx, ty = bot.waypoints[bot.current_target_index]
    return _dist(bot.pos_x, bot.pos_y, tx, ty) < bot.personality.waypoint_arrival_radius


# =====================================================================
# Blob
# =====================================================================

@dataclass
class Blob:
    id: int
    pos_x: float
    pos_y: float
    field_pos_x: float
    field_pos_y: float
    grid_slot: Optional[int] = None
    counted_by: set = field(default_factory=set)
    marked_by: Optional[str] = None
    # Animation
    anim_from_x: float = 0.0
    anim_from_y: float = 0.0
    anim_to_x: float = 0.0
    anim_to_y: float = 0.0
    anim_progress: float = 0.0
    animating: bool = False
    # Deferred count: blob is flying to grid but hasn't "landed" yet
    pending_grid_placement: bool = False


def _start_blob_transition(blob: Blob, target_x: float, target_y: float):
    blob.anim_from_x = blob.pos_x
    blob.anim_from_y = blob.pos_y
    blob.anim_to_x = target_x
    blob.anim_to_y = target_y
    blob.anim_progress = 0.0
    blob.animating = True


def _update_blob_animations(blobs: list, grid=None):
    for blob in blobs:
        if blob.animating:
            blob.anim_progress = min(1.0, blob.anim_progress + ANIM_SPEED * DT)
            t = _ease_out_cubic(blob.anim_progress)
            blob.pos_x = blob.anim_from_x + (blob.anim_to_x - blob.anim_from_x) * t
            blob.pos_y = blob.anim_from_y + (blob.anim_to_y - blob.anim_from_y) * t
            if blob.anim_progress >= 1.0:
                blob.pos_x = blob.anim_to_x
                blob.pos_y = blob.anim_to_y
                blob.animating = False
                blob.anim_progress = 0.0
                # Deferred count: blob has landed on grid
                if blob.pending_grid_placement and grid is not None:
                    grid.filled_count += 1
                    blob.pending_grid_placement = False


# =====================================================================
# Grid
# =====================================================================

@dataclass
class GridState:
    slots: list  # list of (x, y) tuples — 25 slot positions
    occupancy: list  # slot -> blob_id (-1 = empty)
    placement_order: list  # FILO stack of blob indices
    filled_count: int = 0
    grid_left: float = 0.0
    grid_top: float = 0.0


def _create_grid() -> GridState:
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - GRID_COLS * GRID_CELL_SIZE
    grid_height = GRID_ROWS * GRID_CELL_SIZE
    grid_top = (WORLD_HEIGHT - grid_height) / 2

    slots = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = grid_left + col * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
            y = grid_top + (GRID_ROWS - 1 - row) * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
            slots.append((x, y))

    return GridState(
        slots=slots,
        occupancy=[-1] * len(slots),
        placement_order=[],
        filled_count=0,
        grid_left=grid_left,
        grid_top=grid_top,
    )


def _create_line_grid(count: int) -> GridState:
    """Create a vertical line of slots on the right side of the world.
    Same region as the 5x5 grid. Slots fill bottom-to-top."""
    region_width = GRID_COLS * GRID_CELL_SIZE  # 250
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - region_width
    cx = grid_left + region_width / 2  # center x of the region
    margin = 80
    usable_h = WORLD_HEIGHT - 2 * margin
    spacing = usable_h / (count - 1) if count > 1 else 0

    slots = []
    for i in range(count):
        # Bottom-to-top: slot 0 at the bottom (highest y)
        y = WORLD_HEIGHT - margin - i * spacing
        slots.append((cx, y))

    grid_top = margin  # topmost slot position (approximately)
    return GridState(
        slots=slots,
        occupancy=[-1] * len(slots),
        placement_order=[],
        filled_count=0,
        grid_left=grid_left,
        grid_top=grid_top,
    )


def _create_circle_grid(count: int) -> GridState:
    """Create a ring of slots in the grid region.
    Starts from bottom, goes clockwise. Matches frontend circle arrangement."""
    region_width = GRID_COLS * GRID_CELL_SIZE  # 250
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - region_width
    cx = grid_left + region_width / 2
    cy = WORLD_HEIGHT / 2
    radius = min(region_width, 500) / 2 - 20  # fit within region

    slots = []
    for i in range(count):
        # Start from bottom, go clockwise (matches frontend)
        angle = math.pi / 2 + (i * 2 * math.pi) / count
        x = cx + math.cos(angle) * radius
        y = cy + math.sin(angle) * radius
        slots.append((x, y))

    grid_top = cy - radius - 20
    return GridState(
        slots=slots,
        occupancy=[-1] * len(slots),
        placement_order=[],
        filled_count=0,
        grid_left=grid_left,
        grid_top=grid_top,
    )


def _create_scatter_grid(count: int) -> GridState:
    """Create a tight disorganized cluster of slots in the grid region.
    Blobs get compressed into a smaller area — no organized structure.
    Uses rejection sampling to avoid overlap while keeping things tight."""
    region_width = GRID_COLS * GRID_CELL_SIZE  # 250
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - region_width
    cx = grid_left + region_width / 2
    cy = WORLD_HEIGHT / 2
    # Tight cluster: ~60% of region size
    cluster_w = region_width * 0.6
    cluster_h = 300  # compact vertical extent
    min_sep = 30  # tighter than normal min separation

    slots = []
    max_attempts = 2000
    for _ in range(count):
        placed = False
        for _attempt in range(max_attempts):
            x = cx + (random.random() - 0.5) * cluster_w
            y = cy + (random.random() - 0.5) * cluster_h
            # Check min separation from existing slots
            too_close = False
            for sx, sy in slots:
                if _dist(x, y, sx, sy) < min_sep:
                    too_close = True
                    break
            if not too_close:
                slots.append((x, y))
                placed = True
                break
        if not placed:
            # Fallback: place with slight jitter from center
            x = cx + (random.random() - 0.5) * cluster_w
            y = cy + (random.random() - 0.5) * cluster_h
            slots.append((x, y))

    grid_top = cy - cluster_h / 2
    return GridState(
        slots=slots,
        occupancy=[-1] * len(slots),
        placement_order=[],
        filled_count=0,
        grid_left=grid_left,
        grid_top=grid_top,
    )


def _create_target_grid(target_arrangement: str, count: int) -> GridState:
    """Factory: create target slot layout based on arrangement type."""
    if target_arrangement == "line":
        return _create_line_grid(count)
    if target_arrangement == "circle":
        return _create_circle_grid(count)
    if target_arrangement == "scatter":
        return _create_scatter_grid(count)
    return _create_grid()


def _count_blob_to_grid(blob: Blob, blob_index: int, grid: GridState, bot_id: str, bot_color: str) -> bool:
    if blob.grid_slot is not None:
        return False
    next_slot = len(grid.placement_order)
    if next_slot >= len(grid.slots):
        return False

    slot_idx = next_slot
    grid.occupancy[slot_idx] = blob_index
    grid.placement_order.append(blob_index)
    # NOTE: grid.filled_count is NOT incremented here — deferred until blob lands

    blob.grid_slot = slot_idx
    blob.counted_by.add(bot_id)
    blob.pending_grid_placement = True
    _start_blob_transition(blob, grid.slots[slot_idx][0], grid.slots[slot_idx][1])
    return True


def _uncount_blob_from_grid(blobs: list, grid: GridState, bot_id: str):
    if not grid.placement_order:
        return None

    blob_index = grid.placement_order.pop()
    blob = blobs[blob_index]
    slot_idx = blob.grid_slot

    if slot_idx is not None:
        grid.occupancy[slot_idx] = -1
    # Only decrement filled_count if blob had already landed (was counted)
    if blob.pending_grid_placement:
        blob.pending_grid_placement = False
    else:
        grid.filled_count -= 1

    # Clear counting state
    blob.counted_by.discard(bot_id)
    if blob.marked_by == bot_id:
        blob.marked_by = None
    blob.grid_slot = None

    # Scatter back to field
    scatter_x, scatter_y = _find_scatter_position(blobs, grid.grid_left)
    _start_blob_transition(blob, scatter_x, scatter_y)
    blob.field_pos_x = scatter_x
    blob.field_pos_y = scatter_y

    return blob_index


def _get_next_uncount_target(blobs: list, grid: GridState):
    if not grid.placement_order:
        return None
    blob_index = grid.placement_order[-1]
    blob = blobs[blob_index]
    return (blob.pos_x, blob.pos_y)


def _find_scatter_position(blobs: list, grid_left_edge: float):
    field_max_x = min(grid_left_edge - 80, WORLD_WIDTH * 0.55)
    min_spacing = 40
    for _ in range(50):
        x = MARGIN + random.random() * (field_max_x - MARGIN)
        y = MARGIN + random.random() * (WORLD_HEIGHT - 2 * MARGIN)
        too_close = False
        for b in blobs:
            if b.grid_slot is not None:
                continue
            if _dist(b.pos_x, b.pos_y, x, y) < min_spacing:
                too_close = True
                break
        if not too_close:
            return x, y
    # Fallback
    return (MARGIN + random.random() * (field_max_x - MARGIN),
            MARGIN + random.random() * (WORLD_HEIGHT - 2 * MARGIN))


# =====================================================================
# Arrangements
# =====================================================================

def _scattered_positions(count, w, h, margin, min_sep):
    positions = []
    attempts = 0
    while len(positions) < count and attempts < count * 80:
        attempts += 1
        x = margin + random.random() * (w - 2 * margin)
        y = margin + random.random() * (h - 2 * margin)
        if all(_dist(px, py, x, y) >= min_sep for px, py in positions):
            positions.append((x, y))
    return positions


def _clustered_positions(count, w, h, margin, min_sep):
    num_clusters = 2 + int(random.random() * 2)
    centers = []
    for _ in range(num_clusters):
        for _a in range(100):
            cx = margin * 2 + random.random() * (w - 4 * margin)
            cy = margin * 2 + random.random() * (h - 4 * margin)
            min_cluster_dist = min(w, h) * 0.25
            if all(_dist(ccx, ccy, cx, cy) >= min_cluster_dist for ccx, ccy in centers):
                centers.append((cx, cy))
                break
    if not centers:
        return _scattered_positions(count, w, h, margin, min_sep)

    positions = []
    cluster_radius = min(w, h) * 0.12
    attempts = 0
    while len(positions) < count and attempts < count * 80:
        attempts += 1
        ccx, ccy = random.choice(centers)
        angle = random.random() * math.pi * 2
        r = random.random() * cluster_radius
        x = ccx + math.cos(angle) * r
        y = ccy + math.sin(angle) * r
        if x < margin or x > w - margin or y < margin or y > h - margin:
            continue
        if all(_dist(px, py, x, y) >= min_sep for px, py in positions):
            positions.append((x, y))
    return positions


def _grid_like_positions(count, w, h, margin, min_sep):
    cols = math.ceil(math.sqrt(count * (w / h)))
    rows = math.ceil(count / cols)
    cell_w = (w - 2 * margin) / cols
    cell_h = (h - 2 * margin) / rows
    jitter = min(cell_w, cell_h) * 0.25

    positions = []
    for r in range(rows):
        if len(positions) >= count:
            break
        for c in range(cols):
            if len(positions) >= count:
                break
            base_x = margin + (c + 0.5) * cell_w
            base_y = margin + (r + 0.5) * cell_h
            x = base_x + (random.random() - 0.5) * 2 * jitter
            y = base_y + (random.random() - 0.5) * 2 * jitter
            x = max(margin, min(w - margin, x))
            y = max(margin, min(h - margin, y))
            positions.append((x, y))
    return positions


def _mixed_positions(count, w, h, margin, min_sep):
    cluster_count = count // 2
    scatter_count = count - cluster_count
    clustered = _clustered_positions(cluster_count, w, h, margin, min_sep)
    scattered = []
    attempts = 0
    while len(scattered) < scatter_count and attempts < scatter_count * 80:
        attempts += 1
        x = margin + random.random() * (w - 2 * margin)
        y = margin + random.random() * (h - 2 * margin)
        all_existing = clustered + scattered
        if all(_dist(px, py, x, y) >= min_sep for px, py in all_existing):
            scattered.append((x, y))
    return clustered + scattered


_ARRANGEMENT_FUNCS = {
    "scattered": _scattered_positions,
    "clustered": _clustered_positions,
    "grid-like": _grid_like_positions,
    "mixed": _mixed_positions,
}
_ARRANGEMENT_TYPES = list(_ARRANGEMENT_FUNCS.keys())


def _random_arrangement():
    return random.choice(_ARRANGEMENT_TYPES)


def _generate_arrangement(count, width, height, margin, min_sep, style):
    return _ARRANGEMENT_FUNCS.get(style, _scattered_positions)(count, width, height, margin, min_sep)


# =====================================================================
# Bot personality selection by stage
# =====================================================================

def _select_personality(stage: int) -> BotPersonality:
    if stage == 1:
        return MARKING_BOT
    elif stage == 2:
        return CONFUSED_BOT
    elif stage == 3:
        return MARKING_BOT if random.random() < 0.5 else CONFUSED_BOT
    elif stage == 4:
        pool = [MARKING_BOT, CONFUSED_BOT, ORGANIZING_BOT, GRID_BOT]
        return random.choice(pool)
    else:  # stage 5
        return random.choice(ALL_PERSONALITIES)


# =====================================================================
# Episode State
# =====================================================================

@dataclass
class EpisodeState:
    bot: Bot
    blobs: list  # list of Blob
    grid: GridState
    phase: str  # "counting", "unmarking", "predict"
    step_count: int = 0
    done: bool = False
    truncated: bool = False
    bot_total_distance: float = 0.0
    prev_bot_x: float = 0.0
    prev_bot_y: float = 0.0
    blob_count_at_start: int = 0
    bot_type: str = ""
    arrangement_type: str = ""
    reward: float = 0.0
    bidirectional: bool = False
    uncount_pause_timer: int = 0
    # Confused bot: stop after N
    confused_stop_after: int = 0
    # Ablation: mask out grid filled count from observation (indices 80-81)
    mask_count: bool = False
    # Ablation: mask out slot assignments from observation (indices 53-77)
    mask_slots: bool = False
    # Ablation: shuffle blob ordering in observation every step
    shuffle_blobs: bool = False
    # Ablation: random orthogonal projection of observation (preserves distances, destroys semantics)
    random_project: bool = False
    # Ablation: random permutation of observation dimensions (reorders without mixing)
    random_permute: bool = False


# =====================================================================
# Random orthogonal projection matrix (lazy-init, fixed seed)
# =====================================================================

_PROJECTION_MATRIX = None
_PERMUTATION_ORDER = None


def _get_projection_matrix():
    global _PROJECTION_MATRIX
    if _PROJECTION_MATRIX is None:
        from scipy.stats import ortho_group
        _PROJECTION_MATRIX = ortho_group.rvs(OBS_SIZE, random_state=np.random.RandomState(42_000)).astype(np.float32)
    return _PROJECTION_MATRIX


def _get_permutation_order():
    global _PERMUTATION_ORDER
    if _PERMUTATION_ORDER is None:
        _PERMUTATION_ORDER = np.random.RandomState(42_001).permutation(OBS_SIZE)
    return _PERMUTATION_ORDER


# =====================================================================
# Observation vector
# =====================================================================

def _get_observation(state: EpisodeState) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    bot = state.bot
    blobs = state.blobs
    grid = state.grid

    obs[0] = bot.pos_x / WORLD_WIDTH
    obs[1] = bot.pos_y / WORLD_HEIGHT

    # Bot state: 0=seeking, 1=paused, 2=done
    if bot.all_visited:
        obs[2] = 2.0
    elif bot.pause_timer > 0:
        obs[2] = 1.0
    else:
        obs[2] = 0.0

    # Blob positions (padded to MAX_BLOB_COUNT)
    blob_offset = 3
    for i in range(min(len(blobs), MAX_BLOB_COUNT)):
        obs[blob_offset + i * 2] = blobs[i].pos_x / WORLD_WIDTH
        obs[blob_offset + i * 2 + 1] = blobs[i].pos_y / WORLD_HEIGHT

    # Grid slot assignments
    slot_offset = blob_offset + MAX_BLOB_COUNT * 2  # 53
    max_slots = GRID_COLS * GRID_ROWS
    for i in range(min(len(blobs), MAX_BLOB_COUNT)):
        if blobs[i].grid_slot is not None:
            obs[slot_offset + i] = (blobs[i].grid_slot + 1) / max_slots
        else:
            obs[slot_offset + i] = 0.0

    # Actual blob count (index 78)
    obs[slot_offset + MAX_BLOB_COUNT] = len(blobs)

    # Phase indicator (index 79): 0=counting, 0.5=unmarking, 1=predict
    if state.phase == "predict":
        obs[slot_offset + MAX_BLOB_COUNT + 1] = 1.0
    elif state.phase == "unmarking":
        obs[slot_offset + MAX_BLOB_COUNT + 1] = 0.5
    else:
        obs[slot_offset + MAX_BLOB_COUNT + 1] = 0.0

    # Grid filled normalized (index 80)
    obs[slot_offset + MAX_BLOB_COUNT + 2] = grid.filled_count / MAX_BLOB_COUNT

    # Grid filled raw (index 81)
    obs[slot_offset + MAX_BLOB_COUNT + 3] = grid.filled_count

    # Shuffle blob ordering if requested (ablation: prevent tracking individual blobs by array index)
    if state.shuffle_blobs:
        perm = np.random.permutation(MAX_BLOB_COUNT)
        blob_positions = obs[blob_offset:blob_offset + MAX_BLOB_COUNT * 2].reshape(MAX_BLOB_COUNT, 2)
        obs[blob_offset:blob_offset + MAX_BLOB_COUNT * 2] = blob_positions[perm].flatten()

    # Mask out slot assignments if requested (ablation: agent has only blob positions)
    if state.mask_slots:
        for i in range(MAX_BLOB_COUNT):
            obs[slot_offset + i] = 0.0  # indices 53-77

    # Mask out count signals if requested (ablation: force agent to infer count from physical process)
    if state.mask_count:
        obs[slot_offset + MAX_BLOB_COUNT + 2] = 0.0  # index 80
        obs[slot_offset + MAX_BLOB_COUNT + 3] = 0.0  # index 81

    # Random orthogonal projection (preserves all pairwise distances, destroys spatial semantics)
    if state.random_project:
        obs = _get_projection_matrix() @ obs

    # Random permutation (reorders dimensions without mixing — tests coordinate expectations vs feature mixing)
    if state.random_permute:
        obs = obs[_get_permutation_order()]

    return obs


# =====================================================================
# Core: reset and step
# =====================================================================

def _reset_env(stage, conservation, blob_count_min, blob_count_max, max_steps, bidirectional, target_arrangement="grid", mask_count=False, mask_slots=False, shuffle_blobs=False, random_project=False, random_permute=False) -> EpisodeState:
    blob_count = blob_count_min + int(random.random() * (blob_count_max - blob_count_min + 1))

    # Arrangement in field zone (left 55%)
    arrangement_type = _random_arrangement()
    field_width = int(WORLD_WIDTH * 0.55)
    positions = _generate_arrangement(blob_count, field_width, WORLD_HEIGHT, MARGIN, MIN_SEPARATION, arrangement_type)

    # Create blobs
    blobs = []
    for i, (px, py) in enumerate(positions):
        blobs.append(Blob(
            id=i,
            pos_x=px, pos_y=py,
            field_pos_x=px, field_pos_y=py,
        ))

    # Create target shape (grid or line)
    grid = _create_target_grid(target_arrangement, blob_count)

    # Select personality
    personality = _select_personality(stage)

    # Random bot start in field area
    start_x = MARGIN + random.random() * (field_width - 2 * MARGIN)
    start_y = MARGIN + random.random() * (WORLD_HEIGHT - 2 * MARGIN)

    # Waypoints = blob positions
    waypoints = [(b.pos_x, b.pos_y) for b in blobs]

    bot = Bot(
        personality=personality,
        pos_x=start_x, pos_y=start_y,
        waypoints=waypoints,
        current_target_index=_select_initial_target(personality, waypoints),
        wander_angle=random.random() * math.pi * 2,
    )

    # Confused bot: compute stop-after
    if personality.name == "confused":
        bot._confused_stop_after = len(blobs) + 3 + int(random.random() * 4)

    state = EpisodeState(
        bot=bot,
        blobs=blobs,
        grid=grid,
        phase="counting",
        blob_count_at_start=len(blobs),
        bot_type=personality.name,
        arrangement_type=arrangement_type,
        bidirectional=bidirectional,
        prev_bot_x=start_x,
        prev_bot_y=start_y,
        mask_count=mask_count,
        mask_slots=mask_slots,
        shuffle_blobs=shuffle_blobs,
        random_project=random_project,
        random_permute=random_permute,
    )
    return state


def _handle_arrival(state: EpisodeState) -> bool:
    """Handle bot arriving at a waypoint. Returns whether to mark as visited."""
    bot = state.bot
    idx = bot.current_target_index
    if idx >= len(state.blobs):
        return True
    blob = state.blobs[idx]
    p_name = bot.personality.name

    if p_name == "confused":
        if blob.grid_slot is not None:
            return False  # skip already-gridded
        placed = _count_blob_to_grid(blob, idx, state.grid, bot.personality.name, bot.personality.color)
        if placed:
            forgets = random.random() < 0.15
            if bot.count_tally >= bot._confused_stop_after:
                for i in range(len(bot.waypoints)):
                    bot.visited_indices.add(i)
                return True
            return not forgets
        return False
    elif p_name == "marking":
        if blob.grid_slot is not None:
            return False
        placed = _count_blob_to_grid(blob, idx, state.grid, bot.personality.name, bot.personality.color)
        if placed:
            blob.marked_by = bot.personality.name
        return placed
    else:
        # organizing, grid, unconventional
        if blob.grid_slot is not None:
            return False
        return _count_blob_to_grid(blob, idx, state.grid, bot.personality.name, bot.personality.color)


def _mutate_blobs(blobs: list, grid: GridState):
    """Conservation OFF: 5% chance of adding/removing a blob."""
    if random.random() >= 0.05:
        return
    if random.random() < 0.5 and len(blobs) > 1:
        # Remove uncounted, non-gridded
        candidates = [i for i, b in enumerate(blobs) if not b.counted_by and b.grid_slot is None]
        if candidates:
            idx = random.choice(candidates)
            blobs.pop(idx)
            # Re-index
            for i, b in enumerate(blobs):
                b.id = i
    else:
        # Add blob
        field_max_x = WORLD_WIDTH * 0.55
        x = MARGIN + random.random() * (field_max_x - 2 * MARGIN)
        y = MARGIN + random.random() * (WORLD_HEIGHT - 2 * MARGIN)
        new_id = max((b.id for b in blobs), default=-1) + 1
        blobs.append(Blob(id=new_id, pos_x=x, pos_y=y, field_pos_x=x, field_pos_y=y))


def _step_env(state: EpisodeState, config_conservation, config_max_steps, action) -> dict:
    # Predict phase: compute reward and end
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
        return {
            "obs": _get_observation(state),
            "reward": reward,
            "done": True,
            "info": _build_info(state, config_conservation),
        }

    state.step_count += 1

    # Conservation OFF: maybe mutate
    if not config_conservation and state.phase != "unmarking":
        _mutate_blobs(state.blobs, state.grid)

    # Update blob animations (pass grid for deferred count-at-placement)
    _update_blob_animations(state.blobs, state.grid)

    bot = state.bot

    # --- Unmarking phase ---
    if state.phase == "unmarking":
        if state.uncount_pause_timer > 0:
            state.uncount_pause_timer -= 1
            bot.vel_x *= 0.85
            bot.vel_y *= 0.85
        else:
            target = _get_next_uncount_target(state.blobs, state.grid)
            if target is not None:
                bot.dynamic_target = target
                d = _dist(bot.pos_x, bot.pos_y, target[0], target[1])
                if d < UNCOUNT_ARRIVAL_RADIUS:
                    _uncount_blob_from_grid(state.blobs, state.grid, bot.personality.name)
                    state.uncount_pause_timer = UNCOUNT_PAUSE_FRAMES
                    bot.dynamic_target = None

            if state.grid.filled_count == 0:
                state.phase = "predict"

        _update_bot(bot)

        # Boundary push
        if bot.pos_x < BOUNDARY_MARGIN:
            bot.vel_x += 0.3
        if bot.pos_x > WORLD_WIDTH - BOUNDARY_MARGIN:
            bot.vel_x -= 0.3
        if bot.pos_y < BOUNDARY_MARGIN:
            bot.vel_y += 0.3
        if bot.pos_y > WORLD_HEIGHT - BOUNDARY_MARGIN:
            bot.vel_y -= 0.3

    # --- Counting phase ---
    elif not bot.all_visited:
        # Check waypoint arrival BEFORE updating bot (matches TS: arrival check is at start of update)
        if _check_waypoint_arrival(bot):
            should_mark_visited = _handle_arrival(state)
            bot.count_tally += 1
            if should_mark_visited:
                bot.visited_indices.add(bot.current_target_index)
            # Check all visited
            if len(bot.waypoints) - len(bot.visited_indices) == 0:
                bot.all_visited = True
            # Pause
            p = bot.personality
            pause_var = round((random.random() - 0.5) * 2 * p.pause_variance)
            bot.pause_timer = max(0, p.pause_duration + pause_var)
            # Select next
            if not bot.all_visited:
                bot.current_target_index = _select_next_target(bot)
        else:
            _update_bot(bot)

        # Boundary push
        if bot.pos_x < BOUNDARY_MARGIN:
            bot.vel_x += 0.3
        if bot.pos_x > WORLD_WIDTH - BOUNDARY_MARGIN:
            bot.vel_x -= 0.3
        if bot.pos_y < BOUNDARY_MARGIN:
            bot.vel_y += 0.3
        if bot.pos_y > WORLD_HEIGHT - BOUNDARY_MARGIN:
            bot.vel_y -= 0.3

    # Track distance
    dx = bot.pos_x - state.prev_bot_x
    dy = bot.pos_y - state.prev_bot_y
    state.bot_total_distance += math.sqrt(dx * dx + dy * dy)
    state.prev_bot_x = bot.pos_x
    state.prev_bot_y = bot.pos_y

    # Phase transition: counting → unmarking or predict
    if bot.all_visited and state.phase == "counting":
        # Force-finalize any blobs still flying to the grid
        for blob in state.blobs:
            if blob.pending_grid_placement:
                blob.pos_x = blob.anim_to_x
                blob.pos_y = blob.anim_to_y
                blob.animating = False
                blob.anim_progress = 0.0
                blob.pending_grid_placement = False
                state.grid.filled_count += 1
        if state.bidirectional:
            state.phase = "unmarking"
            bot.all_visited = False
            bot.dynamic_target = None
            bot.visited_indices.clear()
            bot.waypoints = []
        else:
            state.phase = "predict"

    # Truncation
    if state.step_count >= config_max_steps:
        state.truncated = True
        state.done = True
        return {
            "obs": _get_observation(state),
            "reward": 0.0,
            "done": True,
            "info": _build_info(state, config_conservation),
        }

    return {
        "obs": _get_observation(state),
        "reward": 0.0,
        "done": False,
        "info": _build_info(state, config_conservation),
    }


def _build_info(state: EpisodeState, conservation: bool) -> dict:
    return {
        "bot_tally": state.bot.count_tally,
        "blob_count_start": state.blob_count_at_start,
        "blob_count_end": len(state.blobs),
        "bot_type": state.bot_type,
        "episode_length": state.step_count,
        "bot_distance": state.bot_total_distance,
        "arrangement_type": state.arrangement_type,
        "phase": state.phase,
        "conservation": conservation,
        "truncated": state.truncated,
        "bidirectional": state.bidirectional,
        "grid_filled": state.grid.filled_count,
    }


# =====================================================================
# Gym Environment (drop-in replacement for counting_env.CountingWorldEnv)
# =====================================================================

MAX_TALLY = 25


class CountingWorldEnv(gym.Env):
    """Pure Python counting world — no Node.js subprocess."""

    metadata = {"render.modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.stage = kwargs.get("stage", 1)
        self.conservation = kwargs.get("conservation", True)
        self.blob_count_min = kwargs.get("blob_count_min", 6)
        self.blob_count_max = kwargs.get("blob_count_max", 25)
        self.max_steps = kwargs.get("max_steps", 5000)
        self.bidirectional = kwargs.get("bidirectional", False)
        self.target_arrangement = kwargs.get("target_arrangement", "grid")
        self.mask_count = kwargs.get("mask_count", False)
        self.mask_slots = kwargs.get("mask_slots", False)
        self.shuffle_blobs = kwargs.get("shuffle_blobs", False)
        self.random_project = kwargs.get("random_project", False)
        self.random_permute = kwargs.get("random_permute", False)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=float(self.blob_count_max),
            shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(MAX_TALLY)

        self._state: Optional[EpisodeState] = None
        self._current_obs: Optional[np.ndarray] = None

    def reset(self):
        self._state = _reset_env(
            self.stage, self.conservation,
            self.blob_count_min, self.blob_count_max,
            self.max_steps, self.bidirectional,
            self.target_arrangement,
            self.mask_count,
            self.mask_slots,
            self.shuffle_blobs,
            self.random_project,
            self.random_permute,
        )
        self._current_obs = _get_observation(self._state)
        return self._current_obs

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        action_val = int(action) if action is not None else None
        phase_indicator = self._current_obs[OBS_SIZE - 3] if self._current_obs is not None else 0
        send_action = action_val if phase_indicator == 1.0 else None

        result = _step_env(self._state, self.conservation, self.max_steps, send_action)
        self._current_obs = result["obs"]
        return result["obs"], result["reward"], result["done"], result["info"]

    def close(self):
        self._state = None


# Smoke test
if __name__ == "__main__":
    env = CountingWorldEnv(stage=1, conservation=True, bidirectional=True,
                           blob_count_min=10, blob_count_max=10, max_steps=10000)

    import time
    t0 = time.time()
    n_episodes = 5
    total_steps = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_steps = 0
        while not done:
            obs, reward, done, info = env.step(0)
            ep_steps += 1
        total_steps += ep_steps
        print(f"  Episode {ep+1}: {ep_steps} steps, tally={info['bot_tally']}, "
              f"grid_filled={info['grid_filled']}, phase={info['phase']}")

    elapsed = time.time() - t0
    print(f"\n{total_steps} steps in {elapsed:.2f}s = {total_steps/elapsed:.0f} steps/sec")
    env.close()
    print("Smoke test passed!")
