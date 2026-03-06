#!/usr/bin/env python3
"""
Real-time DreamerV3 counting visualization.

Shows the pure Python counting environment alongside the RSSM's internal
representation (horseshoe manifold) with live probe predictions.

Usage:
    # Headless (Docker) — saves frames, stitch with ffmpeg:
    python3 visualize_counting.py --headless --episodes 1 --output /tmp/counting_demo

    # Live window (Mac/Linux with display):
    python3 visualize_counting.py

    # Then stitch:
    ffmpeg -framerate 30 -i /tmp/counting_demo/frames/%05d.png \
           -c:v libx264 -pix_fmt yuv420p /tmp/counting_demo/counting_demo.mp4
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
BRIDGE_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(BRIDGE_SCRIPTS))

# Models dir: try multiple locations (Docker vs Mac)
_MODELS_CANDIDATES = [
    BRIDGE_SCRIPTS.parent / "models",                            # bridge/models/
    BRIDGE_SCRIPTS.parent.parent / "projects" / "jamstack-v1" / "packages" / "signal-app" / "public" / "models",  # Docker
    Path("/workspace/projects/jamstack-v1/packages/signal-app/public/models"),  # Docker absolute
    Path.home() / "anim-bridge" / "models",                     # Mac home
]
MODELS_DIR = None
for _p in _MODELS_CANDIDATES:
    if (_p / "dreamer_weights.bin").exists():
        MODELS_DIR = _p
        break

# ── Imports from existing modules ────────────────────────────────────
from counting_env_pure import (
    CountingWorldEnv,
    WORLD_WIDTH,
    WORLD_HEIGHT,
)
from counting_env_embodied import EmbodiedCountingWorldEnv
from counting_env_multidim import MultiDimCountingWorldEnv
from export_deter_centroids import FastRSSM, load_weights as _load_weights_orig


# ── State accessors (works for both 2D and multi-dim states) ─────────
def _bot_xy(state):
    """Get bot (x, y) from any state type."""
    bot = state.bot
    if hasattr(bot, 'pos_x'):
        return bot.pos_x, bot.pos_y
    return float(bot.pos[0]), float(bot.pos[1])


def _bot_dir(state):
    """Get bot facing direction (dx, dy) from any state type."""
    bot = state.bot
    if hasattr(bot, 'heading'):
        import math as _m
        return _m.cos(bot.heading), _m.sin(bot.heading)
    if hasattr(bot, 'vel_x'):
        vx, vy = bot.vel_x, bot.vel_y
    else:
        vx, vy = float(bot.vel[0]), float(bot.vel[1])
    speed = (vx * vx + vy * vy) ** 0.5
    if speed > 0.1:
        return vx / speed, vy / speed
    if hasattr(bot, 'prev_dir_x'):
        dx, dy = bot.prev_dir_x, bot.prev_dir_y
        return (dx, dy) if dx != 0 or dy != 0 else (1.0, 0.0)
    return 1.0, 0.0


def _blob_xy(blob):
    """Get blob (x, y) from any blob type."""
    if hasattr(blob, 'pos_x'):
        return blob.pos_x, blob.pos_y
    return float(blob.pos[0]), float(blob.pos[1])


def _slot_xy(slot):
    """Get grid slot (x, y) from tuple or ndarray."""
    if isinstance(slot, np.ndarray):
        return float(slot[0]), float(slot[1])
    return slot[0], slot[1]


def _bot_color(state):
    """Get bot color from any state type."""
    bot = state.bot
    if hasattr(bot, 'personality'):
        return hex_to_rgb(bot.personality.color)
    return (52, 211, 153)  # default teal


def load_weights(models_dir=None):
    """Load RSSM weights, using resolved MODELS_DIR if needed."""
    d = models_dir or MODELS_DIR
    if d is None:
        raise FileNotFoundError(
            "Cannot find dreamer_weights.bin. Use --models-dir or copy models to bridge/models/")
    # Temporarily patch the module-level paths
    import export_deter_centroids as _edc
    orig_bin = _edc.BIN_PATH
    orig_man = _edc.MANIFEST_PATH
    _edc.BIN_PATH = d / "dreamer_weights.bin"
    _edc.MANIFEST_PATH = d / "dreamer_manifest.json"
    try:
        return _load_weights_orig()
    finally:
        _edc.BIN_PATH = orig_bin
        _edc.MANIFEST_PATH = orig_man


# ── Layout constants ─────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 700
PANEL_W = 600
PANEL_H = 520
STATUS_H = SCREEN_H - PANEL_H  # 180

# Colors
BG = (18, 18, 24)
PANEL_BG = (24, 24, 32)
STATUS_BG = (14, 14, 20)
GRID_EMPTY = (50, 50, 70)
GRID_FILLED = (96, 165, 250)
BLOB_FIELD = (52, 145, 130)
BLOB_ANIM = (251, 191, 36)
BLOB_GRID = (96, 165, 250)
TRAIL_COLOR = (120, 255, 120)
LIVE_DOT = (80, 255, 120)
SEPARATOR = (60, 60, 80)
TEXT_WHITE = (240, 240, 240)
TEXT_DIM = (120, 120, 140)
TEXT_GREEN = (74, 222, 128)
TEXT_YELLOW = (250, 204, 21)
TEXT_RED = (248, 113, 113)
CENTROID_LINE = (60, 60, 80)

# Trail
TRAIL_LEN = 80


def hex_to_rgb(h: str) -> tuple:
    """'#34D399' → (52, 211, 153)"""
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ── viridis-ish colormap (26 values) ────────────────────────────────
def viridis(t: float) -> tuple:
    """Simple viridis approximation, t in [0,1] → (r,g,b)."""
    # Key stops: dark purple → teal → yellow
    if t < 0.25:
        s = t / 0.25
        return (int(68 + s * (-4)), int(1 + s * 83), int(84 + s * 76))
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (int(64 - s * 31), int(84 + s * 80), int(160 - s * 20))
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (int(33 + s * 60), int(164 + s * 25), int(140 - s * 50))
    else:
        s = (t - 0.75) / 0.25
        return (int(93 + s * 160), int(189 + s * 50), int(90 - s * 60))


# ── Data loading ─────────────────────────────────────────────────────
def load_model_data(models_dir=None, randproj=False, explicit_dir=False) -> dict:
    """Load PCA components/mean/centroids and linear probe from JSON."""
    d = models_dir or MODELS_DIR
    # When --models-dir is explicit, files are named normally (embed_pca.json).
    # When using default dir, randproj files use *_randproj.json suffix.
    use_suffix = randproj and not explicit_dir
    pca_file = "embed_pca_randproj.json" if use_suffix else "embed_pca.json"
    probe_file = "embed_probe_randproj.json" if use_suffix else "embed_probe.json"
    with open(d / pca_file) as f:
        pca = json.load(f)
    with open(d / probe_file) as f:
        probe = json.load(f)

    # Build 512-dim centroid matrix for nearest-centroid prediction
    c512 = pca.get("centroids_512", {})
    n_centroids = len(c512)
    centroid_matrix = np.zeros((n_centroids, 512), dtype=np.float32)
    centroid_labels = np.zeros(n_centroids, dtype=np.int32)
    for idx, (count_str, vec) in enumerate(sorted(c512.items(), key=lambda x: int(x[0]))):
        centroid_matrix[idx] = vec
        centroid_labels[idx] = int(count_str)

    result = {
        "pca_components": np.array(pca["pca_components"], dtype=np.float32),   # (2, 512)
        "pca_mean": np.array(pca["pca_mean"], dtype=np.float32),               # (512,)
        "centroids_2d": np.array(pca["centroids_2d"], dtype=np.float32),       # (26, 2)
        "centroid_counts": np.array(pca["centroid_counts"], dtype=np.float32), # (26,)
        "probe_weights": np.array(probe["weights"], dtype=np.float32),         # (512,)
        "probe_bias": float(probe["bias"]),
        "centroid_matrix": centroid_matrix,    # (26, 512) for nearest-centroid
        "centroid_labels": centroid_labels,    # (26,) count labels
    }

    # Load Bayesian parameters if present in probe JSON
    if "bayesian_gaussians" in probe:
        result["bayesian_means"] = np.array(
            [g["mean"] for g in probe["bayesian_gaussians"]], dtype=np.float64)
        result["bayesian_stds"] = np.array(
            [g["std"] for g in probe["bayesian_gaussians"]], dtype=np.float64)
    if "bayesian_boundaries" in probe:
        result["bayesian_boundaries"] = np.array(
            probe["bayesian_boundaries"], dtype=np.float64)

    # Load training point cloud if available
    states_file = d / "baseline_states.npz"
    if states_file.exists():
        data = np.load(states_file)
        h_t = data["h_t"]      # (N, 512)
        counts = data["counts"]  # (N,)
        # Project through same PCA as centroids
        pca_comp = result["pca_components"]
        pca_mean = result["pca_mean"]
        points_2d = (h_t - pca_mean) @ pca_comp.T  # (N, 2)
        result["cloud_points"] = points_2d.astype(np.float32)
        result["cloud_counts"] = counts
    return result


# ── Coordinate transforms ───────────────────────────────────────────
def make_world_transform():
    """Returns fn: world (x,y) → screen (sx,sy) for left panel."""
    scale = min((PANEL_W - 20) / WORLD_WIDTH, (PANEL_H - 20) / WORLD_HEIGHT)
    scaled_w = WORLD_WIDTH * scale
    scaled_h = WORLD_HEIGHT * scale
    ox = (PANEL_W - scaled_w) / 2
    oy = (PANEL_H - scaled_h) / 2
    def transform(wx, wy):
        return int(ox + wx * scale), int(oy + wy * scale)
    return transform, scale


def make_pca_transform(centroids_2d: np.ndarray):
    """Returns fn: pca (px,py) → screen (sx,sy) for right panel."""
    margin = 40
    pad = 0.15
    xs = centroids_2d[:, 0]
    ys = centroids_2d[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * pad
    x_max += x_range * pad
    y_min -= y_range * pad
    y_max += y_range * pad
    x_range = x_max - x_min
    y_range = y_max - y_min

    draw_w = PANEL_W - 2 * margin
    draw_h = PANEL_H - 2 * margin
    scale = min(draw_w / x_range, draw_h / y_range)
    cx = PANEL_W + margin + draw_w / 2
    cy = margin + draw_h / 2

    def transform(px, py):
        sx = cx + (px - (x_min + x_max) / 2) * scale
        sy = cy + (py - (y_min + y_max) / 2) * scale
        return int(sx), int(sy)
    return transform


# ── Drawing functions ────────────────────────────────────────────────
def draw_world(screen, pygame, state, w2s, scale):
    """Draw environment: grid, blobs, bot. Works with 2D, embodied, and multi-dim states."""
    # Background for left panel
    pygame.draw.rect(screen, PANEL_BG, (0, 0, PANEL_W, PANEL_H))

    grid = state.grid
    blobs = state.blobs

    # Grid slots
    slot_r = max(3, int(20 * scale))
    for i, slot in enumerate(grid.slots):
        sx, sy = _slot_xy(slot)
        px, py = w2s(sx, sy)
        if grid.occupancy[i] != -1:
            pygame.draw.rect(screen, GRID_FILLED,
                             (px - slot_r, py - slot_r, slot_r * 2, slot_r * 2))
        else:
            pygame.draw.rect(screen, GRID_EMPTY,
                             (px - slot_r, py - slot_r, slot_r * 2, slot_r * 2), 1)

    # Blobs
    blob_r = max(2, int(12 * scale))
    for b in blobs:
        bx, by = _blob_xy(b)
        px, py = w2s(bx, by)
        if b.animating:
            color = BLOB_ANIM
        elif b.grid_slot is not None:
            color = BLOB_GRID
        else:
            color = BLOB_FIELD
        pygame.draw.circle(screen, color, (px, py), blob_r)

    # Bot — triangle pointing in facing direction
    bot_x, bot_y = _bot_xy(state)
    bx, by = w2s(bot_x, bot_y)
    dx, dy = _bot_dir(state)

    bot_size = max(4, int(18 * scale))
    tip = (bx + dx * bot_size, by + dy * bot_size)
    perp_x, perp_y = -dy, dx
    left = (bx - dx * bot_size * 0.6 + perp_x * bot_size * 0.5,
            by - dy * bot_size * 0.6 + perp_y * bot_size * 0.5)
    right = (bx - dx * bot_size * 0.6 - perp_x * bot_size * 0.5,
             by - dy * bot_size * 0.6 - perp_y * bot_size * 0.5)
    pygame.draw.polygon(screen, _bot_color(state), [tip, left, right])

    # Panel label
    font_sm = pygame.font.SysFont("monospace", 13)
    label = font_sm.render("Environment", True, TEXT_DIM)
    screen.blit(label, (10, 5))


def prerender_point_cloud(pygame, cloud_points, cloud_counts, p2s):
    """Pre-render training point cloud as a static SRCALPHA surface."""
    surf = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    n = len(cloud_points)
    # Subsample if too many points (keep rendering fast)
    max_pts = 12000
    if n > max_pts:
        idx = np.random.RandomState(0).choice(n, max_pts, replace=False)
        pts = cloud_points[idx]
        cts = cloud_counts[idx]
    else:
        pts, cts = cloud_points, cloud_counts

    for i in range(len(pts)):
        px, py = p2s(pts[i, 0], pts[i, 1])
        # Shift to panel-local coords (p2s returns screen coords with PANEL_W offset)
        lx = px - PANEL_W
        ly = py
        if 0 <= lx < PANEL_W and 0 <= ly < PANEL_H:
            c = int(cts[i])
            r, g, b = viridis(c / 25.0)
            surf.set_at((lx, ly), (r, g, b, 30))
            # 2px dot: fill adjacent pixels for visibility
            for dx, dy in [(1, 0), (0, 1), (1, 1)]:
                nx, ny = lx + dx, ly + dy
                if 0 <= nx < PANEL_W and 0 <= ny < PANEL_H:
                    surf.set_at((nx, ny), (r, g, b, 22))
    return surf


def draw_manifold(screen, pygame, centroids_2d, counts, live_2d, trail, p2s,
                  cloud_surface=None):
    """Draw horseshoe manifold with centroid backbone, trail, and live dot."""
    # Background for right panel
    pygame.draw.rect(screen, PANEL_BG, (PANEL_W, 0, PANEL_W, PANEL_H))

    # Point cloud layer (pre-rendered)
    if cloud_surface is not None:
        screen.blit(cloud_surface, (PANEL_W, 0))

    n = len(centroids_2d)

    # Centroid backbone line
    points = [p2s(centroids_2d[i, 0], centroids_2d[i, 1]) for i in range(n)]
    if len(points) >= 2:
        pygame.draw.lines(screen, CENTROID_LINE, False, points, 2)

    # Centroid dots with viridis coloring
    label_set = {0, 5, 10, 15, 20, 25}
    font_sm = pygame.font.SysFont("monospace", 11)
    for i in range(n):
        c = int(counts[i])
        color = viridis(c / 25.0)
        px, py = points[i]
        pygame.draw.circle(screen, color, (px, py), 5)
        if c in label_set:
            lbl = font_sm.render(str(c), True, TEXT_DIM)
            screen.blit(lbl, (px + 7, py - 6))

    # Trail (fading)
    trail_list = list(trail)
    for i, pt in enumerate(trail_list):
        alpha = int(40 + 180 * (i / max(len(trail_list) - 1, 1)))
        px, py = p2s(pt[0], pt[1])
        r = max(1, 2 + int(3 * i / max(len(trail_list) - 1, 1)))
        # Use alpha surface for trail dots
        s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        brightness = int(80 + 175 * (i / max(len(trail_list) - 1, 1)))
        s_color = (brightness, 255, brightness, alpha)
        pygame.draw.circle(s, s_color, (r + 1, r + 1), r)
        screen.blit(s, (px - r - 1, py - r - 1))

    # Live dot with glow
    if live_2d is not None:
        lx, ly = p2s(live_2d[0], live_2d[1])
        # Glow rings
        glow = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow, (80, 255, 120, 30), (20, 20), 18)
        pygame.draw.circle(glow, (80, 255, 120, 60), (20, 20), 12)
        screen.blit(glow, (lx - 20, ly - 20))
        # Solid center
        pygame.draw.circle(screen, LIVE_DOT, (lx, ly), 6)

    # Panel label
    label = font_sm.render("RSSM Manifold (PCA)", True, TEXT_DIM)
    screen.blit(label, (PANEL_W + 10, 5))


def draw_status(screen, pygame, gt, pred, raw, phase, episode, step, total_exact, total_steps,
                model_label="", embodied=False):
    """Draw bottom status bar with counts, phase, accuracy."""
    pygame.draw.rect(screen, STATUS_BG, (0, PANEL_H, SCREEN_W, STATUS_H))

    font_lg = pygame.font.SysFont("monospace", 38)
    font_md = pygame.font.SysFont("monospace", 22)
    font_sm = pygame.font.SysFont("monospace", 14)

    y_base = PANEL_H + 20

    # Ground truth
    gt_label = font_md.render("Ground Truth:", True, TEXT_DIM)
    gt_val = font_lg.render(str(gt), True, TEXT_WHITE)
    screen.blit(gt_label, (30, y_base))
    screen.blit(gt_val, (30, y_base + 30))

    # Predicted
    diff = abs(gt - pred)
    if diff == 0:
        pred_color = TEXT_GREEN
    elif diff == 1:
        pred_color = TEXT_YELLOW
    else:
        pred_color = TEXT_RED

    pred_label = font_md.render("Predicted:", True, TEXT_DIM)
    pred_val = font_lg.render(str(pred), True, pred_color)
    screen.blit(pred_label, (250, y_base))
    screen.blit(pred_val, (250, y_base + 30))

    # Raw probe value
    raw_text = font_sm.render(f"raw: {raw:.2f}", True, TEXT_DIM)
    screen.blit(raw_text, (250, y_base + 78))

    # Phase badge
    phase_upper = phase.upper()
    badge_colors = {
        "COUNTING": (52, 211, 153),
        "UNMARKING": (251, 191, 36),
        "PREDICT": (248, 113, 113),
    }
    badge_color = badge_colors.get(phase_upper, TEXT_DIM)
    phase_text = font_md.render(phase_upper, True, badge_color)
    screen.blit(phase_text, (480, y_base))

    # Episode / step
    info_text = font_sm.render(f"Episode {episode}  |  Step {step}", True, TEXT_DIM)
    screen.blit(info_text, (480, y_base + 35))

    # Running accuracy
    if total_steps > 0:
        exact_pct = 100 * total_exact / total_steps
        within1 = exact_pct  # we track ±1 separately below
        acc_text = font_sm.render(f"Exact: {exact_pct:.1f}%", True, TEXT_DIM)
        screen.blit(acc_text, (480, y_base + 58))

    # Separator line at top of status bar
    pygame.draw.line(screen, SEPARATOR, (0, PANEL_H), (SCREEN_W, PANEL_H), 1)

    # Title
    if embodied:
        title = font_md.render("DreamerV3 World Model — Embodied", True, TEXT_WHITE)
        screen.blit(title, (700, y_base))
        subtitle = font_sm.render("agent-controlled steering", True, TEXT_DIM)
        screen.blit(subtitle, (700, y_base + 30))
        subtitle2 = font_sm.render("navigates to blobs, picks up", True, TEXT_DIM)
        screen.blit(subtitle2, (700, y_base + 50))
        subtitle3 = font_sm.render("on proximity, fills grid", True, TEXT_DIM)
        screen.blit(subtitle3, (700, y_base + 68))
    else:
        title = font_md.render("DreamerV3 World Model — Counting", True, TEXT_WHITE)
        screen.blit(title, (700, y_base))
        subtitle = font_sm.render("passive observer, zero embodiment", True, TEXT_DIM)
        screen.blit(subtitle, (700, y_base + 30))
        subtitle2 = font_sm.render("learns numerical structure from", True, TEXT_DIM)
        screen.blit(subtitle2, (700, y_base + 50))
        subtitle3 = font_sm.render("observation alone", True, TEXT_DIM)
        screen.blit(subtitle3, (700, y_base + 68))
    if model_label:
        ml = font_sm.render(model_label, True, (140, 140, 180))
        screen.blit(ml, (700, y_base + 88))


# ── Embodied heuristic steering ──────────────────────────────────────
def heuristic_steering(state) -> float:
    """Simple approach-nearest-blob heuristic. Returns steering angle in radians."""
    import math
    bot = state.bot
    best_d = float("inf")
    target_x, target_y = None, None

    for blob in state.blobs:
        if blob.grid_slot is not None or blob.pending_grid_placement:
            continue
        dx = blob.pos_x - bot.pos_x
        dy = blob.pos_y - bot.pos_y
        d = math.sqrt(dx * dx + dy * dy)
        if d < best_d:
            best_d = d
            target_x, target_y = blob.pos_x, blob.pos_y

    if target_x is None:
        # All blobs placed — just keep going straight
        return bot.heading

    return math.atan2(target_y - bot.pos_y, target_x - bot.pos_x)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DreamerV3 counting visualization")
    parser.add_argument("--headless", action="store_true",
                        help="No display — save frames as PNGs")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes to run")
    parser.add_argument("--output", type=str, default="/tmp/counting_demo",
                        help="Output directory for headless frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (live mode)")
    parser.add_argument("--blob-count", type=int, default=25,
                        help="Fixed blob count")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Path to models directory (dreamer_weights.bin, embed_*.json)")
    parser.add_argument("--randproj", action="store_true",
                        help="Use random projection (82x82 orthogonal, seed 42000)")
    parser.add_argument("--embodied", action="store_true",
                        help="Embodied mode: agent-controlled steering with heuristic policy")
    parser.add_argument("--multidim", type=int, default=None, metavar="D",
                        help="Multi-dim mode: use D-dimensional counting env (e.g. --multidim 2)")
    parser.add_argument("--prediction", choices=["round", "bayes", "bayes+monotonic"],
                        default="round",
                        help="Prediction mode: round (default), bayes (Gaussian discriminant), "
                             "bayes+monotonic (Bayesian + monotonicity during gathering)")
    args = parser.parse_args()

    # Override MODELS_DIR if specified
    global MODELS_DIR
    if args.models_dir:
        MODELS_DIR = Path(args.models_dir)
    elif args.embodied:
        # Default to embodied_baseline subdir
        _ep = BRIDGE_SCRIPTS.parent / "models" / "embodied_baseline"
        if (_ep / "dreamer_weights.bin").exists():
            MODELS_DIR = _ep
    elif args.multidim is not None:
        # Default to multidim_d{N} subdir
        _ep = BRIDGE_SCRIPTS.parent / "models" / f"multidim_d{args.multidim}"
        if (_ep / "dreamer_weights.bin").exists():
            MODELS_DIR = _ep
    if MODELS_DIR is None:
        print("ERROR: Cannot find model files. Use --models-dir /path/to/models/")
        sys.exit(1)

    headless = args.headless

    # Set SDL driver before importing pygame
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame
    pygame.init()

    if headless:
        screen = pygame.Surface((SCREEN_W, SCREEN_H))
        frames_dir = Path(args.output) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Headless mode — saving frames to {frames_dir}")
    else:
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("DreamerV3 Counting Visualization")

    clock = pygame.time.Clock()

    # Random projection matrix (82x82 orthogonal, seed 42000)
    proj_matrix = None
    if args.randproj:
        from scipy.stats import ortho_group
        proj_matrix = ortho_group.rvs(82, random_state=np.random.RandomState(42_000)).astype(np.float32)
        print("Random projection mode — 82x82 orthogonal matrix (seed 42000)")

    # Load model data
    print(f"Loading model weights from {MODELS_DIR} ...")
    weights = load_weights(MODELS_DIR)
    # Determine obs dimension from encoder weight shape (82 for 2D, 128 for multi-dim)
    rssm_obs_size = weights["enc_linear0_w"].shape[1]
    print(f"  RSSM obs dimension: {rssm_obs_size}")
    model_data = load_model_data(MODELS_DIR, randproj=args.randproj,
                                  explicit_dir=args.models_dir is not None)
    pca_components = model_data["pca_components"]  # (2, 512)
    pca_mean = model_data["pca_mean"]              # (512,)
    centroids_2d = model_data["centroids_2d"]      # (26, 2)
    centroid_counts = model_data["centroid_counts"] # (26,)
    probe_w = model_data["probe_weights"]          # (512,)
    probe_b = model_data["probe_bias"]
    centroid_matrix = model_data["centroid_matrix"]  # (26, 512)
    centroid_labels = model_data["centroid_labels"]  # (26,)

    # Bayesian prediction parameters
    bayes_means = model_data.get("bayesian_means")    # (26,) or None
    bayes_stds = model_data.get("bayesian_stds")      # (26,) or None
    prediction_mode = args.prediction
    if prediction_mode != "round" and bayes_means is None:
        print(f"WARNING: --prediction {prediction_mode} requires Bayesian parameters in "
              f"embed_probe.json. Run bayesian_probe_optimizer.py --export-probe first. "
              f"Falling back to 'round'.")
        prediction_mode = "round"
    if prediction_mode != "round":
        print(f"  Prediction mode: {prediction_mode}")
    prev_predicted = 0  # for monotonic mode

    # Coordinate transforms
    w2s, world_scale = make_world_transform()
    p2s = make_pca_transform(centroids_2d)

    # Pre-render training point cloud
    cloud_surface = None
    if "cloud_points" in model_data:
        print(f"Pre-rendering point cloud ({len(model_data['cloud_points'])} states)...")
        cloud_surface = prerender_point_cloud(
            pygame, model_data["cloud_points"], model_data["cloud_counts"], p2s)
        print("  Done.")

    # Environment + RSSM
    embodied = args.embodied
    multidim = args.multidim
    num_actions = weights["img_in_w"].shape[1] - 1024
    if embodied:
        env = EmbodiedCountingWorldEnv(
            blob_count_min=args.blob_count,
            blob_count_max=args.blob_count,
            max_steps=8000,
        )
        print(f"Embodied mode: heuristic steering, {num_actions} action dims")
    elif multidim is not None:
        env = MultiDimCountingWorldEnv(
            blob_count_min=args.blob_count,
            blob_count_max=args.blob_count,
            fixed_dim=multidim,
            proj_dim=rssm_obs_size,
        )
        print(f"Multi-dim mode: D={multidim}, proj_dim={rssm_obs_size}, {num_actions} action dims")
    else:
        env = CountingWorldEnv(
            blob_count_min=args.blob_count,
            blob_count_max=args.blob_count,
            conservation=True,
        )
    rssm = FastRSSM(weights)

    # State
    trail = deque(maxlen=TRAIL_LEN)
    frame_num = 0
    episode = 0
    total_exact = 0
    total_counted = 0  # steps where phase == counting (meaningful predictions)
    restart_countdown = 0

    obs = env.reset()
    rssm.reset()
    episode += 1
    episodes_done = 0

    running = True
    while running:
        # Handle events
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        rssm.reset()
                        trail.clear()
                        restart_countdown = 0
                        prev_predicted = 0

        # Episode restart countdown
        if restart_countdown > 0:
            restart_countdown -= 1
            if restart_countdown == 0:
                obs = env.reset()
                rssm.reset()
                trail.clear()
                prev_predicted = 0
                episode += 1
                episodes_done += 1
                if episodes_done >= args.episodes:
                    running = False
                    continue
        else:
            # Process current obs through RSSM BEFORE stepping env
            # Multi-dim env returns pre-projected obs; 2D envs need slicing
            if multidim is not None:
                obs_for_rssm = obs[:rssm_obs_size].astype(np.float32)
            else:
                obs_for_rssm = obs.copy()
                if proj_matrix is not None:
                    obs_for_rssm = (proj_matrix @ obs_for_rssm.astype(np.float32))
                obs_for_rssm = obs_for_rssm[:rssm_obs_size].astype(np.float32)

            # Compute action for RSSM
            if embodied:
                import math as _m
                steering = heuristic_steering(env._state)
                rssm_action = np.zeros(num_actions, dtype=np.float32)
                rssm_action[0] = steering / _m.pi  # normalize to [-1, 1]
            else:
                rssm_action = 0.0

            # RSSM step
            deter = rssm.step(obs_for_rssm, rssm_action)

            # PCA projection
            live_2d = (deter - pca_mean) @ pca_components.T  # (2,)
            trail.append(live_2d.copy())

            # Probe prediction
            raw_probe = float(deter @ probe_w + probe_b)
            if prediction_mode == "round":
                predicted = int(np.clip(np.round(raw_probe), 0, 25))
            else:
                # Bayesian: pick count c maximizing N(raw_probe; mu_c, sigma_c)
                log_probs = -0.5 * ((raw_probe - bayes_means) / bayes_stds) ** 2 - np.log(bayes_stds)
                predicted = int(np.argmax(log_probs))
                if prediction_mode == "bayes+monotonic":
                    _phase = env._state.phase
                    _gathering = _phase in ("counting", "gathering")
                    if _gathering:
                        if predicted < prev_predicted:
                            predicted = prev_predicted  # can't decrease during gathering
                        elif predicted > prev_predicted + 1:
                            predicted = prev_predicted + 1  # max +1 per step
                    else:
                        prev_predicted = 0  # reset at non-gathering phases
                prev_predicted = predicted
            # Ground truth: from obs[81] for 2D envs, from state for multi-dim
            if multidim is not None:
                gt = int(env._state.grid.filled_count)
            else:
                gt = int(obs[81])  # grid_filled_raw

            # Track accuracy during counting phase
            state = env._state
            phase_name = state.phase
            # Embodied uses "gathering" vs passive uses "counting"
            is_counting = phase_name in ("counting", "gathering")
            if is_counting:
                total_counted += 1
                if predicted == gt:
                    total_exact += 1

            # Step environment for next frame
            if embodied:
                obs, reward, done, info = env.step([steering])
            elif multidim is not None:
                obs, reward, done, info = env.step(-0.995)
            else:
                obs, reward, done, info = env.step(0)

            if done:
                restart_countdown = 90  # 3 seconds at 30fps

        # ── Render ───────────────────────────────────────────────────
        screen.fill(BG)

        state = env._state

        # Left panel: environment
        draw_world(screen, pygame, state, w2s, world_scale)

        # Separator
        pygame.draw.line(screen, SEPARATOR, (PANEL_W, 0), (PANEL_W, PANEL_H), 1)

        # Right panel: manifold
        if restart_countdown > 0:
            live_pt = trail[-1] if trail else None
        else:
            live_pt = live_2d if 'live_2d' in dir() else None
        draw_manifold(screen, pygame, centroids_2d, centroid_counts,
                      live_pt, trail, p2s, cloud_surface=cloud_surface)

        # Status bar
        if restart_countdown > 0:
            gt_show = gt if 'gt' in dir() else 0
            pred_show = predicted if 'predicted' in dir() else 0
            raw_show = raw_probe if 'raw_probe' in dir() else 0.0
        else:
            gt_show = gt
            pred_show = predicted
            raw_show = raw_probe

        step_num = state.step_count if state else 0
        if embodied:
            mlabel = "embodied baseline (heuristic steering)"
        elif multidim is not None:
            mlabel = f"multi-dim D={multidim} (128-dim projection)"
        elif args.randproj:
            mlabel = "random projection (seed 42000)"
        else:
            mlabel = "baseline"
        # Map embodied phase names to display names
        phase_display = state.phase if state else "counting"
        if phase_display == "gathering":
            phase_display = "counting"
        draw_status(screen, pygame, gt_show, pred_show, raw_show,
                    phase_display,
                    episode, step_num, total_exact, total_counted,
                    model_label=mlabel,
                    embodied=embodied)

        # Save or display
        if headless:
            frame_num += 1
            pygame.image.save(screen, str(frames_dir / f"{frame_num:05d}.png"))
            if frame_num % 100 == 0:
                pct = total_exact / max(total_counted, 1) * 100
                print(f"  Frame {frame_num}, step {step_num}, "
                      f"gt={gt_show}, pred={pred_show}, exact={pct:.1f}%")
        else:
            pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()

    if headless:
        pct = total_exact / max(total_counted, 1) * 100
        print(f"\nDone. {frame_num} frames saved to {frames_dir}")
        print(f"Accuracy: {pct:.1f}% exact over {total_counted} counting steps")
        print(f"\nTo create video:")
        print(f"  ffmpeg -framerate 30 -i {frames_dir}/%05d.png \\")
        print(f"         -c:v libx264 -pix_fmt yuv420p {args.output}/counting_demo.mp4")


if __name__ == "__main__":
    main()
