#!/usr/bin/env python3
"""
Synchronized 2D World + 3D Informed Hidden State Visualization.

Left panel:  The counting world — bot gathers blobs into a grid.
Right panel: The model's 512-dim hidden state projected onto 3 informed axes:
  Axis 1: Counting direction (probe weight vector)
  Axis 2: Spatial state (within-count variance, orthogonal to Axis 1)
  Axis 3: Transition dynamics (velocity at count changes, orthogonal to 1 & 2)

Controls:
  Space         Play/pause
  Left/Right    Step backward/forward one frame
  Shift+L/R     Jump to previous/next transition
  Up/Down       Adjust speed (0.25x, 0.5x, 1x, 2x, 4x)
  A             Toggle anticipation overlay (prior dot)
  T             Toggle transition detail mode (auto slow-mo + annotations)
  D             Toggle dimension spotlight (top counting dim bar)
  P             Toggle probe number line
  H             Toggle PCA horseshoe inset
  R             Reset to beginning
  Mouse drag    Rotate 3D view (right panel)
  Scroll        Zoom 3D view

Usage:
    python3 educational_viz.py --episode path/to/episode_data.npz
    python3 educational_viz.py  # uses default path
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    print("pygame required: pip install pygame")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SCREEN_W, SCREEN_H = 1400, 800
PANEL_W = SCREEN_W // 2  # 700 each
PANEL_H = 620
STATUS_H = SCREEN_H - PANEL_H  # 180

# Colors
BG = (14, 14, 20)
PANEL_BG = (22, 22, 30)
GRID_EMPTY = (45, 45, 65)
GRID_FILLED = (90, 160, 245)
BLOB_FIELD = (48, 140, 125)
BLOB_ANIM = (248, 190, 34)
BLOB_GRID = (90, 160, 245)
BOT_COLOR = (52, 211, 153)
TRAIL_COLOR = (80, 255, 120)
TEXT_COLOR = (235, 235, 235)
TEXT_DIM = (110, 110, 130)
TEXT_GREEN = (74, 222, 128)
TEXT_YELLOW = (250, 204, 21)
TEXT_RED = (248, 113, 113)
AXIS_COLOR = (70, 70, 90)
CENTROID_LINE = (60, 60, 80)

# 3D view defaults
DEFAULT_AZIMUTH = -25.0
DEFAULT_ELEVATION = 20.0
DEFAULT_ZOOM = 1.0

# Playback speeds
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_IDX = 2  # 1.0x

FPS = 60
BASE_STEP_RATE = 30  # frames per second at 1x speed


# ═══════════════════════════════════════════════════════════════════════════
# Viridis-like color ramp (count 0=dark purple, max=yellow)
# ═══════════════════════════════════════════════════════════════════════════

def viridis(t):
    """Attempt viridis-ish colormap. t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        s = t / 0.25
        r = int(68 * (1 - s) + 33 * s)
        g = int(1 * (1 - s) + 145 * s)
        b = int(84 * (1 - s) + 140 * s)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r = int(33 * (1 - s) + 32 * s)
        g = int(145 * (1 - s) + 190 * s)
        b = int(140 * (1 - s) + 107 * s)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r = int(32 * (1 - s) + 160 * s)
        g = int(190 * (1 - s) + 218 * s)
        b = int(107 * (1 - s) + 57 * s)
    else:
        s = (t - 0.75) / 0.25
        r = int(160 * (1 - s) + 253 * s)
        g = int(218 * (1 - s) + 231 * s)
        b = int(57 * (1 - s) + 37 * s)
    return (r, g, b)


# ═══════════════════════════════════════════════════════════════════════════
# 3D projection engine (custom, no matplotlib dependency)
# ═══════════════════════════════════════════════════════════════════════════

def rotation_matrix(azimuth_deg, elevation_deg):
    """Build a rotation matrix from azimuth and elevation angles."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    # Rotation around Y (azimuth), then around X (elevation)
    cos_a, sin_a = math.cos(az), math.sin(az)
    cos_e, sin_e = math.cos(el), math.sin(el)
    # Combined rotation matrix
    R = np.array([
        [cos_a, 0, sin_a],
        [sin_a * sin_e, cos_e, -cos_a * sin_e],
        [-sin_a * cos_e, sin_e, cos_a * cos_e],
    ], dtype=np.float64)
    return R


def project_3d_to_2d(points_3d, azimuth, elevation, zoom, cx, cy):
    """Project 3D points to 2D screen coordinates.

    points_3d: (N, 3) array
    Returns: (N, 2) screen coords, (N,) depth values
    """
    R = rotation_matrix(azimuth, elevation)
    rotated = (R @ points_3d.T).T  # (N, 3)
    scale = 1.0 * zoom
    screen_x = cx + rotated[:, 0] * scale
    screen_y = cy - rotated[:, 1] * scale  # Y flipped for screen
    depth = rotated[:, 2]
    return np.column_stack([screen_x, screen_y]), depth


# ═══════════════════════════════════════════════════════════════════════════
# Compute informed axes from episode data
# ═══════════════════════════════════════════════════════════════════════════

def compute_informed_axes(data):
    """Compute 3 orthogonal, semantically meaningful axes from hidden states.

    Axis 1: Probe direction (counting)
    Axis 2: PC1 of residuals after removing Axis 1 (spatial state)
    Axis 3: PC1 of transition velocities, orthogonalized (transition dynamics)
    """
    deter = data["deter"]  # (T, 512)
    gt = data["gt_count"]  # (T,)
    transitions = data["transition"]  # (T,)

    # --- Axis 1: Probe direction ---
    probe_w = data["probe_weights"]  # (512,)
    axis1 = probe_w / np.linalg.norm(probe_w)

    # --- Axis 2: Spatial state (WITHIN-count variance, orthogonal to axis1) ---
    # Remove counting direction from all hidden states
    projections_on_1 = deter @ axis1  # (T,)
    residuals = deter - np.outer(projections_on_1, axis1)  # (T, 512)
    # Compute within-class scatter matrix (LDA-style)
    # This captures variance WITHIN each count, not between counts
    counts = sorted(set(gt))
    S_w = np.zeros((512, 512), dtype=np.float64)
    for c in counts:
        mask = gt == c
        if mask.sum() < 2:
            continue
        res_c = residuals[mask]
        mean_c = res_c.mean(axis=0)
        centered_c = res_c - mean_c
        S_w += centered_c.T @ centered_c
    # PC1 of within-class scatter = direction of maximum within-count variance
    eigvals, eigvecs = np.linalg.eigh(S_w)
    axis2 = eigvecs[:, -1]  # largest eigenvalue
    # Ensure orthogonal to axis1 (should already be, but numerical safety)
    axis2 = axis2 - (axis2 @ axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)

    # --- Axis 3: Transition dynamics (velocity at transitions) ---
    trans_idx = np.where(transitions)[0]
    if len(trans_idx) > 1:
        # Velocity vectors at transition points
        velocities = []
        for idx in trans_idx:
            if idx > 0:
                v = deter[idx] - deter[idx - 1]
                velocities.append(v)
        if len(velocities) > 2:
            vel_arr = np.array(velocities)
            vel_mean = vel_arr.mean(axis=0)
            vel_centered = vel_arr - vel_mean
            _, _, Vt_vel = np.linalg.svd(vel_centered, full_matrices=False)
            axis3 = Vt_vel[0]
        else:
            # Fallback: use mean velocity direction
            axis3 = np.mean(velocities, axis=0)
            axis3 = axis3 / np.linalg.norm(axis3)
    else:
        # Fallback: use PCA PC2 of residuals
        axis3 = Vt[1]

    # Gram-Schmidt: orthogonalize axis3 against axis1 and axis2
    axis3 = axis3 - (axis3 @ axis1) * axis1
    axis3 = axis3 - (axis3 @ axis2) * axis2
    norm3 = np.linalg.norm(axis3)
    if norm3 < 1e-8:
        # Degenerate: use PC2 of residuals
        axis3 = Vt[1]
        axis3 = axis3 - (axis3 @ axis1) * axis1
        axis3 = axis3 - (axis3 @ axis2) * axis2
        axis3 = axis3 / np.linalg.norm(axis3)
    else:
        axis3 = axis3 / norm3

    return axis1, axis2, axis3


def compute_3d_positions(data, axis1, axis2, axis3):
    """Project all hidden states onto the 3 informed axes."""
    deter = data["deter"]  # (T, 512)
    # Center on mean
    mean = deter.mean(axis=0)
    centered = deter - mean
    coords = np.column_stack([
        centered @ axis1,
        centered @ axis2,
        centered @ axis3,
    ])
    return coords, mean


def compute_centroids_3d(data, axis1, axis2, axis3, mean):
    """Compute per-count centroids in the 3D informed space."""
    deter = data["deter"]
    gt = data["gt_count"]
    counts = sorted(set(gt))
    centroids = {}
    for c in counts:
        mask = gt == c
        if mask.sum() > 0:
            cent = deter[mask].mean(axis=0) - mean
            centroids[c] = np.array([cent @ axis1, cent @ axis2, cent @ axis3])
    return centroids


# ═══════════════════════════════════════════════════════════════════════════
# PCA horseshoe (for the inset toggle)
# ═══════════════════════════════════════════════════════════════════════════

def compute_pca_2d(data):
    """Standard PCA projection for horseshoe inset."""
    deter = data["deter"]
    mean = deter.mean(axis=0)
    centered = deter - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1, pc2 = Vt[0], Vt[1]
    coords = np.column_stack([centered @ pc1, centered @ pc2])
    return coords


# ═══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_aa_circle(surface, color, center, radius, alpha=255):
    """Draw an anti-aliased filled circle."""
    x, y = int(center[0]), int(center[1])
    r = max(1, int(radius))
    if alpha < 255:
        s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(s, r + 1, r + 1, r, (*color, alpha))
        pygame.gfxdraw.filled_circle(s, r + 1, r + 1, r, (*color, alpha))
        surface.blit(s, (x - r - 1, y - r - 1))
    else:
        try:
            pygame.gfxdraw.aacircle(surface, x, y, r, color)
            pygame.gfxdraw.filled_circle(surface, x, y, r, color)
        except (OverflowError, ValueError):
            pass


def draw_triangle(surface, color, cx, cy, size, dx, dy):
    """Draw a filled triangle pointing in direction (dx, dy)."""
    angle = math.atan2(dy, dx)
    pts = []
    for i in range(3):
        a = angle + i * (2 * math.pi / 3)
        if i == 0:
            r = size * 1.2  # nose longer
        else:
            r = size * 0.7
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.aalines(surface, color, True, pts)


# ═══════════════════════════════════════════════════════════════════════════
# 2D World Panel
# ═══════════════════════════════════════════════════════════════════════════

def draw_world_panel(surface, data, frame, panel_rect):
    """Draw the 2D counting world."""
    x0, y0, pw, ph = panel_rect
    # Background
    pygame.draw.rect(surface, PANEL_BG, panel_rect, border_radius=4)

    n_blobs = int(data["n_blobs"][0])
    world_w = float(data["world_width"][0])
    world_h = float(data["world_height"][0])

    # Scale to fit panel with margin
    margin = 12
    scale = min((pw - 2 * margin) / world_w, (ph - 2 * margin) / world_h)
    ox = x0 + margin + (pw - 2 * margin - world_w * scale) / 2
    oy = y0 + margin + (ph - 2 * margin - world_h * scale) / 2

    def to_screen(wx, wy):
        return ox + wx * scale, oy + wy * scale

    # Grid (5x5) — positioned at right side of world
    grid_cell = 50
    grid_cols, grid_rows = 5, 5
    grid_right_margin = 120
    grid_left = world_w - grid_right_margin - grid_cols * grid_cell
    grid_height = grid_rows * grid_cell
    grid_top = (world_h - grid_height) / 2

    for row in range(grid_rows):
        for col in range(grid_cols):
            slot_idx = row * grid_cols + col
            wx = grid_left + col * grid_cell + grid_cell / 2
            wy = grid_top + row * grid_cell + grid_cell / 2
            sx, sy = to_screen(wx, wy)
            cell_size = grid_cell * scale
            rect = (sx - cell_size / 2, sy - cell_size / 2, cell_size, cell_size)
            filled = data["grid_filled"][frame, slot_idx] if slot_idx < 25 else False
            color = GRID_FILLED if filled else GRID_EMPTY
            pygame.draw.rect(surface, color, rect, border_radius=2)
            pygame.draw.rect(surface, (35, 35, 50), rect, 1, border_radius=2)

    # Blobs
    for i in range(n_blobs):
        bx = data["blob_x"][frame, i]
        by = data["blob_y"][frame, i]
        if bx == 0 and by == 0:
            continue
        sx, sy = to_screen(bx, by)
        on_grid = data["blob_on_grid"][frame, i]
        animating = data["blob_animating"][frame, i]

        if animating:
            color = BLOB_ANIM
            radius = max(4, 8 * scale)
        elif on_grid:
            color = BLOB_GRID
            radius = max(3, 6 * scale)
        else:
            color = BLOB_FIELD
            radius = max(4, 7 * scale)

        draw_aa_circle(surface, color, (sx, sy), radius)

    # Bot
    bx = data["bot_x"][frame]
    by = data["bot_y"][frame]
    dx = data["bot_dx"][frame]
    dy = data["bot_dy"][frame]
    sx, sy = to_screen(bx, by)
    bot_size = max(8, 14 * scale)
    draw_triangle(surface, BOT_COLOR, sx, sy, bot_size, dx, dy)

    # Count display
    gt = int(data["gt_count"][frame])
    count_text = f"Count: {gt}"
    font = pygame.font.SysFont("monospace", 16, bold=True)
    txt = font.render(count_text, True, TEXT_COLOR)
    surface.blit(txt, (x0 + pw - txt.get_width() - 15, y0 + 8))

    # Panel label
    label_font = pygame.font.SysFont("monospace", 11)
    label = label_font.render("2D COUNTING WORLD", True, TEXT_DIM)
    surface.blit(label, (x0 + 10, y0 + 8))


# ═══════════════════════════════════════════════════════════════════════════
# 3D Informed Panel
# ═══════════════════════════════════════════════════════════════════════════

def draw_3d_panel(surface, data, frame, panel_rect, coords_3d, centroids_3d,
                  azimuth, elevation, zoom, trail_length=80,
                  show_anticipation=False, show_transition_detail=False,
                  pca_2d=None, show_horseshoe=False,
                  show_dim_spotlight=False, show_probe_line=False,
                  transition_text=None, max_count=25,
                  precomputed=None):
    """Draw the 3D informed hidden state visualization."""
    x0, y0, pw, ph = panel_rect
    pygame.draw.rect(surface, PANEL_BG, panel_rect, border_radius=4)

    cx = x0 + pw / 2
    cy = y0 + ph / 2 - 10

    # Scale coordinates for display
    all_coords = coords_3d  # (T, 3)
    range_max = np.abs(all_coords).max() + 1e-8
    display_scale = min(pw, ph) * 0.35 / range_max * zoom

    def project(pts):
        """Project (N,3) to screen coordinates."""
        pts_scaled = pts * display_scale
        screen, depth = project_3d_to_2d(pts_scaled, azimuth, elevation, 1.0, cx, cy)
        return screen, depth

    # --- Draw axes ---
    axis_len = range_max * 1.1
    axis_pts = np.array([
        [0, 0, 0], [axis_len, 0, 0],   # X: counting
        [0, 0, 0], [0, axis_len, 0],    # Y: spatial
        [0, 0, 0], [0, 0, axis_len],    # Z: transition
    ]) * display_scale
    ax_screen, _ = project_3d_to_2d(axis_pts, azimuth, elevation, 1.0, cx, cy)

    axis_labels = ["Count \u2192", "Spatial \u2191", "Transition \u2197"]
    axis_colors = [(120, 180, 255), (120, 255, 150), (255, 180, 120)]
    for i in range(3):
        p0 = ax_screen[i * 2].astype(int)
        p1 = ax_screen[i * 2 + 1].astype(int)
        pygame.draw.aaline(surface, (*axis_colors[i][:3],), tuple(p0), tuple(p1))
        font = pygame.font.SysFont("monospace", 10)
        txt = font.render(axis_labels[i], True, axis_colors[i])
        surface.blit(txt, (int(p1[0]) + 4, int(p1[1]) - 6))

    # --- Draw centroid backbone ---
    cent_keys = sorted(centroids_3d.keys())
    if len(cent_keys) > 1:
        cent_pts = np.array([centroids_3d[k] for k in cent_keys])
        cent_screen, cent_depth = project(cent_pts)

        # Connecting line
        for i in range(len(cent_keys) - 1):
            p0 = cent_screen[i].astype(int)
            p1 = cent_screen[i + 1].astype(int)
            pygame.draw.aaline(surface, CENTROID_LINE, tuple(p0), tuple(p1))

        # Centroid dots
        for i, c in enumerate(cent_keys):
            sx, sy = cent_screen[i]
            t = c / max(max_count, 1)
            col = viridis(t)
            draw_aa_circle(surface, col, (sx, sy), 4, alpha=180)
            # Label every few counts
            if c % 5 == 0 or c == max_count:
                font = pygame.font.SysFont("monospace", 9)
                txt = font.render(str(c), True, TEXT_DIM)
                surface.blit(txt, (int(sx) + 6, int(sy) - 5))

    # --- Point cloud backdrop (subsampled trajectory as faint dots) ---
    if precomputed and "cloud_screen" not in precomputed:
        # Precompute on first call
        cloud_step = max(1, len(coords_3d) // 800)
        cloud_pts = coords_3d[::cloud_step]
        cloud_counts = data["gt_count"][::cloud_step]
        precomputed["cloud_pts"] = cloud_pts
        precomputed["cloud_counts"] = cloud_counts
    if precomputed and "cloud_pts" in precomputed:
        cloud_pts = precomputed["cloud_pts"]
        cloud_counts = precomputed["cloud_counts"]
        cloud_screen, cloud_depth = project(cloud_pts)
        # Sort by depth (back to front) for proper alpha blending
        depth_order = np.argsort(-cloud_depth)
        for i in depth_order:
            csx, csy = cloud_screen[i]
            if x0 + 5 < csx < x0 + pw - 5 and y0 + 5 < csy < y0 + ph - 5:
                ct = cloud_counts[i] / max(max_count, 1)
                col = viridis(ct)
                try:
                    draw_aa_circle(surface, col, (csx, csy), 2, alpha=45)
                except (OverflowError, ValueError):
                    pass

    # --- Trail ---
    trail_start = max(0, frame - trail_length)
    trail_pts = coords_3d[trail_start:frame + 1]
    if len(trail_pts) > 1:
        trail_screen, trail_depth = project(trail_pts)
        for i in range(len(trail_pts) - 1):
            alpha = int(40 + 180 * (i / len(trail_pts)))
            p0 = trail_screen[i].astype(int)
            p1 = trail_screen[i + 1].astype(int)
            # Draw trail segment
            t = data["gt_count"][trail_start + i] / max(max_count, 1)
            col = viridis(t)
            try:
                pygame.draw.aaline(surface, col, tuple(p0), tuple(p1))
            except (OverflowError, ValueError):
                pass

    # --- Current position dot ---
    cur_pt = coords_3d[frame:frame + 1]
    cur_screen, cur_depth = project(cur_pt)
    sx, sy = cur_screen[0]
    gt = int(data["gt_count"][frame])
    t = gt / max(max_count, 1)
    dot_color = viridis(t)

    # Glow rings
    for r, a in [(12, 30), (8, 60), (5, 120)]:
        draw_aa_circle(surface, dot_color, (sx, sy), r, alpha=a)
    draw_aa_circle(surface, dot_color, (sx, sy), 4)

    # --- Anticipation overlay (prior dot) ---
    if show_anticipation and frame > 0:
        # Prior uses same deter, but show where the prior probe points
        # Since prior_probe ≈ probe_pred (same deter), we show the prior's
        # stochastic state projected through a simplified path
        prior_probe = data["prior_probe"][frame]
        post_probe = data["probe_pred"][frame]
        # Visual indicator: if prior and posterior differ, show a ghost dot
        # For now show a dimmer dot slightly offset in the counting direction
        diff = prior_probe - gt
        if abs(diff) > 0.3:
            # Show prior position as ghost
            ghost_3d = coords_3d[frame].copy()
            ghost_3d[0] += diff * display_scale * 0.01  # slight offset
            ghost_screen, _ = project(ghost_3d.reshape(1, 3))
            gsx, gsy = ghost_screen[0]
            draw_aa_circle(surface, (200, 200, 255), (gsx, gsy), 5, alpha=80)
            font = pygame.font.SysFont("monospace", 9)
            txt = font.render(f"prior: {prior_probe:.1f}", True, (180, 180, 220))
            surface.blit(txt, (int(gsx) + 8, int(gsy) - 5))

    # --- Transition detail annotations ---
    if show_transition_detail and transition_text:
        font = pygame.font.SysFont("monospace", 11)
        lines = transition_text.split("\n")
        for i, line in enumerate(lines):
            txt = font.render(line, True, TEXT_YELLOW)
            surface.blit(txt, (x0 + 12, y0 + ph - 70 - i * 16))

    # --- Dimension spotlight ---
    if show_dim_spotlight and precomputed:
        _draw_dim_spotlight(surface, data, frame, x0 + pw - 110, y0 + 30, precomputed)

    # --- Probe number line ---
    if show_probe_line:
        _draw_probe_line(surface, data, frame, x0 + 20, y0 + ph - 30, pw - 40, max_count)

    # --- PCA horseshoe inset ---
    if show_horseshoe and pca_2d is not None:
        _draw_horseshoe_inset(surface, data, frame, pca_2d,
                              x0 + pw - 155, y0 + ph - 155, 140, 140, max_count)

    # Panel label
    label_font = pygame.font.SysFont("monospace", 11)
    label = label_font.render("3D INFORMED HIDDEN STATE", True, TEXT_DIM)
    surface.blit(label, (x0 + 10, y0 + 8))


def _draw_dim_spotlight(surface, data, frame, x, y, precomputed):
    """Draw a bar chart of the top counting dimension's raw value."""
    top_dim = precomputed["top_dim"]
    top_corr = precomputed["top_corr"]
    dim_min = precomputed["dim_min"]
    dim_max = precomputed["dim_max"]
    val = data["deter"][frame, top_dim]

    # Draw box
    w, h = 95, 110
    pygame.draw.rect(surface, (30, 30, 42), (x, y, w, h), border_radius=3)
    pygame.draw.rect(surface, (50, 50, 65), (x, y, w, h), 1, border_radius=3)

    font = pygame.font.SysFont("monospace", 9)
    txt = font.render(f"Dim {top_dim}", True, TEXT_DIM)
    surface.blit(txt, (x + 5, y + 4))
    txt = font.render(f"|r|={top_corr:.3f}", True, TEXT_DIM)
    surface.blit(txt, (x + 5, y + 16))

    # Bar
    bar_x = x + 15
    bar_y = y + 35
    bar_w = 65
    bar_h = 60
    val_range = max(abs(dim_min), abs(dim_max))
    if val_range < 1e-8:
        val_range = 1.0
    bar_fill = val / val_range  # -1 to 1
    # Draw bar background
    pygame.draw.rect(surface, (40, 40, 55), (bar_x, bar_y, bar_w, bar_h))
    # Draw zero line
    zero_y = bar_y + bar_h / 2
    pygame.draw.line(surface, TEXT_DIM, (bar_x, int(zero_y)), (bar_x + bar_w, int(zero_y)))
    # Draw fill
    fill_h = abs(bar_fill) * bar_h / 2
    if bar_fill > 0:
        pygame.draw.rect(surface, TEXT_GREEN,
                         (bar_x + 2, int(zero_y - fill_h), bar_w - 4, int(fill_h)))
    else:
        pygame.draw.rect(surface, TEXT_RED,
                         (bar_x + 2, int(zero_y), bar_w - 4, int(fill_h)))

    # Value text
    txt = font.render(f"{val:.2f}", True, TEXT_COLOR)
    surface.blit(txt, (bar_x + 5, bar_y + bar_h + 3))


def _draw_probe_line(surface, data, frame, x, y, w, max_count):
    """Draw a 1D probe number line."""
    h = 20
    # Background bar
    pygame.draw.rect(surface, (35, 35, 48), (x, y, w, h), border_radius=3)

    # Tick marks
    font = pygame.font.SysFont("monospace", 8)
    for c in range(0, max_count + 1, 5):
        tx = x + (c / max_count) * w
        pygame.draw.line(surface, TEXT_DIM, (int(tx), y), (int(tx), y + 5))
        txt = font.render(str(c), True, TEXT_DIM)
        surface.blit(txt, (int(tx) - 3, y + 6))

    # Ground truth marker
    gt = int(data["gt_count"][frame])
    gt_x = x + (gt / max_count) * w
    pygame.draw.rect(surface, (80, 80, 100), (int(gt_x) - 1, y, 3, h))

    # Probe marker
    probe_val = float(data["probe_pred"][frame])
    probe_x = x + (np.clip(probe_val, 0, max_count) / max_count) * w
    draw_aa_circle(surface, TRAIL_COLOR, (probe_x, y + h / 2), 5)

    # Label
    label = font.render("PROBE", True, TEXT_DIM)
    surface.blit(label, (x - 40, y + 4))


def _draw_horseshoe_inset(surface, data, frame, pca_2d, x, y, w, h, max_count):
    """Draw a small PCA horseshoe projection inset."""
    # Background
    pygame.draw.rect(surface, (25, 25, 35), (x, y, w, h), border_radius=3)
    pygame.draw.rect(surface, (50, 50, 65), (x, y, w, h), 1, border_radius=3)

    # Scale PCA coordinates to fit
    pca_range = np.abs(pca_2d).max() + 1e-8
    margin = 15

    def to_inset(px, py):
        sx = x + margin + (px / pca_range + 1) * 0.5 * (w - 2 * margin)
        sy = y + margin + (-py / pca_range + 1) * 0.5 * (h - 2 * margin)
        return sx, sy

    # Draw faint cloud (subsample)
    step = max(1, len(pca_2d) // 300)
    for i in range(0, len(pca_2d), step):
        sx, sy = to_inset(pca_2d[i, 0], pca_2d[i, 1])
        t = data["gt_count"][i] / max(max_count, 1)
        col = viridis(t)
        try:
            surface.set_at((int(sx), int(sy)), (*col, 80))
        except (IndexError, OverflowError):
            pass

    # Current point
    sx, sy = to_inset(pca_2d[frame, 0], pca_2d[frame, 1])
    draw_aa_circle(surface, TRAIL_COLOR, (sx, sy), 3)

    # Label
    font = pygame.font.SysFont("monospace", 8)
    txt = font.render("PCA", True, TEXT_DIM)
    surface.blit(txt, (x + 3, y + 3))


# ═══════════════════════════════════════════════════════════════════════════
# Status bar
# ═══════════════════════════════════════════════════════════════════════════

def draw_status_bar(surface, data, frame, status_rect, speed_idx, playing,
                    show_anticipation, show_transition_detail,
                    show_dim_spotlight=False, show_probe_line=False,
                    show_horseshoe=False):
    """Draw the bottom status bar."""
    x0, y0, sw, sh = status_rect
    pygame.draw.rect(surface, (18, 18, 26), status_rect)

    n_frames = int(data["n_frames"][0])
    gt = int(data["gt_count"][frame])
    probe_val = float(data["probe_pred"][frame])
    probe_int = int(round(np.clip(probe_val, 0, 25)))
    prior_val = float(data["prior_probe"][frame])

    # Probe accuracy color
    diff = abs(probe_int - gt)
    if diff == 0:
        probe_color = TEXT_GREEN
    elif diff == 1:
        probe_color = TEXT_YELLOW
    else:
        probe_color = TEXT_RED

    # Phase
    phase_val = int(data["phase"][frame])
    phase_names = ["Seeking blob", "Unmarking", "Predict"]
    phase_str = phase_names[min(phase_val, 2)]

    # Detect carrying/animating
    if data["blob_animating"][frame].any():
        phase_str = "Blob in flight"

    speed = SPEEDS[speed_idx]
    play_str = f"\u25B6 {speed}x" if playing else f"\u23F8 {speed}x"

    font = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 11)

    # Row 1: frame info
    row1_y = y0 + 12
    items = [
        (f"Frame {frame}/{n_frames}", TEXT_DIM, 20),
        (f"Count: {gt}", TEXT_COLOR, 200),
        (f"Probe: {probe_val:.1f} ({probe_int})", probe_color, 360),
        (f"Prior: {prior_val:.1f}", TEXT_DIM, 560),
        (play_str, TEXT_COLOR, 700),
        (phase_str, TEXT_DIM, 800),
    ]
    for text, color, xpos in items:
        txt = font.render(text, True, color)
        surface.blit(txt, (x0 + xpos, row1_y))

    # Row 2: toggles
    row2_y = y0 + 35
    toggles = [
        ("A", "Anticipation", show_anticipation),
        ("T", "Transition detail", show_transition_detail),
        ("D", "Dim spotlight", show_dim_spotlight),
        ("P", "Probe line", show_probe_line),
        ("H", "PCA horseshoe", show_horseshoe),
    ]
    tx = x0 + 20
    for key, label, active in toggles:
        col = TEXT_GREEN if active else TEXT_DIM
        txt = font_small.render(f"[{key}] {label}", True, col)
        surface.blit(txt, (tx, row2_y))
        tx += txt.get_width() + 20

    # Row 3: controls help
    row3_y = y0 + 55
    help_text = "Space: play/pause  \u2190\u2192: step  Shift+\u2190\u2192: jump transition  \u2191\u2193: speed  R: reset  Drag: rotate 3D  Scroll: zoom"
    txt = font_small.render(help_text, True, (60, 60, 75))
    surface.blit(txt, (x0 + 20, row3_y))

    # Progress bar
    bar_y = y0 + 2
    bar_w = sw - 40
    bar_x = x0 + 20
    pygame.draw.rect(surface, (35, 35, 50), (bar_x, bar_y, bar_w, 4))
    progress = frame / max(n_frames - 1, 1)
    pygame.draw.rect(surface, TEXT_GREEN, (bar_x, bar_y, int(bar_w * progress), 4))

    # Transition markers on progress bar
    trans = np.where(data["transition"])[0]
    for t_frame in trans:
        tx = bar_x + (t_frame / max(n_frames - 1, 1)) * bar_w
        pygame.draw.line(surface, TEXT_YELLOW, (int(tx), bar_y), (int(tx), bar_y + 6))


# ═══════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Educational visualization")
    parser.add_argument("--episode",
                        default="/workspace/bridge/artifacts/episodes/randproj_episode.npz",
                        help="Path to episode .npz file")
    parser.add_argument("--headless", action="store_true",
                        help="Render key frames and save PNGs (no window)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="Directory to save every Nth frame as PNG for video export")
    parser.add_argument("--frame-step", type=int, default=3,
                        help="Step between saved frames (with --save-frames)")
    args = parser.parse_args()

    # Load data
    print(f"Loading episode: {args.episode}")
    data = dict(np.load(args.episode, allow_pickle=True))
    n_frames = int(data["n_frames"][0])
    max_count = int(data["gt_count"].max())
    print(f"  {n_frames} frames, count range 0-{max_count}")

    # Precompute informed axes
    print("Computing informed axes...")
    axis1, axis2, axis3 = compute_informed_axes(data)
    coords_3d, mean_deter = compute_3d_positions(data, axis1, axis2, axis3)
    centroids_3d = compute_centroids_3d(data, axis1, axis2, axis3, mean_deter)
    pca_2d = compute_pca_2d(data)

    # Precompute top counting dimension (for spotlight)
    gt_float = data["gt_count"].astype(float)
    corr = np.array([abs(np.corrcoef(data["deter"][:, d], gt_float)[0, 1])
                     if data["deter"][:, d].std() > 1e-8 else 0
                     for d in range(512)])
    top_dim = int(np.argmax(corr))
    print(f"  Top counting dim: {top_dim} (|r|={corr[top_dim]:.3f})")

    # Find transition frames
    transition_frames = np.where(data["transition"])[0]
    print(f"  {len(transition_frames)} transitions")

    # Precomputed data for efficient rendering
    precomputed = {
        "top_dim": top_dim,
        "top_corr": corr[top_dim],
        "dim_min": float(data["deter"][:, top_dim].min()),
        "dim_max": float(data["deter"][:, top_dim].max()),
    }

    # Init pygame
    if args.headless or args.save_frames:
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Anim \u2014 Counting Manifold Visualization")
    clock = pygame.time.Clock()

    # --- Headless mode: render key frames and exit ---
    if args.headless:
        out_dir = Path(args.episode).parent
        key_frames = [0]
        # Add transition frames (10 frames before each)
        for tf in transition_frames:
            key_frames.extend([max(0, int(tf) - 10), int(tf), int(tf) + 5])
        # Add evenly spaced frames
        for f in range(0, n_frames, n_frames // 8):
            key_frames.append(f)
        key_frames.append(n_frames - 1)
        key_frames = sorted(set(f for f in key_frames if 0 <= f < n_frames))

        for fi, f in enumerate(key_frames):
            screen.fill(BG)
            draw_world_panel(screen, data, f, (6, 6, PANEL_W - 9, PANEL_H - 6))
            draw_3d_panel(screen, data, f, (PANEL_W + 3, 6, PANEL_W - 9, PANEL_H - 6),
                          coords_3d, centroids_3d, DEFAULT_AZIMUTH, DEFAULT_ELEVATION,
                          DEFAULT_ZOOM, show_probe_line=True, show_dim_spotlight=True,
                          max_count=max_count, precomputed=precomputed)
            draw_status_bar(screen, data, f, (0, PANEL_H, SCREEN_W, STATUS_H),
                            DEFAULT_SPEED_IDX, False, False, False, True, True, False)
            fname = out_dir / f"frame_{f:05d}.png"
            pygame.image.save(screen, str(fname))
        print(f"  Saved {len(key_frames)} key frames to {out_dir}/")
        pygame.quit()
        return

    # --- Save-frames mode: render every Nth frame ---
    if args.save_frames:
        save_dir = Path(args.save_frames)
        save_dir.mkdir(parents=True, exist_ok=True)
        step = args.frame_step
        total = len(range(0, n_frames, step))
        for fi, f in enumerate(range(0, n_frames, step)):
            screen.fill(BG)
            draw_world_panel(screen, data, f, (6, 6, PANEL_W - 9, PANEL_H - 6))
            draw_3d_panel(screen, data, f, (PANEL_W + 3, 6, PANEL_W - 9, PANEL_H - 6),
                          coords_3d, centroids_3d, DEFAULT_AZIMUTH, DEFAULT_ELEVATION,
                          DEFAULT_ZOOM, show_probe_line=True, show_dim_spotlight=True,
                          max_count=max_count, precomputed=precomputed)
            draw_status_bar(screen, data, f, (0, PANEL_H, SCREEN_W, STATUS_H),
                            DEFAULT_SPEED_IDX, True, False, False, True, True, False)
            pygame.image.save(screen, str(save_dir / f"frame_{fi:06d}.png"))
            if fi % 100 == 0:
                print(f"  {fi}/{total} frames...", flush=True)
        print(f"  Saved {total} frames to {save_dir}/")
        print(f"  To make video: ffmpeg -framerate 30 -i {save_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
        pygame.quit()
        return

    # State
    frame = 0
    playing = False
    speed_idx = DEFAULT_SPEED_IDX
    azimuth = DEFAULT_AZIMUTH
    elevation = DEFAULT_ELEVATION
    zoom = DEFAULT_ZOOM
    dragging = False
    drag_start = (0, 0)
    drag_az_start = azimuth
    drag_el_start = elevation

    # Toggles
    show_anticipation = False
    show_transition_detail = False
    show_dim_spotlight = False
    show_probe_line = False
    show_horseshoe = False

    # Transition detail state
    transition_slowmo = False
    transition_text = None
    transition_restore_speed = DEFAULT_SPEED_IDX
    frames_since_transition = 999

    # Accumulator for sub-frame stepping
    step_accum = 0.0

    print("Ready. Press Space to play.")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_LEFT:
                    if mods & pygame.KMOD_SHIFT:
                        # Jump to previous transition
                        prev_trans = transition_frames[transition_frames < frame]
                        if len(prev_trans) > 0:
                            frame = int(prev_trans[-1]) - 10
                            frame = max(0, frame)
                    else:
                        frame = max(0, frame - 1)
                elif event.key == pygame.K_RIGHT:
                    if mods & pygame.KMOD_SHIFT:
                        # Jump to next transition
                        next_trans = transition_frames[transition_frames > frame]
                        if len(next_trans) > 0:
                            frame = max(0, int(next_trans[0]) - 10)
                    else:
                        frame = min(n_frames - 1, frame + 1)
                elif event.key == pygame.K_UP:
                    speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)
                elif event.key == pygame.K_DOWN:
                    speed_idx = max(0, speed_idx - 1)
                elif event.key == pygame.K_r:
                    frame = 0
                    playing = False
                elif event.key == pygame.K_a:
                    show_anticipation = not show_anticipation
                elif event.key == pygame.K_t:
                    show_transition_detail = not show_transition_detail
                elif event.key == pygame.K_d:
                    show_dim_spotlight = not show_dim_spotlight
                elif event.key == pygame.K_p:
                    show_probe_line = not show_probe_line
                elif event.key == pygame.K_h:
                    show_horseshoe = not show_horseshoe
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if click is in 3D panel area
                    mx, my = event.pos
                    if mx >= PANEL_W:
                        dragging = True
                        drag_start = event.pos
                        drag_az_start = azimuth
                        drag_el_start = elevation
                elif event.button == 4:  # scroll up
                    zoom *= 1.1
                elif event.button == 5:  # scroll down
                    zoom /= 1.1
                    zoom = max(0.1, zoom)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    azimuth = drag_az_start + dx * 0.5
                    elevation = drag_el_start + dy * 0.5
                    elevation = max(-89, min(89, elevation))

        # Advance frame if playing
        if playing:
            speed = SPEEDS[speed_idx]
            step_accum += speed * BASE_STEP_RATE * dt
            steps = int(step_accum)
            step_accum -= steps
            frame = min(n_frames - 1, frame + steps)
            if frame >= n_frames - 1:
                playing = False

        # Transition detail mode
        if show_transition_detail:
            is_trans = bool(data["transition"][frame]) if frame < n_frames else False
            if is_trans and frames_since_transition > 30:
                # Entering transition
                gt_now = int(data["gt_count"][frame])
                gt_prev = int(data["gt_count"][max(0, frame - 1)])
                transition_text = f"Blob landed: count {gt_prev} \u2192 {gt_now}"
                if not transition_slowmo:
                    transition_restore_speed = speed_idx
                    speed_idx = 0  # 0.25x
                    transition_slowmo = True
                frames_since_transition = 0
            else:
                frames_since_transition += 1
                if frames_since_transition > 60 and transition_slowmo:
                    speed_idx = transition_restore_speed
                    transition_slowmo = False
                    transition_text = None

        # ---- Draw ----
        screen.fill(BG)

        # Panels
        world_rect = (6, 6, PANEL_W - 9, PANEL_H - 6)
        state3d_rect = (PANEL_W + 3, 6, PANEL_W - 9, PANEL_H - 6)
        status_rect = (0, PANEL_H, SCREEN_W, STATUS_H)

        draw_world_panel(screen, data, frame, world_rect)
        draw_3d_panel(screen, data, frame, state3d_rect, coords_3d, centroids_3d,
                      azimuth, elevation, zoom,
                      show_anticipation=show_anticipation,
                      show_transition_detail=show_transition_detail,
                      pca_2d=pca_2d, show_horseshoe=show_horseshoe,
                      show_dim_spotlight=show_dim_spotlight,
                      show_probe_line=show_probe_line,
                      transition_text=transition_text,
                      max_count=max_count,
                      precomputed=precomputed)
        draw_status_bar(screen, data, frame, status_rect, speed_idx, playing,
                        show_anticipation, show_transition_detail,
                        show_dim_spotlight, show_probe_line, show_horseshoe)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
