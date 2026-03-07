#!/usr/bin/env python3
"""
Preview all 12 proposed blob arrangement types in a pygame grid.

Usage:
    cd ~/anim-bridge/scripts && python3 preview_arrangements.py
    python3 preview_arrangements.py --blobs 13   # fewer blobs
    python3 preview_arrangements.py --reseed      # press R to reseed live
"""
import math
import random
import sys

# ── Existing arrangements (imported from counting_env_pure) ──────────

from counting_env_pure import (
    _scattered_positions,
    _clustered_positions,
    _grid_like_positions,
    _mixed_positions,
    _dist,
    WORLD_WIDTH, WORLD_HEIGHT, MARGIN, MIN_SEPARATION,
)

# ── 8 new proposed arrangement functions ─────────────────────────────

def _tight_cluster_positions(count, w, h, margin, min_sep):
    """All blobs within a small radius — tests counting in tight clusters."""
    radius = max(min_sep * 1.5, min(w, h) * 0.06)
    cx = margin + radius + random.random() * (w - 2 * margin - 2 * radius)
    cy = margin + radius + random.random() * (h - 2 * margin - 2 * radius)
    positions = []
    attempts = 0
    while len(positions) < count and attempts < count * 200:
        attempts += 1
        angle = random.random() * math.pi * 2
        r = random.random() * radius
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        if x < margin or x > w - margin or y < margin or y > h - margin:
            continue
        if all(_dist(px, py, x, y) >= min_sep for px, py in positions):
            positions.append((x, y))
    if len(positions) < count:
        return _scattered_positions(count, w, h, margin, min_sep)
    return positions


def _max_spread_positions(count, w, h, margin, min_sep):
    """Greedy farthest-point sampling — maximize minimum pairwise distance."""
    positions = [(
        margin + random.random() * (w - 2 * margin),
        margin + random.random() * (h - 2 * margin),
    )]
    for _ in range(count - 1):
        best = None
        best_min_d = -1
        for _attempt in range(60):
            cx = margin + random.random() * (w - 2 * margin)
            cy = margin + random.random() * (h - 2 * margin)
            min_d = min(_dist(px, py, cx, cy) for px, py in positions)
            if min_d > best_min_d:
                best_min_d = min_d
                best = (cx, cy)
        if best:
            positions.append(best)
    return positions


def _linear_positions(count, w, h, margin, min_sep):
    """Blobs along a line at a random angle."""
    theta = random.random() * math.pi  # [0, pi)
    cx = margin + random.random() * (w - 2 * margin)
    cy = margin + random.random() * (h - 2 * margin)
    dx, dy = math.cos(theta), math.sin(theta)

    # Find max extent that fits
    max_t = 0
    for t_test in range(10, 2000, 10):
        x1 = cx - dx * t_test
        y1 = cy - dy * t_test
        x2 = cx + dx * t_test
        y2 = cy + dy * t_test
        if (margin <= x1 <= w - margin and margin <= y1 <= h - margin and
                margin <= x2 <= w - margin and margin <= y2 <= h - margin):
            max_t = t_test
        else:
            break

    if max_t < min_sep * (count - 1) / 2:
        # Line doesn't fit — fallback
        return _scattered_positions(count, w, h, margin, min_sep)

    spacing = 2 * max_t / max(count - 1, 1)
    jitter = min_sep * 0.3
    positions = []
    for i in range(count):
        t = -max_t + i * spacing
        px = cx + dx * t + (random.random() - 0.5) * 2 * jitter * (-dy)
        py = cy + dy * t + (random.random() - 0.5) * 2 * jitter * dx
        px = max(margin, min(w - margin, px))
        py = max(margin, min(h - margin, py))
        positions.append((px, py))
    return positions


def _ring_positions(count, w, h, margin, min_sep):
    """Blobs evenly spaced on a circle."""
    max_radius = min(w - 2 * margin, h - 2 * margin) / 2 * 0.85
    # Radius must be large enough for min_sep between adjacent
    min_radius = min_sep * count / (2 * math.pi) if count > 1 else 0
    radius = max(min_radius, max_radius * 0.4)
    radius = min(radius, max_radius)

    cx = margin + radius + random.random() * max(0, w - 2 * margin - 2 * radius)
    cy = margin + radius + random.random() * max(0, h - 2 * margin - 2 * radius)
    start_angle = random.random() * 2 * math.pi
    radial_jitter = radius * 0.08

    positions = []
    for i in range(count):
        angle = start_angle + 2 * math.pi * i / count
        r = radius + (random.random() - 0.5) * 2 * radial_jitter
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        x = max(margin, min(w - margin, x))
        y = max(margin, min(h - margin, y))
        positions.append((x, y))
    return positions


def _two_clusters_positions(count, w, h, margin, min_sep):
    """Two separated groups of blobs."""
    sep = min(w, h) * 0.3
    # Place two centers
    for _attempt in range(200):
        c1x = margin * 2 + random.random() * (w - 4 * margin)
        c1y = margin * 2 + random.random() * (h - 4 * margin)
        c2x = margin * 2 + random.random() * (w - 4 * margin)
        c2y = margin * 2 + random.random() * (h - 4 * margin)
        if _dist(c1x, c1y, c2x, c2y) >= sep:
            break
    else:
        return _scattered_positions(count, w, h, margin, min_sep)

    n1 = count // 2
    n2 = count - n1
    cluster_radius = min(w, h) * 0.10

    positions = []
    for cx, cy, n in [(c1x, c1y, n1), (c2x, c2y, n2)]:
        attempts = 0
        while len(positions) < (count - n2 if n == n1 else count) and attempts < n * 80:
            attempts += 1
            angle = random.random() * 2 * math.pi
            r = random.random() * cluster_radius
            x = cx + math.cos(angle) * r
            y = cy + math.sin(angle) * r
            if x < margin or x > w - margin or y < margin or y > h - margin:
                continue
            if all(_dist(px, py, x, y) >= min_sep for px, py in positions):
                positions.append((x, y))
    return positions


def _three_clusters_positions(count, w, h, margin, min_sep):
    """Three separated groups of blobs."""
    sep = min(w, h) * 0.25
    centers = []
    for _c in range(3):
        for _attempt in range(200):
            cx = margin * 2 + random.random() * (w - 4 * margin)
            cy = margin * 2 + random.random() * (h - 4 * margin)
            if all(_dist(ox, oy, cx, cy) >= sep for ox, oy in centers):
                centers.append((cx, cy))
                break
    if len(centers) < 3:
        return _scattered_positions(count, w, h, margin, min_sep)

    splits = [count // 3, count // 3, count - 2 * (count // 3)]
    cluster_radius = min(w, h) * 0.09
    positions = []
    target = 0
    for ci, (cx, cy) in enumerate(centers):
        target += splits[ci]
        attempts = 0
        while len(positions) < target and attempts < splits[ci] * 80:
            attempts += 1
            angle = random.random() * 2 * math.pi
            r = random.random() * cluster_radius
            x = cx + math.cos(angle) * r
            y = cy + math.sin(angle) * r
            if x < margin or x > w - margin or y < margin or y > h - margin:
                continue
            if all(_dist(px, py, x, y) >= min_sep for px, py in positions):
                positions.append((x, y))
    return positions


def _spiral_positions(count, w, h, margin, min_sep):
    """Blobs along an Archimedean spiral."""
    max_radius = min(w - 2 * margin, h - 2 * margin) / 2 * 0.8
    cx = w / 2
    cy = h / 2
    # Archimedean spiral: r = a + b*theta
    # We want total arc from r_min to max_radius, with count points
    total_turns = 2.0 + count / 10.0
    max_theta = total_turns * 2 * math.pi
    b = max_radius / max_theta
    start_angle = random.random() * 2 * math.pi

    positions = []
    for i in range(count):
        theta = max_theta * i / max(count - 1, 1)
        r = b * theta + min_sep * 0.5  # small inner offset
        r = min(r, max_radius)
        angle = start_angle + theta
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        x = max(margin, min(w - margin, x))
        y = max(margin, min(h - margin, y))
        positions.append((x, y))
    return positions


def _adversarial_positions(count, w, h, margin, min_sep):
    """Maximize total travel distance (greedy worst-case nearest-neighbor tour)."""
    best_positions = None
    best_tour = -1

    for _ in range(10):
        positions = _scattered_positions(count, w, h, margin, min_sep)
        if len(positions) < count:
            continue
        # Compute nearest-neighbor tour length
        remaining = list(range(len(positions)))
        current = remaining.pop(0)
        tour_len = 0
        while remaining:
            nearest_idx = min(remaining, key=lambda j: _dist(
                positions[current][0], positions[current][1],
                positions[j][0], positions[j][1]))
            tour_len += _dist(
                positions[current][0], positions[current][1],
                positions[nearest_idx][0], positions[nearest_idx][1])
            remaining.remove(nearest_idx)
            current = nearest_idx
        if tour_len > best_tour:
            best_tour = tour_len
            best_positions = positions

    return best_positions or _scattered_positions(count, w, h, margin, min_sep)


# ── All 12 arrangements ─────────────────────────────────────────────

ALL_ARRANGEMENTS = [
    ("scattered", _scattered_positions),
    ("clustered", _clustered_positions),
    ("grid-like", _grid_like_positions),
    ("mixed", _mixed_positions),
    ("tight-cluster", _tight_cluster_positions),
    ("max-spread", _max_spread_positions),
    ("linear", _linear_positions),
    ("ring", _ring_positions),
    ("two-clusters", _two_clusters_positions),
    ("three-clusters", _three_clusters_positions),
    ("spiral", _spiral_positions),
    ("adversarial", _adversarial_positions),
]


# ── Pygame rendering ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preview all 12 blob arrangement types")
    parser.add_argument("--blobs", type=int, default=25, help="Number of blobs (default 25)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    import pygame
    pygame.init()

    COLS, ROWS = 4, 3
    CELL_W, CELL_H = 320, 260
    PAD = 10
    LABEL_H = 28
    SCREEN_W = COLS * CELL_W + PAD
    SCREEN_H = ROWS * CELL_H + PAD

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Blob Arrangements Preview — {args.blobs} blobs (R=reseed, Q=quit)")

    font = pygame.font.SysFont("Menlo,Monaco,monospace", 16)
    font_sm = pygame.font.SysFont("Menlo,Monaco,monospace", 12)

    BG = (18, 18, 24)
    CELL_BG = (28, 30, 38)
    FIELD_BG = (22, 26, 32)
    GRID_AREA = (35, 38, 48)
    BLOB_COLOR = (80, 200, 255)
    BLOB_OUTLINE = (40, 120, 180)
    LABEL_COLOR = (200, 210, 230)
    COUNT_COLOR = (120, 130, 150)
    EXISTING_BADGE = (60, 180, 100)
    NEW_BADGE = (220, 140, 50)

    field_w = int(WORLD_WIDTH * 0.55)

    def generate_all():
        results = []
        for name, func in ALL_ARRANGEMENTS:
            positions = func(args.blobs, field_w, WORLD_HEIGHT, MARGIN, MIN_SEPARATION)
            results.append((name, positions))
        return results

    arrangements = generate_all()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    arrangements = generate_all()

        screen.fill(BG)

        for idx, (name, positions) in enumerate(arrangements):
            col = idx % COLS
            row = idx // COLS
            ox = col * CELL_W + PAD
            oy = row * CELL_H + PAD

            # Cell background
            cell_rect = pygame.Rect(ox, oy, CELL_W - PAD, CELL_H - PAD)
            pygame.draw.rect(screen, CELL_BG, cell_rect, border_radius=6)

            # Field area (left 55%)
            draw_w = CELL_W - PAD - 8
            draw_h = CELL_H - PAD - LABEL_H - 8
            field_draw_w = int(draw_w * 0.55)
            field_rect = pygame.Rect(ox + 4, oy + LABEL_H, field_draw_w, draw_h)
            pygame.draw.rect(screen, FIELD_BG, field_rect)

            # Grid area (right 45%)
            grid_rect = pygame.Rect(ox + 4 + field_draw_w, oy + LABEL_H,
                                    draw_w - field_draw_w, draw_h)
            pygame.draw.rect(screen, GRID_AREA, grid_rect)

            # Scale positions to cell
            sx = field_draw_w / field_w
            sy = draw_h / WORLD_HEIGHT

            # Draw blobs
            for px, py in positions:
                bx = int(ox + 4 + px * sx)
                by = int(oy + LABEL_H + py * sy)
                pygame.draw.circle(screen, BLOB_COLOR, (bx, by), 5)
                pygame.draw.circle(screen, BLOB_OUTLINE, (bx, by), 5, 1)

            # Label
            is_existing = idx < 4
            badge_color = EXISTING_BADGE if is_existing else NEW_BADGE
            badge_text = "existing" if is_existing else "NEW"
            badge_surf = font_sm.render(badge_text, True, badge_color)
            name_surf = font.render(name, True, LABEL_COLOR)
            count_surf = font_sm.render(f"n={len(positions)}", True, COUNT_COLOR)

            screen.blit(name_surf, (ox + 6, oy + 4))
            screen.blit(badge_surf, (ox + 6 + name_surf.get_width() + 8, oy + 8))
            screen.blit(count_surf, (ox + CELL_W - PAD - count_surf.get_width() - 8, oy + 8))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
