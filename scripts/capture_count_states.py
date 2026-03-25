#!/usr/bin/env python3
"""
Capture screenshots of the binary counting environment at each discrete state (0-14).
Renders a clean frame at each count value when the machine is idle (not mid-cascade).
"""

import os
import sys
import math
from pathlib import Path

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_DIR = Path("/workspace/projects/jamstack-v1/bridge/scripts")
sys.path.insert(0, str(_ENV_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from binary_counting_env import (
    BinaryCountingEnv, OBS_SIZE,
    ARENA_W, ARENA_H, FIELD_X_MAX,
    MACHINE_X_START, MACHINE_Y_TOP, MACHINE_Y_BOT,
    MACHINE_COL_SPACING, MACHINE_INPUT_X, MACHINE_INPUT_Y,
    NUM_COLUMNS, COLUMN_X, COLUMN_Y, BLOB_SIZES,
    BOT_RADIUS, BLOB_BASE_SIZE, PICKUP_RADIUS,
    MAX_FIELD_BLOBS, PHASE_IDLE,
)

import pygame as pg
pg.init()
pg.font.init()

# Render at 2x for crisp screenshots
RENDER_W = 1200
RENDER_H = 800
MARGIN = 20


def w2s(wx, wy):
    """World-to-screen transform."""
    scale_x = (RENDER_W - 2 * MARGIN) / ARENA_W
    scale_y = (RENDER_H - 80 - 2 * MARGIN) / ARENA_H  # Leave room for title
    scale = min(scale_x, scale_y)
    ox = MARGIN + (RENDER_W - 2 * MARGIN - ARENA_W * scale) / 2
    oy = 60 + MARGIN + (RENDER_H - 80 - 2 * MARGIN - ARENA_H * scale) / 2
    return int(ox + wx * scale), int(oy + wy * scale), scale


# Pre-compute scale
_, _, SCALE = w2s(0, 0)


def w2s_pt(wx, wy):
    sx, sy, _ = w2s(wx, wy)
    return sx, sy


def render_state(state, count, n_remaining, step):
    """Render the full arena to a pygame surface."""
    screen = pg.Surface((RENDER_W, RENDER_H))
    screen.fill((18, 18, 24))

    # Title bar
    font_title = pg.font.SysFont("monospace", 28, bold=True)
    font_sub = pg.font.SysFont("monospace", 18)

    bits_str = ""
    for i in range(NUM_COLUMNS - 1, -1, -1):
        b = 1 if state.columns[i].occupied else 0
        bits_str += str(b)

    title = f"Count = {count}   ({bits_str})"
    title_surf = font_title.render(title, True, (240, 240, 240))
    screen.blit(title_surf, (RENDER_W // 2 - title_surf.get_width() // 2, 15))

    # Arena background
    ax1, ay1 = w2s_pt(0, 0)
    ax2, ay2 = w2s_pt(ARENA_W, ARENA_H)
    pg.draw.rect(screen, (24, 24, 32), (ax1, ay1, ax2 - ax1, ay2 - ay1))

    # Field boundary
    fx, fy1 = w2s_pt(FIELD_X_MAX, 0)
    _, fy2 = w2s_pt(FIELD_X_MAX, ARENA_H)
    pg.draw.line(screen, (50, 50, 70), (fx, fy1), (fx, fy2), 2)

    # Field label
    font_label = pg.font.SysFont("monospace", 14)
    fl = font_label.render("Gathering Field", True, (80, 80, 100))
    field_cx = (w2s_pt(0, 0)[0] + fx) // 2
    screen.blit(fl, (field_cx - fl.get_width() // 2, ay2 + 5))

    # Machine label
    ml = font_label.render("Binary Counter", True, (80, 80, 100))
    machine_cx = (fx + w2s_pt(ARENA_W, 0)[0]) // 2
    screen.blit(ml, (machine_cx - ml.get_width() // 2, ay2 + 5))

    # Field blobs
    for b in state.field_blobs:
        if not b.exists:
            continue
        sx, sy = w2s_pt(b.pos_x, b.pos_y)
        r = max(4, int(b.size * SCALE * 0.5))
        pg.draw.circle(screen, (100, 180, 255), (sx, sy), r)
        pg.draw.circle(screen, (140, 200, 255), (sx, sy), r, 1)

    # Machine background
    mx1, my1 = w2s_pt(MACHINE_X_START - 20, MACHINE_Y_TOP - 30)
    mx2, my2 = w2s_pt(ARENA_W - 10, MACHINE_Y_BOT + 70)
    pg.draw.rect(screen, (30, 30, 45), (mx1, my1, mx2 - mx1, my2 - my1))
    pg.draw.rect(screen, (80, 80, 120), (mx1, my1, mx2 - mx1, my2 - my1), 2)

    # Column slots with labels
    bit_names = ["1", "2", "4", "8"]
    font_col = pg.font.SysFont("monospace", max(12, int(16 * SCALE)), bold=True)
    font_bit = pg.font.SysFont("monospace", max(18, int(24 * SCALE)), bold=True)

    for i in range(NUM_COLUMNS):
        cx, cy = w2s_pt(COLUMN_X[i], COLUMN_Y)
        col = state.columns[i]
        slot_r = max(8, int(BLOB_SIZES[i] * SCALE * 0.6))

        if col.occupied:
            pg.draw.circle(screen, (80, 160, 255), (cx, cy), slot_r)
            pg.draw.circle(screen, (200, 220, 255), (cx, cy), slot_r, 2)
            # "1" on occupied
            bit_surf = font_bit.render("1", True, (255, 255, 255))
            screen.blit(bit_surf, (cx - bit_surf.get_width() // 2,
                                    cy - bit_surf.get_height() // 2))
        else:
            pg.draw.circle(screen, (40, 40, 55), (cx, cy), slot_r, 2)
            # "0" on empty
            bit_surf = font_bit.render("0", True, (60, 60, 80))
            screen.blit(bit_surf, (cx - bit_surf.get_width() // 2,
                                    cy - bit_surf.get_height() // 2))

        # Column label below
        label = font_col.render(bit_names[i], True, (120, 120, 140))
        screen.blit(label, (cx - label.get_width() // 2, cy + slot_r + 8))

    # Binary readout above machine
    font_readout = pg.font.SysFont("monospace", max(14, int(18 * SCALE)), bold=True)
    readout = f"[{bits_str}] = {count}"
    readout_surf = font_readout.render(readout, True, (100, 220, 255))
    screen.blit(readout_surf, (
        (mx1 + mx2) // 2 - readout_surf.get_width() // 2,
        my1 - 25
    ))

    # Input zone
    ix, iy = w2s_pt(MACHINE_INPUT_X, MACHINE_INPUT_Y)
    pg.draw.circle(screen, (80, 80, 100), (ix, iy), max(4, int(8 * SCALE)), 1)
    inp_label = font_label.render("input", True, (60, 60, 80))
    screen.blit(inp_label, (ix - inp_label.get_width() // 2, iy + 10))

    # Bot
    bot = state.bot
    bx, by = w2s_pt(bot.pos_x, bot.pos_y)
    heading = bot.heading
    bot_size = max(6, int(BOT_RADIUS * SCALE))

    tip_x = bx + int(math.cos(heading) * bot_size)
    tip_y = by + int(math.sin(heading) * bot_size)
    left_x = bx + int(math.cos(heading + 2.4) * bot_size * 0.7)
    left_y = by + int(math.sin(heading + 2.4) * bot_size * 0.7)
    right_x = bx + int(math.cos(heading - 2.4) * bot_size * 0.7)
    right_y = by + int(math.sin(heading - 2.4) * bot_size * 0.7)

    bot_col = (250, 200, 60) if bot.carrying else (80, 200, 180)
    pg.draw.polygon(screen, bot_col,
                     [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)])

    # Carried blob
    if bot.carrying:
        cr = max(3, int(BLOB_BASE_SIZE * SCALE * 0.4))
        pg.draw.circle(screen, (255, 200, 80),
                        (tip_x + int(math.cos(heading) * 8),
                         tip_y + int(math.sin(heading) * 8)), cr)

    # Info overlay
    info_str = f"Step {step}  |  {n_remaining} blobs remaining"
    if bot.carrying:
        info_str += "  |  carrying"
    info_surf = font_label.render(info_str, True, (100, 100, 120))
    screen.blit(info_surf, (10, RENDER_H - 22))

    return screen


def main():
    out_dir = Path("/workspace/bridge/figures/count_states")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BinaryCountingEnv(n_blobs=15, max_steps=2000, seed=42)
    obs = env.reset()

    captured = set()
    # Capture count=0 immediately (before any steps)
    state = env._state
    n_remaining = sum(1 for b in state.field_blobs if b.exists)
    surface = render_state(state, 0, n_remaining, 0)
    pg.image.save(surface, str(out_dir / "count_00.png"))
    captured.add(0)
    print(f"  Captured count=0 (step 0)")

    done = False
    step = 0
    while not done:
        obs, _, done, info = env.step(0)
        step += 1
        state = env._state
        count = info["decimal_count"]
        phase = info["cascade_phase"]
        n_remaining = sum(1 for b in state.field_blobs if b.exists)

        # Capture when machine is idle (settled) and we haven't captured this count yet
        if phase == "idle" and count not in captured:
            surface = render_state(state, count, n_remaining, step)
            pg.image.save(surface, str(out_dir / f"count_{count:02d}.png"))
            captured.add(count)
            print(f"  Captured count={count} (step {step})")

        if len(captured) == 15:  # Got all 0-14
            break

    print(f"\nSaved {len(captured)} screenshots to {out_dir}/")
    print(f"Counts captured: {sorted(captured)}")

    pg.quit()


if __name__ == "__main__":
    main()
