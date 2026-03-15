#!/usr/bin/env python3
"""Imagination Rollout with Column-State Probes (matched to carry propagation).

Re-runs the imagination rollout using probes trained on ACTUAL COLUMN STATES
(from battery.npz bits), not decimal_count-derived bits. This matches the
probe calibration used in binary_carry_propagation.py, allowing direct
comparison between posterior and imagination timing.

The carry propagation analysis showed sequential bit crossings over ~10 steps
in the posterior. This script tests whether imagination (prior-only, no obs)
reproduces that sequential structure.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ─── Path setup ───────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = Path('/workspace/projects/jamstack-v1/bridge')
sys.path.insert(0, str(BRIDGE_DIR / 'scripts'))

from binary_counting_env import BinaryCountingEnv, OBS_SIZE

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Import RSSM and weight loading from existing script ──────────────────────

from imagination_rollout_binary import (
    FastRSSMWithImagination, load_exported_weights, carry_depth
)

NUM_COLUMNS = 4


# ─── Probe training on battery column-state data ─────────────────────────────

def train_colstate_probes():
    """Train probes on actual column states from battery.npz (matching carry propagation)."""
    print('Loading battery data for column-state probe training...')
    data = np.load(str(BATTERY_PATH), allow_pickle=True)
    h_t = data['h_t']          # (13280, 512)
    bits = data['bits']        # (13280, 4) — actual column states
    counts = data['counts']    # (13280,) — decimal_count

    print(f'  Battery: {h_t.shape[0]} timesteps, {len(np.unique(data["episode_ids"]))} episodes')

    probes = []
    weights = []
    accs = []
    for b in range(4):
        p = Ridge(alpha=1.0)
        p.fit(h_t, bits[:, b])
        probes.append(p)
        w = p.coef_
        weights.append(w / np.linalg.norm(w))
        acc = (np.round(p.predict(h_t)).astype(int) == bits[:, b]).mean()
        accs.append(acc)
        print(f'  Bit {b} probe accuracy: {acc:.4f}')

    # Compute global on/off normalization (matching carry propagation)
    bit_means = {'off': np.zeros(4), 'on': np.zeros(4)}
    for b in range(4):
        preds = probes[b].predict(h_t)
        bit_means['off'][b] = preds[bits[:, b] == 0].mean()
        bit_means['on'][b] = preds[bits[:, b] == 1].mean()
        print(f'  Bit {b}: off_mean={bit_means["off"][b]:.4f}, on_mean={bit_means["on"][b]:.4f}')

    return probes, weights, accs, bit_means


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_data(weights, n_episodes=30, max_steps=900,
                 fork_offset=20, imagination_horizon=35,
                 state_buffer_size=25, seed=42):
    """Run episodes, fork to imagination at carry transitions.

    Records ACTUAL COLUMN STATES at each step (not just decimal_count).
    """
    rng = np.random.RandomState(seed)
    rssm = FastRSSMWithImagination(weights)

    transitions_by_depth = {0: [], 1: [], 2: [], 3: []}

    # Collect all posterior h_t and column bits for each episode
    all_posterior_h = []
    all_posterior_colbits = []

    for ep in range(n_episodes):
        env = BinaryCountingEnv(seed=int(rng.randint(0, 100000)))
        obs = env.reset()
        rssm.reset()

        state_buffer = []
        posterior_h_list = []
        posterior_colbits_list = []
        prev_count = 0

        for t in range(max_steps):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            deter = rssm.step(obs_vec, action=0)

            # Read ACTUAL column states (what the carry propagation uses)
            col_bits = [1 if env._state.columns[i].occupied else 0
                        for i in range(NUM_COLUMNS)]
            cur_count = env._state.decimal_count

            state_buffer.append({
                'state': rssm.get_state(),
                'deter': deter.copy(),
                'count': cur_count,
                'col_bits': col_bits[:],
                't': t,
            })
            if len(state_buffer) > state_buffer_size:
                state_buffer.pop(0)

            posterior_h_list.append(deter)
            posterior_colbits_list.append(col_bits)

            # Detect transition (decimal_count changes)
            if cur_count > prev_count and cur_count == prev_count + 1 and prev_count < 14:
                c_from = prev_count
                c_to = cur_count
                depth = carry_depth(c_from)

                # Fork from buffer
                buf_idx = max(0, len(state_buffer) - 1 - fork_offset)
                fork_state = state_buffer[buf_idx]
                actual_fork_offset = t - fork_state['t']

                # Posterior trajectory: capture enough to see the cascade
                # For a depth-3 cascade (~14 steps), we need at least 20 steps back
                post_start = max(0, len(posterior_h_list) - 25)
                post_traj = np.array(posterior_h_list[post_start:])
                post_colbits = np.array(posterior_colbits_list[post_start:])
                transition_idx = len(post_traj) - 1

                # Fork to imagination
                saved_current = rssm.get_state()
                rssm.set_state(fork_state['state'])
                imag_h = [fork_state['deter'].copy()]
                for s in range(imagination_horizon):
                    d = rssm.imagine_step(action=0)
                    imag_h.append(d)
                imag_h = np.array(imag_h)

                rssm.set_state(saved_current)

                transitions_by_depth[depth].append({
                    'episode': ep,
                    'transition_t': t,
                    'c_from': c_from,
                    'c_to': c_to,
                    'depth': depth,
                    'posterior_traj': post_traj,
                    'posterior_colbits': post_colbits,
                    'transition_idx': transition_idx,
                    'imagination_traj': imag_h,
                    'fork_offset': actual_fork_offset,
                })

            prev_count = cur_count
            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']

            if env._state.decimal_count >= 14:
                for extra in range(20):
                    obs_vec = obs[:OBS_SIZE].astype(np.float32)
                    deter = rssm.step(obs_vec, action=0)
                    col_bits = [1 if env._state.columns[i].occupied else 0
                                for i in range(NUM_COLUMNS)]
                    posterior_h_list.append(deter)
                    posterior_colbits_list.append(col_bits)
                break

        # Extend posterior trajectories for already-collected transitions
        all_post_h = np.array(posterior_h_list)
        all_post_cb = np.array(posterior_colbits_list)
        for depth_events in transitions_by_depth.values():
            for ev in depth_events:
                if ev['episode'] == ep:
                    t_idx = ev['transition_idx']
                    post_start = max(0, len(posterior_h_list) - 25 -
                                     (len(posterior_h_list) - ev['transition_t'] - 1))
                    # Re-capture with extended posterior
                    ps = max(0, ev['transition_t'] - 24)
                    pe = min(len(all_post_h), ev['transition_t'] + 21)
                    ev['posterior_traj'] = all_post_h[ps:pe]
                    ev['posterior_colbits'] = all_post_cb[ps:pe]
                    ev['transition_idx'] = ev['transition_t'] - ps

        all_posterior_h.append(np.array(posterior_h_list))
        all_posterior_colbits.append(np.array(posterior_colbits_list))

        if (ep + 1) % 5 == 0 or ep + 1 == n_episodes:
            counts = {d: len(v) for d, v in transitions_by_depth.items()}
            print(f'  Episode {ep+1}/{n_episodes}: {counts}')

    return transitions_by_depth


# ─── Projection and timing measurement ──────────────────────────────────────

def project_and_normalize(h_traj, probes, bit_means):
    """Project hidden states through probes and normalize to [0,1].

    Uses probe.predict() with normalization matching carry propagation.
    0 = off mean, 1 = on mean.
    """
    proj = np.zeros((len(h_traj), 4))
    for b in range(4):
        preds = probes[b].predict(h_traj)
        off_val = bit_means['off'][b]
        on_val = bit_means['on'][b]
        rng = on_val - off_val
        if abs(rng) > 1e-8:
            proj[:, b] = (preds - off_val) / rng
        else:
            proj[:, b] = preds
    return proj


def measure_crossing_time(trace, x_vals, direction, midpoint=0.5):
    """Find when trace crosses midpoint, with sub-step interpolation.

    Matches carry propagation methodology exactly.
    """
    for i in range(1, len(trace)):
        if direction == 'on':
            if trace[i-1] < midpoint and trace[i] >= midpoint:
                frac = (midpoint - trace[i-1]) / (trace[i] - trace[i-1])
                return x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1])
        else:
            if trace[i-1] > midpoint and trace[i] <= midpoint:
                frac = (trace[i-1] - midpoint) / (trace[i-1] - trace[i])
                return x_vals[i-1] + frac * (x_vals[i] - x_vals[i-1])
    return None


def analyze_transitions(transitions_by_depth, probes, bit_means):
    """Analyze posterior and imagination timing with column-state probes."""
    results = {}

    for depth in sorted(transitions_by_depth.keys()):
        events = transitions_by_depth[depth]
        if not events:
            continue

        c_from = events[0]['c_from']
        c_to = events[0]['c_to']

        # Which bits change?
        bits_that_change = []
        bits_that_stay = []
        for b in range(4):
            bf = (c_from >> b) & 1
            bt = (c_to >> b) & 1
            if bf != bt:
                bits_that_change.append(b)
            else:
                bits_that_stay.append(b)

        # Collect crossings for posterior and imagination
        post_crossings = {b: [] for b in bits_that_change}
        imag_crossings = {b: [] for b in bits_that_change}
        post_stay_devs = {b: [] for b in bits_that_stay}
        imag_stay_devs = {b: [] for b in bits_that_stay}

        post_projs_all = []
        imag_projs_all = []

        for ev in events:
            # Posterior: project and normalize
            post_proj = project_and_normalize(ev['posterior_traj'], probes, bit_means)
            tidx = ev['transition_idx']
            x_post = np.arange(len(post_proj)) - tidx  # Centered on transition

            post_projs_all.append({
                'proj': post_proj, 'x': x_post, 'tidx': tidx,
            })

            # Posterior crossing times
            for b in bits_that_change:
                bf_val = (c_from >> b) & 1
                bt_val = (c_to >> b) & 1
                direction = 'on' if bt_val > bf_val else 'off'
                ct = measure_crossing_time(post_proj[:, b], x_post, direction)
                if ct is not None:
                    post_crossings[b].append(ct)

            # Posterior carry bleed
            for b in bits_that_stay:
                w_start = max(0, tidx - 15)
                w_end = min(len(post_proj), tidx + 20)
                window_vals = post_proj[w_start:w_end, b]
                if len(window_vals) > 0:
                    post_stay_devs[b].append(float(np.abs(window_vals - window_vals[0]).max()))

            # Imagination: project and normalize
            imag_proj = project_and_normalize(ev['imagination_traj'], probes, bit_means)
            fork_off = ev['fork_offset']
            x_imag = np.arange(len(imag_proj)) - fork_off  # Centered on fork point
            # Re-center on where the transition WOULD happen
            x_imag_trans = x_imag  # Already relative to fork; transition is at +fork_offset

            imag_projs_all.append({
                'proj': imag_proj, 'x': x_imag, 'fork_off': fork_off,
            })

            # Imagination crossing times (relative to transition point)
            for b in bits_that_change:
                bf_val = (c_from >> b) & 1
                bt_val = (c_to >> b) & 1
                direction = 'on' if bt_val > bf_val else 'off'
                ct = measure_crossing_time(imag_proj[:, b], x_imag, direction)
                if ct is not None:
                    imag_crossings[b].append(ct)

            # Imagination carry bleed
            for b in bits_that_stay:
                window_vals = imag_proj[fork_off:min(len(imag_proj), fork_off + 25), b]
                if len(window_vals) > 0:
                    imag_stay_devs[b].append(float(np.abs(window_vals - window_vals[0]).max()))

        # Compute summary statistics
        post_means = {}
        imag_means = {}
        for b in bits_that_change:
            if post_crossings[b]:
                post_means[b] = float(np.mean(post_crossings[b]))
            if imag_crossings[b]:
                imag_means[b] = float(np.mean(imag_crossings[b]))

        # Span = max crossing - min crossing (sequential = large span)
        post_vals = [post_means[b] for b in bits_that_change if b in post_means]
        imag_vals = [imag_means[b] for b in bits_that_change if b in imag_means]
        post_span = (max(post_vals) - min(post_vals)) if len(post_vals) > 1 else 0.0
        imag_span = (max(imag_vals) - min(imag_vals)) if len(imag_vals) > 1 else 0.0

        # Sequential = bits cross in LSB→MSB order
        post_ordered = all(post_means.get(bits_that_change[i], float('inf')) <=
                          post_means.get(bits_that_change[i+1], float('inf'))
                          for i in range(len(bits_that_change)-1)) if len(bits_that_change) > 1 else True
        imag_ordered = all(imag_means.get(bits_that_change[i], float('inf')) <=
                          imag_means.get(bits_that_change[i+1], float('inf'))
                          for i in range(len(bits_that_change)-1)) if len(bits_that_change) > 1 else True

        post_bleed = max((max(post_stay_devs[b]) if post_stay_devs[b] else 0)
                        for b in bits_that_stay) if bits_that_stay else 0
        imag_bleed = max((max(imag_stay_devs[b]) if imag_stay_devs[b] else 0)
                        for b in bits_that_stay) if bits_that_stay else 0

        results[depth] = {
            'c_from': c_from, 'c_to': c_to, 'depth': depth,
            'bits_that_change': bits_that_change,
            'bits_that_stay': bits_that_stay,
            'n_events': len(events),
            'post_crossings': {str(b): post_crossings[b] for b in bits_that_change},
            'imag_crossings': {str(b): imag_crossings[b] for b in bits_that_change},
            'post_means': {str(b): post_means.get(b) for b in bits_that_change},
            'imag_means': {str(b): imag_means.get(b) for b in bits_that_change},
            'post_span': post_span, 'imag_span': imag_span,
            'post_sequential': post_ordered, 'imag_sequential': imag_ordered,
            'post_bleed': post_bleed, 'imag_bleed': imag_bleed,
            'post_projs': post_projs_all,
            'imag_projs': imag_projs_all,
        }

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(results):
    """2×4 grid: posterior (top) vs imagination (bottom), 4 carry depths."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    bit_labels = ['Bit0 (1s)', 'Bit1 (2s)', 'Bit2 (4s)', 'Bit3 (8s)']

    depth_order = [0, 1, 2, 3]
    window = 25  # Show ±25 steps

    for col, depth in enumerate(depth_order):
        if depth not in results:
            continue
        r = results[depth]

        # ─── Posterior (top row) ─────────────────────────────────
        ax = axes[0, col]
        for pp in r['post_projs']:
            proj, x = pp['proj'], pp['x']
            mask = (x >= -window) & (x <= window)
            for b in range(4):
                ax.plot(x[mask], proj[mask, b], color=colors[b], alpha=0.15, linewidth=0.7)

        # Mean trajectory
        aligned = []
        for pp in r['post_projs']:
            proj, x = pp['proj'], pp['x']
            interp_x = np.arange(-window, window + 1)
            interp_proj = np.zeros((len(interp_x), 4))
            for b in range(4):
                interp_proj[:, b] = np.interp(interp_x, x, proj[:, b],
                                              left=proj[0, b], right=proj[-1, b])
            aligned.append(interp_proj)
        if aligned:
            mean_proj = np.mean(aligned, axis=0)
            for b in range(4):
                ax.plot(interp_x, mean_proj[:, b], color=colors[b], linewidth=2.5,
                        label=bit_labels[b] if col == 0 else None)

        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_title(f'{r["c_from"]}→{r["c_to"]} (depth {depth})\nPOSTERIOR (with obs)',
                     fontsize=11, fontweight='bold')
        ax.set_ylim(-0.3, 1.4)
        if col == 0:
            ax.set_ylabel('Normalized bit-probe\nactivation (0=off, 1=on)', fontsize=10)

        # Mark crossing times
        for b in r['bits_that_change']:
            bkey = str(b)
            if r['post_means'].get(bkey) is not None:
                ax.axvline(r['post_means'][bkey], color=colors[b],
                           linestyle=':', linewidth=1, alpha=0.6)

        # ─── Imagination (bottom row) ───────────────────────────
        ax = axes[1, col]
        for ip in r['imag_projs']:
            proj, x = ip['proj'], ip['x']
            mask = (x >= -window) & (x <= window)
            for b in range(4):
                ax.plot(x[mask], proj[mask, b], color=colors[b], alpha=0.15, linewidth=0.7)

        # Mean trajectory
        aligned = []
        for ip in r['imag_projs']:
            proj, x = ip['proj'], ip['x']
            interp_proj = np.zeros((len(interp_x), 4))
            for b in range(4):
                interp_proj[:, b] = np.interp(interp_x, x, proj[:, b],
                                              left=proj[0, b], right=proj[-1, b])
            aligned.append(interp_proj)
        if aligned:
            mean_proj = np.mean(aligned, axis=0)
            for b in range(4):
                ax.plot(interp_x, mean_proj[:, b], color=colors[b], linewidth=2.5)

        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_title(f'{r["c_from"]}→{r["c_to"]} (depth {depth})\nIMAGINATION (no obs)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Timesteps relative to transition', fontsize=10)
        ax.set_ylim(-0.3, 1.4)
        if col == 0:
            ax.set_ylabel('Normalized bit-probe\nactivation (0=off, 1=on)', fontsize=10)

        # Mark crossing times
        for b in r['bits_that_change']:
            bkey = str(b)
            if r['imag_means'].get(bkey) is not None:
                ax.axvline(r['imag_means'][bkey], color=colors[b],
                           linestyle=':', linewidth=1, alpha=0.6)

    fig.suptitle('Does the RSSM Simulate Carries Without Observations?\n'
                 '(Column-state probes, matched to carry propagation analysis)',
                 fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('='*90)
    print('IMAGINATION ROLLOUT — Column-State Probes (matched methodology)')
    print('='*90)

    # 1. Train probes on battery column-state data
    probes, weights, accs, bit_means = train_colstate_probes()

    # 2. Load RSSM weights
    print('\nLoading binary specialist weights...')
    rssm_weights = load_exported_weights(str(CKPT_DIR))
    print(f'  Loaded {len(rssm_weights)} weight tensors')

    # 3. Collect data
    print('\nCollecting imagination rollout data (30 episodes)...')
    transitions = collect_data(rssm_weights, n_episodes=30, fork_offset=20,
                               imagination_horizon=35, state_buffer_size=25)

    for d in sorted(transitions.keys()):
        print(f'  Depth {d}: {len(transitions[d])} transitions')

    # 4. Analyze timing
    print('\nAnalyzing timing with column-state probes...')
    results = analyze_transitions(transitions, probes, bit_means)

    # 5. Print results table
    print('\n' + '='*100)
    print('RESULTS: Posterior vs Imagination Cascade Timing (Column-State Probes)')
    print('='*100)
    print(f'{"Trans":<10} {"Depth":<6} {"n":<5} '
          f'{"Post span":<12} {"Post seq?":<12} '
          f'{"Imag span":<12} {"Imag seq?":<12} '
          f'{"Post bleed":<12} {"Imag bleed":<12}')
    print('-'*100)

    for depth in sorted(results.keys()):
        r = results[depth]
        print(f'{r["c_from"]}→{r["c_to"]:<6}  {depth:<6} {r["n_events"]:<5} '
              f'{r["post_span"]:<12.1f} {"Yes" if r["post_sequential"] else "No":<12} '
              f'{r["imag_span"]:<12.1f} {"Yes" if r["imag_sequential"] else "No":<12} '
              f'{r["post_bleed"]:<12.3f} {r["imag_bleed"]:<12.3f}')

        # Per-bit crossing times
        for b in r['bits_that_change']:
            bkey = str(b)
            bf = (r['c_from'] >> b) & 1
            bt = (r['c_to'] >> b) & 1
            direction = '↑' if bt > bf else '↓'
            post_t = r['post_means'].get(bkey)
            imag_t = r['imag_means'].get(bkey)
            post_n = len(r['post_crossings'].get(bkey, []))
            imag_n = len(r['imag_crossings'].get(bkey, []))
            post_str = f'{post_t:+.1f} (n={post_n})' if post_t is not None else 'no cross'
            imag_str = f'{imag_t:+.1f} (n={imag_n})' if imag_t is not None else 'no cross'
            print(f'    bit{b} ({direction}): posterior {post_str}, imagination {imag_str}')

    # 6. Outcome classification
    print('\n' + '='*90)
    print('OUTCOME CLASSIFICATION')
    print('='*90)

    # Check if imagination shows sequential cascades at depth >= 2
    d3 = results.get(3)
    d2 = results.get(2)
    d1 = results.get(1)

    if d3 and d3['imag_sequential'] and d3['imag_span'] > 2.0:
        outcome = 'A'
        desc = ('OUTCOME A (REVISED): Sequential cascade in imagination!\n'
                '  The column-state probes reveal sequential bit flips in imagination.\n'
                '  The model IS simulating the carry cascade from internal dynamics.')
    elif d3 and not d3['imag_sequential']:
        # Check if there's any sequential structure at lower depths
        has_partial = (d2 and d2['imag_sequential'] and d2['imag_span'] > 1.0) or \
                      (d1 and d1['imag_sequential'] and d1['imag_span'] > 0.5)
        if has_partial:
            outcome = 'C'
            desc = ('OUTCOME C (CONFIRMED): Partial cascade in imagination.\n'
                    '  Some sequential structure at lower depths, but depth-3 is not sequential.\n'
                    '  Confirmed with matched probes — not a measurement artifact.')
        else:
            outcome = 'B'
            desc = ('OUTCOME B: Simultaneous/unordered flip in imagination.\n'
                    '  The model predicts WHAT happens but not HOW it unfolds.')
    elif d3 and d3['imag_sequential'] and d3['imag_span'] <= 2.0:
        outcome = 'C'
        desc = ('OUTCOME C: Partial cascade — some ordering but compressed span.\n'
                f'  Imagination span = {d3["imag_span"]:.1f} vs posterior span = {d3["post_span"]:.1f}')
    else:
        outcome = 'D'
        desc = 'OUTCOME D: Insufficient data to classify.'

    print(f'\n  {desc}')

    # Compare posterior timing to carry propagation results
    print('\n  Comparison to carry propagation analysis:')
    if d3 and '0' in (r.get('post_means') or {}):
        cp_timings = {'0': -10.4, '1': -6.4, '2': -2.5, '3': -0.5}
        print(f'    Carry prop (battery):  bit0={cp_timings["0"]:+.1f}, '
              f'bit1={cp_timings["1"]:+.1f}, bit2={cp_timings["2"]:+.1f}, '
              f'bit3={cp_timings["3"]:+.1f}, span=9.9')
        d3_post = results[3]['post_means']
        print(f'    This run (posterior):  ', end='')
        for b in range(4):
            bkey = str(b)
            if d3_post.get(bkey) is not None:
                print(f'bit{b}={d3_post[bkey]:+.1f}, ', end='')
        print(f'span={results[3]["post_span"]:.1f}')

    # 7. Generate figure
    print('\nGenerating comparison figure...')
    fig = plot_comparison(results)
    fig_path = str(FIGURES_DIR / 'imagination_vs_posterior_colstate.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fig_path}')

    # 8. Save results (exclude numpy arrays)
    save_results = {
        'outcome': outcome,
        'probe_type': 'column_state (from battery.npz bits)',
        'n_episodes': 30,
        'probe_accs': accs,
    }
    for depth, r in results.items():
        save_results[f'depth_{depth}'] = {
            'c_from': r['c_from'], 'c_to': r['c_to'],
            'n_events': r['n_events'],
            'bits_that_change': r['bits_that_change'],
            'bits_that_stay': r['bits_that_stay'],
            'post_span': r['post_span'], 'imag_span': r['imag_span'],
            'post_sequential': r['post_sequential'],
            'imag_sequential': r['imag_sequential'],
            'post_bleed': r['post_bleed'], 'imag_bleed': r['imag_bleed'],
            'post_means': r['post_means'],
            'imag_means': r['imag_means'],
            'post_crossings': {k: v for k, v in r['post_crossings'].items()},
            'imag_crossings': {k: v for k, v in r['imag_crossings'].items()},
        }

    json_path = str(ARTIFACTS_DIR / 'imagination_rollout_colstate.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'  Results saved to {json_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
