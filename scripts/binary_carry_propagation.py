#!/usr/bin/env python3
"""Sequential carry propagation — timestep-level analysis.

Does the binary specialist's hidden state track carry cascades sequentially
(bit 0 flips first, then bit 1, etc.) or as a simultaneous jump?

Uses existing battery.npz data (15 episodes, 13280 timesteps, per-timestep h_t).
The binary counting environment propagates carries over ~2 steps per phase,
so a 7→8 cascade takes ~14-16 observation timesteps. The hidden state
should track this sequential process if the model has learned the mechanism.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ─── Paths ────────────────────────────────────────────────────────────────────

BATTERY_PATH = '/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz'
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'binary_successor')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Window around transition to analyze
WINDOW_BEFORE = 15
WINDOW_AFTER = 20  # Carry cascades take up to ~16 steps

# Carry depth for each transition
CARRY_DEPTH = {}
for n in range(15):
    n1 = n + 1
    if n1 > 14:
        break
    depth = 0
    for b in range(4):
        if (n >> b) & 1 == 1 and (n1 >> b) & 1 == 0:
            depth += 1
        else:
            break
    CARRY_DEPTH[n] = depth


def load_data():
    data = np.load(BATTERY_PATH, allow_pickle=True)
    return (data['h_t'], data['counts'], data['bits'],
            data['episode_ids'], data['timesteps'])


def train_bit_probes(h_t, bits):
    """Train 4 per-bit Ridge probes. Return probe objects and unit weight vectors."""
    probes = []
    weights = []
    for b in range(4):
        p = Ridge(alpha=1.0)
        p.fit(h_t, bits[:, b])
        w = p.coef_
        probes.append(p)
        weights.append(w / np.linalg.norm(w))
    return probes, weights


def find_transitions(counts, episode_ids, timesteps):
    """Find all transition points in episode-ordered data.

    Returns list of dicts with episode, transition index in sorted episode,
    count_from, count_to, carry_depth.
    """
    episodes = sorted(np.unique(episode_ids))
    transitions = []

    for ep in episodes:
        ep_mask = episode_ids == ep
        ep_indices = np.where(ep_mask)[0]
        ep_ts = timesteps[ep_mask]
        order = np.argsort(ep_ts)
        ep_indices_sorted = ep_indices[order]
        ep_counts_sorted = counts[ep_indices_sorted]

        for i in range(1, len(ep_counts_sorted)):
            if ep_counts_sorted[i] != ep_counts_sorted[i-1]:
                c_from = int(ep_counts_sorted[i-1])
                c_to = int(ep_counts_sorted[i])
                if c_to == c_from + 1 and c_from < 14:
                    transitions.append({
                        'episode': int(ep),
                        'global_idx': int(ep_indices_sorted[i]),
                        'local_idx': i,
                        'sorted_indices': ep_indices_sorted,
                        'count_from': c_from,
                        'count_to': c_to,
                        'carry_depth': CARRY_DEPTH[c_from],
                    })

    return transitions


def extract_probe_trajectories(h_t, transitions, probe_weights):
    """For each transition, extract 4 bit-probe activations in a window."""
    results = {}

    for tr in transitions:
        key = f'{tr["count_from"]}_{tr["count_to"]}'
        if key not in results:
            results[key] = {
                'carry_depth': tr['carry_depth'],
                'count_from': tr['count_from'],
                'count_to': tr['count_to'],
                'trajectories': [],
            }

        idx = tr['local_idx']
        sorted_idx = tr['sorted_indices']
        start = max(0, idx - WINDOW_BEFORE)
        end = min(len(sorted_idx), idx + WINDOW_AFTER + 1)

        # Get hidden states in window
        window_global = sorted_idx[start:end]
        h_window = h_t[window_global]

        # Project onto 4 bit-probe directions
        projs = np.zeros((len(h_window), 4))
        for b in range(4):
            projs[:, b] = h_window @ probe_weights[b]

        # Transition is at position (idx - start) in the window
        transition_pos = idx - start

        results[key]['trajectories'].append({
            'projections': projs,
            'transition_pos': transition_pos,
            'window_len': len(h_window),
        })

    return results


def normalize_projections(probe_trajectories, probes, h_t, counts):
    """Normalize probe activations to [0, 1] scale per bit.

    0 = mean activation when bit is off, 1 = mean activation when bit is on.
    """
    bit_means = {'off': np.zeros(4), 'on': np.zeros(4)}
    for b in range(4):
        all_preds = probes[b].predict(h_t)
        # Use actual bit values to compute on/off means
        bits_b = np.array([(c >> b) & 1 for c in counts])
        bit_means['off'][b] = all_preds[bits_b == 0].mean()
        bit_means['on'][b] = all_preds[bits_b == 1].mean()

    return bit_means


def plot_transition_probes(probe_traj, bit_means, probes, h_t, transition_key,
                           title_extra=''):
    """Plot 4 bit-probe activations through a specific transition type."""
    info = probe_traj[transition_key]
    trajs = info['trajectories']
    c_from = info['count_from']
    c_to = info['count_to']
    depth = info['carry_depth']

    # Align all trajectories at the transition point
    aligned = []
    for tr in trajs:
        pos = tr['transition_pos']
        projs = tr['projections']
        # Use raw probe predictions (not unit-vector projections)
        aligned.append((projs, pos))

    # Find common window size
    max_before = min(tr['transition_pos'] for tr in trajs)
    max_after = min(tr['window_len'] - tr['transition_pos'] - 1 for tr in trajs)
    max_before = min(max_before, WINDOW_BEFORE)
    max_after = min(max_after, WINDOW_AFTER)

    # Align and average
    n_steps = max_before + max_after + 1
    all_projs = np.zeros((len(trajs), n_steps, 4))
    for i, (projs, pos) in enumerate(aligned):
        start = pos - max_before
        end = pos + max_after + 1
        all_projs[i] = projs[start:end]

    # Normalize each bit to [0, 1] scale
    for b in range(4):
        off_val = bit_means['off'][b]
        on_val = bit_means['on'][b]
        rng = on_val - off_val
        if abs(rng) > 1e-8:
            all_projs[:, :, b] = (all_projs[:, :, b] - off_val) / rng

    mean_projs = all_projs.mean(axis=0)
    std_projs = all_projs.std(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(-max_before, max_after + 1)
    colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
    bit_labels = ['Bit 0 (1s)', 'Bit 1 (2s)', 'Bit 2 (4s)', 'Bit 3 (8s)']

    for b in range(4):
        ax.plot(x, mean_projs[:, b], color=colors[b], linewidth=2.5,
                label=bit_labels[b], zorder=3)
        ax.fill_between(x, mean_projs[:, b] - std_projs[:, b],
                         mean_projs[:, b] + std_projs[:, b],
                         alpha=0.12, color=colors[b])

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='Count changes')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(1, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # Annotate expected bit states
    bits_from = ''.join(str((c_from >> (3-b)) & 1) for b in range(4))
    bits_to = ''.join(str((c_to >> (3-b)) & 1) for b in range(4))
    ax.set_title(f'Carry Propagation: {c_from}→{c_to} ({bits_from}→{bits_to}), '
                 f'depth={depth}{title_extra}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Timesteps relative to count change', fontsize=12)
    ax.set_ylabel('Normalized bit-probe activation (0=off, 1=on)', fontsize=12)
    ax.legend(fontsize=10, loc='center right')
    ax.set_ylim(-0.3, 1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig, mean_projs, std_projs, x


def measure_transition_timing(mean_projs, x, c_from, c_to):
    """Measure when each bit crosses the midpoint (0.5) during a transition."""
    midpoint = 0.5
    timings = {}

    for b in range(4):
        bit_from = (c_from >> b) & 1
        bit_to = (c_to >> b) & 1

        if bit_from == bit_to:
            timings[b] = {'changes': False, 'direction': None, 'crossing_time': None}
            continue

        direction = 'on' if bit_to > bit_from else 'off'
        # Find crossing point in the post-transition region
        trace = mean_projs[:, b]
        crossing = None

        if direction == 'on':
            # Looking for trace crossing 0.5 upward
            for i in range(1, len(trace)):
                if trace[i-1] < midpoint and trace[i] >= midpoint:
                    # Linear interpolation for sub-timestep precision
                    frac = (midpoint - trace[i-1]) / (trace[i] - trace[i-1])
                    crossing = x[i-1] + frac * (x[i] - x[i-1])
                    break
        else:
            # Looking for trace crossing 0.5 downward
            for i in range(1, len(trace)):
                if trace[i-1] > midpoint and trace[i] <= midpoint:
                    frac = (trace[i-1] - midpoint) / (trace[i-1] - trace[i])
                    crossing = x[i-1] + frac * (x[i] - x[i-1])
                    break

        timings[b] = {
            'changes': True,
            'direction': direction,
            'crossing_time': float(crossing) if crossing is not None else None,
        }

    return timings


def carry_stops_here_analysis(probe_traj, bit_means, probes, h_t):
    """Analyze whether non-participating bits show transient activity."""
    print('\n' + '='*90)
    print('ANALYSIS: "Carry Stops Here" — Non-Participating Bit Activity')
    print('='*90)

    # For each transition type, find the first non-participating bit
    # and measure its max deviation during the transition window
    results = {}

    for key, info in sorted(probe_traj.items()):
        c_from = info['count_from']
        c_to = info['count_to']
        depth = info['carry_depth']
        trajs = info['trajectories']

        # Find non-participating bits (bits that don't change)
        non_participating = []
        for b in range(4):
            if (c_from >> b) & 1 == (c_to >> b) & 1:
                non_participating.append(b)

        if not non_participating:
            continue  # All bits change (e.g., 7→8)

        # First non-participating bit above the carry
        first_non = None
        for b in range(4):
            if b > depth - 1 and b in non_participating:
                # This is the bit just above the carry's reach
                first_non = b
                break

        if first_non is None:
            continue

        # Measure deviation of non-participating bit during transition
        deviations = []
        for tr in trajs:
            pos = tr['transition_pos']
            projs = tr['projections']
            # Normalize
            off_val = bit_means['off'][first_non]
            on_val = bit_means['on'][first_non]
            rng = on_val - off_val
            if abs(rng) < 1e-8:
                continue
            normed = (projs[:, first_non] - off_val) / rng

            # Expected value for this bit
            expected = float((c_from >> first_non) & 1)

            # Max deviation from expected in the transition window
            # (look at region around and after transition)
            window_start = max(0, pos - 3)
            window_end = min(len(normed), pos + 15)
            window_vals = normed[window_start:window_end]
            max_dev = np.max(np.abs(window_vals - expected))
            deviations.append(max_dev)

        mean_dev = np.mean(deviations) if deviations else 0
        results[key] = {
            'count_from': c_from, 'count_to': c_to,
            'carry_depth': depth,
            'non_participating_bit': first_non,
            'expected_state': int((c_from >> first_non) & 1),
            'mean_max_deviation': float(mean_dev),
            'n_instances': len(deviations),
        }

    print(f'\n{"Transition":<12} {"Depth":<8} {"Monitor bit":<14} {"Expected":<10} '
          f'{"Max deviation":<16} {"Interpretation"}')
    print('-'*80)
    for key, r in sorted(results.items(), key=lambda x: x[1]['carry_depth']):
        interp = 'Flat (no blip)' if r['mean_max_deviation'] < 0.1 else \
                 'Small blip' if r['mean_max_deviation'] < 0.2 else 'Significant activity'
        print(f'{r["count_from"]:>2}→{r["count_to"]:<2}       '
              f'{r["carry_depth"]:<8} '
              f'bit {r["non_participating_bit"]:<10} '
              f'{r["expected_state"]:<10} '
              f'{r["mean_max_deviation"]:<16.4f} '
              f'{interp}')

    return results


def main():
    print('Loading battery data...')
    h_t, counts, bits, episode_ids, timesteps = load_data()
    print(f'  {h_t.shape[0]} timesteps, {len(np.unique(episode_ids))} episodes')

    print('\nTraining bit probes...')
    probes, probe_weights = train_bit_probes(h_t, bits)
    for b in range(4):
        acc = ((probes[b].predict(h_t) > 0.5).astype(int) == bits[:, b]).mean()
        print(f'  Bit {b}: accuracy = {acc:.4f}')

    print('\nFinding transitions...')
    transitions = find_transitions(counts, episode_ids, timesteps)
    print(f'  Found {len(transitions)} transitions')
    by_depth = {}
    for tr in transitions:
        d = tr['carry_depth']
        by_depth[d] = by_depth.get(d, 0) + 1
    for d in sorted(by_depth):
        print(f'    Depth {d}: {by_depth[d]} instances')

    print('\nExtracting probe trajectories...')
    probe_traj = extract_probe_trajectories(h_t, transitions, probe_weights)

    # Normalize projections using raw probe predictions instead of unit-vector projections
    # Re-extract using raw probe predictions for proper normalization
    probe_traj_raw = {}
    for key, info in probe_traj.items():
        trajs_raw = []
        for tr_data in info['trajectories']:
            # Re-project using full probe (not unit vector)
            projs_raw = np.zeros_like(tr_data['projections'])
            # Find the original global indices
            # We need to reconstruct from the aligned data
            trajs_raw.append(tr_data)  # Keep the unit-vector projections for now
        probe_traj_raw[key] = info

    # Compute normalization constants from raw probe predictions
    bit_means = normalize_projections(probe_traj, probes, h_t, counts)
    print(f'\n  Bit activation ranges (off → on):')
    for b in range(4):
        print(f'    Bit {b}: {bit_means["off"][b]:.4f} → {bit_means["on"][b]:.4f}')

    # Re-extract with actual probe predictions for correct normalization
    probe_traj_actual = {}
    for key, info in probe_traj.items():
        trajs_actual = []
        for orig_tr in info['trajectories']:
            # Recompute using actual probe.predict
            idx = orig_tr['transition_pos']
            projs = orig_tr['projections']
            # The unit-vector projections are proportional to probe predictions
            # but we need actual probe values for normalization
            # Since probe predictions = h @ coef_ + intercept_ and our projections
            # are h @ (coef_/||coef_||), we can recover:
            actual_projs = np.zeros_like(projs)
            trajs_actual.append(orig_tr)

        probe_traj_actual[key] = info

    # Actually, let me re-extract properly with probe.predict()
    print('\nRe-extracting with raw probe predictions...')
    for key, info in probe_traj.items():
        for tr_data in info['trajectories']:
            # We stored projections as h @ unit_weight
            # Raw probe prediction = h @ coef_ + intercept_
            # So actual = projection * ||coef_|| + intercept_
            for b in range(4):
                norm_w = np.linalg.norm(probes[b].coef_)
                tr_data['projections'][:, b] = (
                    tr_data['projections'][:, b] * norm_w + probes[b].intercept_
                )

    # ─── Key transition plots ─────────────────────────────────────────────

    all_timings = {}
    transitions_to_plot = [
        # Full cascade
        ('7_8', '7→8 (Full Cascade)'),
        # Depth-2 carries
        ('3_4', '3→4 (Depth-2 Carry)'),
        ('11_12', '11→12 (Depth-2 Carry)'),
        # Depth-1 carries
        ('1_2', '1→2 (Depth-1 Carry)'),
        # Simple flips (controls)
        ('0_1', '0→1 (Simple Flip, Control)'),
        ('4_5', '4→5 (Simple Flip, Control)'),
        ('8_9', '8→9 (Simple Flip, Control)'),
    ]

    for key, label in transitions_to_plot:
        if key not in probe_traj:
            print(f'  Skipping {key} (not found)')
            continue

        fig, mean_projs, std_projs, x = plot_transition_probes(
            probe_traj, bit_means, probes, h_t, key)

        path = os.path.join(FIGURES_DIR, f'carry_propagation_{key}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')

        # Measure timing
        info = probe_traj[key]
        timings = measure_transition_timing(
            mean_projs, x, info['count_from'], info['count_to'])
        all_timings[key] = timings

    # ─── Combined figure: key transitions side by side ────────────────────

    key_transitions = ['0_1', '1_2', '3_4', '7_8']
    available = [k for k in key_transitions if k in probe_traj]

    if len(available) >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
        bit_labels = ['Bit 0 (1s)', 'Bit 1 (2s)', 'Bit 2 (4s)', 'Bit 3 (8s)']

        for ax_idx, key in enumerate(available):
            ax = axes[ax_idx]
            info = probe_traj[key]
            trajs = info['trajectories']
            c_from, c_to = info['count_from'], info['count_to']
            depth = info['carry_depth']

            # Align and average
            max_before = min(tr['transition_pos'] for tr in trajs)
            max_after = min(tr['window_len'] - tr['transition_pos'] - 1 for tr in trajs)
            max_before = min(max_before, WINDOW_BEFORE)
            max_after = min(max_after, WINDOW_AFTER)
            n_steps = max_before + max_after + 1

            all_projs = np.zeros((len(trajs), n_steps, 4))
            for i, tr in enumerate(trajs):
                pos = tr['transition_pos']
                s = pos - max_before
                e = pos + max_after + 1
                all_projs[i] = tr['projections'][s:e]

            # Normalize
            for b in range(4):
                off_val = bit_means['off'][b]
                on_val = bit_means['on'][b]
                rng = on_val - off_val
                if abs(rng) > 1e-8:
                    all_projs[:, :, b] = (all_projs[:, :, b] - off_val) / rng

            mean_p = all_projs.mean(axis=0)
            std_p = all_projs.std(axis=0)
            x = np.arange(-max_before, max_after + 1)

            for b in range(4):
                ax.plot(x, mean_p[:, b], color=colors[b], linewidth=2,
                        label=bit_labels[b])
                ax.fill_between(x, mean_p[:, b] - std_p[:, b],
                                mean_p[:, b] + std_p[:, b],
                                alpha=0.1, color=colors[b])

            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
            ax.axhline(1, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

            bits_from = ''.join(str((c_from >> (3-b)) & 1) for b in range(4))
            bits_to = ''.join(str((c_to >> (3-b)) & 1) for b in range(4))
            ax.set_title(f'{c_from}→{c_to} ({bits_from}→{bits_to}), depth={depth}',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Timesteps relative to count change', fontsize=10)
            ax.set_ylabel('Normalized activation', fontsize=10)
            ax.set_ylim(-0.3, 1.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if ax_idx == 0:
                ax.legend(fontsize=9, loc='center right')

        fig.suptitle('Carry Propagation Through Binary Specialist Hidden State',
                     fontsize=15, fontweight='bold', y=1.01)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, 'carry_propagation_combined.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')

    # ─── Transition duration table ────────────────────────────────────────

    print('\n' + '='*90)
    print('TRANSITION TIMING ANALYSIS')
    print('='*90)
    print(f'\n{"Transition":<12} {"Depth":<8} {"Bits":<6} ', end='')
    print(f'{"Bit0 crossing":<16} {"Bit1 crossing":<16} {"Bit2 crossing":<16} {"Bit3 crossing":<16} {"Span"}')
    print('-'*105)

    timing_summary = []
    for key in ['0_1', '1_2', '3_4', '7_8', '4_5', '5_6', '8_9', '9_10', '11_12', '13_14']:
        if key not in all_timings:
            continue
        timings = all_timings[key]
        info = probe_traj[key]
        c_from, c_to = info['count_from'], info['count_to']
        depth = info['carry_depth']

        crossing_times = []
        crossing_strs = []
        for b in range(4):
            t = timings[b]
            if t['changes'] and t['crossing_time'] is not None:
                crossing_times.append(t['crossing_time'])
                crossing_strs.append(f'{t["crossing_time"]:>+.1f} ({t["direction"]})')
            elif t['changes']:
                crossing_strs.append('no cross')
            else:
                crossing_strs.append('--')

        span = (max(crossing_times) - min(crossing_times)) if len(crossing_times) > 1 else 0
        n_changing = sum(1 for t in timings.values() if t['changes'])

        row = f'{c_from:>2}→{c_to:<2}       {depth:<8} {n_changing:<6} '
        for s in crossing_strs:
            row += f'{s:<16} '
        row += f'{span:.1f}'
        print(row)

        timing_summary.append({
            'transition': f'{c_from}→{c_to}',
            'carry_depth': depth,
            'bits_changing': n_changing,
            'crossing_times': {str(b): timings[b]['crossing_time'] for b in range(4)
                              if timings[b]['changes']},
            'span': float(span),
        })

    # ─── Propagation order ────────────────────────────────────────────────

    print('\n' + '='*90)
    print('PROPAGATION ORDER ANALYSIS')
    print('='*90)

    for key in ['1_2', '3_4', '7_8', '5_6', '9_10', '11_12', '13_14']:
        if key not in all_timings:
            continue
        timings = all_timings[key]
        info = probe_traj[key]
        c_from, c_to = info['count_from'], info['count_to']
        depth = info['carry_depth']

        if depth == 0:
            continue

        changing_bits = []
        for b in range(4):
            if timings[b]['changes'] and timings[b]['crossing_time'] is not None:
                changing_bits.append((timings[b]['crossing_time'], b, timings[b]['direction']))

        if len(changing_bits) < 2:
            continue

        changing_bits.sort()
        order_str = ' → '.join(f'bit{b}({d})' for _, b, d in changing_bits)
        span = changing_bits[-1][0] - changing_bits[0][0]

        print(f'  {c_from}→{c_to} (depth {depth}): {order_str} [span={span:.1f} steps]')

    # ─── "Carry stops here" analysis ──────────────────────────────────────

    stops_results = carry_stops_here_analysis(probe_traj, bit_means, probes, h_t)

    # ─── Save results ─────────────────────────────────────────────────────

    results = {
        'environment_carry_mechanism': {
            'type': 'multi-timestep cascade',
            'steps_per_phase': 2,
            'phases': ['entering', 'merging', 'carrying'],
            'full_cascade_steps': '14-16',
            'note': 'Carry propagates visibly over multiple observation timesteps. '
                    'Count recomputed from column occupancy each step.',
        },
        'timing_summary': timing_summary,
        'carry_stops_here': {k: v for k, v in stops_results.items()},
    }

    out_path = os.path.join(ARTIFACTS_DIR, 'carry_propagation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results saved to {out_path}')


if __name__ == '__main__':
    main()
