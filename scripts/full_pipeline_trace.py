#!/usr/bin/env python3
"""Task A: Full Pipeline Trace — The Complete Machine from 0 to 14.

Five analyses mapping how the binary specialist RSSM represents the complete
counting sequence:

1. Full episode trajectory — bit-probe activations across entire episode
2. Phase duration analysis — idle/anticipation/transition/settling per transition
3. PCA trajectory geometry — 3D trajectory colored by count
4. Bit-axis occupation heatmap — hierarchical binary structure
5. Inter-transition dynamics — stability during idle periods

All probes use column-state calibration (matching carry propagation methodology).
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# ─── Path setup ───────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = Path('/workspace/projects/jamstack-v1/bridge')
sys.path.insert(0, str(BRIDGE_DIR / 'scripts'))

from binary_counting_env import BinaryCountingEnv, OBS_SIZE
from imagination_rollout_binary import (
    FastRSSMWithImagination, load_exported_weights, carry_depth
)

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLUMNS = 4


# ─── Probe training (column-state, from battery.npz) ─────────────────────────

def train_colstate_probes():
    """Train Ridge probes on actual column states from battery.npz."""
    data = np.load(str(BATTERY_PATH), allow_pickle=True)
    h_t = data['h_t']
    bits = data['bits']
    print(f'  Battery: {h_t.shape[0]} timesteps, {len(np.unique(data["episode_ids"]))} episodes')

    probes = []
    bit_means = {'off': np.zeros(4), 'on': np.zeros(4)}
    for b in range(4):
        p = Ridge(alpha=1.0)
        p.fit(h_t, bits[:, b])
        probes.append(p)
        preds = p.predict(h_t)
        bit_means['off'][b] = preds[bits[:, b] == 0].mean()
        bit_means['on'][b] = preds[bits[:, b] == 1].mean()
        acc = (np.round(preds).astype(int) == bits[:, b]).mean()
        print(f'  Bit {b}: acc={acc:.4f}, off={bit_means["off"][b]:.4f}, on={bit_means["on"][b]:.4f}')

    return probes, bit_means


def project_and_normalize(h_arr, probes, bit_means):
    """Project hidden states through probes, normalize to [0,1]."""
    proj = np.zeros((len(h_arr), 4))
    for b in range(4):
        preds = probes[b].predict(h_arr)
        off_val = bit_means['off'][b]
        on_val = bit_means['on'][b]
        rng = on_val - off_val
        if abs(rng) > 1e-8:
            proj[:, b] = (preds - off_val) / rng
        else:
            proj[:, b] = preds
    return proj


# ─── Data collection: full episode ───────────────────────────────────────────

def collect_full_episode(rssm_weights, seed=42, max_steps=1200):
    """Run a complete episode from 0 to 14, recording everything."""
    rssm = FastRSSMWithImagination(rssm_weights)
    env = BinaryCountingEnv(seed=seed)
    obs = env.reset()
    rssm.reset()

    h_list = []
    col_bits_list = []
    decimal_counts = []
    timesteps = []

    for t in range(max_steps):
        obs_vec = obs[:OBS_SIZE].astype(np.float32)
        deter = rssm.step(obs_vec, action=0)

        col_bits = [1 if env._state.columns[i].occupied else 0
                    for i in range(NUM_COLUMNS)]
        dc = env._state.decimal_count

        h_list.append(deter)
        col_bits_list.append(col_bits)
        decimal_counts.append(dc)
        timesteps.append(t)

        result = env.step(0)
        obs = result[0] if isinstance(result, tuple) else result['obs']

        if dc >= 14:
            # Run a few more steps after reaching 14
            for extra in range(30):
                t += 1
                obs_vec = obs[:OBS_SIZE].astype(np.float32)
                deter = rssm.step(obs_vec, action=0)
                col_bits = [1 if env._state.columns[i].occupied else 0
                            for i in range(NUM_COLUMNS)]
                h_list.append(deter)
                col_bits_list.append(col_bits)
                decimal_counts.append(env._state.decimal_count)
                timesteps.append(t)
                result = env.step(0)
                obs = result[0] if isinstance(result, tuple) else result['obs']
            break

    return {
        'h': np.array(h_list),
        'col_bits': np.array(col_bits_list),
        'decimal_counts': np.array(decimal_counts),
        'timesteps': np.array(timesteps),
    }


def collect_multiple_episodes(rssm_weights, n_episodes=5, max_steps=1200):
    """Collect multiple full episodes for averaging."""
    episodes = []
    for ep in range(n_episodes):
        print(f'  Episode {ep+1}/{n_episodes}...')
        data = collect_full_episode(rssm_weights, seed=42 + ep, max_steps=max_steps)
        episodes.append(data)
        print(f'    {len(data["h"])} steps, reached count={data["decimal_counts"].max()}')
    return episodes


# ─── Analysis 1: Full Episode Trajectory ──────────────────────────────────────

def analysis_1_trajectory(episodes, probes, bit_means):
    """Master plot of 4 bit-probe activations across a full episode."""
    print('\n--- Analysis 1: Full Episode Trajectory ---')

    fig, axes = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    bit_labels = ['Bit0 (1s)', 'Bit1 (2s)', 'Bit2 (4s)', 'Bit3 (8s)']

    # Use first episode for main trace, overlay others faintly
    ep0 = episodes[0]
    proj0 = project_and_normalize(ep0['h'], probes, bit_means)

    ax = axes[0]
    # Faint traces from other episodes
    for ep in episodes[1:]:
        proj = project_and_normalize(ep['h'], probes, bit_means)
        for b in range(4):
            ax.plot(range(len(proj)), proj[:, b], color=colors[b], alpha=0.08, linewidth=0.5)

    # Main trace
    for b in range(4):
        ax.plot(range(len(proj0)), proj0[:, b], color=colors[b], linewidth=1.5,
                label=bit_labels[b], alpha=0.9)

    # Mark transitions (where decimal_count changes)
    dc = ep0['decimal_counts']
    for t in range(1, len(dc)):
        if dc[t] != dc[t-1]:
            depth = carry_depth(dc[t-1])
            lw = 0.5 + depth * 0.5
            ax.axvline(t, color='black', linestyle='--', linewidth=lw, alpha=0.3)
            # Label count above
            if dc[t] <= 14:
                ax.text(t, 1.35, str(dc[t]), ha='center', va='bottom', fontsize=7,
                        fontweight='bold' if depth >= 2 else 'normal')

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axhline(1, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.set_ylabel('Normalized bit-probe activation\n(0=off, 1=on)', fontsize=11)
    ax.set_ylim(-0.3, 1.5)
    ax.set_xlim(0, len(proj0))
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Full Episode: Binary Specialist Hidden State (0→14)',
                 fontsize=14, fontweight='bold')

    # Bottom panel: decimal count
    ax2 = axes[1]
    ax2.step(range(len(dc)), dc, where='post', color='#333333', linewidth=1.5)
    ax2.set_ylabel('Decimal\ncount', fontsize=11)
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylim(-0.5, 15.5)
    ax2.set_xlim(0, len(dc))
    ax2.set_yticks(range(0, 16, 2))

    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_full_trajectory.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')
    return proj0


# ─── Analysis 2: Phase Duration Analysis ─────────────────────────────────────

def analysis_2_phase_durations(episodes, probes, bit_means):
    """Identify idle/anticipation/transition/settling phases for each transition."""
    print('\n--- Analysis 2: Phase Duration Analysis ---')

    # Collect phase durations across episodes
    all_phases = {i: [] for i in range(14)}  # 0→1 through 13→14

    for ep in episodes:
        proj = project_and_normalize(ep['h'], probes, bit_means)
        dc = ep['decimal_counts']

        # Find transition points (where decimal_count changes)
        transitions = []
        for t in range(1, len(dc)):
            if dc[t] == dc[t-1] + 1 and dc[t-1] < 14:
                transitions.append((t, dc[t-1], dc[t]))

        for i, (t_change, c_from, c_to) in enumerate(transitions):
            depth = carry_depth(c_from)
            bits_changing = []
            for b in range(4):
                if ((c_from >> b) & 1) != ((c_to >> b) & 1):
                    bits_changing.append(b)

            if not bits_changing:
                continue

            # Find the FIRST bit to start changing (anticipation start)
            # Look backwards from t_change for first bit crossing 0.5
            first_move = t_change  # default: at decimal change
            for scan_back in range(min(20, t_change)):
                t_check = t_change - scan_back - 1
                if t_check < 0:
                    break
                for b in bits_changing:
                    expected_before = float((c_from >> b) & 1)
                    deviation = abs(proj[t_check, b] - expected_before)
                    if deviation > 0.15:  # Started moving
                        first_move = min(first_move, t_check)

            # Find the LAST bit to settle (settling end)
            last_settle = t_change
            for scan_fwd in range(min(25, len(proj) - t_change - 1)):
                t_check = t_change + scan_fwd
                settled = True
                for b in bits_changing:
                    expected_after = float((c_to >> b) & 1)
                    deviation = abs(proj[t_check, b] - expected_after)
                    if deviation > 0.15:
                        settled = False
                if settled:
                    last_settle = t_check
                    break
            else:
                last_settle = t_change + 20

            # Phase boundaries
            anticipation_start = first_move
            transition_start = t_change - depth * 2  # LSB starts ~2 steps per carry depth before
            settling_end = last_settle

            # Find previous transition end (or start of episode)
            if i > 0:
                prev_t = transitions[i-1][0]
                # Find when previous transition settled
                idle_start = prev_t + 5  # Rough estimate
            else:
                idle_start = 0

            phases = {
                'c_from': c_from,
                'c_to': c_to,
                'depth': depth,
                'idle_duration': max(0, anticipation_start - idle_start),
                'anticipation_duration': max(0, t_change - anticipation_start),
                'cascade_duration': max(1, settling_end - t_change),
                'total_transition': max(1, settling_end - anticipation_start),
                't_change': t_change,
                'first_move': first_move,
                'last_settle': last_settle,
            }
            all_phases[c_from].append(phases)

    # Print table
    print(f'\n{"Trans":<8} {"Depth":<6} {"n":<4} '
          f'{"Idle":<10} {"Anticip":<10} {"Cascade":<10} {"Total":<10}')
    print('-' * 65)

    summary = []
    for c_from in range(14):
        if not all_phases[c_from]:
            continue
        phases_list = all_phases[c_from]
        n = len(phases_list)
        depth = phases_list[0]['depth']
        idle_mean = np.mean([p['idle_duration'] for p in phases_list])
        antic_mean = np.mean([p['anticipation_duration'] for p in phases_list])
        casc_mean = np.mean([p['cascade_duration'] for p in phases_list])
        total_mean = np.mean([p['total_transition'] for p in phases_list])

        print(f'{c_from}→{c_from+1:<4} {depth:<6} {n:<4} '
              f'{idle_mean:<10.1f} {antic_mean:<10.1f} {casc_mean:<10.1f} {total_mean:<10.1f}')

        summary.append({
            'transition': f'{c_from}→{c_from+1}',
            'depth': depth,
            'n': n,
            'idle_mean': float(idle_mean),
            'anticipation_mean': float(antic_mean),
            'cascade_mean': float(casc_mean),
            'total_mean': float(total_mean),
        })

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(summary))
    labels = [s['transition'] for s in summary]
    depths = [s['depth'] for s in summary]

    idle = [s['idle_mean'] for s in summary]
    antic = [s['anticipation_mean'] for s in summary]
    cascade = [s['cascade_mean'] for s in summary]

    ax.bar(x, idle, label='Idle', color='#AECBEB', edgecolor='white')
    ax.bar(x, antic, bottom=idle, label='Anticipation', color='#FFD580', edgecolor='white')
    bottoms = [i + a for i, a in zip(idle, antic)]
    ax.bar(x, cascade, bottom=bottoms, label='Cascade', color='#FF8080', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Duration (timesteps)', fontsize=11)
    ax.set_title('Phase Duration by Transition\n(idle → anticipation → cascade)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Annotate depth
    for i, d in enumerate(depths):
        if d > 0:
            ax.text(i, bottoms[i] + cascade[i] + 1, f'd={d}',
                    ha='center', fontsize=8, color='#666')

    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_phase_durations.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return summary


# ─── Analysis 3: PCA Trajectory Geometry ──────────────────────────────────────

def analysis_3_pca_trajectory(episodes):
    """3D PCA trajectory of hidden state, colored by count."""
    print('\n--- Analysis 3: PCA Trajectory Geometry ---')

    # Use all episodes for PCA fit, plot first
    all_h = np.concatenate([ep['h'] for ep in episodes], axis=0)
    all_dc = np.concatenate([ep['decimal_counts'] for ep in episodes])

    pca = PCA(n_components=3)
    pca.fit(all_h)
    var_explained = pca.explained_variance_ratio_
    print(f'  PCA variance explained: {var_explained[0]:.3f}, {var_explained[1]:.3f}, {var_explained[2]:.3f}')
    print(f'  Total: {sum(var_explained):.3f}')

    # Compute centroids for each count
    centroids = {}
    for c in range(15):
        mask = all_dc == c
        if mask.sum() > 0:
            centroids[c] = pca.transform(all_h[mask].mean(axis=0, keepdims=True))[0]

    # Project first episode
    ep0 = episodes[0]
    proj = pca.transform(ep0['h'])
    dc = ep0['decimal_counts']

    # 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by count
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 14)

    # Plot trajectory as colored line segments
    for t in range(len(proj) - 1):
        color = cmap(norm(dc[t]))
        ax.plot(proj[t:t+2, 0], proj[t:t+2, 1], proj[t:t+2, 2],
                color=color, linewidth=0.8, alpha=0.5)

    # Plot centroids as large spheres
    for c, cent in sorted(centroids.items()):
        ax.scatter(*cent, s=120, c=[cmap(norm(c))], edgecolors='black',
                   linewidths=1.5, zorder=5)
        ax.text(cent[0], cent[1], cent[2] + 0.3, str(c),
                fontsize=9, fontweight='bold', ha='center')

    # Connect centroids with lines to show counting path
    cent_list = [centroids[c] for c in sorted(centroids.keys()) if c in centroids]
    cent_arr = np.array(cent_list)
    ax.plot(cent_arr[:, 0], cent_arr[:, 1], cent_arr[:, 2],
            'k--', linewidth=1.5, alpha=0.4, label='Centroid path')

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})', fontsize=10)
    ax.set_zlabel(f'PC3 ({var_explained[2]:.1%})', fontsize=10)
    ax.set_title('Hidden State Trajectory: PCA of Binary Specialist (0→14)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_pca_trajectory.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    # Also make a 2D version for cleaner visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    proj2d = proj[:, :2]

    # Color by count
    scatter = ax.scatter(proj2d[:, 0], proj2d[:, 1], c=dc, cmap='viridis',
                         s=3, alpha=0.4, vmin=0, vmax=14)
    plt.colorbar(scatter, ax=ax, label='Count')

    # Centroids
    for c, cent in sorted(centroids.items()):
        ax.scatter(cent[0], cent[1], s=200, c=[cmap(norm(c))], edgecolors='black',
                   linewidths=2, zorder=5)
        ax.text(cent[0] + 0.1, cent[1] + 0.1, str(c), fontsize=11, fontweight='bold')

    # Centroid path
    ax.plot(cent_arr[:, 0], cent_arr[:, 1], 'k--', linewidth=1.5, alpha=0.4)

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})', fontsize=11)
    ax.set_title('Hidden State Trajectory: PCA of Binary Specialist (2D)',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_pca_trajectory_2d.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    # Centroid distances
    dists = {}
    for c in range(14):
        if c in centroids and c + 1 in centroids:
            d = np.linalg.norm(centroids[c + 1] - centroids[c])
            dists[f'{c}→{c+1}'] = float(d)
            depth = carry_depth(c)
            print(f'    {c}→{c+1} (depth={depth}): centroid distance = {d:.4f}')

    return {'var_explained': var_explained.tolist(), 'centroid_distances': dists,
            'centroids': {str(k): v.tolist() for k, v in centroids.items()}}


# ─── Analysis 4: Bit-Axis Occupation Heatmap ─────────────────────────────────

def analysis_4_heatmap(episodes, probes, bit_means):
    """4-row heatmap showing hierarchical binary counting structure."""
    print('\n--- Analysis 4: Bit-Axis Occupation Heatmap ---')

    ep0 = episodes[0]
    proj = project_and_normalize(ep0['h'], probes, bit_means)
    dc = ep0['decimal_counts']

    fig, axes = plt.subplots(5, 1, figsize=(20, 8),
                              gridspec_kw={'height_ratios': [1, 1, 1, 1, 0.5]})

    bit_labels = ['Bit3 (8s)', 'Bit2 (4s)', 'Bit1 (2s)', 'Bit0 (1s)']
    bit_order = [3, 2, 1, 0]  # MSB at top

    for row, b in enumerate(bit_order):
        ax = axes[row]
        data = proj[:, b].reshape(1, -1)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=1.1,
                        interpolation='nearest')
        ax.set_yticks([0])
        ax.set_yticklabels([bit_labels[row]], fontsize=10)
        ax.set_xticks([])

        # Mark transitions
        for t in range(1, len(dc)):
            if dc[t] != dc[t-1]:
                ax.axvline(t, color='white', linewidth=0.5, alpha=0.7)

    # Bottom row: count
    ax = axes[4]
    count_data = dc.reshape(1, -1).astype(float)
    im2 = ax.imshow(count_data, aspect='auto', cmap='viridis', vmin=0, vmax=14,
                     interpolation='nearest')
    ax.set_yticks([0])
    ax.set_yticklabels(['Count'], fontsize=10)
    ax.set_xlabel('Timestep', fontsize=11)

    for t in range(1, len(dc)):
        if dc[t] != dc[t-1]:
            ax.axvline(t, color='white', linewidth=0.5, alpha=0.7)

    fig.suptitle('Bit-Axis Occupation: Binary Counting in Hidden State',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_bit_heatmap.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Analysis 5: Inter-Transition Dynamics ───────────────────────────────────

def analysis_5_stability(episodes, probes, bit_means):
    """Measure representational stability during idle periods."""
    print('\n--- Analysis 5: Inter-Transition Dynamics ---')

    stability_results = []

    for ep_idx, ep in enumerate(episodes):
        proj = project_and_normalize(ep['h'], probes, bit_means)
        dc = ep['decimal_counts']

        # Find idle segments (between transitions)
        transitions = []
        for t in range(1, len(dc)):
            if dc[t] != dc[t-1]:
                transitions.append(t)

        # Add boundaries
        boundaries = [0] + transitions + [len(dc)]

        for seg_idx in range(len(boundaries) - 1):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            if end - start < 5:
                continue

            # Trim 3 steps from each end to avoid transition effects
            trim_start = start + 3
            trim_end = end - 3
            if trim_end <= trim_start:
                continue

            segment = proj[trim_start:trim_end]
            count_val = dc[trim_start]

            # Compute stability metrics
            mean_act = segment.mean(axis=0)
            std_act = segment.std(axis=0)
            max_drift = np.abs(segment - segment[0]).max(axis=0)

            # Expected bit states for this count
            expected = np.array([(count_val >> b) & 1 for b in range(4)], dtype=float)
            mean_error = np.abs(mean_act - expected)

            stability_results.append({
                'episode': ep_idx,
                'count': int(count_val),
                'duration': int(trim_end - trim_start),
                'mean_activation': mean_act.tolist(),
                'std_activation': std_act.tolist(),
                'max_drift': max_drift.tolist(),
                'mean_error': mean_error.tolist(),
            })

    # Aggregate by count
    print(f'\n{"Count":<8} {"n":<5} {"Duration":<10} '
          f'{"Mean std":<12} {"Mean drift":<12} {"Mean error":<12}')
    print('-' * 60)

    count_summary = {}
    for c in range(15):
        segs = [s for s in stability_results if s['count'] == c]
        if not segs:
            continue
        n = len(segs)
        dur = np.mean([s['duration'] for s in segs])
        mean_std = np.mean([np.mean(s['std_activation']) for s in segs])
        mean_drift = np.mean([np.mean(s['max_drift']) for s in segs])
        mean_err = np.mean([np.mean(s['mean_error']) for s in segs])

        print(f'{c:<8} {n:<5} {dur:<10.1f} {mean_std:<12.4f} {mean_drift:<12.4f} {mean_err:<12.4f}')
        count_summary[c] = {
            'n': n, 'duration': float(dur),
            'mean_std': float(mean_std), 'mean_drift': float(mean_drift),
            'mean_error': float(mean_err),
        }

    # Plot stability over time for first episode
    ep0 = episodes[0]
    proj0 = project_and_normalize(ep0['h'], probes, bit_means)
    dc0 = ep0['decimal_counts']

    # Compute rolling std (window=5)
    window = 5
    rolling_std = np.zeros((len(proj0) - window, 4))
    for t in range(len(proj0) - window):
        rolling_std[t] = proj0[t:t+window].std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    bit_labels = ['Bit0', 'Bit1', 'Bit2', 'Bit3']

    ax = axes[0]
    for b in range(4):
        ax.plot(range(len(rolling_std)), rolling_std[:, b],
                color=colors[b], linewidth=1, alpha=0.7, label=bit_labels[b])
    ax.set_ylabel('Rolling std (window=5)', fontsize=11)
    ax.set_title('Representational Stability During Idle Periods', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Mark transitions
    for t in range(1, len(dc0)):
        if dc0[t] != dc0[t-1]:
            ax.axvline(t, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    # Bottom: compute error from expected
    error = np.zeros((len(proj0), 4))
    for t in range(len(proj0)):
        c = dc0[t]
        expected = np.array([(c >> b) & 1 for b in range(4)], dtype=float)
        error[t] = np.abs(proj0[t] - expected)

    ax2 = axes[1]
    total_error = error.mean(axis=1)
    ax2.plot(range(len(total_error)), total_error, color='#333', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Mean |activation - expected|', fontsize=11)
    ax2.set_xlabel('Timestep', fontsize=11)

    for t in range(1, len(dc0)):
        if dc0[t] != dc0[t-1]:
            ax2.axvline(t, color='black', linestyle='--', linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / 'pipeline_stability.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return count_summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 90)
    print('TASK A: Full Pipeline Trace — The Complete Machine from 0 to 14')
    print('=' * 90)

    # 1. Train probes
    print('\nTraining column-state probes...')
    probes, bit_means = train_colstate_probes()

    # 2. Load RSSM
    print('\nLoading RSSM weights...')
    rssm_weights = load_exported_weights(str(CKPT_DIR))
    print(f'  Loaded {len(rssm_weights)} tensors')

    # 3. Collect episodes
    print('\nCollecting full episodes (0→14)...')
    episodes = collect_multiple_episodes(rssm_weights, n_episodes=5)

    # 4. Run all 5 analyses
    proj0 = analysis_1_trajectory(episodes, probes, bit_means)
    phase_summary = analysis_2_phase_durations(episodes, probes, bit_means)
    pca_results = analysis_3_pca_trajectory(episodes)
    analysis_4_heatmap(episodes, probes, bit_means)
    stability_results = analysis_5_stability(episodes, probes, bit_means)

    # 5. Save combined results
    results = {
        'n_episodes': len(episodes),
        'episode_lengths': [len(ep['h']) for ep in episodes],
        'max_counts': [int(ep['decimal_counts'].max()) for ep in episodes],
        'phase_durations': phase_summary,
        'pca': pca_results,
        'stability': {str(k): v for k, v in stability_results.items()},
    }

    out_path = ARTIFACTS_DIR / 'full_pipeline_trace.json'
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
