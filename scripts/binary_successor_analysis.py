#!/usr/bin/env python3
"""Binary successor function — deep investigation.

Characterizes what every count transition (0→1, ..., 13→14) looks like inside
the binary specialist's hidden state. Answers: how does the model represent "+1"
when "+1" means completely different physical events depending on carry depth?
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine as cosine_dist

# ─── Paths ────────────────────────────────────────────────────────────────────

BATTERY_PATH = '/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz'
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'binary_successor')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─── Carry depth table ───────────────────────────────────────────────────────

TRANSITIONS = []
for n in range(15):
    bits_n = [(n >> b) & 1 for b in range(4)]
    n1 = n + 1
    if n1 > 14:
        break
    bits_n1 = [(n1 >> b) & 1 for b in range(4)]
    flipped = sum(a != b for a, b in zip(bits_n, bits_n1))
    # Carry depth = number of consecutive carries from bit 0
    carry = 0
    for b in range(4):
        if bits_n[b] == 1 and bits_n1[b] == 0:
            carry += 1
        else:
            break
    TRANSITIONS.append({
        'from': n, 'to': n1,
        'bits_from': ''.join(str((n >> (3-b)) & 1) for b in range(4)),
        'bits_to': ''.join(str((n1 >> (3-b)) & 1) for b in range(4)),
        'carry_depth': carry,
        'bits_flipped': flipped,
    })


def load_data():
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']          # (13280, 512)
    counts = data['counts']     # (13280,)
    bits = data['bits']         # (13280, 4)
    episode_ids = data['episode_ids']  # (13280,)
    timesteps = data['timesteps']      # (13280,)
    return h_t, counts, bits, episode_ids, timesteps


# ─── Analysis 1: Step vectors for every transition ───────────────────────────

def compute_step_vectors(h_t, counts):
    """Compute step vectors as centroid(count=n+1) - centroid(count=n)."""
    centroids = {}
    for c in range(15):
        mask = counts == c
        if mask.sum() > 0:
            centroids[c] = h_t[mask].mean(axis=0)

    step_vectors = []
    magnitudes = []
    for t in TRANSITIONS:
        sv = centroids[t['to']] - centroids[t['from']]
        step_vectors.append(sv)
        magnitudes.append(np.linalg.norm(sv))

    return np.array(step_vectors), magnitudes, centroids


def analysis_1_table(magnitudes):
    """Print the full transition table."""
    print('\n' + '='*90)
    print('ANALYSIS 1: Step Vectors for Every Transition')
    print('='*90)
    print(f'{"Transition":<12} {"Binary change":<16} {"Carry depth":<14} {"Step magnitude":<16} {"Bits flipped"}')
    print('-'*70)
    for i, t in enumerate(TRANSITIONS):
        print(f'{t["from"]:>2}→{t["to"]:<2}       '
              f'{t["bits_from"]}→{t["bits_to"]}     '
              f'{t["carry_depth"]:<14} '
              f'{magnitudes[i]:<16.4f} '
              f'{t["bits_flipped"]}')

    # Correlation between carry depth and magnitude
    depths = [t['carry_depth'] for t in TRANSITIONS]
    corr = np.corrcoef(depths, magnitudes)[0, 1]
    print(f'\nCorrelation (carry depth vs magnitude): r = {corr:.4f}')

    # Group by carry depth
    print('\nBy carry depth:')
    for d in sorted(set(depths)):
        mags = [magnitudes[i] for i, t in enumerate(TRANSITIONS) if t['carry_depth'] == d]
        print(f'  Depth {d}: mean={np.mean(mags):.4f}, std={np.std(mags):.4f}, n={len(mags)}')

    return corr


# ─── Analysis 2: Step vector PCA ─────────────────────────────────────────────

def analysis_2_pca(step_vectors):
    """PCA on 14 step vectors."""
    print('\n' + '='*90)
    print('ANALYSIS 2: Step Vector PCA')
    print('='*90)

    pca = PCA()
    projected = pca.fit_transform(step_vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    for thresh in [0.5, 0.9, 0.95, 0.99]:
        n = np.searchsorted(cumvar, thresh) + 1
        print(f'  Components for {thresh*100:.0f}% variance: {n}')

    print(f'  PC1 explains: {pca.explained_variance_ratio_[0]*100:.1f}%')
    print(f'  PC2 explains: {pca.explained_variance_ratio_[1]*100:.1f}%')
    print(f'  PC3 explains: {pca.explained_variance_ratio_[2]*100:.1f}%')

    # Plot PC1 vs PC2 colored by carry depth
    fig, ax = plt.subplots(figsize=(9, 7))
    depths = [t['carry_depth'] for t in TRANSITIONS]
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, 3)

    for i, t in enumerate(TRANSITIONS):
        color = cmap(norm(t['carry_depth']))
        ax.scatter(projected[i, 0], projected[i, 1], c=[color], s=120,
                   edgecolors='black', linewidth=0.8, zorder=3)
        label = f'{t["from"]}→{t["to"]}'
        ax.annotate(label, (projected[i, 0], projected[i, 1]),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=8, color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.set_label('Carry depth', fontsize=11)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Binary Successor Step Vectors in PCA Space', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'binary_step_pca.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return pca, projected, cumvar


# ─── Analysis 3: Cosine similarity matrix ────────────────────────────────────

def analysis_3_cosine(step_vectors):
    """14x14 cosine similarity matrix."""
    print('\n' + '='*90)
    print('ANALYSIS 3: Cosine Similarity Matrix')
    print('='*90)

    n = len(step_vectors)
    cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_sim[i, j] = 1.0 - cosine_dist(step_vectors[i], step_vectors[j])

    # Group analysis
    depth_groups = {}
    for i, t in enumerate(TRANSITIONS):
        d = t['carry_depth']
        if d not in depth_groups:
            depth_groups[d] = []
        depth_groups[d].append(i)

    print('\nWithin-group mean cosine similarity:')
    for d in sorted(depth_groups.keys()):
        indices = depth_groups[d]
        if len(indices) < 2:
            print(f'  Depth {d}: only {len(indices)} transition (no within-group comparison)')
            continue
        sims = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                sims.append(cos_sim[indices[i], indices[j]])
        print(f'  Depth {d}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, n={len(sims)} pairs')

    print('\nBetween-group mean cosine similarity:')
    for d1 in sorted(depth_groups.keys()):
        for d2 in sorted(depth_groups.keys()):
            if d2 <= d1:
                continue
            sims = []
            for i in depth_groups[d1]:
                for j in depth_groups[d2]:
                    sims.append(cos_sim[i, j])
            print(f'  Depth {d1} vs {d2}: mean={np.mean(sims):.4f}, n={len(sims)} pairs')

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [f'{t["from"]}→{t["to"]}\n(d={t["carry_depth"]})' for t in TRANSITIONS]
    im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=7)

    # Add carry depth separators
    for val in cos_sim.flatten():
        pass  # annotation would be too dense

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cosine similarity', fontsize=11)
    ax.set_title('Step Vector Cosine Similarity\n(annotated by carry depth)', fontsize=14, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'binary_step_cosine_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return cos_sim


# ─── Analysis 4: Per-bit decomposition ───────────────────────────────────────

def analysis_4_bit_decomposition(h_t, counts, bits, step_vectors):
    """Decompose step vectors into per-bit contributions using probe weight directions."""
    print('\n' + '='*90)
    print('ANALYSIS 4: Per-Bit Decomposition of Step Vectors')
    print('='*90)

    # Train per-bit Ridge probes
    probe_weights = []
    for b in range(4):
        probe = Ridge(alpha=1.0)
        probe.fit(h_t, bits[:, b])
        w = probe.coef_ / np.linalg.norm(probe.coef_)  # unit direction
        probe_weights.append(w)
        acc = (np.round(probe.predict(h_t)).astype(int) == bits[:, b]).mean()
        print(f'  Bit {b} probe accuracy: {acc:.4f}')

    # Project each step vector onto each probe direction
    print(f'\n{"Transition":<12} {"Bit0(1s)":<10} {"Bit1(2s)":<10} {"Bit2(4s)":<10} {"Bit3(8s)":<10} {"Expected"}')
    print('-'*70)

    decomposition = []
    for i, t in enumerate(TRANSITIONS):
        sv = step_vectors[i]
        projections = [np.dot(sv, w) for w in probe_weights]
        decomposition.append(projections)

        # Expected: which bits change and in which direction
        n_from, n_to = t['from'], t['to']
        expected = []
        for b in range(4):
            bit_from = (n_from >> b) & 1
            bit_to = (n_to >> b) & 1
            if bit_to > bit_from:
                expected.append('+')
            elif bit_to < bit_from:
                expected.append('-')
            else:
                expected.append('0')

        print(f'{t["from"]:>2}→{t["to"]:<2}       '
              f'{projections[0]:>+8.3f}  '
              f'{projections[1]:>+8.3f}  '
              f'{projections[2]:>+8.3f}  '
              f'{projections[3]:>+8.3f}  '
              f'{"".join(expected)}')

    # Check sign agreement
    correct_signs = 0
    total_changes = 0
    for i, t in enumerate(TRANSITIONS):
        n_from, n_to = t['from'], t['to']
        for b in range(4):
            bit_from = (n_from >> b) & 1
            bit_to = (n_to >> b) & 1
            if bit_from != bit_to:
                total_changes += 1
                expected_sign = 1 if bit_to > bit_from else -1
                actual_sign = 1 if decomposition[i][b] > 0 else -1
                if expected_sign == actual_sign:
                    correct_signs += 1

    print(f'\nSign agreement (changed bits only): {correct_signs}/{total_changes} = {correct_signs/total_changes:.1%}')

    return decomposition, probe_weights


# ─── Analysis 5: Anticipation analysis ───────────────────────────────────────

def analysis_5_anticipation(h_t, counts, episode_ids, timesteps, step_vectors, centroids):
    """Does the hidden state start shifting before a transition happens?"""
    print('\n' + '='*90)
    print('ANALYSIS 5: Anticipation Analysis')
    print('='*90)

    # Find transition points in each episode
    episodes = sorted(np.unique(episode_ids))
    window = 30  # Look back 30 timesteps

    anticipation_by_depth = {0: [], 1: [], 2: [], 3: []}

    for ep in episodes:
        ep_mask = episode_ids == ep
        ep_h = h_t[ep_mask]
        ep_counts = counts[ep_mask]
        ep_ts = timesteps[ep_mask]

        # Sort by timestep within episode
        order = np.argsort(ep_ts)
        ep_h = ep_h[order]
        ep_counts = ep_counts[order]

        # Find transition indices
        for idx in range(1, len(ep_counts)):
            if ep_counts[idx] != ep_counts[idx-1] and ep_counts[idx] == ep_counts[idx-1] + 1:
                c_from = ep_counts[idx-1]
                c_to = ep_counts[idx]
                if c_from >= 14:
                    continue

                t_info = TRANSITIONS[c_from]
                sv = step_vectors[c_from]
                sv_unit = sv / np.linalg.norm(sv)

                # Look at projection onto step vector in preceding timesteps
                start = max(0, idx - window)
                pre_h = ep_h[start:idx+1]  # Include the transition point
                projections = pre_h @ sv_unit

                # Normalize: 0 = at centroid of c_from, 1 = at centroid of c_to
                proj_from = centroids[c_from] @ sv_unit
                proj_to = centroids[c_to] @ sv_unit
                if abs(proj_to - proj_from) > 1e-8:
                    normed = (projections - proj_from) / (proj_to - proj_from)
                else:
                    continue

                anticipation_by_depth[t_info['carry_depth']].append({
                    'normed_trajectory': normed,
                    'actual_len': len(normed),
                    'transition': f'{c_from}→{c_to}',
                })

    # Measure anticipation: at what point does projection cross 0.1 (10% of the way)?
    print(f'\nAnticipation onset (timesteps before transition where projection > 10%):')
    onset_by_depth = {}
    for d in sorted(anticipation_by_depth.keys()):
        events = anticipation_by_depth[d]
        onsets = []
        for ev in events:
            traj = ev['normed_trajectory']
            # Find first point where trajectory exceeds 0.1, counting from the end
            for k in range(len(traj)-1, -1, -1):
                if traj[k] < 0.1:
                    onset = len(traj) - 1 - k - 1
                    onsets.append(onset)
                    break
            else:
                onsets.append(len(traj) - 1)  # Never drops below 0.1

        onset_by_depth[d] = onsets
        if onsets:
            print(f'  Depth {d}: mean onset = {np.mean(onsets):.1f} ± {np.std(onsets):.1f} timesteps before, '
                  f'n={len(onsets)} events')

    # Plot average anticipation trajectory by carry depth
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    labels_depth = ['Depth 0 (simple flip)', 'Depth 1 (1-bit carry)',
                    'Depth 2 (2-bit carry)', 'Depth 3 (full cascade)']

    for d in sorted(anticipation_by_depth.keys()):
        events = anticipation_by_depth[d]
        if not events:
            continue
        # Align trajectories at transition point (last element)
        max_len = min(window + 1, min(e['actual_len'] for e in events))
        aligned = []
        for ev in events:
            traj = ev['normed_trajectory']
            aligned.append(traj[-max_len:])

        aligned = np.array(aligned)
        mean_traj = aligned.mean(axis=0)
        std_traj = aligned.std(axis=0)
        x = np.arange(-max_len + 1, 1)

        ax.plot(x, mean_traj, color=colors[d], linewidth=2.5, label=labels_depth[d])
        ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                        alpha=0.15, color=colors[d])

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axhline(1, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Timesteps relative to transition', fontsize=12)
    ax.set_ylabel('Normalized projection onto step vector', fontsize=12)
    ax.set_title('Anticipation of Count Transitions by Carry Depth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'binary_anticipation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return anticipation_by_depth, onset_by_depth


# ─── Analysis 6: Step magnitude vs carry depth plot ──────────────────────────

def plot_magnitude_vs_carry(magnitudes):
    """Bar chart / scatter of step magnitude vs carry depth."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_by_depth = {0: '#55A868', 1: '#4C72B0', 2: '#DD8452', 3: '#C44E52'}
    x_labels = [f'{t["from"]}→{t["to"]}' for t in TRANSITIONS]

    bars = ax.bar(range(len(magnitudes)), magnitudes, width=0.7, edgecolor='white', linewidth=1)
    for i, t in enumerate(TRANSITIONS):
        bars[i].set_color(colors_by_depth[t['carry_depth']])

    # Add carry depth labels above bars
    for i, (mag, t) in enumerate(zip(magnitudes, TRANSITIONS)):
        ax.text(i, mag + 0.01, f'd={t["carry_depth"]}', ha='center', va='bottom',
                fontsize=7, color=colors_by_depth[t['carry_depth']])

    ax.set_xticks(range(len(magnitudes)))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Step vector magnitude (L2 norm)', fontsize=12)
    ax.set_title('Binary Successor: Step Magnitude by Transition', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors_by_depth[d], label=f'Depth {d}') for d in [0, 1, 2, 3]]
    ax.legend(handles=patches, fontsize=10)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'binary_step_magnitude.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Analysis 7: Grid vs binary comparison ───────────────────────────────────

def analysis_7_comparison(magnitudes, pca, cos_sim):
    """Compare binary successor to grid specialist."""
    print('\n' + '='*90)
    print('ANALYSIS 7: Grid vs Binary Comparison')
    print('='*90)

    # Grid specialist data from counting manifold results
    grid_data_path = '/workspace/bridge/artifacts/tools/data/successor_battery_results.json'
    grid_comparison = {}
    if os.path.exists(grid_data_path):
        with open(grid_data_path) as f:
            grid_results = json.load(f)
        grid_comparison = grid_results
        print(f'  Loaded grid successor results: {list(grid_results.keys())[:10]}...')
    else:
        print(f'  Grid successor results not found at {grid_data_path}')

    # Binary statistics
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    mag_cv = np.std(magnitudes) / np.mean(magnitudes)

    # Adjacent step cosine similarities
    adj_cos = []
    for i in range(len(cos_sim) - 1):
        adj_cos.append(cos_sim[i, i+1])
    mean_adj_cos = np.mean(adj_cos)

    print(f'\n  {"Property":<40} {"Binary specialist":<20}')
    print(f'  {"-"*60}')
    print(f'  {"PCA components for 90%":<40} {n_90:<20}')
    print(f'  {"Step magnitude CV":<40} {mag_cv:<20.4f}')
    print(f'  {"Adjacent step cosine similarity":<40} {mean_adj_cos:<20.4f}')
    print(f'  {"Max step magnitude":<40} {max(magnitudes):<20.4f}')
    print(f'  {"Min step magnitude":<40} {min(magnitudes):<20.4f}')
    print(f'  {"Max/min ratio":<40} {max(magnitudes)/min(magnitudes):<20.4f}')

    # Transition with max and min
    imax = int(np.argmax(magnitudes))
    imin = int(np.argmin(magnitudes))
    print(f'  {"Max transition":<40} {TRANSITIONS[imax]["from"]}→{TRANSITIONS[imax]["to"]} (depth {TRANSITIONS[imax]["carry_depth"]})')
    print(f'  {"Min transition":<40} {TRANSITIONS[imin]["from"]}→{TRANSITIONS[imin]["to"]} (depth {TRANSITIONS[imin]["carry_depth"]})')

    return {
        'n_components_90pct': n_90,
        'magnitude_cv': float(mag_cv),
        'mean_adjacent_cosine': float(mean_adj_cos),
        'max_magnitude': float(max(magnitudes)),
        'min_magnitude': float(min(magnitudes)),
        'max_min_ratio': float(max(magnitudes)/min(magnitudes)),
        'max_transition': f'{TRANSITIONS[imax]["from"]}→{TRANSITIONS[imax]["to"]}',
        'min_transition': f'{TRANSITIONS[imin]["from"]}→{TRANSITIONS[imin]["to"]}',
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('Loading battery data...')
    h_t, counts, bits, episode_ids, timesteps = load_data()
    print(f'  h_t: {h_t.shape}, counts range: {counts.min()}-{counts.max()}, '
          f'{len(np.unique(episode_ids))} episodes')

    # Analysis 1: Step vectors
    step_vectors, magnitudes, centroids = compute_step_vectors(h_t, counts)
    depth_mag_corr = analysis_1_table(magnitudes)

    # Step magnitude plot (key visual)
    plot_magnitude_vs_carry(magnitudes)

    # Analysis 2: PCA
    pca, projected, cumvar = analysis_2_pca(step_vectors)

    # Analysis 3: Cosine similarity
    cos_sim = analysis_3_cosine(step_vectors)

    # Analysis 4: Per-bit decomposition
    decomposition, probe_weights = analysis_4_bit_decomposition(h_t, counts, bits, step_vectors)

    # Analysis 5: Anticipation
    anticipation, onset_by_depth = analysis_5_anticipation(
        h_t, counts, episode_ids, timesteps, step_vectors, centroids)

    # Analysis 7: Comparison
    comparison = analysis_7_comparison(magnitudes, pca, cos_sim)

    # ─── Save results ─────────────────────────────────────────────────────

    results = {
        'transitions': TRANSITIONS,
        'magnitudes': [float(m) for m in magnitudes],
        'depth_magnitude_correlation': float(depth_mag_corr),
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'cosine_similarity_matrix': cos_sim.tolist(),
        'bit_decomposition': [[float(x) for x in row] for row in decomposition],
        'anticipation_onsets': {str(k): [float(x) for x in v] for k, v in onset_by_depth.items()},
        'comparison': comparison,
    }

    out_path = os.path.join(ARTIFACTS_DIR, 'successor_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {out_path}')

    # Summary
    print('\n' + '='*90)
    print('SUMMARY')
    print('='*90)
    print(f'  Carry depth ↔ step magnitude correlation: r = {depth_mag_corr:.4f}')
    print(f'  PCA components for 90%: {comparison["n_components_90pct"]}')
    print(f'  Step magnitude range: {comparison["min_magnitude"]:.4f} → {comparison["max_magnitude"]:.4f} '
          f'(ratio {comparison["max_min_ratio"]:.2f}x)')
    print(f'  Largest step: {comparison["max_transition"]}')
    print(f'  Smallest step: {comparison["min_transition"]}')


if __name__ == '__main__':
    main()
