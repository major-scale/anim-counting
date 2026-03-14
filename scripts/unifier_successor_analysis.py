#!/usr/bin/env python3
"""Does compositional structure survive unification?

Runs the binary successor decomposition analysis on unifier hidden states (h_u, 256-dim)
instead of specialist hidden states (h_t, 512-dim). Compares across 7 unifier conditions
to test whether clean factored bit-level structure survives the adapter + GRU integration.

Three possible outcomes:
  A) Structure survives everywhere — adapters faithfully relay compositional structure
  B) Structure survives in some conditions — alignment loss or VICReg modulates fidelity
  C) Structure degrades everywhere — GRU integration necessarily destroys factoring

Uses battery_final.npz (h_a, h_b, counts) as canonical inputs, processes them
through each checkpoint independently.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine as cosine_dist

import torch
import torch.nn as nn

# Add scripts dir to path for unifier_model import
SCRIPTS_DIR = '/workspace/projects/jamstack-v1/bridge/scripts'
sys.path.insert(0, SCRIPTS_DIR)
from unifier_model import UnifierRSSM

# ─── Paths ────────────────────────────────────────────────────────────────────

BATTERY_PATH = '/workspace/projects/jamstack-v1/bridge/artifacts/checkpoints/unifier_s0/battery_final.npz'
CKPT_BASE = '/workspace/projects/jamstack-v1/bridge/artifacts/checkpoints'
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'binary_successor')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─── Conditions to test ──────────────────────────────────────────────────────

CONDITIONS = [
    ('lambda_0.1',      f'{CKPT_BASE}/unifier_s0/best.pt',             'λ=0.1 (base)'),
    ('lambda_0.005',    f'{CKPT_BASE}/sweep_lambda_0.005/best.pt',     'λ=0.005 (sweet spot)'),
    ('lambda_0.05',     f'{CKPT_BASE}/sweep_lambda_0.05/best.pt',      'λ=0.05'),
    ('no_align',        f'{CKPT_BASE}/unifier_no_align_s0/best.pt',    'λ=0.0 (no alignment)'),
    ('vicreg_70k',      f'{CKPT_BASE}/unifier_vicreg_s0/best.pt',      'VICReg (70K steps)'),
    ('onset_5000',      f'{CKPT_BASE}/imprinting_onset_5000/best.pt',  'Late VICReg (onset 5000)'),
    ('onset_0',         f'{CKPT_BASE}/imprinting_N400_onset0/best.pt', 'Early VICReg (onset 0)'),
]

# ─── Carry depth table ───────────────────────────────────────────────────────

def build_transitions(max_count):
    transitions = []
    for n in range(max_count):
        bits_n = [(n >> b) & 1 for b in range(4)]
        n1 = n + 1
        bits_n1 = [(n1 >> b) & 1 for b in range(4)]
        carry = 0
        for b in range(4):
            if bits_n[b] == 1 and bits_n1[b] == 0:
                carry += 1
            else:
                break
        transitions.append({
            'from': n, 'to': n1,
            'bits_from': ''.join(str((n >> (3-b)) & 1) for b in range(4)),
            'bits_to': ''.join(str((n1 >> (3-b)) & 1) for b in range(4)),
            'carry_depth': carry,
        })
    return transitions


# ─── Load data and run through checkpoints ────────────────────────────────────

def load_battery():
    """Load canonical (h_a, h_b, counts) from battery_final.npz."""
    data = np.load(BATTERY_PATH)
    h_a = data['h_a']      # (2800, 512)
    h_b = data['h_b']      # (2800, 512)
    counts = data['counts'] # (2800,)
    # Derive bits from counts
    bits = np.array([[(c >> b) & 1 for b in range(4)] for c in counts])
    return h_a, h_b, counts, bits


def collect_h_u(checkpoint_path, h_a, h_b, batch_size=512):
    """Run (h_a, h_b) through a unifier checkpoint, return h_u."""
    device = torch.device('cpu')
    model = UnifierRSSM().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    N = len(h_a)
    all_h_u = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            ha = torch.tensor(h_a[start:end], dtype=torch.float32, device=device)
            hb = torch.tensor(h_b[start:end], dtype=torch.float32, device=device)
            B = ha.shape[0]

            h_prev, z_prev = model.initial_state(B, device)
            out = model.step(ha, hb, h_prev, z_prev)
            all_h_u.append(out['h_u'].numpy())

    return np.concatenate(all_h_u, axis=0)  # (N, 256)


# ─── Successor analysis functions ────────────────────────────────────────────

def compute_centroids(h, counts, max_count):
    """Compute per-count centroids."""
    centroids = {}
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids[c] = h[mask].mean(axis=0)
    return centroids


def compute_step_vectors(centroids, transitions):
    """Step vectors = centroid(n+1) - centroid(n)."""
    step_vectors = []
    magnitudes = []
    for t in transitions:
        if t['from'] in centroids and t['to'] in centroids:
            sv = centroids[t['to']] - centroids[t['from']]
            step_vectors.append(sv)
            magnitudes.append(np.linalg.norm(sv))
        else:
            step_vectors.append(np.zeros(list(centroids.values())[0].shape))
            magnitudes.append(0.0)
    return np.array(step_vectors), magnitudes


def train_bit_probes(h, bits):
    """Train 4 Ridge probes for individual bits."""
    probe_weights = []
    accuracies = []
    for b in range(4):
        probe = Ridge(alpha=1.0)
        probe.fit(h, bits[:, b])
        w = probe.coef_ / np.linalg.norm(probe.coef_)  # unit direction
        probe_weights.append(w)
        acc = (np.round(probe.predict(h)).astype(int) == bits[:, b]).mean()
        accuracies.append(acc)
    return probe_weights, accuracies


def bit_decomposition(step_vectors, probe_weights, transitions):
    """Project step vectors onto probe directions."""
    decomposition = []
    for i, t in enumerate(transitions):
        sv = step_vectors[i]
        projections = [float(np.dot(sv, w)) for w in probe_weights]
        decomposition.append(projections)
    return decomposition


def sign_agreement(decomposition, transitions):
    """Check: does the projection sign match expected bit changes?"""
    correct = 0
    total = 0
    for i, t in enumerate(transitions):
        n_from, n_to = t['from'], t['to']
        for b in range(4):
            bit_from = (n_from >> b) & 1
            bit_to = (n_to >> b) & 1
            if bit_from != bit_to:
                total += 1
                expected_sign = 1 if bit_to > bit_from else -1
                actual_sign = 1 if decomposition[i][b] > 0 else -1
                if expected_sign == actual_sign:
                    correct += 1
    return correct, total


def cross_talk_ratio(decomposition, transitions):
    """Ratio of projection on non-participating bits vs participating bits."""
    participating_proj = []
    non_participating_proj = []

    for i, t in enumerate(transitions):
        n_from, n_to = t['from'], t['to']
        for b in range(4):
            bit_from = (n_from >> b) & 1
            bit_to = (n_to >> b) & 1
            proj_mag = abs(decomposition[i][b])
            if bit_from != bit_to:
                participating_proj.append(proj_mag)
            else:
                non_participating_proj.append(proj_mag)

    if not participating_proj:
        return float('inf')
    mean_part = np.mean(participating_proj)
    mean_nonpart = np.mean(non_participating_proj)
    return float(mean_nonpart / mean_part) if mean_part > 0 else float('inf')


def cosine_matrix(step_vectors):
    """Full cosine similarity matrix."""
    n = len(step_vectors)
    cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(step_vectors[i])
            norm_j = np.linalg.norm(step_vectors[j])
            if norm_i > 1e-10 and norm_j > 1e-10:
                cos_sim[i, j] = 1.0 - cosine_dist(step_vectors[i], step_vectors[j])
    return cos_sim


def cosine_within_between(cos_sim, transitions):
    """Mean cosine within depth-0 and between depth-0 vs depth-1."""
    depth_groups = {}
    for i, t in enumerate(transitions):
        d = t['carry_depth']
        if d not in depth_groups:
            depth_groups[d] = []
        depth_groups[d].append(i)

    # Within depth-0
    within_0 = []
    if 0 in depth_groups and len(depth_groups[0]) >= 2:
        for i in range(len(depth_groups[0])):
            for j in range(i+1, len(depth_groups[0])):
                within_0.append(cos_sim[depth_groups[0][i], depth_groups[0][j]])

    # Between depth-0 and depth-1
    between_01 = []
    if 0 in depth_groups and 1 in depth_groups:
        for i in depth_groups[0]:
            for j in depth_groups[1]:
                between_01.append(cos_sim[i, j])

    return (
        float(np.mean(within_0)) if within_0 else float('nan'),
        float(np.mean(between_01)) if between_01 else float('nan'),
    )


def pca_components_90(step_vectors):
    """Number of PCA components for 90% variance."""
    pca = PCA()
    pca.fit(step_vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    return n_90, pca


# ─── Run full analysis for one condition ──────────────────────────────────────

def analyze_condition(h, counts, bits, transitions, label):
    """Run full successor decomposition on hidden states h."""
    max_count = int(counts.max())
    centroids = compute_centroids(h, counts, max_count)
    step_vectors, magnitudes = compute_step_vectors(centroids, transitions)

    # Magnitude-depth correlation
    depths = [t['carry_depth'] for t in transitions]
    mag_depth_corr = float(np.corrcoef(depths, magnitudes)[0, 1])

    # Per-bit probes
    probe_weights, probe_accs = train_bit_probes(h, bits)

    # Decomposition
    decomp = bit_decomposition(step_vectors, probe_weights, transitions)
    correct, total = sign_agreement(decomp, transitions)
    sign_pct = correct / total if total > 0 else 0.0
    xtalk = cross_talk_ratio(decomp, transitions)

    # Cosine similarity
    cos_sim = cosine_matrix(step_vectors)
    within_0, between_01 = cosine_within_between(cos_sim, transitions)

    # PCA
    n_90, pca = pca_components_90(step_vectors)

    results = {
        'label': label,
        'mag_depth_corr': mag_depth_corr,
        'sign_agreement': f'{correct}/{total}',
        'sign_pct': sign_pct,
        'cross_talk': xtalk,
        'within_depth0_cosine': within_0,
        'between_d0_d1_cosine': between_01,
        'pca_90': n_90,
        'magnitudes': [float(m) for m in magnitudes],
        'probe_accs': [float(a) for a in probe_accs],
        'decomposition': decomp,
        'cosine_matrix': cos_sim.tolist(),
        'pca_variance': pca.explained_variance_ratio_.tolist(),
    }

    return results


# ─── Specialist baseline ─────────────────────────────────────────────────────

def specialist_baseline(h_b, counts, bits, transitions):
    """Run same analysis on raw binary specialist h_b (512-dim) for comparison."""
    return analyze_condition(h_b, counts, bits, transitions, 'Binary specialist (512-d)')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('=' * 90)
    print('DOES COMPOSITIONAL STRUCTURE SURVIVE UNIFICATION?')
    print('=' * 90)

    # Load canonical inputs
    print('\nLoading battery_final.npz...')
    h_a, h_b, counts, bits = load_battery()
    max_count = int(counts.max())
    transitions = build_transitions(max_count)
    print(f'  h_a: {h_a.shape}, h_b: {h_b.shape}, counts: 0-{max_count}')
    print(f'  Transitions: {len(transitions)} ({transitions[0]["from"]}→{transitions[0]["to"]} to '
          f'{transitions[-1]["from"]}→{transitions[-1]["to"]})')

    # Specialist baseline
    print('\n--- Binary Specialist (baseline) ---')
    spec_results = specialist_baseline(h_b, counts, bits, transitions)
    print(f'  mag-depth r={spec_results["mag_depth_corr"]:.4f}, '
          f'sign={spec_results["sign_agreement"]} ({spec_results["sign_pct"]:.1%}), '
          f'cross-talk={spec_results["cross_talk"]:.4f}, '
          f'PCA-90={spec_results["pca_90"]}')

    # Process each condition
    all_results = {'specialist': spec_results}
    for key, ckpt_path, label in CONDITIONS:
        print(f'\n--- {label} ---')
        print(f'  Loading {os.path.basename(os.path.dirname(ckpt_path))}/{os.path.basename(ckpt_path)}...')
        h_u = collect_h_u(ckpt_path, h_a, h_b)
        print(f'  h_u: {h_u.shape}')
        results = analyze_condition(h_u, counts, bits, transitions, label)
        all_results[key] = results
        print(f'  mag-depth r={results["mag_depth_corr"]:.4f}, '
              f'sign={results["sign_agreement"]} ({results["sign_pct"]:.1%}), '
              f'cross-talk={results["cross_talk"]:.4f}, '
              f'PCA-90={results["pca_90"]}')

    # ─── Summary comparison table ─────────────────────────────────────────

    print('\n' + '=' * 120)
    print('SUMMARY: COMPOSITIONAL STRUCTURE COMPARISON')
    print('=' * 120)
    header = f'{"Condition":<28} {"mag-depth r":<14} {"Sign agr.":<12} {"Cross-talk":<12} {"cos(d0,d0)":<12} {"cos(d0,d1)":<12} {"PCA-90":<8} {"Bit0 acc":<10} {"Bit3 acc":<10}'
    print(header)
    print('-' * 120)

    for key in ['specialist'] + [c[0] for c in CONDITIONS]:
        r = all_results[key]
        print(f'{r["label"]:<28} '
              f'{r["mag_depth_corr"]:>+10.4f}   '
              f'{r["sign_agreement"]:>8}   '
              f'{r["cross_talk"]:>10.4f}   '
              f'{r["within_depth0_cosine"]:>10.4f}   '
              f'{r["between_d0_d1_cosine"]:>10.4f}   '
              f'{r["pca_90"]:>5}   '
              f'{r["probe_accs"][0]:>8.3f}   '
              f'{r["probe_accs"][3]:>8.3f}')

    # ─── Determine outcome ────────────────────────────────────────────────

    print('\n' + '=' * 90)
    print('OUTCOME CLASSIFICATION')
    print('=' * 90)

    unifier_signs = [all_results[c[0]]['sign_pct'] for c in CONDITIONS]
    unifier_corrs = [all_results[c[0]]['mag_depth_corr'] for c in CONDITIONS]
    spec_corr = spec_results['mag_depth_corr']

    all_signs_perfect = all(s >= 0.99 for s in unifier_signs)
    contrastive_keys = ['lambda_0.1', 'lambda_0.005', 'lambda_0.05']
    non_contrastive_keys = ['no_align', 'vicreg_70k', 'onset_5000', 'onset_0']

    contrastive_corrs = [all_results[k]['mag_depth_corr'] for k in contrastive_keys]
    non_contrastive_corrs = [all_results[k]['mag_depth_corr'] for k in non_contrastive_keys]

    if all_signs_perfect:
        print('  SCENARIO A (MODIFIED): Factored bit structure SURVIVES everywhere')
        print(f'  Sign agreement: 100% across all 7 conditions (matches specialist)')
        print(f'  Cross-talk: {np.mean([all_results[c[0]]["cross_talk"] for c in CONDITIONS]):.4f} '
              f'(~{np.mean([all_results[c[0]]["cross_talk"] for c in CONDITIONS])/spec_results["cross_talk"]:.0f}x specialist)')
        print()

    if max(contrastive_corrs) < 0.2 and min(non_contrastive_corrs) > 0.7:
        print('  BUT: Contrastive loss ERASES carry-depth ↔ magnitude correlation')
        print(f'    Contrastive (λ>0):     r = {np.mean(contrastive_corrs):+.3f}  (specialist: {spec_corr:+.3f})')
        print(f'    Non-contrastive (λ=0): r = {np.mean(non_contrastive_corrs):+.3f}  (specialist: {spec_corr:+.3f})')
        print()
        print('  INTERPRETATION: The contrastive InfoNCE loss treats all count mismatches')
        print('  equally, normalizing representational distances regardless of carry depth.')
        print('  This erases the magnitude-depth relationship while perfectly preserving')
        print('  the factored bit directions. This is the accuracy-geometry dissociation')
        print('  in microcosm: all information is retained, but the geometric organization')
        print('  (which distances encode carry complexity) is reorganized by the loss.')

    # ─── Cross-talk comparison ────────────────────────────────────────────

    print('\n  Cross-talk comparison (lower = better separation):')
    print(f'    Specialist:    {spec_results["cross_talk"]:.4f}')
    for c in CONDITIONS:
        r = all_results[c[0]]
        ratio = r['cross_talk'] / spec_results['cross_talk'] if spec_results['cross_talk'] > 0 else float('inf')
        print(f'    {c[2]:<25} {r["cross_talk"]:.4f}  ({ratio:.1f}x specialist)')

    # ─── Figures ──────────────────────────────────────────────────────────

    plot_comparison_bars(all_results, transitions)
    plot_decomposition_heatmap(all_results, transitions)

    # ─── Save results ────────────────────────────────────────────────────

    # Remove large arrays for JSON serialization
    save_results = {}
    for key, r in all_results.items():
        save_r = {k: v for k, v in r.items() if k != 'cosine_matrix'}
        save_results[key] = save_r

    out_path = os.path.join(ARTIFACTS_DIR, 'unifier_successor_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'\n  Results saved to {out_path}')


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_comparison_bars(all_results, transitions):
    """Bar chart comparing key metrics across conditions."""
    keys = ['specialist'] + [c[0] for c in CONDITIONS]
    labels = [all_results[k]['label'] for k in keys]
    # Shorten labels for plot
    short_labels = ['Specialist\n(512-d)', 'λ=0.1', 'λ=0.005', 'λ=0.05', 'λ=0.0',
                    'VICReg\n70K', 'Late\nVICReg', 'Early\nVICReg']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Colors: specialist in green, unifiers in blues/oranges
    colors = ['#55A868'] + ['#4C72B0', '#DD8452', '#8172B2', '#C44E52',
                            '#64B5CD', '#CCB974', '#8C8C8C']

    # 1) Magnitude-depth correlation
    vals = [all_results[k]['mag_depth_corr'] for k in keys]
    axes[0].bar(range(len(vals)), vals, color=colors, edgecolor='white')
    axes[0].set_title('Magnitude-Depth\nCorrelation (r)', fontsize=11, fontweight='bold')
    axes[0].set_ylim(-0.2, 1.1)
    axes[0].axhline(0, color='gray', linewidth=0.5)

    # 2) Sign agreement
    vals = [all_results[k]['sign_pct'] * 100 for k in keys]
    axes[1].bar(range(len(vals)), vals, color=colors, edgecolor='white')
    axes[1].set_title('Sign Agreement\n(%)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 110)
    axes[1].axhline(100, color='gray', linewidth=0.5, linestyle='--')

    # 3) Cross-talk ratio
    vals = [all_results[k]['cross_talk'] for k in keys]
    axes[2].bar(range(len(vals)), vals, color=colors, edgecolor='white')
    axes[2].set_title('Cross-Talk Ratio\n(lower = better)', fontsize=11, fontweight='bold')

    # 4) PCA components for 90%
    vals = [all_results[k]['pca_90'] for k in keys]
    axes[3].bar(range(len(vals)), vals, color=colors, edgecolor='white')
    axes[3].set_title('PCA Components\nfor 90% Variance', fontsize=11, fontweight='bold')

    for ax in axes:
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Compositional Structure: Specialist vs Unifier Conditions',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'unifier_successor_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_decomposition_heatmap(all_results, transitions):
    """Per-bit decomposition heatmaps for specialist and key conditions."""
    # Select key conditions: specialist, λ=0.1, λ=0.005, onset_5000
    plot_keys = ['specialist', 'lambda_0.1', 'lambda_0.005', 'onset_5000']
    plot_labels = ['Binary Specialist (512-d)', 'Unifier λ=0.1', 'Unifier λ=0.005', 'Late VICReg (onset 5000)']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    for ax_idx, (key, title) in enumerate(zip(plot_keys, plot_labels)):
        r = all_results[key]
        decomp = np.array(r['decomposition'])  # (n_transitions, 4)

        # Normalize by row max for visualization
        row_max = np.abs(decomp).max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        normed = decomp / row_max

        ax = axes[ax_idx]
        im = ax.imshow(normed, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Transition labels
        t_labels = [f'{t["from"]}→{t["to"]}' for t in transitions]
        ax.set_yticks(range(len(transitions)))
        ax.set_yticklabels(t_labels, fontsize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Bit0\n(1s)', 'Bit1\n(2s)', 'Bit2\n(4s)', 'Bit3\n(8s)'], fontsize=8)
        ax.set_title(f'{title}\nsign={r["sign_agreement"]}, xtalk={r["cross_talk"]:.3f}',
                     fontsize=9, fontweight='bold')

        # Mark expected changes with boxes
        for i, t in enumerate(transitions):
            for b in range(4):
                bit_from = (t['from'] >> b) & 1
                bit_to = (t['to'] >> b) & 1
                if bit_from != bit_to:
                    rect = plt.Rectangle((b - 0.5, i - 0.5), 1, 1,
                                         linewidth=1.5, edgecolor='black',
                                         facecolor='none')
                    ax.add_patch(rect)

    fig.colorbar(im, ax=axes, shrink=0.6, label='Normalized projection')
    fig.suptitle('Per-Bit Decomposition: Does Factored Structure Survive?',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'unifier_successor_decomposition.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    main()
