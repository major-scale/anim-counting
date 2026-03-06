#!/usr/bin/env python3
"""
Generate publication-quality figures for the counting manifold paper (v2).
Updated with multi-seed ablation data and trained-vs-untrained comparison.

Figures:
  fig1_hero: Environment + manifold + arc-length (3-panel)
  fig2_untrained: Trained vs Untrained PCA + RDMs (4-panel)
  fig3_ablation: Multi-seed ablation with error bars (2-panel)

Usage:
    python generate_figures_v2.py [--data_dir /path/to/h_t_data] [--output_dir /path/to/figures]
"""

import os, sys, argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    print("SciencePlots not available, using default style")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def load_h_t_data(data_dir, label):
    path = Path(data_dir) / f"{label}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        'h_t': data['h_t'],
        'counts': data['counts'],
        'episode_ids': data['episode_ids'],
        'timesteps': data['timesteps'],
    }


def compute_centroids(h_t, counts):
    max_count = int(counts.max())
    centroids, valid_counts = [], []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def compute_geodesic_distances(centroids, k=6):
    n = len(centroids)
    A = kneighbors_graph(centroids, n_neighbors=min(k, n-1), mode='distance')
    A = 0.5 * (A + A.T)
    return shortest_path(A, directed=False)


def compute_arc_length(centroids, geo):
    n = len(centroids)
    consec = [geo[i, i+1] for i in range(n-1)]
    return np.cumsum([0] + consec), consec


# =====================================================================
# Figure 1: Hero (kept from v1, uses seed 0)
# =====================================================================
def fig1_hero(data_dir, output_dir):
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP fig1: no grid_baseline_seed0 data")
        return

    h_t, counts = data['h_t'], data['counts']
    centroids, valid_counts = compute_centroids(h_t, counts)
    n = len(valid_counts)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2), gridspec_kw={'width_ratios': [1.0, 1.2, 1.0]})

    # Panel A: Environment schematic
    ax = axes[0]
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    np.random.seed(42)
    for _ in range(12):
        bx = np.random.uniform(1, 6)
        by = np.random.uniform(1.5, 8.5)
        ax.plot(bx, by, 'o', color='#4CAF50', markersize=4, alpha=0.7)
    for row in range(3):
        for col in range(3):
            gx = 9.5 + col * 1.2
            gy = 3 + row * 1.5
            ax.plot(gx, gy, 's', color='#2196F3', markersize=5, markeredgecolor='#1565C0', markeredgewidth=0.5)
    ax.plot(5.5, 5, '^', color='#FF9800', markersize=8, markeredgecolor='#E65100', markeredgewidth=0.5)
    ax.annotate('', xy=(8.5, 5), xytext=(6.5, 5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.text(7, 0.5, 'gather', ha='center', fontsize=7, color='#666', style='italic')
    ax.set_title('(a) Environment', fontsize=8, fontweight='bold')
    ax.axis('off')

    # Panel B: PCA manifold (switched from Isomap for consistency with untrained comparison)
    ax = axes[1]
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(centroids)
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=valid_counts,
                    cmap='viridis', s=25, edgecolors='k', linewidths=0.3, zorder=2)
    for i in range(n - 1):
        ax.plot([embedding[i, 0], embedding[i+1, 0]],
                [embedding[i, 1], embedding[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Count', fontsize=7)
    pv = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({pv[0]:.0%})', fontsize=7)
    ax.set_ylabel(f'PC2 ({pv[1]:.0%})', fontsize=7)
    ax.set_title('(b) Learned manifold (PCA)', fontsize=8, fontweight='bold')

    # Panel C: Arc-length vs count
    ax = axes[2]
    geo = compute_geodesic_distances(centroids)
    arc, _ = compute_arc_length(centroids, geo)
    ax.scatter(valid_counts, arc, c=valid_counts, cmap='viridis', s=20,
               edgecolors='k', linewidths=0.3, zorder=2)
    coeffs = np.polyfit(valid_counts, arc, 1)
    fit_line = np.polyval(coeffs, valid_counts)
    ss_res = np.sum((arc - fit_line)**2)
    ss_tot = np.sum((arc - np.mean(arc))**2)
    r2 = 1 - ss_res / ss_tot
    ax.plot(valid_counts, fit_line, 'k--', linewidth=1, alpha=0.7, label=f'$R^2$ = {r2:.3f}')
    ax.set_xlabel('Count', fontsize=7)
    ax.set_ylabel('Geodesic arc-length', fontsize=7)
    ax.set_title('(c) Linearity', fontsize=8, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig1_hero.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig1_hero")


# =====================================================================
# Figure 2: Trained vs Untrained (NEW)
# =====================================================================
def fig2_untrained(data_dir, output_dir):
    """Four-panel: PCA trained, PCA untrained, RDM trained, RDM untrained."""
    trained_data = load_h_t_data(data_dir, 'trained_same_run')
    untrained_data = load_h_t_data(data_dir, 'untrained_baseline')

    if trained_data is None or untrained_data is None:
        print("  SKIP fig2: missing trained_same_run or untrained_baseline")
        return

    # Compute centroids for both
    t_centroids, t_counts = compute_centroids(trained_data['h_t'], trained_data['counts'])
    u_centroids, u_counts = compute_centroids(untrained_data['h_t'], untrained_data['counts'])
    n_t, n_u = len(t_counts), len(u_counts)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))

    # --- Top row: PCA projections ---
    # Trained PCA
    ax = axes[0, 0]
    pca_t = PCA(n_components=2)
    emb_t = pca_t.fit_transform(t_centroids)
    sc = ax.scatter(emb_t[:, 0], emb_t[:, 1], c=t_counts,
                    cmap='viridis', s=30, edgecolors='k', linewidths=0.3, zorder=2)
    for i in range(n_t - 1):
        ax.plot([emb_t[i, 0], emb_t[i+1, 0]],
                [emb_t[i, 1], emb_t[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    pv_t = pca_t.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({pv_t[0]:.0%})', fontsize=7)
    ax.set_ylabel(f'PC2 ({pv_t[1]:.0%})', fontsize=7)
    ax.set_title('(a) Trained — PCA projection', fontsize=8, fontweight='bold')

    # Untrained PCA
    ax = axes[0, 1]
    pca_u = PCA(n_components=2)
    emb_u = pca_u.fit_transform(u_centroids)
    sc = ax.scatter(emb_u[:, 0], emb_u[:, 1], c=u_counts,
                    cmap='viridis', s=30, edgecolors='k', linewidths=0.3, zorder=2)
    for i in range(n_u - 1):
        ax.plot([emb_u[i, 0], emb_u[i+1, 0]],
                [emb_u[i, 1], emb_u[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    pv_u = pca_u.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({pv_u[0]:.0%})', fontsize=7)
    ax.set_ylabel(f'PC2 ({pv_u[1]:.0%})', fontsize=7)
    ax.set_title('(b) Untrained — PCA projection', fontsize=8, fontweight='bold')

    # --- Bottom row: RDMs ---
    # Trained RDM
    ax = axes[1, 0]
    rdm_t = squareform(pdist(t_centroids, metric='euclidean'))
    im = ax.imshow(rdm_t, cmap='viridis', origin='lower', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Distance', fontsize=6)
    tick_pos = [0, 5, 10, 15, 20, 25]
    tick_lab = [str(t_counts[i]) if i < len(t_counts) else '' for i in tick_pos]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=6)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab, fontsize=6)
    ax.set_xlabel('Count', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    # Compute RSA
    rdm_ideal = np.abs(t_counts[:, None] - t_counts[None, :]).astype(float)
    triu = np.triu_indices(n_t, k=1)
    rsa_t, _ = spearmanr(rdm_t[triu], rdm_ideal[triu])
    ax.set_title(f'(c) Trained RDM ($\\rho$ = {rsa_t:.3f})', fontsize=8, fontweight='bold')

    # Untrained RDM
    ax = axes[1, 1]
    rdm_u = squareform(pdist(u_centroids, metric='euclidean'))
    im = ax.imshow(rdm_u, cmap='viridis', origin='lower', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Distance', fontsize=6)
    tick_lab_u = [str(u_counts[i]) if i < len(u_counts) else '' for i in tick_pos]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab_u, fontsize=6)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab_u, fontsize=6)
    ax.set_xlabel('Count', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    rdm_ideal_u = np.abs(u_counts[:, None] - u_counts[None, :]).astype(float)
    triu_u = np.triu_indices(n_u, k=1)
    rsa_u, _ = spearmanr(rdm_u[triu_u], rdm_ideal_u[triu_u])
    ax.set_title(f'(d) Untrained RDM ($\\rho$ = {rsa_u:.3f})', fontsize=8, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig2_untrained.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated fig2_untrained (RSA trained={rsa_t:.3f}, untrained={rsa_u:.3f})")


# =====================================================================
# Figure 3: Multi-seed Ablation (UPDATED)
# =====================================================================
def fig3_ablation(data_dir, output_dir):
    """Two-panel: information starvation with error bars + arrangement comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [5, 4]})

    # Panel A: Information starvation (multi-seed with error bars)
    conditions = [
        {'name': 'Grid\nbaseline', 'mean': 0.329, 'std': 0.027, 'n': 6, 'color': '#2ca02c'},
        {'name': 'No-count', 'mean': 0.336, 'std': 0.091, 'n': 3, 'color': '#8c564b'},
        {'name': 'No-slots\nno-count', 'mean': 0.344, 'std': 0.045, 'n': 3, 'color': '#ff7f0e'},
        {'name': 'Shuffle\nno-slots\nno-count', 'mean': 0.367, 'std': 0.081, 'n': 3, 'color': '#e377c2'},
        {'name': 'Untrained', 'mean': 0.395, 'std': 0, 'n': 1, 'color': '#999999'},
    ]

    x = np.arange(len(conditions))
    means = [c['mean'] for c in conditions]
    stds = [c['std'] for c in conditions]
    colors = [c['color'] for c in conditions]
    names = [c['name'] for c in conditions]

    bars = ax1.bar(x, means, yerr=stds, color=colors, edgecolor='k', linewidth=0.5,
                   width=0.6, capsize=3, error_kw={'linewidth': 1})
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=5.5)
    ax1.set_ylabel('GHE (mean $\\pm$ std)', fontsize=8)
    ax1.set_title('(a) Information starvation', fontsize=8, fontweight='bold')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Threshold')
    ax1.set_ylim(0, 0.6)

    # Annotate with n= and mean
    for i, c in enumerate(conditions):
        label = f'{c["mean"]:.3f}'
        if c['n'] > 1:
            label += f'\n(n={c["n"]})'
        ax1.text(i, c['mean'] + c['std'] + 0.015, label,
                ha='center', va='bottom', fontsize=5.5, fontweight='bold')

    # Bracket showing "all valid"
    ax1.annotate('', xy=(0, 0.52), xytext=(3, 0.52),
                arrowprops=dict(arrowstyle='|-|', color='green', lw=1.5))
    ax1.text(1.5, 0.535, 'all valid (< 0.5)', ha='center', fontsize=6, color='green', fontweight='bold')

    ax1.legend(fontsize=6, loc='upper left')

    # Panel B: Arrangement comparison (single-seed)
    arrangements = [
        {'name': 'Line', 'ghe': 0.288, 'color': '#1f77b4'},
        {'name': 'Grid', 'ghe': 0.329, 'color': '#2ca02c'},
        {'name': 'Scatter', 'ghe': 0.334, 'color': '#d62728'},
        {'name': 'Circle', 'ghe': 0.394, 'color': '#9467bd'},
    ]

    x2 = np.arange(len(arrangements))
    ghes = [a['ghe'] for a in arrangements]
    colors2 = [a['color'] for a in arrangements]
    names2 = [a['name'] for a in arrangements]

    bars2 = ax2.bar(x2, ghes, color=colors2, edgecolor='k', linewidth=0.5, width=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names2, fontsize=7)
    ax2.set_ylabel('GHE', fontsize=8)
    ax2.set_title('(b) Spatial arrangement', fontsize=8, fontweight='bold')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0, 0.6)
    for bar, ghe in zip(bars2, ghes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ghe:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Use baseline mean for Grid bar
    ax2.text(bars2[1].get_x() + bars2[1].get_width()/2, -0.035,
            '(6 seeds)', ha='center', fontsize=5, color='gray')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig3_ablation.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig3_ablation")


# =====================================================================
# Supplementary: Topology (persistence barcodes)
# =====================================================================
def supp_topology(data_dir, output_dir):
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP supp_topology: no data")
        return
    try:
        from ripser import ripser
    except ImportError:
        print("  SKIP supp_topology: ripser not installed")
        return

    centroids, _ = compute_centroids(data['h_t'], data['counts'])
    result = ripser(centroids, maxdim=1)
    h0, h1 = result['dgms'][0], result['dgms'][1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    h0_sorted = sorted(h0, key=lambda x: x[1] - x[0], reverse=True)
    for i, (birth, death) in enumerate(h0_sorted):
        if death == np.inf:
            death = max(h0[h0[:, 1] != np.inf][:, 1].max() * 1.3, 1)
            ax1.barh(i, death - birth, left=birth, height=0.6, color='#2196F3', edgecolor='k', linewidth=0.3)
            ax1.plot(death, i, '>', color='#2196F3', markersize=4)
        else:
            ax1.barh(i, death - birth, left=birth, height=0.6, color='#2196F3', edgecolor='k', linewidth=0.3)
    ax1.set_xlabel('Filtration value', fontsize=7)
    ax1.set_ylabel('Feature', fontsize=7)
    ax1.set_title(r'$H_0$ (components): $\beta_0=1$', fontsize=8, fontweight='bold')
    ax1.invert_yaxis()

    if len(h1) > 0:
        h1_sorted = sorted(h1, key=lambda x: x[1] - x[0], reverse=True)
        for i, (birth, death) in enumerate(h1_sorted):
            d = death if death != np.inf else max(h1[h1[:, 1] != np.inf][:, 1].max() * 1.3, 1)
            ax2.barh(i, d - birth, left=birth, height=0.6, color='#F44336', edgecolor='k', linewidth=0.3)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No persistent features\n(no loops)', transform=ax2.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
    ax2.set_xlabel('Filtration value', fontsize=7)
    ax2.set_ylabel('Feature', fontsize=7)
    ax2.set_title(r'$H_1$ (loops): $\beta_1=0$', fontsize=8, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'supp_topology.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated supp_topology")


# =====================================================================
# Supplementary: Curvature profile
# =====================================================================
def supp_curvature(data_dir, output_dir):
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP supp_curvature: no data")
        return

    centroids, valid_counts = compute_centroids(data['h_t'], data['counts'])
    n = len(valid_counts)

    kappas, kappa_counts = [], []
    for i in range(1, n - 1):
        p1, p2, p3 = centroids[i-1], centroids[i], centroids[i+1]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        if a > 1e-10 and b > 1e-10 and c > 1e-10:
            cos_a = np.clip(np.dot(p2-p1, p3-p1) / (np.linalg.norm(p2-p1) * np.linalg.norm(p3-p1)), -1, 1)
            sin_a = np.sqrt(1 - cos_a**2)
            area = 0.5 * np.linalg.norm(p2-p1) * np.linalg.norm(p3-p1) * sin_a
            kappas.append(2 * area / (a * b * c))
        else:
            kappas.append(0)
        kappa_counts.append(valid_counts[i])

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax.plot(kappa_counts, kappas, 'o-', color='#2ca02c', markersize=3, linewidth=1,
            markeredgecolor='k', markeredgewidth=0.3)
    ax.set_xlabel('Count', fontsize=8)
    ax.set_ylabel('Menger curvature $\\kappa$', fontsize=8)
    ax.set_title('Curvature profile (grid baseline)', fontsize=8, fontweight='bold')
    ax.axhline(y=np.mean(kappas), color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(1, np.mean(kappas) * 1.05, f'mean={np.mean(kappas):.3f}', fontsize=6, color='gray')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'supp_curvature.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated supp_curvature")


# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/workspace/bridge/artifacts/h_t_data")
    parser.add_argument("--output_dir", default="/workspace/bridge/artifacts/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating figures (v2)...")
    print()

    # Main figures
    fig1_hero(args.data_dir, args.output_dir)
    fig2_untrained(args.data_dir, args.output_dir)
    fig3_ablation(args.data_dir, args.output_dir)

    # Supplementary
    supp_topology(args.data_dir, args.output_dir)
    supp_curvature(args.data_dir, args.output_dir)

    print()
    print("Done! Figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
