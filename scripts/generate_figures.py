#!/usr/bin/env python3
"""
Generate publication-quality figures for the counting manifold paper.
Figures use SciencePlots styling, viridis colormap, and PDF output.

Usage:
    python generate_figures.py [--data_dir /path/to/h_t_data] [--output_dir /path/to/figures]
"""

import os, sys, argparse, json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try SciencePlots
try:
    plt.style.use(['science', 'no-latex'])
except Exception:
    print("SciencePlots not available, using default style")

plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
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

# Consistent condition colors (tab10)
CONDITION_COLORS = {
    'Line': '#1f77b4',
    'No-slots+no-count': '#ff7f0e',
    'Grid (baseline)': '#2ca02c',
    'Scatter': '#d62728',
    'Circle': '#9467bd',
    'No-count': '#8c564b',
    'Shuffled': '#e377c2',
}

# Full results table (metric-only, no h_t needed)
RESULTS = [
    {'name': 'Line',              'ghe': 0.288, 'r2': 0.998, 'b0': 1, 'b1': 0, 'rsa': 0.982, 'steps': '100K'},
    {'name': 'No-slots+no-count', 'ghe': 0.303, 'r2': 0.997, 'b0': 1, 'b1': 0, 'rsa': 0.981, 'steps': '200K'},
    {'name': 'Grid (baseline)',   'ghe': 0.327, 'r2': 0.998, 'b0': 1, 'b1': 0, 'rsa': 0.982, 'steps': '300K'},
    {'name': 'Scatter',           'ghe': 0.334, 'r2': 0.996, 'b0': 1, 'b1': 0, 'rsa': 0.980, 'steps': '100K'},
    {'name': 'Circle',            'ghe': 0.394, 'r2': 0.996, 'b0': 1, 'b1': 0, 'rsa': 0.978, 'steps': '100K'},
    {'name': 'No-count',          'ghe': 0.440, 'r2': 0.998, 'b0': 1, 'b1': 0, 'rsa': 0.981, 'steps': '100K'},
]


def load_h_t_data(data_dir, label):
    """Load h_t data from .npz file."""
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
    """Compute count-averaged centroids."""
    max_count = int(counts.max())
    centroids = []
    valid_counts = []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def compute_geodesic_distances(centroids, k=6):
    """Compute geodesic distance matrix."""
    n = len(centroids)
    A = kneighbors_graph(centroids, n_neighbors=min(k, n-1), mode='distance')
    A = 0.5 * (A + A.T)
    geo = shortest_path(A, directed=False)
    return geo


def compute_arc_length(centroids, geo):
    """Compute cumulative arc-length along the manifold."""
    n = len(centroids)
    consec = [geo[i, i+1] for i in range(n-1)]
    arc = np.cumsum([0] + consec)
    return arc, consec


def menger_curvature(p1, p2, p3):
    """Compute Menger curvature from three points in R^n."""
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0
    cos_angle = np.clip(np.dot(p2 - p1, p3 - p1) / (np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1)), -1, 1)
    sin_angle = np.sqrt(1 - cos_angle**2)
    area = 0.5 * np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1) * sin_angle
    return 2 * area / (a * b * c)


# =====================================================================
# Figure 1: Hero Figure
# =====================================================================
def fig1_hero(data_dir, output_dir):
    """Three-panel hero figure: Setup → Manifold → Arc-length."""
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
    # Field zone (left)
    np.random.seed(42)
    for _ in range(12):
        bx = np.random.uniform(1, 6)
        by = np.random.uniform(1.5, 8.5)
        ax.plot(bx, by, 'o', color='#4CAF50', markersize=4, alpha=0.7)
    # Target grid (right)
    for row in range(3):
        for col in range(3):
            gx = 9.5 + col * 1.2
            gy = 3 + row * 1.5
            ax.plot(gx, gy, 's', color='#2196F3', markersize=5, markeredgecolor='#1565C0', markeredgewidth=0.5)
    # Bot
    ax.plot(5.5, 5, '^', color='#FF9800', markersize=8, markeredgecolor='#E65100', markeredgewidth=0.5)
    # Arrow
    ax.annotate('', xy=(8.5, 5), xytext=(6.5, 5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    ax.text(7, 0.5, 'gather', ha='center', fontsize=7, color='#666', style='italic')
    ax.set_title('(a) Environment', fontsize=8, fontweight='bold')
    ax.axis('off')

    # Panel B: Isomap manifold projection
    ax = axes[1]
    iso = Isomap(n_neighbors=min(12, n-1), n_components=2)
    embedding = iso.fit_transform(centroids)
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=valid_counts,
                    cmap='viridis', s=25, edgecolors='k', linewidths=0.3, zorder=2)
    # Connect consecutive points
    for i in range(n - 1):
        ax.plot([embedding[i, 0], embedding[i+1, 0]],
                [embedding[i, 1], embedding[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Count', fontsize=7)
    ax.set_xlabel('Isomap dim 1', fontsize=7)
    ax.set_ylabel('Isomap dim 2', fontsize=7)
    ax.set_title('(b) Learned manifold', fontsize=8, fontweight='bold')

    # Panel C: Arc-length vs count
    ax = axes[2]
    geo = compute_geodesic_distances(centroids)
    arc, _ = compute_arc_length(centroids, geo)
    ax.scatter(valid_counts, arc, c=valid_counts, cmap='viridis', s=20,
               edgecolors='k', linewidths=0.3, zorder=2)
    # Linear fit
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
# Figure 3: Ablation Cascade (bar chart + available manifold panels)
# =====================================================================
def fig3_ablation_cascade(data_dir, output_dir):
    """Ablation cascade: bar chart of GHE across conditions."""

    # Row 1: Ablation ladder
    ablation_row = [
        {'name': 'Grid\n(baseline)', 'ghe': 0.327, 'color': CONDITION_COLORS['Grid (baseline)']},
        {'name': 'No-count', 'ghe': 0.440, 'color': CONDITION_COLORS['No-count']},
        {'name': 'No-slots\nno-count', 'ghe': 0.303, 'color': CONDITION_COLORS['No-slots+no-count']},
    ]
    # Row 2: Arrangement comparison
    arrangement_row = [
        {'name': 'Line', 'ghe': 0.288, 'color': CONDITION_COLORS['Line']},
        {'name': 'Grid', 'ghe': 0.327, 'color': CONDITION_COLORS['Grid (baseline)']},
        {'name': 'Scatter', 'ghe': 0.334, 'color': CONDITION_COLORS['Scatter']},
        {'name': 'Circle', 'ghe': 0.394, 'color': CONDITION_COLORS['Circle']},
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={'width_ratios': [3, 4]})

    # Ablation ladder bar chart
    names = [r['name'] for r in ablation_row]
    ghes = [r['ghe'] for r in ablation_row]
    colors = [r['color'] for r in ablation_row]
    bars = ax1.bar(range(len(names)), ghes, color=colors, edgecolor='k', linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=6)
    ax1.set_ylabel('GHE', fontsize=8)
    ax1.set_title('Ablation ladder', fontsize=8, fontweight='bold')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='threshold')
    ax1.set_ylim(0, 0.6)
    # Annotate values
    for bar, ghe in zip(bars, ghes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ghe:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    # Star the surprising result
    ax1.annotate('*', xy=(2, 0.303), fontsize=14, color='red', ha='center',
                fontweight='bold', xytext=(2, 0.34),
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    # Arrangement comparison bar chart
    names = [r['name'] for r in arrangement_row]
    ghes = [r['ghe'] for r in arrangement_row]
    colors = [r['color'] for r in arrangement_row]
    bars = ax2.bar(range(len(names)), ghes, color=colors, edgecolor='k', linewidth=0.5, width=0.6)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=7)
    ax2.set_ylabel('GHE', fontsize=8)
    ax2.set_title('Arrangement comparison', fontsize=8, fontweight='bold')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0, 0.6)
    for bar, ghe in zip(bars, ghes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ghe:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig3_ablation_cascade.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig3_ablation_cascade")


# =====================================================================
# Figure 4: Curvature Profile
# =====================================================================
def fig4_curvature(data_dir, output_dir):
    """Menger curvature profile along the manifold."""
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP fig4: no grid_baseline_seed0 data")
        return

    h_t, counts = data['h_t'], data['counts']
    centroids, valid_counts = compute_centroids(h_t, counts)
    n = len(valid_counts)

    # Compute Menger curvature
    kappas = []
    kappa_counts = []
    for i in range(1, n - 1):
        k = menger_curvature(centroids[i-1], centroids[i], centroids[i+1])
        kappas.append(k)
        kappa_counts.append(valid_counts[i])

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax.plot(kappa_counts, kappas, 'o-', color=CONDITION_COLORS['Grid (baseline)'],
            markersize=3, linewidth=1, markeredgecolor='k', markeredgewidth=0.3)
    ax.set_xlabel('Count', fontsize=8)
    ax.set_ylabel('Menger curvature $\\kappa$', fontsize=8)
    ax.set_title('Curvature profile (grid baseline)', fontsize=8, fontweight='bold')
    ax.axhline(y=np.mean(kappas), color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(1, np.mean(kappas) * 1.05, f'mean={np.mean(kappas):.3f}', fontsize=6, color='gray')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig4_curvature.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig4_curvature")


# =====================================================================
# Figure 5: Topology — Persistence Barcodes
# =====================================================================
def fig5_topology(data_dir, output_dir):
    """Persistent homology barcodes."""
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP fig5: no grid_baseline_seed0 data")
        return

    try:
        from ripser import ripser
    except ImportError:
        print("  SKIP fig5: ripser not installed")
        return

    h_t, counts = data['h_t'], data['counts']
    centroids, valid_counts = compute_centroids(h_t, counts)

    result = ripser(centroids, maxdim=1)
    h0 = result['dgms'][0]
    h1 = result['dgms'][1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # H0 barcode
    h0_sorted = sorted(h0, key=lambda x: x[1] - x[0], reverse=True)
    for i, (birth, death) in enumerate(h0_sorted):
        if death == np.inf:
            death = max(h0[h0[:, 1] != np.inf][:, 1].max() * 1.3 if len(h0[h0[:, 1] != np.inf]) > 0 else 10, 1)
            ax1.barh(i, death - birth, left=birth, height=0.6, color='#2196F3', edgecolor='k', linewidth=0.3)
            ax1.plot(death, i, '>', color='#2196F3', markersize=4)
        else:
            ax1.barh(i, death - birth, left=birth, height=0.6, color='#2196F3', edgecolor='k', linewidth=0.3)
    ax1.set_xlabel('Filtration value', fontsize=7)
    ax1.set_ylabel('Feature', fontsize=7)
    ax1.set_title(f'$H_0$ (components): $\\beta_0=1$', fontsize=8, fontweight='bold')
    ax1.invert_yaxis()

    # H1 barcode
    if len(h1) > 0:
        h1_sorted = sorted(h1, key=lambda x: x[1] - x[0], reverse=True)
        for i, (birth, death) in enumerate(h1_sorted):
            if death == np.inf:
                death = max(h1[h1[:, 1] != np.inf][:, 1].max() * 1.3 if len(h1[h1[:, 1] != np.inf]) > 0 else 10, 1)
            ax2.barh(i, death - birth, left=birth, height=0.6, color='#F44336', edgecolor='k', linewidth=0.3)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No persistent features\n(no loops)', transform=ax2.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
    ax2.set_xlabel('Filtration value', fontsize=7)
    ax2.set_ylabel('Feature', fontsize=7)
    ax2.set_title(f'$H_1$ (loops): $\\beta_1=0$', fontsize=8, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig5_topology.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig5_topology")


# =====================================================================
# Supplementary: Arc-length R² for all conditions (metric-only version)
# =====================================================================
def supp3_arclength_summary(data_dir, output_dir):
    """Summary bar chart of arc-length R² across all conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

    names = [r['name'] for r in RESULTS]
    r2s = [r['r2'] for r in RESULTS]
    colors = [CONDITION_COLORS.get(r['name'], '#999') for r in RESULTS]

    bars = ax.bar(range(len(names)), r2s, color=colors, edgecolor='k', linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6, rotation=15, ha='right')
    ax.set_ylabel('Arc-length $R^2$', fontsize=8)
    ax.set_title('Geodesic linearity across conditions', fontsize=8, fontweight='bold')
    ax.set_ylim(0.99, 1.001)
    for bar, r2 in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=5.5, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'supp3_arclength_summary.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated supp3_arclength_summary")


# =====================================================================
# RSA heatmap (from baseline h_t data)
# =====================================================================
def fig6_rsm(data_dir, output_dir):
    """Representational Similarity Matrix for baseline."""
    data = load_h_t_data(data_dir, 'grid_baseline_seed0')
    if data is None:
        print("  SKIP fig6: no grid_baseline_seed0 data")
        return

    h_t, counts = data['h_t'], data['counts']
    centroids, valid_counts = compute_centroids(h_t, counts)

    rdm = squareform(pdist(centroids, metric='euclidean'))

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))
    im = ax.imshow(rdm, cmap='viridis', origin='lower', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Euclidean distance', fontsize=7)
    ax.set_xlabel('Count', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title('Representational Dissimilarity Matrix', fontsize=8, fontweight='bold')
    # Set ticks at intervals
    tick_positions = [0, 5, 10, 15, 20, 25]
    tick_labels = [str(valid_counts[i]) if i < len(valid_counts) else '' for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=6)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(Path(output_dir) / f'fig6_rsm.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Generated fig6_rsm")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/workspace/bridge/artifacts/h_t_data")
    parser.add_argument("--output_dir", default="/workspace/bridge/artifacts/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    print()

    # Figures that need h_t data
    fig1_hero(args.data_dir, args.output_dir)
    fig4_curvature(args.data_dir, args.output_dir)
    fig5_topology(args.data_dir, args.output_dir)
    fig6_rsm(args.data_dir, args.output_dir)

    # Figures that only need metric numbers
    fig3_ablation_cascade(args.data_dir, args.output_dir)
    supp3_arclength_summary(args.data_dir, args.output_dir)

    print()
    print("Done! Figures saved to:", args.output_dir)
    print()

    # List what's missing
    missing = []
    for label in ['line_seed0', 'scatter_seed0', 'circle_seed0', 'nocount_seed0', 'noslots_seed0']:
        if not (Path(args.data_dir) / f"{label}.npz").exists():
            missing.append(label)
    if missing:
        print("Missing h_t data for arrangement invariance panels:")
        for m in missing:
            print(f"  - {m}")
        print("Re-run those conditions and extract h_t to enable Figures 2 and full Figure 3.")


if __name__ == "__main__":
    main()
