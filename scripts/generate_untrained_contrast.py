#!/usr/bin/env python3
"""
Generate supplementary trained-vs-untrained contrast plots.

Three paired visualizations that make the trained/untrained gap obvious:
  Plot 1: PC1 vs. Count (smooth curve vs scattered cloud)
  Plot 2: Step size — consecutive Euclidean distance (uniform vs wild)
  Plot 3: Nearest-neighbor accuracy (per-count heatmap + summary bars)

Uses the same dual-eval data as fig2_untrained (117K timesteps).
"""

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

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


DATA_DIR = Path("/workspace/bridge/artifacts/h_t_data")
OUTPUT_DIR = Path("/workspace/bridge/artifacts/figures")


def load_data(label):
    path = DATA_DIR / f"{label}.npz"
    data = np.load(path, allow_pickle=True)
    return data['h_t'], data['counts']


def compute_centroids(h_t, counts):
    max_count = int(counts.max())
    centroids, valid_counts = [], []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def main():
    print("Loading dual-eval data...")
    t_ht, t_counts = load_data('trained_same_run')
    u_ht, u_counts = load_data('untrained_baseline')

    t_centroids, t_valid = compute_centroids(t_ht, t_counts)
    u_centroids, u_valid = compute_centroids(u_ht, u_counts)

    n = len(t_valid)  # should be 26 (counts 0-25)
    print(f"  {n} count levels, {len(t_ht)} timesteps")

    # ================================================================
    # Plot 1: PC1 vs Count
    # ================================================================
    print("Generating plot 1: PC1 vs Count...")

    # Fit PCA on each independently (same as fig2)
    pca_t = PCA(n_components=2).fit(t_centroids)
    pca_u = PCA(n_components=2).fit(u_centroids)
    t_pc1 = pca_t.transform(t_centroids)[:, 0]
    u_pc1 = pca_u.transform(u_centroids)[:, 0]

    # Ensure PC1 is positively correlated with count (flip if needed)
    if np.corrcoef(t_valid, t_pc1)[0, 1] < 0:
        t_pc1 = -t_pc1
    if np.corrcoef(u_valid, u_pc1)[0, 1] < 0:
        u_pc1 = -u_pc1

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax = axes[0]
    ax.scatter(t_valid, t_pc1, c=t_valid, cmap='viridis', s=30,
               edgecolors='k', linewidths=0.3, zorder=2)
    ax.plot(t_valid, t_pc1, '-', color='gray', alpha=0.4, linewidth=1, zorder=1)
    r2_t = np.corrcoef(t_valid, t_pc1)[0, 1] ** 2
    ax.set_xlabel('Count')
    ax.set_ylabel(f'PC1 ({pca_t.explained_variance_ratio_[0]:.0%} var)')
    ax.set_title(f'(a) Trained — PC1 vs count ($r^2$={r2_t:.3f})',
                 fontsize=8, fontweight='bold')

    ax = axes[1]
    ax.scatter(u_valid, u_pc1, c=u_valid, cmap='viridis', s=30,
               edgecolors='k', linewidths=0.3, zorder=2)
    ax.plot(u_valid, u_pc1, '-', color='gray', alpha=0.4, linewidth=1, zorder=1)
    r2_u = np.corrcoef(u_valid, u_pc1)[0, 1] ** 2
    ax.set_xlabel('Count')
    ax.set_ylabel(f'PC1 ({pca_u.explained_variance_ratio_[0]:.0%} var)')
    ax.set_title(f'(b) Untrained — PC1 vs count ($r^2$={r2_u:.3f})',
                 fontsize=8, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'supp_pc1_vs_count.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved supp_pc1_vs_count (trained r²={r2_t:.3f}, untrained r²={r2_u:.3f})")

    # ================================================================
    # Plot 2: Step Size (consecutive Euclidean distance)
    # ================================================================
    print("Generating plot 2: Step size...")

    t_steps = np.linalg.norm(np.diff(t_centroids, axis=0), axis=1)
    u_steps = np.linalg.norm(np.diff(u_centroids, axis=0), axis=1)
    transitions = [f'{i}\u2192{i+1}' for i in range(n - 1)]
    x = np.arange(n - 1)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    ax = axes[0]
    ax.bar(x, t_steps, color='#2196F3', edgecolor='#1565C0', linewidth=0.3, width=0.8)
    mean_t = t_steps.mean()
    ax.axhline(mean_t, color='k', linestyle='--', linewidth=0.8, alpha=0.6,
               label=f'mean = {mean_t:.2f}')
    cv_t = t_steps.std() / mean_t
    ax.set_xlabel('Count transition')
    ax.set_ylabel('Euclidean distance')
    ax.set_title(f'(a) Trained — step size (CV={cv_t:.2f})',
                 fontsize=8, fontweight='bold')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([transitions[i] for i in range(0, n-1, 5)], fontsize=6)
    ax.legend(fontsize=6, loc='upper right')

    ax = axes[1]
    ax.bar(x, u_steps, color='#FF9800', edgecolor='#E65100', linewidth=0.3, width=0.8)
    mean_u = u_steps.mean()
    ax.axhline(mean_u, color='k', linestyle='--', linewidth=0.8, alpha=0.6,
               label=f'mean = {mean_u:.2f}')
    cv_u = u_steps.std() / mean_u
    ax.set_xlabel('Count transition')
    ax.set_ylabel('Euclidean distance')
    ax.set_title(f'(b) Untrained — step size (CV={cv_u:.2f})',
                 fontsize=8, fontweight='bold')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([transitions[i] for i in range(0, n-1, 5)], fontsize=6)
    ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'supp_step_size.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved supp_step_size (trained CV={cv_t:.2f}, untrained CV={cv_u:.2f})")

    # ================================================================
    # Plot 3: Nearest-Neighbor Accuracy
    # ================================================================
    print("Generating plot 3: Nearest-neighbor accuracy...")

    # For each count, find nearest centroid (by Euclidean distance, excluding self)
    # Check if it's an adjacent count (±1)
    t_dist = cdist(t_centroids, t_centroids, 'euclidean')
    u_dist = cdist(u_centroids, u_centroids, 'euclidean')

    def nn_accuracy_per_count(dist_mat, valid_counts):
        n = len(valid_counts)
        correct = np.zeros(n, dtype=bool)
        nn_idx = np.zeros(n, dtype=int)
        for i in range(n):
            dists = dist_mat[i].copy()
            dists[i] = np.inf  # exclude self
            nearest = np.argmin(dists)
            nn_idx[i] = nearest
            # Check if nearest is adjacent count
            if abs(valid_counts[nearest] - valid_counts[i]) <= 1:
                correct[i] = True
        return correct, nn_idx

    t_correct, t_nn = nn_accuracy_per_count(t_dist, t_valid)
    u_correct, u_nn = nn_accuracy_per_count(u_dist, u_valid)

    t_acc = t_correct.mean()
    u_acc = u_correct.mean()

    # Combined figure: per-count heatmap + summary bars
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5),
                              gridspec_kw={'width_ratios': [2, 2, 1.2]})

    # (a) Trained per-count heatmap
    ax = axes[0]
    colors_t = ['#4CAF50' if c else '#F44336' for c in t_correct]
    ax.bar(t_valid, np.ones(n), color=colors_t, edgecolor='white', linewidth=0.5, width=0.9)
    # Show NN label on wrong ones
    for i in range(n):
        if not t_correct[i]:
            ax.text(t_valid[i], 0.5, f'{t_valid[t_nn[i]]}', ha='center', va='center',
                    fontsize=5, color='white', fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_yticks([])
    ax.set_title(f'(a) Trained NN (accuracy={t_acc:.0%})',
                 fontsize=8, fontweight='bold')
    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#4CAF50'), Patch(color='#F44336')],
              ['Adjacent NN', 'Non-adjacent NN'], fontsize=6, loc='lower right')

    # (b) Untrained per-count heatmap
    ax = axes[1]
    colors_u = ['#4CAF50' if c else '#F44336' for c in u_correct]
    ax.bar(u_valid, np.ones(n), color=colors_u, edgecolor='white', linewidth=0.5, width=0.9)
    for i in range(n):
        if not u_correct[i]:
            ax.text(u_valid[i], 0.5, f'{u_valid[u_nn[i]]}', ha='center', va='center',
                    fontsize=5, color='white', fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_yticks([])
    ax.set_title(f'(b) Untrained NN (accuracy={u_acc:.0%})',
                 fontsize=8, fontweight='bold')

    # (c) Summary bar chart
    ax = axes[2]
    bars = ax.bar(['Trained', 'Untrained'], [t_acc * 100, u_acc * 100],
                  color=['#2196F3', '#FF9800'], edgecolor=['#1565C0', '#E65100'],
                  linewidth=0.5, width=0.6)
    ax.set_ylabel('NN accuracy (%)')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, [t_acc, u_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{val:.0%}', ha='center', fontsize=7, fontweight='bold')
    ax.set_title('(c) Summary', fontsize=8, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'supp_nn_accuracy.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved supp_nn_accuracy (trained={t_acc:.0%}, untrained={u_acc:.0%})")

    print("\nDone!")


if __name__ == '__main__':
    main()
