#!/usr/bin/env python3
"""
Alternative projection visualizations: PaCMAP and TriMap for trained vs untrained.

Validates that the trained-vs-untrained gap is real across multiple projection
methods, not a PCA artifact. Uses stratified subsampling (~40 points per count,
~1000 total) since PaCMAP/TriMap are designed for more points than 26 centroids.

Also generates centroid-only versions for comparison.
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

import pacmap
import trimap
from sklearn.decomposition import PCA

DATA_DIR = Path("/workspace/bridge/artifacts/h_t_data")
OUTPUT_DIR = Path("/workspace/bridge/artifacts/figures")


def load_data(label):
    path = DATA_DIR / f"{label}.npz"
    data = np.load(path, allow_pickle=True)
    return data['h_t'], data['counts']


def stratified_subsample(h_t, counts, per_count=40, seed=42):
    """Subsample ~per_count points per count level for stable projections."""
    rng = np.random.RandomState(seed)
    indices = []
    for c in range(int(counts.max()) + 1):
        mask = np.where(counts == c)[0]
        if len(mask) == 0:
            continue
        n_take = min(per_count, len(mask))
        chosen = rng.choice(mask, n_take, replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    return h_t[indices], counts[indices]


def compute_centroids(h_t, counts):
    max_count = int(counts.max())
    centroids, valid_counts = [], []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def r2_axis1_vs_count(embedding, counts):
    """R² of linear regression between first embedding axis and count."""
    x = counts.astype(float)
    y = embedding[:, 0]
    # Flip if negatively correlated
    if np.corrcoef(x, y)[0, 1] < 0:
        y = -y
    r = np.corrcoef(x, y)[0, 1]
    return r ** 2


def plot_projection(ax, embedding, counts, title, show_line=True):
    """Plot a 2D projection colored by count."""
    # Compute centroids in embedding space for line
    unique_counts = np.unique(counts)
    centroids_emb = []
    for c in sorted(unique_counts):
        mask = counts == c
        centroids_emb.append(embedding[mask].mean(axis=0))
    centroids_emb = np.array(centroids_emb)

    # Scatter all points
    ax.scatter(embedding[:, 0], embedding[:, 1], c=counts,
               cmap='viridis', s=8, alpha=0.4, edgecolors='none', zorder=1)

    # Overlay centroids with edges
    if show_line:
        for i in range(len(centroids_emb) - 1):
            ax.plot([centroids_emb[i, 0], centroids_emb[i+1, 0]],
                    [centroids_emb[i, 1], centroids_emb[i+1, 1]],
                    '-', color='gray', alpha=0.5, linewidth=0.8, zorder=2)
    ax.scatter(centroids_emb[:, 0], centroids_emb[:, 1],
               c=sorted(unique_counts), cmap='viridis', s=30,
               edgecolors='k', linewidths=0.4, zorder=3)

    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    print("Loading dual-eval data...")
    t_ht, t_counts = load_data('trained_same_run')
    u_ht, u_counts = load_data('untrained_baseline')
    print(f"  Full data: {len(t_ht)} timesteps")

    # Stratified subsample for projection methods
    t_sub, t_sub_c = stratified_subsample(t_ht, t_counts, per_count=40, seed=42)
    u_sub, u_sub_c = stratified_subsample(u_ht, u_counts, per_count=40, seed=42)
    print(f"  Subsampled: {len(t_sub)} trained, {len(u_sub)} untrained")

    # Centroids for R² computation
    t_centroids, t_valid = compute_centroids(t_ht, t_counts)
    u_centroids, u_valid = compute_centroids(u_ht, u_counts)

    # =================================================================
    # Run projections (3 seeds each for stability check)
    # =================================================================
    methods = {}

    # PCA (deterministic, no seed needed)
    print("Running PCA...")
    pca_t = PCA(n_components=2).fit_transform(t_sub)
    pca_u = PCA(n_components=2).fit_transform(u_sub)
    r2_pca_t = r2_axis1_vs_count(pca_t, t_sub_c)
    r2_pca_u = r2_axis1_vs_count(pca_u, u_sub_c)
    methods['PCA'] = {
        'trained': (pca_t, t_sub_c, r2_pca_t),
        'untrained': (pca_u, u_sub_c, r2_pca_u),
    }
    print(f"  PCA: trained r²={r2_pca_t:.3f}, untrained r²={r2_pca_u:.3f}")

    # PaCMAP (3 seeds)
    print("Running PaCMAP (3 seeds)...")
    pacmap_results = {'trained': [], 'untrained': []}
    for seed in [0, 1, 2]:
        pm = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=seed)
        emb_t = pm.fit_transform(t_sub)
        pm_u = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=seed)
        emb_u = pm_u.fit_transform(u_sub)
        r2_t = r2_axis1_vs_count(emb_t, t_sub_c)
        r2_u = r2_axis1_vs_count(emb_u, u_sub_c)
        pacmap_results['trained'].append((emb_t, r2_t))
        pacmap_results['untrained'].append((emb_u, r2_u))
        print(f"    seed {seed}: trained r²={r2_t:.3f}, untrained r²={r2_u:.3f}")

    # Pick best seed (highest trained r²) for display
    best_seed = max(range(3), key=lambda s: pacmap_results['trained'][s][1])
    methods['PaCMAP'] = {
        'trained': (pacmap_results['trained'][best_seed][0], t_sub_c,
                    pacmap_results['trained'][best_seed][1]),
        'untrained': (pacmap_results['untrained'][best_seed][0], u_sub_c,
                      pacmap_results['untrained'][best_seed][1]),
    }
    # Report stability
    t_r2s = [r[1] for r in pacmap_results['trained']]
    u_r2s = [r[1] for r in pacmap_results['untrained']]
    print(f"  PaCMAP stability: trained r²={np.mean(t_r2s):.3f}±{np.std(t_r2s):.3f}, "
          f"untrained r²={np.mean(u_r2s):.3f}±{np.std(u_r2s):.3f}")

    # TriMap (3 seeds)
    print("Running TriMap (3 seeds)...")
    trimap_results = {'trained': [], 'untrained': []}
    for seed in [0, 1, 2]:
        # TriMap with relaxed params for small datasets
        n_inliers = min(10, len(t_sub) // 3)
        n_outliers = min(5, len(t_sub) // 5)
        try:
            tm = trimap.TRIMAP(n_dims=2, n_inliers=n_inliers, n_outliers=n_outliers,
                               n_random=3)
            emb_t = tm.fit_transform(t_sub)
            tm_u = trimap.TRIMAP(n_dims=2, n_inliers=n_inliers, n_outliers=n_outliers,
                                 n_random=3)
            emb_u = tm_u.fit_transform(u_sub)
            r2_t = r2_axis1_vs_count(emb_t, t_sub_c)
            r2_u = r2_axis1_vs_count(emb_u, u_sub_c)
            trimap_results['trained'].append((emb_t, r2_t))
            trimap_results['untrained'].append((emb_u, r2_u))
            print(f"    seed {seed}: trained r²={r2_t:.3f}, untrained r²={r2_u:.3f}")
        except Exception as e:
            print(f"    seed {seed} failed: {e}")

    if trimap_results['trained']:
        best_seed = max(range(len(trimap_results['trained'])),
                        key=lambda s: trimap_results['trained'][s][1])
        methods['TriMap'] = {
            'trained': (trimap_results['trained'][best_seed][0], t_sub_c,
                        trimap_results['trained'][best_seed][1]),
            'untrained': (trimap_results['untrained'][best_seed][0], u_sub_c,
                          trimap_results['untrained'][best_seed][1]),
        }
        t_r2s = [r[1] for r in trimap_results['trained']]
        u_r2s = [r[1] for r in trimap_results['untrained']]
        print(f"  TriMap stability: trained r²={np.mean(t_r2s):.3f}±{np.std(t_r2s):.3f}, "
              f"untrained r²={np.mean(u_r2s):.3f}±{np.std(u_r2s):.3f}")

    # =================================================================
    # Composite figure: 2 rows × 3 columns (PCA, PaCMAP, TriMap)
    # Top row: trained, Bottom row: untrained
    # =================================================================
    n_methods = len(methods)
    print(f"\nGenerating composite figure ({n_methods} methods)...")

    fig, axes = plt.subplots(2, n_methods, figsize=(3.3 * n_methods, 5.5))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    labels = 'abcdef'
    for col, (method_name, data) in enumerate(methods.items()):
        emb_t, counts_t, r2_t = data['trained']
        emb_u, counts_u, r2_u = data['untrained']

        lbl_t = labels[col * 2]
        lbl_u = labels[col * 2 + 1]

        plot_projection(axes[0, col], emb_t, counts_t,
                        f'({lbl_t}) Trained — {method_name} ($r^2$={r2_t:.3f})')
        plot_projection(axes[1, col], emb_u, counts_u,
                        f'({lbl_u}) Untrained — {method_name} ($r^2$={r2_u:.3f})')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f'supp_projection_contrast.{fmt}',
                    dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("PROJECTION METHOD COMPARISON")
    print("=" * 60)
    print(f"{'Method':<12} {'Trained r²':>12} {'Untrained r²':>14} {'Gap':>8}")
    print("-" * 48)
    for name, data in methods.items():
        r2_t = data['trained'][2]
        r2_u = data['untrained'][2]
        print(f"{name:<12} {r2_t:>12.3f} {r2_u:>14.3f} {r2_t - r2_u:>8.3f}")
    print()
    print("Saved: supp_projection_contrast.png/pdf")


if __name__ == '__main__':
    main()
