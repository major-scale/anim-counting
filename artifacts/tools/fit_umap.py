"""
Fit UMAP — Phase 2
====================
Loads trajectories.npz, fits 3 UMAP configs + PCA,
generates static brain map PNGs and cluster analysis.

Output: data/umap_embeddings.npz + figures/*.png
"""

import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import umap

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
FIG_DIR = _SCRIPT_DIR / "figures"

def _make_count_colors(n):
    """Generate n distinct colors."""
    if n <= 10:
        cmap = plt.cm.tab10
        return [cmap(i / 10) for i in range(n)]
    else:
        cmap = plt.cm.turbo
        return [cmap(i / max(n - 1, 1)) for i in range(n)]


def load_data():
    d = np.load(DATA_DIR / "trajectories.npz")
    print(f"Loaded {len(d['true_count'])} timesteps from {len(np.unique(d['episode_id']))} episodes")
    return d


def compute_cluster_centers(embedding, counts):
    """Compute mean UMAP position per count."""
    centers = {}
    for c in range(int(counts.max()) + 1):
        mask = counts == c
        if mask.sum() > 0:
            centers[c] = embedding[mask].mean(axis=0)
    return centers


def compute_label_purity(embedding, counts, k=15):
    """k-NN label purity: for each point, what fraction of k neighbors share its label."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embedding, counts)
    # Get neighbor indices
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    # For each point, fraction of neighbors with same label
    purities = []
    for i in range(len(counts)):
        neighbor_labels = counts[indices[i]]
        purity = (neighbor_labels == counts[i]).mean()
        purities.append(purity)
    return np.array(purities)


def plot_brain_map(embedding, counts, centers, title, filename, params_str=""):
    """Generate a static brain map PNG."""
    max_count = int(counts.max())
    colors = _make_count_colors(max_count + 1)
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background points
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=counts, cmap=cmap, vmin=0, vmax=max_count,
        alpha=0.3, s=5, rasterized=True,
    )

    # Cluster centers with connecting lines
    sorted_counts = sorted(centers.keys())
    center_coords = np.array([centers[c] for c in sorted_counts])

    # Dashed lines connecting adjacent centers
    for i in range(len(sorted_counts) - 1):
        ax.plot(
            [center_coords[i, 0], center_coords[i+1, 0]],
            [center_coords[i, 1], center_coords[i+1, 1]],
            "k--", alpha=0.5, linewidth=1.5,
        )

    # Center dots with labels
    for i, c in enumerate(sorted_counts):
        ax.scatter(
            center_coords[i, 0], center_coords[i, 1],
            c=[colors[c]], s=300, edgecolors="black",
            linewidths=2, zorder=10,
        )
        ax.text(
            center_coords[i, 0], center_coords[i, 1], str(c),
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", zorder=11,
        )

    # Tick labels: show every Nth tick if too many
    tick_step = 1 if max_count <= 10 else (2 if max_count <= 20 else 5)
    ticks = list(range(0, max_count + 1, tick_step))
    cbar = plt.colorbar(scatter, ax=ax, label="Marked Count", ticks=ticks)
    ax.set_title(f"{title}\n{params_str}", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIG_DIR / filename}")


def plot_inter_cluster_distances(all_centers):
    """Bar chart of inter-cluster distances for each UMAP config."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (config_name, centers) in enumerate(all_centers.items()):
        ax = axes[idx]
        sorted_counts = sorted(centers.keys())
        pairs = []
        distances = []
        for i in range(len(sorted_counts) - 1):
            c1, c2 = sorted_counts[i], sorted_counts[i+1]
            d = np.linalg.norm(centers[c2] - centers[c1])
            pairs.append(f"{c1}→{c2}")
            distances.append(d)

        n_colors = max(sorted_counts) + 1 if sorted_counts else 1
        plot_colors = _make_count_colors(n_colors)
        bar_colors = [plot_colors[sorted_counts[i]] for i in range(len(pairs))]
        ax.bar(pairs, distances, color=bar_colors, edgecolor="black")
        ax.set_title(f"{config_name}", fontsize=12)
        ax.set_ylabel("Distance")
        ax.set_xlabel("Transition")

    plt.suptitle("Inter-Cluster Distances by UMAP Config", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "inter_cluster_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIG_DIR / 'inter_cluster_distances.png'}")


def main():
    print("=" * 60)
    print("Phase 2: Fit UMAP + Static Brain Maps")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    h_t = data["h_t"]
    counts = data["true_count"]
    n = len(counts)

    # UMAP configs
    umap_configs = [
        ("default", {"n_neighbors": 15, "min_dist": 0.1}),
        ("local",   {"n_neighbors": 5,  "min_dist": 0.05}),
        ("global",  {"n_neighbors": 30, "min_dist": 0.3}),
    ]

    embeddings = {}
    all_centers = {}

    for name, params in umap_configs:
        print(f"\n[UMAP {name}] n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric="euclidean",
            random_state=42,
        )
        emb = reducer.fit_transform(h_t)
        embeddings[name] = emb
        print(f"  Done. Shape: {emb.shape}")

        centers = compute_cluster_centers(emb, counts)
        all_centers[name] = centers

        params_str = f"n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, N={n}"
        plot_brain_map(emb, counts, centers, f"Brain Map — {name.title()}", f"brain_map_{name}.png", params_str)

    # PCA
    print(f"\n[PCA] Fitting PCA(n_components=2)...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(h_t)
    embeddings["pca"] = emb_pca
    var_ratio = pca.explained_variance_ratio_
    print(f"  PC1: {var_ratio[0]:.4f} ({var_ratio[0]:.1%}), PC2: {var_ratio[1]:.4f} ({var_ratio[1]:.1%})")

    centers_pca = compute_cluster_centers(emb_pca, counts)
    all_centers["pca"] = centers_pca
    plot_brain_map(
        emb_pca, counts, centers_pca, "Brain Map — PCA",
        "brain_map_pca.png",
        f"PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%}, N={n}",
    )

    # Save embeddings
    save_dict = {}
    for name, emb in embeddings.items():
        save_dict[f"emb_{name}"] = emb
    # Also save cluster centers
    for name, centers in all_centers.items():
        for c, pos in centers.items():
            save_dict[f"center_{name}_{c}"] = pos

    np.savez_compressed(DATA_DIR / "umap_embeddings.npz", **save_dict)
    print(f"\nSaved embeddings to {DATA_DIR / 'umap_embeddings.npz'}")

    # Inter-cluster distance chart
    print("\nPlotting inter-cluster distances...")
    plot_inter_cluster_distances(all_centers)

    # Label purity analysis
    print("\nComputing label purity (k-NN)...")
    for name, emb in embeddings.items():
        purities = compute_label_purity(emb, counts, k=15)
        mean_purity = purities.mean()
        per_count = {c: purities[counts == c].mean() for c in range(int(counts.max()) + 1) if (counts == c).sum() > 0}
        print(f"  {name}: mean purity={mean_purity:.3f}")
        for c, p in sorted(per_count.items()):
            print(f"    count {c}: {p:.3f}")

    print(f"\n{'=' * 60}")
    print("Phase 2 complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
