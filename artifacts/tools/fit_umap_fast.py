"""
Fast UMAP + PCA visualization for large datasets.
Subsamples to ~20K points for UMAP (standard practice), full PCA.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
FIG_DIR = _SCRIPT_DIR / "figures"

UMAP_SAMPLE = 20000  # subsample for UMAP


def _make_count_colors(n):
    if n <= 10:
        cmap = plt.cm.tab10
        return [cmap(i / 10) for i in range(n)]
    else:
        cmap = plt.cm.turbo
        return [cmap(i / max(n - 1, 1)) for i in range(n)]


def subsample_stratified(h_t, counts, n_total, rng):
    """Stratified subsample: proportional per count, min 10 per count."""
    unique_counts = np.unique(counts)
    n_counts = len(unique_counts)
    per_count = max(10, n_total // n_counts)

    indices = []
    for c in unique_counts:
        idx = np.where(counts == c)[0]
        n_take = min(per_count, len(idx))
        chosen = rng.choice(idx, n_take, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    rng.shuffle(indices)
    return indices[:n_total]


def compute_cluster_centers(embedding, counts):
    centers = {}
    for c in range(int(counts.max()) + 1):
        mask = counts == c
        if mask.sum() > 0:
            centers[c] = embedding[mask].mean(axis=0)
    return centers


def plot_brain_map(embedding, counts, centers, title, filename, params_str=""):
    max_count = int(counts.max())
    colors = _make_count_colors(max_count + 1)
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(14, 11))

    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=counts, cmap=cmap, vmin=0, vmax=max_count,
        alpha=0.25, s=3, rasterized=True,
    )

    sorted_counts = sorted(centers.keys())
    center_coords = np.array([centers[c] for c in sorted_counts])

    # Connecting lines
    for i in range(len(sorted_counts) - 1):
        ax.plot(
            [center_coords[i, 0], center_coords[i + 1, 0]],
            [center_coords[i, 1], center_coords[i + 1, 1]],
            "k-", alpha=0.4, linewidth=1.5,
        )

    # Center dots with labels
    for i, c in enumerate(sorted_counts):
        ax.scatter(
            center_coords[i, 0], center_coords[i, 1],
            c=[colors[c]], s=250, edgecolors="black",
            linewidths=2, zorder=10,
        )
        ax.text(
            center_coords[i, 0], center_coords[i, 1], str(c),
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", zorder=11,
        )

    tick_step = 1 if max_count <= 10 else (2 if max_count <= 20 else 5)
    ticks = list(range(0, max_count + 1, tick_step))
    plt.colorbar(scatter, ax=ax, label="Marked Count", ticks=ticks)
    ax.set_title(f"{title}\n{params_str}", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def main():
    print("=" * 60)
    print("Fast UMAP + PCA Brain Maps")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA_DIR / "trajectories.npz")
    h_t = d["h_t"]
    counts = d["true_count"]
    n = len(counts)
    max_count = int(counts.max())
    print(f"Loaded {n} timesteps, max count={max_count}")

    rng = np.random.RandomState(42)

    # --- PCA on full dataset (fast) ---
    print(f"\n[PCA] Full dataset ({n} points)...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(h_t)
    var = pca.explained_variance_ratio_
    print(f"  PC1={var[0]:.1%}, PC2={var[1]:.1%}")

    centers_pca = compute_cluster_centers(emb_pca, counts)
    plot_brain_map(
        emb_pca, counts, centers_pca, "Brain Map — PCA (full dataset)",
        "brain_map_pca.png",
        f"PC1={var[0]:.1%}, PC2={var[1]:.1%}, N={n}",
    )

    # --- Subsample for UMAP ---
    n_sample = min(UMAP_SAMPLE, n)
    print(f"\n[Subsample] {n_sample} points (stratified)...")
    sample_idx = subsample_stratified(h_t, counts, n_sample, rng)
    h_sub = h_t[sample_idx]
    c_sub = counts[sample_idx]
    print(f"  Count distribution: {dict(zip(*np.unique(c_sub, return_counts=True)))}")

    # --- UMAP configs ---
    import umap

    umap_configs = [
        ("default", {"n_neighbors": 15, "min_dist": 0.1}),
        ("local",   {"n_neighbors": 5,  "min_dist": 0.05}),
        ("global",  {"n_neighbors": 30, "min_dist": 0.3}),
    ]

    all_centers = {"pca": centers_pca}
    embeddings = {"pca": (emb_pca, counts)}

    for name, params in umap_configs:
        print(f"\n[UMAP {name}] n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']} on {n_sample} pts...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            metric="euclidean",
            random_state=42,
        )
        emb = reducer.fit_transform(h_sub)
        print(f"  Done.")

        centers = compute_cluster_centers(emb, c_sub)
        all_centers[name] = centers
        embeddings[name] = (emb, c_sub)

        plot_brain_map(
            emb, c_sub, centers, f"Brain Map — UMAP {name.title()}",
            f"brain_map_{name}.png",
            f"n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}, N={n_sample}",
        )

    # --- Inter-cluster distances ---
    print("\nPlotting inter-cluster distances...")
    n_configs = len(all_centers)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (config_name, centers) in enumerate(all_centers.items()):
        ax = axes[idx]
        sorted_c = sorted(centers.keys())
        pairs = []
        distances = []
        for i in range(len(sorted_c) - 1):
            c1, c2 = sorted_c[i], sorted_c[i + 1]
            d = np.linalg.norm(centers[c2] - centers[c1])
            pairs.append(f"{c1}→{c2}")
            distances.append(d)

        colors = _make_count_colors(max(sorted_c) + 1)
        bar_colors = [colors[sorted_c[i]] for i in range(len(pairs))]
        ax.bar(range(len(pairs)), distances, color=bar_colors, edgecolor="black", width=0.8)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=90, fontsize=6)
        ax.set_title(config_name, fontsize=12)
        ax.set_ylabel("Distance")

    plt.suptitle("Inter-Cluster Distances by Config", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "inter_cluster_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved inter_cluster_distances.png")

    # --- Save embeddings ---
    save_dict = {}
    for name, (emb, c) in embeddings.items():
        save_dict[f"emb_{name}"] = emb
        save_dict[f"counts_{name}"] = c
    for name, centers in all_centers.items():
        for c, pos in centers.items():
            save_dict[f"center_{name}_{c}"] = pos

    np.savez_compressed(DATA_DIR / "umap_embeddings.npz", **save_dict)
    print(f"\nSaved embeddings to {DATA_DIR / 'umap_embeddings.npz'}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
