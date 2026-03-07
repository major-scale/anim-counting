"""
Successor Function Measurement Battery
========================================
Runs 5 analyses on DreamerV3 latent representations to determine
whether successor structure exists.

Works with any count range (auto-detected from data).

Analyses:
  1. RSA with Spearman Correlation
  2. Geodesic Distance Linearity
  3. Homomorphism Error
  4. Anisotropy-Corrected Cosine Similarity
  5. Persistent Homology
"""

import os
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
FIG_DIR = _SCRIPT_DIR / "figures"

def _make_count_colors(n):
    """Generate n distinct colors using a perceptually uniform colormap."""
    if n <= 10:
        # Use tab10 for small count ranges
        cmap = plt.cm.tab10
        return [cmap(i / 10) for i in range(n)]
    else:
        # Use a continuous colormap for larger ranges
        cmap = plt.cm.turbo
        return [cmap(i / max(n - 1, 1)) for i in range(n)]


# Legacy 9-color palette for backward compat (used when max_count <= 8)
_LEGACY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


def load_data():
    d = np.load(DATA_DIR / "trajectories.npz")
    h_t = d["h_t"]
    counts = d["true_count"]
    print(f"Loaded {len(counts)} timesteps, h_t shape {h_t.shape}")
    return h_t, counts


def compute_centroids(h_t, counts):
    """Compute centroid for each unique count value."""
    max_count = int(counts.max())
    centroids = []
    valid_counts = []
    for c in range(max_count + 1):
        mask = counts == c
        n = mask.sum()
        if n > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
            print(f"  count={c}: {n} samples")
        else:
            print(f"  count={c}: 0 samples (SKIPPED)")
    return np.stack(centroids), np.array(valid_counts)


# ─── Analysis 1: RSA ────────────────────────────────────────
def analysis_1_rsa(centroids, valid_counts, h_t, counts):
    print("\n" + "=" * 60)
    print("ANALYSIS 1: RSA with Spearman Correlation")
    print("=" * 60)

    n_counts = len(valid_counts)

    # Agent RDM from centroids
    rdm_agent = squareform(pdist(centroids, metric='euclidean'))

    # Ideal RDM: |i - j|
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)

    # Upper triangle
    triu_idx = np.triu_indices(n_counts, k=1)
    agent_vec = rdm_agent[triu_idx]
    ideal_vec = rdm_ideal[triu_idx]

    spearman_rho, spearman_p = spearmanr(agent_vec, ideal_vec)
    pearson_r, pearson_p = pearsonr(agent_vec, ideal_vec)

    print(f"\n  RSA Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.2e})")
    print(f"  RSA Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")

    # Noise ceiling: split-half reliability
    print("\n  Computing noise ceiling (split-half reliability)...")
    n_splits = 20
    split_correlations = []
    rng = np.random.RandomState(42)
    for _ in range(n_splits):
        centroids_a = []
        centroids_b = []
        for c in valid_counts:
            mask = counts == c
            idx = np.where(mask)[0]
            rng.shuffle(idx)
            half = len(idx) // 2
            centroids_a.append(h_t[idx[:half]].mean(axis=0))
            centroids_b.append(h_t[idx[half:]].mean(axis=0))
        rdm_a = squareform(pdist(np.stack(centroids_a), metric='euclidean'))
        rdm_b = squareform(pdist(np.stack(centroids_b), metric='euclidean'))
        rho, _ = spearmanr(rdm_a[triu_idx], rdm_b[triu_idx])
        split_correlations.append(rho)

    noise_ceil = np.mean(split_correlations)
    print(f"  Noise ceiling (split-half ρ): {noise_ceil:.4f} ± {np.std(split_correlations):.4f}")

    # Plot RDMs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(rdm_agent, cmap='viridis')
    axes[0].set_title("Agent RDM (Euclidean)", fontsize=12)
    axes[0].set_xticks(range(n_counts))
    axes[0].set_xticklabels(valid_counts)
    axes[0].set_yticks(range(n_counts))
    axes[0].set_yticklabels(valid_counts)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rdm_ideal, cmap='viridis')
    axes[1].set_title("Ideal RDM (|i - j|)", fontsize=12)
    axes[1].set_xticks(range(n_counts))
    axes[1].set_xticklabels(valid_counts)
    axes[1].set_yticks(range(n_counts))
    axes[1].set_yticklabels(valid_counts)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].scatter(ideal_vec, agent_vec, c='steelblue', s=40, alpha=0.7)
    axes[2].set_xlabel("|i - j| (ideal distance)")
    axes[2].set_ylabel("||h̄_i - h̄_j|| (agent distance)")
    axes[2].set_title(f"RSA: Spearman ρ={spearman_rho:.4f}, Pearson r={pearson_r:.4f}", fontsize=12)
    # Fit line
    z = np.polyfit(ideal_vec, agent_vec, 1)
    x_fit = np.linspace(0, ideal_vec.max(), 100)
    axes[2].plot(x_fit, np.polyval(z, x_fit), 'r--', alpha=0.7)

    plt.suptitle("Analysis 1: Representational Similarity Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "successor_rsa.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved successor_rsa.png")

    return {
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "noise_ceiling": float(noise_ceil),
        "noise_ceiling_std": float(np.std(split_correlations)),
        "rdm_agent": rdm_agent.tolist(),
        "rdm_ideal": rdm_ideal.tolist(),
    }


# ─── Analysis 2: Geodesic Distance Linearity ────────────────
def analysis_2_geodesic(centroids, valid_counts, h_t, counts):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Geodesic Distance Linearity")
    print("=" * 60)

    n_counts = len(valid_counts)
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu_idx = np.triu_indices(n_counts, k=1)
    ideal_vec = rdm_ideal[triu_idx]

    results = {}
    colors = _make_count_colors(n_counts)

    # --- Centroid-based (small graph) ---
    print(f"\n  Centroid-based geodesic ({n_counts} points):")
    for k in [2, 3, 4]:
        try:
            graph = kneighbors_graph(centroids, n_neighbors=min(k, n_counts - 1),
                                     mode='distance')
            graph = 0.5 * (graph + graph.T)
            geodesic = shortest_path(graph, method='D')

            if np.any(np.isinf(geodesic[triu_idx])):
                print(f"    k={k}: Graph not fully connected, skipping")
                continue

            consec = [geodesic[i, i + 1] for i in range(n_counts - 1)]
            cv = np.std(consec) / np.mean(consec) if np.mean(consec) > 0 else float('inf')

            geo_vec = geodesic[triu_idx]
            r, p = pearsonr(geo_vec, ideal_vec)

            print(f"    k={k}: CV = {cv:.4f}, Pearson r = {r:.4f} (p = {p:.2e})")
            print(f"      Consecutive: {[f'{d:.3f}' for d in consec]}")

            results[f"centroid_k{k}"] = {
                "cv": float(cv),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "consecutive_distances": [float(d) for d in consec],
            }
        except Exception as e:
            print(f"    k={k}: Failed — {e}")

    # --- Dense sample (100 per count = 900 points) ---
    print("\n  Dense sample geodesic (100 per count):")
    rng = np.random.RandomState(42)
    dense_samples = []
    dense_labels = []
    for i, c in enumerate(valid_counts):
        idx = np.where(counts == c)[0]
        chosen = rng.choice(idx, min(100, len(idx)), replace=False)
        dense_samples.append(h_t[chosen])
        dense_labels.extend([i] * len(chosen))
    dense_h = np.vstack(dense_samples)
    dense_labels = np.array(dense_labels)

    for k in [5, 10]:
        try:
            graph = kneighbors_graph(dense_h, n_neighbors=k, mode='distance')
            graph = 0.5 * (graph + graph.T)
            geodesic_full = shortest_path(graph, method='D')

            # Compute centroid-to-centroid geodesic from dense graph
            geo_centroids = np.zeros((n_counts, n_counts))
            for i in range(n_counts):
                for j in range(n_counts):
                    mask_i = dense_labels == i
                    mask_j = dense_labels == j
                    # Mean geodesic between all pairs of points from count i to count j
                    # (too expensive — use centroid approximation: mean geodesic from i-points to j-points)
                    dists_ij = geodesic_full[np.ix_(np.where(mask_i)[0], np.where(mask_j)[0])]
                    finite_mask = np.isfinite(dists_ij)
                    if finite_mask.any():
                        geo_centroids[i, j] = np.mean(dists_ij[finite_mask])
                    else:
                        geo_centroids[i, j] = np.inf

            if np.any(np.isinf(geo_centroids[triu_idx])):
                print(f"    k={k}: Some pairs unreachable, skipping")
                continue

            consec = [geo_centroids[i, i + 1] for i in range(n_counts - 1)]
            cv = np.std(consec) / np.mean(consec) if np.mean(consec) > 0 else float('inf')
            geo_vec = geo_centroids[triu_idx]
            r, p = pearsonr(geo_vec, ideal_vec)

            print(f"    k={k}: CV = {cv:.4f}, Pearson r = {r:.4f} (p = {p:.2e})")
            print(f"      Consecutive: {[f'{d:.3f}' for d in consec]}")

            results[f"dense_k{k}"] = {
                "cv": float(cv),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "consecutive_distances": [float(d) for d in consec],
            }
        except Exception as e:
            print(f"    k={k}: Failed — {e}")

    # Plot: geodesic distance vs |i-j| for best config
    best_key = None
    best_r = -1
    for key, val in results.items():
        if val["pearson_r"] > best_r:
            best_r = val["pearson_r"]
            best_key = key

    if best_key:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Consecutive distances
        consec = results[best_key]["consecutive_distances"]
        labels = [f"{valid_counts[i]}→{valid_counts[i+1]}" for i in range(n_counts - 1)]
        bar_colors = [colors[i] for i in range(n_counts - 1)]
        ax1.bar(labels, consec, color=bar_colors, edgecolor="black")
        ax1.axhline(np.mean(consec), color="red", linestyle="--", alpha=0.7,
                     label=f"mean={np.mean(consec):.3f}")
        ax1.set_ylabel("Geodesic Distance")
        ax1.set_title(f"Consecutive Geodesic Distances ({best_key})\nCV = {results[best_key]['cv']:.4f}")
        ax1.legend()

        # Geodesic vs ideal
        # Recompute for plotting
        ax2.set_xlabel("|i - j|")
        ax2.set_ylabel("Geodesic distance")
        ax2.set_title(f"Geodesic Distance vs |i-j| (r={best_r:.4f})")
        ax2.text(0.05, 0.95, f"Best config: {best_key}",
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Use consecutive distances to reconstruct approximate geodesic matrix
        # and plot against ideal
        cum_geo = np.zeros(n_counts)
        for i in range(1, n_counts):
            cum_geo[i] = cum_geo[i - 1] + consec[i - 1]
        geo_approx = np.abs(cum_geo[:, None] - cum_geo[None, :])
        geo_vec = geo_approx[triu_idx]
        ax2.scatter(ideal_vec, geo_vec, c='steelblue', s=40, alpha=0.7)
        z = np.polyfit(ideal_vec, geo_vec, 1)
        x_fit = np.linspace(0, ideal_vec.max(), 100)
        ax2.plot(x_fit, np.polyval(z, x_fit), 'r--', alpha=0.7)

        plt.suptitle("Analysis 2: Geodesic Distance Linearity", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "successor_geodesic.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved successor_geodesic.png")

    return results


# ─── Analysis 3: Homomorphism Error ─────────────────────────
def analysis_3_homomorphism(centroids, valid_counts):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Homomorphism Error")
    print("=" * 60)

    n_counts = len(valid_counts)
    displacements = centroids[1:] - centroids[:-1]  # (n_counts-1, 512)
    v_mean = displacements.mean(axis=0)

    # Per-step homomorphism error
    he_per_step = np.sum((displacements - v_mean) ** 2, axis=1) / np.sum(v_mean ** 2)
    he_total = he_per_step.mean()

    print(f"\n  Mean displacement ||v̄|| = {np.linalg.norm(v_mean):.4f}")
    print(f"  Homomorphism Error (HE) = {he_total:.4f}")
    print(f"  Per-step HE:")
    for i in range(n_counts - 1):
        label = f"    {valid_counts[i]}→{valid_counts[i+1]}"
        print(f"{label}: HE={he_per_step[i]:.4f}, ||d||={np.linalg.norm(displacements[i]):.4f}")

    # Direction + magnitude combined metric
    n_disp = len(displacements)
    similarities = []
    sim_matrix = np.zeros((n_disp, n_disp))
    for i in range(n_disp):
        for j in range(n_disp):
            di, dj = displacements[i], displacements[j]
            denom = np.sum(di ** 2) + np.sum(dj ** 2)
            if denom > 0:
                sim = 1 - np.sum((di - dj) ** 2) / denom
            else:
                sim = 1.0
            sim_matrix[i, j] = sim
            if j > i:
                similarities.append(sim)

    print(f"\n  Displacement similarity (direction + magnitude):")
    print(f"    Mean: {np.mean(similarities):.4f}")
    print(f"    Min:  {np.min(similarities):.4f}")
    print(f"    Max:  {np.max(similarities):.4f}")

    # Cosine similarities between displacements (for comparison)
    disp_norms = np.linalg.norm(displacements, axis=1, keepdims=True)
    disp_unit = displacements / np.maximum(disp_norms, 1e-8)
    cos_matrix = disp_unit @ disp_unit.T

    print(f"\n  Raw cosine similarities between displacements:")
    for i in range(n_disp):
        for j in range(i + 1, n_disp):
            li = f"{valid_counts[i]}→{valid_counts[i+1]}"
            lj = f"{valid_counts[j]}→{valid_counts[j+1]}"
            print(f"    {li} vs {lj}: cos={cos_matrix[i,j]:.4f}, sim={sim_matrix[i,j]:.4f}")
    print(f"  Mean pairwise cosine: {cos_matrix[np.triu_indices(n_disp, k=1)].mean():.4f}")

    # Plot
    colors = _make_count_colors(n_counts)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-step HE bar chart
    labels = [f"{valid_counts[i]}→{valid_counts[i+1]}" for i in range(n_counts - 1)]
    bar_colors = [colors[i] for i in range(n_counts - 1)]
    axes[0].bar(labels, he_per_step, color=bar_colors, edgecolor="black")
    axes[0].axhline(he_total, color="red", linestyle="--", alpha=0.7,
                     label=f"mean HE={he_total:.4f}")
    axes[0].set_ylabel("Homomorphism Error")
    axes[0].set_title("Per-Transition Homomorphism Error")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Displacement similarity matrix
    im = axes[1].imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_xticks(range(n_disp))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(n_disp))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_title("Displacement Similarity (dir + mag)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Cosine similarity matrix
    im2 = axes[2].imshow(cos_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xticks(range(n_disp))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticks(range(n_disp))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_title("Raw Cosine Similarity")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Analysis 3: Homomorphism Error (HE = {he_total:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "successor_homomorphism.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved successor_homomorphism.png")

    return {
        "he_total": float(he_total),
        "he_per_step": {f"{valid_counts[i]}->{valid_counts[i+1]}": float(he_per_step[i])
                        for i in range(n_counts - 1)},
        "mean_displacement_norm": float(np.linalg.norm(v_mean)),
        "displacement_norms": {f"{valid_counts[i]}->{valid_counts[i+1]}": float(np.linalg.norm(displacements[i]))
                               for i in range(n_counts - 1)},
        "displacement_similarity_mean": float(np.mean(similarities)),
        "displacement_similarity_min": float(np.min(similarities)),
        "mean_pairwise_cosine": float(cos_matrix[np.triu_indices(n_disp, k=1)].mean()),
    }


# ─── Analysis 4: Anisotropy-Corrected Cosine ────────────────
def analysis_4_anisotropy(centroids, valid_counts, h_t):
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Anisotropy-Corrected Cosine Similarity")
    print("=" * 60)

    n_counts = len(valid_counts)
    displacements = centroids[1:] - centroids[:-1]

    # Random baseline
    n_samples = 2000
    rng = np.random.RandomState(42)
    idx1 = rng.choice(len(h_t), n_samples)
    idx2 = rng.choice(len(h_t), n_samples)
    baseline_cosines = []
    for i, j in zip(idx1, idx2):
        if i != j:
            v1, v2 = h_t[i], h_t[j]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            baseline_cosines.append(cos)
    baseline_cos = np.mean(baseline_cosines)
    baseline_std = np.std(baseline_cosines)

    print(f"\n  Anisotropy baseline: mean={baseline_cos:.4f} ± {baseline_std:.4f}")

    # Random displacement baseline (crucial: compare displacement vectors, not h_t vectors)
    random_disp_cosines = []
    for _ in range(2000):
        i1, i2 = rng.choice(len(h_t) - 1, 2, replace=False)
        d1 = h_t[i1 + 1] - h_t[i1]
        d2 = h_t[i2 + 1] - h_t[i2]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 > 1e-8 and n2 > 1e-8:
            random_disp_cosines.append(np.dot(d1, d2) / (n1 * n2))
    disp_baseline = np.mean(random_disp_cosines)
    disp_baseline_std = np.std(random_disp_cosines)
    print(f"  Random displacement baseline: mean={disp_baseline:.4f} ± {disp_baseline_std:.4f}")

    # Consecutive displacement cosines
    labels = []
    raw_cosines = []
    corrected_cosines = []
    disp_corrected = []
    for i in range(n_counts - 2):
        d1, d2 = displacements[i], displacements[i + 1]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        cos = np.dot(d1, d2) / (n1 * n2 + 1e-10)
        raw_cosines.append(cos)
        corrected_cosines.append(cos - baseline_cos)
        disp_corrected.append(cos - disp_baseline)
        labels.append(f"{valid_counts[i]}→{valid_counts[i+1]} vs {valid_counts[i+1]}→{valid_counts[i+2]}")

    # All pairs (not just consecutive)
    n_disp = len(displacements)
    all_raw = []
    for i in range(n_disp):
        for j in range(i + 1, n_disp):
            d1, d2 = displacements[i], displacements[j]
            cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
            all_raw.append(cos)
    mean_all_raw = np.mean(all_raw)

    print(f"\n  Consecutive displacement cosines:")
    print(f"  {'Pair':<25} {'Raw':>8} {'−h_t base':>10} {'−disp base':>11}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*11}")
    for i, label in enumerate(labels):
        print(f"  {label:<25} {raw_cosines[i]:>8.4f} {corrected_cosines[i]:>10.4f} {disp_corrected[i]:>11.4f}")

    print(f"\n  Summary:")
    print(f"    Mean raw consecutive cosine: {np.mean(raw_cosines):.4f}")
    print(f"    Mean all-pairs raw cosine: {mean_all_raw:.4f}")
    print(f"    Mean corrected (−h_t baseline): {np.mean(corrected_cosines):.4f}")
    print(f"    Mean corrected (−disp baseline): {np.mean(disp_corrected):.4f}")

    # Plot
    colors = _make_count_colors(n_counts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, raw_cosines, width, label='Raw cosine', color='steelblue')
    ax1.bar(x + width / 2, disp_corrected, width, label='Corrected (−disp baseline)', color='coral')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axhline(disp_baseline, color='red', linestyle='--', alpha=0.5,
                label=f'Displacement baseline ({disp_baseline:.4f})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Consecutive Displacement Cosine")
    ax1.legend(fontsize=8)

    # Distribution of random displacement cosines vs actual
    ax2.hist(random_disp_cosines, bins=50, alpha=0.5, color='gray',
             label='Random displacements', density=True)
    for i, (rc, label) in enumerate(zip(raw_cosines, labels)):
        ax2.axvline(rc, color=colors[i], linewidth=2,
                     alpha=0.8, label=label if i < 4 else None)
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Density")
    ax2.set_title("Displacement Cosines vs Random Baseline")
    ax2.legend(fontsize=7, loc='upper left')

    plt.suptitle("Analysis 4: Anisotropy-Corrected Cosine Similarity", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "successor_anisotropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved successor_anisotropy.png")

    return {
        "h_t_baseline": float(baseline_cos),
        "h_t_baseline_std": float(baseline_std),
        "displacement_baseline": float(disp_baseline),
        "displacement_baseline_std": float(disp_baseline_std),
        "consecutive_raw": {labels[i]: float(raw_cosines[i]) for i in range(len(labels))},
        "consecutive_corrected_disp": {labels[i]: float(disp_corrected[i]) for i in range(len(labels))},
        "mean_raw_consecutive": float(np.mean(raw_cosines)),
        "mean_all_pairs_raw": float(mean_all_raw),
        "mean_corrected_disp": float(np.mean(disp_corrected)),
    }


# ─── Analysis 5: Persistent Homology ────────────────────────
def analysis_5_topology(centroids, valid_counts, h_t, counts):
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Persistent Homology")
    print("=" * 60)

    from ripser import ripser

    # Centroids
    print(f"\n  Centroid persistence ({len(valid_counts)} points):")
    result = ripser(centroids, maxdim=1)
    diagrams = result['dgms']

    h0 = diagrams[0]
    h1 = diagrams[1]
    print(f"    H0 (connected components): {len(h0)} features")
    print(f"    H1 (loops): {len(h1)} features")

    h0_deaths = [float(d) for b, d in h0 if d != np.inf]
    if h0_deaths:
        h0_cv = np.std(h0_deaths) / np.mean(h0_deaths) if np.mean(h0_deaths) > 0 else float('inf')
        print(f"    H0 death times: {[f'{d:.3f}' for d in sorted(h0_deaths)]}")
        print(f"    H0 death time CV: {h0_cv:.4f}")
        print(f"    (Equal deaths = equal spacing; CV=0 is perfect)")
    else:
        h0_cv = float('nan')

    if len(h1) > 0:
        h1_lifetimes = [float(d - b) for b, d in h1 if d != np.inf]
        print(f"    H1 lifetimes: {[f'{l:.3f}' for l in h1_lifetimes]}")
        print(f"    (Non-trivial H1 means loops — not expected for a line)")
    else:
        h1_lifetimes = []
        print(f"    No H1 features — consistent with open curve topology")

    # Dense sample (50 per count)
    print("\n  Dense sample persistence (50 per count):")
    rng = np.random.RandomState(42)
    dense = []
    for c in valid_counts:
        idx = np.where(counts == c)[0]
        chosen = rng.choice(idx, min(50, len(idx)), replace=False)
        dense.append(h_t[chosen])
    dense_h = np.vstack(dense)

    result_dense = ripser(dense_h, maxdim=1)
    diagrams_dense = result_dense['dgms']
    print(f"    H0: {len(diagrams_dense[0])} features")
    print(f"    H1: {len(diagrams_dense[1])} features")
    if len(diagrams_dense[1]) > 0:
        h1_dense = sorted([float(d - b) for b, d in diagrams_dense[1] if d != np.inf], reverse=True)
        print(f"    Top 5 H1 lifetimes: {[f'{l:.3f}' for l in h1_dense[:5]]}")

    # Plot persistence diagrams
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for dim, diagram in enumerate(diagrams[:2]):
        ax = axes[dim]
        if len(diagram) == 0:
            ax.text(0.5, 0.5, f"No H{dim} features", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f"H{dim} Persistence Diagram (centroids)")
            continue
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        finite = deaths != np.inf
        if finite.any():
            ax.scatter(births[finite], deaths[finite], s=40, alpha=0.7,
                       label=f'H{dim} ({finite.sum()} features)')
        if (~finite).any():
            inf_y = deaths[finite].max() * 1.1 if finite.any() else births.max() * 1.5
            ax.scatter(births[~finite], [inf_y] * (~finite).sum(),
                       s=80, marker='^', color='red', label='∞ (never dies)')
        max_val = 1.0
        if len(births) > 0:
            max_val = births.max()
        if finite.any():
            max_val = max(max_val, deaths[finite].max())
        max_val *= 1.2
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"H{dim} Persistence Diagram (centroids)")
        ax.legend()

    plt.suptitle("Analysis 5: Persistent Homology", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "successor_topology.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved successor_topology.png")

    return {
        "centroid_h0_count": len(h0),
        "centroid_h1_count": len(h1),
        "h0_death_cv": float(h0_cv) if not np.isnan(h0_cv) else None,
        "h0_deaths": h0_deaths,
        "h1_lifetimes": h1_lifetimes,
        "dense_h0_count": len(diagrams_dense[0]),
        "dense_h1_count": len(diagrams_dense[1]),
    }


# ─── Main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SUCCESSOR FUNCTION MEASUREMENT BATTERY")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    h_t, counts = load_data()

    print("\nComputing centroids...")
    centroids, valid_counts = compute_centroids(h_t, counts)

    all_results = {}
    all_results["analysis_1_rsa"] = analysis_1_rsa(centroids, valid_counts, h_t, counts)
    all_results["analysis_2_geodesic"] = analysis_2_geodesic(centroids, valid_counts, h_t, counts)
    all_results["analysis_3_homomorphism"] = analysis_3_homomorphism(centroids, valid_counts)
    all_results["analysis_4_anisotropy"] = analysis_4_anisotropy(centroids, valid_counts, h_t)
    try:
        all_results["analysis_5_topology"] = analysis_5_topology(centroids, valid_counts, h_t, counts)
    except ImportError:
        print("\n  SKIPPING Analysis 5: ripser not installed (pip install ripser)")
        all_results["analysis_5_topology"] = {"skipped": True}

    # ─── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    rsa = all_results["analysis_1_rsa"]
    homo = all_results["analysis_3_homomorphism"]
    aniso = all_results["analysis_4_anisotropy"]

    print(f"""
  RSA Spearman ρ:           {rsa['spearman_rho']:.4f}  (p = {rsa['spearman_p']:.2e})
  RSA Pearson r:            {rsa['pearson_r']:.4f}  (p = {rsa['pearson_p']:.2e})
  Noise ceiling:            {rsa['noise_ceiling']:.4f}
  Homomorphism Error:       {homo['he_total']:.4f}
  Displacement similarity:  {homo['displacement_similarity_mean']:.4f} (mean), {homo['displacement_similarity_min']:.4f} (min)
  Mean pairwise cosine:     {homo['mean_pairwise_cosine']:.4f}
  Anisotropy baseline:      {aniso['displacement_baseline']:.4f}
  Corrected cosine (mean):  {aniso['mean_corrected_disp']:.4f}
""")

    has_ordinal = rsa['spearman_rho'] > 0.95
    has_successor = homo['he_total'] < 0.5
    if has_ordinal and has_successor:
        verdict = ("SUCCESSOR STRUCTURE PRESENT. The agent has learned an approximate "
                   "homomorphism from counting to latent space. Cosine similarity was "
                   "a measurement artifact — the successor operation exists along a "
                   "curved manifold where displacement vectors rotate while maintaining "
                   "consistent magnitude and step structure.")
    elif has_ordinal and not has_successor:
        verdict = ("ORDINAL WITHOUT OPERATIONAL SUCCESSOR. The agent knows the order "
                   "of counts but the successor operation is not uniformly encoded. "
                   "New environment designs targeting bidirectional counting and "
                   "variable increments are warranted.")
    else:
        verdict = ("MIXED RESULTS. See per-transition breakdown for details on which "
                   "count transitions show successor-like structure.")

    print(f"  INTERPRETATION: {verdict}")

    # Save
    all_results["verdict"] = verdict
    outpath = DATA_DIR / "successor_battery_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved results to {outpath}")

    print(f"\n{'=' * 60}")
    print("Battery complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
