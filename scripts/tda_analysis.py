#!/usr/bin/env python3
"""
TDA Analysis — Persistent Homology on Binary RSSM Count Centroids
=================================================================
Computes persistent homology on the 15 count centroids in 512-d hidden
state space. Tests whether the topology matches a 4-dimensional hypercube
subset.

Also runs on the full hidden state point cloud (subsampled) for higher-
resolution topological features.

Output: artifacts/binary_successor/tda_analysis.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

SCRIPT_DIR = Path(__file__).resolve().parent
BATTERY_PATH = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")
OUT_DIR = Path("/workspace/bridge/artifacts/binary_successor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_centroids(h_t, counts):
    """Compute mean hidden state per count value."""
    unique_counts = sorted(np.unique(counts))
    centroids = []
    for c in unique_counts:
        mask = counts == c
        centroids.append(h_t[mask].mean(axis=0))
    return np.array(centroids), unique_counts


def betti_at_scale(diagrams, scale):
    """Count features alive at a given filtration scale."""
    bettis = []
    for dim_dgm in diagrams:
        if len(dim_dgm) == 0:
            bettis.append(0)
            continue
        alive = np.sum((dim_dgm[:, 0] <= scale) & (dim_dgm[:, 1] > scale))
        bettis.append(int(alive))
    return bettis


def persistence_summary(diagrams):
    """Summarize persistence diagrams."""
    summary = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            summary[f"H{dim}"] = {"n_features": 0, "births": [], "deaths": [],
                                   "lifetimes": [], "max_lifetime": 0}
            continue
        # Filter out infinite death features for lifetime calc
        finite = dgm[dgm[:, 1] < np.inf]
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
        inf_count = int(np.sum(dgm[:, 1] == np.inf))

        summary[f"H{dim}"] = {
            "n_features": int(len(dgm)),
            "n_finite": int(len(finite)),
            "n_infinite": inf_count,
            "births": dgm[:, 0].tolist(),
            "deaths": [d if d < np.inf else "inf" for d in dgm[:, 1].tolist()],
            "lifetimes": lifetimes.tolist() if len(lifetimes) > 0 else [],
            "max_lifetime": float(lifetimes.max()) if len(lifetimes) > 0 else 0,
            "mean_lifetime": float(lifetimes.mean()) if len(lifetimes) > 0 else 0,
        }
    return summary


def hypercube_distances():
    """Compute pairwise Hamming distances for 4-bit binary codes 0-14."""
    codes = []
    for i in range(15):
        bits = [(i >> b) & 1 for b in range(4)]
        codes.append(bits)
    codes = np.array(codes)
    return squareform(pdist(codes, metric='hamming')) * 4  # Scale to actual Hamming distance


def main():
    print("Loading battery data...")
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']      # (13280, 512)
    counts = data['counts']  # (13280,)
    bits = data['bits']      # (13280, 4)

    print(f"  h_t: {h_t.shape}, counts: {counts.shape}")

    # Compute centroids
    centroids, unique_counts = compute_centroids(h_t, counts)
    print(f"  Centroids: {centroids.shape} for counts {unique_counts}")

    results = {}

    # --- Experiment 1: Persistent homology on centroids ---
    print("\n[1] Persistent homology on 15 count centroids...")
    rips = ripser(centroids, maxdim=3)
    diagrams = rips['dgms']

    results["centroids"] = {
        "n_points": int(len(centroids)),
        "ambient_dim": int(centroids.shape[1]),
        "persistence": persistence_summary(diagrams),
    }

    # Betti numbers at multiple scales
    dists = squareform(pdist(centroids))
    dist_range = np.linspace(0, dists.max(), 50)
    betti_curves = {}
    for scale in dist_range:
        bettis = betti_at_scale(diagrams, scale)
        for dim, b in enumerate(bettis):
            betti_curves.setdefault(f"beta_{dim}", []).append(int(b))
    results["centroids"]["betti_curves"] = betti_curves
    results["centroids"]["filtration_scales"] = dist_range.tolist()

    # Key Betti numbers at natural scales
    median_dist = float(np.median(dists[dists > 0]))
    mean_dist = float(np.mean(dists[dists > 0]))
    min_dist = float(np.min(dists[dists > 0]))
    max_dist = float(np.max(dists))

    print(f"  Distance stats: min={min_dist:.2f}, median={median_dist:.2f}, "
          f"mean={mean_dist:.2f}, max={max_dist:.2f}")

    for label, scale in [("min_dist", min_dist), ("median_dist", median_dist),
                          ("mean_dist", mean_dist), ("75pct", np.percentile(dists[dists>0], 75))]:
        bettis = betti_at_scale(diagrams, scale)
        results["centroids"][f"betti_at_{label}"] = {
            "scale": float(scale),
            "beta_0": bettis[0] if len(bettis) > 0 else 0,
            "beta_1": bettis[1] if len(bettis) > 1 else 0,
            "beta_2": bettis[2] if len(bettis) > 2 else 0,
            "beta_3": bettis[3] if len(bettis) > 3 else 0,
        }
        print(f"  Betti at {label} ({scale:.2f}): β₀={bettis[0]}, β₁={bettis[1]}, "
              f"β₂={bettis[2] if len(bettis) > 2 else '?'}, β₃={bettis[3] if len(bettis) > 3 else '?'}")

    # Long-lived features (lifetime > 25% of distance range)
    threshold = 0.25 * max_dist
    long_lived = {}
    for dim, dgm in enumerate(diagrams):
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            significant = lifetimes > threshold
            long_lived[f"H{dim}"] = {
                "n_significant": int(significant.sum()),
                "threshold": float(threshold),
                "lifetimes": lifetimes[significant].tolist(),
            }
        else:
            long_lived[f"H{dim}"] = {"n_significant": 0, "threshold": float(threshold)}
    results["centroids"]["long_lived_features"] = long_lived
    print(f"\n  Long-lived features (lifetime > {threshold:.2f}):")
    for dim, info in long_lived.items():
        print(f"    {dim}: {info['n_significant']} features")

    # --- Experiment 2: Compare centroid distances to Hamming distances ---
    print("\n[2] Centroid distances vs Hamming distances...")
    centroid_dists = squareform(pdist(centroids))
    hamming_dists = hypercube_distances()

    # Correlation between centroid distances and Hamming distances
    upper_tri = np.triu_indices(15, k=1)
    cd = centroid_dists[upper_tri]
    hd = hamming_dists[upper_tri]

    from scipy.stats import spearmanr, pearsonr
    spearman_r, spearman_p = spearmanr(cd, hd)
    pearson_r, pearson_p = pearsonr(cd, hd)

    results["hamming_comparison"] = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mean_centroid_dist_by_hamming": {},
    }

    # Mean centroid distance grouped by Hamming distance
    for h in sorted(np.unique(hd)):
        mask = hd == h
        results["hamming_comparison"]["mean_centroid_dist_by_hamming"][str(int(h))] = {
            "mean": float(cd[mask].mean()),
            "std": float(cd[mask].std()),
            "n": int(mask.sum()),
        }
        print(f"  Hamming={int(h)}: centroid dist = {cd[mask].mean():.2f} ± {cd[mask].std():.2f} (n={mask.sum()})")

    print(f"  Spearman r={spearman_r:.3f} (p={spearman_p:.1e})")
    print(f"  Pearson r={pearson_r:.3f} (p={pearson_p:.1e})")

    # --- Experiment 3: Persistent homology on full point cloud (subsampled) ---
    print("\n[3] Persistent homology on subsampled point cloud...")
    # Subsample to ~500 points for tractability
    rng = np.random.RandomState(42)
    n_subsample = min(500, len(h_t))
    indices = rng.choice(len(h_t), n_subsample, replace=False)
    h_sub = h_t[indices]

    # Use PCA to reduce dimensionality for faster computation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    h_pca = pca.fit_transform(h_sub)
    pca_var = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA 20 components: {pca_var[-1]*100:.1f}% variance explained")

    rips_cloud = ripser(h_pca, maxdim=2)
    diagrams_cloud = rips_cloud['dgms']

    results["point_cloud"] = {
        "n_points": n_subsample,
        "pca_components": 20,
        "pca_variance_explained": float(pca_var[-1]),
        "persistence": persistence_summary(diagrams_cloud),
    }

    for dim, dgm in enumerate(diagrams_cloud):
        finite = dgm[dgm[:, 1] < np.inf]
        n_feat = len(dgm)
        n_significant = 0
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            threshold_cloud = np.percentile(lifetimes, 90) if len(lifetimes) > 0 else 0
            n_significant = int((lifetimes > threshold_cloud).sum())
        print(f"  H{dim}: {n_feat} features total, {n_significant} above 90th percentile lifetime")

    # --- Experiment 4: Per-count persistent homology ---
    print("\n[4] Per-count cloud topology...")
    results["per_count_topology"] = {}
    for c in unique_counts:
        mask = counts == c
        h_c = h_t[mask]
        if len(h_c) < 10:
            continue
        # Subsample if too large
        if len(h_c) > 200:
            idx = rng.choice(len(h_c), 200, replace=False)
            h_c = h_c[idx]

        pca_c = PCA(n_components=min(10, len(h_c)-1))
        h_c_pca = pca_c.fit_transform(h_c)

        rips_c = ripser(h_c_pca, maxdim=1)
        dgms_c = rips_c['dgms']

        # Count significant H1 features (cycles)
        h1 = dgms_c[1] if len(dgms_c) > 1 else np.array([])
        n_h1 = len(h1)
        max_h1_life = 0
        if len(h1) > 0:
            finite_h1 = h1[h1[:, 1] < np.inf]
            if len(finite_h1) > 0:
                max_h1_life = float((finite_h1[:, 1] - finite_h1[:, 0]).max())

        results["per_count_topology"][str(c)] = {
            "n_points": int(len(h_c)),
            "beta_0_inf": int(np.sum(dgms_c[0][:, 1] == np.inf)) if len(dgms_c[0]) > 0 else 0,
            "n_h1_features": n_h1,
            "max_h1_lifetime": max_h1_life,
        }
        print(f"  Count {c:>2}: n={len(h_c):>3}, β₀∞={results['per_count_topology'][str(c)]['beta_0_inf']}, "
              f"H1 features={n_h1}, max H1 life={max_h1_life:.3f}")

    # Save results
    out_path = OUT_DIR / "tda_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
