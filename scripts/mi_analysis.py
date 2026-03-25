#!/usr/bin/env python3
"""
Mutual Information Analysis — Block-Diagonal Structure in Binary RSSM
=====================================================================
Tests whether hidden state dimensions show block-diagonal MI structure
with individual bits, confirming compositional representation.

Also computes:
- MI(h_dim, count) vs MI(h_dim, individual_bits) to test compositionality
- Bit-specific subspace identification via top-MI dimensions
- Overlap between bit subspaces (Jaccard on top-k dimensions)
- Conditional MI: I(h; bit_i | bit_j) to test independence

Output: artifacts/binary_successor/mi_analysis.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

BATTERY_PATH = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")
OUT_DIR = Path("/workspace/bridge/artifacts/binary_successor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_mi_matrix(h_t, bits, n_dims=100):
    """Compute MI between top-variance hidden dims and each bit.

    Returns (n_dims, 4) matrix where entry [i,j] = MI(h_dim_i, bit_j).
    Uses top-variance dims for efficiency and signal.
    """
    # Select top-variance dimensions
    variances = h_t.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:n_dims]
    h_selected = h_t[:, top_dims]

    mi_matrix = np.zeros((n_dims, bits.shape[1]))
    for j in range(bits.shape[1]):
        # MI between each hidden dim and bit j
        mi_vals = mutual_info_classif(h_selected, bits[:, j], random_state=42)
        mi_matrix[:, j] = mi_vals

    return mi_matrix, top_dims


def compute_mi_count(h_t, counts, n_dims=100):
    """Compute MI between top-variance hidden dims and decimal count."""
    variances = h_t.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:n_dims]
    h_selected = h_t[:, top_dims]

    mi_count = mutual_info_classif(h_selected, counts, random_state=42)
    return mi_count, top_dims


def bit_subspace_overlap(mi_matrix, top_k=20):
    """Compute Jaccard overlap between top-k MI dimensions for each bit pair."""
    n_bits = mi_matrix.shape[1]
    overlaps = {}
    bit_top_dims = {}

    for j in range(n_bits):
        top_j = set(np.argsort(mi_matrix[:, j])[::-1][:top_k])
        bit_top_dims[j] = sorted(top_j)

    for j1 in range(n_bits):
        for j2 in range(j1 + 1, n_bits):
            s1 = set(bit_top_dims[j1])
            s2 = set(bit_top_dims[j2])
            jaccard = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
            overlaps[f"bit{j1}_bit{j2}"] = {
                "jaccard": float(jaccard),
                "intersection": len(s1 & s2),
                "union": len(s1 | s2),
            }

    return overlaps, bit_top_dims


def block_diagonal_score(mi_matrix):
    """Measure how block-diagonal the MI matrix is.

    Cluster hidden dims by their MI profile, then measure
    within-cluster vs between-cluster MI concentration.
    """
    n_dims, n_bits = mi_matrix.shape

    # Normalize MI matrix rows to unit vectors for clustering
    norms = np.linalg.norm(mi_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    mi_normed = mi_matrix / norms

    # Hierarchical clustering on MI profiles
    # Use correlation distance
    if n_dims < 3:
        return {"error": "too few dims"}

    Z = linkage(mi_normed, method='ward')

    # Try clustering into n_bits clusters
    labels = fcluster(Z, t=n_bits, criterion='maxclust')

    # For each cluster, find its dominant bit (highest mean MI)
    cluster_bit_assignment = {}
    within_mi = 0
    total_mi = mi_matrix.sum()

    for c in range(1, n_bits + 1):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cluster_mi = mi_matrix[mask].sum(axis=0)  # MI per bit for this cluster
        dominant_bit = int(np.argmax(cluster_mi))
        cluster_bit_assignment[int(c)] = {
            "dominant_bit": dominant_bit,
            "n_dims": int(mask.sum()),
            "mi_per_bit": cluster_mi.tolist(),
        }
        within_mi += cluster_mi[dominant_bit]

    score = float(within_mi / total_mi) if total_mi > 0 else 0

    return {
        "block_diagonal_score": score,
        "n_clusters": n_bits,
        "cluster_assignments": cluster_bit_assignment,
        "interpretation": "1.0 = perfect block-diagonal, 0.25 = uniform (no structure)",
    }


def specialization_index(mi_matrix):
    """For each hidden dim, compute how specialized it is to one bit.

    Specialization = max(MI_bits) / sum(MI_bits) for each dim.
    1.0 = perfectly specialized to one bit.
    0.25 = uniform across 4 bits.
    """
    row_sums = mi_matrix.sum(axis=1)
    row_maxs = mi_matrix.max(axis=1)

    # Only compute for dims with nonzero MI
    nonzero = row_sums > 0
    specs = np.zeros(len(mi_matrix))
    specs[nonzero] = row_maxs[nonzero] / row_sums[nonzero]

    return {
        "mean_specialization": float(specs[nonzero].mean()) if nonzero.any() else 0,
        "median_specialization": float(np.median(specs[nonzero])) if nonzero.any() else 0,
        "std_specialization": float(specs[nonzero].std()) if nonzero.any() else 0,
        "n_specialized_dims": int((specs > 0.5).sum()),
        "n_generalist_dims": int((specs <= 0.5).sum() & nonzero.sum()),
        "histogram": {
            "bins": [0, 0.3, 0.5, 0.7, 0.9, 1.0],
            "counts": [int(((specs >= lo) & (specs < hi)).sum())
                      for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]],
        },
    }


def conditional_mi_test(h_t, bits, n_dims=50):
    """Test conditional independence: I(h; bit_i | bit_j).

    If bits are independently encoded, conditioning on bit_j should
    not reduce MI with bit_i.
    """
    variances = h_t.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:n_dims]
    h_selected = h_t[:, top_dims]

    n_bits = bits.shape[1]
    results = {}

    for i in range(n_bits):
        for j in range(n_bits):
            if i == j:
                continue

            # Compute MI(h, bit_i) for each value of bit_j
            mi_given_0 = []
            mi_given_1 = []

            mask_0 = bits[:, j] == 0
            mask_1 = bits[:, j] == 1

            if mask_0.sum() > 100 and mask_1.sum() > 100:
                mi_0 = mutual_info_classif(h_selected[mask_0], bits[mask_0, i], random_state=42)
                mi_1 = mutual_info_classif(h_selected[mask_1], bits[mask_1, i], random_state=42)

                results[f"I(h;bit{i}|bit{j}=0)"] = float(mi_0.mean())
                results[f"I(h;bit{i}|bit{j}=1)"] = float(mi_1.mean())
                results[f"I(h;bit{i})_marginal"] = float(
                    mutual_info_classif(h_selected, bits[:, i], random_state=42).mean()
                )

    return results


def main():
    print("Loading battery data...")
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']      # (13280, 512)
    counts = data['counts']  # (13280,)
    bits = data['bits']      # (13280, 4)

    print(f"  h_t: {h_t.shape}, counts: {counts.shape}, bits: {bits.shape}")

    results = {}

    # --- Experiment 1: MI matrix (top 100 dims x 4 bits) ---
    print("\n[1] Computing MI matrix (100 dims x 4 bits)...")
    mi_matrix, top_dims = compute_mi_matrix(h_t, bits, n_dims=100)

    # Summary stats per bit
    for j in range(4):
        col = mi_matrix[:, j]
        print(f"  Bit {j}: mean MI={col.mean():.4f}, max MI={col.max():.4f}, "
              f"n_dims(MI>0.1)={int((col > 0.1).sum())}")

    results["mi_matrix_summary"] = {
        "n_dims": 100,
        "n_bits": 4,
        "per_bit": {
            f"bit{j}": {
                "mean_mi": float(mi_matrix[:, j].mean()),
                "max_mi": float(mi_matrix[:, j].max()),
                "std_mi": float(mi_matrix[:, j].std()),
                "n_dims_above_0.1": int((mi_matrix[:, j] > 0.1).sum()),
                "n_dims_above_0.3": int((mi_matrix[:, j] > 0.3).sum()),
                "top5_dims": [int(d) for d in np.argsort(mi_matrix[:, j])[::-1][:5]],
                "top5_mi": mi_matrix[np.argsort(mi_matrix[:, j])[::-1][:5], j].tolist(),
            }
            for j in range(4)
        },
        "total_mi": float(mi_matrix.sum()),
    }

    # --- Experiment 2: MI with count vs bits ---
    print("\n[2] MI(h, count) vs MI(h, bits)...")
    mi_count, _ = compute_mi_count(h_t, counts, n_dims=100)
    mi_bits_sum = mi_matrix.sum(axis=1)  # Sum MI across bits per dim

    from scipy.stats import spearmanr
    corr, p = spearmanr(mi_count, mi_bits_sum)
    print(f"  Spearman(MI_count, sum_MI_bits) = {corr:.3f} (p={p:.1e})")
    print(f"  Mean MI(h,count)={mi_count.mean():.4f}, Mean sum MI(h,bits)={mi_bits_sum.mean():.4f}")

    results["count_vs_bits_mi"] = {
        "spearman_r": float(corr),
        "spearman_p": float(p),
        "mean_mi_count": float(mi_count.mean()),
        "mean_mi_bits_sum": float(mi_bits_sum.mean()),
        "max_mi_count": float(mi_count.max()),
        "max_mi_bits_sum": float(mi_bits_sum.max()),
    }

    # --- Experiment 3: Block-diagonal score ---
    print("\n[3] Block-diagonal structure...")
    block_result = block_diagonal_score(mi_matrix)
    print(f"  Block-diagonal score: {block_result.get('block_diagonal_score', 'N/A'):.3f}")
    if 'cluster_assignments' in block_result:
        for c, info in block_result['cluster_assignments'].items():
            print(f"    Cluster {c}: dominant bit={info['dominant_bit']}, "
                  f"n_dims={info['n_dims']}")
    results["block_diagonal"] = block_result

    # --- Experiment 4: Specialization index ---
    print("\n[4] Specialization index...")
    spec = specialization_index(mi_matrix)
    print(f"  Mean specialization: {spec['mean_specialization']:.3f}")
    print(f"  Median specialization: {spec['median_specialization']:.3f}")
    print(f"  Specialized dims (>0.5): {spec['n_specialized_dims']}")
    results["specialization"] = spec

    # --- Experiment 5: Bit subspace overlap ---
    print("\n[5] Bit subspace overlap (top-20 dims)...")
    overlaps, bit_top_dims = bit_subspace_overlap(mi_matrix, top_k=20)
    for pair, info in overlaps.items():
        print(f"  {pair}: Jaccard={info['jaccard']:.3f} "
              f"(intersection={info['intersection']}, union={info['union']})")
    results["subspace_overlap"] = overlaps

    # --- Experiment 6: Conditional MI ---
    print("\n[6] Conditional MI test (50 dims)...")
    cond_mi = conditional_mi_test(h_t, bits, n_dims=50)
    results["conditional_mi"] = cond_mi

    # Print summary
    print("\n  Sample conditional MI values:")
    for key in sorted(cond_mi.keys())[:6]:
        print(f"    {key} = {cond_mi[key]:.4f}")

    # --- Experiment 7: LSB signal concentration ---
    print("\n[7] LSB signal concentration...")
    # Compare MI for bit 0 (LSB, flips every step) vs bit 3 (MSB, flips rarely)
    lsb_mi = mi_matrix[:, 0]
    msb_mi = mi_matrix[:, 3]
    results["lsb_vs_msb"] = {
        "bit0_mean_mi": float(lsb_mi.mean()),
        "bit3_mean_mi": float(msb_mi.mean()),
        "bit0_max_mi": float(lsb_mi.max()),
        "bit3_max_mi": float(msb_mi.max()),
        "ratio_mean": float(lsb_mi.mean() / msb_mi.mean()) if msb_mi.mean() > 0 else float('inf'),
        "ratio_max": float(lsb_mi.max() / msb_mi.max()) if msb_mi.max() > 0 else float('inf'),
    }
    print(f"  Bit 0 (LSB) mean MI: {lsb_mi.mean():.4f}, max: {lsb_mi.max():.4f}")
    print(f"  Bit 3 (MSB) mean MI: {msb_mi.mean():.4f}, max: {msb_mi.max():.4f}")
    print(f"  LSB/MSB ratio (mean): {results['lsb_vs_msb']['ratio_mean']:.2f}x")

    # Save results
    out_path = OUT_DIR / "mi_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
