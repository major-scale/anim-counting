#!/usr/bin/env python3
"""
Curve Type Characterization — What Shape Did the Agent Learn?

Analyzes the geometry of the learned count manifold in DreamerV3's
512-dim GRU hidden state space. Tests for W-curve structure (constant
curvature on flat torus, Klein & Lie 1871).

Usage:
    python curve_characterization.py <eval_dump_dir> [--name seed1_step155K]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def compute_centroids(h_t, counts):
    """Average h_t vectors per count value."""
    unique_counts = sorted(np.unique(counts))
    centroids = []
    for c in unique_counts:
        mask = counts == c
        centroids.append(h_t[mask].mean(axis=0))
    return np.stack(centroids), np.array(unique_counts)


# =====================================================================
# Measurement 1: Menger Curvature Profile
# =====================================================================

def menger_curvature(p1, p2, p3):
    """Curvature of the circle through three points in R^n."""
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)

    v1 = p2 - p1
    v2 = p3 - p1
    cross_mag = np.sqrt(
        max(0, np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) ** 2)
    )
    area = 0.5 * cross_mag

    denom = a * b * c
    if denom < 1e-10:
        return 0.0
    return 4.0 * area / denom


def measure_menger(C):
    """Compute Menger curvature profile along centroids."""
    kappas = []
    for i in range(1, len(C) - 1):
        kappas.append(menger_curvature(C[i - 1], C[i], C[i + 1]))
    kappas = np.array(kappas)

    mean_k = float(kappas.mean())
    std_k = float(kappas.std())
    cv = std_k / mean_k if mean_k > 1e-10 else float("inf")

    return {
        "kappa_mean": mean_k,
        "kappa_std": std_k,
        "kappa_cv": cv,
        "kappa_min": float(kappas.min()),
        "kappa_max": float(kappas.max()),
        "kappa_profile": kappas.tolist(),
    }


# =====================================================================
# Measurement 2: PCA Frequency Decomposition
# =====================================================================

def sinusoid(n, A, f, phi):
    return A * np.sin(2 * np.pi * f * n + phi)


def measure_pca_decomposition(C):
    """Fit sinusoids to PCA component pairs — tests for W-curve structure."""
    n_components = min(20, len(C))
    pca = PCA(n_components=n_components)
    C_pca = pca.fit_transform(C)

    n_values = np.arange(len(C), dtype=float)
    pc_fits = []

    for pc_idx in range(0, n_components - 1, 2):
        pc_even = C_pca[:, pc_idx]
        pc_odd = C_pca[:, pc_idx + 1]

        try:
            popt_even, _ = curve_fit(
                sinusoid, n_values, pc_even,
                p0=[np.std(pc_even), 0.05, 0],
                maxfev=5000,
            )
            residuals_even = pc_even - sinusoid(n_values, *popt_even)
            ss_res = np.sum(residuals_even ** 2)
            ss_tot = np.sum((pc_even - pc_even.mean()) ** 2)
            r2_even = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            popt_odd, _ = curve_fit(
                sinusoid, n_values, pc_odd,
                p0=[np.std(pc_odd), 0.05, 0],
                maxfev=5000,
            )
            residuals_odd = pc_odd - sinusoid(n_values, *popt_odd)
            ss_res = np.sum(residuals_odd ** 2)
            ss_tot = np.sum((pc_odd - pc_odd.mean()) ** 2)
            r2_odd = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            freq_match = abs(abs(popt_even[1]) - abs(popt_odd[1])) < 0.01
            amp_ratio = min(abs(popt_even[0]), abs(popt_odd[0])) / max(abs(popt_even[0]), abs(popt_odd[0]), 1e-10)
            amp_match = amp_ratio > 0.8

            pc_fits.append({
                "pc_pair": [pc_idx, pc_idx + 1],
                "freq_even": float(abs(popt_even[1])),
                "freq_odd": float(abs(popt_odd[1])),
                "amp_even": float(abs(popt_even[0])),
                "amp_odd": float(abs(popt_odd[0])),
                "r2_even": float(r2_even),
                "r2_odd": float(r2_odd),
                "freq_match": bool(freq_match),
                "amp_match": bool(amp_match),
                "variance_explained": float(
                    pca.explained_variance_ratio_[pc_idx]
                    + pca.explained_variance_ratio_[pc_idx + 1]
                ),
            })
        except RuntimeError:
            pc_fits.append({
                "pc_pair": [pc_idx, pc_idx + 1],
                "fit_failed": True,
                "variance_explained": float(
                    pca.explained_variance_ratio_[pc_idx]
                    + pca.explained_variance_ratio_[pc_idx + 1]
                ),
            })

    good_pairs = sum(
        1 for f in pc_fits
        if not f.get("fit_failed")
        and f["r2_even"] > 0.9
        and f["r2_odd"] > 0.9
        and f["freq_match"]
        and f["amp_match"]
    )
    total_var = sum(
        f["variance_explained"]
        for f in pc_fits
        if not f.get("fit_failed")
        and f["r2_even"] > 0.9
        and f["r2_odd"] > 0.9
        and f["freq_match"]
        and f["amp_match"]
    )

    dominant_freq = None
    if pc_fits and not pc_fits[0].get("fit_failed"):
        dominant_freq = pc_fits[0]["freq_even"]

    return {
        "n_w_curve_pairs": good_pairs,
        "total_variance_w_curve": float(total_var),
        "dominant_frequency": dominant_freq,
        "pairs": pc_fits,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


# =====================================================================
# Measurement 3: Discrete Frenet-Serret Frame
# =====================================================================

def measure_frenet(C):
    """Compute discrete curvature and torsion along the curve."""
    T = np.diff(C, axis=0)
    T_norms = np.linalg.norm(T, axis=1, keepdims=True)
    T_hat = T / (T_norms + 1e-10)

    dT = np.diff(T_hat, axis=0)
    kappa_frenet = np.linalg.norm(dT, axis=1)

    N_hat = dT / (np.linalg.norm(dT, axis=1, keepdims=True) + 1e-10)

    dN = np.diff(N_hat, axis=0)
    tau_values = []
    for i in range(len(dN)):
        dn = dN[i]
        t_comp = np.dot(dn, T_hat[i + 1]) * T_hat[i + 1]
        n_comp = np.dot(dn, N_hat[i + 1]) * N_hat[i + 1]
        dn_binormal = dn - t_comp - n_comp
        tau_values.append(np.linalg.norm(dn_binormal))
    tau_values = np.array(tau_values)

    k_mean = float(kappa_frenet.mean())
    k_cv = float(kappa_frenet.std() / k_mean) if k_mean > 1e-10 else float("inf")
    t_mean = float(tau_values.mean()) if len(tau_values) > 0 else 0
    t_cv = float(tau_values.std() / t_mean) if t_mean > 1e-10 else float("inf")

    return {
        "kappa_frenet_mean": k_mean,
        "kappa_frenet_cv": k_cv,
        "kappa_frenet_profile": kappa_frenet.tolist(),
        "tau_mean": t_mean,
        "tau_cv": t_cv,
        "tau_profile": tau_values.tolist(),
        "curvature_constant": k_cv < 0.1,
        "torsion_constant": t_cv < 0.1,
        "is_w_curve": k_cv < 0.1 and t_cv < 0.1,
    }


# =====================================================================
# Measurement 4: Regional GHE Breakdown
# =====================================================================

def _compute_geodesic_distances(C, k=6):
    """Build k-NN graph on centroids and compute pairwise geodesic distances."""
    n = len(C)
    best_geo = None
    for k_try in [4, 5, 6, 7, 8]:
        try:
            A = kneighbors_graph(C, n_neighbors=min(k_try, n - 1), mode="distance")
            A = 0.5 * (A + A.T)
            geo = shortest_path(A, directed=False)
            if not np.any(np.isinf(geo)):
                best_geo = geo
                break
        except Exception:
            continue
    return best_geo


def _euclidean_he_for_region(C, idx):
    """Compute Euclidean HE for a subset of consecutive centroids."""
    sub_C = C[idx]
    displacements = sub_C[1:] - sub_C[:-1]
    if len(displacements) < 2:
        return None
    v_mean = displacements.mean(axis=0)
    v_mag_sq = np.sum(v_mean ** 2)
    if v_mag_sq < 1e-10:
        return float("inf")
    he_per_step = np.sum((displacements - v_mean) ** 2, axis=1) / v_mag_sq
    return float(he_per_step.mean())


def _geodesic_he_for_region(geo_matrix, idx):
    """Compute geodesic HE (CV of consecutive geodesic distances) for a region."""
    consec = []
    for i in range(len(idx) - 1):
        consec.append(geo_matrix[idx[i], idx[i + 1]])
    consec = np.array(consec)
    if len(consec) < 2 or np.mean(consec) < 1e-10:
        return None
    return float(np.std(consec) / np.mean(consec))


def measure_regional_he(C, valid_counts):
    """Compute both Euclidean and Geodesic HE separately for low/mid/high count regions."""
    regions = {"low": (0, 8), "mid": (9, 17), "high": (18, 25)}
    results = {}

    # Euclidean HE (original, potentially confounds curvature with quality)
    for name, (lo, hi) in regions.items():
        mask = (valid_counts >= lo) & (valid_counts <= hi)
        idx = np.where(mask)[0]
        if len(idx) < 3:
            results[f"euclidean_HE_{name}"] = None
            continue
        results[f"euclidean_HE_{name}"] = _euclidean_he_for_region(C, idx)

    he_low = results.get("euclidean_HE_low")
    he_high = results.get("euclidean_HE_high")
    if he_low and he_high and he_low > 0:
        results["euclidean_HE_ratio_high_to_low"] = float(he_high / he_low)
    else:
        results["euclidean_HE_ratio_high_to_low"] = None

    # Geodesic HE (curvature-aware — the metric that matters)
    geo = _compute_geodesic_distances(C)
    if geo is not None:
        for name, (lo, hi) in regions.items():
            mask = (valid_counts >= lo) & (valid_counts <= hi)
            idx = np.where(mask)[0]
            if len(idx) < 3:
                results[f"geodesic_GHE_{name}"] = None
                continue
            results[f"geodesic_GHE_{name}"] = _geodesic_he_for_region(geo, idx)

        ghe_low = results.get("geodesic_GHE_low")
        ghe_high = results.get("geodesic_GHE_high")
        if ghe_low and ghe_high and ghe_low > 0:
            results["geodesic_GHE_ratio_high_to_low"] = float(ghe_high / ghe_low)
        else:
            results["geodesic_GHE_ratio_high_to_low"] = None

        # Full GHE (for reference)
        all_consec = [geo[i, i + 1] for i in range(len(C) - 1)]
        results["geodesic_GHE_full"] = float(np.std(all_consec) / np.mean(all_consec))
    else:
        for name in regions:
            results[f"geodesic_GHE_{name}"] = None
        results["geodesic_GHE_ratio_high_to_low"] = None
        results["geodesic_GHE_full"] = None

    return results


# =====================================================================
# Measurement 5: W-Curve Confidence Score
# =====================================================================

def compute_w_curve_score(menger, pca_decomp, frenet, regional_he):
    """Combine all measurements into a W-curve confidence score."""
    score = 0.0
    max_score = 0.0
    breakdown = {}

    # Menger curvature constancy (0-30)
    max_score += 30
    kappa_cv = menger["kappa_cv"]
    if kappa_cv < 0.05:
        s = 30
    elif kappa_cv < 0.1:
        s = 25
    elif kappa_cv < 0.2:
        s = 15
    elif kappa_cv < 0.3:
        s = 5
    else:
        s = 0
    score += s
    breakdown["menger_curvature"] = {"score": s, "max": 30, "kappa_cv": kappa_cv}

    # PCA sinusoidal decomposition (0-30)
    max_score += 30
    good_pairs = pca_decomp["n_w_curve_pairs"]
    s = min(30, good_pairs * 10)
    score += s
    breakdown["pca_sinusoidal"] = {"score": s, "max": 30, "good_pairs": good_pairs}

    # Frenet torsion constancy (0-20)
    max_score += 20
    tau_cv = frenet["tau_cv"]
    if tau_cv < 0.1:
        s = 20
    elif tau_cv < 0.2:
        s = 15
    elif tau_cv < 0.3:
        s = 8
    else:
        s = 0
    score += s
    breakdown["frenet_torsion"] = {"score": s, "max": 20, "tau_cv": tau_cv}

    # Regional HE uniformity (0-20) — use geodesic GHE ratio (curvature-aware)
    max_score += 20
    ghe_ratio = regional_he.get("geodesic_GHE_ratio_high_to_low")
    eucl_ratio = regional_he.get("euclidean_HE_ratio_high_to_low")
    # Prefer geodesic; fall back to euclidean
    he_ratio = ghe_ratio if ghe_ratio is not None else eucl_ratio
    if he_ratio is not None:
        if he_ratio < 1.2:
            s = 20
        elif he_ratio < 1.5:
            s = 15
        elif he_ratio < 2.0:
            s = 8
        else:
            s = 0
    else:
        s = 10  # can't measure, give neutral score
    score += s
    breakdown["regional_he"] = {
        "score": s, "max": 20,
        "geodesic_ratio": ghe_ratio,
        "euclidean_ratio": eucl_ratio,
        "used_metric": "geodesic" if ghe_ratio is not None else "euclidean",
    }

    confidence = score / max_score

    # Find limiting factor
    min_frac = 1.0
    limiting = None
    for k, v in breakdown.items():
        frac = v["score"] / v["max"]
        if frac < min_frac:
            min_frac = frac
            limiting = k

    if confidence > 0.8:
        classification = "Strong W-curve"
    elif confidence > 0.5:
        classification = "Partial W-curve"
    else:
        classification = "Non-W-curve"

    return {
        "w_curve_confidence": float(confidence),
        "classification": classification,
        "limiting_factor": limiting,
        "score": float(score),
        "max_score": float(max_score),
        "breakdown": breakdown,
    }


# =====================================================================
# Plotting
# =====================================================================

def generate_plots(C, valid_counts, menger, pca_decomp, frenet, regional_he, out_dir):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. Menger curvature vs count
    fig, ax = plt.subplots(figsize=(10, 5))
    kappas = menger["kappa_profile"]
    x = valid_counts[1:-1]  # interior points
    ax.bar(x, kappas, color="steelblue", alpha=0.7)
    ax.axhline(menger["kappa_mean"], color="red", linestyle="--", label=f'mean={menger["kappa_mean"]:.4f}')
    ax.fill_between(
        [x[0] - 0.5, x[-1] + 0.5],
        menger["kappa_mean"] - menger["kappa_std"],
        menger["kappa_mean"] + menger["kappa_std"],
        alpha=0.15, color="red",
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Menger Curvature")
    ax.set_title(f'Menger Curvature Profile (CV={menger["kappa_cv"]:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "menger_curvature.png"), dpi=150)
    plt.close(fig)

    # 2. PC pair scatter plots
    pca = PCA(n_components=min(8, len(C)))
    C_pca = pca.fit_transform(C)
    n_pairs = min(4, C_pca.shape[1] // 2)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    for i in range(n_pairs):
        ax = axes[i]
        sc = ax.scatter(
            C_pca[:, 2 * i], C_pca[:, 2 * i + 1],
            c=valid_counts, cmap="viridis", s=60, edgecolors="k", linewidth=0.5,
        )
        ax.set_xlabel(f"PC{2*i}")
        ax.set_ylabel(f"PC{2*i+1}")
        ax.set_title(f"PC{2*i}-{2*i+1}")
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes[-1], label="Count")
    fig.suptitle("PCA Pair Scatter (colored by count)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_pairs.png"), dpi=150)
    plt.close(fig)

    # 3. Sinusoidal fits on first 8 PCs
    n_pcs = min(8, C_pca.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    n_values = np.arange(len(C), dtype=float)
    for pc_idx in range(n_pcs):
        ax = axes[pc_idx // 4][pc_idx % 4]
        pc_data = C_pca[:, pc_idx]
        ax.plot(valid_counts, pc_data, "o-", markersize=4, label="data")
        try:
            popt, _ = curve_fit(
                sinusoid, n_values, pc_data,
                p0=[np.std(pc_data), 0.05, 0], maxfev=5000,
            )
            fit_vals = sinusoid(n_values, *popt)
            ss_res = np.sum((pc_data - fit_vals) ** 2)
            ss_tot = np.sum((pc_data - pc_data.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.plot(valid_counts, fit_vals, "--", color="red", label=f"sin R²={r2:.2f}")
        except RuntimeError:
            pass
        ax.set_title(f"PC{pc_idx}")
        ax.legend(fontsize=7)
    fig.suptitle("PCA Components with Sinusoidal Fits")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_sinusoidal_fits.png"), dpi=150)
    plt.close(fig)

    # 4. Frenet curvature and torsion
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    kf = frenet["kappa_frenet_profile"]
    ax1.plot(valid_counts[1:-1], kf, "o-", color="steelblue")
    ax1.axhline(frenet["kappa_frenet_mean"], color="red", linestyle="--")
    ax1.set_ylabel("Frenet Curvature κ")
    ax1.set_title(f'Frenet Curvature (CV={frenet["kappa_frenet_cv"]:.3f})')

    tau = frenet["tau_profile"]
    if len(tau) > 0:
        ax2.plot(valid_counts[2:-1], tau, "o-", color="darkorange")
        ax2.axhline(frenet["tau_mean"], color="red", linestyle="--")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Torsion τ")
    ax2.set_title(f'Frenet Torsion (CV={frenet["tau_cv"]:.3f})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frenet_curvature_torsion.png"), dpi=150)
    plt.close(fig)

    # 5. Regional HE — side-by-side Euclidean vs Geodesic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = ["low\n(0-8)", "mid\n(9-17)", "high\n(18-25)"]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    eucl_vals = [
        regional_he.get("euclidean_HE_low", 0) or 0,
        regional_he.get("euclidean_HE_mid", 0) or 0,
        regional_he.get("euclidean_HE_high", 0) or 0,
    ]
    ax1.bar(names, eucl_vals, color=colors)
    ax1.set_ylabel("Euclidean HE")
    ax1.set_title("Euclidean HE (confounds curvature)")

    geo_vals = [
        regional_he.get("geodesic_GHE_low", 0) or 0,
        regional_he.get("geodesic_GHE_mid", 0) or 0,
        regional_he.get("geodesic_GHE_high", 0) or 0,
    ]
    ax2.bar(names, geo_vals, color=colors)
    ax2.set_ylabel("Geodesic GHE")
    ax2.set_title("Geodesic GHE (curvature-aware)")

    fig.suptitle("Regional Breakdown: Euclidean vs Geodesic")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regional_he.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Curve Type Characterization")
    parser.add_argument("eval_dump_dir", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    dump_dir = args.eval_dump_dir
    name = args.name or os.path.basename(dump_dir.rstrip("/"))

    print(f"=" * 60)
    print(f"Curve Type Characterization")
    print(f"=" * 60)
    print(f"  Eval dump: {dump_dir}")
    print(f"  Name: {name}")
    print()

    h_t = np.load(os.path.join(dump_dir, "h_t.npy"))
    counts = np.load(os.path.join(dump_dir, "counts.npy"))
    print(f"  h_t: {h_t.shape}, counts: {counts.shape}")
    print(f"  Count range: {counts.min()}-{counts.max()}, unique: {len(np.unique(counts))}")

    # Centroids
    print("\nComputing centroids...")
    C, valid_counts = compute_centroids(h_t, counts)
    print(f"  {len(valid_counts)} centroids in R^{C.shape[1]}")

    # Measurement 1
    print("\n--- Measurement 1: Menger Curvature ---")
    menger = measure_menger(C)
    print(f"  κ mean={menger['kappa_mean']:.4f}, std={menger['kappa_std']:.4f}, CV={menger['kappa_cv']:.4f}")
    cv_interp = "constant" if menger["kappa_cv"] < 0.1 else "moderate" if menger["kappa_cv"] < 0.3 else "irregular"
    print(f"  Interpretation: {cv_interp} curvature")

    # Measurement 2
    print("\n--- Measurement 2: PCA Frequency Decomposition ---")
    pca_decomp = measure_pca_decomposition(C)
    print(f"  W-curve pairs: {pca_decomp['n_w_curve_pairs']}")
    print(f"  Variance in W-curve pairs: {pca_decomp['total_variance_w_curve']:.3f}")
    if pca_decomp["dominant_frequency"]:
        print(f"  Dominant frequency: {pca_decomp['dominant_frequency']:.4f}")
    for p in pca_decomp["pairs"][:4]:
        if p.get("fit_failed"):
            print(f"    PC{p['pc_pair']}: fit failed")
        else:
            match_str = "✓" if p["freq_match"] and p["amp_match"] else "✗"
            print(
                f"    PC{p['pc_pair']}: R²={p['r2_even']:.2f}/{p['r2_odd']:.2f} "
                f"f={p['freq_even']:.3f}/{p['freq_odd']:.3f} "
                f"A={p['amp_even']:.2f}/{p['amp_odd']:.2f} {match_str}"
            )

    # Measurement 3
    print("\n--- Measurement 3: Frenet-Serret Frame ---")
    frenet = measure_frenet(C)
    print(f"  Curvature: mean={frenet['kappa_frenet_mean']:.4f}, CV={frenet['kappa_frenet_cv']:.4f}")
    print(f"  Torsion:   mean={frenet['tau_mean']:.4f}, CV={frenet['tau_cv']:.4f}")
    print(f"  Curvature constant: {frenet['curvature_constant']}")
    print(f"  Torsion constant:   {frenet['torsion_constant']}")
    print(f"  Is W-curve (Frenet): {frenet['is_w_curve']}")

    # Measurement 4
    print("\n--- Measurement 4: Regional HE (Euclidean + Geodesic) ---")
    regional_he = measure_regional_he(C, valid_counts)
    print("  Euclidean HE (confounds curvature):")
    for k in ["euclidean_HE_low", "euclidean_HE_mid", "euclidean_HE_high", "euclidean_HE_ratio_high_to_low"]:
        v = regional_he.get(k)
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")
    print("  Geodesic GHE (curvature-aware — the metric that matters):")
    for k in ["geodesic_GHE_low", "geodesic_GHE_mid", "geodesic_GHE_high", "geodesic_GHE_ratio_high_to_low", "geodesic_GHE_full"]:
        v = regional_he.get(k)
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")

    # Measurement 5
    print("\n--- Measurement 5: W-Curve Confidence ---")
    summary = compute_w_curve_score(menger, pca_decomp, frenet, regional_he)
    print(f"  Confidence: {summary['w_curve_confidence']:.2f}")
    print(f"  Classification: {summary['classification']}")
    print(f"  Limiting factor: {summary['limiting_factor']}")
    for k, v in summary["breakdown"].items():
        print(f"    {k}: {v['score']}/{v['max']}")

    # Save results
    results = {
        "checkpoint": name,
        "menger": menger,
        "pca_decomposition": pca_decomp,
        "frenet": frenet,
        "regional_he": regional_he,
        "summary": summary,
    }

    out_json = os.path.join(dump_dir, f"curve_characterization.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Plots
    if not args.no_plots:
        plot_dir = os.path.join(dump_dir, "plots")
        generate_plots(C, valid_counts, menger, pca_decomp, frenet, regional_he, plot_dir)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {summary['classification']} (confidence={summary['w_curve_confidence']:.2f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
