#!/usr/bin/env python3
"""
Full Analysis Battery + Cross-Modal Comparison for Symbolic Binary Specialist
===============================================================================
Runs the complete Anim analysis battery on a trained symbolic RSSM checkpoint,
then compares representations against the physical binary specialist.

Usage:
    python eval_symbolic_battery.py <checkpoint_path> [--env clean|rich] [--physical_battery <path>]
    python eval_symbolic_battery.py --three-way <ckpt_A> <ckpt_B> [--physical_battery <path>]

Outputs:
    artifacts/symbolic_binary_s{seed}/
        battery_results.json     — all metrics
        battery_data.npz         — h_t, counts, bits, etc.
        cross_modal.json         — cross-modal comparison metrics
        cross_modal_data.npz     — RDMs, CKA, stitching results
        figures/                  — publication-quality figures
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from symbolic_binary_env import (
    SymbolicBinaryEnv, NUM_BITS, MAX_COUNT, int_to_bits, carry_depth,
)
from symbolic_binary_env_rich import RichSymbolicBinaryEnv
from symbolic_rssm import SymbolicRSSM, DETER_DIM, STOCH_FLAT, FEAT_DIM
from train_symbolic import make_env

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 9

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

# Default paths
PHYSICAL_BATTERY = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")


# ---------------------------------------------------------------------------
# 1. Data collection
# ---------------------------------------------------------------------------

def collect_hidden_states(model, device, n_cycles=2, n_episodes=10, seed=42,
                          env_type="clean"):
    """Collect h_t (deter) from symbolic specialist across multiple episodes."""
    model.eval()

    all_h_t, all_counts, all_bits = [], [], []
    all_carry, all_ep_ids, all_timesteps = [], [], []

    for ep_idx in range(n_episodes):
        env = make_env(env_type, steps_per_count=20, n_cycles=n_cycles,
                       seed=seed + ep_idx)
        obs = env.reset()
        h, z = model.initial_state(1, device)
        action = torch.zeros(1, 1, device=device)
        done = False
        step = 0

        while not done:
            tokens = torch.tensor(obs["tokens"], dtype=torch.long,
                                 device=device).unsqueeze(0)
            if obs["is_first"]:
                h, z = model.initial_state(1, device)

            with torch.no_grad():
                out = model.observe_step(tokens, action, h, z)

            h = out["deter"]
            z = out["stoch"]

            meta = env.metadata
            count = meta["decimal_count"]

            all_h_t.append(h.squeeze(0).cpu().numpy())
            all_counts.append(count)
            all_bits.append(int_to_bits(count))
            all_carry.append(carry_depth(count))
            all_ep_ids.append(ep_idx)
            all_timesteps.append(step)

            obs, _, done, _ = env.step(0)
            step += 1

        print(f"  Episode {ep_idx+1}/{n_episodes}: {step} steps", end="\r")

    print()
    return {
        "h_t": np.stack(all_h_t).astype(np.float32),
        "counts": np.array(all_counts, dtype=np.int32),
        "bits": np.stack(all_bits).astype(np.int32),
        "carry_depth": np.array(all_carry, dtype=np.int32),
        "episode_ids": np.array(all_ep_ids, dtype=np.int32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# 2. Standard battery
# ---------------------------------------------------------------------------

def compute_centroids(h_t, counts, max_count=MAX_COUNT):
    centroids, valid_counts = [], []
    for c in range(max_count):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def run_standard_battery(h_t, counts, bits, carry):
    """Run full measurement battery matching physical specialist."""
    results = {}
    n_samples = len(counts)
    centroids, valid_counts = compute_centroids(h_t, counts)
    n = len(valid_counts)

    # --- Linear probes ---
    # Count probe (Ridge regression)
    ridge = Ridge(alpha=1.0)
    cv_scores = cross_val_score(ridge, h_t, counts, cv=5, scoring="r2")
    results["count_probe_r2"] = float(np.mean(cv_scores))

    ridge.fit(h_t, counts)
    preds = ridge.predict(h_t)
    results["count_probe_exact"] = float(np.mean(np.round(preds).astype(int) == counts))
    results["count_probe_mae"] = float(np.mean(np.abs(preds - counts)))

    # Per-bit probes
    for b in range(NUM_BITS):
        ridge_b = Ridge(alpha=1.0)
        ridge_b.fit(h_t, bits[:, b])
        bit_preds = (ridge_b.predict(h_t) > 0.5).astype(int)
        results[f"bit_probe_{b}_accuracy"] = float(np.mean(bit_preds == bits[:, b]))
        results[f"bit_probe_{b}_weight_norm"] = float(np.linalg.norm(ridge_b.coef_))

    # Probe SNR
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(h_t, counts)
    probe_w = ridge_full.coef_ / (np.linalg.norm(ridge_full.coef_) + 1e-12)
    projections = h_t @ probe_w
    unique_counts = np.unique(counts)
    per_count_means, per_count_vars = [], []
    for c in unique_counts:
        mask = counts == c
        proj_c = projections[mask]
        if len(proj_c) > 1:
            per_count_means.append(np.mean(proj_c))
            per_count_vars.append(np.var(proj_c))
    between_var = np.var(per_count_means)
    within_var = np.mean(per_count_vars) if per_count_vars else 1e-12
    results["probe_snr"] = float(between_var / within_var) if within_var > 0 else float("inf")

    # Carry probe
    if len(np.unique(carry)) > 1:
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            carry_binary = (carry > 0).astype(int)
            lr.fit(h_t, carry_binary)
            results["carry_probe_accuracy"] = float(lr.score(h_t, carry_binary))
        except Exception:
            results["carry_probe_accuracy"] = None

    # --- GHE + arc-length R² ---
    best_ghe, best_geo = None, None
    for k in [4, 5, 6, 7, 8]:
        try:
            A = kneighbors_graph(centroids, n_neighbors=min(k, n-1), mode="distance")
            A = 0.5 * (A + A.T)
            geo = shortest_path(A, directed=False)
            if np.any(np.isinf(geo)):
                continue
            consec = [geo[i, i+1] for i in range(n-1)]
            ghe = np.std(consec) / np.mean(consec)
            if best_ghe is None or ghe < best_ghe:
                best_ghe = ghe
                best_geo = consec
        except Exception:
            continue

    results["ghe"] = float(best_ghe) if best_ghe is not None else None

    if best_geo:
        arc = np.cumsum([0] + best_geo)
        coeffs = np.polyfit(np.arange(n), arc, 1)
        fit = np.polyval(coeffs, np.arange(n))
        ss_res = np.sum((arc - fit)**2)
        ss_tot = np.sum((arc - np.mean(arc))**2)
        results["arc_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    else:
        results["arc_r2"] = None

    # --- Topology ---
    try:
        from ripser import ripser
        rips = ripser(centroids, maxdim=1)
        h0, h1 = rips["dgms"][0], rips["dgms"][1]
        results["beta_0"] = int(np.sum(h0[:, 1] == np.inf))
        results["beta_1"] = int(np.sum(h1[:, 1] == np.inf)) if len(h1) > 0 else 0
    except ImportError:
        results["beta_0"] = None
        results["beta_1"] = None

    # --- RSA ---
    rdm_agent = squareform(pdist(centroids, metric="euclidean"))
    # Ordinal RSA
    rdm_ordinal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu = np.triu_indices(n, k=1)
    rsa_ord, _ = spearmanr(rdm_agent[triu], rdm_ordinal[triu])
    results["rsa_ordinal"] = float(rsa_ord)

    # Hamming RSA
    hamming_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bi = int_to_bits(valid_counts[i])
            bj = int_to_bits(valid_counts[j])
            hamming_matrix[i, j] = np.sum(bi != bj)
    rsa_ham, _ = spearmanr(rdm_agent[triu], hamming_matrix[triu])
    results["rsa_hamming"] = float(rsa_ham)

    # Pairwise R² (Hamming vs Euclidean)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(hamming_matrix[triu].reshape(-1, 1), rdm_agent[triu])
    results["pairwise_r2_hamming"] = float(lr.score(
        hamming_matrix[triu].reshape(-1, 1), rdm_agent[triu]))
    lr.fit(rdm_ordinal[triu].reshape(-1, 1), rdm_agent[triu])
    results["pairwise_r2_decimal"] = float(lr.score(
        rdm_ordinal[triu].reshape(-1, 1), rdm_agent[triu]))

    # --- PCA ---
    pca = PCA(n_components=min(10, n))
    pca.fit(centroids)
    results["pca_variance_explained"] = [float(v) for v in pca.explained_variance_ratio_]
    results["pca_90pct_dims"] = int(np.searchsorted(
        np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)

    # --- Effective rank ---
    cov = np.cov(h_t.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    probs = eigenvalues / eigenvalues.sum()
    results["erank"] = float(np.exp(-np.sum(probs * np.log(probs + 1e-30))))

    # --- DCI (simplified) ---
    # Fit 4 bit probes, measure disentanglement
    importance_matrix = np.zeros((4, DETER_DIM))
    for b in range(NUM_BITS):
        ridge_b = Ridge(alpha=1.0)
        ridge_b.fit(h_t, bits[:, b])
        importance_matrix[b] = np.abs(ridge_b.coef_)

    # DCI disentanglement: for each dim, how concentrated is its importance across factors?
    dim_total = importance_matrix.sum(axis=0)
    dim_total[dim_total < 1e-12] = 1e-12
    dim_probs = importance_matrix / dim_total[None, :]
    dim_entropy = -np.sum(dim_probs * np.log(dim_probs + 1e-30), axis=0)
    max_entropy = np.log(NUM_BITS)
    dci_d = 1 - np.mean(dim_entropy / max_entropy)
    results["dci_disentanglement"] = float(dci_d)

    # DCI compactness: for each factor, how concentrated is its importance across dims?
    factor_total = importance_matrix.sum(axis=1)
    factor_total[factor_total < 1e-12] = 1e-12
    factor_probs = importance_matrix / factor_total[:, None]
    factor_entropy = -np.sum(factor_probs * np.log(factor_probs + 1e-30), axis=1)
    max_dim_entropy = np.log(DETER_DIM)
    dci_c = 1 - np.mean(factor_entropy / max_dim_entropy)
    results["dci_compactness"] = float(dci_c)

    # --- NN accuracy ---
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(h_t)
    _, indices = nn.kneighbors(h_t)
    nn_counts = counts[indices[:, 1]]
    adjacent = np.abs(nn_counts.astype(int) - counts.astype(int)) <= 1
    results["nn_accuracy"] = float(np.mean(adjacent))

    results["n_samples"] = n_samples
    results["n_counts"] = n
    results["rdm_agent"] = rdm_agent.tolist()

    return results, centroids, valid_counts


# ---------------------------------------------------------------------------
# 3. Cross-modal comparison
# ---------------------------------------------------------------------------

def run_cross_modal(h_t_sym, counts_sym, bits_sym,
                    h_t_phys, counts_phys, bits_phys):
    """Compare symbolic and physical specialist representations."""
    results = {}

    # Compute centroids for both
    cent_sym, vc_sym = compute_centroids(h_t_sym, counts_sym)
    cent_phys, vc_phys = compute_centroids(h_t_phys, counts_phys)

    # Find common counts
    common = np.intersect1d(vc_sym, vc_phys)
    n = len(common)
    print(f"  Cross-modal comparison on {n} common counts: {common.tolist()}")

    # Align centroids to common counts
    idx_sym = [np.where(vc_sym == c)[0][0] for c in common]
    idx_phys = [np.where(vc_phys == c)[0][0] for c in common]
    cs = cent_sym[idx_sym]
    cp = cent_phys[idx_phys]

    # --- RDM comparison ---
    rdm_sym = squareform(pdist(cs, metric="euclidean"))
    rdm_phys = squareform(pdist(cp, metric="euclidean"))
    triu = np.triu_indices(n, k=1)
    rho, p = spearmanr(rdm_sym[triu], rdm_phys[triu])
    results["rdm_spearman_rho"] = float(rho)
    results["rdm_spearman_p"] = float(p)

    # --- RSA with Hamming model (for each specialist) ---
    hamming = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hamming[i, j] = np.sum(int_to_bits(common[i]) != int_to_bits(common[j]))
    rsa_ham_sym, _ = spearmanr(rdm_sym[triu], hamming[triu])
    rsa_ham_phys, _ = spearmanr(rdm_phys[triu], hamming[triu])
    results["rsa_hamming_symbolic"] = float(rsa_ham_sym)
    results["rsa_hamming_physical"] = float(rsa_ham_phys)

    # --- Probe direction cosine ---
    # Train per-bit probes on each specialist's full h_t
    probe_cosines = []
    for b in range(NUM_BITS):
        ridge_sym = Ridge(alpha=1.0)
        ridge_sym.fit(h_t_sym, bits_sym[:, b])
        w_sym = ridge_sym.coef_ / (np.linalg.norm(ridge_sym.coef_) + 1e-12)

        ridge_phys = Ridge(alpha=1.0)
        ridge_phys.fit(h_t_phys, bits_phys[:, b])
        w_phys = ridge_phys.coef_ / (np.linalg.norm(ridge_phys.coef_) + 1e-12)

        cos = float(np.dot(w_sym, w_phys))
        probe_cosines.append(cos)
        results[f"probe_cosine_bit{b}"] = cos

    results["probe_cosine_mean"] = float(np.mean(np.abs(probe_cosines)))

    # --- Model stitching (affine map: symbolic → physical) ---
    # Align by count label
    aligned_sym, aligned_phys = [], []
    for c in common:
        mask_s = counts_sym == c
        mask_p = counts_phys == c
        n_min = min(mask_s.sum(), mask_p.sum())
        if n_min > 0:
            aligned_sym.append(h_t_sym[mask_s][:n_min])
            aligned_phys.append(h_t_phys[mask_p][:n_min])
    if aligned_sym:
        X = np.concatenate(aligned_sym)
        Y = np.concatenate(aligned_phys)
        # Split 80/20
        n_train = int(0.8 * len(X))
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X[:n_train], Y[:n_train])
        r2_train = lr.score(X[:n_train], Y[:n_train])
        r2_test = lr.score(X[n_train:], Y[n_train:])
        results["stitching_r2_train"] = float(r2_train)
        results["stitching_r2_test"] = float(r2_test)
        results["stitching_n_samples"] = len(X)

    # --- CKA (linear) ---
    # Align by count: use count-conditional means
    cka = linear_cka(cs, cp)
    results["cka_centroids"] = float(cka)

    # Also on full aligned data (subsampled for memory)
    if aligned_sym:
        max_n = 5000
        if len(X) > max_n:
            idx = np.random.RandomState(42).choice(len(X), max_n, replace=False)
            X_sub, Y_sub = X[idx], Y[idx]
        else:
            X_sub, Y_sub = X, Y
        results["cka_full"] = float(linear_cka(X_sub, Y_sub))

    # --- Eigenvalue shape comparison ---
    # Compare sorted eigenvalue spectra of covariance matrices
    cov_sym = np.cov(h_t_sym.T)
    cov_phys = np.cov(h_t_phys.T)
    eig_sym = np.sort(np.linalg.eigvalsh(cov_sym))[::-1]
    eig_phys = np.sort(np.linalg.eigvalsh(cov_phys))[::-1]
    # Normalize to sum to 1
    eig_sym_norm = eig_sym / (eig_sym.sum() + 1e-12)
    eig_phys_norm = eig_phys / (eig_phys.sum() + 1e-12)
    # Spearman correlation of eigenvalue magnitudes
    min_len = min(len(eig_sym_norm), len(eig_phys_norm))
    eig_rho, _ = spearmanr(eig_sym_norm[:min_len], eig_phys_norm[:min_len])
    results["eigenvalue_shape_rho"] = float(eig_rho)

    return results, {
        "rdm_sym": rdm_sym,
        "rdm_phys": rdm_phys,
        "hamming_matrix": hamming,
        "common_counts": common,
        "centroids_sym": cs,
        "centroids_phys": cp,
        "eig_sym": eig_sym_norm[:50],
        "eig_phys": eig_phys_norm[:50],
        "probe_cosines": np.array(probe_cosines),
    }


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment (Kornblith et al. 2019)."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------

def make_figures(results, cross_results, cross_data, output_dir):
    """Generate publication-quality figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if cross_data is None:
        return

    # --- RDM side-by-side ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    common = cross_data["common_counts"]

    im0 = axes[0].imshow(cross_data["rdm_sym"], cmap="viridis")
    axes[0].set_title("Symbolic Specialist RDM")
    axes[0].set_xticks(range(len(common)))
    axes[0].set_xticklabels(common, fontsize=6)
    axes[0].set_yticks(range(len(common)))
    axes[0].set_yticklabels(common, fontsize=6)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(cross_data["rdm_phys"], cmap="viridis")
    axes[1].set_title("Physical Specialist RDM")
    axes[1].set_xticks(range(len(common)))
    axes[1].set_xticklabels(common, fontsize=6)
    axes[1].set_yticks(range(len(common)))
    axes[1].set_yticklabels(common, fontsize=6)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(cross_data["hamming_matrix"], cmap="viridis")
    axes[2].set_title("Hamming Distance (theoretical)")
    axes[2].set_xticks(range(len(common)))
    axes[2].set_xticklabels(common, fontsize=6)
    axes[2].set_yticks(range(len(common)))
    axes[2].set_yticklabels(common, fontsize=6)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    rho = cross_results.get("rdm_spearman_rho", "?")
    fig.suptitle(f"Cross-Modal RDM Comparison (Spearman ρ = {rho:.3f})" if isinstance(rho, float) else "RDMs")
    plt.tight_layout()
    plt.savefig(fig_dir / "rdm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Eigenvalue comparison ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(cross_data["eig_sym"], "b-", label="Symbolic", alpha=0.8)
    ax.semilogy(cross_data["eig_phys"], "r-", label="Physical", alpha=0.8)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Normalized eigenvalue")
    ax.set_title(f"Eigenvalue Shape Comparison (ρ = {cross_results.get('eigenvalue_shape_rho', '?'):.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eigenvalue_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Probe cosine matrix ---
    cosines = cross_data["probe_cosines"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(NUM_BITS), np.abs(cosines), color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_xticks(range(NUM_BITS))
    ax.set_xticklabels([f"Bit {i}" for i in range(NUM_BITS)])
    ax.set_ylabel("|Cosine similarity|")
    ax.set_title(f"Probe Direction Cosines (mean |cos| = {np.mean(np.abs(cosines)):.3f})")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="Random baseline (~0.05)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "probe_cosines.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Symbolic binary specialist evaluation battery")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file or directory")
    parser.add_argument("--env", type=str, default="clean", choices=["clean", "rich"],
                       help="Environment variant (clean=A, rich=B)")
    parser.add_argument("--physical_battery", type=str, default=str(PHYSICAL_BATTERY),
                       help="Path to physical specialist battery.npz")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--n_cycles", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "latest.pt"

    env_type = args.env

    print("=" * 60)
    print(f"Symbolic Binary Specialist — Full Analysis Battery (env={env_type})")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = SymbolicRSSM().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded model from step {ckpt.get('step', '?')}")

    # Output directory
    output_dir = ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect hidden states
    print(f"\nCollecting hidden states ({args.n_episodes} episodes, {args.n_cycles} cycles)...")
    data = collect_hidden_states(model, device, n_cycles=args.n_cycles,
                                n_episodes=args.n_episodes, seed=args.seed,
                                env_type=env_type)
    print(f"  Collected {len(data['h_t'])} timesteps")

    # Save battery data
    np.savez(output_dir / "battery_data.npz", **data)

    # Run standard battery
    print("\nRunning standard battery...")
    results, centroids, valid_counts = run_standard_battery(
        data["h_t"], data["counts"], data["bits"], data["carry_depth"])

    for k, v in sorted(results.items()):
        if isinstance(v, (int, float)) and v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save battery results
    with open(output_dir / "battery_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Cross-modal comparison
    cross_results, cross_data = None, None
    phys_path = Path(args.physical_battery)
    if phys_path.exists():
        print(f"\nRunning cross-modal comparison with {phys_path}...")
        phys = np.load(phys_path, allow_pickle=True)
        h_t_phys = phys["h_t"]
        counts_phys = phys["counts"]
        bits_phys = phys["bits"]

        print(f"  Physical specialist: {len(h_t_phys)} timesteps, "
              f"counts {counts_phys.min()}-{counts_phys.max()}")

        cross_results, cross_data = run_cross_modal(
            data["h_t"], data["counts"], data["bits"],
            h_t_phys, counts_phys, bits_phys,
        )

        print("\n  Cross-modal results:")
        for k, v in sorted(cross_results.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

        # Save
        with open(output_dir / "cross_modal.json", "w") as f:
            json.dump(cross_results, f, indent=2)

        # Save data arrays
        save_data = {k: v for k, v in cross_data.items() if isinstance(v, np.ndarray)}
        np.savez(output_dir / "cross_modal_data.npz", **save_data)
    else:
        print(f"\n  Physical battery not found at {phys_path}, skipping cross-modal comparison")

    # Generate figures
    print("\nGenerating figures...")
    make_figures(results, cross_results, cross_data, output_dir)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — Symbolic Binary Specialist")
    print("=" * 60)
    print(f"  Count exact accuracy:  {results['count_probe_exact']:.4f}")
    bit_accs = [f"{results[f'bit_probe_{b}_accuracy']:.3f}" for b in range(NUM_BITS)]
    print(f"  Per-bit accuracy:      {bit_accs}")
    print(f"  Probe SNR:             {results['probe_snr']:.1f}")
    print(f"  Effective rank:        {results['erank']:.1f}")
    print(f"  RSA ordinal:           {results['rsa_ordinal']:.4f}")
    print(f"  RSA Hamming:           {results['rsa_hamming']:.4f}")
    print(f"  pR² Hamming:           {results['pairwise_r2_hamming']:.4f}")
    print(f"  pR² Decimal:           {results['pairwise_r2_decimal']:.4f}")
    print(f"  GHE:                   {results['ghe']}")
    print(f"  DCI disentanglement:   {results['dci_disentanglement']:.4f}")
    print(f"  DCI compactness:       {results['dci_compactness']:.4f}")

    if cross_results:
        print(f"\n  --- Cross-Modal ---")
        print(f"  RDM Spearman ρ:        {cross_results['rdm_spearman_rho']:.4f}")
        print(f"  RSA Hamming (sym):     {cross_results['rsa_hamming_symbolic']:.4f}")
        print(f"  RSA Hamming (phys):    {cross_results['rsa_hamming_physical']:.4f}")
        print(f"  Probe cosine (mean):   {cross_results['probe_cosine_mean']:.4f}")
        print(f"  Stitching R² (test):   {cross_results.get('stitching_r2_test', 'N/A')}")
        print(f"  CKA (centroids):       {cross_results['cka_centroids']:.4f}")
        print(f"  Eigenvalue shape ρ:    {cross_results['eigenvalue_shape_rho']:.4f}")

    print(f"\nAll outputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# 6. Three-way comparison (Physical × Symbolic A × Symbolic B)
# ---------------------------------------------------------------------------

def three_way_comparison():
    """Run pairwise cross-modal comparison across all three specialists."""
    parser = argparse.ArgumentParser(description="Three-way specialist comparison")
    parser.add_argument("--ckpt_a", type=str, required=True, help="Symbolic A checkpoint")
    parser.add_argument("--ckpt_b", type=str, required=True, help="Symbolic B checkpoint")
    parser.add_argument("--physical_battery", type=str, default=str(PHYSICAL_BATTERY))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Three-Way Comparison: Physical × Symbolic A × Symbolic B")
    print("=" * 60)

    # Load all three datasets
    # Physical
    phys = np.load(args.physical_battery, allow_pickle=True)
    h_phys, c_phys, b_phys = phys["h_t"], phys["counts"], phys["bits"]
    print(f"  Physical: {len(h_phys)} timesteps")

    # Symbolic A (clean)
    ckpt_a = Path(args.ckpt_a)
    if ckpt_a.is_dir():
        ckpt_a = ckpt_a / "latest.pt"
    model_a = SymbolicRSSM().to(device)
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device)["model_state_dict"])
    model_a.eval()
    data_a = collect_hidden_states(model_a, device, n_episodes=args.n_episodes,
                                   seed=args.seed, env_type="clean")
    print(f"  Symbolic A: {len(data_a['h_t'])} timesteps")

    # Symbolic B (rich)
    ckpt_b = Path(args.ckpt_b)
    if ckpt_b.is_dir():
        ckpt_b = ckpt_b / "latest.pt"
    model_b = SymbolicRSSM().to(device)
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device)["model_state_dict"])
    model_b.eval()
    data_b = collect_hidden_states(model_b, device, n_episodes=args.n_episodes,
                                   seed=args.seed, env_type="rich")
    print(f"  Symbolic B: {len(data_b['h_t'])} timesteps")

    # Run all 3 pairwise comparisons
    pairs = [
        ("Physical ↔ Symbolic A", h_phys, c_phys, b_phys,
         data_a["h_t"], data_a["counts"], data_a["bits"]),
        ("Physical ↔ Symbolic B", h_phys, c_phys, b_phys,
         data_b["h_t"], data_b["counts"], data_b["bits"]),
        ("Symbolic A ↔ Symbolic B", data_a["h_t"], data_a["counts"], data_a["bits"],
         data_b["h_t"], data_b["counts"], data_b["bits"]),
    ]

    all_results = {}
    for name, h1, c1, b1, h2, c2, b2 in pairs:
        print(f"\n  {name}...")
        results, _ = run_cross_modal(h1, c1, b1, h2, c2, b2)
        all_results[name] = results

    # Print comparison table
    print("\n" + "=" * 80)
    print("THREE-WAY COMPARISON TABLE")
    print("=" * 80)
    metrics = ["rdm_spearman_rho", "probe_cosine_mean", "cka_centroids",
               "stitching_r2_test", "eigenvalue_shape_rho"]
    header = f"{'Metric':<25}" + "".join(f"{p[:20]:<22}" for p, *_ in pairs)
    print(header)
    print("-" * 80)
    for m in metrics:
        row = f"{m:<25}"
        for name, *_ in pairs:
            v = all_results[name].get(m)
            row += f"{v:<22.4f}" if isinstance(v, float) else f"{'N/A':<22}"
        print(row)

    # Save
    out = Path(args.output_dir) if args.output_dir else Path(ckpt_a).parent.parent
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "three_way_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out / 'three_way_comparison.json'}")


if __name__ == "__main__":
    if "--three-way" in sys.argv:
        sys.argv.remove("--three-way")
        three_way_comparison()
    else:
        main()
