#!/usr/bin/env python3
"""
Successor Function Analysis — Visualizing S(n) = n+1 in the RSSM Hidden State
==============================================================================

Analyzes the geometric structure of the counting successor function in
DreamerV3's 512-dimensional hidden state space.

Uses the clean randproj model (highest probe SNR, cleanest linear decodability).

Output: JSON results + matplotlib plots saved to output directory.
"""

import sys
import json
import os
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def collect_episode_trajectories(weights_dir, n_episodes=5, blob_count=25,
                                 randproj=False):
    """Collect full episode trajectories with per-step hidden states and counts."""
    from export_deter_centroids import FastRSSM, load_weights, OBS_SIZE
    from counting_env_pure import CountingWorldEnv

    # Load weights
    wdir = Path(weights_dir)
    with open(wdir / "dreamer_manifest.json") as f:
        manifest = json.load(f)
    with open(wdir / "dreamer_weights.bin", "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"],
                            offset=entry["offset"]).copy()
        weights[name] = arr.reshape(entry["shape"])
    obs_size = weights["enc_linear0_w"].shape[1]

    # Random projection matrix
    proj_matrix = None
    if randproj:
        from scipy.stats import ortho_group
        proj_matrix = ortho_group.rvs(obs_size,
                                       random_state=np.random.RandomState(42_000)).astype(np.float32)

    episodes = []
    for ep in range(n_episodes):
        env = CountingWorldEnv(
            blob_count_min=blob_count, blob_count_max=blob_count,
            conservation=True,
        )
        model = FastRSSM(weights)
        vec = env.reset()
        done = False

        ep_data = {
            "hidden_states": [],
            "counts": [],
            "phases": [],
            "steps": [],
        }
        step = 0

        while not done:
            obs = vec.copy()
            if proj_matrix is not None:
                obs = (proj_matrix @ obs.astype(np.float32))
            obs = obs[:obs_size].astype(np.float32)

            count = int(vec[81])  # grid_filled_raw
            phase = env._state.phase

            deter = model.step(obs, 0.0)
            ep_data["hidden_states"].append(deter.copy())
            ep_data["counts"].append(count)
            ep_data["phases"].append(phase)
            ep_data["steps"].append(step)

            vec, reward, done, info = env.step(0)
            step += 1

        env.close()
        episodes.append(ep_data)
        print(f"  Episode {ep+1}/{n_episodes}: {step} steps, "
              f"max count={max(ep_data['counts'])}")

    return episodes, weights


def compute_centroids(episodes):
    """Compute per-count centroids from all episodes."""
    by_count = defaultdict(list)
    for ep in episodes:
        for h, c in zip(ep["hidden_states"], ep["counts"]):
            by_count[c].append(h)

    centroids = {}
    for c in sorted(by_count.keys()):
        centroids[c] = np.stack(by_count[c]).mean(axis=0)
    return centroids


def analysis_1_transition_profile(episodes, probe_w, probe_b):
    """Plot 1D probe projection over time alongside ground truth count."""
    print("\n=== Analysis 1: Transition Profile ===")

    results = []
    for i, ep in enumerate(episodes[:3]):  # First 3 episodes
        H = np.stack(ep["hidden_states"])
        projections = H @ probe_w + probe_b
        counts = np.array(ep["counts"])

        # Find transition points (where count changes)
        transitions = []
        for t in range(1, len(counts)):
            if counts[t] != counts[t-1]:
                transitions.append({
                    "step": t,
                    "from_count": int(counts[t-1]),
                    "to_count": int(counts[t]),
                })

        # Measure transition characteristics
        for tr in transitions:
            t = tr["step"]
            # Look at window around transition
            window = 50  # steps before/after
            t_start = max(0, t - window)
            t_end = min(len(projections), t + window)
            segment = projections[t_start:t_end]

            # Measure transition duration: how many steps to go from
            # 10% to 90% of the step
            pre_val = projections[max(0, t-5):t].mean() if t > 5 else projections[0]
            post_val = projections[t:min(len(projections), t+20)].mean()
            step_size = post_val - pre_val

            if abs(step_size) > 0.01:
                threshold_10 = pre_val + 0.1 * step_size
                threshold_90 = pre_val + 0.9 * step_size

                # Find 10% and 90% crossing points
                t10 = t  # default
                t90 = t
                for tt in range(t_start, t_end):
                    if step_size > 0:
                        if projections[tt] >= threshold_10 and tt < t10:
                            t10 = tt
                        if projections[tt] >= threshold_90:
                            t90 = tt
                            break
                    else:
                        if projections[tt] <= threshold_10 and tt < t10:
                            t10 = tt
                        if projections[tt] <= threshold_90:
                            t90 = tt
                            break

                tr["transition_duration"] = t90 - t10
                tr["anticipation"] = t - t10  # positive = starts before landing
                tr["step_size_probe"] = float(step_size)

        ep_result = {
            "episode": i,
            "n_steps": len(ep["counts"]),
            "n_transitions": len(transitions),
            "transitions": transitions,
            "projections": projections.tolist(),
            "counts": [int(c) for c in counts],
        }

        # Summary stats
        durations = [t["transition_duration"] for t in transitions
                     if "transition_duration" in t]
        anticipations = [t["anticipation"] for t in transitions
                         if "anticipation" in t]
        if durations:
            ep_result["mean_transition_duration"] = float(np.mean(durations))
            ep_result["std_transition_duration"] = float(np.std(durations))
        if anticipations:
            ep_result["mean_anticipation"] = float(np.mean(anticipations))

        results.append(ep_result)
        print(f"  Ep {i}: {len(transitions)} transitions, "
              f"mean duration={np.mean(durations):.1f} steps" if durations else
              f"  Ep {i}: {len(transitions)} transitions")

    return results


def analysis_2_step_vectors(centroids):
    """Analyze the step vectors Δ_n = μ_{n+1} − μ_n."""
    print("\n=== Analysis 2: Step Vectors (The Successor Function) ===")

    counts = sorted(centroids.keys())
    consecutive = [(c, c+1) for c in counts if c+1 in centroids]

    step_vectors = []
    step_labels = []
    for c_from, c_to in consecutive:
        delta = centroids[c_to] - centroids[c_from]
        step_vectors.append(delta)
        step_labels.append(c_from)

    step_vectors = np.stack(step_vectors)  # (N, 512)
    magnitudes = np.linalg.norm(step_vectors, axis=1)

    # Pairwise cosine similarities
    norms = step_vectors / magnitudes[:, None]
    cosine_matrix = norms @ norms.T

    # Extract upper triangle (excluding diagonal)
    n = len(step_vectors)
    triu_idx = np.triu_indices(n, k=1)
    pairwise_cosines = cosine_matrix[triu_idx]

    # Adjacent cosines (Δ_n vs Δ_{n+1})
    adjacent_cosines = [float(cosine_matrix[i, i+1]) for i in range(n-1)]

    results = {
        "n_step_vectors": n,
        "count_range": [int(step_labels[0]), int(step_labels[-1])],
        "magnitudes": {
            "mean": float(magnitudes.mean()),
            "std": float(magnitudes.std()),
            "cv": float(magnitudes.std() / magnitudes.mean()),
            "min": float(magnitudes.min()),
            "max": float(magnitudes.max()),
            "per_step": [float(m) for m in magnitudes],
        },
        "cosine_similarity": {
            "mean_all_pairs": float(pairwise_cosines.mean()),
            "std_all_pairs": float(pairwise_cosines.std()),
            "min_all_pairs": float(pairwise_cosines.min()),
            "mean_adjacent": float(np.mean(adjacent_cosines)),
            "adjacent_cosines": adjacent_cosines,
        },
        "step_labels": [int(l) for l in step_labels],
    }

    print(f"  {n} step vectors (counts {step_labels[0]}→{step_labels[-1]+1})")
    print(f"  Magnitude: mean={magnitudes.mean():.4f}, CV={magnitudes.std()/magnitudes.mean():.3f}")
    print(f"  Cosine similarity (all pairs): mean={pairwise_cosines.mean():.4f}, "
          f"std={pairwise_cosines.std():.4f}")
    print(f"  Cosine similarity (adjacent): mean={np.mean(adjacent_cosines):.4f}")

    return results, step_vectors


def analysis_3_instantaneous_velocity(episodes, centroids):
    """Analyze velocity vectors during transitions vs centroid step vectors."""
    print("\n=== Analysis 3: Instantaneous Successor Velocity ===")

    counts_sorted = sorted(centroids.keys())
    step_vectors = {}
    for c in counts_sorted:
        if c + 1 in centroids:
            step_vectors[c] = centroids[c + 1] - centroids[c]

    # For each transition, compute alignment of velocity with step vector
    transition_alignments = []
    non_transition_alignments = []

    for ep in episodes:
        H = np.stack(ep["hidden_states"])
        counts = np.array(ep["counts"])

        for t in range(len(H) - 1):
            v_t = H[t+1] - H[t]  # instantaneous velocity
            c = counts[t]
            v_norm = np.linalg.norm(v_t)

            if c in step_vectors and v_norm > 1e-8:
                delta = step_vectors[c]
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-8:
                    cosine = float(np.dot(v_t, delta) / (v_norm * delta_norm))

                    is_transition = counts[t+1] != counts[t]
                    entry = {
                        "cosine": cosine,
                        "velocity_magnitude": float(v_norm),
                        "count": int(c),
                    }

                    if is_transition:
                        transition_alignments.append(entry)
                    else:
                        non_transition_alignments.append(entry)

    # Summary
    trans_cos = [a["cosine"] for a in transition_alignments]
    non_trans_cos = [a["cosine"] for a in non_transition_alignments]

    results = {
        "n_transition_steps": len(trans_cos),
        "n_non_transition_steps": len(non_trans_cos),
        "transition_alignment": {
            "mean_cosine": float(np.mean(trans_cos)) if trans_cos else None,
            "std_cosine": float(np.std(trans_cos)) if trans_cos else None,
            "mean_velocity": float(np.mean([a["velocity_magnitude"]
                                            for a in transition_alignments])) if transition_alignments else None,
        },
        "non_transition_alignment": {
            "mean_cosine": float(np.mean(non_trans_cos)) if non_trans_cos else None,
            "std_cosine": float(np.std(non_trans_cos)) if non_trans_cos else None,
            "mean_velocity": float(np.mean([a["velocity_magnitude"]
                                            for a in non_transition_alignments])) if non_transition_alignments else None,
        },
    }

    if trans_cos and non_trans_cos:
        print(f"  Transition steps:     mean cosine = {np.mean(trans_cos):.4f} "
              f"(±{np.std(trans_cos):.4f}), n={len(trans_cos)}")
        print(f"  Non-transition steps: mean cosine = {np.mean(non_trans_cos):.4f} "
              f"(±{np.std(non_trans_cos):.4f}), n={len(non_trans_cos)}")
        print(f"  Velocity at transitions: {results['transition_alignment']['mean_velocity']:.4f}")
        print(f"  Velocity during seeking: {results['non_transition_alignment']['mean_velocity']:.4f}")

    return results


def analysis_4_step_vector_pca(step_vectors):
    """PCA on step vectors — is the successor one direction?"""
    print("\n=== Analysis 4: Is the Successor One Direction? (PCA on Step Vectors) ===")

    mean = step_vectors.mean(axis=0)
    centered = step_vectors - mean

    # Full eigendecomposition
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Variance explained
    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var
    cumulative = np.cumsum(explained)

    results = {
        "n_vectors": len(step_vectors),
        "dimensionality": step_vectors.shape[1],
        "eigenspectrum_top10": [float(e) for e in explained[:10]],
        "cumulative_top10": [float(c) for c in cumulative[:10]],
        "pc1_explained": float(explained[0]),
        "pc1_pc2_explained": float(cumulative[1]),
        "pcs_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "pcs_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "pcs_for_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
    }

    print(f"  PC1 explains {explained[0]:.1%} of step vector variance")
    print(f"  PC1+PC2 explains {cumulative[1]:.1%}")
    print(f"  PCs for 90%: {results['pcs_for_90pct']}")
    print(f"  PCs for 95%: {results['pcs_for_95pct']}")
    print(f"  PCs for 99%: {results['pcs_for_99pct']}")

    if explained[0] > 0.90:
        print("  → The successor IS essentially one direction in 512-dim space")
    elif explained[0] > 0.70:
        print("  → Mostly one direction, but with significant curvature")
    else:
        print("  → The successor changes direction with count (curved manifold)")

    return results


def analysis_5_linear_reconstruction(centroids):
    """Can count be reconstructed by repeatedly adding the mean step vector?"""
    print("\n=== Analysis 5: Linear Reconstruction from μ_0 + n·Δ_mean ===")

    counts = sorted(centroids.keys())
    if 0 not in centroids:
        print("  No count=0 centroid, skipping")
        return None

    # Compute mean step vector
    deltas = []
    for c in counts:
        if c + 1 in centroids:
            deltas.append(centroids[c + 1] - centroids[c])
    delta_mean = np.stack(deltas).mean(axis=0)

    # Reconstruct
    mu_0 = centroids[0]
    reconstructed = {}
    errors = {}
    for c in counts:
        recon = mu_0 + c * delta_mean
        reconstructed[c] = recon
        error = np.linalg.norm(recon - centroids[c])
        errors[c] = float(error)

    # Also compute centroid-to-centroid distances for normalization
    inter_centroid_dists = []
    for c in counts:
        if c + 1 in centroids:
            inter_centroid_dists.append(
                float(np.linalg.norm(centroids[c+1] - centroids[c])))
    mean_step_dist = np.mean(inter_centroid_dists)

    # Normalized errors (relative to mean step size)
    norm_errors = {c: e / mean_step_dist for c, e in errors.items()}

    results = {
        "mean_step_magnitude": float(np.linalg.norm(delta_mean)),
        "mean_inter_centroid_dist": float(mean_step_dist),
        "reconstruction_errors": errors,
        "normalized_errors": {str(c): float(e) for c, e in norm_errors.items()},
        "mean_normalized_error": float(np.mean(list(norm_errors.values()))),
        "max_normalized_error": float(np.max(list(norm_errors.values()))),
        "error_at_count_5": float(norm_errors.get(5, -1)),
        "error_at_count_10": float(norm_errors.get(10, -1)),
        "error_at_count_15": float(norm_errors.get(15, -1)),
        "error_at_count_20": float(norm_errors.get(20, -1)),
        "error_at_count_25": float(norm_errors.get(25, -1)),
    }

    print(f"  Mean step vector magnitude: {np.linalg.norm(delta_mean):.4f}")
    print(f"  Mean inter-centroid distance: {mean_step_dist:.4f}")
    print(f"  Mean normalized reconstruction error: {results['mean_normalized_error']:.4f}")
    print(f"  Max normalized error: {results['max_normalized_error']:.4f}")
    print(f"  Error growth with count:")
    for c in [0, 5, 10, 15, 20, 25]:
        if c in norm_errors:
            print(f"    count={c:2d}: {norm_errors[c]:.4f} "
                  f"({'< 1 step' if norm_errors[c] < 1.0 else '> 1 step ⚠'})")

    return results


def generate_plots(episodes, centroids, probe_w, probe_b, step_vectors,
                   step_labels, output_dir):
    """Generate matplotlib plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    odir = Path(output_dir)

    # --- Plot 1: Transition profile (1D projection over time) ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for i, ep in enumerate(episodes[:3]):
        H = np.stack(ep["hidden_states"])
        projections = H @ probe_w + probe_b
        counts = np.array(ep["counts"])
        steps = np.arange(len(counts))

        ax = axes[i]
        ax.plot(steps, projections, color="#2196F3", linewidth=0.5, alpha=0.8,
                label="Probe projection")
        ax.plot(steps, counts, color="#FF5722", linewidth=1.0, alpha=0.7,
                label="Ground truth count", linestyle="--")

        # Mark transitions
        for t in range(1, len(counts)):
            if counts[t] != counts[t-1]:
                ax.axvline(t, color="#4CAF50", alpha=0.3, linewidth=0.5)

        ax.set_ylabel("Count / Projection")
        ax.set_title(f"Episode {i+1}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Transition Profile: Probe Projection vs Ground Truth Count",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(odir / "transition_profile.png", dpi=150)
    plt.close()
    print(f"  Saved transition_profile.png")

    # --- Plot 2: Step vector magnitudes and cosines ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    mags = np.linalg.norm(step_vectors, axis=1)
    ax1.bar(step_labels, mags, color="#2196F3", alpha=0.7)
    ax1.axhline(mags.mean(), color="#FF5722", linestyle="--",
                label=f"Mean={mags.mean():.4f}")
    ax1.set_xlabel("Count transition (n → n+1)")
    ax1.set_ylabel("Step vector magnitude ||Δ_n||")
    ax1.set_title("Successor Step Vector Magnitudes")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Adjacent cosines
    norms_mat = step_vectors / mags[:, None]
    adj_cos = [float(np.dot(norms_mat[i], norms_mat[i+1]))
               for i in range(len(norms_mat)-1)]
    ax2.bar(step_labels[:-1], adj_cos, color="#4CAF50", alpha=0.7)
    ax2.axhline(np.mean(adj_cos), color="#FF5722", linestyle="--",
                label=f"Mean={np.mean(adj_cos):.4f}")
    ax2.set_xlabel("Count transition pair (Δ_n vs Δ_{n+1})")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Adjacent Step Vector Alignment")
    ax2.set_ylim(-0.2, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(odir / "step_vectors.png", dpi=150)
    plt.close()
    print(f"  Saved step_vectors.png")

    # --- Plot 3: Full cosine similarity matrix ---
    cosine_matrix = norms_mat @ norms_mat.T
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cosine_matrix, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels([str(l) for l in step_labels], fontsize=7)
    ax.set_yticks(range(len(step_labels)))
    ax.set_yticklabels([str(l) for l in step_labels], fontsize=7)
    ax.set_xlabel("Step vector Δ_n")
    ax.set_ylabel("Step vector Δ_m")
    ax.set_title("Pairwise Cosine Similarity of Successor Step Vectors")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(odir / "cosine_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved cosine_matrix.png")

    # --- Plot 4: Linear reconstruction error ---
    counts_sorted = sorted(centroids.keys())
    delta_mean = step_vectors.mean(axis=0)
    mu_0 = centroids[0]
    norm_errors = []
    mean_step_dist = mags.mean()
    for c in counts_sorted:
        recon = mu_0 + c * delta_mean
        err = np.linalg.norm(recon - centroids[c]) / mean_step_dist
        norm_errors.append(err)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(counts_sorted, norm_errors, "o-", color="#9C27B0", markersize=4)
    ax.axhline(1.0, color="#FF5722", linestyle="--", alpha=0.5,
               label="1 step-size error")
    ax.set_xlabel("Count")
    ax.set_ylabel("Normalized reconstruction error\n(multiples of mean step size)")
    ax.set_title("Linear Reconstruction: μ₀ + n·Δ_mean vs actual centroid μ_n")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(odir / "linear_reconstruction.png", dpi=150)
    plt.close()
    print(f"  Saved linear_reconstruction.png")

    # --- Plot 5: Step vector PCA eigenspectrum ---
    centered = step_vectors - step_vectors.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(1, 11), explained[:10] * 100, color="#FF9800", alpha=0.7)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Step Vector PCA — Eigenspectrum")
    ax1.grid(True, alpha=0.2)

    ax2.plot(range(1, 11), cumulative[:10] * 100, "o-", color="#FF9800")
    ax2.axhline(90, color="#FF5722", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(95, color="#F44336", linestyle="--", alpha=0.5, label="95%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Step Vector PCA — Cumulative")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(odir / "step_vector_pca.png", dpi=150)
    plt.close()
    print(f"  Saved step_vector_pca.png")


def main():
    parser = argparse.ArgumentParser(
        description="Successor function analysis in RSSM hidden states")
    parser.add_argument("--weights-dir", type=str,
                        default="/workspace/bridge/models/randproj_clean")
    parser.add_argument("--output-dir", type=str,
                        default="/workspace/bridge/artifacts/reports/successor_analysis")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--blob-count", type=int, default=25)
    parser.add_argument("--randproj", action="store_true", default=True,
                        help="Apply random projection (default True for randproj model)")
    parser.add_argument("--no-randproj", action="store_false", dest="randproj")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    odir = Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Successor Function Analysis")
    print("=" * 60)
    print(f"  Model:    {args.weights_dir}")
    print(f"  Randproj: {args.randproj}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Blobs:    {args.blob_count}")
    print(f"  Output:   {odir}")
    print()

    # Load probe
    wdir = Path(args.weights_dir)
    with open(wdir / "embed_probe.json") as f:
        probe = json.load(f)
    probe_w = np.array(probe["weights"])
    probe_b = probe["bias"]

    # Collect data
    print("Collecting episode trajectories...")
    t0 = time.time()
    episodes, weights = collect_episode_trajectories(
        args.weights_dir, args.episodes, args.blob_count, args.randproj)
    print(f"  Done in {time.time()-t0:.0f}s\n")

    # Compute centroids
    centroids = compute_centroids(episodes)
    print(f"Centroids computed for {len(centroids)} counts\n")

    # Run analyses
    all_results = {}

    all_results["transition_profile"] = analysis_1_transition_profile(
        episodes, probe_w, probe_b)

    step_results, step_vectors = analysis_2_step_vectors(centroids)
    all_results["step_vectors"] = step_results

    all_results["instantaneous_velocity"] = analysis_3_instantaneous_velocity(
        episodes, centroids)

    all_results["step_vector_pca"] = analysis_4_step_vector_pca(step_vectors)

    all_results["linear_reconstruction"] = analysis_5_linear_reconstruction(centroids)

    # Save results
    with open(odir / "successor_analysis.json", "w") as f:
        # Remove non-serializable data from transition profiles
        serializable = {}
        for k, v in all_results.items():
            if k == "transition_profile":
                # Strip large arrays for JSON
                clean = []
                for ep_result in v:
                    c = dict(ep_result)
                    c.pop("projections", None)  # too large
                    clean.append(c)
                serializable[k] = clean
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {odir}/successor_analysis.json")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(episodes, centroids, probe_w, probe_b,
                       step_vectors, step_results["step_labels"], odir)

    print("\nDone!")


if __name__ == "__main__":
    main()
