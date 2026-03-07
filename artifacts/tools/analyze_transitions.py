"""
Analyze Transitions — Phase 4
================================
Quantitative analysis: transition dynamics, stability, count=8 deep dive,
dimensionality analysis, and anticipation signal.

Output: data/analysis_results.json + figures/*.png
"""

import os
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
FIG_DIR = _SCRIPT_DIR / "figures"

COUNT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


def load_data():
    traj = np.load(DATA_DIR / "trajectories.npz")
    emb = np.load(DATA_DIR / "umap_embeddings.npz")
    return traj, emb


def get_episode_slices(episode_ids):
    """Return dict mapping episode_id -> (start, end) indices."""
    slices = {}
    unique_eps = np.unique(episode_ids)
    for ep in unique_eps:
        indices = np.where(episode_ids == ep)[0]
        slices[ep] = (indices[0], indices[-1] + 1)
    return slices


def analyze_transition_dynamics(h_t, counts, episode_ids, umap_emb, ep_slices):
    """4a: Transition speed and path analysis at marking events."""
    print("\n[4a] Transition dynamics...")

    results = {}

    for c in range(8):
        transitions = []
        for ep, (start, end) in ep_slices.items():
            ep_counts = counts[start:end]
            ep_h = h_t[start:end]
            ep_umap = umap_emb[start:end]

            for i in range(len(ep_counts) - 1):
                if ep_counts[i] == c and ep_counts[i + 1] == c + 1:
                    # Find when we entered the c+1 cluster (stay there for 3+ steps)
                    t_mark = i + 1
                    # Look backward: how many steps was the trajectory already drifting?
                    # Compute UMAP path length from t-10 to t+10 around the mark
                    window_start = max(0, i - 10)
                    window_end = min(len(ep_counts), i + 11)
                    path = ep_umap[window_start:window_end]
                    if len(path) > 1:
                        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                        straight = np.linalg.norm(path[-1] - path[0])
                        ratio = path_length / max(straight, 1e-8)
                    else:
                        path_length, straight, ratio = 0, 0, 1

                    # h_t jump magnitude
                    h_jump = np.linalg.norm(ep_h[i + 1] - ep_h[i])

                    transitions.append({
                        "path_length": float(path_length),
                        "straight_line": float(straight),
                        "path_ratio": float(ratio),
                        "h_jump": float(h_jump),
                    })

        if transitions:
            results[f"{c}->{c+1}"] = {
                "n": len(transitions),
                "mean_path_ratio": float(np.mean([t["path_ratio"] for t in transitions])),
                "std_path_ratio": float(np.std([t["path_ratio"] for t in transitions])),
                "mean_h_jump": float(np.mean([t["h_jump"] for t in transitions])),
                "std_h_jump": float(np.std([t["h_jump"] for t in transitions])),
            }
            print(f"  {c}->{c+1}: n={len(transitions)}, "
                  f"path_ratio={results[f'{c}->{c+1}']['mean_path_ratio']:.2f}±{results[f'{c}->{c+1}']['std_path_ratio']:.2f}, "
                  f"h_jump={results[f'{c}->{c+1}']['mean_h_jump']:.2f}±{results[f'{c}->{c+1}']['std_h_jump']:.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = sorted(results.keys())
    ratios = [results[k]["mean_path_ratio"] for k in labels]
    ratio_errs = [results[k]["std_path_ratio"] for k in labels]
    jumps = [results[k]["mean_h_jump"] for k in labels]
    jump_errs = [results[k]["std_h_jump"] for k in labels]
    colors = [COUNT_COLORS[int(k[0])] for k in labels]

    ax1.bar(labels, ratios, yerr=ratio_errs, color=colors, edgecolor="black", capsize=3)
    ax1.set_ylabel("Path Length / Straight Line")
    ax1.set_title("UMAP Path Tortuosity per Transition")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    ax2.bar(labels, jumps, yerr=jump_errs, color=colors, edgecolor="black", capsize=3)
    ax2.set_ylabel("||h_{t+1} - h_t|| at marking event")
    ax2.set_title("Latent Jump Magnitude per Transition")

    plt.suptitle("Transition Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "transition_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved transition_dynamics.png")

    return results


def analyze_stability(h_t, counts, episode_ids, ep_slices):
    """4b: h_t drift rate during non-marking periods."""
    print("\n[4b] Between-mark stability...")

    drift_by_count = {c: [] for c in range(9)}

    for ep, (start, end) in ep_slices.items():
        ep_h = h_t[start:end]
        ep_counts = counts[start:end]

        for i in range(1, len(ep_h)):
            drift = np.linalg.norm(ep_h[i] - ep_h[i - 1])
            c = ep_counts[i]
            if ep_counts[i] == ep_counts[i - 1]:  # non-marking
                drift_by_count[c].append(drift)

    results = {}
    for c in range(9):
        if drift_by_count[c]:
            drifts = np.array(drift_by_count[c])
            results[str(c)] = {
                "mean_drift": float(drifts.mean()),
                "std_drift": float(drifts.std()),
                "n": len(drifts),
            }
            print(f"  count={c}: drift={drifts.mean():.3f}±{drifts.std():.3f} (n={len(drifts)})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cs = sorted([int(c) for c in results.keys()])
    means = [results[str(c)]["mean_drift"] for c in cs]
    stds = [results[str(c)]["std_drift"] for c in cs]
    ax.bar([str(c) for c in cs], means, yerr=stds, color=[COUNT_COLORS[c] for c in cs],
           edgecolor="black", capsize=3)
    ax.set_xlabel("Count")
    ax.set_ylabel("||h_t - h_{t-1}|| (non-marking steps)")
    ax.set_title("Latent Drift Rate by Count Level")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stability_by_count.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def analyze_count8(h_t, counts, actor_pred, umap_emb):
    """4c: Count=8 deep dive."""
    print("\n[4c] Count=8 deep dive...")

    mask7 = counts == 7
    mask8 = counts == 8
    n7, n8 = mask7.sum(), mask8.sum()
    print(f"  count=7: {n7} timesteps, count=8: {n8} timesteps")

    results = {}

    if n8 > 0 and n7 > 0:
        # UMAP separation
        center7 = umap_emb[mask7].mean(axis=0)
        center8 = umap_emb[mask8].mean(axis=0)
        separation = np.linalg.norm(center8 - center7)
        results["umap_separation_7_8"] = float(separation)
        print(f"  UMAP center distance (7 vs 8): {separation:.3f}")

        # h_t separation
        h_center7 = h_t[mask7].mean(axis=0)
        h_center8 = h_t[mask8].mean(axis=0)
        h_separation = np.linalg.norm(h_center8 - h_center7)
        results["h_separation_7_8"] = float(h_separation)
        print(f"  h_t center distance (7 vs 8): {h_separation:.3f}")

        # Actor predictions at count=8
        preds8 = actor_pred[mask8]
        results["count8_pred_mean"] = float(preds8.mean())
        results["count8_pred_std"] = float(preds8.std())
        results["count8_pred_median"] = float(np.median(preds8))
        print(f"  Count=8 predictions: mean={preds8.mean():.3f}±{preds8.std():.3f}, "
              f"median={np.median(preds8):.3f}")

        preds7 = actor_pred[mask7]
        results["count7_pred_mean"] = float(preds7.mean())

        # Plot prediction distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(preds7, bins=50, range=(5, 8.5), alpha=0.6, color=COUNT_COLORS[7],
                label=f"count=7 (n={n7})", density=True)
        ax.hist(preds8, bins=50, range=(5, 8.5), alpha=0.6, color=COUNT_COLORS[8],
                label=f"count=8 (n={n8})", density=True)
        ax.axvline(7, color=COUNT_COLORS[7], linestyle="--", linewidth=2)
        ax.axvline(8, color=COUNT_COLORS[8], linestyle="--", linewidth=2)
        ax.set_xlabel("Actor Prediction")
        ax.set_ylabel("Density")
        ax.set_title("Actor Predictions: Count=7 vs Count=8")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "count8_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved count8_analysis.png")
    else:
        results["count8_note"] = f"Insufficient data: n7={n7}, n8={n8}"

    results["n_count7"] = int(n7)
    results["n_count8"] = int(n8)
    return results


def analyze_dimensionality(h_t, counts, bot_x, bot_y):
    """4d: PCA spectrum and residual probes."""
    print("\n[4d] Dimensionality analysis...")

    pca = PCA(n_components=min(50, h_t.shape[1]))
    pca.fit(h_t)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n99 = int(np.searchsorted(cumvar, 0.99)) + 1
    print(f"  Components for 90% variance: {n90}")
    print(f"  Components for 95% variance: {n95}")
    print(f"  Components for 99% variance: {n99}")

    results = {
        "n_90pct": n90,
        "n_95pct": n95,
        "n_99pct": n99,
        "top5_variance": [float(v) for v in pca.explained_variance_ratio_[:5]],
    }

    # Residual analysis: project out PC1 (count direction), probe for bot_x, bot_y
    pca1 = PCA(n_components=1)
    h_pc1 = pca1.fit_transform(h_t)
    h_residual = h_t - pca1.inverse_transform(h_pc1)

    for name, target in [("bot_x", bot_x), ("bot_y", bot_y)]:
        X_train, X_test, y_train, y_test = train_test_split(
            h_residual, target, test_size=0.2, random_state=42
        )
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        results[f"residual_r2_{name}"] = float(r2)
        print(f"  Residual probe {name}: R²={r2:.4f}")

    # Also probe from full h_t for comparison
    for name, target in [("bot_x", bot_x), ("bot_y", bot_y)]:
        X_train, X_test, y_train, y_test = train_test_split(
            h_t, target, test_size=0.2, random_state=42
        )
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        results[f"full_r2_{name}"] = float(r2)
        print(f"  Full h_t probe {name}: R²={r2:.4f}")

    # PCA spectrum plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    n_plot = min(30, len(pca.explained_variance_ratio_))
    ax1.bar(range(n_plot), pca.explained_variance_ratio_[:n_plot], color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("PCA Spectrum of h_t")

    ax2.plot(range(1, len(cumvar) + 1), cumvar, "b-", linewidth=2)
    ax2.axhline(0.90, color="red", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(0.95, color="orange", linestyle="--", alpha=0.5, label="95%")
    ax2.axhline(0.99, color="green", linestyle="--", alpha=0.5, label="99%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()

    plt.suptitle("Dimensionality of h_t", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dimensionality_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dimensionality_analysis.png")

    return results


def analyze_anticipation(h_t, counts, episode_ids, ep_slices):
    """4e: Anticipation signal — does h_t start moving toward next count before the marking event?"""
    print("\n[4e] Anticipation analysis...")

    # Compute cluster centers in 512-dim
    cluster_centers = {}
    for c in range(9):
        mask = counts == c
        if mask.sum() > 0:
            cluster_centers[c] = h_t[mask].mean(axis=0)

    # For each marking event, compute cosine alignment
    window = 15  # t-10 to t+5
    t_offsets = list(range(-10, 6))  # -10, -9, ..., 0, ..., 5
    all_alignments = {t: [] for t in t_offsets}

    for ep, (start, end) in ep_slices.items():
        ep_h = h_t[start:end]
        ep_counts = counts[start:end]
        ep_len = end - start

        for i in range(1, ep_len):
            if ep_counts[i] != ep_counts[i - 1]:  # marking event at i
                c_before = ep_counts[i - 1]
                c_after = ep_counts[i]
                if c_after not in cluster_centers:
                    continue

                target_dir = cluster_centers[c_after] - cluster_centers[c_before]
                target_norm = np.linalg.norm(target_dir)
                if target_norm < 1e-8:
                    continue
                target_dir = target_dir / target_norm

                for t_off in t_offsets:
                    idx = i + t_off
                    if idx < 1 or idx >= ep_len:
                        continue
                    # h_t movement direction at this timestep
                    movement = ep_h[idx] - ep_h[idx - 1]
                    move_norm = np.linalg.norm(movement)
                    if move_norm < 1e-8:
                        all_alignments[t_off].append(0.0)
                        continue
                    cos_sim = np.dot(movement / move_norm, target_dir)
                    all_alignments[t_off].append(float(cos_sim))

    # Compute mean ± SEM
    results = {}
    mean_alignments = []
    sem_alignments = []
    valid_offsets = []

    for t_off in t_offsets:
        vals = all_alignments[t_off]
        if vals:
            arr = np.array(vals)
            mean_alignments.append(arr.mean())
            sem_alignments.append(arr.std() / np.sqrt(len(arr)))
            valid_offsets.append(t_off)
            results[f"t{t_off:+d}"] = {
                "mean": float(arr.mean()),
                "sem": float(arr.std() / np.sqrt(len(arr))),
                "n": len(vals),
            }

    mean_alignments = np.array(mean_alignments)
    sem_alignments = np.array(sem_alignments)
    valid_offsets = np.array(valid_offsets)

    print("  Alignment by offset:")
    for t_off, mean, sem in zip(valid_offsets, mean_alignments, sem_alignments):
        marker = " <<<" if t_off == 0 else ""
        print(f"    t{t_off:+d}: {mean:.4f} ± {sem:.4f}{marker}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(valid_offsets, mean_alignments - 2*sem_alignments,
                     mean_alignments + 2*sem_alignments, alpha=0.2, color="steelblue")
    ax.plot(valid_offsets, mean_alignments, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axvline(0, color="red", linewidth=2, alpha=0.7, label="Marking event (t=0)")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5, linestyle="--")
    ax.set_xlabel("Timestep relative to marking event", fontsize=12)
    ax.set_ylabel("Cosine alignment with target direction", fontsize=12)
    ax.set_title("Anticipation Signal: Does h_t move toward next count before marking?", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-10.5, 5.5)

    # Annotate key region
    pre_mean = mean_alignments[valid_offsets < 0].mean() if (valid_offsets < 0).any() else 0
    post_mean = mean_alignments[valid_offsets > 0].mean() if (valid_offsets > 0).any() else 0
    ax.text(0.02, 0.98, f"Pre-mark mean: {pre_mean:.4f}\nPost-mark mean: {post_mean:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "anticipation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved anticipation_analysis.png")

    results["pre_mark_mean"] = float(pre_mean)
    results["post_mark_mean"] = float(post_mean)
    return results


def main():
    print("=" * 60)
    print("Phase 4: Quantitative Transition Analysis")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traj, emb_data = load_data()

    h_t = traj["h_t"]
    counts = traj["true_count"]
    episode_ids = traj["episode_id"]
    actor_pred = traj["actor_prediction"]
    bot_x = traj["bot_x"]
    bot_y = traj["bot_y"]
    umap_default = emb_data["emb_default"]

    ep_slices = get_episode_slices(episode_ids)
    print(f"Loaded {len(counts)} timesteps, {len(ep_slices)} episodes")

    all_results = {}

    # 4a: Transition dynamics
    all_results["transition_dynamics"] = analyze_transition_dynamics(
        h_t, counts, episode_ids, umap_default, ep_slices
    )

    # 4b: Stability
    all_results["stability"] = analyze_stability(h_t, counts, episode_ids, ep_slices)

    # 4c: Count=8
    all_results["count8"] = analyze_count8(h_t, counts, actor_pred, umap_default)

    # 4d: Dimensionality
    all_results["dimensionality"] = analyze_dimensionality(h_t, counts, bot_x, bot_y)

    # 4e: Anticipation
    all_results["anticipation"] = analyze_anticipation(h_t, counts, episode_ids, ep_slices)

    # Save results
    outpath = DATA_DIR / "analysis_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved analysis results to {outpath}")

    print(f"\n{'=' * 60}")
    print("Phase 4 complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
