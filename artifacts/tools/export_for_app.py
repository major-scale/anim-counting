"""
Export for App — Phase 2→3 Bridge
===================================
Creates brain_explorer_data.json with everything the HTML app needs:
- 4 embedding coordinate sets (downsampled background)
- Full trajectory data for all episodes
- Cluster centers per config
- Episode metadata
- Pre-computed anticipation signal

Output: data/brain_explorer_data.json
"""

import os
import json
import pathlib
import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"

MAX_BACKGROUND_POINTS = 5000


def main():
    print("=" * 60)
    print("Phase 2→3: Export Data for Browser App")
    print("=" * 60)

    # Load data
    traj = np.load(DATA_DIR / "trajectories.npz")
    emb_data = np.load(DATA_DIR / "umap_embeddings.npz")

    h_t = traj["h_t"]
    counts = traj["true_count"]
    episode_ids = traj["episode_id"]
    timesteps = traj["timestep"]
    actor_pred = traj["actor_prediction"]
    reward = traj["reward"]
    bot_x = traj["bot_x"]
    bot_y = traj["bot_y"]
    is_marking = traj["is_marking_event"]
    total_blobs = traj["total_blobs"]

    n = len(counts)
    unique_eps = np.unique(episode_ids)
    n_eps = len(unique_eps)

    print(f"Loaded {n} timesteps from {n_eps} episodes")

    # Load embeddings
    embedding_names = ["default", "local", "global", "pca"]
    embeddings = {}
    for name in embedding_names:
        key = f"emb_{name}"
        if key in emb_data:
            embeddings[name] = emb_data[key]
            print(f"  Embedding '{name}': {emb_data[key].shape}")

    # Compute cluster centers
    cluster_centers = {}
    for name, emb in embeddings.items():
        centers = {}
        for c in range(9):
            mask = counts == c
            if mask.sum() > 0:
                center = emb[mask].mean(axis=0)
                centers[str(c)] = [float(center[0]), float(center[1])]
        cluster_centers[name] = centers

    # Downsample background points (stratified by count)
    print(f"\nDownsampling to {MAX_BACKGROUND_POINTS} background points...")
    rng = np.random.RandomState(42)
    bg_indices = []
    per_count = MAX_BACKGROUND_POINTS // 9
    for c in range(9):
        idx = np.where(counts == c)[0]
        if len(idx) > per_count:
            chosen = rng.choice(idx, per_count, replace=False)
        else:
            chosen = idx
        bg_indices.extend(chosen)
    bg_indices = np.array(sorted(bg_indices))
    print(f"  Selected {len(bg_indices)} background points")

    # Background data (for scatter plot)
    background = {
        "count": counts[bg_indices].tolist(),
    }
    for name, emb in embeddings.items():
        background[f"x_{name}"] = emb[bg_indices, 0].tolist()
        background[f"y_{name}"] = emb[bg_indices, 1].tolist()

    # Per-episode full trajectory data
    episodes_data = []
    for ep in unique_eps:
        mask = episode_ids == ep
        ep_start = np.where(mask)[0][0]
        ep_end = np.where(mask)[0][-1] + 1
        ep_len = ep_end - ep_start

        ep_blobs = int(total_blobs[mask][0])
        ep_counts = counts[mask]
        ep_preds = actor_pred[mask]
        ep_errors = np.abs(ep_preds - ep_counts)
        max_error = float(ep_errors.max())
        mean_error = float(ep_errors.mean())

        # Tag episodes
        tags = []
        if max_error < 0.3:
            tags.append("clean")
        if max_error > 1.0:
            tags.append("interesting")
        if ep_counts.max() == 8:
            tags.append("reaches-8")
        if mean_error > 0.5:
            tags.append("noisy")

        ep_data = {
            "id": int(ep),
            "blob_count": ep_blobs,
            "length": int(ep_len),
            "max_error": round(max_error, 3),
            "mean_error": round(mean_error, 3),
            "tags": tags,
            "timestep": timesteps[mask].tolist(),
            "count": ep_counts.tolist(),
            "prediction": [round(float(v), 3) for v in ep_preds],
            "reward": [round(float(v), 3) for v in reward[mask]],
            "bot_x": [round(float(v), 4) for v in bot_x[mask]],
            "bot_y": [round(float(v), 4) for v in bot_y[mask]],
            "is_marking": is_marking[mask].tolist(),
        }

        # UMAP coordinates for this episode's trajectory
        for name, emb in embeddings.items():
            ep_data[f"x_{name}"] = [round(float(v), 4) for v in emb[mask, 0]]
            ep_data[f"y_{name}"] = [round(float(v), 4) for v in emb[mask, 1]]

        episodes_data.append(ep_data)

    # Sort by blob count for the dropdown
    episodes_data.sort(key=lambda e: (e["blob_count"], e["id"]))

    # Load analysis results if available
    analysis_path = DATA_DIR / "analysis_results.json"
    analysis = {}
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        print(f"  Loaded analysis results")

    # Build final JSON
    output = {
        "embedding_names": embedding_names,
        "cluster_centers": cluster_centers,
        "background": background,
        "episodes": episodes_data,
        "analysis": analysis,
        "metadata": {
            "n_timesteps": n,
            "n_episodes": n_eps,
            "n_background": len(bg_indices),
        },
    }

    outpath = DATA_DIR / "brain_explorer_data.json"
    with open(outpath, "w") as f:
        json.dump(output, f)
    size_mb = outpath.stat().st_size / 1e6
    print(f"\nSaved to {outpath} ({size_mb:.1f} MB)")

    print(f"\n{'=' * 60}")
    print("Export complete! Open brain_explorer.html to explore.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
