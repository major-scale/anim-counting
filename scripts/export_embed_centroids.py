#!/usr/bin/env python3
"""
Compute 512-dim encoder embed centroids per count for browser inference.

Uses the pure Python counting env to collect obs across all count values,
runs them through the DreamerV3 encoder WITH symlog, and computes:
  1. Mean 512-dim embed per count (for nearest-centroid prediction)
  2. PCA on centroids (for 2D visualization)
  3. Ridge probe (for linear prediction backup)

Output: embed_pca.json and embed_probe.json in signal-app/public/models/
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────
OBS_SIZE = 82
HIDDEN = 512
LN_EPS = 1e-3

MODELS_DIR = Path("/workspace/projects/jamstack-v1/packages/signal-app/public/models")
BIN_PATH = MODELS_DIR / "dreamer_weights.bin"
MANIFEST_PATH = MODELS_DIR / "dreamer_manifest.json"


# ─── Weight loading ───────────────────────────────────────────────────
def load_weights(bin_path: Path, manifest_path: Path) -> dict[str, np.ndarray]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(bin_path, "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"], offset=entry["offset"]).copy()
        arr = arr.reshape(entry["shape"])
        weights[name] = arr
    return weights


# ─── Encoder forward pass with symlog ─────────────────────────────────
def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log(np.abs(x) + 1)


def layer_norm(x, w, b):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    inv_std = 1.0 / np.sqrt(var + LN_EPS)
    return (x - mean) * inv_std * w + b


def silu(x):
    return x / (1.0 + np.exp(-x))


def encode_obs(obs80: np.ndarray, weights: dict) -> np.ndarray:
    """Encoder: symlog → 3x(Linear → LN → SiLU) → embed[512]."""
    x = symlog(obs80[:OBS_SIZE].astype(np.float32))

    h = weights["enc_linear0_w"] @ x
    h = layer_norm(h, weights["enc_norm0_w"], weights["enc_norm0_b"])
    h = silu(h)

    h = weights["enc_linear1_w"] @ h
    h = layer_norm(h, weights["enc_norm1_w"], weights["enc_norm1_b"])
    h = silu(h)

    h = weights["enc_linear2_w"] @ h
    h = layer_norm(h, weights["enc_norm2_w"], weights["enc_norm2_b"])
    h = silu(h)

    return h.copy()


# ─── Data collection ──────────────────────────────────────────────────
def collect_embeddings(weights: dict, n_episodes: int = 30, blob_count_fixed: int | None = None):
    """Run pure Python env, collect encoder embeds per count.

    If blob_count_fixed is set, only collect that blob count (e.g. 25 for browser).
    Otherwise collect all counts 3-25.
    """
    sys.path.insert(0, "/workspace/bridge/scripts")
    from counting_env_pure import CountingWorldEnv

    embeds_by_count = defaultdict(list)
    obs_by_count = defaultdict(list)
    total_samples = 0

    blob_range = [blob_count_fixed] if blob_count_fixed else range(3, 26)

    for ep in range(n_episodes):
        for blob_count in blob_range:
            env = CountingWorldEnv(
                blob_count_min=blob_count,
                blob_count_max=blob_count,
            )
            vec = env.reset()  # returns numpy array [82]
            done = False

            while not done:
                count = int(vec[81])  # grid_filled_raw
                obs80 = vec[:OBS_SIZE].astype(np.float32)
                embed = encode_obs(obs80, weights)

                embeds_by_count[count].append(embed)
                obs_by_count[count].append(obs80.copy())
                total_samples += 1

                # action doesn't matter for obs collection (env is autonomous)
                vec, reward, done, info = env.step(-0.995)

            env.close()

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: {total_samples} samples so far")

    print(f"\n  Total: {total_samples} samples across {len(embeds_by_count)} unique counts")
    for c in sorted(embeds_by_count.keys()):
        print(f"    count={c:2d}: {len(embeds_by_count[c])} samples")

    return embeds_by_count, obs_by_count


# ─── Centroid computation ─────────────────────────────────────────────
def compute_centroids(embeds_by_count: dict) -> tuple[dict, np.ndarray, list]:
    """Compute mean 512-dim embed per count."""
    centroids = {}
    for count in sorted(embeds_by_count.keys()):
        arr = np.stack(embeds_by_count[count])
        centroids[count] = arr.mean(axis=0)

    counts = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in counts])
    return centroids, centroid_matrix, counts


# ─── PCA ──────────────────────────────────────────────────────────────
def fit_pca(centroid_matrix: np.ndarray, n_components: int = 2):
    """Simple PCA on centroid matrix."""
    mean = centroid_matrix.mean(axis=0)
    centered = centroid_matrix - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components].T  # [n_components, 512]
    explained = eigenvalues[:n_components] / eigenvalues.sum()
    projected = centered @ components.T  # [N, n_components]

    return components, mean, explained, projected


# ─── Ridge probe ──────────────────────────────────────────────────────
def train_probe(embeds_by_count: dict, alpha: float = 10.0):
    """Train Ridge regression probe: embed[512] → count."""
    X_list, y_list = [], []
    for count, embs in embeds_by_count.items():
        for emb in embs:
            X_list.append(emb)
            y_list.append(count)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float64)

    # Ridge regression: w = (X^T X + alpha*I)^{-1} X^T y
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    bias = y.mean() - w @ X.mean(axis=0)

    # Evaluate
    preds = X @ w + bias
    preds_rounded = np.clip(np.round(preds), 0, 25).astype(int)
    exact = (preds_rounded == y.astype(int)).mean()
    within1 = (np.abs(preds_rounded - y) <= 1).mean()

    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    print(f"\n  Ridge probe (alpha={alpha}):")
    print(f"    R² = {r_squared:.6f}")
    print(f"    Exact = {exact:.1%}")
    print(f"    Within ±1 = {within1:.1%}")
    print(f"    N = {len(y)}")

    return w, bias, r_squared


# ─── Export ───────────────────────────────────────────────────────────
def export_all(centroid_matrix, counts, components, pca_mean, explained, projected,
               centroids_512, probe_w, probe_bias, probe_r2, output_dir):
    """Export embed_pca.json and embed_probe.json."""
    os.makedirs(output_dir, exist_ok=True)

    # embed_pca.json: PCA + 2D centroids + 512-dim centroids
    pca_data = {
        "pca_components": components.tolist(),
        "pca_mean": pca_mean.tolist(),
        "pca_explained_variance": explained.tolist(),
        "centroids_2d": projected.tolist(),
        "centroid_counts": counts,
        "centroids_512": {str(c): centroids_512[c].tolist() for c in counts},
    }

    pca_path = os.path.join(output_dir, "embed_pca.json")
    with open(pca_path, "w") as f:
        json.dump(pca_data, f)
    size_kb = os.path.getsize(pca_path) / 1e3
    print(f"\n  {pca_path}: {size_kb:.1f} KB")
    print(f"    {len(counts)} centroids, PCA explained: {explained.sum():.1%}")
    print(f"    512-dim centroids included for nearest-centroid prediction")

    # embed_probe.json: linear probe weights
    probe_data = {
        "weights": probe_w.tolist(),
        "bias": float(probe_bias),
        "r_squared": float(probe_r2),
        "description": f"Ridge(alpha=10) on {len(counts)} counts, symlog encoder, pure python env",
    }

    probe_path = os.path.join(output_dir, "embed_probe.json")
    with open(probe_path, "w") as f:
        json.dump(probe_data, f)
    size_kb = os.path.getsize(probe_path) / 1e3
    print(f"  {probe_path}: {size_kb:.1f} KB")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=== Export Embed Centroids ===\n")

    print("Loading encoder weights...")
    weights = load_weights(BIN_PATH, MANIFEST_PATH)
    print(f"  {len(weights)} tensors loaded\n")

    # Collect embeddings for 25-blob episodes only (matches browser which always has 25 blobs)
    print("Collecting embeddings (25 blobs only, matching browser)...")
    embeds_by_count, obs_by_count = collect_embeddings(
        weights, n_episodes=50, blob_count_fixed=25
    )

    print("\nComputing 512-dim centroids...")
    centroids_512, centroid_matrix, counts = compute_centroids(embeds_by_count)

    print("Fitting PCA on centroids...")
    components, pca_mean, explained, projected = fit_pca(centroid_matrix)
    print(f"  PC1: {explained[0]:.1%}, PC2: {explained[1]:.1%}")

    print("\nTraining Ridge probe...")
    probe_w, probe_bias, probe_r2 = train_probe(embeds_by_count, alpha=10.0)

    print("\nExporting...")
    export_all(
        centroid_matrix, counts, components, pca_mean, explained, projected,
        centroids_512, probe_w, probe_bias, probe_r2,
        str(MODELS_DIR),
    )

    # Verify nearest-centroid accuracy
    print("\n=== Nearest-Centroid Accuracy (512-dim) ===")
    exact, within1, total = 0, 0, 0
    for count, embs in embeds_by_count.items():
        for emb in embs:
            # Find nearest centroid in 512-dim
            dists = np.array([np.linalg.norm(emb - centroids_512[c]) for c in counts])
            pred = counts[np.argmin(dists)]
            total += 1
            if pred == count:
                exact += 1
            if abs(pred - count) <= 1:
                within1 += 1

    print(f"  Exact: {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"  Within ±1: {within1}/{total} ({100*within1/total:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
