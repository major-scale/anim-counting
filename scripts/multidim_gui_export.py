#!/usr/bin/env python3
"""Export centroids, probe, and compute probe SNR for multi-dim model at specified D."""

import sys, json, os, time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--blob-count", type=int, default=13)
    args = parser.parse_args()

    wdir = Path(args.weights_dir)
    odir = Path(args.output_dir)
    os.makedirs(odir, exist_ok=True)
    D = args.dim

    # Load weights
    print(f"=== Multi-dim D={D} Export ===")
    with open(wdir / "dreamer_manifest.json") as f:
        manifest = json.load(f)
    with open(wdir / "dreamer_weights.bin", "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"], offset=entry["offset"]).copy()
        weights[name] = arr.reshape(entry["shape"])
    OBS_SIZE = weights["enc_linear0_w"].shape[1]
    print(f"  OBS_SIZE={OBS_SIZE}, num_actions={weights['img_in_w'].shape[1] - 1024}")

    from export_deter_centroids import FastRSSM
    from counting_env_multidim import MultiDimCountingWorldEnv

    # Collect deter states
    print(f"\nCollecting D={D} deter states ({args.episodes} episodes, {args.blob_count} blobs)...")
    deter_by_count = defaultdict(list)
    total = 0
    t0 = time.time()

    for ep in range(args.episodes):
        env = MultiDimCountingWorldEnv(
            blob_count_min=args.blob_count, blob_count_max=args.blob_count,
            fixed_dim=D, proj_dim=OBS_SIZE,
        )
        model = FastRSSM(weights)
        vec = env.reset()
        done = False

        while not done:
            obs = vec[:OBS_SIZE].astype(np.float32)
            count = int(env._state.grid.filled_count)
            deter = model.step(obs, 0.0)
            deter_by_count[count].append(deter)
            total += 1
            vec, reward, done, info = env.step(-0.995)
        env.close()

        elapsed = time.time() - t0
        fps = total / elapsed if elapsed > 0 else 0
        print(f"  Ep {ep+1}/{args.episodes}: {total} samples ({fps:.0f} steps/s), "
              f"filled={env._state.grid.filled_count}")

    print(f"\nTotal: {total} samples, {len(deter_by_count)} unique counts")
    for c in sorted(deter_by_count.keys()):
        print(f"  count={c:2d}: {len(deter_by_count[c])}")

    # Centroids
    centroids = {}
    for c in sorted(deter_by_count.keys()):
        centroids[c] = np.stack(deter_by_count[c]).mean(axis=0)
    counts = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in counts])

    # PCA
    mean = centroid_matrix.mean(axis=0)
    centered = centroid_matrix - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :2].T
    explained = eigenvalues[:2] / eigenvalues.sum()
    projected = centered @ components.T
    print(f"\nPCA: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    # Ridge probe
    X_list, y_list = [], []
    for c, deters in deter_by_count.items():
        for d in deters:
            X_list.append(d)
            y_list.append(c)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float64)
    alpha = 10.0
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    probe_w = np.linalg.solve(XtX, Xty)
    probe_bias = y.mean() - probe_w @ X.mean(axis=0)
    preds = X @ probe_w + probe_bias
    preds_r = np.clip(np.round(preds), 0, args.blob_count).astype(int)
    exact = (preds_r == y.astype(int)).mean()
    within1 = (np.abs(preds_r - y) <= 1).mean()
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\nRidge probe: R²={r2:.4f}, exact={exact:.1%}, within±1={within1:.1%}")

    # === Probe SNR ===
    probe_dir = probe_w / np.linalg.norm(probe_w)  # unit vector
    projections = X @ probe_dir  # scalar projection of each hidden state

    # Between-count variance: variance of per-count means
    count_means = {}
    for c in sorted(deter_by_count.keys()):
        idxs = (y == c)
        count_means[c] = projections[idxs].mean()
    between_var = np.var(list(count_means.values()))

    # Within-count variance: mean of per-count variances
    within_vars = []
    for c in sorted(deter_by_count.keys()):
        idxs = (y == c)
        if idxs.sum() > 1:
            within_vars.append(projections[idxs].var())
    within_var = np.mean(within_vars) if within_vars else 0.0

    snr = between_var / within_var if within_var > 0 else float("inf")
    print(f"\n=== PROBE SNR D={D} ===")
    print(f"  Between-count variance: {between_var:.4f}")
    print(f"  Within-count variance:  {within_var:.6f}")
    print(f"  Probe SNR:              {snr:.1f}")

    # Save centroids
    pca_data = {
        "pca_components": components.tolist(),
        "pca_mean": mean.tolist(),
        "pca_explained_variance": explained.tolist(),
        "centroids_2d": projected.tolist(),
        "centroid_counts": counts,
        "centroids_512": {str(c): centroids[c].tolist() for c in counts},
    }
    with open(odir / "embed_pca.json", "w") as f:
        json.dump(pca_data, f)

    probe_data = {
        "weights": probe_w.tolist(),
        "bias": float(probe_bias),
        "r_squared": float(r2),
        "probe_snr": float(snr),
        "between_var": float(between_var),
        "within_var": float(within_var),
    }
    with open(odir / "embed_probe.json", "w") as f:
        json.dump(probe_data, f)

    print(f"\nSaved to {odir}/embed_pca.json and embed_probe.json")
    print("Done!")


if __name__ == "__main__":
    main()
