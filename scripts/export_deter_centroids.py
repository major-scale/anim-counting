#!/usr/bin/env python3
"""
Compute 512-dim RSSM deter state centroids per count for browser inference.

Uses the full RSSM (encoder + GRU + posterior) with action=0, matching the
browser's inference mode. The deter state accumulates temporal context and
gives 98.7% exact accuracy vs ~70% with encoder-only.

Output: embed_pca.json and embed_probe.json in signal-app/public/models/
(overwrites the encoder-based versions)
"""

import json
import sys
import os
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────
OBS_SIZE = 82  # default, overridden by weight shape at load time
DETER = 512
HIDDEN = 512
STOCH_CATS = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_CATS * STOCH_CLASSES  # 1024
LN_EPS = 1e-3

MODELS_DIR = Path("/workspace/projects/jamstack-v1/packages/signal-app/public/models")
BIN_PATH = MODELS_DIR / "dreamer_weights.bin"
MANIFEST_PATH = MODELS_DIR / "dreamer_manifest.json"


# ─── Weight loading ───────────────────────────────────────────────────
def load_weights():
    global OBS_SIZE
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(BIN_PATH, "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"], offset=entry["offset"]).copy()
        arr = arr.reshape(entry["shape"])
        weights[name] = arr
    # Derive obs dim from encoder input weight shape
    if "enc_linear0_w" in weights:
        OBS_SIZE = weights["enc_linear0_w"].shape[1]
    return weights


# ─── Vectorized RSSM (from test_rssm_zero_action.py) ─────────────────
def ln(x, w, b):
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + LN_EPS) * w + b


def silu_v(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sigmoid_v(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def argmax_one_hot(logits):
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    reshaped = logits.reshape(STOCH_CATS, STOCH_CLASSES)
    indices = reshaped.argmax(axis=1)
    for c in range(STOCH_CATS):
        out[c * STOCH_CLASSES + indices[c]] = 1.0
    return out


class FastRSSM:
    def __init__(self, w):
        self.w = w
        self.deter = np.tanh(w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True

        # Compute initial prior
        h = silu_v(ln(w["img_out_w"] @ self.deter, w["img_out_norm_w"], w["img_out_norm_b"]))
        logits = w["imgs_stat_w"] @ h + w["imgs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)

    def reset(self):
        self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        h = silu_v(ln(self.w["img_out_w"] @ self.deter, self.w["img_out_norm_w"], self.w["img_out_norm_b"]))
        logits = self.w["imgs_stat_w"] @ h + self.w["imgs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)
        self.is_first = True

    def step(self, obs_raw, action=0.0):
        """Full obs_step: encode → img_step → posterior. Returns deter copy.

        action can be a scalar (passive, num_actions=1) or array (embodied, num_actions=2).
        """
        w = self.w

        # Symlog + encode
        x = np.sign(obs_raw) * np.log(np.abs(obs_raw) + 1)
        h = silu_v(ln(w["enc_linear0_w"] @ x, w["enc_norm0_w"], w["enc_norm0_b"]))
        h = silu_v(ln(w["enc_linear1_w"] @ h, w["enc_norm1_w"], w["enc_norm1_b"]))
        embed = silu_v(ln(w["enc_linear2_w"] @ h, w["enc_norm2_w"], w["enc_norm2_b"]))

        # img_step — action is scalar or array depending on model
        act = np.zeros_like(np.atleast_1d(np.asarray(action, dtype=np.float32))) if self.is_first else np.atleast_1d(np.asarray(action, dtype=np.float32))
        self.is_first = False
        cat = np.concatenate([self.stoch, act]).astype(np.float32)
        img_h = silu_v(ln(w["img_in_w"] @ cat, w["img_in_norm_w"], w["img_in_norm_b"]))

        gru_in = np.concatenate([img_h, self.deter]).astype(np.float32)
        gru_ln = ln(w["gru_w"] @ gru_in, w["gru_norm_w"], w["gru_norm_b"])

        # Vectorized GRU gates
        reset = sigmoid_v(gru_ln[:DETER])
        cand = np.tanh(gru_ln[DETER:2*DETER] * reset)
        update = sigmoid_v(gru_ln[2*DETER:] - 1.0)
        self.deter = (update * cand + (1.0 - update) * self.deter).astype(np.float32)

        # Posterior
        inp = np.concatenate([self.deter, embed]).astype(np.float32)
        h = silu_v(ln(w["obs_out_w"] @ inp, w["obs_out_norm_w"], w["obs_out_norm_b"]))
        logits = w["obs_stat_w"] @ h + w["obs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)

        return self.deter.copy()


# ─── Data collection ──────────────────────────────────────────────────
def _heuristic_steering(state):
    """Simple approach-nearest-blob heuristic for embodied data collection."""
    import math
    bot = state.bot
    best_d = float("inf")
    target_x, target_y = None, None
    for blob in state.blobs:
        if blob.grid_slot is not None or blob.pending_grid_placement:
            continue
        dx = blob.pos_x - bot.pos_x
        dy = blob.pos_y - bot.pos_y
        d = math.sqrt(dx * dx + dy * dy)
        if d < best_d:
            best_d = d
            target_x, target_y = blob.pos_x, blob.pos_y
    if target_x is None:
        return bot.heading
    return math.atan2(target_y - bot.pos_y, target_x - bot.pos_x)


def collect_deter_states(w, n_episodes=50, blob_count_fixed=25, proj_matrix=None,
                         embodied=False, multidim=None, proj_dim=128):
    """Run pure Python env with RSSM, collect deter states per count.

    proj_matrix: optional (82,82) orthogonal matrix to project obs before RSSM
                 (required for randproj models trained on projected obs)
    embodied: if True, use EmbodiedCountingWorldEnv with heuristic steering
    """
    import math
    sys.path.insert(0, "/workspace/bridge/scripts")
    from counting_env_pure import CountingWorldEnv

    # Detect num_actions from weight shape
    num_actions = w["img_in_w"].shape[1] - 1024

    deter_by_count = defaultdict(list)
    total = 0
    t0 = time.time()

    for ep in range(n_episodes):
        if multidim is not None:
            from counting_env_multidim import MultiDimCountingWorldEnv
            env = MultiDimCountingWorldEnv(
                blob_count_min=blob_count_fixed,
                blob_count_max=blob_count_fixed,
                fixed_dim=multidim,
                proj_dim=proj_dim,
            )
        elif embodied:
            from counting_env_embodied import EmbodiedCountingWorldEnv
            env = EmbodiedCountingWorldEnv(
                blob_count_min=blob_count_fixed,
                blob_count_max=blob_count_fixed,
                max_steps=8000,
            )
        else:
            env = CountingWorldEnv(
                blob_count_min=blob_count_fixed,
                blob_count_max=blob_count_fixed,
            )
        model = FastRSSM(w)
        vec = env.reset()
        done = False

        while not done:
            if multidim is not None:
                obs = vec[:OBS_SIZE].astype(np.float32)
                count = int(env._state.grid.filled_count)
            elif proj_matrix is not None:
                obs = (proj_matrix @ vec.astype(np.float32))[:OBS_SIZE]
                count = int(vec[81])
            else:
                obs = vec[:OBS_SIZE].astype(np.float32)
                count = int(vec[81])

            # Build action for RSSM
            if embodied:
                steering = _heuristic_steering(env._state)
                action = np.zeros(num_actions, dtype=np.float32)
                action[0] = steering / math.pi  # normalize to [-1, 1]
            else:
                action = 0.0

            deter = model.step(obs, action)
            deter_by_count[count].append(deter)
            total += 1

            # Step env
            if embodied:
                vec, reward, done, info = env.step([steering])
            else:
                vec, reward, done, info = env.step(-0.995)  # works for both 2D and multi-dim

        env.close()

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            fps = total / elapsed if elapsed > 0 else 0
            print(f"  Episode {ep+1}/{n_episodes}: {total} samples ({fps:.0f} steps/s)")

    print(f"\n  Total: {total} samples across {len(deter_by_count)} unique counts")
    for c in sorted(deter_by_count.keys()):
        print(f"    count={c:2d}: {len(deter_by_count[c])} samples")

    return deter_by_count


# ─── Centroid computation ─────────────────────────────────────────────
def compute_centroids(deter_by_count):
    centroids = {}
    for count in sorted(deter_by_count.keys()):
        arr = np.stack(deter_by_count[count])
        centroids[count] = arr.mean(axis=0)
    counts = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in counts])
    return centroids, centroid_matrix, counts


# ─── PCA ──────────────────────────────────────────────────────────────
def fit_pca(centroid_matrix, n_components=2):
    mean = centroid_matrix.mean(axis=0)
    centered = centroid_matrix - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components].T
    explained = eigenvalues[:n_components] / eigenvalues.sum()
    projected = centered @ components.T

    return components, mean, explained, projected


# ─── Ridge probe ──────────────────────────────────────────────────────
def train_probe(deter_by_count, alpha=10.0):
    X_list, y_list = [], []
    for count, deters in deter_by_count.items():
        for d in deters:
            X_list.append(d)
            y_list.append(count)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float64)

    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    bias = y.mean() - w @ X.mean(axis=0)

    preds = X @ w + bias
    preds_rounded = np.clip(np.round(preds), 0, 25).astype(int)
    exact = (preds_rounded == y.astype(int)).mean()
    within1 = (np.abs(preds_rounded - y) <= 1).mean()

    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    print(f"\n  Ridge probe (alpha={alpha}) on RSSM deter states:")
    print(f"    R² = {r_squared:.6f}")
    print(f"    Exact = {exact:.1%}")
    print(f"    Within ±1 = {within1:.1%}")
    print(f"    N = {len(y)}")

    return w, bias, r_squared


# ─── Export ───────────────────────────────────────────────────────────
def export_all(centroid_matrix, counts, components, pca_mean, explained, projected,
               centroids_512, probe_w, probe_bias, probe_r2, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Also compute probe values at each centroid for debugging
    centroid_probevals = {}
    for c in counts:
        val = float(probe_w @ centroids_512[c] + probe_bias)
        centroid_probevals[str(c)] = val

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
    print(f"    {len(counts)} centroids, PCA explained: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    probe_data = {
        "weights": probe_w.tolist(),
        "bias": float(probe_bias),
        "r_squared": float(probe_r2),
        "description": "Ridge(alpha=10) on RSSM deter states, action=0, 25-blob pure python env",
        "centroid_probevals": centroid_probevals,
    }

    probe_path = os.path.join(output_dir, "embed_probe.json")
    with open(probe_path, "w") as f:
        json.dump(probe_data, f)
    size_kb = os.path.getsize(probe_path) / 1e3
    print(f"  {probe_path}: {size_kb:.1f} KB")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export RSSM deter centroids for visualization")
    parser.add_argument("--randproj", action="store_true",
                        help="Apply 82x82 orthogonal projection (seed 42000) to obs before RSSM")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: MODELS_DIR)")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory containing dreamer_weights.bin and manifest")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Number of episodes to collect")
    parser.add_argument("--embodied", action="store_true",
                        help="Use embodied env with heuristic steering for data collection")
    parser.add_argument("--multidim", type=int, default=None, metavar="D",
                        help="Use multi-dim env at fixed dimensionality D")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection dim for multi-dim env (default 128)")
    args = parser.parse_args()

    # Override paths if specified
    global BIN_PATH, MANIFEST_PATH, MODELS_DIR
    if args.weights_dir:
        wdir = Path(args.weights_dir)
        BIN_PATH = wdir / "dreamer_weights.bin"
        MANIFEST_PATH = wdir / "dreamer_manifest.json"

    proj_label = " + randproj" if args.randproj else ""
    emb_label = " + embodied" if args.embodied else ""
    print(f"=== Export RSSM Deter Centroids (action={'heuristic' if args.embodied else '0'}{proj_label}{emb_label}) ===\n")

    print("Loading weights...")
    w = load_weights()
    print(f"  {len(w)} tensors loaded\n")

    # Random projection matrix
    proj_matrix = None
    if args.randproj:
        from scipy.stats import ortho_group
        proj_matrix = ortho_group.rvs(82, random_state=np.random.RandomState(42_000)).astype(np.float32)
        print("  Using random projection (82x82 orthogonal, seed 42000)\n")

    if args.multidim is not None:
        act_label = f"multi-dim D={args.multidim}"
    elif args.embodied:
        act_label = "heuristic steering"
    else:
        act_label = "action=0"
    print(f"Collecting RSSM deter states (25 blobs, {act_label})...")
    deter_by_count = collect_deter_states(w, n_episodes=args.episodes, blob_count_fixed=25,
                                           proj_matrix=proj_matrix, embodied=args.embodied,
                                           multidim=args.multidim, proj_dim=args.proj_dim)

    print("\nComputing 512-dim centroids...")
    centroids_512, centroid_matrix, counts = compute_centroids(deter_by_count)

    print("Fitting PCA on deter centroids...")
    components, pca_mean, explained, projected = fit_pca(centroid_matrix)
    print(f"  PC1: {explained[0]:.1%}, PC2: {explained[1]:.1%}")

    print("\nTraining Ridge probe on deter states...")
    probe_w, probe_bias, probe_r2 = train_probe(deter_by_count, alpha=10.0)

    output_dir = args.output_dir or str(MODELS_DIR)
    print(f"\nExporting to {output_dir}...")
    export_all(
        centroid_matrix, counts, components, pca_mean, explained, projected,
        centroids_512, probe_w, probe_bias, probe_r2,
        output_dir,
    )

    # Verify nearest-centroid accuracy (512-dim)
    print("\n=== Nearest-Centroid Accuracy (512-dim deter space) ===")
    exact, within1, total = 0, 0, 0
    for count, deters in deter_by_count.items():
        for d in deters:
            dists = np.array([np.linalg.norm(d - centroids_512[c]) for c in counts])
            pred = counts[np.argmin(dists)]
            total += 1
            if pred == count:
                exact += 1
            if abs(pred - count) <= 1:
                within1 += 1

    print(f"  Exact: {exact}/{total} ({100*exact/total:.1f}%)")
    print(f"  Within ±1: {within1}/{total} ({100*within1/total:.1f}%)")

    print("\nDone! Browser manifold visualization now uses RSSM deter space.")


if __name__ == "__main__":
    main()
