#!/usr/bin/env python3
"""
Hidden State Anatomy — AI-native analysis of RSSM 512-dim hidden states.

Three analyses designed for AI interpretation, not human visualization:
1. Dimension importance ranking: which dims carry counting signal, per D
2. Dimension clustering by behavior: functional groups in the 512 dims
3. Subspace tracking over time: does the counting subspace rotate?

Plus cross-dimensional comparison: are the counting dims the SAME across D values?

Usage:
    python3 hidden_state_anatomy.py --checkpoint-dir DIR --dims 2 3 4 5 --episodes 10
    python3 hidden_state_anatomy.py --checkpoint-dir DIR --dims 2 3 --episodes 5 --blob-count 13
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# RSSM / env loading (reuse existing patterns)
# ---------------------------------------------------------------------------

STOCH_DIM = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_DIM * STOCH_CLASSES  # 1024
DETER_DIM = 512


def load_weights(checkpoint_dir):
    """Load RSSM weights from checkpoint directory.

    Handles two formats:
    1. Exported weights (dreamer_manifest.json + dreamer_weights.bin)
    2. dreamerv3-torch PyTorch checkpoint (latest.pt with agent_state_dict)
    """
    ckpt = Path(checkpoint_dir)

    # Try dreamer_manifest.json first (exported weights — old format)
    manifest_path = ckpt / "dreamer_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(ckpt / "dreamer_weights.bin", "rb") as f:
            raw = f.read()
        weights = {}
        for name, entry in manifest["tensors"].items():
            arr = np.frombuffer(
                raw, dtype="<f4", count=entry["length"], offset=entry["offset"]
            ).copy()
            weights[name] = arr.reshape(entry["shape"])
        weights["_format"] = "exported"
        return weights

    # Fall back to PyTorch checkpoint (dreamerv3-torch format)
    import torch

    ckpt_file = ckpt / "latest.pt"
    data = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # dreamerv3-torch uses agent_state_dict key
    if "agent_state_dict" in data:
        agent_state = data["agent_state_dict"]
    elif "agent" in data:
        agent_state = data["agent"]
    else:
        agent_state = data

    weights = {}
    # Map dreamerv3-torch keys to our internal names
    key_map = {
        # Encoder (no bias on linear layers, norm has bias)
        "_wm.encoder._mlp.layers.Encoder_linear0.weight": "enc_linear0_w",
        "_wm.encoder._mlp.layers.Encoder_norm0.weight": "enc_ln0_w",
        "_wm.encoder._mlp.layers.Encoder_norm0.bias": "enc_ln0_b",
        "_wm.encoder._mlp.layers.Encoder_linear1.weight": "enc_linear1_w",
        "_wm.encoder._mlp.layers.Encoder_norm1.weight": "enc_ln1_w",
        "_wm.encoder._mlp.layers.Encoder_norm1.bias": "enc_ln1_b",
        "_wm.encoder._mlp.layers.Encoder_linear2.weight": "enc_linear2_w",
        "_wm.encoder._mlp.layers.Encoder_norm2.weight": "enc_ln2_w",
        "_wm.encoder._mlp.layers.Encoder_norm2.bias": "enc_ln2_b",
        # Dynamics — img_in (no bias on linear, norm has bias)
        "_wm.dynamics._img_in_layers.0.weight": "img_in_w",
        "_wm.dynamics._img_in_layers.1.weight": "img_in_ln_w",
        "_wm.dynamics._img_in_layers.1.bias": "img_in_ln_b",
        # Dynamics — GRU (combined weight + LayerNorm)
        "_wm.dynamics._cell.layers.GRU_linear.weight": "gru_w",
        "_wm.dynamics._cell.layers.GRU_norm.weight": "gru_ln_w",
        "_wm.dynamics._cell.layers.GRU_norm.bias": "gru_ln_b",
        # Dynamics — deter init
        "_wm.dynamics.W": "deter_init_w",
        # Dynamics — prior (img_out)
        "_wm.dynamics._img_out_layers.0.weight": "prior0_w",
        "_wm.dynamics._img_out_layers.1.weight": "prior0_ln_w",
        "_wm.dynamics._img_out_layers.1.bias": "prior0_ln_b",
        "_wm.dynamics._imgs_stat_layer.weight": "prior_out_w",
        "_wm.dynamics._imgs_stat_layer.bias": "prior_out_b",
        # Dynamics — posterior (obs_out)
        "_wm.dynamics._obs_out_layers.0.weight": "post0_w",
        "_wm.dynamics._obs_out_layers.1.weight": "post0_ln_w",
        "_wm.dynamics._obs_out_layers.1.bias": "post0_ln_b",
        "_wm.dynamics._obs_stat_layer.weight": "post_out_w",
        "_wm.dynamics._obs_stat_layer.bias": "post_out_b",
    }
    for k, v in key_map.items():
        if k in agent_state:
            t = agent_state[k]
            weights[v] = t.numpy() if hasattr(t, "numpy") else np.array(t)

    weights["_format"] = "dreamerv3-torch"
    return weights


# Lightweight RSSM forward pass (numpy only)
def _ln(x, w, b, eps=1e-5):
    mu = x.mean()
    var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _gumbel_sample(logits):
    """Straight-through Gumbel softmax."""
    uniform = np.random.uniform(1e-5, 1 - 1e-5, logits.shape).astype(np.float32)
    gumbel = -np.log(-np.log(uniform))
    sample = np.zeros_like(logits)
    idx = (logits + gumbel).argmax(axis=-1)
    for i, j in enumerate(idx):
        sample[i, j] = 1.0
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return (sample + probs - probs).flatten().astype(np.float32)


class FastRSSM:
    """RSSM forward pass in pure numpy. Handles both exported and dreamerv3-torch formats."""

    GRU_UPDATE_BIAS = -1.0  # dreamerv3-torch default

    def __init__(self, weights):
        self.w = weights
        self.fmt = weights.get("_format", "exported")
        # Encoder has bias in exported format, not in dreamerv3-torch
        self.enc_has_bias = "enc_linear0_b" in weights
        self.deter = np.tanh(weights["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        self._compute_prior()

    def _encode(self, obs_raw):
        """Symlog + 3-layer encoder MLP."""
        w = self.w
        x = np.sign(obs_raw) * np.log(np.abs(obs_raw) + 1)
        if self.enc_has_bias:
            h = _silu(_ln(w["enc_linear0_w"] @ x + w["enc_linear0_b"],
                           w["enc_ln0_w"], w["enc_ln0_b"]))
            h = _silu(_ln(w["enc_linear1_w"] @ h + w["enc_linear1_b"],
                           w["enc_ln1_w"], w["enc_ln1_b"]))
            return _silu(_ln(w["enc_linear2_w"] @ h + w["enc_linear2_b"],
                              w["enc_ln2_w"], w["enc_ln2_b"]))
        else:
            h = _silu(_ln(w["enc_linear0_w"] @ x,
                           w["enc_ln0_w"], w["enc_ln0_b"]))
            h = _silu(_ln(w["enc_linear1_w"] @ h,
                           w["enc_ln1_w"], w["enc_ln1_b"]))
            return _silu(_ln(w["enc_linear2_w"] @ h,
                              w["enc_ln2_w"], w["enc_ln2_b"]))

    def _img_in(self, stoch, action):
        """img_in MLP: stoch + action -> hidden."""
        w = self.w
        cat_in = np.concatenate([stoch, np.atleast_1d(np.float32(action))])
        if "img_in_b" in w:
            return _silu(_ln(w["img_in_w"] @ cat_in + w["img_in_b"],
                              w["img_in_ln_w"], w["img_in_ln_b"]))
        else:
            return _silu(_ln(w["img_in_w"] @ cat_in,
                              w["img_in_ln_w"], w["img_in_ln_b"]))

    def _gru_step(self, h_in):
        """GRU forward pass. Handles both formats."""
        w = self.w
        N = DETER_DIM

        if "gru_w" in w:
            # dreamerv3-torch: combined weight + LayerNorm
            combined = np.concatenate([h_in, self.deter])
            out = w["gru_w"] @ combined
            out = _ln(out, w["gru_ln_w"], w["gru_ln_b"])
            reset = _sigmoid(out[:N])
            cand = np.tanh(reset * out[N : 2 * N])
            update = _sigmoid(out[2 * N :] + self.GRU_UPDATE_BIAS)
            self.deter = (update * cand + (1 - update) * self.deter).astype(np.float32)
        else:
            # Exported format: separate ih/hh weights
            gi = w["gru_ih_w"] @ h_in + w["gru_ih_b"]
            gh = w["gru_hh_w"] @ self.deter + w["gru_hh_b"]
            r = _sigmoid(gi[:N] + gh[:N])
            z = _sigmoid(gi[N : 2 * N] + gh[N : 2 * N])
            n = np.tanh(gi[2 * N :] + r * gh[2 * N :])
            self.deter = ((1 - z) * n + z * self.deter).astype(np.float32)

    def _posterior(self, embed):
        """Compute posterior from deter + embed."""
        w = self.w
        cat_post = np.concatenate([self.deter, embed])
        if "post0_b" in w:
            h = _silu(_ln(w["post0_w"] @ cat_post + w["post0_b"],
                           w["post0_ln_w"], w["post0_ln_b"]))
        else:
            h = _silu(_ln(w["post0_w"] @ cat_post,
                           w["post0_ln_w"], w["post0_ln_b"]))
        logits = (w["post_out_w"] @ h + w["post_out_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES
        )
        self.stoch = _gumbel_sample(logits)

    def _compute_prior(self):
        """Compute prior from deter + stoch."""
        w = self.w
        if "prior0_b" in w:
            h = _silu(_ln(w["prior0_w"] @ self.deter + w["prior0_b"],
                           w["prior0_ln_w"], w["prior0_ln_b"]))
        else:
            h = _silu(_ln(w["prior0_w"] @ self.deter,
                           w["prior0_ln_w"], w["prior0_ln_b"]))
        logits = (w["prior_out_w"] @ h + w["prior_out_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES
        )
        self.stoch = _gumbel_sample(logits)

    def reset(self):
        self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        self._compute_prior()

    def step(self, obs_raw, action=0.0):
        """Full RSSM step: encode -> img_step -> posterior. Returns deter copy."""
        # 1. Encode
        embed = self._encode(obs_raw)

        # 2. img_step: stoch + action -> GRU update
        act = np.zeros(1, dtype=np.float32) if self.is_first else np.atleast_1d(
            np.float32(action)
        )
        self.is_first = False
        h_in = self._img_in(self.stoch.copy(), act)
        self._gru_step(h_in)

        # 3. Posterior from deter + embed
        self._posterior(embed)

        return self.deter.copy()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_episodes(weights, D, n_episodes, blob_count, proj_dim):
    """Run episodes at dimension D, collect full hidden states + counts."""
    from counting_env_multidim import MultiDimCountingWorldEnv

    obs_size = weights["enc_linear0_w"].shape[1]
    env = MultiDimCountingWorldEnv(
        blob_count_min=blob_count,
        blob_count_max=blob_count,
        fixed_dim=D,
        proj_dim=obs_size,
    )
    model = FastRSSM(weights)

    episodes = []
    for ep in range(n_episodes):
        vec = env.reset()
        model.reset()
        done = False
        ep_data = {"h": [], "count": [], "step": []}
        step_i = 0
        while not done:
            obs = vec[:obs_size].astype(np.float32)
            count = int(env._state.grid.filled_count)
            deter = model.step(obs, 0.0)
            ep_data["h"].append(deter)
            ep_data["count"].append(count)
            ep_data["step"].append(step_i)
            vec, reward, done, info = env.step(-0.995)
            step_i += 1
        ep_data["h"] = np.stack(ep_data["h"])  # (T, 512)
        ep_data["count"] = np.array(ep_data["count"])
        ep_data["step"] = np.array(ep_data["step"])
        episodes.append(ep_data)
    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Analysis 1: Dimension importance ranking
# ---------------------------------------------------------------------------


def dimension_importance(H, counts, top_k=30):
    """Rank each of 512 dims by |correlation| with ground truth count.

    Returns sorted indices, correlations, and cumulative R² of top-k dims.
    """
    n_dims = H.shape[1]
    correlations = np.zeros(n_dims)
    for d in range(n_dims):
        if H[:, d].std() < 1e-10:
            correlations[d] = 0.0
        else:
            correlations[d] = np.corrcoef(H[:, d], counts)[0, 1]

    abs_corr = np.abs(correlations)
    ranking = np.argsort(-abs_corr)  # descending

    # Cumulative R² from ridge regression using top-k dims
    cum_r2 = []
    y = counts.astype(np.float32)
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    for k in range(1, top_k + 1):
        X_k = H[:, ranking[:k]]
        # Ridge regression (alpha=10)
        XtX = X_k.T @ X_k + 10.0 * np.eye(k)
        Xty = X_k.T @ y
        w = np.linalg.solve(XtX, Xty)
        preds = X_k @ w
        ss_res = np.sum((y - preds) ** 2)
        cum_r2.append(1.0 - ss_res / ss_tot)

    # How many dims for 95% and 99% of full-model R²
    full_r2 = cum_r2[-1] if cum_r2 else 0.0
    dims_for_95 = next((k + 1 for k, r in enumerate(cum_r2) if r >= 0.95 * full_r2), top_k)
    dims_for_99 = next((k + 1 for k, r in enumerate(cum_r2) if r >= 0.99 * full_r2), top_k)

    return {
        "ranking": ranking[:top_k].tolist(),
        "correlations": correlations[ranking[:top_k]].tolist(),
        "abs_correlations": abs_corr[ranking[:top_k]].tolist(),
        "cumulative_r2": cum_r2,
        "dims_for_95pct": dims_for_95,
        "dims_for_99pct": dims_for_99,
        "top_10_dims": ranking[:10].tolist(),
        "top_10_corr": correlations[ranking[:10]].tolist(),
        "mean_abs_corr_top20": float(abs_corr[ranking[:20]].mean()),
        "mean_abs_corr_all": float(abs_corr.mean()),
        "n_dims_above_0.1": int((abs_corr > 0.1).sum()),
        "n_dims_above_0.3": int((abs_corr > 0.3).sum()),
        "n_dims_above_0.5": int((abs_corr > 0.5).sum()),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Dimension clustering by temporal behavior
# ---------------------------------------------------------------------------


def dimension_clustering(H, counts, n_clusters=8):
    """Cluster 512 dims by temporal correlation, label clusters functionally.

    Uses hierarchical clustering on pairwise correlation of dim time series.
    Labels clusters by their mean correlation with count.
    """
    n_dims = H.shape[1]

    # Correlation matrix of dim time series
    # Use a subsample for speed if very long
    max_samples = 20000
    if H.shape[0] > max_samples:
        idx = np.random.choice(H.shape[0], max_samples, replace=False)
        idx.sort()
        H_sub = H[idx]
        c_sub = counts[idx]
    else:
        H_sub = H
        c_sub = counts

    # Standardize
    H_std = (H_sub - H_sub.mean(axis=0)) / (H_sub.std(axis=0) + 1e-10)

    # Pairwise correlation matrix (512 x 512)
    corr_mat = (H_std.T @ H_std) / H_std.shape[0]
    np.fill_diagonal(corr_mat, 1.0)

    # Convert to distance for clustering
    dist = 1.0 - np.abs(corr_mat)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)
    dist = (dist + dist.T) / 2  # ensure symmetric

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Characterize each cluster
    count_corrs = np.array([np.corrcoef(H_sub[:, d], c_sub)[0, 1]
                            if H_sub[:, d].std() > 1e-10 else 0.0
                            for d in range(n_dims)])

    clusters = {}
    for cl in range(1, n_clusters + 1):
        dims_in_cluster = np.where(labels == cl)[0]
        cl_count_corrs = count_corrs[dims_in_cluster]

        # Mean within-cluster correlation (how tightly coupled)
        if len(dims_in_cluster) > 1:
            sub_corr = corr_mat[np.ix_(dims_in_cluster, dims_in_cluster)]
            mask = ~np.eye(len(dims_in_cluster), dtype=bool)
            mean_internal_corr = float(sub_corr[mask].mean())
        else:
            mean_internal_corr = 1.0

        # Variance explained by this cluster for count prediction
        if len(dims_in_cluster) > 0:
            X_cl = H_sub[:, dims_in_cluster]
            y = c_sub.astype(np.float32)
            k = X_cl.shape[1]
            XtX = X_cl.T @ X_cl + 10.0 * np.eye(k)
            Xty = X_cl.T @ y
            w = np.linalg.solve(XtX, Xty)
            preds = X_cl @ w
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            cluster_r2 = float(1.0 - ss_res / ss_tot)
        else:
            cluster_r2 = 0.0

        # Activity level: mean std across dims
        mean_std = float(H_sub[:, dims_in_cluster].std(axis=0).mean())

        # Label by function
        mean_count_corr = float(cl_count_corrs.mean())
        abs_mean = float(np.abs(cl_count_corrs).mean())
        if abs_mean > 0.4:
            role = "counting_primary"
        elif abs_mean > 0.2:
            role = "counting_secondary"
        elif mean_std < 0.05:
            role = "near_constant"
        else:
            role = "spatial_or_other"

        clusters[int(cl)] = {
            "n_dims": int(len(dims_in_cluster)),
            "dims": dims_in_cluster.tolist(),
            "mean_count_corr": mean_count_corr,
            "abs_mean_count_corr": abs_mean,
            "mean_internal_corr": mean_internal_corr,
            "cluster_r2_for_count": cluster_r2,
            "mean_activity_std": mean_std,
            "role": role,
        }

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "dim_labels": labels.tolist(),
        "count_correlations_all": count_corrs.tolist(),
    }


# ---------------------------------------------------------------------------
# Analysis 3: Subspace tracking over time
# ---------------------------------------------------------------------------


def subspace_tracking(episodes, window_size=50, stride=10):
    """Track which dims are most count-predictive in sliding windows.

    For each window, compute per-dim |correlation| with count.
    Returns how the "counting subspace" shifts over the episode.
    """
    results = []

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]  # (T, 512)
        counts = ep["count"]
        T = len(counts)

        if T < window_size + 10:
            continue

        windows = []
        for start in range(0, T - window_size, stride):
            end = start + window_size
            H_win = H[start:end]
            c_win = counts[start:end]

            # Skip windows with constant count (no signal)
            if c_win.std() < 0.5:
                continue

            # Per-dim correlation with count in this window
            corrs = np.zeros(512)
            for d in range(512):
                if H_win[:, d].std() > 1e-10:
                    corrs[d] = np.abs(np.corrcoef(H_win[:, d], c_win)[0, 1])

            top_20 = np.argsort(-corrs)[:20]
            mean_count = float(c_win.mean())

            windows.append({
                "start": int(start),
                "end": int(end),
                "mean_count": mean_count,
                "top_20_dims": top_20.tolist(),
                "top_20_corrs": corrs[top_20].tolist(),
                "max_corr": float(corrs.max()),
                "n_dims_above_0.3": int((corrs > 0.3).sum()),
            })

        if not windows:
            continue

        # Compute subspace stability: Jaccard overlap of top-20 between consecutive windows
        overlaps = []
        for i in range(1, len(windows)):
            set_a = set(windows[i - 1]["top_20_dims"])
            set_b = set(windows[i]["top_20_dims"])
            jaccard = len(set_a & set_b) / len(set_a | set_b)
            overlaps.append(jaccard)

        # Compute subspace stability between LOW count windows and HIGH count windows
        low_windows = [w for w in windows if w["mean_count"] < 5]
        high_windows = [w for w in windows if w["mean_count"] > 10]
        if low_windows and high_windows:
            low_dims = set()
            for w in low_windows:
                low_dims.update(w["top_20_dims"][:10])
            high_dims = set()
            for w in high_windows:
                high_dims.update(w["top_20_dims"][:10])
            low_high_jaccard = len(low_dims & high_dims) / len(low_dims | high_dims) if (low_dims | high_dims) else 0
        else:
            low_high_jaccard = None

        results.append({
            "episode": ep_idx,
            "n_windows": len(windows),
            "mean_jaccard_consecutive": float(np.mean(overlaps)) if overlaps else 0.0,
            "std_jaccard_consecutive": float(np.std(overlaps)) if overlaps else 0.0,
            "low_vs_high_count_jaccard": low_high_jaccard,
            "windows": windows,
        })

    # Aggregate across episodes
    if results:
        all_jaccards = [r["mean_jaccard_consecutive"] for r in results]
        low_high = [r["low_vs_high_count_jaccard"] for r in results
                    if r["low_vs_high_count_jaccard"] is not None]
    else:
        all_jaccards = []
        low_high = []

    return {
        "window_size": window_size,
        "stride": stride,
        "per_episode": results,
        "summary": {
            "mean_consecutive_jaccard": float(np.mean(all_jaccards)) if all_jaccards else 0.0,
            "mean_low_vs_high_jaccard": float(np.mean(low_high)) if low_high else None,
            "interpretation": (
                "HIGH consecutive Jaccard (>0.7) = stable subspace. "
                "LOW low-vs-high Jaccard (<0.3) = subspace rotates with count."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Cross-dimensional comparison
# ---------------------------------------------------------------------------


def cross_dim_comparison(dim_importance_results):
    """Compare which dims carry counting signal across dimensions.

    Core question: are the counting dims the SAME across D values?
    """
    dims_available = sorted(dim_importance_results.keys())
    if len(dims_available) < 2:
        return {"error": "Need at least 2 dimensions to compare"}

    # Jaccard overlap of top-k dims between each pair of D values
    results = {}
    for k in [10, 20, 30]:
        pair_jaccards = {}
        for i, d1 in enumerate(dims_available):
            for d2 in dims_available[i + 1:]:
                set1 = set(dim_importance_results[d1]["ranking"][:k])
                set2 = set(dim_importance_results[d2]["ranking"][:k])
                jaccard = len(set1 & set2) / len(set1 | set2)
                pair_jaccards[f"D{d1}_vs_D{d2}"] = round(jaccard, 4)
        results[f"top_{k}_jaccard"] = pair_jaccards

    # Correlation of importance profiles across D values
    # Each D has a 512-dim vector of |correlation with count|
    importance_vectors = {}
    for d in dims_available:
        vec = np.zeros(512)
        for idx, corr in zip(
            dim_importance_results[d]["ranking"],
            dim_importance_results[d]["abs_correlations"],
        ):
            vec[idx] = corr
        # Fill remaining from full correlations if available
        importance_vectors[d] = vec

    profile_corrs = {}
    for i, d1 in enumerate(dims_available):
        for d2 in dims_available[i + 1:]:
            r = np.corrcoef(importance_vectors[d1], importance_vectors[d2])[0, 1]
            profile_corrs[f"D{d1}_vs_D{d2}"] = round(float(r), 4)
    results["importance_profile_correlation"] = profile_corrs

    # Shared vs unique counting dims
    all_top20 = {d: set(dim_importance_results[d]["ranking"][:20]) for d in dims_available}
    shared = set.intersection(*all_top20.values()) if all_top20 else set()
    union = set.union(*all_top20.values()) if all_top20 else set()
    results["shared_top20"] = sorted(shared)
    results["n_shared_top20"] = len(shared)
    results["n_union_top20"] = len(union)
    results["sharing_ratio"] = round(len(shared) / 20, 3) if dims_available else 0

    # Interpretation
    mean_top20_jaccard = np.mean(
        [v for v in results["top_20_jaccard"].values()]
    ) if results.get("top_20_jaccard") else 0

    if mean_top20_jaccard > 0.6:
        verdict = "SHARED_SUBSPACE"
        explanation = "Most counting dims are the same across D — probe transfer should work"
    elif mean_top20_jaccard > 0.3:
        verdict = "PARTIAL_OVERLAP"
        explanation = "Some counting dims shared, some D-specific — partial transfer possible"
    else:
        verdict = "DISJOINT_SUBSPACES"
        explanation = "Counting dims are mostly different per D — explains probe transfer failure"

    results["verdict"] = verdict
    results["explanation"] = explanation
    results["mean_top20_jaccard"] = round(float(mean_top20_jaccard), 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hidden State Anatomy")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--blob-count", type=int, default=13)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <checkpoint-dir>/anatomy.json)")
    args = parser.parse_args()

    output_path = args.output or str(Path(args.checkpoint_dir) / "anatomy.json")

    print("=" * 60)
    print("Hidden State Anatomy")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Dims: {args.dims}")
    print(f"  Episodes per dim: {args.episodes}")
    print(f"  Blob count: {args.blob_count}")
    print()

    # Load weights
    print("Loading weights...", flush=True)
    weights = load_weights(args.checkpoint_dir)
    obs_size = weights["enc_linear0_w"].shape[1]  # (512, obs_size)
    fmt = weights.get("_format", "unknown")
    print(f"  Format: {fmt}, OBS_SIZE={obs_size}")

    all_results = {
        "checkpoint": args.checkpoint_dir,
        "obs_size": obs_size,
        "episodes_per_dim": args.episodes,
        "blob_count": args.blob_count,
        "dims_analyzed": args.dims,
    }

    dim_importance_results = {}

    for D in args.dims:
        print(f"\n--- Dimension D={D} ---", flush=True)
        t0 = time.time()

        # Collect episodes
        print(f"  Collecting {args.episodes} episodes...", end=" ", flush=True)
        episodes = collect_episodes(weights, D, args.episodes, args.blob_count, args.proj_dim)
        n_steps = sum(len(ep["count"]) for ep in episodes)
        print(f"{n_steps} steps in {time.time()-t0:.1f}s")

        # Stack all hidden states
        H = np.concatenate([ep["h"] for ep in episodes])
        counts = np.concatenate([ep["count"] for ep in episodes])

        # Analysis 1: Dimension importance
        print("  Analysis 1: Dimension importance...", end=" ", flush=True)
        t1 = time.time()
        importance = dimension_importance(H, counts)
        dim_importance_results[D] = importance
        print(f"{time.time()-t1:.1f}s — top dim={importance['top_10_dims'][0]}, "
              f"|r|={importance['abs_correlations'][0]:.3f}, "
              f"dims for 95%: {importance['dims_for_95pct']}")

        # Analysis 2: Dimension clustering
        print("  Analysis 2: Dimension clustering...", end=" ", flush=True)
        t2 = time.time()
        clustering = dimension_clustering(H, counts)
        counting_clusters = [
            (cl_id, cl) for cl_id, cl in clustering["clusters"].items()
            if cl["role"].startswith("counting")
        ]
        print(f"{time.time()-t2:.1f}s — {len(counting_clusters)} counting clusters, "
              f"{sum(cl[1]['n_dims'] for cl in counting_clusters)} counting dims")

        # Analysis 3: Subspace tracking
        print("  Analysis 3: Subspace tracking...", end=" ", flush=True)
        t3 = time.time()
        tracking = subspace_tracking(episodes)
        summary = tracking["summary"]
        print(f"{time.time()-t3:.1f}s — consecutive Jaccard={summary['mean_consecutive_jaccard']:.3f}, "
              f"low-vs-high={summary['mean_low_vs_high_jaccard']}")

        # Don't save full window data (too large) — keep summary + first episode
        tracking_compact = {
            "summary": tracking["summary"],
            "per_episode_summary": [
                {
                    "episode": ep["episode"],
                    "n_windows": ep["n_windows"],
                    "mean_jaccard_consecutive": ep["mean_jaccard_consecutive"],
                    "low_vs_high_count_jaccard": ep["low_vs_high_count_jaccard"],
                }
                for ep in tracking["per_episode"]
            ],
        }
        # Keep full window data for first episode only
        if tracking["per_episode"]:
            tracking_compact["first_episode_windows"] = tracking["per_episode"][0]["windows"]

        all_results[f"D{D}"] = {
            "n_samples": int(len(counts)),
            "n_unique_counts": int(len(np.unique(counts))),
            "count_range": [int(counts.min()), int(counts.max())],
            "dimension_importance": importance,
            "dimension_clustering": clustering,
            "subspace_tracking": tracking_compact,
        }

    # Cross-dimensional comparison
    if len(args.dims) >= 2:
        print("\n--- Cross-Dimensional Comparison ---", flush=True)
        cross_dim = cross_dim_comparison(dim_importance_results)
        all_results["cross_dimensional"] = cross_dim
        print(f"  Verdict: {cross_dim['verdict']}")
        print(f"  Mean top-20 Jaccard: {cross_dim['mean_top20_jaccard']}")
        print(f"  Shared top-20 dims: {cross_dim['n_shared_top20']}/20")
        print(f"  Importance profile correlations: {cross_dim['importance_profile_correlation']}")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dim':>4} {'Top dim':>8} {'|r|':>6} {'95% dims':>9} {'#>0.3':>6} "
          f"{'Cnt clusters':>13} {'Consec J':>9} {'Lo/Hi J':>8}")
    for D in args.dims:
        r = all_results[f"D{D}"]
        imp = r["dimension_importance"]
        trk = r["subspace_tracking"]["summary"]
        cnt_cl = sum(1 for cl in r["dimension_clustering"]["clusters"].values()
                     if cl["role"].startswith("counting"))
        cnt_dims = sum(cl["n_dims"] for cl in r["dimension_clustering"]["clusters"].values()
                       if cl["role"].startswith("counting"))
        lh = trk["mean_low_vs_high_jaccard"]
        lh_str = f"{lh:.3f}" if lh is not None else "N/A"
        print(f"D={D:>2} {imp['top_10_dims'][0]:>8} {imp['abs_correlations'][0]:>6.3f} "
              f"{imp['dims_for_95pct']:>9} {imp['n_dims_above_0.3']:>6} "
              f"{cnt_cl:>5}({cnt_dims:>3}d) "
              f"{trk['mean_consecutive_jaccard']:>9.3f} {lh_str:>8}")

    if "cross_dimensional" in all_results:
        cd = all_results["cross_dimensional"]
        print(f"\nCross-dim verdict: {cd['verdict']}")
        print(f"  {cd['explanation']}")


if __name__ == "__main__":
    main()
