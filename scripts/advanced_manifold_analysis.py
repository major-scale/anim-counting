#!/usr/bin/env python3
"""
Advanced Manifold Analysis — jPCA, Gromov-Wasserstein, Prior vs Posterior.

Three principled analyses of the RSSM counting manifold:
1. jPCA: Find planes of maximal rotational dynamics (Churchland et al. 2012)
2. Gromov-Wasserstein: Geometry-aware comparison of manifold structures
3. Prior vs Posterior: Test whether anticipation is genuine prediction

Usage (randproj model):
    python3 advanced_manifold_analysis.py --weights-dir models/randproj_clean --mode randproj
    python3 advanced_manifold_analysis.py --weights-dir models/randproj_clean --mode randproj --analyses jpca gw prior
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# RSSM loading — reuse from export_deter_centroids for exported weights
# ---------------------------------------------------------------------------

STOCH_DIM = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_DIM * STOCH_CLASSES
DETER_DIM = 512


def _ln(x, w, b, eps=1e-5):
    mu = x.mean()
    var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _gumbel_sample(logits):
    uniform = np.random.uniform(1e-5, 1 - 1e-5, logits.shape).astype(np.float32)
    gumbel = -np.log(-np.log(uniform))
    sample = np.zeros_like(logits)
    idx = (logits + gumbel).argmax(axis=-1)
    for i, j in enumerate(idx):
        sample[i, j] = 1.0
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return (sample + probs - probs).flatten().astype(np.float32)


def load_exported_weights(weights_dir):
    """Load from dreamer_manifest.json + dreamer_weights.bin."""
    p = Path(weights_dir)
    with open(p / "dreamer_manifest.json") as f:
        manifest = json.load(f)
    with open(p / "dreamer_weights.bin", "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"],
                            offset=entry["offset"]).copy()
        weights[name] = arr.reshape(entry["shape"])
    return weights


class FastRSSMWithPrior:
    """RSSM that also exposes the prior distribution for analysis.

    Handles the exported weight format:
      enc_linear{0,1,2}_w, enc_norm{0,1,2}_{w,b}  (no linear bias)
      img_in_w, img_in_norm_{w,b}                   (no linear bias)
      gru_w, gru_norm_{w,b}                         (combined weight + LN)
      img_out_w, img_out_norm_{w,b}, imgs_stat_{w,b} (prior)
      obs_out_w, obs_out_norm_{w,b}, obs_stat_{w,b}  (posterior)
      deter_init_w
    """

    GRU_UPDATE_BIAS = -1.0

    def __init__(self, weights):
        self.w = weights
        self.obs_size = weights["enc_linear0_w"].shape[1]
        self.reset()

    def reset(self):
        self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        self._compute_prior()

    def _compute_prior(self):
        w = self.w
        h = _silu(_ln(w["img_out_w"] @ self.deter,
                       w["img_out_norm_w"], w["img_out_norm_b"]))
        logits = (w["imgs_stat_w"] @ h + w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES
        )
        self.stoch = _gumbel_sample(logits)

    def step(self, obs_raw, action=0.0):
        """Full RSSM step. Returns (deter, prior_stoch_probs, post_stoch_probs)."""
        w = self.w

        # 1. Encode (no linear bias, norm has bias)
        x = np.sign(obs_raw) * np.log(np.abs(obs_raw) + 1)
        h = _silu(_ln(w["enc_linear0_w"] @ x, w["enc_norm0_w"], w["enc_norm0_b"]))
        h = _silu(_ln(w["enc_linear1_w"] @ h, w["enc_norm1_w"], w["enc_norm1_b"]))
        embed = _silu(_ln(w["enc_linear2_w"] @ h, w["enc_norm2_w"], w["enc_norm2_b"]))

        # 2. img_step
        act = np.zeros(1, dtype=np.float32) if self.is_first else np.atleast_1d(
            np.float32(action))
        self.is_first = False
        cat_in = np.concatenate([self.stoch.copy(), act])
        h_in = _silu(_ln(w["img_in_w"] @ cat_in, w["img_in_norm_w"], w["img_in_norm_b"]))

        # GRU (combined weight + LayerNorm, dreamerv3-torch style)
        combined = np.concatenate([h_in, self.deter])
        out = w["gru_w"] @ combined
        out = _ln(out, w["gru_norm_w"], w["gru_norm_b"])
        N = DETER_DIM
        reset = _sigmoid(out[:N])
        cand = np.tanh(reset * out[N:2*N])
        update = _sigmoid(out[2*N:] + self.GRU_UPDATE_BIAS)
        self.deter = (update * cand + (1 - update) * self.deter).astype(np.float32)

        # 3. Prior (model's prediction before seeing obs)
        h_prior = _silu(_ln(w["img_out_w"] @ self.deter,
                             w["img_out_norm_w"], w["img_out_norm_b"]))
        prior_logits = (w["imgs_stat_w"] @ h_prior + w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES
        )
        prior_probs = np.exp(prior_logits - prior_logits.max(axis=-1, keepdims=True))
        prior_probs /= prior_probs.sum(axis=-1, keepdims=True)
        prior_stoch = prior_probs.flatten().copy()

        # 4. Posterior (after seeing obs)
        cat_post = np.concatenate([self.deter, embed])
        h_post = _silu(_ln(w["obs_out_w"] @ cat_post,
                            w["obs_out_norm_w"], w["obs_out_norm_b"]))
        post_logits = (w["obs_stat_w"] @ h_post + w["obs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES
        )
        prior_probs_post = np.exp(post_logits - post_logits.max(axis=-1, keepdims=True))
        prior_probs_post /= prior_probs_post.sum(axis=-1, keepdims=True)

        # Sample posterior
        self.stoch = _gumbel_sample(post_logits)
        post_stoch = prior_probs_post.flatten().copy()

        return self.deter.copy(), prior_stoch, post_stoch


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_episodes(weights_dir, n_episodes, mode="baseline"):
    """Collect episodes with deter states, prior/posterior stoch, and counts."""
    from counting_env_pure import CountingWorldEnv as PureCountingWorldEnv

    weights = load_exported_weights(weights_dir)
    obs_size = weights["enc_linear0_w"].shape[1]
    model = FastRSSMWithPrior(weights)

    use_randproj = (mode == "randproj")
    if use_randproj:
        from scipy.stats import ortho_group
        proj_matrix = ortho_group.rvs(obs_size,
                                      random_state=np.random.RandomState(42_000)).astype(np.float32)

    env = PureCountingWorldEnv()
    episodes = []

    for ep in range(n_episodes):
        state = env.reset()
        model.reset()
        done = False
        ep_data = {"deter": [], "prior_stoch": [], "post_stoch": [],
                   "count": [], "step": []}
        step_i = 0

        while not done:
            obs = state[:obs_size].astype(np.float32)
            if use_randproj:
                obs = (proj_matrix @ obs)[:obs_size]

            count = int(env._state.grid.filled_count)
            deter, prior_s, post_s = model.step(obs, 0.0)

            ep_data["deter"].append(deter)
            ep_data["prior_stoch"].append(prior_s)
            ep_data["post_stoch"].append(post_s)
            ep_data["count"].append(count)
            ep_data["step"].append(step_i)

            state, reward, done, info = env.step(0)
            step_i += 1

        ep_data["deter"] = np.stack(ep_data["deter"])
        ep_data["prior_stoch"] = np.stack(ep_data["prior_stoch"])
        ep_data["post_stoch"] = np.stack(ep_data["post_stoch"])
        ep_data["count"] = np.array(ep_data["count"])
        episodes.append(ep_data)

    env.close()
    return episodes, weights


# ---------------------------------------------------------------------------
# Analysis 1: jPCA — rotational dynamics
# ---------------------------------------------------------------------------

def run_jpca(episodes, n_pca_dims=6, n_jpca_planes=3):
    """jPCA: find planes of maximal rotational dynamics.

    Method (Churchland et al. 2012):
    1. PCA-reduce centroids to n_pca_dims
    2. Compute derivatives dx/dt = x_{n+1} - x_n
    3. Fit dx = M_skew @ x where M_skew is skew-symmetric
    4. Eigendecompose M_skew to find rotation planes
    """
    print("  Computing centroids...", flush=True)

    # Compute per-count centroids from all episodes
    by_count = defaultdict(list)
    for ep in episodes:
        for h, c in zip(ep["deter"], ep["count"]):
            by_count[c].append(h)

    counts = sorted(by_count.keys())
    centroids = np.stack([np.mean(by_count[c], axis=0) for c in counts])
    n_counts = len(counts)

    # Step 1: PCA reduce
    centroid_mean = centroids.mean(axis=0)
    X = centroids - centroid_mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_pca = X @ Vt[:n_pca_dims].T  # (n_counts, n_pca_dims)
    pca_explained = (S[:n_pca_dims] ** 2) / (S ** 2).sum()

    print(f"  PCA: top {n_pca_dims} explain {pca_explained.sum()*100:.1f}% "
          f"({', '.join(f'{p*100:.1f}%' for p in pca_explained)})")

    # Step 2: Derivatives (consecutive centroids)
    X_t = X_pca[:-1]   # x(t)
    dX = X_pca[1:] - X_pca[:-1]  # dx/dt

    # Step 3: Fit M_skew (skew-symmetric constraint: M = M_raw - M_raw.T) / 2
    # Solve: dX = X_t @ M.T  =>  M.T = pinv(X_t) @ dX
    M_raw = np.linalg.lstsq(X_t, dX, rcond=None)[0].T  # (n_pca_dims, n_pca_dims)
    M_skew = (M_raw - M_raw.T) / 2  # Force skew-symmetry
    M_symm = (M_raw + M_raw.T) / 2  # Symmetric part (expansion/contraction)

    # Reconstruction quality
    dX_pred_skew = X_t @ M_skew.T
    dX_pred_full = X_t @ M_raw.T
    ss_total = np.sum(dX ** 2)
    r2_skew = 1 - np.sum((dX - dX_pred_skew) ** 2) / ss_total
    r2_full = 1 - np.sum((dX - dX_pred_full) ** 2) / ss_total

    print(f"  Dynamics fit: R²(skew)={r2_skew:.4f}, R²(full)={r2_full:.4f}, "
          f"rotation fraction={r2_skew/r2_full:.3f}" if r2_full > 0 else "")

    # Step 4: Eigendecompose M_skew
    # Skew-symmetric matrices have purely imaginary eigenvalues (±iω)
    eigenvalues, eigenvectors = np.linalg.eig(M_skew)

    # Sort by |imaginary part| (rotation frequency)
    imag_parts = np.abs(eigenvalues.imag)
    order = np.argsort(-imag_parts)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Extract rotation planes (conjugate pairs)
    planes = []
    used = set()
    for i in range(len(eigenvalues)):
        if i in used:
            continue
        if abs(eigenvalues[i].imag) < 1e-8:
            continue  # Skip real eigenvalues (no rotation)

        # Find conjugate pair
        for j in range(i + 1, len(eigenvalues)):
            if j in used:
                continue
            if abs(eigenvalues[i] + eigenvalues[j]) < 1e-6:
                # Conjugate pair found
                used.add(i)
                used.add(j)

                # Rotation plane basis vectors (real and imaginary parts)
                v1 = eigenvectors[:, i].real
                v2 = eigenvectors[:, i].imag
                # Normalize
                v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                v2 = v2 / (np.linalg.norm(v2) + 1e-10)

                # Project centroids onto this plane
                proj = X_pca @ np.column_stack([v1, v2])

                # Compute rotation angle per step
                angles = []
                for k in range(len(proj) - 1):
                    dx = proj[k + 1, 0] - proj[k, 0]
                    dy = proj[k + 1, 1] - proj[k, 1]
                    angle = np.arctan2(dy, dx)
                    angles.append(np.degrees(angle))

                # Variance explained by this plane
                proj_var = np.sum(proj ** 2)
                total_var = np.sum(X_pca ** 2)
                var_explained = proj_var / total_var

                freq = abs(eigenvalues[i].imag)
                planes.append({
                    "eigenvalue": complex(eigenvalues[i]),
                    "frequency": float(freq),
                    "period_in_counts": float(2 * np.pi / freq) if freq > 0 else float("inf"),
                    "variance_explained": float(var_explained),
                    "mean_angle_per_step": float(np.mean(angles)),
                    "std_angle_per_step": float(np.std(angles)),
                    "projections": proj.tolist(),
                    "counts": counts,
                })
                if len(planes) >= n_jpca_planes:
                    break
        if len(planes) >= n_jpca_planes:
            break

    # Summary
    result = {
        "analysis": "jPCA",
        "n_counts": n_counts,
        "n_pca_dims": n_pca_dims,
        "pca_explained": pca_explained.tolist(),
        "r2_skew": float(r2_skew),
        "r2_full": float(r2_full),
        "rotation_fraction": float(r2_skew / r2_full) if r2_full > 0 else 0,
        "n_planes": len(planes),
        "planes": [{
            "frequency": p["frequency"],
            "period_in_counts": p["period_in_counts"],
            "variance_explained": p["variance_explained"],
            "mean_angle_per_step": p["mean_angle_per_step"],
            "std_angle_per_step": p["std_angle_per_step"],
            "projections": p["projections"],
            "counts": p["counts"],
        } for p in planes],
        "M_skew_norm": float(np.linalg.norm(M_skew)),
        "M_symm_norm": float(np.linalg.norm(M_symm)),
        "skew_to_symm_ratio": float(np.linalg.norm(M_skew) / (np.linalg.norm(M_symm) + 1e-10)),
    }

    if planes:
        print(f"  Found {len(planes)} rotation plane(s):")
        for i, p in enumerate(planes):
            print(f"    Plane {i+1}: freq={p['frequency']:.3f}, "
                  f"period={p['period_in_counts']:.1f} counts, "
                  f"var={p['variance_explained']*100:.1f}%, "
                  f"angle/step={p['mean_angle_per_step']:.1f}°±{p['std_angle_per_step']:.1f}°")
        print(f"  Rotation fraction: {result['rotation_fraction']:.3f} "
              f"(1.0 = pure rotation, 0.0 = pure expansion)")
    else:
        print("  No rotation planes found")

    return result


# ---------------------------------------------------------------------------
# Analysis 2: Gromov-Wasserstein distance
# ---------------------------------------------------------------------------

def run_gromov_wasserstein(episodes_list, labels, metric="euclidean"):
    """Compare manifold geometry across conditions using GW distance.

    episodes_list: list of episode lists (one per condition)
    labels: condition names (e.g., ["D=2", "D=3"])
    """
    import ot

    print("  Computing per-condition distance matrices...", flush=True)

    # Compute per-count centroids for each condition
    condition_centroids = {}
    for cond_idx, (eps, label) in enumerate(zip(episodes_list, labels)):
        by_count = defaultdict(list)
        for ep in eps:
            for h, c in zip(ep["deter"], ep["count"]):
                by_count[c].append(h)
        counts = sorted(by_count.keys())
        centroids = np.stack([np.mean(by_count[c], axis=0) for c in counts])
        condition_centroids[label] = {"centroids": centroids, "counts": counts}

    # Compute pairwise GW distances
    results = {"analysis": "gromov_wasserstein", "conditions": labels, "pairs": {}}

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if j <= i:
                continue

            C1 = condition_centroids[l1]["centroids"]
            C2 = condition_centroids[l2]["centroids"]

            # Intra-condition distance matrices
            from scipy.spatial.distance import cdist
            D1 = cdist(C1, C1, metric=metric)
            D2 = cdist(C2, C2, metric=metric)

            # Normalize to [0, 1]
            D1 = D1 / (D1.max() + 1e-10)
            D2 = D2 / (D2.max() + 1e-10)

            # Uniform weights
            n1, n2 = len(C1), len(C2)
            p = np.ones(n1) / n1
            q = np.ones(n2) / n2

            # GW distance
            gw_dist, log = ot.gromov.gromov_wasserstein2(
                D1, D2, p, q, loss_fun="square_loss", log=True
            )

            # Also compute RSA (Spearman correlation of upper triangles)
            triu1 = D1[np.triu_indices(n1, k=1)]
            triu2 = D2[np.triu_indices(n2, k=1)]
            if len(triu1) == len(triu2):
                from scipy.stats import spearmanr
                rsa, _ = spearmanr(triu1, triu2)
            else:
                rsa = None

            pair_key = f"{l1}_vs_{l2}"
            results["pairs"][pair_key] = {
                "gw_distance": float(gw_dist),
                "n_points_1": n1,
                "n_points_2": n2,
                "rsa_spearman": float(rsa) if rsa is not None else None,
                "counts_1": condition_centroids[l1]["counts"],
                "counts_2": condition_centroids[l2]["counts"],
            }
            print(f"    {pair_key}: GW={gw_dist:.6f}, RSA={rsa:.4f}" if rsa else
                  f"    {pair_key}: GW={gw_dist:.6f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Prior vs Posterior during transitions
# ---------------------------------------------------------------------------

def run_prior_vs_posterior(episodes, weights):
    """Compare prior and posterior count representations during transitions.

    If the prior shifts toward the next count BEFORE the blob lands,
    that's genuine prediction, not input encoding.
    """
    print("  Training probe on posterior stoch...", flush=True)

    # First, train a ridge probe on posterior stoch -> count
    all_post = np.concatenate([ep["post_stoch"] for ep in episodes])
    all_counts = np.concatenate([ep["count"] for ep in episodes])

    # Ridge regression
    alpha = 10.0
    X = all_post
    y = all_counts.astype(np.float32)
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    probe_w = np.linalg.solve(XtX, Xty)
    probe_b = y.mean() - probe_w @ X.mean(axis=0)

    # Evaluate probe on posterior
    preds = X @ probe_w + probe_b
    r2_post = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  Posterior probe R² = {r2_post:.4f}")

    # Evaluate same probe on prior (WITHOUT retraining)
    all_prior = np.concatenate([ep["prior_stoch"] for ep in episodes])
    preds_prior = all_prior @ probe_w + probe_b
    r2_prior = 1 - np.sum((y - preds_prior) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  Prior probe R² = {r2_prior:.4f} (using posterior-trained probe)")

    # Now the key analysis: around transitions, does the prior anticipate?
    print("  Analyzing prior anticipation at transitions...", flush=True)

    transition_results = []
    for ep_idx, ep in enumerate(episodes):
        counts = ep["count"]
        prior_s = ep["prior_stoch"]
        post_s = ep["post_stoch"]
        T = len(counts)

        for t in range(1, T):
            if counts[t] != counts[t - 1] and t > 10 and t + 10 < T:
                from_c = int(counts[t - 1])
                to_c = int(counts[t])

                # Prior and posterior projections in a window around transition
                window = range(max(0, t - 10), min(T, t + 11))
                prior_proj = np.array([prior_s[s] @ probe_w + probe_b for s in window])
                post_proj = np.array([post_s[s] @ probe_w + probe_b for s in window])
                ground_truth = np.array([counts[s] for s in window])

                # Prior projection at transition
                prior_at_t = prior_s[t] @ probe_w + probe_b
                post_at_t = post_s[t] @ probe_w + probe_b
                prior_before = prior_s[t - 1] @ probe_w + probe_b

                # Does the prior start shifting before the posterior does?
                # Measure: how much does prior projection change in pre-transition window?
                pre_prior = np.array([prior_s[s] @ probe_w + probe_b
                                      for s in range(max(0, t - 10), t)])
                pre_post = np.array([post_s[s] @ probe_w + probe_b
                                     for s in range(max(0, t - 10), t)])

                if len(pre_prior) >= 3:
                    # Linear trend in prior projection before transition
                    x_time = np.arange(len(pre_prior), dtype=np.float32)
                    prior_slope = np.polyfit(x_time, pre_prior, 1)[0]
                    post_slope = np.polyfit(x_time, pre_post, 1)[0]

                    # Anticipation: positive slope toward next count
                    direction = 1.0 if to_c > from_c else -1.0
                    prior_anticipation = prior_slope * direction
                    post_anticipation = post_slope * direction

                    transition_results.append({
                        "episode": ep_idx,
                        "step": int(t),
                        "transition": f"{from_c}->{to_c}",
                        "prior_at_t": float(prior_at_t),
                        "post_at_t": float(post_at_t),
                        "prior_before_t": float(prior_before),
                        "prior_slope_pre": float(prior_slope),
                        "post_slope_pre": float(post_slope),
                        "prior_anticipation": float(prior_anticipation),
                        "post_anticipation": float(post_anticipation),
                        "prior_projects": prior_proj.tolist(),
                        "post_projects": post_proj.tolist(),
                        "ground_truth": ground_truth.tolist(),
                    })

    # Aggregate
    if transition_results:
        prior_ants = [t["prior_anticipation"] for t in transition_results]
        post_ants = [t["post_anticipation"] for t in transition_results]
        mean_prior_ant = float(np.mean(prior_ants))
        mean_post_ant = float(np.mean(post_ants))
        frac_prior_positive = float(np.mean([a > 0 for a in prior_ants]))
        frac_post_positive = float(np.mean([a > 0 for a in post_ants]))
    else:
        mean_prior_ant = mean_post_ant = 0
        frac_prior_positive = frac_post_positive = 0

    print(f"  Prior anticipation: mean slope={mean_prior_ant:.4f}, "
          f"positive in {frac_prior_positive*100:.0f}% of transitions")
    print(f"  Post anticipation:  mean slope={mean_post_ant:.4f}, "
          f"positive in {frac_post_positive*100:.0f}% of transitions")
    if mean_prior_ant > 0.01:
        print(f"  VERDICT: Prior shows genuine anticipation (slope > 0)")
    else:
        print(f"  VERDICT: No clear anticipation in prior")

    result = {
        "analysis": "prior_vs_posterior",
        "r2_posterior": float(r2_post),
        "r2_prior": float(r2_prior),
        "n_transitions": len(transition_results),
        "mean_prior_anticipation": mean_prior_ant,
        "mean_post_anticipation": mean_post_ant,
        "frac_prior_positive": frac_prior_positive,
        "frac_post_positive": frac_post_positive,
        # Keep first 5 transitions for detail
        "example_transitions": transition_results[:5],
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced Manifold Analysis")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--mode", choices=["baseline", "randproj"], default="baseline")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--analyses", nargs="+", default=["jpca", "gw", "prior"],
                        choices=["jpca", "gw", "prior"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.weights_dir) / "advanced_analysis.json"
    )

    print("=" * 60)
    print("Advanced Manifold Analysis")
    print("=" * 60)
    print(f"  Weights: {args.weights_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Analyses: {args.analyses}")
    print()

    # Collect episodes
    print("Collecting episodes...", flush=True)
    t0 = time.time()
    episodes, weights = collect_episodes(args.weights_dir, args.episodes, args.mode)
    n_steps = sum(len(ep["count"]) for ep in episodes)
    print(f"  {n_steps} steps in {time.time()-t0:.1f}s")

    results = {
        "weights_dir": args.weights_dir,
        "mode": args.mode,
        "episodes": args.episodes,
        "n_steps": n_steps,
    }

    # Analysis 1: jPCA
    if "jpca" in args.analyses:
        print("\n--- jPCA: Rotational Dynamics ---", flush=True)
        t1 = time.time()
        results["jpca"] = run_jpca(episodes)
        print(f"  Done in {time.time()-t1:.1f}s")

    # Analysis 2: Gromov-Wasserstein (self-comparison: first half vs second half episodes)
    if "gw" in args.analyses:
        print("\n--- Gromov-Wasserstein: Geometry Comparison ---", flush=True)
        t1 = time.time()
        half = len(episodes) // 2
        if half >= 2:
            results["gw"] = run_gromov_wasserstein(
                [episodes[:half], episodes[half:]],
                ["episodes_1st_half", "episodes_2nd_half"]
            )
            print(f"  (Self-comparison: split episodes into halves to test consistency)")
        else:
            print("  Skipped: need >= 4 episodes for split comparison")
        print(f"  Done in {time.time()-t1:.1f}s")

    # Analysis 3: Prior vs Posterior
    if "prior" in args.analyses:
        print("\n--- Prior vs Posterior: Anticipation Test ---", flush=True)
        t1 = time.time()
        results["prior_vs_posterior"] = run_prior_vs_posterior(episodes, weights)
        print(f"  Done in {time.time()-t1:.1f}s")

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x:
                  x.tolist() if hasattr(x, "tolist") else
                  str(x) if isinstance(x, complex) else x)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
