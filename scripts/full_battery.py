#!/usr/bin/env python3
"""
Full measurement battery for counting manifold analysis.

Collects h_t from a trained checkpoint, then runs:
  - PCA projection + figure
  - PaCMAP projection + figure (if available)
  - TriMap projection + figure (if available)
  - Per-projection R² (coordinates vs ground truth count)
  - RSA (Spearman correlation of RDMs)
  - Nearest neighbor accuracy
  - Step size coefficient of variation
  - Linear probe R²
  - Probe SNR (between-count / within-count variance along probe direction)
  - GHE, arc-length R², topology (β₀, β₁)

Usage:
    python full_battery.py <checkpoint_dir> [--output_dir <dir>] [--episodes_per 5] [--label randproj_s0]

Environment variables respected: COUNTING_RANDOM_PROJECT, COUNTING_MASK_COUNT, etc.
"""

import os, sys, time, pathlib, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch"))
sys.path.insert(0, "/workspace/bridge/scripts")

from ruamel.yaml import YAML as _YAML
class _YAMLCompat:
    @staticmethod
    def safe_load(text):
        y = _YAML(typ='safe', pure=True)
        return y.load(text)
yaml = _YAMLCompat()

import tools, networks

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 8

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.linear_model import Ridge
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

BLOB_COUNTS = [3, 5, 8, 10, 12, 15, 20, 25]
GRID_FILLED_RAW_IDX = 81


def _detect_device():
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_config(checkpoint_dir):
    configs_dir = os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")
    configs = yaml.safe_load(pathlib.Path(configs_dir, "configs.yaml").read_text())
    defaults = {}
    for name in ["defaults", "counting_continuous"]:
        for k, v in configs[name].items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v
    defaults["logdir"] = str(checkpoint_dir)
    defaults["device"] = _detect_device()
    defaults["compile"] = False
    defaults["steps"] = 100000
    defaults["action_repeat"] = 1
    defaults["eval_every"] = 10000
    defaults["time_limit"] = 6000
    defaults["log_every"] = 1000
    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])
    config.num_actions = 1
    return config


def create_agent(config):
    import gym, gym.spaces
    from counting_env_pure import OBS_SIZE
    from dreamer import make_dataset, Dreamer

    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(0.0, 25.0, (OBS_SIZE,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    logdir = pathlib.Path(config.logdir)
    logger = tools.Logger(logdir, 0)
    train_eps = tools.load_episodes(logdir / "train_eps", limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return agent


def collect_h_t(agent, blob_counts, episodes_per, device, seed=42):
    """Collect hidden states across blob counts."""
    from envs.counting import CountingWorld

    all_h_t, all_counts, all_ep_ids, all_timesteps = [], [], [], []
    ep_num = 0
    t0 = time.time()

    for n_blobs in blob_counts:
        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        env = CountingWorld("counting_world", seed=seed)

        for ep in range(episodes_per):
            obs_raw = env.reset()
            state = None
            done = False
            step = 0

            while not done:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else int(vec[GRID_FILLED_RAW_IDX])

                obs_dict = {
                    "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device),
                    "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]], dtype=torch.float32).to(device),
                    "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]], dtype=torch.float32).to(device),
                    "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]], dtype=torch.float32).to(device),
                }
                reset = np.array([obs_raw["is_first"]])

                with torch.no_grad():
                    policy_output, state = agent(obs_dict, reset, state, training=False)

                latent, _ = state
                h_t = latent["deter"].cpu().numpy().squeeze(0)

                all_h_t.append(h_t)
                all_counts.append(marked_count)
                all_ep_ids.append(ep_num)
                all_timesteps.append(step)

                obs_raw, reward, done, info = env.step(policy_output["action"].cpu().numpy())
                step += 1

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num} (blobs={n_blobs}, steps={step}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return {
        "h_t": np.stack(all_h_t),
        "counts": np.array(all_counts, dtype=np.int32),
        "episode_ids": np.array(all_ep_ids, dtype=np.int32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
    }


def compute_centroids(h_t, counts):
    max_count = int(counts.max())
    centroids, valid_counts = [], []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def run_battery(h_t, counts, centroids, valid_counts, output_dir, label):
    """Run full measurement battery. Returns metrics dict."""
    n = len(valid_counts)
    results = {"label": label, "n_counts": n, "n_timesteps": len(counts)}

    # --- GHE + arc-length R² ---
    best_ghe, best_geo = None, None
    for k in [4, 5, 6, 7, 8]:
        try:
            A = kneighbors_graph(centroids, n_neighbors=min(k, n-1), mode='distance')
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

    results["ghe"] = float(best_ghe) if best_ghe else None

    if best_geo:
        arc_length = np.cumsum([0] + best_geo)
        coeffs = np.polyfit(np.arange(n), arc_length, 1)
        fit_line = np.polyval(coeffs, np.arange(n))
        ss_res = np.sum((arc_length - fit_line)**2)
        ss_tot = np.sum((arc_length - np.mean(arc_length))**2)
        results["arc_length_r2"] = float(1 - ss_res / ss_tot)
    else:
        results["arc_length_r2"] = None

    # --- Topology ---
    try:
        from ripser import ripser
        rips = ripser(centroids, maxdim=1)
        h0, h1 = rips['dgms'][0], rips['dgms'][1]
        results["beta_0"] = int(len(h0[h0[:, 1] == np.inf]))
        results["beta_1"] = int(len([1 for b, d in h1 if d == np.inf])) if len(h1) > 0 else 0
    except Exception:
        results["beta_0"] = None
        results["beta_1"] = None

    # --- RSA ---
    rdm_agent = squareform(pdist(centroids, metric='euclidean'))
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu = np.triu_indices(n, k=1)
    rsa_rho, _ = spearmanr(rdm_agent[triu], rdm_ideal[triu])
    results["rsa_spearman"] = float(rsa_rho)

    # --- PCA ---
    pca = PCA(n_components=min(10, n))
    pca.fit(centroids)
    results["pca_variance_explained"] = [float(v) for v in pca.explained_variance_ratio_]
    results["pca_pc1"] = float(pca.explained_variance_ratio_[0])

    # --- Linear probe R² (from full h_t, not centroids) ---
    from sklearn.model_selection import cross_val_score
    ridge = Ridge(alpha=1.0)
    cv_scores = cross_val_score(ridge, h_t, counts, cv=5, scoring='r2')
    results["linear_probe_r2"] = float(np.mean(cv_scores))

    # --- Probe SNR (linear decodability quality) ---
    # Fit probe on full data to get weight vector, then measure how cleanly
    # count information separates from nuisance variation along the probe direction.
    # SNR = between-count variance / mean within-count variance of probe projections.
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(h_t, counts)
    probe_w = ridge_full.coef_  # (512,)
    probe_w_norm = probe_w / (np.linalg.norm(probe_w) + 1e-12)
    projections = h_t @ probe_w_norm  # scalar per sample

    unique_counts = np.unique(counts)
    per_count_means = []
    per_count_vars = []
    for c in unique_counts:
        mask = counts == c
        proj_c = projections[mask]
        if len(proj_c) > 1:
            per_count_means.append(np.mean(proj_c))
            per_count_vars.append(np.var(proj_c))
    between_var = np.var(per_count_means)
    within_var = np.mean(per_count_vars)
    probe_snr = float(between_var / within_var) if within_var > 0 else float('inf')
    results["probe_snr"] = round(probe_snr, 1)

    # --- Nearest neighbor accuracy ---
    # For each timestep, check if 1-NN is from adjacent count (±1)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn.fit(h_t)
    distances, indices = nn.kneighbors(h_t)
    nn_counts = counts[indices[:, 1]]  # nearest neighbor (not self)
    adjacent = np.abs(nn_counts.astype(int) - counts.astype(int)) <= 1
    results["nn_accuracy"] = float(np.mean(adjacent))

    # --- Step size CV ---
    if best_geo:
        results["step_size_cv"] = float(np.std(best_geo) / np.mean(best_geo))
        results["step_sizes"] = [float(s) for s in best_geo]
    else:
        results["step_size_cv"] = None

    # --- Per-projection R² ---
    projection_r2 = {}

    # PCA R²
    pca2 = PCA(n_components=2)
    pca_emb = pca2.fit_transform(centroids)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(pca_emb, valid_counts)
    projection_r2["pca"] = float(lr.score(pca_emb, valid_counts))

    # PaCMAP
    try:
        import pacmap
        pac = pacmap.PaCMAP(n_components=2, n_neighbors=min(10, n-1))
        pac_emb = pac.fit_transform(centroids)
        lr.fit(pac_emb, valid_counts)
        projection_r2["pacmap"] = float(lr.score(pac_emb, valid_counts))
    except ImportError:
        projection_r2["pacmap"] = None

    # TriMap
    try:
        import trimap
        tri = trimap.TRIMAP(n_dims=2)
        tri_emb = tri.fit_transform(centroids)
        lr.fit(tri_emb, valid_counts)
        projection_r2["trimap"] = float(lr.score(tri_emb, valid_counts))
    except ImportError:
        projection_r2["trimap"] = None

    results["projection_r2"] = projection_r2

    # =========================================================================
    # FIGURES
    # =========================================================================
    fig_dir = pathlib.Path(output_dir) / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmap = 'viridis'

    # --- PCA figure ---
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(pca_emb[:, 0], pca_emb[:, 1], c=valid_counts,
                    cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
    for i in range(n - 1):
        ax.plot([pca_emb[i, 0], pca_emb[i+1, 0]],
                [pca_emb[i, 1], pca_emb[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=7)
    pv = pca2.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({pv[0]:.0%})', fontsize=8)
    ax.set_ylabel(f'PC2 ({pv[1]:.0%})', fontsize=8)
    ax.set_title(f'PCA — {label}', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / 'pca.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- PaCMAP figure ---
    if projection_r2.get("pacmap") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        sc = ax.scatter(pac_emb[:, 0], pac_emb[:, 1], c=valid_counts,
                        cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
        for i in range(n - 1):
            ax.plot([pac_emb[i, 0], pac_emb[i+1, 0]],
                    [pac_emb[i, 1], pac_emb[i+1, 1]],
                    'k-', alpha=0.2, linewidth=0.5, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=7)
        ax.set_xlabel('PaCMAP 1', fontsize=8)
        ax.set_ylabel('PaCMAP 2', fontsize=8)
        ax.set_title(f'PaCMAP — {label}', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'pacmap.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- TriMap figure ---
    if projection_r2.get("trimap") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        sc = ax.scatter(tri_emb[:, 0], tri_emb[:, 1], c=valid_counts,
                        cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
        for i in range(n - 1):
            ax.plot([tri_emb[i, 0], tri_emb[i+1, 0]],
                    [tri_emb[i, 1], tri_emb[i+1, 1]],
                    'k-', alpha=0.2, linewidth=0.5, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=7)
        ax.set_xlabel('TriMap 1', fontsize=8)
        ax.set_ylabel('TriMap 2', fontsize=8)
        ax.set_title(f'TriMap — {label}', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'trimap.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- RDM figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(rdm_agent, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title(f'Agent RDM ($\\rho$={rsa_rho:.3f})', fontsize=8, fontweight='bold')
    ax1.set_xlabel('Count', fontsize=7)
    ax1.set_ylabel('Count', fontsize=7)

    im2 = ax2.imshow(rdm_ideal, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title('Ideal ordinal RDM', fontsize=8, fontweight='bold')
    ax2.set_xlabel('Count', fontsize=7)
    ax2.set_ylabel('Count', fontsize=7)

    plt.tight_layout()
    fig.savefig(fig_dir / 'rdm.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Arc-length figure ---
    if best_geo:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        arc_length = np.cumsum([0] + best_geo)
        ax.scatter(valid_counts, arc_length, c=valid_counts, cmap=cmap, s=20,
                   edgecolors='k', linewidths=0.3, zorder=2)
        coeffs = np.polyfit(valid_counts, arc_length, 1)
        fit_line = np.polyval(coeffs, valid_counts)
        r2 = results["arc_length_r2"]
        ax.plot(valid_counts, fit_line, 'k--', linewidth=1, alpha=0.7, label=f'$R^2$ = {r2:.3f}')
        ax.set_xlabel('Count', fontsize=8)
        ax.set_ylabel('Geodesic arc-length', fontsize=8)
        ax.set_title(f'Arc-length linearity — {label}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(fig_dir / 'arc_length.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- Step size figure ---
    if best_geo:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.bar(range(n-1), best_geo, color='#2ca02c', edgecolor='k', linewidth=0.3)
        ax.axhline(y=np.mean(best_geo), color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Count transition', fontsize=8)
        ax.set_ylabel('Geodesic step size', fontsize=8)
        cv = results["step_size_cv"]
        ax.set_title(f'Step sizes (CV={cv:.2f}) — {label}', fontsize=8, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'step_sizes.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Figures saved to {fig_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Full measurement battery")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--label", type=str, default="analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else ckpt_dir / "battery"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FULL MEASUREMENT BATTERY: {args.label}")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Episodes:   {args.episodes_per} per blob count")
    print(f"  Blob counts: {BLOB_COUNTS}")
    print(f"  Device:     {_detect_device()}")
    rand_proj = os.environ.get("COUNTING_RANDOM_PROJECT", "false")
    rand_perm = os.environ.get("COUNTING_RANDOM_PERMUTE", "false")
    print(f"  Random proj: {rand_proj}")
    print(f"  Random perm: {rand_perm}")
    print()

    # Env setup
    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_MAX_STEPS'] = '10000'

    print("Loading agent...")
    config = load_config(ckpt_dir)
    agent = create_agent(config)
    print()

    print("Collecting hidden states...")
    t0 = time.time()
    data = collect_h_t(agent, BLOB_COUNTS, args.episodes_per, config.device, seed=args.seed)
    collect_time = time.time() - t0
    print(f"Collected {len(data['counts'])} timesteps in {collect_time:.0f}s")
    print()

    # Save raw data
    np.savez_compressed(output_dir / f"{args.label}.npz", **data)

    # Compute centroids
    centroids, valid_counts = compute_centroids(data['h_t'], data['counts'])
    print(f"Centroids: {len(valid_counts)} counts ({valid_counts[0]}-{valid_counts[-1]})")
    print()

    # Run battery
    print("Running measurement battery...")
    results = run_battery(data['h_t'], data['counts'], centroids, valid_counts,
                          output_dir, args.label)
    results["collect_time_seconds"] = round(collect_time, 1)

    # Save results
    with open(output_dir / f"{args.label}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  GHE:              {results.get('ghe', 'N/A')}")
    print(f"  Arc-length R²:    {results.get('arc_length_r2', 'N/A')}")
    print(f"  Topology:         β₀={results.get('beta_0')}, β₁={results.get('beta_1')}")
    print(f"  RSA (Spearman):   {results.get('rsa_spearman', 'N/A')}")
    print(f"  PCA PC1:          {results.get('pca_pc1', 'N/A')}")
    print(f"  Linear probe R²:  {results.get('linear_probe_r2', 'N/A')}")
    print(f"  Probe SNR:        {results.get('probe_snr', 'N/A')}")
    print(f"  NN accuracy:      {results.get('nn_accuracy', 'N/A')}")
    print(f"  Step size CV:     {results.get('step_size_cv', 'N/A')}")
    pr2 = results.get('projection_r2', {})
    print(f"  Projection R²:    PCA={pr2.get('pca', 'N/A')}, PaCMAP={pr2.get('pacmap', 'N/A')}, TriMap={pr2.get('trimap', 'N/A')}")
    print()
    print(f"Results saved to: {output_dir / f'{args.label}_metrics.json'}")


if __name__ == "__main__":
    main()
