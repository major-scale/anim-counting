#!/usr/bin/env python3
"""
Evaluate an untrained DreamerV3 agent using a trained agent's policy.

The trained agent drives the environment (generating the observation stream),
while the untrained agent observes the same sequence and produces h_t vectors.
This ensures both trained and untrained models see identical data.

Usage:
    python eval_untrained_v2.py <trained_checkpoint_dir> [--episodes_per 5]
"""

import os, sys, time, pathlib, argparse, json
import numpy as np
import torch

sys.path.insert(0, "/workspace/dreamerv3-torch")
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
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])
    config.num_actions = 1
    return config


def create_agent(config, load_checkpoint=True):
    """Create a DreamerV3 agent, optionally loading checkpoint."""
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

    if load_checkpoint:
        checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])

    agent.eval()
    return agent


def run_dual_eval(trained_agent, untrained_agent, blob_counts, episodes_per, seed=42):
    """
    Run eval episodes using the trained agent's policy.
    Both agents observe the same data. Collect h_t from both.
    """
    from envs.counting import CountingWorld

    trained_h_t = []
    untrained_h_t = []
    all_counts = []
    all_episode_ids = []
    all_timesteps = []
    total_eps = len(blob_counts) * episodes_per
    ep_num = 0
    t0 = time.time()

    for n_blobs in blob_counts:
        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        env = CountingWorld("counting_world", seed=seed)

        for ep in range(episodes_per):
            obs_raw = env.reset()
            trained_state = None
            untrained_state = None
            done = False
            step_in_ep = 0

            while not done:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else int(vec[GRID_FILLED_RAW_IDX])

                obs_dict = {
                    "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
                    "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]], dtype=torch.float32),
                    "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]], dtype=torch.float32),
                    "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]], dtype=torch.float32),
                }
                reset = np.array([obs_raw["is_first"]])

                # Forward pass through TRAINED agent (for policy + h_t)
                with torch.no_grad():
                    trained_policy, trained_state = trained_agent(obs_dict, reset, trained_state, training=False)

                # Forward pass through UNTRAINED agent (for h_t only, policy ignored)
                with torch.no_grad():
                    _, untrained_state = untrained_agent(obs_dict, reset, untrained_state, training=False)

                # Extract h_t from both
                t_latent, _ = trained_state
                u_latent, _ = untrained_state
                trained_h_t.append(t_latent["deter"].cpu().numpy().squeeze(0))
                untrained_h_t.append(u_latent["deter"].cpu().numpy().squeeze(0))
                all_counts.append(marked_count)
                all_episode_ids.append(ep_num)
                all_timesteps.append(step_in_ep)

                # Use TRAINED agent's action to step the env
                action = trained_policy["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)
                step_in_ep += 1

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num}/{total_eps} (blobs={n_blobs}, steps={step_in_ep}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return {
        "trained_h_t": np.stack(trained_h_t),
        "untrained_h_t": np.stack(untrained_h_t),
        "counts": np.array(all_counts, dtype=np.int32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
    }


def compute_metrics(h_t, counts, label=""):
    """Full measurement battery."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    # Centroids
    max_count = int(counts.max())
    centroids = []
    valid_counts = []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    centroids = np.stack(centroids)
    valid_counts = np.array(valid_counts)
    n = len(valid_counts)

    print(f"  [{label}] Unique counts: {n} (range {valid_counts[0]}-{valid_counts[-1]})")

    if n < 3:
        return {"error": "too_few_counts", "n_counts": n}

    # GHE
    best_ghe = None
    best_geo = None
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
        except:
            continue

    # Arc-length R²
    r2 = None
    if best_geo:
        arc_length = np.cumsum([0] + best_geo)
        count_values = np.arange(n)
        coeffs = np.polyfit(count_values, arc_length, 1)
        fit_line = np.polyval(coeffs, count_values)
        ss_res = np.sum((arc_length - fit_line)**2)
        ss_tot = np.sum((arc_length - np.mean(arc_length))**2)
        r2 = 1 - ss_res / ss_tot

    # Persistent homology
    try:
        from ripser import ripser
        result = ripser(centroids, maxdim=1)
        h0 = result['dgms'][0]
        h1 = result['dgms'][1]
        beta_0 = len(h0[h0[:, 1] == np.inf])
        beta_1 = len([1 for b, d in h1 if d == np.inf]) if len(h1) > 0 else 0
    except:
        beta_0 = None
        beta_1 = None

    # RSA
    rdm_agent = squareform(pdist(centroids, metric='euclidean'))
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu_idx = np.triu_indices(n, k=1)
    rsa_rho, _ = spearmanr(rdm_agent[triu_idx], rdm_ideal[triu_idx])

    # PCA
    pca = PCA(n_components=min(5, n))
    pca.fit(centroids)
    var_explained = pca.explained_variance_ratio_

    return {
        "ghe": float(best_ghe) if best_ghe else None,
        "arc_length_r2": float(r2) if r2 else None,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "rsa_spearman": float(rsa_rho),
        "n_counts": n,
        "n_timesteps": len(counts),
        "count_range": [int(counts.min()), int(counts.max())],
        "pca_variance_explained": [float(v) for v in var_explained],
    }


def main():
    parser = argparse.ArgumentParser(description="Dual eval: trained vs untrained on same data")
    parser.add_argument("checkpoint_dir", type=str, help="Trained agent checkpoint")
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--untrained_seed", type=int, default=12345,
                        help="Random seed for untrained agent initialization")
    parser.add_argument("--output_dir", type=str, default="/workspace/bridge/artifacts/h_t_data")
    args = parser.parse_args()

    print("=" * 60)
    print("DUAL EVAL: TRAINED vs UNTRAINED (SAME OBSERVATIONS)")
    print("=" * 60)
    print(f"  Trained checkpoint: {args.checkpoint_dir}")
    print(f"  Untrained seed:     {args.untrained_seed}")
    print(f"  Episodes per count: {args.episodes_per}")
    print(f"  Device:             {_detect_device()}")
    print()

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_BRIDGE_SCRIPT'] = '/workspace/bridge/dist/bridge.js'
    os.environ['COUNTING_MAX_STEPS'] = '10000'

    config = load_config(args.checkpoint_dir)

    print("Loading trained agent...")
    trained_agent = create_agent(config, load_checkpoint=True)

    print("Creating untrained agent (random weights)...")
    torch.manual_seed(args.untrained_seed)
    np.random.seed(args.untrained_seed)
    untrained_agent = create_agent(config, load_checkpoint=False)
    print(f"  Parameters: {sum(p.numel() for p in untrained_agent.parameters()):,}")
    print()

    print("Running dual eval (trained policy drives env, both agents observe)...")
    t0 = time.time()
    data = run_dual_eval(
        trained_agent, untrained_agent, BLOB_COUNTS,
        args.episodes_per, seed=args.seed
    )
    eval_time = time.time() - t0
    print(f"Eval done: {len(data['counts'])} timesteps in {eval_time:.0f}s")
    print()

    # Compute metrics for both
    print("Computing TRAINED metrics...")
    trained_metrics = compute_metrics(data["trained_h_t"], data["counts"], label="trained")

    print("Computing UNTRAINED metrics...")
    untrained_metrics = compute_metrics(data["untrained_h_t"], data["counts"], label="untrained")

    # Display comparison
    print(f"\n{'=' * 60}")
    print(f"RESULTS COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'Trained':>10} {'Untrained':>10} {'Delta':>10}")
    print("-" * 52)

    for key in ["ghe", "arc_length_r2", "rsa_spearman"]:
        t_val = trained_metrics.get(key)
        u_val = untrained_metrics.get(key)
        if t_val is not None and u_val is not None:
            delta = t_val - u_val
            print(f"{key:<20} {t_val:>10.4f} {u_val:>10.4f} {delta:>+10.4f}")
        else:
            print(f"{key:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    for key in ["beta_0", "beta_1"]:
        t_val = trained_metrics.get(key)
        u_val = untrained_metrics.get(key)
        print(f"{key:<20} {str(t_val):>10} {str(u_val):>10}")

    t_pca = trained_metrics.get("pca_variance_explained", [])
    u_pca = untrained_metrics.get("pca_variance_explained", [])
    if t_pca and u_pca:
        print(f"{'PCA PC1':<20} {t_pca[0]:>10.3f} {u_pca[0]:>10.3f}")
        if len(t_pca) > 1 and len(u_pca) > 1:
            print(f"{'PCA PC1+PC2':<20} {sum(t_pca[:2]):>10.3f} {sum(u_pca[:2]):>10.3f}")

    # Save data
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "untrained_baseline.npz",
        h_t=data["untrained_h_t"],
        counts=data["counts"],
        episode_ids=data["episode_ids"],
        timesteps=data["timesteps"],
        train_step=np.array([0]),
        label=np.array(["untrained_baseline"]),
    )

    # Also save the trained h_t from same run for direct comparison
    np.savez_compressed(
        output_dir / "trained_same_run.npz",
        h_t=data["trained_h_t"],
        counts=data["counts"],
        episode_ids=data["episode_ids"],
        timesteps=data["timesteps"],
    )

    # Save metrics
    combined_metrics = {
        "trained": trained_metrics,
        "untrained": untrained_metrics,
        "eval_time_seconds": round(eval_time, 1),
        "same_observations": True,
        "trained_checkpoint": args.checkpoint_dir,
        "untrained_seed": args.untrained_seed,
    }
    with open(output_dir / "untrained_comparison.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
