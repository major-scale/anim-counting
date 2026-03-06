#!/usr/bin/env python3
"""
Evaluate an untrained (random weights) DreamerV3 agent.
Establishes the "floor" — what geometric structure exists before learning.

Usage:
    python eval_untrained.py [--episodes_per 5] [--seed 42]
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


def create_untrained_agent(seed=42):
    """Create a DreamerV3 agent with random weights (no training)."""
    import gym, gym.spaces
    from counting_env_pure import OBS_SIZE
    from dreamer import make_dataset, Dreamer

    torch.manual_seed(seed)
    np.random.seed(seed)

    configs_dir = os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")
    configs = yaml.safe_load(pathlib.Path(configs_dir, "configs.yaml").read_text())
    defaults = {}
    for name in ["defaults", "counting_continuous"]:
        for k, v in configs[name].items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v

    # Use a temp logdir
    logdir = pathlib.Path("/tmp/untrained_dreamer")
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "train_eps").mkdir(exist_ok=True)

    defaults["logdir"] = str(logdir)
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

    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(0.0, 25.0, (OBS_SIZE,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    logger = tools.Logger(logdir, 0)
    train_eps = {}  # Empty — no training data
    train_dataset = make_dataset(train_eps, config)

    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    return agent, config


def run_eval_untrained(agent, blob_counts, episodes_per, seed=42, max_steps_per_ep=2000):
    """Run eval with untrained agent. Uses shorter episodes since random policy is slow."""
    from envs.counting import CountingWorld

    all_h_t = []
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
            state = None
            done = False
            step_in_ep = 0
            while not done and step_in_ep < max_steps_per_ep:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else int(vec[GRID_FILLED_RAW_IDX])

                obs_dict = {
                    "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
                    "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]], dtype=torch.float32),
                    "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]], dtype=torch.float32),
                    "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]], dtype=torch.float32),
                }
                reset = np.array([obs_raw["is_first"]])
                with torch.no_grad():
                    policy_output, state = agent(obs_dict, reset, state, training=False)

                latent, _ = state
                h_t = latent["deter"].cpu().numpy().squeeze(0)
                all_h_t.append(h_t)
                all_counts.append(marked_count)
                all_episode_ids.append(ep_num)
                all_timesteps.append(step_in_ep)

                action = policy_output["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)
                step_in_ep += 1

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num}/{total_eps} (blobs={n_blobs}, steps={step_in_ep}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return (np.stack(all_h_t), np.array(all_counts, dtype=np.int32),
            np.array(all_episode_ids, dtype=np.int32), np.array(all_timesteps, dtype=np.int32))


def compute_metrics(h_t, counts):
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

    print(f"  Unique counts with data: {n} (range {valid_counts[0]}-{valid_counts[-1]})")
    count_distribution = {int(c): int((counts == c).sum()) for c in valid_counts}
    print(f"  Count distribution: {count_distribution}")

    if n < 3:
        print("  WARNING: fewer than 3 unique counts, metrics may be unreliable")
        return {"error": "too_few_counts", "n_counts": n}

    # GHE (geodesic)
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
    parser = argparse.ArgumentParser(description="Evaluate untrained DreamerV3")
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Max steps per episode (random policy is slow)")
    parser.add_argument("--output_dir", type=str, default="/workspace/bridge/artifacts/h_t_data")
    args = parser.parse_args()

    print("=" * 60)
    print("UNTRAINED DREAMERV3 BASELINE")
    print("=" * 60)
    print(f"  Seed:          {args.seed}")
    print(f"  Episodes per:  {args.episodes_per}")
    print(f"  Max steps/ep:  {args.max_steps}")
    print(f"  Device:        {_detect_device()}")
    print()

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_BRIDGE_SCRIPT'] = '/workspace/bridge/dist/bridge.js'
    os.environ['COUNTING_MAX_STEPS'] = str(args.max_steps)

    print("Creating untrained agent...")
    agent, config = create_untrained_agent(seed=args.seed)
    print(f"  Agent created with random weights")
    print(f"  Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print()

    print("Running eval episodes (random policy)...")
    t0 = time.time()
    h_t, counts, episode_ids, timesteps = run_eval_untrained(
        agent, BLOB_COUNTS, args.episodes_per, seed=args.seed,
        max_steps_per_ep=args.max_steps
    )
    eval_time = time.time() - t0
    print(f"Eval done: {len(counts)} timesteps in {eval_time:.0f}s")
    print()

    print("Computing metrics...")
    results = compute_metrics(h_t, counts)
    results["eval_time_seconds"] = round(eval_time, 1)
    results["train_steps"] = 0

    print(f"\n{'=' * 60}")
    print(f"UNTRAINED BASELINE RESULTS")
    print(f"{'=' * 60}")
    if "error" not in results:
        print(f"  GHE:            {results['ghe']:.4f}" if results['ghe'] else "  GHE:            N/A (disconnected graph)")
        print(f"  Arc-length R²:  {results['arc_length_r2']:.4f}" if results['arc_length_r2'] else "  Arc-length R²:  N/A")
        print(f"  β₀={results['beta_0']}, β₁={results['beta_1']}")
        print(f"  RSA Spearman:   {results['rsa_spearman']:.4f}")
        print(f"  PCA var:        {[f'{v:.3f}' for v in results['pca_variance_explained'][:3]]}")
    else:
        print(f"  ERROR: {results['error']}")
    print(f"  Count range:    {results.get('count_range', 'N/A')}")
    print(f"  Timesteps:      {results.get('n_timesteps', 'N/A')}")

    # Save h_t data
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "untrained_baseline.npz",
        h_t=h_t,
        counts=counts,
        episode_ids=episode_ids,
        timesteps=timesteps,
        train_step=np.array([0]),
        label=np.array(["untrained_baseline"]),
    )
    print(f"\nSaved h_t data to {output_dir / 'untrained_baseline.npz'}")

    # Save metrics
    metrics_path = output_dir / "untrained_baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
