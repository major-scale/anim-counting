#!/usr/bin/env python3
"""
Quick GHE check for embodied agent checkpoints.
Fork of quick_ghe.py adapted for num_actions=2 and embodied env.

Usage:
    python quick_ghe_embodied.py <checkpoint_dir> [--episodes_per 5] [--arrangement grid]
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
COUNT_IDX = 78


def _detect_device():
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _coerce_yaml_numbers(d):
    """PyYAML 6.0+ parses scientific notation as strings. Fix that."""
    for k, v in d.items():
        if isinstance(v, dict):
            _coerce_yaml_numbers(v)
        elif isinstance(v, str):
            try:
                fv = float(v)
                d[k] = int(fv) if fv == int(fv) and "." not in v else fv
            except (ValueError, OverflowError):
                pass


def load_config_embodied(checkpoint_dir):
    """Load config using embodied_continuous profile."""
    # Try bridge configs first (has embodied_continuous), fall back to dreamerv3-torch
    bridge_configs = pathlib.Path("/workspace/bridge/scripts/configs.yaml")
    dreamer_configs = pathlib.Path(os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")) / "configs.yaml"

    if bridge_configs.exists():
        configs = yaml.safe_load(bridge_configs.read_text())
    else:
        configs = yaml.safe_load(dreamer_configs.read_text())

    defaults = {}
    for name in ["defaults", "embodied_continuous"]:
        if name not in configs:
            print(f"WARNING: config profile '{name}' not found, skipping")
            continue
        for k, v in configs[name].items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v

    _coerce_yaml_numbers(defaults)

    defaults["logdir"] = str(checkpoint_dir)
    defaults["device"] = _detect_device()
    defaults["compile"] = False
    defaults["steps"] = 100000
    defaults["action_repeat"] = 1
    defaults["eval_every"] = 10000
    defaults["time_limit"] = 8000
    defaults["log_every"] = 1000

    # Displacement loss defaults
    defaults.setdefault("disp_lambda", 0.0)
    defaults.setdefault("disp_ema_decay", 0.99)
    defaults.setdefault("disp_warmup", 50)

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])
    config.num_actions = 2  # Embodied: steering + count prediction
    return config


def load_agent_embodied(config):
    """Load agent with 2-dim action space."""
    import gym, gym.spaces
    from counting_env_pure import OBS_SIZE
    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(-50.0, 50.0, (OBS_SIZE,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    logdir = pathlib.Path(config.logdir)
    logger = tools.Logger(logdir, 0)
    train_eps = tools.load_episodes(logdir / "train_eps", limit=config.dataset_size)
    from dreamer import make_dataset, Dreamer
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return agent


def run_eval(agent, blob_counts, episodes_per, seed=42, arrangement="grid",
             random_project=False, random_permute=False):
    """Run eval episodes using embodied env and collect h_t + counts."""
    from envs.embodied import EmbodiedCountingWorld

    all_h_t = []
    all_counts = []
    total_eps = len(blob_counts) * episodes_per
    ep_num = 0
    t0 = time.time()

    for n_blobs in blob_counts:
        os.environ['EMBODIED_BLOB_MIN'] = str(n_blobs)
        os.environ['EMBODIED_BLOB_MAX'] = str(n_blobs)
        os.environ['EMBODIED_ARRANGEMENT'] = arrangement
        if random_project:
            os.environ['EMBODIED_RANDOM_PROJECT'] = 'true'
        else:
            os.environ.pop('EMBODIED_RANDOM_PROJECT', None)
        if random_permute:
            os.environ['EMBODIED_RANDOM_PERMUTE'] = 'true'
        else:
            os.environ.pop('EMBODIED_RANDOM_PERMUTE', None)

        env = EmbodiedCountingWorld("embodied_world", seed=seed)

        for ep in range(episodes_per):
            obs_raw = env.reset()
            state = None
            done = False
            while not done:
                vec = obs_raw["vector"]
                # Read count from inner env state
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

                action = policy_output["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num}/{total_eps} (blobs={n_blobs}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return np.stack(all_h_t), np.array(all_counts, dtype=np.int32)


def compute_ghe(h_t, counts):
    """Compute GHE, arc-length R², persistent homology, and RSA."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

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

    # HE (Euclidean)
    displacements = centroids[1:] - centroids[:-1]
    v_mean = displacements.mean(axis=0)
    he_per_step = np.sum((displacements - v_mean)**2, axis=1) / np.sum(v_mean**2)
    he = he_per_step.mean()

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
    if best_geo:
        arc_length = np.cumsum([0] + best_geo)
        count_values = np.arange(n)
        coeffs = np.polyfit(count_values, arc_length, 1)
        fit_line = np.polyval(coeffs, count_values)
        ss_res = np.sum((arc_length - fit_line)**2)
        ss_tot = np.sum((arc_length - np.mean(arc_length))**2)
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = None

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
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    rdm_agent = squareform(pdist(centroids, metric='euclidean'))
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu_idx = np.triu_indices(n, k=1)
    rsa_rho, _ = spearmanr(rdm_agent[triu_idx], rdm_ideal[triu_idx])

    return {
        "he": float(he),
        "ghe": float(best_ghe) if best_ghe else None,
        "arc_length_r2": float(r2) if r2 else None,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "rsa_spearman": float(rsa_rho),
        "n_counts": n,
        "n_timesteps": len(counts),
        "count_range": [int(counts.min()), int(counts.max())],
    }


def main():
    parser = argparse.ArgumentParser(description="Quick GHE check for embodied agent")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrangement", type=str, default="grid")
    parser.add_argument("--random_project", action="store_true")
    parser.add_argument("--random_permute", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    if not pathlib.Path(checkpoint_dir, "latest.pt").exists():
        print(f"No latest.pt in {checkpoint_dir}")
        sys.exit(1)

    metrics_path = pathlib.Path(checkpoint_dir) / "metrics.jsonl"
    train_step = "?"
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                d = json.loads(line)
                if "step" in d:
                    train_step = d["step"]

    device = _detect_device()
    print(f"=" * 50)
    print(f"Quick GHE Check — Embodied Agent ({device})")
    print(f"=" * 50)
    print(f"  Checkpoint:    {checkpoint_dir}")
    print(f"  Train step:    {train_step}")
    print(f"  Device:        {device}")
    print(f"  Arrangement:   {args.arrangement}")
    print(f"  Random proj:   {args.random_project}")
    print(f"  Episodes:      {args.episodes_per} x {len(BLOB_COUNTS)} counts = {args.episodes_per * len(BLOB_COUNTS)}")
    print()

    config = load_config_embodied(checkpoint_dir)
    print("Loading agent...")
    agent = load_agent_embodied(config)

    print("Running eval episodes...")
    t0 = time.time()
    h_t, counts = run_eval(agent, BLOB_COUNTS, args.episodes_per, seed=args.seed,
                           arrangement=args.arrangement,
                           random_project=args.random_project,
                           random_permute=args.random_permute)
    eval_time = time.time() - t0
    print(f"Eval done: {len(counts)} timesteps in {eval_time:.0f}s")

    print("\nComputing metrics...")
    results = compute_ghe(h_t, counts)
    results["train_step"] = train_step
    results["eval_time_seconds"] = round(eval_time, 1)
    results["experiment"] = "embodied"
    results["arrangement"] = args.arrangement
    results["random_project"] = args.random_project
    results["random_permute"] = args.random_permute

    print(f"\n{'=' * 50}")
    print(f"RESULTS (step {train_step})")
    print(f"{'=' * 50}")
    print(f"  GHE:            {results['ghe']:.4f}" if results['ghe'] else "  GHE:            N/A")
    print(f"  HE:             {results['he']:.4f}")
    print(f"  Arc-length R²:  {results['arc_length_r2']:.4f}" if results['arc_length_r2'] else "  Arc-length R²:  N/A")
    print(f"  β₀={results['beta_0']}, β₁={results['beta_1']}")
    print(f"  RSA Spearman:   {results['rsa_spearman']:.4f}")
    print(f"  Count range:    {results['count_range']}")
    print(f"  Timesteps:      {results['n_timesteps']}")

    out_path = pathlib.Path(checkpoint_dir) / "quick_ghe.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
