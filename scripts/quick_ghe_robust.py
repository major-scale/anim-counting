#!/usr/bin/env python3
"""
Quick GHE check for robustness experiments — extends standard battery with:
  - Per-arrangement probe: force each arrangement type, 5 episodes, report per-type accuracy
  - Cold-start probe: force start_count at [0, 5, 10, 15, 20], check probe at t=0

Usage:
    python quick_ghe_robust.py <checkpoint_dir> [--episodes_per 5] [--random_project] [--varied_arrangements] [--random_start_count]
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

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

BLOB_COUNTS = [3, 5, 8, 10, 12, 15, 20, 25]
GRID_FILLED_RAW_IDX = 81
COUNT_IDX = 78

# Import arrangement types from counting_env_pure
from counting_env_pure import _ALL_TYPES as ALL_ARRANGEMENT_TYPES


def _detect_device():
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_config_cpu(checkpoint_dir):
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


def load_agent_cpu(config):
    import gym, gym.spaces
    from counting_env_pure import OBS_SIZE
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
    from dreamer import make_dataset, Dreamer
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return agent


def run_eval(agent, blob_counts, episodes_per, seed=42):
    """Run standard eval episodes and collect h_t + counts."""
    from envs.counting import CountingWorld

    all_h_t = []
    all_counts = []
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
            print(f"  Standard eval {ep_num}/{total_eps} (blobs={n_blobs}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return np.stack(all_h_t), np.array(all_counts, dtype=np.int32)


def run_per_arrangement_eval(agent, episodes_per=5, n_blobs=15, seed=42):
    """Run episodes forcing each arrangement type, collect probe accuracy per type."""
    from envs.counting import CountingWorld
    from sklearn.linear_model import Ridge

    results = {}
    t0 = time.time()
    total = len(ALL_ARRANGEMENT_TYPES)

    for arr_idx, arr_type in enumerate(ALL_ARRANGEMENT_TYPES):
        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        # Disable varied arrangements for controlled testing
        os.environ['COUNTING_VARIED_ARRANGEMENTS'] = 'false'
        os.environ['COUNTING_RANDOM_START_COUNT'] = 'false'
        env = CountingWorld("counting_world", seed=seed)

        arr_h_t = []
        arr_counts = []

        for ep in range(episodes_per):
            # Force arrangement type by temporarily patching
            import counting_env_pure
            orig_func = counting_env_pure._random_arrangement
            counting_env_pure._random_arrangement = lambda varied=False: arr_type

            obs_raw = env.reset()

            counting_env_pure._random_arrangement = orig_func

            state = None
            done = False
            while not done:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else 0

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
                arr_h_t.append(h_t)
                arr_counts.append(marked_count)

                action = policy_output["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)

        env.close()

        if len(arr_h_t) > 10:
            X = np.stack(arr_h_t)
            y = np.array(arr_counts, dtype=np.float32)
            probe = Ridge(alpha=1.0).fit(X, y)
            preds = probe.predict(X)
            r2 = float(probe.score(X, y))
            exact_acc = float(np.mean(np.round(preds) == y))
            within_1 = float(np.mean(np.abs(np.round(preds) - y) <= 1))
        else:
            r2, exact_acc, within_1 = 0, 0, 0

        results[arr_type] = {
            "probe_r2": round(r2, 4),
            "exact_accuracy": round(exact_acc, 4),
            "within_1_accuracy": round(within_1, 4),
            "n_timesteps": len(arr_h_t),
        }

        elapsed = time.time() - t0
        print(f"  Arrangement {arr_idx+1}/{total} ({arr_type}): R²={r2:.3f}, acc={exact_acc:.1%} [{elapsed:.0f}s]")

    return results


def run_cold_start_eval(agent, start_counts=(0, 5, 10, 15, 20), episodes_per=5, n_blobs=25, seed=42):
    """Force specific start counts and check probe at t=0."""
    from envs.counting import CountingWorld

    results = {}
    t0 = time.time()

    for sc in start_counts:
        if sc >= n_blobs - 4:
            continue  # skip if would leave <5 in field

        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        os.environ['COUNTING_VARIED_ARRANGEMENTS'] = 'false'
        os.environ['COUNTING_RANDOM_START_COUNT'] = 'false'
        env = CountingWorld("counting_world", seed=seed)

        first_h_t = []
        first_counts = []
        all_h_t = []
        all_counts = []

        for ep in range(episodes_per):
            # Disable random start, we'll manually pre-place after reset
            env._env.random_start_count = False
            obs_raw = env.reset()

            # Manually pre-place sc blobs on grid
            if sc > 0 and env._env._state and sc < len(env._env._state.blobs):
                state_ep = env._env._state
                grid = state_ep.grid
                for i in range(sc):
                    blob = state_ep.blobs[i]
                    if blob.grid_slot is not None:
                        continue
                    slot_idx = len(grid.placement_order)
                    if slot_idx >= len(grid.slots):
                        break
                    grid.occupancy[slot_idx] = blob.id
                    grid.placement_order.append(blob.id)
                    blob.grid_slot = slot_idx
                    blob.pos_x = grid.slots[slot_idx][0]
                    blob.pos_y = grid.slots[slot_idx][1]
                    blob.pending_grid_placement = False
                grid.filled_count = sc
                state_ep.bot.count_tally = sc
                state_ep.start_count = sc
                state_ep.bot.waypoints = [(b.pos_x, b.pos_y) for b in state_ep.blobs if b.grid_slot is None]
                # Re-observe with updated state
                from counting_env_pure import _get_observation
                env._env._current_obs = _get_observation(state_ep)
                obs_raw = {
                    "vector": env._env._current_obs,
                    "is_first": True,
                    "is_last": False,
                    "is_terminal": False,
                }

            state_agent = None
            done = False
            step_in_ep = 0
            while not done:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else 0

                obs_dict = {
                    "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
                    "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]], dtype=torch.float32),
                    "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]], dtype=torch.float32),
                    "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]], dtype=torch.float32),
                }
                reset = np.array([obs_raw["is_first"]])
                with torch.no_grad():
                    policy_output, state_agent = agent(obs_dict, reset, state_agent, training=False)

                latent, _ = state_agent
                h_t = latent["deter"].cpu().numpy().squeeze(0)

                if step_in_ep == 0:
                    first_h_t.append(h_t)
                    first_counts.append(marked_count)

                all_h_t.append(h_t)
                all_counts.append(marked_count)

                action = policy_output["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)
                step_in_ep += 1

        env.close()

        # Fit probe on all timesteps
        if len(all_h_t) > 10:
            from sklearn.linear_model import Ridge
            X = np.stack(all_h_t)
            y = np.array(all_counts, dtype=np.float32)
            probe = Ridge(alpha=1.0).fit(X, y)
            r2 = float(probe.score(X, y))

            # Check probe on first timesteps
            if first_h_t:
                X_first = np.stack(first_h_t)
                y_first = np.array(first_counts, dtype=np.float32)
                preds_first = probe.predict(X_first)
                first_errors = np.abs(np.round(preds_first) - y_first)
                first_exact = float(np.mean(first_errors == 0))
                first_mean_err = float(first_errors.mean())
            else:
                first_exact, first_mean_err = 0, 0
        else:
            r2, first_exact, first_mean_err = 0, 0, 0

        results[f"start_{sc}"] = {
            "start_count": sc,
            "probe_r2": round(r2, 4),
            "t0_exact_accuracy": round(first_exact, 4),
            "t0_mean_error": round(first_mean_err, 2),
            "n_timesteps": len(all_h_t),
        }

        elapsed = time.time() - t0
        print(f"  Cold start={sc}: t0_acc={first_exact:.1%}, t0_err={first_mean_err:.1f} [{elapsed:.0f}s]")

    return results


def compute_ghe(h_t, counts):
    """Compute GHE, arc-length R^2, and persistent homology."""
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

    # HE
    displacements = centroids[1:] - centroids[:-1]
    v_mean = displacements.mean(axis=0)
    he_per_step = np.sum((displacements - v_mean)**2, axis=1) / np.sum(v_mean**2)
    he = he_per_step.mean()

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

    # Arc-length R^2
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
    parser = argparse.ArgumentParser(description="Quick GHE + robustness battery")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrangement", type=str, default="grid")
    parser.add_argument("--random_project", action="store_true")
    parser.add_argument("--random_permute", action="store_true")
    parser.add_argument("--varied_arrangements", action="store_true")
    parser.add_argument("--random_start_count", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not pathlib.Path(checkpoint_dir, "latest.pt").exists():
        print(f"No latest.pt in {checkpoint_dir}")
        sys.exit(1)

    # Get training step
    metrics_path = pathlib.Path(checkpoint_dir) / "metrics.jsonl"
    train_step = "?"
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                d = json.loads(line)
                if "step" in d:
                    train_step = d["step"]

    device = _detect_device()
    print(f"=" * 60)
    print(f"Robustness GHE Battery ({device})")
    print(f"=" * 60)
    print(f"  Checkpoint:   {checkpoint_dir}")
    print(f"  Train step:   {train_step}")
    print(f"  Device:       {device}")
    print(f"  Random proj:  {args.random_project}")
    print(f"  Varied arr:   {args.varied_arrangements}")
    print(f"  Random start: {args.random_start_count}")
    print()

    # Setup env
    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = args.arrangement
    os.environ['COUNTING_BRIDGE_SCRIPT'] = '/workspace/bridge/dist/bridge.js'
    os.environ['COUNTING_MAX_STEPS'] = '10000'
    if args.random_project:
        os.environ['COUNTING_RANDOM_PROJECT'] = 'true'
    if args.random_permute:
        os.environ['COUNTING_RANDOM_PERMUTE'] = 'true'
    if args.varied_arrangements:
        os.environ['COUNTING_VARIED_ARRANGEMENTS'] = 'true'
    if args.random_start_count:
        os.environ['COUNTING_RANDOM_START_COUNT'] = 'true'

    config = load_config_cpu(checkpoint_dir)
    print("Loading agent...")
    agent = load_agent_cpu(config)

    # --- Phase 1: Standard GHE battery ---
    print("\n--- Phase 1: Standard GHE Battery ---")
    t0 = time.time()
    h_t, counts = run_eval(agent, BLOB_COUNTS, args.episodes_per, seed=args.seed)
    eval_time = time.time() - t0
    print(f"Eval done: {len(counts)} timesteps in {eval_time:.0f}s")

    print("\nComputing GHE metrics...")
    ghe_results = compute_ghe(h_t, counts)
    ghe_results["train_step"] = train_step
    ghe_results["eval_time_seconds"] = round(eval_time, 1)

    print(f"\n  GHE:            {ghe_results['ghe']:.4f}" if ghe_results['ghe'] else "  GHE:            N/A")
    print(f"  Arc-length R2:  {ghe_results['arc_length_r2']:.4f}" if ghe_results['arc_length_r2'] else "  Arc-length R2:  N/A")
    print(f"  B0={ghe_results['beta_0']}, B1={ghe_results['beta_1']}")
    print(f"  RSA:            {ghe_results['rsa_spearman']:.4f}")

    # --- Phase 2: Per-arrangement probe ---
    print("\n--- Phase 2: Per-Arrangement Probe ---")
    arr_results = run_per_arrangement_eval(agent, episodes_per=args.episodes_per, seed=args.seed)

    # --- Phase 3: Cold-start probe ---
    print("\n--- Phase 3: Cold-Start Probe ---")
    cold_results = run_cold_start_eval(agent, episodes_per=args.episodes_per, seed=args.seed)

    # --- Combine and save ---
    all_results = {
        "standard_ghe": ghe_results,
        "per_arrangement": arr_results,
        "cold_start": cold_results,
        "config": {
            "random_project": args.random_project,
            "varied_arrangements": args.varied_arrangements,
            "random_start_count": args.random_start_count,
        },
    }

    out_path = pathlib.Path(checkpoint_dir) / "quick_ghe.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save battery NPZ for offline analysis
    npz_path = pathlib.Path(checkpoint_dir) / "battery.npz"
    np.savez_compressed(str(npz_path), h_t=h_t, counts=counts)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY (step {train_step})")
    print(f"{'=' * 60}")
    print(f"  GHE:  {ghe_results['ghe']:.4f}" if ghe_results['ghe'] else "  GHE:  N/A")
    print(f"  RSA:  {ghe_results['rsa_spearman']:.4f}")
    print(f"  Topology: B0={ghe_results['beta_0']}, B1={ghe_results['beta_1']}")
    print()
    print(f"  Per-arrangement probe (mean R2): {np.mean([v['probe_r2'] for v in arr_results.values()]):.3f}")
    print(f"  Per-arrangement probe (mean acc): {np.mean([v['exact_accuracy'] for v in arr_results.values()]):.1%}")
    if cold_results:
        print(f"  Cold-start t0 accuracy: {np.mean([v['t0_exact_accuracy'] for v in cold_results.values()]):.1%}")
    print(f"\n  Saved to {out_path}")
    print(f"  Battery NPZ: {npz_path}")


if __name__ == "__main__":
    main()
