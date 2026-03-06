#!/usr/bin/env python3
"""
Extract h_t (GRU hidden state) vectors and count labels from a checkpoint.
Saves numpy arrays for downstream visualization.

Usage:
    python extract_h_t.py <checkpoint_dir> [--episodes_per 5] [--output_dir /path/]
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
    """Run eval episodes and collect h_t + counts + episode metadata."""
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
                all_episode_ids.append(ep_num)
                all_timesteps.append(step_in_ep)

                action = policy_output["action"].cpu().numpy().flatten()
                obs_raw, reward, done, info = env.step(action)
                step_in_ep += 1

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num}/{total_eps} (blobs={n_blobs}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    return (np.stack(all_h_t), np.array(all_counts, dtype=np.int32),
            np.array(all_episode_ids, dtype=np.int32), np.array(all_timesteps, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description="Extract h_t vectors from checkpoint")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--episodes_per", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrangement", type=str, default="grid")
    parser.add_argument("--output_dir", type=str, default="/workspace/bridge/artifacts/h_t_data")
    parser.add_argument("--label", type=str, default=None, help="Label for this condition")
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

    label = args.label or pathlib.Path(checkpoint_dir).name
    device = _detect_device()
    print(f"Extracting h_t: {label}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Train step: {train_step}")
    print(f"  Device:     {device}")

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = args.arrangement
    os.environ['COUNTING_BRIDGE_SCRIPT'] = '/workspace/bridge/dist/bridge.js'
    os.environ['COUNTING_MAX_STEPS'] = '10000'

    config = load_config_cpu(checkpoint_dir)
    print("Loading agent...")
    agent = load_agent_cpu(config)

    print("Running eval episodes...")
    t0 = time.time()
    h_t, counts, episode_ids, timesteps = run_eval(
        agent, BLOB_COUNTS, args.episodes_per, seed=args.seed
    )
    eval_time = time.time() - t0
    print(f"Eval done: {len(counts)} timesteps in {eval_time:.0f}s")

    # Save
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{label}.npz"
    np.savez_compressed(
        out_file,
        h_t=h_t,
        counts=counts,
        episode_ids=episode_ids,
        timesteps=timesteps,
        train_step=np.array([train_step]),
        label=np.array([label]),
    )
    print(f"Saved {h_t.shape} h_t vectors to {out_file}")
    print(f"  h_t shape: {h_t.shape}")
    print(f"  Count range: [{counts.min()}, {counts.max()}]")
    print(f"  Unique counts: {len(np.unique(counts))}")


if __name__ == "__main__":
    main()
