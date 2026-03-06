#!/usr/bin/env python3
"""
Record embodied agent episodes for visualization replay.

Uses the full DreamerV3 agent (PyTorch) to run episodes, recording:
  - observations (for RSSM replay)
  - actions (steering angle)
  - env state snapshots (bot pos, blob positions, grid filled count, phase)

Output: .npz file that visualize_counting.py can replay.

Usage:
    python record_embodied_episodes.py <checkpoint_dir> [--episodes 3] [--output embodied_replay.npz]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

import tools

import warnings
warnings.filterwarnings("ignore")


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
    for k, v in d.items():
        if isinstance(v, dict):
            _coerce_yaml_numbers(v)
        elif isinstance(v, str):
            try:
                fv = float(v)
                d[k] = int(fv) if fv == int(fv) and "." not in v else fv
            except (ValueError, OverflowError):
                pass


def load_config(checkpoint_dir):
    bridge_configs = Path("/workspace/bridge/scripts/configs.yaml")
    dreamer_configs = Path(os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")) / "configs.yaml"

    if bridge_configs.exists():
        configs = yaml.safe_load(bridge_configs.read_text())
    else:
        configs = yaml.safe_load(dreamer_configs.read_text())

    defaults = {}
    for name in ["defaults", "embodied_continuous"]:
        if name not in configs:
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
    defaults.setdefault("disp_lambda", 0.0)
    defaults.setdefault("disp_ema_decay", 0.99)
    defaults.setdefault("disp_warmup", 50)

    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])
    config.num_actions = 2
    return config


def load_agent(config):
    import gym, gym.spaces
    from counting_env_pure import OBS_SIZE
    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(-50.0, 50.0, (OBS_SIZE,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    logdir = Path(config.logdir)
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


def record_episodes(agent, n_episodes=3, blob_count=25, seed=42):
    """Record episodes with the trained policy."""
    from counting_env_embodied import EmbodiedCountingWorldEnv

    all_episodes = []

    for ep in range(n_episodes):
        env = EmbodiedCountingWorldEnv(
            blob_count_min=blob_count,
            blob_count_max=blob_count,
            max_steps=8000,
        )

        obs_raw = env.reset()
        state_agent = None

        ep_obs = []       # [T, 82]
        ep_actions = []   # [T, 2] (steering, count_pred)
        ep_bot_x = []
        ep_bot_y = []
        ep_bot_heading = []
        ep_grid_filled = []
        ep_phase = []
        ep_blobs_x = []   # [T, 25]
        ep_blobs_y = []   # [T, 25]
        ep_reward = []

        done = False
        step = 0

        while not done:
            vec = obs_raw.astype(np.float32)
            ep_obs.append(vec.copy())

            # Record env state
            s = env._state
            ep_bot_x.append(s.bot.pos_x)
            ep_bot_y.append(s.bot.pos_y)
            ep_bot_heading.append(s.bot.heading)
            ep_grid_filled.append(s.grid.filled_count)
            ep_phase.append(0.0 if s.phase == "gathering" else 1.0)

            bx = [b.pos_x for b in s.blobs[:25]]
            by = [b.pos_y for b in s.blobs[:25]]
            while len(bx) < 25:
                bx.append(0.0)
                by.append(0.0)
            ep_blobs_x.append(bx)
            ep_blobs_y.append(by)

            # Agent inference
            obs_dict = {
                "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
                "is_first": torch.tensor([[1.0 if step == 0 else 0.0]], dtype=torch.float32),
                "is_last": torch.tensor([[0.0]], dtype=torch.float32),
                "is_terminal": torch.tensor([[0.0]], dtype=torch.float32),
            }
            reset = np.array([step == 0])
            with torch.no_grad():
                policy_output, state_agent = agent(obs_dict, reset, state_agent, training=False)

            action = policy_output["action"].cpu().numpy().flatten()
            ep_actions.append(action.copy())

            # Step env with steering angle (action[0] maps [-1,1] → [-π,π])
            import math
            steering = float(action[0]) * math.pi
            obs_raw, reward, done, info = env.step([steering])
            ep_reward.append(reward)
            step += 1

        env.close()

        episode = {
            "obs": np.array(ep_obs, dtype=np.float32),
            "actions": np.array(ep_actions, dtype=np.float32),
            "bot_x": np.array(ep_bot_x, dtype=np.float32),
            "bot_y": np.array(ep_bot_y, dtype=np.float32),
            "bot_heading": np.array(ep_bot_heading, dtype=np.float32),
            "grid_filled": np.array(ep_grid_filled, dtype=np.int32),
            "phase": np.array(ep_phase, dtype=np.float32),
            "blobs_x": np.array(ep_blobs_x, dtype=np.float32),
            "blobs_y": np.array(ep_blobs_y, dtype=np.float32),
            "reward": np.array(ep_reward, dtype=np.float32),
        }
        all_episodes.append(episode)

        total_reward = sum(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: {step} steps, "
              f"grid_filled={ep_grid_filled[-1]}/{blob_count}, "
              f"total_reward={total_reward:.1f}")

    return all_episodes


def main():
    parser = argparse.ArgumentParser(description="Record embodied episodes for GUI replay")
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--blob_count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.checkpoint_dir) / "embodied_replay.npz")

    device = _detect_device()
    print(f"Recording embodied episodes ({device})")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Blobs:      {args.blob_count}")
    print()

    config = load_config(args.checkpoint_dir)
    print("Loading agent...")
    agent = load_agent(config)

    print("Recording episodes...")
    t0 = time.time()
    episodes = record_episodes(agent, args.episodes, args.blob_count, args.seed)
    elapsed = time.time() - t0
    print(f"\nRecording done in {elapsed:.0f}s")

    # Save as npz — prefix each array with ep{N}_
    save_dict = {"n_episodes": np.array(len(episodes))}
    for i, ep in enumerate(episodes):
        for key, arr in ep.items():
            save_dict[f"ep{i}_{key}"] = arr

    np.savez_compressed(args.output, **save_dict)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
