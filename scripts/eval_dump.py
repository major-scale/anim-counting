#!/usr/bin/env python3
"""
Eval Dump — Run eval episodes on a frozen DreamerV3 agent and save
all per-timestep tensors (h_t, z_t, counts, positions) for offline
analysis by CCC.

Reads job params from JOB_PARAMS env var (YAML file):
    checkpoint:    path to checkpoint dir (default: logs/continuous_s10)
    n_blobs:       fixed blob count, or null for env default (3-8)
    n_blobs_min:   min blob count (ignored if n_blobs is set)
    n_blobs_max:   max blob count (ignored if n_blobs is set)
    n_blobs_values: list of blob counts to sweep, e.g. [3,5,8,9,10,12,15]
                    (overrides n_blobs/n_blobs_min/n_blobs_max)
    n_episodes:    number of eval episodes (default: 50, used when n_blobs_values is not set)
    n_episodes_per: episodes per blob count (default: 10, used with n_blobs_values)
    output_dir:    where to save .npy files (default: auto from job id)
    seed:          env seed (default: 42)

Output:
    {output_dir}/h_t.npy          (total_timesteps, 512)
    {output_dir}/z_t.npy          (total_timesteps, 32, 32)
    {output_dir}/counts.npy       (total_timesteps,)
    {output_dir}/bot_positions.npy (total_timesteps, 2)
    {output_dir}/timesteps.npy    (total_timesteps,)
    {output_dir}/episodes.npy     (total_timesteps,)
    {output_dir}/actor_predictions.npy (total_timesteps,)
    {output_dir}/rewards.npy      (total_timesteps,)
    {output_dir}/total_blobs.npy  (total_timesteps,)
    {output_dir}/metadata.json
"""

import json
import os
import pathlib
import sys
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, "~/dreamerv3-torch")
sys.path.insert(0, "~/anim-bridge/scripts")

from ruamel.yaml import YAML as _YAML
import yaml as pyyaml


class _YAMLCompat:
    @staticmethod
    def safe_load(text):
        y = _YAML(typ='safe', pure=True)
        return y.load(text)


yaml = _YAMLCompat()

import tools
import networks

# Current env layout (25 blobs, OBS_SIZE=82, grid-based counting)
NEW_MAX_BLOBS = 25
NEW_GRID_SLOT_START = 53   # 3 + 25*2 — grid slot assignments (normalized slot index, 0 if in field)
NEW_GRID_SLOT_END = 78     # 53 + 25
NEW_COUNT_IDX = 78
NEW_PHASE_IDX = 79
NEW_GRID_FILLED_NORM_IDX = 80
NEW_GRID_FILLED_RAW_IDX = 81

# Legacy layout (20 blobs, OBS_SIZE=65) — for frozen checkpoints trained on old env
OLD_MAX_BLOBS = 20
OLD_OBS_SIZE = 65
OLD_MARK_START = 43   # 3 + 20*2
OLD_MARK_END = 63     # 43 + 20
OLD_COUNT_IDX = 63
OLD_PHASE_IDX = 64

DEFAULT_CHECKPOINT = pathlib.Path.home() / "anim-training" / "logs" / "continuous_s10"


def load_params():
    """Load job parameters from JOB_PARAMS env var."""
    params_path = os.environ.get("JOB_PARAMS")
    if params_path and os.path.exists(params_path):
        with open(params_path) as f:
            return pyyaml.safe_load(f) or {}
    return {}


def write_progress(step, total_steps, episode, total_episodes, elapsed):
    """Write progress JSON for CCC to poll."""
    artifacts = os.environ.get("ANIM_ARTIFACTS")
    job_id = os.environ.get("JOB_ID", "eval_dump")
    if not artifacts:
        return
    status_dir = pathlib.Path(artifacts) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    eta = (elapsed / max(step, 1)) * (total_steps - step) if step > 0 else 0
    progress = {
        "step": step,
        "total_steps": total_steps,
        "percent": round(100 * step / max(total_steps, 1), 1),
        "episode": episode,
        "total_episodes": total_episodes,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
    }
    with open(status_dir / f"{job_id}_progress.json", "w") as f:
        json.dump(progress, f)


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def load_config(checkpoint_dir):
    configs = yaml.safe_load(
        (pathlib.Path("~/dreamerv3-torch/configs.yaml")).read_text()
    )
    defaults = {}
    for name in ["defaults", "counting_continuous"]:
        recursive_update(defaults, configs[name])
    defaults["logdir"] = str(checkpoint_dir)
    defaults["device"] = "mps"
    defaults["compile"] = False
    defaults["steps"] = 100000
    defaults["action_repeat"] = 1
    defaults["eval_every"] = 10000
    defaults["time_limit"] = 6000
    defaults["log_every"] = 1000
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args([])
    config.num_actions = 1
    return config


def detect_checkpoint_obs_size(checkpoint_dir):
    """Detect the obs size a checkpoint was trained with by inspecting encoder weights."""
    ckpt_path = pathlib.Path(checkpoint_dir) / "latest.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = checkpoint["agent_state_dict"]
    encoder_key = "_wm.encoder._mlp.layers.Encoder_linear0.weight"
    if encoder_key in sd:
        return sd[encoder_key].shape[1]
    return None


def remap_obs_80_to_65(obs_80):
    """Remap an 80-dim obs (25 blobs) to 65-dim (20 blobs) for legacy checkpoints.

    80-dim: [bot_x, bot_y, state, 25×pos(50), 25×mark(25), count, phase]
    65-dim: [bot_x, bot_y, state, 20×pos(40), 20×mark(20), count, phase]

    Blobs 21-25 are simply dropped (model never saw them during training).
    """
    obs_65 = np.zeros(OLD_OBS_SIZE, dtype=np.float32)
    # bot_x, bot_y, state (indices 0-2 are the same)
    obs_65[0:3] = obs_80[0:3]
    # First 20 blob positions: old [3:43], new [3:43] (same offset, just fewer)
    obs_65[3:43] = obs_80[3:43]
    # First 20 mark flags: old [43:63], new [53:73]
    obs_65[43:63] = obs_80[53:73]
    # Blob count: old [63], new [78]
    obs_65[63] = obs_80[78]
    # Phase: old [64], new [79]
    obs_65[64] = obs_80[79]
    return obs_65


def load_agent(config, blob_max=8, obs_size=None):
    import gym
    import gym.spaces
    from counting_env import OBS_SIZE

    # Use detected obs_size if provided, otherwise use current env's OBS_SIZE
    actual_obs_size = obs_size if obs_size is not None else OBS_SIZE

    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(0.0, float(blob_max), (actual_obs_size,), dtype=np.float32),
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
    print(f"Loaded checkpoint from {logdir / 'latest.pt'} (obs_size={actual_obs_size})")
    return agent


def action_to_prediction(action_val, blob_max=25):
    return np.clip((action_val + 1.0) * (blob_max / 2.0), 0.0, float(blob_max))


def make_env(n_blobs, seed=42):
    """Create a CountingWorld env with a fixed blob count."""
    os.environ["COUNTING_ACTION_SPACE"] = "continuous"
    os.environ["COUNTING_BLOB_MIN"] = str(n_blobs)
    os.environ["COUNTING_BLOB_MAX"] = str(n_blobs)
    from envs.counting import CountingWorld
    return CountingWorld("counting_world", seed=seed)


def run_episodes_on_env(agent, env, device, num_episodes, episode_offset=0,
                        start_time=None, total_episodes_global=None,
                        steps_so_far=0, legacy_obs=False):
    """Run num_episodes on an already-created env. Returns data dict + total steps.

    If legacy_obs=True, remaps 80-dim env obs to 65-dim before feeding to model,
    but records ground truth from the full 80-dim obs.
    """

    all_h_t = []
    all_z_t = []
    all_counts = []
    all_actor_pred = []
    all_rewards = []
    all_bot_pos = []
    all_episodes = []
    all_timesteps = []
    all_total_blobs = []

    if start_time is None:
        start_time = time.time()
    if total_episodes_global is None:
        total_episodes_global = num_episodes

    for local_ep in range(num_episodes):
        global_ep = episode_offset + local_ep
        obs_raw = env.reset()
        done = False
        state = None
        step = 0

        while not done:
            # Extract ground truth from full 82-dim obs (always use new layout)
            full_vec = obs_raw["vector"]
            marked_count = int(full_vec[NEW_GRID_FILLED_RAW_IDX])
            total_blobs = int(full_vec[NEW_COUNT_IDX])
            bot_x = float(full_vec[0])
            bot_y = float(full_vec[1])

            # Feed model — remap to 65-dim if legacy checkpoint
            model_vec = remap_obs_80_to_65(full_vec) if legacy_obs else full_vec

            obs_dict = {
                "vector": torch.tensor(model_vec, dtype=torch.float32)
                    .unsqueeze(0).to(device),
                "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]],
                    dtype=torch.float32).to(device),
                "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]],
                    dtype=torch.float32).to(device),
                "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]],
                    dtype=torch.float32).to(device),
            }
            reset = np.array([obs_raw["is_first"]])

            with torch.no_grad():
                policy_output, state = agent(obs_dict, reset, state, training=False)

            latent, action_tensor = state
            h_t = latent["deter"].cpu().numpy().squeeze(0)
            z_t = latent["stoch"].cpu().numpy().squeeze(0)

            raw_action = action_tensor.cpu().numpy().squeeze()
            if raw_action.ndim == 0:
                raw_action = float(raw_action)
            else:
                raw_action = float(raw_action[0])

            obs_raw, reward, done, info = env.step(policy_output["action"].cpu().numpy())

            all_h_t.append(h_t)
            all_z_t.append(z_t)
            all_counts.append(marked_count)
            all_actor_pred.append(action_to_prediction(raw_action, blob_max=8 if legacy_obs else 25))
            all_rewards.append(reward)
            all_bot_pos.append([bot_x, bot_y])
            all_episodes.append(global_ep)
            all_timesteps.append(step)
            all_total_blobs.append(total_blobs)

            step += 1

        total_so_far = steps_so_far + len(all_counts)
        elapsed = time.time() - start_time
        avg_steps_per_ep = total_so_far / max(global_ep + 1, 1)
        total_steps_est = int(avg_steps_per_ep * total_episodes_global)
        write_progress(total_so_far, total_steps_est,
                       global_ep + 1, total_episodes_global, elapsed)

        print(f"  Episode {global_ep + 1}/{total_episodes_global}: {step} steps, "
              f"blobs={total_blobs}, elapsed={elapsed:.0f}s")

    return {
        "h_t": np.stack(all_h_t),
        "z_t": np.stack(all_z_t),
        "counts": np.array(all_counts, dtype=np.int32),
        "actor_predictions": np.array(all_actor_pred, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "bot_positions": np.stack(all_bot_pos).astype(np.float32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
        "episodes": np.array(all_episodes, dtype=np.int32),
        "total_blobs": np.array(all_total_blobs, dtype=np.int32),
    }


def run_eval_episodes(agent, config, num_episodes, n_blobs=None,
                      n_blobs_min=None, n_blobs_max=None, seed=42,
                      legacy_obs=False):
    """Single blob-count mode (original behavior)."""
    os.environ["COUNTING_ACTION_SPACE"] = "continuous"
    if n_blobs is not None:
        os.environ["COUNTING_BLOB_MIN"] = str(n_blobs)
        os.environ["COUNTING_BLOB_MAX"] = str(n_blobs)
    else:
        if n_blobs_min is not None:
            os.environ["COUNTING_BLOB_MIN"] = str(n_blobs_min)
        if n_blobs_max is not None:
            os.environ["COUNTING_BLOB_MAX"] = str(n_blobs_max)

    from envs.counting import CountingWorld
    env = CountingWorld("counting_world", seed=seed)
    data = run_episodes_on_env(agent, env, config.device, num_episodes,
                               legacy_obs=legacy_obs)
    env.close()
    return data


def run_sweep(agent, config, n_blobs_values, n_episodes_per, seed=42,
              legacy_obs=False):
    """Sweep mode: for each blob count in the list, run n_episodes_per episodes."""
    total_episodes = len(n_blobs_values) * n_episodes_per
    print(f"\n  Sweep: {n_blobs_values} x {n_episodes_per} eps = {total_episodes} total")

    all_data = {}
    episode_offset = 0
    steps_so_far = 0
    start_time = time.time()

    for blob_count in n_blobs_values:
        print(f"\n--- n_blobs = {blob_count} ---")
        env = make_env(blob_count, seed=seed)
        data = run_episodes_on_env(
            agent, env, config.device, n_episodes_per,
            episode_offset=episode_offset,
            start_time=start_time,
            total_episodes_global=total_episodes,
            steps_so_far=steps_so_far,
            legacy_obs=legacy_obs,
        )
        env.close()

        if not all_data:
            all_data = data
        else:
            for k in all_data:
                all_data[k] = np.concatenate([all_data[k], data[k]], axis=0)

        episode_offset += n_episodes_per
        steps_so_far = len(all_data["counts"])

    return all_data


def main():
    params = load_params()
    job_id = os.environ.get("JOB_ID", "eval_dump")
    artifacts = os.environ.get("ANIM_ARTIFACTS",
                               str(pathlib.Path.home() / "anim-bridge" / "artifacts"))

    checkpoint_dir = pathlib.Path(os.path.expanduser(
        params.get("checkpoint", str(DEFAULT_CHECKPOINT))
    ))
    n_blobs_values = params.get("n_blobs_values")  # e.g. [3, 5, 8, 9, 10, 12, 15]
    n_episodes_per = params.get("n_episodes_per", 10)
    n_episodes = params.get("n_episodes", 50)
    n_blobs = params.get("n_blobs")
    n_blobs_min = params.get("n_blobs_min")
    n_blobs_max = params.get("n_blobs_max")
    seed = params.get("seed", 42)
    output_dir = pathlib.Path(params.get("output_dir",
                              f"{artifacts}/tensors/{job_id}"))

    # Target arrangement (grid, line, etc.) — must match training arrangement
    arrangement = params.get("arrangement", "grid")
    os.environ["COUNTING_ARRANGEMENT"] = arrangement

    sweep_mode = n_blobs_values is not None and len(n_blobs_values) > 0

    print("=" * 60)
    print(f"Eval Dump: {job_id}")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_dir}")
    if sweep_mode:
        total_eps = len(n_blobs_values) * n_episodes_per
        print(f"  Mode:       SWEEP over {n_blobs_values}")
        print(f"  Episodes:   {n_episodes_per} per count = {total_eps} total")
    else:
        print(f"  Episodes:   {n_episodes}")
        print(f"  Blobs:      {n_blobs or f'{n_blobs_min or 3}-{n_blobs_max or 8}'}")
    print(f"  Output:     {output_dir}")
    print(f"  Seed:       {seed}")
    if arrangement != "grid":
        print(f"  Arrangement: {arrangement}")

    print("\nLoading model...")
    config = load_config(checkpoint_dir)

    # Detect if checkpoint was trained with old 65-dim obs
    ckpt_obs_size = detect_checkpoint_obs_size(checkpoint_dir)
    from counting_env import OBS_SIZE as env_obs_size
    legacy_obs = (ckpt_obs_size is not None and ckpt_obs_size < env_obs_size)
    if legacy_obs:
        print(f"  Legacy checkpoint detected: trained with obs_size={ckpt_obs_size}, "
              f"env produces {env_obs_size}. Will remap observations.")

    blob_max = max(n_blobs_values) if sweep_mode else (n_blobs or n_blobs_max or 8)
    agent = load_agent(config, blob_max=blob_max, obs_size=ckpt_obs_size)

    if sweep_mode:
        print(f"\nRunning sweep: {n_blobs_values} x {n_episodes_per} episodes...")
        data = run_sweep(agent, config, n_blobs_values, n_episodes_per, seed=seed,
                         legacy_obs=legacy_obs)
        actual_n_episodes = len(n_blobs_values) * n_episodes_per
    else:
        print(f"\nRunning {n_episodes} eval episodes...")
        data = run_eval_episodes(agent, config, n_episodes,
                                 n_blobs=n_blobs, n_blobs_min=n_blobs_min,
                                 n_blobs_max=n_blobs_max, seed=seed,
                                 legacy_obs=legacy_obs)
        actual_n_episodes = n_episodes

    # Save arrays
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in data.items():
        np.save(output_dir / f"{name}.npy", arr)
        print(f"  Saved {name}.npy  shape={arr.shape}  dtype={arr.dtype}")

    # Save metadata
    metadata = {
        "job_id": job_id,
        "checkpoint": str(checkpoint_dir),
        "n_episodes": actual_n_episodes,
        "seed": seed,
        "total_timesteps": len(data["counts"]),
        "count_range": [int(data["counts"].min()), int(data["counts"].max())],
        "mean_actor_error": float(np.mean(np.abs(
            data["actor_predictions"] - data["counts"]))),
        "shapes": {k: list(v.shape) for k, v in data.items()},
    }
    if sweep_mode:
        metadata["mode"] = "sweep"
        metadata["n_blobs_values"] = n_blobs_values
        metadata["n_episodes_per"] = n_episodes_per
    else:
        metadata["mode"] = "single"
        metadata["n_blobs"] = n_blobs
        metadata["n_blobs_min"] = n_blobs_min
        metadata["n_blobs_max"] = n_blobs_max

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")

    n = len(data["counts"])
    print(f"\n--- Summary ---")
    print(f"  Total timesteps: {n}")
    print(f"  Episodes: {actual_n_episodes}")
    print(f"  h_t shape: {data['h_t'].shape}")
    print(f"  Count range: {data['counts'].min()} - {data['counts'].max()}")
    print(f"  Mean actor error: {metadata['mean_actor_error']:.3f}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
