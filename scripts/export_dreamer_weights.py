#!/usr/bin/env python3
"""
Export DreamerV3 world model weights (encoder + RSSM) for browser inference.

Binary format: dreamer_weights.bin (~10.7MB) + dreamer_manifest.json (offsets)
Each tensor is stored as contiguous float32 little-endian bytes.

Also exports PCA components and count centroids.

Usage:
    python export_dreamer_weights.py [--checkpoint_dir DIR] [--output DIR]
"""

import os, sys, json, argparse, struct
import numpy as np
import torch


def export_weights(checkpoint_dir, output_dir):
    """Extract encoder + RSSM weights and write as binary + manifest."""
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["agent_state_dict"]

    prefix = "_expl_behavior._world_model."

    # Ordered list of (short_name, checkpoint_key, expected_shape)
    weight_list = [
        ("enc_linear0_w", "encoder._mlp.layers.Encoder_linear0.weight",   None),  # dynamic: [512, obs_dim]
        ("enc_norm0_w",   "encoder._mlp.layers.Encoder_norm0.weight",     [512]),
        ("enc_norm0_b",   "encoder._mlp.layers.Encoder_norm0.bias",       [512]),
        ("enc_linear1_w", "encoder._mlp.layers.Encoder_linear1.weight",   [512, 512]),
        ("enc_norm1_w",   "encoder._mlp.layers.Encoder_norm1.weight",     [512]),
        ("enc_norm1_b",   "encoder._mlp.layers.Encoder_norm1.bias",       [512]),
        ("enc_linear2_w", "encoder._mlp.layers.Encoder_linear2.weight",   [512, 512]),
        ("enc_norm2_w",   "encoder._mlp.layers.Encoder_norm2.weight",     [512]),
        ("enc_norm2_b",   "encoder._mlp.layers.Encoder_norm2.bias",       [512]),
        ("img_in_w",      "dynamics._img_in_layers.0.weight",             None),  # dynamic: [512, 1024+num_actions]
        ("img_in_norm_w", "dynamics._img_in_layers.1.weight",             [512]),
        ("img_in_norm_b", "dynamics._img_in_layers.1.bias",               [512]),
        ("gru_w",         "dynamics._cell.layers.GRU_linear.weight",      [1536, 1024]),
        ("gru_norm_w",    "dynamics._cell.layers.GRU_norm.weight",        [1536]),
        ("gru_norm_b",    "dynamics._cell.layers.GRU_norm.bias",          [1536]),
        ("img_out_w",     "dynamics._img_out_layers.0.weight",            [512, 512]),
        ("img_out_norm_w","dynamics._img_out_layers.1.weight",            [512]),
        ("img_out_norm_b","dynamics._img_out_layers.1.bias",              [512]),
        ("imgs_stat_w",   "dynamics._imgs_stat_layer.weight",             [1024, 512]),
        ("imgs_stat_b",   "dynamics._imgs_stat_layer.bias",               [1024]),
        # Posterior (obs_step): obs_out + obs_stat
        ("obs_out_w",     "dynamics._obs_out_layers.0.weight",            [512, 1024]),
        ("obs_out_norm_w","dynamics._obs_out_layers.1.weight",            [512]),
        ("obs_out_norm_b","dynamics._obs_out_layers.1.bias",              [512]),
        ("obs_stat_w",    "dynamics._obs_stat_layer.weight",              [1024, 512]),
        ("obs_stat_b",    "dynamics._obs_stat_layer.bias",                [1024]),
        # Initial deter
        ("deter_init_w",  "dynamics.W",                                    [1, 512]),
    ]

    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "dreamer_weights.bin")
    manifest = {"format": "float32_le", "tensors": {}}
    offset = 0

    with open(bin_path, "wb") as f:
        for short_name, key, expected_shape in weight_list:
            full_key = prefix + key
            if full_key not in state:
                print(f"  WARNING: key not found: {full_key}")
                continue
            tensor = state[full_key].float()
            actual_shape = list(tensor.shape)
            if expected_shape is not None and actual_shape != expected_shape:
                print(f"  WARNING: {short_name} shape mismatch: expected {expected_shape}, got {actual_shape}")
            shape = actual_shape
            t = tensor.numpy().flatten()
            n_bytes = len(t) * 4
            f.write(t.tobytes())
            manifest["tensors"][short_name] = {
                "offset": offset,
                "length": len(t),
                "shape": shape,
            }
            print(f"  {short_name}: {shape} → {n_bytes} bytes @ offset {offset}")
            offset += n_bytes

    manifest["total_bytes"] = offset

    # Infer num_actions and obs_size from weight shapes
    img_in_shape = manifest["tensors"].get("img_in_w", {}).get("shape", [512, 1025])
    manifest["num_actions"] = img_in_shape[1] - 1024  # stoch_flat=1024, remainder is actions
    enc_shape = manifest["tensors"].get("enc_linear0_w", {}).get("shape", [512, 80])
    manifest["obs_size"] = enc_shape[1]

    manifest_path = os.path.join(output_dir, "dreamer_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = os.path.getsize(bin_path) / 1e6
    print(f"\n  {bin_path}: {size_mb:.1f} MB")
    print(f"  {manifest_path}: {os.path.getsize(manifest_path)} bytes")

    # Also delete the old JSON if it exists
    old_json = os.path.join(output_dir, "dreamer_weights.json")
    if os.path.exists(old_json):
        os.remove(old_json)
        print(f"  Removed old {old_json}")


def export_pca_and_centroids(output_dir):
    """Export PCA components and count centroids from existing h_t data."""
    h_t_all, counts_all = None, None

    # Try video episode data first (has the most data after a collection run)
    vid_path = "/workspace/bridge/artifacts/video/episode_data.npz"
    if os.path.isfile(vid_path):
        data = np.load(vid_path)
        if "dreamer_h_t" in data:
            h_t_all = data["dreamer_h_t"]
            # video data uses 'grid_filled' not 'counts'
            counts_all = data["grid_filled"] if "grid_filled" in data else data["counts"]
            print(f"  Loaded video episode_data: {len(h_t_all)} samples")

    # Fallback: tensors from eval runs
    if h_t_all is None:
        for name in ["extrap_frozen_v3", "extrap_frozen_v2", "test_eval"]:
            h_path = f"/workspace/bridge/artifacts/tensors/{name}/h_t.npy"
            c_path = f"/workspace/bridge/artifacts/tensors/{name}/counts.npy"
            if os.path.isfile(h_path):
                h_t_all = np.load(h_path)
                counts_all = np.load(c_path)
                print(f"  Loaded {name}: {len(h_t_all)} samples")
                break

    if h_t_all is None:
        print("  WARNING: No h_t data found. Skipping PCA export.")
        return

    print(f"  Total samples: {len(h_t_all)}, deter dim: {h_t_all.shape[1]}")

    # Compute centroids per count
    unique_counts = sorted(np.unique(counts_all).astype(int))
    centroids = {}
    for c in unique_counts:
        mask = counts_all == c
        if mask.sum() > 0:
            centroids[int(c)] = h_t_all[mask].mean(axis=0)

    centroid_matrix = np.stack([centroids[c] for c in sorted(centroids.keys())])
    centroid_counts = sorted(centroids.keys())

    # PCA on centroids
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(centroid_matrix)
    projected_centroids = pca.transform(centroid_matrix)

    result = {
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "centroids_2d": projected_centroids.tolist(),
        "centroid_counts": centroid_counts,
    }

    out_path = os.path.join(output_dir, "manifold_pca.json")
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        json.dump(result, f)
    size_kb = os.path.getsize(out_path) / 1e3
    print(f"  {size_kb:.1f} KB, {len(centroid_counts)} centroids, "
          f"PCA explained: {pca.explained_variance_ratio_.sum():.1%}")


def export_parity_test(checkpoint_dir, output_dir):
    """Run one short episode and export obs+deter for TS verification."""
    sys.path.insert(0, os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch"))
    sys.path.insert(0, "/workspace/bridge/scripts")

    from ruamel.yaml import YAML
    yaml = YAML(typ='safe', pure=True)
    import pathlib, argparse as ap
    import tools, networks

    configs_dir = os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")
    configs = yaml.load(pathlib.Path(configs_dir, "configs.yaml").read_text())
    defaults = {}
    for name in ["defaults", "counting_continuous"]:
        for k, v in configs[name].items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v
    defaults["logdir"] = str(checkpoint_dir)
    defaults["device"] = "cpu"
    defaults["compile"] = False
    defaults["steps"] = 100
    defaults["action_repeat"] = 1
    defaults["eval_every"] = 10000
    defaults["time_limit"] = 6000
    defaults["log_every"] = 1000
    defaults["num_actions"] = 1

    parser = ap.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])

    import gym, gym.spaces
    from dreamer import make_dataset, Dreamer

    OBS_SIZE = 80
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
    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to("cpu")
    agent.requires_grad_(False)
    checkpoint = torch.load(logdir / "latest.pt", map_location="cpu")
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    agent.eval()

    from envs.counting import CountingWorld

    os.environ["COUNTING_BLOB_MIN"] = "10"
    os.environ["COUNTING_BLOB_MAX"] = "10"
    env = CountingWorld("counting_world", seed=42)

    obs_seq, action_seq, deter_seq, stoch_seq, count_seq = [], [], [], [], []
    obs_raw = env.reset()
    state = None
    N_STEPS = 200

    for step in range(N_STEPS):
        vec = obs_raw["vector"][:OBS_SIZE]
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
        z_t = latent["stoch"].cpu().numpy().squeeze(0)
        action = policy_output["action"].cpu().numpy().squeeze(0)

        obs_seq.append(vec.tolist())
        action_seq.append(action.tolist())
        deter_seq.append(h_t.tolist())
        stoch_seq.append(z_t.flatten().tolist())
        count_seq.append(int(obs_raw["vector"][81]) if len(obs_raw["vector"]) > 81 else 0)

        obs_raw, reward, done, info = env.step(action.item() if hasattr(action, 'item') else float(action[0]))
        if done:
            break

    result = {
        "n_steps": len(obs_seq),
        "obs_size": OBS_SIZE,
        "deter_size": 512,
        "stoch_classes": 32,
        "stoch_categoricals": 32,
        "obs": obs_seq,
        "actions": action_seq,
        "deter": deter_seq,
        "stoch": stoch_seq,
        "counts": count_seq,
    }

    out_path = os.path.join(output_dir, "parity_test.json")
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        json.dump(result, f)
    size_kb = os.path.getsize(out_path) / 1e3
    print(f"  {size_kb:.1f} KB, {len(obs_seq)} steps")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="/workspace/bridge/artifacts/checkpoints/train_25blobs_v3")
    parser.add_argument("--output", default="/workspace/projects/jamstack-v1/packages/signal-app/public/models")
    parser.add_argument("--skip_parity", action="store_true")
    args = parser.parse_args()

    print("=== Exporting DreamerV3 weights (binary) ===")
    export_weights(args.checkpoint_dir, args.output)

    print("\n=== Exporting PCA + centroids ===")
    export_pca_and_centroids(args.output)

    if not args.skip_parity:
        print("\n=== Exporting parity test data ===")
        export_parity_test(args.checkpoint_dir, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
