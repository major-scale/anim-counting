#!/usr/bin/env python3
"""
Cross-Dimensional Probe Transfer — PRIMARY evaluation metric for multi-dim v2.

Train a ridge probe on D=2 hidden states, then apply it WITHOUT retraining
to D=3, D=4, D=5 hidden states. If R² stays high, the count representation
is geometrically identical across dimensionalities.

This is stronger than GHE or RSA — those tell you each dimension has a manifold.
The transfer probe tells you it's THE SAME manifold.

Usage:
    python probe_transfer_eval.py --weights-dir /path/to/model \
        --episodes 10 --blob-count 13

    # On RunPod with PyTorch checkpoint:
    python probe_transfer_eval.py --checkpoint-dir /path/to/checkpoint \
        --episodes 10 --blob-count 13
"""

import sys
import json
import os
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def collect_hidden_states_numpy(weights_dir, D, episodes, blob_count):
    """Collect deter states using FastRSSM (numpy, no PyTorch needed)."""
    from export_deter_centroids import FastRSSM
    from counting_env_multidim import MultiDimCountingWorldEnv

    # Load weights
    with open(Path(weights_dir) / "dreamer_manifest.json") as f:
        manifest = json.load(f)
    with open(Path(weights_dir) / "dreamer_weights.bin", "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"],
                            offset=entry["offset"]).copy()
        weights[name] = arr.reshape(entry["shape"])
    obs_size = weights["enc_linear0_w"].shape[1]

    env = MultiDimCountingWorldEnv(
        blob_count_min=blob_count, blob_count_max=blob_count,
        fixed_dim=D, proj_dim=obs_size,
    )
    model = FastRSSM(weights)

    all_h = []
    all_counts = []
    total = 0
    t0 = time.time()

    for ep in range(episodes):
        vec = env.reset()
        model.reset()
        done = False
        while not done:
            obs = vec[:obs_size].astype(np.float32)
            count = int(env._state.grid.filled_count)
            deter = model.step(obs, 0.0)
            all_h.append(deter)
            all_counts.append(count)
            total += 1
            vec, reward, done, info = env.step(-0.995)
        elapsed = time.time() - t0
        fps = total / elapsed if elapsed > 0 else 0
        print(f"    D={D} ep {ep+1}/{episodes}: {total} samples ({fps:.0f} steps/s)")

    env.close()
    return np.stack(all_h), np.array(all_counts)


def collect_hidden_states_pytorch(checkpoint_dir, D, episodes, blob_count, proj_dim=128):
    """Collect deter states using PyTorch agent (for RunPod evaluation)."""
    import torch
    from quick_ghe_multidim import load_config_multidim, load_agent_multidim

    os.environ['MULTIDIM_BLOB_MIN'] = str(blob_count)
    os.environ['MULTIDIM_BLOB_MAX'] = str(blob_count)
    os.environ['MULTIDIM_FIXED_DIM'] = str(D)
    os.environ['MULTIDIM_PROJ_DIM'] = str(proj_dim)

    config = load_config_multidim(checkpoint_dir, proj_dim)
    agent = load_agent_multidim(config, proj_dim)

    from envs.multidim import MultiDimCountingWorld
    env = MultiDimCountingWorld("multidim_world", seed=42)

    all_h = []
    all_counts = []
    total = 0
    t0 = time.time()

    for ep in range(episodes):
        obs_raw = env.reset()
        state = None
        done = False
        while not done:
            vec = obs_raw["vector"]
            inner_state = env._env._state
            count = inner_state.grid.filled_count if inner_state else 0

            obs_dict = {
                "vector": torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
                "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]],
                                         dtype=torch.float32),
                "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]],
                                        dtype=torch.float32),
                "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]],
                                            dtype=torch.float32),
            }
            reset = np.array([obs_raw["is_first"]])
            with torch.no_grad():
                policy_output, state = agent(obs_dict, reset, state, training=False)

            latent, _ = state
            h_t = latent["deter"].cpu().numpy().squeeze(0)
            all_h.append(h_t)
            all_counts.append(count)
            total += 1

            action = policy_output["action"].cpu().numpy().flatten()
            obs_raw, reward, done, info = env.step(action)

        elapsed = time.time() - t0
        fps = total / elapsed if elapsed > 0 else 0
        print(f"    D={D} ep {ep+1}/{episodes}: {total} samples ({fps:.0f} steps/s)")

    env.close()
    return np.stack(all_h), np.array(all_counts)


def train_ridge_probe(X, y, alpha=10.0):
    """Train ridge regression probe. Returns (weights, bias)."""
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y.astype(np.float64)
    w = np.linalg.solve(XtX, Xty)
    b = y.mean() - w @ X.mean(axis=0)
    return w, b


def evaluate_probe(X, y, w, b, max_count):
    """Evaluate a probe on given data. Returns dict of metrics."""
    preds_raw = X @ w + b
    preds_round = np.clip(np.round(preds_raw), 0, max_count).astype(int)
    y_int = y.astype(int)

    exact = (preds_round == y_int).mean()
    within1 = (np.abs(preds_round - y_int) <= 1).mean()

    ss_res = ((y.astype(np.float64) - preds_raw) ** 2).sum()
    ss_tot = ((y.astype(np.float64) - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Probe SNR
    probe_dir = w / np.linalg.norm(w)
    projections = X @ probe_dir
    count_means = {}
    within_vars = []
    for c in sorted(set(y_int)):
        mask = y_int == c
        if mask.sum() > 0:
            count_means[c] = projections[mask].mean()
        if mask.sum() > 1:
            within_vars.append(projections[mask].var())
    between_var = np.var(list(count_means.values()))
    within_var = np.mean(within_vars) if within_vars else 0.0
    snr = between_var / within_var if within_var > 0 else float("inf")

    return {
        "r2": float(r2),
        "exact_accuracy": float(exact),
        "within_1_accuracy": float(within1),
        "probe_snr": float(snr),
        "n_samples": len(y),
        "n_unique_counts": len(set(y_int)),
        "count_range": [int(y.min()), int(y.max())],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dimensional probe transfer evaluation")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Model weights dir (numpy FastRSSM mode)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="PyTorch checkpoint dir (RunPod mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <model-dir>/probe_transfer.json)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per dimensionality")
    parser.add_argument("--blob-count", type=int, default=13,
                        help="Fixed blob count per episode")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Dimensionalities to evaluate")
    parser.add_argument("--train-dim", type=int, default=2,
                        help="Dimensionality to train probe on (default: 2)")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regression alpha")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection dimensionality")
    args = parser.parse_args()

    if not args.weights_dir and not args.checkpoint_dir:
        print("ERROR: Specify --weights-dir or --checkpoint-dir")
        sys.exit(1)

    model_dir = args.weights_dir or args.checkpoint_dir
    use_pytorch = args.checkpoint_dir is not None

    output_path = args.output or str(Path(model_dir) / "probe_transfer.json")

    dims = args.dims
    train_dim = args.train_dim
    if train_dim not in dims:
        dims = [train_dim] + dims

    print("=" * 60)
    print("Cross-Dimensional Probe Transfer Evaluation")
    print("=" * 60)
    print(f"  Model:       {model_dir}")
    print(f"  Mode:        {'PyTorch' if use_pytorch else 'NumPy FastRSSM'}")
    print(f"  Train dim:   D={train_dim}")
    print(f"  Eval dims:   {dims}")
    print(f"  Episodes:    {args.episodes} per dim")
    print(f"  Blob count:  {args.blob_count}")
    print(f"  Alpha:       {args.alpha}")
    print()

    # --- Collect hidden states per dimensionality ---
    data_per_dim = {}
    for D in dims:
        print(f"\n  Collecting D={D}...")
        if use_pytorch:
            h, counts = collect_hidden_states_pytorch(
                args.checkpoint_dir, D, args.episodes, args.blob_count, args.proj_dim)
        else:
            h, counts = collect_hidden_states_numpy(
                args.weights_dir, D, args.episodes, args.blob_count)
        data_per_dim[D] = (h, counts)
        print(f"    {len(counts)} samples, counts {counts.min()}-{counts.max()}")

    # --- Train probe on train_dim ---
    print(f"\n  Training ridge probe on D={train_dim}...")
    X_train, y_train = data_per_dim[train_dim]
    w, b = train_ridge_probe(X_train, y_train, alpha=args.alpha)
    print(f"    Probe trained: {X_train.shape[0]} samples, {X_train.shape[1]}-dim")

    # --- Evaluate on all dimensions ---
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Probe trained on D={train_dim}, applied to all dims")
    print(f"{'=' * 60}")

    results = {
        "train_dim": train_dim,
        "alpha": args.alpha,
        "episodes_per_dim": args.episodes,
        "blob_count": args.blob_count,
        "dims": {},
    }

    # Also train per-dim probes for comparison
    per_dim_probes = {}
    for D in dims:
        X_d, y_d = data_per_dim[D]
        w_d, b_d = train_ridge_probe(X_d, y_d, alpha=args.alpha)
        per_dim_probes[D] = (w_d, b_d)

    header = f"{'Dim':>4} | {'Transfer R²':>12} | {'Native R²':>10} | " \
             f"{'Xfer Exact':>10} | {'Xfer ±1':>8} | {'Xfer SNR':>9} | {'Samples':>8}"
    print(header)
    print("-" * len(header))

    for D in dims:
        X_d, y_d = data_per_dim[D]

        # Transfer probe (trained on train_dim)
        transfer_metrics = evaluate_probe(X_d, y_d, w, b, args.blob_count)

        # Native probe (trained on this dim)
        w_d, b_d = per_dim_probes[D]
        native_metrics = evaluate_probe(X_d, y_d, w_d, b_d, args.blob_count)

        marker = " <-- TRAIN" if D == train_dim else ""
        print(f"D={D:2d} | {transfer_metrics['r2']:11.4f} | {native_metrics['r2']:9.4f} | "
              f"{transfer_metrics['exact_accuracy']:9.1%} | {transfer_metrics['within_1_accuracy']:7.1%} | "
              f"{transfer_metrics['probe_snr']:8.1f} | {transfer_metrics['n_samples']:>7d}{marker}")

        results["dims"][str(D)] = {
            "transfer": transfer_metrics,
            "native": native_metrics,
            "is_train_dim": D == train_dim,
        }

    # --- Headline metric: mean transfer R² across non-train dims ---
    transfer_r2s = [results["dims"][str(D)]["transfer"]["r2"]
                    for D in dims if D != train_dim]
    mean_transfer_r2 = np.mean(transfer_r2s) if transfer_r2s else 0.0
    results["mean_transfer_r2"] = float(mean_transfer_r2)
    results["transfer_r2_std"] = float(np.std(transfer_r2s)) if transfer_r2s else 0.0

    print(f"\n  HEADLINE: Mean transfer R² (excl train dim) = {mean_transfer_r2:.4f} "
          f"± {results['transfer_r2_std']:.4f}")

    if mean_transfer_r2 > 0.95:
        print("  VERDICT: SAME MANIFOLD — probe transfers cleanly across dimensions")
    elif mean_transfer_r2 > 0.80:
        print("  VERDICT: SIMILAR MANIFOLD — probe transfers with some degradation")
    else:
        print("  VERDICT: DIFFERENT MANIFOLDS — probe does not transfer")

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
