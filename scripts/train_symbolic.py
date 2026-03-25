#!/usr/bin/env python3
"""
Training Script for Symbolic Binary Specialist
================================================
Self-contained training loop for the symbolic RSSM on 4-bit binary counting.

Targets CPU/M1 Mac. No GPU required — the model is ~7M params with tiny
token observations, so training should be comparable to or faster than the
physical binary specialist.

Usage:
    python train_symbolic.py [--steps 500000] [--device cpu] [--seed 0]

Saves:
    artifacts/checkpoints/symbolic_binary_s{seed}/
        latest.pt          — latest checkpoint (model + optimizer + step)
        step_{N}.pt        — periodic checkpoints
        metrics.json       — training metrics history
        config.json        — full config for reproducibility
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from symbolic_binary_env import SymbolicBinaryEnv, NUM_BITS, MAX_COUNT, int_to_bits
from symbolic_rssm import (
    SymbolicRSSM, compute_total_loss, count_params,
    DETER_DIM, STOCH_FLAT, NUM_BITS as RSSM_BITS,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Training
    "steps": 500_000,
    "batch_size": 16,
    "batch_length": 64,
    "train_ratio": 32,
    "prefill": 5000,
    "lr": 1e-4,
    "grad_clip": 1000.0,
    "weight_decay": 0.0,
    "seed": 0,

    # Environment
    "steps_per_count": 20,
    "n_cycles": 1,

    # Checkpointing
    "checkpoint_every": 100_000,
    "log_every": 1000,
    "eval_every": 50_000,

    # Device
    "device": "cpu",
}


# ---------------------------------------------------------------------------
# Replay buffer (simple circular buffer of episodes)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple episode-based replay buffer storing token sequences."""

    def __init__(self, max_episodes=10000):
        self.max_episodes = max_episodes
        self.episodes = []
        self._total_steps = 0

    def add_episode(self, episode):
        """episode: dict with 'tokens' (T, 4), 'is_first' (T,), 'is_last' (T,)"""
        self.episodes.append(episode)
        self._total_steps += len(episode["tokens"])
        if len(self.episodes) > self.max_episodes:
            removed = self.episodes.pop(0)
            self._total_steps -= len(removed["tokens"])

    @property
    def total_steps(self):
        return self._total_steps

    def sample_batch(self, batch_size, batch_length, rng):
        """Sample random chunks from random episodes."""
        tokens_batch = []
        is_first_batch = []

        for _ in range(batch_size):
            # Pick random episode
            ep = self.episodes[rng.randint(0, len(self.episodes))]
            T = len(ep["tokens"])

            # Pick random start (allowing overlap with episode boundaries)
            if T <= batch_length:
                start = 0
                # Pad if episode shorter than batch_length
                pad_len = batch_length - T
                tokens = np.concatenate([ep["tokens"],
                                        np.zeros((pad_len, NUM_BITS), dtype=np.int64)])
                is_first = np.concatenate([ep["is_first"],
                                          np.zeros(pad_len, dtype=np.float32)])
            else:
                start = rng.randint(0, T - batch_length)
                tokens = ep["tokens"][start:start + batch_length]
                is_first = ep["is_first"][start:start + batch_length]

            tokens_batch.append(tokens)
            is_first_batch.append(is_first)

        return {
            "tokens": np.stack(tokens_batch),      # (B, T, 4)
            "is_first": np.stack(is_first_batch),   # (B, T)
        }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_episode(env):
    """Run one episode, return dict of numpy arrays."""
    obs = env.reset()
    tokens_list = [obs["tokens"].copy()]
    is_first_list = [1.0]
    counts_list = [env.metadata["decimal_count"]]

    done = False
    while not done:
        obs, _, done, info = env.step(0)
        tokens_list.append(obs["tokens"].copy())
        is_first_list.append(0.0)
        counts_list.append(info.get("decimal_count", 0))

    return {
        "tokens": np.stack(tokens_list).astype(np.int64),    # (T, 4)
        "is_first": np.array(is_first_list, dtype=np.float32),  # (T,)
        "counts": np.array(counts_list, dtype=np.int32),       # (T,) for eval
    }


# ---------------------------------------------------------------------------
# Quick evaluation
# ---------------------------------------------------------------------------

def quick_eval(model, device, seed=42):
    """Run a few episodes and measure token prediction + count probe accuracy."""
    model.eval()
    env = SymbolicBinaryEnv(steps_per_count=20, n_cycles=2, seed=seed)

    all_h_t = []
    all_counts = []
    total_correct = 0
    total_tokens = 0

    for ep_idx in range(5):
        obs = env.reset()
        h, z = model.initial_state(1, device)
        action = torch.zeros(1, 1, device=device)
        done = False
        step = 0

        while not done:
            tokens = torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0)
            is_first = torch.tensor([[1.0 if obs["is_first"] else 0.0]],
                                   dtype=torch.float32, device=device)

            # Reset state at episode boundary
            if obs["is_first"]:
                h, z = model.initial_state(1, device)

            with torch.no_grad():
                out = model.observe_step(tokens, action, h, z)

            h = out["deter"]
            z = out["stoch"]

            # Token prediction accuracy
            logits = model.decoder(out["feat"])  # (1, 4, 2)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            correct = (preds == obs["tokens"]).sum()
            total_correct += correct
            total_tokens += NUM_BITS

            all_h_t.append(h.squeeze(0).cpu().numpy())
            all_counts.append(env.metadata["decimal_count"])

            obs, _, done, _ = env.step(0)
            step += 1

    token_acc = total_correct / total_tokens

    # Linear probe for count
    h_t = np.stack(all_h_t)
    counts = np.array(all_counts)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    # Count exact accuracy via Ridge + rounding
    ridge = Ridge(alpha=1.0)
    ridge.fit(h_t, counts)
    preds = ridge.predict(h_t)
    count_exact = np.mean(np.round(preds).astype(int) == counts)

    # Per-bit probe accuracy
    bits = np.array([int_to_bits(c) for c in counts])
    bit_accs = []
    for b in range(NUM_BITS):
        ridge_bit = Ridge(alpha=1.0)
        ridge_bit.fit(h_t, bits[:, b])
        bit_preds = (ridge_bit.predict(h_t) > 0.5).astype(int)
        bit_accs.append(np.mean(bit_preds == bits[:, b]))

    # Effective rank
    cov = np.cov(h_t.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    probs = eigenvalues / eigenvalues.sum()
    erank = np.exp(-np.sum(probs * np.log(probs + 1e-30)))

    model.train()
    return {
        "token_acc": float(token_acc),
        "count_exact": float(count_exact),
        "bit_accs": [float(a) for a in bit_accs],
        "erank": float(erank),
        "n_timesteps": len(counts),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train symbolic binary specialist")
    for key, val in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    config = vars(args)

    # Device
    device = torch.device(config["device"])
    if config["device"] == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    config["device"] = str(device)

    # Seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Output directory
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    output_dir = artifacts_dir / "checkpoints" / f"symbolic_binary_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Symbolic Binary Specialist — Training")
    print("=" * 60)
    print(f"  Device:         {device}")
    print(f"  Seed:           {seed}")
    print(f"  Steps:          {config['steps']:,}")
    print(f"  Batch:          {config['batch_size']} × {config['batch_length']}")
    print(f"  LR:             {config['lr']}")
    print(f"  Output:         {output_dir}")
    print()

    # Create model
    model = SymbolicRSSM().to(device)
    total_params, parts = count_params(model)
    print(f"  Parameters:     {total_params:,}")
    for name, count in parts.items():
        print(f"    {name}: {count:,}")
    print()

    # Optimizer (matching DreamerV3: Adam with eps=1e-8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        eps=1e-8,
        weight_decay=config["weight_decay"],
    )

    # Replay buffer
    buffer = ReplayBuffer(max_episodes=10000)

    # Environment for data collection
    env = SymbolicBinaryEnv(
        steps_per_count=config["steps_per_count"],
        n_cycles=config["n_cycles"],
        seed=seed,
    )

    # Prefill
    print(f"Prefilling {config['prefill']} steps...")
    while buffer.total_steps < config["prefill"]:
        ep = collect_episode(env)
        buffer.add_episode(ep)
    print(f"  Prefilled: {buffer.total_steps} steps, {len(buffer.episodes)} episodes")
    print()

    # Training
    metrics_history = []
    running_metrics = defaultdict(list)
    batch_steps = config["batch_size"] * config["batch_length"]
    train_every = max(1, batch_steps // config["train_ratio"])

    step = 0
    ep_count = len(buffer.episodes)
    t_start = time.time()

    print("Training...")
    while step < config["steps"]:
        # Collect new episode periodically
        if step % train_every == 0:
            ep = collect_episode(env)
            buffer.add_episode(ep)
            ep_count += 1

        # Sample batch
        batch = buffer.sample_batch(config["batch_size"], config["batch_length"], rng)
        tokens = torch.tensor(batch["tokens"], dtype=torch.long, device=device)
        is_first = torch.tensor(batch["is_first"], dtype=torch.float32, device=device)
        actions = torch.zeros(config["batch_size"], config["batch_length"], 1,
                            dtype=torch.float32, device=device)

        # Forward + loss
        model.train()
        outputs = model(tokens, actions, is_first)
        losses = compute_total_loss(model, outputs, tokens)

        # Backward + clip + step
        optimizer.zero_grad()
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        # Track metrics
        for k, v in losses.items():
            running_metrics[k].append(v.item())
        running_metrics["grad_norm"].append(grad_norm.item())

        step += 1

        # Log
        if step % config["log_every"] == 0:
            elapsed = time.time() - t_start
            sps = step / elapsed
            entry = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "steps_per_sec": round(sps, 1),
                "episodes": ep_count,
                "buffer_steps": buffer.total_steps,
            }
            for k, v in running_metrics.items():
                entry[k] = round(np.mean(v), 4)
            running_metrics.clear()

            metrics_history.append(entry)

            # Print compact status
            print(f"  [{step:>7,}/{config['steps']:,}] "
                  f"loss={entry['total']:.3f} "
                  f"dec={entry['decoder']:.3f} "
                  f"kl={entry['dyn_kl']:.2f}/{entry['rep_kl']:.2f} "
                  f"acc={entry['token_acc']:.3f} "
                  f"gnorm={entry['grad_norm']:.1f} "
                  f"({sps:.0f} steps/s)")

        # Eval checkpoint
        if step % config["eval_every"] == 0:
            print(f"\n  Eval at step {step:,}...")
            eval_metrics = quick_eval(model, device, seed=42)
            print(f"    token_acc={eval_metrics['token_acc']:.4f} "
                  f"count_exact={eval_metrics['count_exact']:.4f} "
                  f"bit_accs={[f'{a:.3f}' for a in eval_metrics['bit_accs']]} "
                  f"erank={eval_metrics['erank']:.1f}")
            entry = {"step": step, "eval": eval_metrics}
            metrics_history.append(entry)
            print()

        # Save checkpoint
        if step % config["checkpoint_every"] == 0 or step == config["steps"]:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }
            torch.save(ckpt, output_dir / f"step_{step}.pt")
            torch.save(ckpt, output_dir / "latest.pt")
            print(f"  Checkpoint saved: step_{step}.pt")

            # Save metrics
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_history, f, indent=2)

    # Final eval
    elapsed = time.time() - t_start
    print(f"\nTraining complete: {config['steps']:,} steps in {elapsed:.0f}s "
          f"({config['steps']/elapsed:.0f} steps/s)")

    print("\nFinal evaluation...")
    eval_metrics = quick_eval(model, device, seed=42)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")

    # Save final metrics
    metrics_history.append({"step": step, "final_eval": eval_metrics})
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def run_random_baseline(device="cpu", n_seeds=3):
    """Run untrained random RSSM on same observation stream for comparison."""
    print("=" * 60)
    print("Random Baseline — Untrained SymbolicRSSM")
    print("=" * 60)

    results = []
    for s in range(n_seeds):
        torch.manual_seed(s + 100)
        model = SymbolicRSSM().to(device)
        model.eval()
        metrics = quick_eval(model, device, seed=42)
        print(f"  Seed {s}: token_acc={metrics['token_acc']:.4f} "
              f"count_exact={metrics['count_exact']:.4f} "
              f"bit_accs={[f'{a:.3f}' for a in metrics['bit_accs']]} "
              f"erank={metrics['erank']:.1f}")
        results.append(metrics)

    # Summarize
    print("\nRandom baseline summary:")
    for key in ["token_acc", "count_exact", "erank"]:
        vals = [r[key] for r in results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return results


if __name__ == "__main__":
    if "--random-baseline" in sys.argv:
        sys.argv.remove("--random-baseline")
        run_random_baseline()
    else:
        main()
