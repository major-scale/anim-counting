#!/usr/bin/env python3
"""
Reservoir Computing Control — Echo State Network vs Trained RSSM
================================================================
Tests whether a random reservoir (ESN) with the same architecture
can match the trained RSSM's binary counting capabilities.

Key distinction from random RSSM baseline:
- Random RSSM uses DreamerV3 architecture but random weights
- ESN uses a standard echo state network architecture
- Both are "random recurrent networks" but different families

If ESN matches trained RSSM → learned structure is just
what any recurrent network gives you for free.
If ESN < trained RSSM → trained structure is genuinely special.

Tests:
1. ESN with same input dimension, same hidden size
2. Probe accuracy: count, per-bit, carry
3. RSA: ordinal, Hamming
4. Imagination: can ESN generate carry cascades?
5. Observation cliff: does ESN show same failure mode?

Output: artifacts/binary_successor/esn_control.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, accuracy_score
from scipy.spatial.distance import pdist, squareform

BATTERY_PATH = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")
OUT_DIR = Path("/workspace/bridge/artifacts/binary_successor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class EchoStateNetwork:
    """Simple Echo State Network for reservoir computing control."""

    def __init__(self, input_dim, hidden_dim, spectral_radius=0.95, input_scale=0.1,
                 leak_rate=0.3, seed=42):
        rng = np.random.RandomState(seed)

        self.hidden_dim = hidden_dim
        self.leak_rate = leak_rate

        # Input weights (sparse)
        self.W_in = rng.randn(hidden_dim, input_dim) * input_scale

        # Recurrent weights (sparse, scaled to spectral radius)
        # Use ~10% connectivity for classic ESN
        W = rng.randn(hidden_dim, hidden_dim)
        mask = rng.rand(hidden_dim, hidden_dim) < 0.1
        W *= mask
        # Scale to desired spectral radius
        current_sr = np.max(np.abs(np.linalg.eigvals(W)))
        if current_sr > 0:
            W *= spectral_radius / current_sr
        self.W = W

        # Bias
        self.b = rng.randn(hidden_dim) * 0.01

    def forward_sequence(self, inputs):
        """Run ESN on input sequence. Returns all hidden states.

        Args:
            inputs: (T, input_dim) array of observations

        Returns:
            h_all: (T, hidden_dim) array of hidden states
        """
        T, _ = inputs.shape
        h = np.zeros(self.hidden_dim)
        h_all = np.zeros((T, self.hidden_dim))

        for t in range(T):
            # ESN update: h = (1-α)h + α·tanh(W_in·x + W·h + b)
            pre = self.W_in @ inputs[t] + self.W @ h + self.b
            h_new = np.tanh(pre)
            h = (1 - self.leak_rate) * h + self.leak_rate * h_new
            h_all[t] = h

        return h_all

    def imagination_step(self, h):
        """Single imagination step (no input, just recurrence).

        Uses zero input — equivalent to RSSM prior mode.
        """
        pre = self.W @ h + self.b  # No input contribution
        h_new = np.tanh(pre)
        h = (1 - self.leak_rate) * h + self.leak_rate * h_new
        return h


def generate_binary_observations(n_episodes=15, steps_per_episode=100, seed=42):
    """Generate binary counting observation sequences.

    Mimics what the binary_counting_env produces:
    - 4-bit counter: counts 0-14 repeatedly
    - Each "step" increments count by 1
    - Observations encode bits visually (simplified to bit vectors here)
    """
    rng = np.random.RandomState(seed)
    all_obs = []
    all_counts = []
    all_bits = []

    for ep in range(n_episodes):
        for step in range(steps_per_episode):
            count = step % 15  # 0-14
            bits_arr = [(count >> b) & 1 for b in range(4)]

            # Create observation vector that encodes bits
            # Use a rich observation like the actual env (72-d)
            obs = np.zeros(72)
            # Encode each bit in a region of the observation
            for b in range(4):
                obs[b * 18:(b + 1) * 18] = bits_arr[b]
                # Add some noise
                obs[b * 18:(b + 1) * 18] += rng.randn(18) * 0.01

            all_obs.append(obs)
            all_counts.append(count)
            all_bits.append(bits_arr)

    return np.array(all_obs), np.array(all_counts), np.array(all_bits)


def compute_rsa(h_t, counts, metric='ordinal'):
    """Compute RSA between hidden state distances and count distances."""
    unique_counts = sorted(np.unique(counts))
    centroids = []
    for c in unique_counts:
        mask = counts == c
        centroids.append(h_t[mask].mean(axis=0))
    centroids = np.array(centroids)

    centroid_dists = squareform(pdist(centroids))

    if metric == 'ordinal':
        # Ordinal distance: |i - j|
        n = len(unique_counts)
        target_dists = np.abs(np.subtract.outer(
            np.array(unique_counts), np.array(unique_counts)
        )).astype(float)
    elif metric == 'hamming':
        # Hamming distance
        codes = np.array([[(c >> b) & 1 for b in range(4)] for c in unique_counts])
        target_dists = squareform(pdist(codes, metric='hamming')) * 4

    upper_tri = np.triu_indices(len(unique_counts), k=1)
    r, p = spearmanr(centroid_dists[upper_tri], target_dists[upper_tri])
    return float(r), float(p)


def train_probes(h_train, y_train, h_test, y_test):
    """Train Ridge probe and return test accuracy/R²."""
    probe = Ridge(alpha=1.0)
    probe.fit(h_train, y_train)
    y_pred = probe.predict(h_test)

    if len(np.unique(y_train)) <= 15:
        # Classification-like: round to nearest integer
        y_pred_round = np.round(y_pred).astype(int)
        y_pred_round = np.clip(y_pred_round, y_train.min(), y_train.max())
        acc = float(accuracy_score(y_test, y_pred_round))
        r2 = float(r2_score(y_test, y_pred))
        return {"accuracy": acc, "r2": r2}
    else:
        r2 = float(r2_score(y_test, y_pred))
        return {"r2": r2}


def run_esn_experiment(h_t_rssm, obs_sequences, counts, bits, episode_boundaries, seed=42):
    """Run full ESN experiment.

    Uses the actual observations from the battery to drive the ESN,
    then compares ESN hidden states to RSSM hidden states.
    """
    results = {}

    # Build ESN with same hidden size as RSSM
    esn = EchoStateNetwork(
        input_dim=72,  # Same as binary env observation
        hidden_dim=512,  # Same as RSSM GRU
        spectral_radius=0.95,
        input_scale=0.1,
        leak_rate=0.3,
        seed=seed,
    )

    # We need actual observations to drive the ESN
    # The battery.npz doesn't store raw obs, so we reconstruct from bits
    print("  Generating observation sequences from bits...")
    n_samples = len(counts)
    obs = np.zeros((n_samples, 72))
    for i in range(n_samples):
        for b in range(4):
            obs[i, b * 18:(b + 1) * 18] = bits[i, b]

    # Run ESN on observation sequences per episode
    print("  Running ESN on observation sequences...")
    h_esn = np.zeros((n_samples, 512))
    ep_starts = [0] + list(np.where(np.diff(episode_boundaries) != 0)[0] + 1)
    ep_starts.append(n_samples)

    for i in range(len(ep_starts) - 1):
        start, end = ep_starts[i], ep_starts[i + 1]
        ep_obs = obs[start:end]
        h_ep = esn.forward_sequence(ep_obs)
        h_esn[start:end] = h_ep

    # Train/test split: first 10 episodes train, last 5 test
    n_train = ep_starts[10] if len(ep_starts) > 10 else int(0.7 * n_samples)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_samples)

    # --- Probe accuracy ---
    print("  Training probes on ESN hidden states...")

    # Count probe
    count_result = train_probes(
        h_esn[train_idx], counts[train_idx],
        h_esn[test_idx], counts[test_idx]
    )
    results["count_probe"] = count_result
    print(f"    Count: acc={count_result['accuracy']:.3f}, R²={count_result['r2']:.3f}")

    # Per-bit probes
    bit_results = {}
    for b in range(4):
        bit_result = train_probes(
            h_esn[train_idx], bits[train_idx, b],
            h_esn[test_idx], bits[test_idx, b]
        )
        bit_results[f"bit{b}"] = bit_result
        print(f"    Bit {b}: acc={bit_result['accuracy']:.3f}, R²={bit_result['r2']:.3f}")
    results["bit_probes"] = bit_results

    # --- RSA ---
    print("  Computing RSA...")
    rsa_ord_r, rsa_ord_p = compute_rsa(h_esn, counts, 'ordinal')
    rsa_ham_r, rsa_ham_p = compute_rsa(h_esn, counts, 'hamming')
    results["rsa"] = {
        "ordinal": {"r": rsa_ord_r, "p": rsa_ord_p},
        "hamming": {"r": rsa_ham_r, "p": rsa_ham_p},
    }
    print(f"    RSA ordinal: r={rsa_ord_r:.3f}")
    print(f"    RSA Hamming: r={rsa_ham_r:.3f}")

    # Also compute RSA on RSSM for direct comparison
    rssm_rsa_ord_r, _ = compute_rsa(h_t_rssm, counts, 'ordinal')
    rssm_rsa_ham_r, _ = compute_rsa(h_t_rssm, counts, 'hamming')
    results["rssm_rsa"] = {
        "ordinal": {"r": rssm_rsa_ord_r},
        "hamming": {"r": rssm_rsa_ham_r},
    }

    # --- Imagination test ---
    print("  Testing imagination (no-input rollout)...")
    # Start from a count-7 state (deepest cascade: 7→8)
    mask_7 = counts == 7
    idx_7 = np.where(mask_7)[0]
    if len(idx_7) > 0:
        # Use last h_esn at count 7
        h_start = h_esn[idx_7[0]]

        # Roll forward 20 steps with no input
        h_imag = [h_start]
        for step in range(20):
            h_next = esn.imagination_step(h_imag[-1])
            h_imag.append(h_next)
        h_imag = np.array(h_imag)

        # Probe the imagined states for count
        count_probe_full = Ridge(alpha=1.0)
        count_probe_full.fit(h_esn, counts)
        imag_counts = count_probe_full.predict(h_imag)

        results["imagination"] = {
            "start_count": 7,
            "n_steps": 20,
            "predicted_counts": imag_counts.tolist(),
            "expected_sequence": list(range(7, min(15, 7 + 21))),
            "first_5_predictions": imag_counts[:5].tolist(),
        }
        print(f"    From count 7, predictions: {[f'{c:.1f}' for c in imag_counts[:10]]}")

    # --- Observation cliff test ---
    print("  Testing observation cliff...")
    # Run ESN normally, then blank observations
    # Use first episode
    ep_obs = obs[:ep_starts[1]]
    h_normal = esn.forward_sequence(ep_obs)

    # Now run with blanked obs after step 50
    ep_obs_blanked = ep_obs.copy()
    ep_obs_blanked[50:] = 0  # Zero out observations
    h_blanked = esn.forward_sequence(ep_obs_blanked)

    # Compare count probe accuracy before/after blanking
    count_probe_full = Ridge(alpha=1.0)
    count_probe_full.fit(h_esn, counts)

    pred_normal = count_probe_full.predict(h_normal)
    pred_blanked = count_probe_full.predict(h_blanked)

    # Accuracy on steps 0-50 vs 51+
    ep_counts = counts[:ep_starts[1]]
    normal_before = np.mean(np.round(pred_normal[:50]) == ep_counts[:50])
    normal_after = np.mean(np.round(pred_normal[50:]) == ep_counts[50:])
    blanked_before = np.mean(np.round(pred_blanked[:50]) == ep_counts[:50])
    blanked_after = np.mean(np.round(pred_blanked[50:]) == ep_counts[50:])

    results["observation_cliff"] = {
        "normal_acc_before50": float(normal_before),
        "normal_acc_after50": float(normal_after),
        "blanked_acc_before50": float(blanked_before),
        "blanked_acc_after50": float(blanked_after),
        "cliff_drop": float(blanked_before - blanked_after),
    }
    print(f"    Normal: {normal_before:.3f} → {normal_after:.3f}")
    print(f"    Blanked: {blanked_before:.3f} → {blanked_after:.3f}")

    return results


def main():
    print("Loading battery data...")
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']      # (13280, 512)
    counts = data['counts']  # (13280,)
    bits = data['bits']      # (13280, 4)

    print(f"  h_t: {h_t.shape}, counts: {counts.shape}")

    # Create episode boundaries (each episode is ~885 steps based on 13280/15)
    ep_size = len(h_t) // 15
    episode_ids = np.repeat(np.arange(15), ep_size)
    # Handle remainder
    if len(episode_ids) < len(h_t):
        episode_ids = np.concatenate([episode_ids, np.full(len(h_t) - len(episode_ids), 14)])
    episode_ids = episode_ids[:len(h_t)]

    all_results = {}

    # Run 3 seeds for stability
    for seed in [42, 123, 456]:
        print(f"\n{'='*60}")
        print(f"ESN Seed {seed}")
        print(f"{'='*60}")
        results = run_esn_experiment(h_t, None, counts, bits, episode_ids, seed=seed)
        all_results[f"seed_{seed}"] = results

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (3 seeds)")
    print(f"{'='*60}")

    agg = {}

    # Count accuracy
    count_accs = [all_results[f"seed_{s}"]["count_probe"]["accuracy"] for s in [42, 123, 456]]
    agg["count_accuracy"] = {
        "mean": float(np.mean(count_accs)),
        "std": float(np.std(count_accs)),
        "values": count_accs,
    }
    print(f"Count accuracy: {np.mean(count_accs):.3f} ± {np.std(count_accs):.3f}")

    # Per-bit accuracy
    for b in range(4):
        bit_accs = [all_results[f"seed_{s}"]["bit_probes"][f"bit{b}"]["accuracy"] for s in [42, 123, 456]]
        agg[f"bit{b}_accuracy"] = {
            "mean": float(np.mean(bit_accs)),
            "std": float(np.std(bit_accs)),
        }
        print(f"Bit {b} accuracy: {np.mean(bit_accs):.3f} ± {np.std(bit_accs):.3f}")

    # RSA
    for metric in ['ordinal', 'hamming']:
        rsas = [all_results[f"seed_{s}"]["rsa"][metric]["r"] for s in [42, 123, 456]]
        agg[f"rsa_{metric}"] = {
            "mean": float(np.mean(rsas)),
            "std": float(np.std(rsas)),
        }
        rssm_r = all_results["seed_42"]["rssm_rsa"][metric]["r"]
        print(f"RSA {metric}: ESN={np.mean(rsas):.3f}±{np.std(rsas):.3f}, RSSM={rssm_r:.3f}")

    all_results["aggregate"] = agg

    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON: ESN vs Trained RSSM vs Random RSSM")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'ESN':<15} {'Trained RSSM':<15} {'Random RSSM':<15}")
    print(f"{'-'*70}")
    print(f"{'Count accuracy':<25} {agg['count_accuracy']['mean']:.3f}±{agg['count_accuracy']['std']:.3f}      {'1.000':<15} {'0.587':<15}")
    print(f"{'RSA ordinal':<25} {agg['rsa_ordinal']['mean']:.3f}±{agg['rsa_ordinal']['std']:.3f}      {'0.500':<15} {'0.470':<15}")
    print(f"{'RSA Hamming':<25} {agg['rsa_hamming']['mean']:.3f}±{agg['rsa_hamming']['std']:.3f}      {'0.558':<15} {'0.337':<15}")

    # Save
    out_path = OUT_DIR / "esn_control.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
