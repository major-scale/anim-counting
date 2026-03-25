#!/usr/bin/env python3
"""
Last-Layer Laplace Uncertainty Analysis
========================================
Fits a last-layer Laplace approximation to count probes to get
calibrated uncertainty estimates. Tests whether the RSSM "knows
what it doesn't know" — particularly around carry cascades and
the observation cliff.

Key questions:
1. Does uncertainty peak at carry transitions (7→8, 3→4, etc.)?
2. Does uncertainty correlate with cascade depth?
3. Does uncertainty spike during imagination (prior-only) vs posterior?
4. Can Laplace detect the observation cliff before it happens?

Approach: Fit Ridge probe, compute Hessian of last layer, get
posterior predictive uncertainty via Laplace approximation.

Output: artifacts/binary_successor/laplace_uncertainty.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

BATTERY_PATH = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")
OUT_DIR = Path("/workspace/bridge/artifacts/binary_successor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class LaplaceProbe:
    """Last-layer Laplace approximation for a Ridge regression probe.

    For Ridge regression y = Xw + b with prior N(0, α⁻¹I):
    - MAP estimate: w* = (X'X + αI)⁻¹X'y
    - Posterior: N(w*, Σ) where Σ = (X'X + αI)⁻¹ · σ²
    - Predictive: N(x'w*, σ² + x'Σx)

    The predictive variance σ²_pred(x) = σ² + x'Σx gives us
    calibrated uncertainty for each input.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.sigma2 = None
        self.Sigma = None
        self.mean_x = None

    def fit(self, X, y):
        """Fit Ridge probe and compute Laplace posterior."""
        n, d = X.shape

        # Center features for numerical stability
        self.mean_x = X.mean(axis=0)
        X_c = X - self.mean_x

        # MAP estimate (Ridge)
        XtX = X_c.T @ X_c
        reg = self.alpha * np.eye(d)
        A = XtX + reg

        # Solve for weights
        self.w = np.linalg.solve(A, X_c.T @ y)
        self.b = y.mean() - self.mean_x @ self.w

        # Residual variance
        y_pred = X_c @ self.w + y.mean()
        residuals = y - y_pred
        self.sigma2 = float(np.mean(residuals ** 2))

        # Posterior covariance: Σ = σ² · (X'X + αI)⁻¹
        # For efficiency, store A⁻¹ and multiply by σ² at prediction time
        self.A_inv = np.linalg.inv(A)

        return self

    def predict(self, X):
        """Predict with uncertainty."""
        X_c = X - self.mean_x

        # Mean prediction
        y_mean = X_c @ self.w + self.b

        # Predictive variance for each point
        # σ²_pred(x) = σ² · (1 + x' A⁻¹ x)
        # Vectorized: diag(X_c @ A_inv @ X_c.T)
        leverage = np.sum((X_c @ self.A_inv) * X_c, axis=1)
        y_var = self.sigma2 * (1 + leverage)
        y_std = np.sqrt(np.maximum(y_var, 0))

        return y_mean, y_std

    def epistemic_uncertainty(self, X):
        """Return just the epistemic (model) uncertainty component."""
        X_c = X - self.mean_x
        leverage = np.sum((X_c @ self.A_inv) * X_c, axis=1)
        return self.sigma2 * leverage

    def aleatoric_uncertainty(self):
        """Return the aleatoric (noise) uncertainty component."""
        return self.sigma2


def carry_depth(count):
    """Compute carry cascade depth for count→count+1 transition."""
    if count >= 14:
        return -1  # No transition from 14
    next_count = count + 1
    depth = 0
    xor = count ^ next_count
    while xor > 0:
        depth += 1
        xor >>= 1
    return depth - 1  # depth 0 = no carry, 1 = 1-bit carry, etc.


def main():
    print("Loading battery data...")
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']      # (13280, 512)
    counts = data['counts']  # (13280,)
    bits = data['bits']      # (13280, 4)

    print(f"  h_t: {h_t.shape}, counts: {counts.shape}")

    # Reduce dimensionality for Laplace (full 512 is expensive for matrix inverse)
    from sklearn.decomposition import PCA
    n_components = 50
    pca = PCA(n_components=n_components)
    h_pca = pca.fit_transform(h_t)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {n_components} dims: {var_explained*100:.1f}% variance explained")

    results = {}

    # --- Experiment 1: Fit Laplace probe and get per-count uncertainty ---
    print("\n[1] Fitting Laplace probe on count...")
    probe = LaplaceProbe(alpha=1.0)
    probe.fit(h_pca, counts.astype(float))

    y_pred, y_std = probe.predict(h_pca)
    accuracy = float(np.mean(np.round(y_pred).astype(int) == counts))
    print(f"  Probe accuracy: {accuracy:.3f}")
    print(f"  Residual σ²: {probe.sigma2:.4f}")

    # Per-count uncertainty
    print("\n  Per-count uncertainty:")
    per_count = {}
    for c in range(15):
        mask = counts == c
        mean_std = float(y_std[mask].mean())
        mean_epist = float(probe.epistemic_uncertainty(h_pca[mask]).mean())
        depth = carry_depth(c)
        per_count[str(c)] = {
            "count": c,
            "carry_depth": depth,
            "mean_predictive_std": mean_std,
            "mean_epistemic_uncertainty": mean_epist,
            "n_samples": int(mask.sum()),
        }
        print(f"    Count {c:>2} (depth={depth}): σ_pred={mean_std:.4f}, σ_epist={mean_epist:.6f}")

    results["per_count_uncertainty"] = per_count

    # --- Experiment 2: Uncertainty vs carry depth correlation ---
    print("\n[2] Uncertainty vs carry depth...")
    depths = np.array([carry_depth(c) for c in counts])
    valid = depths >= 0  # Exclude count 14

    # Per-sample uncertainty
    sample_std = y_std[valid]
    sample_depth = depths[valid]
    sample_epist = probe.epistemic_uncertainty(h_pca[valid])

    r_pred, p_pred = spearmanr(sample_depth, sample_std)
    r_epist, p_epist = spearmanr(sample_depth, sample_epist)

    results["depth_correlation"] = {
        "predictive_std": {"spearman_r": float(r_pred), "p": float(p_pred)},
        "epistemic": {"spearman_r": float(r_epist), "p": float(p_epist)},
    }
    print(f"  Predictive σ vs depth: r={r_pred:.3f} (p={p_pred:.1e})")
    print(f"  Epistemic σ vs depth: r={r_epist:.3f} (p={p_epist:.1e})")

    # Mean uncertainty by depth
    for d in sorted(np.unique(sample_depth)):
        d_mask = sample_depth == d
        print(f"    Depth {int(d)}: pred_σ={sample_std[d_mask].mean():.4f}, "
              f"epist={sample_epist[d_mask].mean():.6f}, n={d_mask.sum()}")
        results["depth_correlation"][f"depth_{int(d)}"] = {
            "mean_pred_std": float(sample_std[d_mask].mean()),
            "mean_epistemic": float(sample_epist[d_mask].mean()),
            "n": int(d_mask.sum()),
        }

    # --- Experiment 3: Transition uncertainty ---
    print("\n[3] Transition vs steady-state uncertainty...")
    # Identify transition steps (where count changes)
    transitions = np.where(np.diff(counts) != 0)[0]
    steady = np.where(np.diff(counts) == 0)[0]

    if len(transitions) > 0 and len(steady) > 0:
        trans_std = y_std[transitions].mean()
        steady_std = y_std[steady].mean()
        trans_epist = probe.epistemic_uncertainty(h_pca[transitions]).mean()
        steady_epist = probe.epistemic_uncertainty(h_pca[steady]).mean()

        results["transition_vs_steady"] = {
            "transition_mean_std": float(trans_std),
            "steady_mean_std": float(steady_std),
            "ratio_std": float(trans_std / steady_std),
            "transition_mean_epist": float(trans_epist),
            "steady_mean_epist": float(steady_epist),
            "ratio_epist": float(trans_epist / steady_epist),
            "n_transitions": len(transitions),
            "n_steady": len(steady),
        }
        print(f"  Transition: pred_σ={trans_std:.4f}, epist={trans_epist:.6f}")
        print(f"  Steady:     pred_σ={steady_std:.4f}, epist={steady_epist:.6f}")
        print(f"  Ratio (trans/steady): pred={trans_std/steady_std:.2f}x, "
              f"epist={trans_epist/steady_epist:.2f}x")

    # --- Experiment 4: Per-bit Laplace probes ---
    print("\n[4] Per-bit Laplace probes...")
    results["per_bit_uncertainty"] = {}
    for b in range(4):
        bit_probe = LaplaceProbe(alpha=1.0)
        bit_probe.fit(h_pca, bits[:, b].astype(float))
        _, bit_std = bit_probe.predict(h_pca)

        # Uncertainty by bit value
        mask_0 = bits[:, b] == 0
        mask_1 = bits[:, b] == 1
        results["per_bit_uncertainty"][f"bit{b}"] = {
            "mean_std_at_0": float(bit_std[mask_0].mean()),
            "mean_std_at_1": float(bit_std[mask_1].mean()),
            "overall_mean_std": float(bit_std.mean()),
            "residual_sigma2": float(bit_probe.sigma2),
        }
        print(f"  Bit {b}: σ(at 0)={bit_std[mask_0].mean():.4f}, "
              f"σ(at 1)={bit_std[mask_1].mean():.4f}, "
              f"σ²_resid={bit_probe.sigma2:.6f}")

    # --- Experiment 5: Temporal uncertainty profile within episodes ---
    print("\n[5] Temporal uncertainty profile...")
    ep_size = len(h_t) // 15
    # Average uncertainty across episodes by step position
    max_steps = ep_size
    step_uncertainty = np.zeros(max_steps)
    step_counts_arr = np.zeros(max_steps)

    for ep in range(15):
        start = ep * ep_size
        end = min(start + ep_size, len(h_t))
        ep_len = end - start
        _, ep_std = probe.predict(h_pca[start:end])
        step_uncertainty[:ep_len] += ep_std[:ep_len] if ep_len <= max_steps else ep_std[:max_steps]
        step_counts_arr[:ep_len] += 1

    valid_steps = step_counts_arr > 0
    step_uncertainty[valid_steps] /= step_counts_arr[valid_steps]

    # Report first 100 steps
    n_report = min(100, int(valid_steps.sum()))
    results["temporal_profile"] = {
        "step_mean_uncertainty": step_uncertainty[:n_report].tolist(),
        "n_steps_reported": n_report,
        "peak_step": int(np.argmax(step_uncertainty[:n_report])),
        "peak_uncertainty": float(step_uncertainty[:n_report].max()),
        "trough_step": int(np.argmin(step_uncertainty[:n_report])),
        "trough_uncertainty": float(step_uncertainty[:n_report].min()),
    }
    print(f"  Peak uncertainty at step {results['temporal_profile']['peak_step']}: "
          f"{results['temporal_profile']['peak_uncertainty']:.4f}")
    print(f"  Trough uncertainty at step {results['temporal_profile']['trough_step']}: "
          f"{results['temporal_profile']['trough_uncertainty']:.4f}")

    # --- Experiment 6: Calibration check ---
    print("\n[6] Calibration check...")
    # For a well-calibrated model, ~68% of true values should fall within ±1σ
    # and ~95% within ±2σ
    residuals_abs = np.abs(counts - y_pred)
    within_1sigma = float(np.mean(residuals_abs < y_std))
    within_2sigma = float(np.mean(residuals_abs < 2 * y_std))
    within_3sigma = float(np.mean(residuals_abs < 3 * y_std))

    results["calibration"] = {
        "within_1sigma": within_1sigma,
        "within_2sigma": within_2sigma,
        "within_3sigma": within_3sigma,
        "expected_1sigma": 0.683,
        "expected_2sigma": 0.954,
        "expected_3sigma": 0.997,
        "overconfident": within_1sigma < 0.6,
        "underconfident": within_1sigma > 0.8,
    }
    print(f"  Within ±1σ: {within_1sigma:.3f} (expected 0.683)")
    print(f"  Within ±2σ: {within_2sigma:.3f} (expected 0.954)")
    print(f"  Within ±3σ: {within_3sigma:.3f} (expected 0.997)")

    # Save
    out_path = OUT_DIR / "laplace_uncertainty.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
