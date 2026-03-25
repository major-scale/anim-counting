#!/usr/bin/env python3
"""
Conformal Prediction for Observation Cliff Detection
=====================================================
Uses split conformal prediction to build distribution-free prediction
intervals for the count probe. Tests whether widening prediction sets
can detect the observation cliff before catastrophic failure.

Key questions:
1. Do conformal prediction sets widen at carry transitions?
2. Can conformal intervals detect when the model is off-manifold?
3. What coverage guarantee holds for imagination vs posterior?
4. Can we build a practical cliff detector from conformal widths?

Approach:
- Split conformal: calibrate on held-out episodes, test on others
- Track prediction set width across time
- Simulate cliff by blanking observations at different points
- Test if conformal width provides early warning

Output: artifacts/binary_successor/conformal_cliff.json
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


def carry_depth(count):
    """Compute carry cascade depth for count→count+1 transition."""
    if count >= 14:
        return -1
    next_count = count + 1
    depth = 0
    xor = count ^ next_count
    while xor > 0:
        depth += 1
        xor >>= 1
    return depth - 1


class ConformalCountPredictor:
    """Split conformal prediction for count regression.

    Uses residual-based conformal prediction:
    1. Fit Ridge probe on training set
    2. Compute nonconformity scores (absolute residuals) on calibration set
    3. Prediction set for new x: [ŷ(x) - q, ŷ(x) + q]
       where q is the (1-α)(1+1/n_cal) quantile of calibration scores
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha  # 1 - coverage
        self.probe = Ridge(alpha=1.0)
        self.quantile = None
        self.cal_scores = None

    def fit_calibrate(self, X_train, y_train, X_cal, y_cal):
        """Fit probe on training data, calibrate on calibration data."""
        self.probe.fit(X_train, y_train)

        # Nonconformity scores on calibration set
        y_cal_pred = self.probe.predict(X_cal)
        self.cal_scores = np.abs(y_cal - y_cal_pred)

        # Quantile for desired coverage
        n_cal = len(self.cal_scores)
        q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        self.quantile = float(np.quantile(self.cal_scores, q_level))

        return self

    def predict_set(self, X):
        """Return prediction sets: (lower, upper, width, point_pred)."""
        y_pred = self.probe.predict(X)
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        width = np.full_like(y_pred, 2 * self.quantile)
        return lower, upper, width, y_pred

    def adaptive_predict_set(self, X, local_scores=None):
        """Locally-weighted conformal prediction.

        If we have local nonconformity scores (from nearby calibration
        points), use those for tighter/wider intervals.
        """
        if local_scores is None:
            return self.predict_set(X)

        y_pred = self.probe.predict(X)
        n = len(local_scores)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        local_q = np.quantile(local_scores, q_level)

        lower = y_pred - local_q
        upper = y_pred + local_q
        width = np.full_like(y_pred, 2 * local_q)
        return lower, upper, width, y_pred


def main():
    print("Loading battery data...")
    data = np.load(BATTERY_PATH, allow_pickle=True)
    h_t = data['h_t']      # (13280, 512)
    counts = data['counts']  # (13280,)
    bits = data['bits']      # (13280, 4)

    print(f"  h_t: {h_t.shape}, counts: {counts.shape}")

    # PCA for efficiency
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    h_pca = pca.fit_transform(h_t)
    print(f"  PCA to 50 dims: {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    # Episode split
    ep_size = len(h_t) // 15
    results = {}

    # --- Experiment 1: Basic conformal prediction ---
    print("\n[1] Split conformal prediction (90% coverage)...")

    # Split: 8 episodes train, 3 calibrate, 4 test
    train_end = 8 * ep_size
    cal_end = 11 * ep_size

    X_train = h_pca[:train_end]
    y_train = counts[:train_end]
    X_cal = h_pca[train_end:cal_end]
    y_cal = counts[train_end:cal_end]
    X_test = h_pca[cal_end:]
    y_test = counts[cal_end:]

    cp = ConformalCountPredictor(alpha=0.1)
    cp.fit_calibrate(X_train, y_train, X_cal, y_cal)

    lower, upper, width, y_pred = cp.predict_set(X_test)

    # Coverage
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mean_width = float(width.mean())

    results["basic_conformal"] = {
        "coverage": coverage,
        "target_coverage": 0.90,
        "mean_width": mean_width,
        "quantile": cp.quantile,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
    }
    print(f"  Coverage: {coverage:.3f} (target: 0.90)")
    print(f"  Mean width: {mean_width:.3f}")
    print(f"  Quantile (q): {cp.quantile:.3f}")

    # --- Experiment 2: Per-count conformal width ---
    print("\n[2] Per-count conformal analysis...")
    results["per_count_conformal"] = {}

    for c in range(15):
        mask = y_test == c
        if mask.sum() == 0:
            continue
        local_coverage = float(np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask])))
        local_accuracy = float(np.mean(np.round(y_pred[mask]) == c))
        depth = carry_depth(c)

        # Adaptive conformal: use calibration scores from same count
        cal_mask = y_cal == c
        if cal_mask.sum() > 5:
            cal_scores_local = cp.cal_scores[cal_mask]  # Already computed
            # Use actual predictions on cal set for this count
            y_cal_pred = cp.probe.predict(X_cal)
            cal_residuals_local = np.abs(y_cal[cal_mask] - y_cal_pred[cal_mask])
            n = len(cal_residuals_local)
            q_level = min(np.ceil(0.9 * (n + 1)) / n, 1.0)
            local_q = float(np.quantile(cal_residuals_local, q_level))
        else:
            local_q = cp.quantile

        results["per_count_conformal"][str(c)] = {
            "count": c,
            "carry_depth": depth,
            "coverage": local_coverage,
            "accuracy": local_accuracy,
            "adaptive_width": 2 * local_q,
            "n_test": int(mask.sum()),
        }
        print(f"  Count {c:>2} (d={depth}): coverage={local_coverage:.3f}, "
              f"acc={local_accuracy:.3f}, adaptive_w={2*local_q:.3f}")

    # --- Experiment 3: Conformal width vs carry depth ---
    print("\n[3] Conformal width vs carry depth...")
    test_depths = np.array([carry_depth(c) for c in y_test])
    valid = test_depths >= 0

    # Use prediction residuals as proxy for conformal width at test time
    test_residuals = np.abs(y_test - y_pred)

    r_depth, p_depth = spearmanr(test_depths[valid], test_residuals[valid])
    results["depth_correlation"] = {
        "residual_vs_depth_r": float(r_depth),
        "residual_vs_depth_p": float(p_depth),
    }
    print(f"  Residual vs depth: r={r_depth:.3f} (p={p_depth:.1e})")

    for d in sorted(np.unique(test_depths[valid])):
        d_mask = test_depths == d
        mean_res = float(test_residuals[d_mask].mean())
        print(f"    Depth {int(d)}: mean |residual|={mean_res:.4f}")
        results["depth_correlation"][f"depth_{int(d)}"] = {
            "mean_residual": mean_res,
            "n": int(d_mask.sum()),
        }

    # --- Experiment 4: Cliff detection via conformal width ---
    print("\n[4] Cliff detection simulation...")

    # Simulate observation blanking at different episode positions
    # Re-run ESN-like dynamics: use the actual h_t but add growing noise
    # to simulate drift from observation loss
    rng = np.random.RandomState(42)

    cliff_results = {}
    for noise_level in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        h_noisy = h_pca.copy()
        # Add noise to test episodes only
        h_noisy[cal_end:] += rng.randn(len(h_noisy[cal_end:]), h_noisy.shape[1]) * noise_level

        lower_n, upper_n, width_n, y_pred_n = cp.predict_set(h_noisy[cal_end:])
        coverage_n = float(np.mean((y_test >= lower_n) & (y_test <= upper_n)))
        accuracy_n = float(np.mean(np.round(y_pred_n) == y_test))

        # Key metric: fraction of prediction sets that are "suspicious" (residual > quantile)
        residuals_n = np.abs(y_test - y_pred_n)
        suspicious = float(np.mean(residuals_n > cp.quantile))

        cliff_results[f"noise_{noise_level}"] = {
            "noise_level": noise_level,
            "coverage": coverage_n,
            "accuracy": accuracy_n,
            "mean_residual": float(residuals_n.mean()),
            "suspicious_fraction": suspicious,
        }
        print(f"  Noise={noise_level:.1f}: coverage={coverage_n:.3f}, "
              f"acc={accuracy_n:.3f}, suspicious={suspicious:.3f}")

    results["cliff_detection"] = cliff_results

    # --- Experiment 5: Temporal conformal width within episodes ---
    print("\n[5] Temporal conformal width profile...")
    # For each position within an episode, compute mean residual
    max_steps = min(ep_size, 100)
    step_residuals = np.zeros(max_steps)
    step_n = np.zeros(max_steps)

    for ep in range(15):
        start = ep * ep_size
        end = min(start + ep_size, len(h_t))
        ep_len = min(end - start, max_steps)

        ep_pred = cp.probe.predict(h_pca[start:start + ep_len])
        ep_res = np.abs(counts[start:start + ep_len] - ep_pred)
        step_residuals[:ep_len] += ep_res
        step_n[:ep_len] += 1

    valid_steps = step_n > 0
    step_residuals[valid_steps] /= step_n[valid_steps]

    # Find periodic peaks (should align with carry transitions)
    results["temporal_profile"] = {
        "mean_residuals_by_step": step_residuals[:max_steps].tolist(),
        "n_steps": max_steps,
    }

    # Identify which steps have highest residuals
    top_steps = np.argsort(step_residuals[:max_steps])[::-1][:10]
    print(f"  Top-10 uncertain steps: {top_steps.tolist()}")
    print(f"  Their counts (mod 15): {[int(s % 15) for s in top_steps]}")
    print(f"  Their carry depths: {[carry_depth(int(s % 15)) for s in top_steps]}")

    results["temporal_profile"]["top_uncertain_steps"] = top_steps.tolist()
    results["temporal_profile"]["top_step_counts"] = [int(s % 15) for s in top_steps]
    results["temporal_profile"]["top_step_depths"] = [carry_depth(int(s % 15)) for s in top_steps]

    # --- Experiment 6: Coverage at multiple confidence levels ---
    print("\n[6] Coverage at multiple confidence levels...")
    results["multi_alpha"] = {}
    for alpha in [0.01, 0.05, 0.10, 0.20, 0.50]:
        cp_a = ConformalCountPredictor(alpha=alpha)
        cp_a.fit_calibrate(X_train, y_train, X_cal, y_cal)
        l, u, w, p = cp_a.predict_set(X_test)
        cov = float(np.mean((y_test >= l) & (y_test <= u)))
        results["multi_alpha"][f"alpha_{alpha}"] = {
            "target": 1 - alpha,
            "achieved": cov,
            "width": float(w.mean()),
            "quantile": cp_a.quantile,
        }
        print(f"  α={alpha:.2f}: target={1-alpha:.2f}, achieved={cov:.3f}, width={w.mean():.3f}")

    # Save
    out_path = OUT_DIR / "conformal_cliff.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
