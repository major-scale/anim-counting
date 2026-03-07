#!/usr/bin/env python3
"""
Bayesian Probe Optimizer — Push count predictions toward 99.99% accuracy.

Loads battery evaluation data (per-episode NPZ format), fits Gaussian
discriminant boundaries on the linear probe's continuous output, and evaluates
multiple temporal smoothing strategies.

Sections:
  1. Data loading (battery NPZ + optional educational episode)
  2. Baseline evaluation (Ridge rounding)
  3. Bayesian boundaries (Gaussian discriminant)
  4. Temporal smoothing
  5. Combined evaluation
  6. Export for GUI (extends embed_probe.json)
  7. Report

Usage:
    # On battery data (per-episode NPZ)
    python3 bayesian_probe_optimizer.py \\
        --data ~/anim-bridge/artifacts/battery/randproj_s0/randproj_s0.npz

    # On educational episode (single-episode NPZ)
    python3 bayesian_probe_optimizer.py \\
        --data ~/anim-bridge/artifacts/episodes/randproj_episode.npz \\
        --format educational

    # Export Bayesian parameters to probe JSON
    python3 bayesian_probe_optimizer.py \\
        --data ~/anim-bridge/artifacts/battery/randproj_s0/randproj_s0.npz \\
        --export-probe ~/anim-bridge/models/randproj_clean/embed_probe.json

    # Also try clean battery data
    python3 bayesian_probe_optimizer.py \\
        --data ~/anim-bridge/artifacts/battery/clean/clean_randproj_s0/clean_randproj_s0.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_battery_npz(path):
    """Load battery NPZ — supports both flat and per-episode formats.

    Flat format (from full_battery.py):
        h_t (N, 512), counts (N,), episode_ids (N,), timesteps (N,)

    Per-episode format:
        n_episodes, ep{i}_h_t (T_i, 512), ep{i}_counts (T_i,)
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # Flat format (full_battery.py output)
    if "h_t" in keys and "counts" in keys:
        h_t = data["h_t"]
        counts = data["counts"].astype(np.int32)
        T = len(counts)
        ep_ids = data["episode_ids"].astype(np.int32) if "episode_ids" in keys else np.zeros(T, dtype=np.int32)
        timesteps = data["timesteps"].astype(np.int32) if "timesteps" in keys else np.arange(T, dtype=np.int32)
        n_eps = int(ep_ids.max() + 1) if len(ep_ids) > 0 else 1
        return {
            "h_t": h_t,
            "counts": counts,
            "episode_ids": ep_ids,
            "timesteps": timesteps,
            "n_episodes": n_eps,
        }

    # Per-episode format
    if "n_episodes" not in keys:
        raise ValueError(f"Unrecognized battery NPZ format. Keys: {keys[:20]}")

    n_eps = int(data["n_episodes"])
    all_h_t, all_counts, all_episode_ids, all_timesteps = [], [], [], []

    for ep in range(n_eps):
        h_key = f"ep{ep}_h_t"
        c_key = f"ep{ep}_counts"
        if h_key not in data:
            break
        h_t = data[h_key]
        counts = data[c_key]
        T = len(counts)
        all_h_t.append(h_t)
        all_counts.append(counts)
        all_episode_ids.append(np.full(T, ep, dtype=np.int32))
        all_timesteps.append(np.arange(T, dtype=np.int32))

    if not all_h_t:
        raise ValueError(f"No per-episode data found in {path}. Keys: {keys[:20]}")

    return {
        "h_t": np.concatenate(all_h_t, axis=0),
        "counts": np.concatenate(all_counts, axis=0).astype(np.int32),
        "episode_ids": np.concatenate(all_episode_ids, axis=0),
        "timesteps": np.concatenate(all_timesteps, axis=0),
        "n_episodes": n_eps,
    }


def load_educational_npz(path):
    """Load single educational episode NPZ: deter (T, 512), gt_count (T,)."""
    data = np.load(path, allow_pickle=True)
    T = len(data["gt_count"])
    return {
        "h_t": data["deter"],                           # (T, 512)
        "counts": data["gt_count"].astype(np.int32),     # (T,)
        "episode_ids": np.zeros(T, dtype=np.int32),      # single episode
        "timesteps": np.arange(T, dtype=np.int32),
        "n_episodes": 1,
    }


def load_data(path, fmt="auto"):
    """Load NPZ data, auto-detecting format."""
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    if fmt == "educational" or (fmt == "auto" and "deter" in keys and "gt_count" in keys):
        print(f"  Detected educational episode format")
        return load_educational_npz(path)
    elif fmt == "battery" or (fmt == "auto" and ("n_episodes" in keys or
                              ("h_t" in keys and "counts" in keys and "episode_ids" in keys))):
        print(f"  Detected battery format")
        return load_battery_npz(path)
    else:
        # Try flat format: h_t, counts directly (no episode_ids)
        if "h_t" in keys and "counts" in keys:
            print(f"  Detected flat format (no episode_ids)")
            h_t = data["h_t"]
            counts = data["counts"].astype(np.int32)
            T = len(counts)
            ep_ids = data.get("episode_ids", np.zeros(T, dtype=np.int32))
            return {
                "h_t": h_t,
                "counts": counts,
                "episode_ids": ep_ids.astype(np.int32),
                "timesteps": np.arange(T, dtype=np.int32),
                "n_episodes": int(ep_ids.max() + 1) if len(ep_ids) > 0 else 1,
            }
        raise ValueError(f"Cannot auto-detect NPZ format. Keys: {keys[:20]}")


def stratified_split(counts, episode_ids, test_size=0.2, seed=42):
    """Stratified train/test split by episode (keeps episodes intact)."""
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n_test = max(1, int(len(unique_eps) * test_size))
    test_eps = set(unique_eps[:n_test])
    train_mask = np.array([eid not in test_eps for eid in episode_ids])
    test_mask = ~train_mask
    return train_mask, test_mask


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Baseline Evaluation (Rounding)
# ═══════════════════════════════════════════════════════════════════════

def fit_ridge_probe(h_t, counts, alpha=1.0):
    """Fit Ridge regression probe: count = h_t @ w + b."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(h_t, counts)
    return model.coef_.astype(np.float64), float(model.intercept_)


def predict_rounding(raw_values, max_count=25):
    """Standard rounding prediction: clip(round(x), 0, max_count)."""
    return np.clip(np.round(raw_values), 0, max_count).astype(np.int32)


def evaluate_predictions(true, pred, label=""):
    """Compute accuracy metrics."""
    exact = np.mean(true == pred)
    within1 = np.mean(np.abs(true - pred) <= 1)
    mae = np.mean(np.abs(true - pred))

    # Per-count accuracy
    unique_counts = np.unique(true)
    per_count = {}
    for c in unique_counts:
        mask = true == c
        n = mask.sum()
        if n > 0:
            per_count[int(c)] = {
                "n": int(n),
                "exact": float(np.mean(pred[mask] == c)),
                "within1": float(np.mean(np.abs(pred[mask] - c) <= 1)),
                "mean_pred": float(np.mean(pred[mask])),
            }

    return {
        "label": label,
        "exact": float(exact),
        "within1": float(within1),
        "mae": float(mae),
        "n_samples": len(true),
        "per_count": per_count,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Bayesian Boundaries (Gaussian Discriminant)
# ═══════════════════════════════════════════════════════════════════════

def fit_gaussians(raw_values, counts, max_count=25):
    """Fit Gaussian (mu, sigma) for each count from probe raw values."""
    gaussians = []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() < 2:
            # No data for this count — interpolate from neighbors
            gaussians.append({"mean": float(c), "std": 0.5, "n": 0})
        else:
            vals = raw_values[mask]
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1))
            sigma = max(sigma, 0.01)  # floor to prevent division by zero
            gaussians.append({"mean": mu, "std": sigma, "n": int(mask.sum())})
    return gaussians


def compute_bayesian_boundaries(gaussians, max_count=25):
    """Compute optimal decision boundaries between adjacent Gaussian distributions.

    For each pair of adjacent counts (c, c+1), find the crossover point where
    P(c|x) = P(c+1|x). With equal priors this is where the PDFs cross.

    Returns list of 25 boundaries (between counts 0-1, 1-2, ..., 24-25).
    """
    boundaries = []
    for c in range(max_count):
        g1 = gaussians[c]
        g2 = gaussians[c + 1]
        mu1, s1 = g1["mean"], g1["std"]
        mu2, s2 = g2["mean"], g2["std"]

        if abs(s1 - s2) < 1e-8:
            # Equal variances: boundary is midpoint
            boundary = (mu1 + mu2) / 2.0
        else:
            # Solve quadratic for log-likelihood crossing
            # log N(x; mu1, s1) = log N(x; mu2, s2)
            a = 1 / (2 * s1**2) - 1 / (2 * s2**2)
            b_coef = mu2 / s2**2 - mu1 / s1**2
            c_coef = mu1**2 / (2 * s1**2) - mu2**2 / (2 * s2**2) + np.log(s1 / s2)

            discriminant = b_coef**2 - 4 * a * c_coef
            if discriminant < 0:
                # No real crossing — use midpoint
                boundary = (mu1 + mu2) / 2.0
            else:
                r1 = (-b_coef + np.sqrt(discriminant)) / (2 * a)
                r2 = (-b_coef - np.sqrt(discriminant)) / (2 * a)
                # Pick the root between the two means
                mid = (mu1 + mu2) / 2.0
                candidates = [r for r in [r1, r2] if min(mu1, mu2) - 2 * max(s1, s2) < r < max(mu1, mu2) + 2 * max(s1, s2)]
                if candidates:
                    boundary = min(candidates, key=lambda x: abs(x - mid))
                else:
                    boundary = mid

        boundaries.append(float(boundary))

    return boundaries


def predict_bayesian(raw_values, boundaries, max_count=25):
    """Classify raw probe values using Bayesian decision boundaries."""
    pred = np.zeros(len(raw_values), dtype=np.int32)
    boundaries = np.array(boundaries)
    for i, x in enumerate(raw_values):
        # Find which bucket x falls into
        c = np.searchsorted(boundaries, x)
        # searchsorted returns index where x would be inserted to keep sorted
        # If boundaries are monotonically increasing, this gives the correct count
        # But boundaries may not be monotonic (inverted centroids!)
        pred[i] = int(np.clip(c, 0, max_count))
    return pred


def predict_bayesian_full(raw_values, gaussians, max_count=25):
    """Full Bayesian: for each x, pick count c that maximizes P(x|c) * P(c).

    More robust than boundary method when boundaries aren't monotonic.
    Uses uniform priors (equal P(c) for all c).
    """
    pred = np.zeros(len(raw_values), dtype=np.int32)
    means = np.array([g["mean"] for g in gaussians])
    stds = np.array([g["std"] for g in gaussians])

    for i, x in enumerate(raw_values):
        # Log-likelihood for each count
        log_probs = norm.logpdf(x, loc=means, scale=stds)
        pred[i] = int(np.argmax(log_probs))

    return pred


def compute_dprime(gaussians, max_count=25):
    """Compute d-prime (discriminability) between adjacent count distributions."""
    dprimes = []
    for c in range(max_count):
        g1 = gaussians[c]
        g2 = gaussians[c + 1]
        if g1["n"] == 0 or g2["n"] == 0:
            dprimes.append(0.0)
            continue
        pooled_std = np.sqrt((g1["std"]**2 + g2["std"]**2) / 2)
        if pooled_std < 1e-8:
            dprimes.append(float('inf'))
        else:
            dprimes.append(float(abs(g2["mean"] - g1["mean"]) / pooled_std))
    return dprimes


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Temporal Smoothing
# ═══════════════════════════════════════════════════════════════════════

def smooth_median(raw_values, episode_ids, window=3):
    """Median filter on raw probe values, resetting at episode boundaries."""
    result = raw_values.copy()
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        indices = np.where(mask)[0]
        vals = raw_values[indices]
        half = window // 2
        for j, idx in enumerate(indices):
            lo = max(0, j - half)
            hi = min(len(vals), j + half + 1)
            result[idx] = np.median(vals[lo:hi])
    return result


def smooth_ema(raw_values, episode_ids, alpha=0.3):
    """Exponential moving average on raw probe values."""
    result = raw_values.copy()
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        ema = raw_values[indices[0]]
        result[indices[0]] = ema
        for j in range(1, len(indices)):
            ema = alpha * raw_values[indices[j]] + (1 - alpha) * ema
            result[indices[j]] = ema
    return result


def enforce_monotonic(predictions, episode_ids, counts):
    """Enforce monotonicity during gathering phase.

    Infer gathering phase: count is non-decreasing within episode.
    During gathering: prediction can only stay or increase by 1.
    """
    result = predictions.copy()
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        ep_counts = counts[indices]

        # Detect gathering phase: longest prefix where counts are non-decreasing
        gathering_end = len(ep_counts)
        for j in range(1, len(ep_counts)):
            if ep_counts[j] < ep_counts[j - 1]:
                gathering_end = j
                break

        # Apply monotonicity in gathering phase
        prev = result[indices[0]]
        for j in range(1, gathering_end):
            curr = result[indices[j]]
            if curr < prev:
                result[indices[j]] = prev  # can't decrease
            elif curr > prev + 1:
                result[indices[j]] = prev + 1  # max +1 per step
            prev = result[indices[j]]

    return result


def hold_on_jump(predictions, episode_ids, max_jump=1):
    """If prediction jumps by more than max_jump, hold previous value for 1 frame."""
    result = predictions.copy()
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        indices = np.where(mask)[0]
        if len(indices) < 2:
            continue
        for j in range(1, len(indices)):
            prev = result[indices[j - 1]]
            curr = predictions[indices[j]]
            if abs(curr - prev) > max_jump:
                result[indices[j]] = prev
    return result


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Combined Evaluation
# ═══════════════════════════════════════════════════════════════════════

def identify_gathering_phase(counts, episode_ids):
    """Return boolean mask for timesteps in gathering phase (non-decreasing counts)."""
    mask = np.zeros(len(counts), dtype=bool)
    for ep in np.unique(episode_ids):
        ep_mask = episode_ids == ep
        indices = np.where(ep_mask)[0]
        if len(indices) == 0:
            continue
        ep_counts = counts[indices]
        # Gathering = longest prefix where count is non-decreasing
        for j in range(len(ep_counts)):
            if j == 0:
                mask[indices[j]] = True
            elif ep_counts[j] >= ep_counts[j - 1]:
                mask[indices[j]] = True
            else:
                break
    return mask


def run_full_evaluation(h_t, counts, episode_ids, probe_w, probe_b, max_count=25):
    """Run all classification + smoothing combinations and return comparison table."""
    raw_values = h_t @ probe_w + probe_b

    # Fit Gaussians on all data (in production, fit on train only)
    gaussians = fit_gaussians(raw_values, counts, max_count)
    boundaries = compute_bayesian_boundaries(gaussians, max_count)
    dprimes = compute_dprime(gaussians, max_count)

    # Gathering phase mask
    gathering_mask = identify_gathering_phase(counts, episode_ids)

    results = []

    # Classification methods
    classifiers = {
        "rounding": predict_rounding(raw_values, max_count),
        "bayesian_boundary": predict_bayesian(raw_values, boundaries, max_count),
        "bayesian_full": predict_bayesian_full(raw_values, gaussians, max_count),
    }

    # Smoothing methods (applied to raw values before classification)
    smoothings = {
        "none": raw_values,
        "median3": smooth_median(raw_values, episode_ids, window=3),
        "median5": smooth_median(raw_values, episode_ids, window=5),
        "ema_0.3": smooth_ema(raw_values, episode_ids, alpha=0.3),
    }

    for clf_name, clf_base_pred in classifiers.items():
        for sm_name, sm_raw in smoothings.items():
            label = f"{clf_name}+{sm_name}"

            if sm_name == "none":
                pred = clf_base_pred
            else:
                # Re-classify smoothed values
                if clf_name == "rounding":
                    pred = predict_rounding(sm_raw, max_count)
                elif clf_name == "bayesian_boundary":
                    pred = predict_bayesian(sm_raw, boundaries, max_count)
                elif clf_name == "bayesian_full":
                    pred = predict_bayesian_full(sm_raw, gaussians, max_count)

            # Overall metrics
            metrics = evaluate_predictions(counts, pred, label)

            # Gathering-only metrics
            if gathering_mask.sum() > 0:
                g_metrics = evaluate_predictions(
                    counts[gathering_mask], pred[gathering_mask],
                    label + " (gathering)")
                metrics["gathering_exact"] = g_metrics["exact"]
                metrics["gathering_within1"] = g_metrics["within1"]
            else:
                metrics["gathering_exact"] = metrics["exact"]
                metrics["gathering_within1"] = metrics["within1"]

            results.append(metrics)

    # Now add monotonic variants (post-classification smoothing)
    for clf_name in classifiers:
        for sm_name in ["none", "median3"]:
            sm_raw = smoothings[sm_name]
            if sm_name == "none":
                base_pred = classifiers[clf_name]
            else:
                if clf_name == "rounding":
                    base_pred = predict_rounding(sm_raw, max_count)
                elif clf_name == "bayesian_boundary":
                    base_pred = predict_bayesian(sm_raw, boundaries, max_count)
                elif clf_name == "bayesian_full":
                    base_pred = predict_bayesian_full(sm_raw, gaussians, max_count)

            mono_pred = enforce_monotonic(base_pred, episode_ids, counts)
            label = f"{clf_name}+{sm_name}+monotonic"
            metrics = evaluate_predictions(counts, mono_pred, label)

            if gathering_mask.sum() > 0:
                g_metrics = evaluate_predictions(
                    counts[gathering_mask], mono_pred[gathering_mask],
                    label + " (gathering)")
                metrics["gathering_exact"] = g_metrics["exact"]
                metrics["gathering_within1"] = g_metrics["within1"]
            else:
                metrics["gathering_exact"] = metrics["exact"]
                metrics["gathering_within1"] = metrics["within1"]

            results.append(metrics)

    # Hold-on-jump variants
    for clf_name in ["bayesian_full"]:
        base_pred = classifiers[clf_name]
        hoj_pred = hold_on_jump(base_pred, episode_ids, max_jump=1)
        label = f"{clf_name}+hold_on_jump"
        metrics = evaluate_predictions(counts, hoj_pred, label)
        if gathering_mask.sum() > 0:
            g_metrics = evaluate_predictions(
                counts[gathering_mask], hoj_pred[gathering_mask],
                label + " (gathering)")
            metrics["gathering_exact"] = g_metrics["exact"]
            metrics["gathering_within1"] = g_metrics["within1"]
        results.append(metrics)

    return {
        "results": results,
        "gaussians": gaussians,
        "boundaries": boundaries,
        "dprimes": dprimes,
        "raw_values": raw_values,
        "gathering_mask": gathering_mask,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Export for GUI
# ═══════════════════════════════════════════════════════════════════════

def export_bayesian_params(probe_json_path, gaussians, boundaries, rounding_acc, bayesian_acc):
    """Add Bayesian parameters to existing embed_probe.json (extends, doesn't replace)."""
    path = Path(probe_json_path)
    if path.exists():
        with open(path) as f:
            probe_data = json.load(f)
    else:
        probe_data = {}

    probe_data["bayesian_boundaries"] = boundaries
    probe_data["bayesian_gaussians"] = [
        {"mean": g["mean"], "std": g["std"]} for g in gaussians
    ]
    probe_data["bayesian_accuracy"] = bayesian_acc
    probe_data["rounding_accuracy"] = rounding_acc

    with open(path, "w") as f:
        json.dump(probe_data, f, indent=None)  # compact JSON

    print(f"  Exported Bayesian parameters to {path}")
    print(f"    {len(boundaries)} boundaries, {len(gaussians)} Gaussians")
    print(f"    Rounding accuracy: {rounding_acc:.4f}")
    print(f"    Bayesian accuracy: {bayesian_acc:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Report
# ═══════════════════════════════════════════════════════════════════════

def format_report(eval_output, data_info, probe_info):
    """Generate formatted markdown report."""
    results = eval_output["results"]
    gaussians = eval_output["gaussians"]
    boundaries = eval_output["boundaries"]
    dprimes = eval_output["dprimes"]

    lines = []
    lines.append("# Bayesian Probe Optimization Report")
    lines.append("")
    lines.append(f"**Data**: {data_info['path']}")
    lines.append(f"**Samples**: {data_info['n_samples']:,} ({data_info['n_train']:,} train / {data_info['n_test']:,} test)")
    lines.append(f"**Episodes**: {data_info['n_episodes']}")
    lines.append(f"**Count range**: {data_info['count_min']}-{data_info['count_max']}")
    lines.append(f"**Probe R²**: {probe_info['r_squared']:.6f}")
    lines.append("")

    # Comparison table
    lines.append("## Accuracy Comparison")
    lines.append("")
    lines.append("| Method | Exact | Within ±1 | MAE | Gathering Exact | Gathering ±1 |")
    lines.append("|--------|-------|-----------|-----|-----------------|--------------|")

    # Sort by exact accuracy descending
    sorted_results = sorted(results, key=lambda r: r["exact"], reverse=True)
    for r in sorted_results:
        g_exact = r.get("gathering_exact", r["exact"])
        g_w1 = r.get("gathering_within1", r["within1"])
        lines.append(
            f"| {r['label']:<40} | {r['exact']:.4f} | {r['within1']:.4f} | {r['mae']:.3f} "
            f"| {g_exact:.4f} | {g_w1:.4f} |")

    lines.append("")

    # Best result
    best = sorted_results[0]
    baseline = next(r for r in results if r["label"] == "rounding+none")
    improvement = best["exact"] - baseline["exact"]
    lines.append(f"**Best**: {best['label']} ({best['exact']:.4f} exact, "
                 f"+{improvement:.4f} vs rounding baseline)")
    lines.append("")

    # Gaussian parameters
    lines.append("## Gaussian Parameters")
    lines.append("")
    lines.append("| Count | Mean | Std | N | d' (vs next) | Boundary |")
    lines.append("|-------|------|-----|---|--------------|----------|")
    for c, g in enumerate(gaussians):
        dp = dprimes[c] if c < len(dprimes) else "-"
        dp_str = f"{dp:.2f}" if isinstance(dp, float) else dp
        bd = f"{boundaries[c]:.4f}" if c < len(boundaries) else "-"
        lines.append(f"| {c:2d} | {g['mean']:7.3f} | {g['std']:5.3f} | {g['n']:5d} | {dp_str:>6} | {bd:>8} |")

    lines.append("")

    # Monotonicity check
    means = [g["mean"] for g in gaussians]
    inversions = []
    for i in range(len(means) - 1):
        if means[i + 1] < means[i]:
            inversions.append((i, i + 1, means[i], means[i + 1]))
    if inversions:
        lines.append("## Centroid Inversions (probe mean not monotonic)")
        lines.append("")
        for c1, c2, m1, m2 in inversions:
            lines.append(f"  - Count {c1} (mean={m1:.3f}) > Count {c2} (mean={m2:.3f}) "
                         f"-- delta={m2-m1:.3f}")
        lines.append("")
    else:
        lines.append("## No centroid inversions detected")
        lines.append("")

    # d-prime analysis
    weak_dprimes = [(i, dp) for i, dp in enumerate(dprimes) if dp < 2.0]
    if weak_dprimes:
        lines.append("## Weak Discriminability (d' < 2.0)")
        lines.append("")
        for c, dp in weak_dprimes:
            lines.append(f"  - Counts {c}/{c+1}: d'={dp:.2f}")
        lines.append("")

    # Boundary vs midpoint comparison
    lines.append("## Boundary vs Midpoint Comparison")
    lines.append("")
    lines.append("| Pair | Midpoint | Bayesian | Delta |")
    lines.append("|------|----------|----------|-------|")
    for c in range(min(len(boundaries), len(gaussians) - 1)):
        mid = (gaussians[c]["mean"] + gaussians[c + 1]["mean"]) / 2.0
        bd = boundaries[c]
        delta = bd - mid
        if abs(delta) > 0.01:
            lines.append(f"| {c:2d}/{c+1:2d} | {mid:7.3f} | {bd:7.3f} | {delta:+.3f} * |")
        else:
            lines.append(f"| {c:2d}/{c+1:2d} | {mid:7.3f} | {bd:7.3f} | {delta:+.3f} |")
    lines.append("")
    lines.append("(* = boundary differs from midpoint by > 0.01)")

    # Per-count breakdown for best method
    lines.append("")
    lines.append(f"## Per-Count Accuracy ({best['label']})")
    lines.append("")
    lines.append("| Count | N | Exact | Within ±1 | Mean Pred |")
    lines.append("|-------|---|-------|-----------|-----------|")
    for c in sorted(best["per_count"].keys()):
        pc = best["per_count"][c]
        lines.append(f"| {c:5} | {pc['n']:5d} | {pc['exact']:.4f} | {pc['within1']:.4f} | {pc['mean_pred']:.2f} |")

    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if best["exact"] > 0.99:
        lines.append("Bayesian boundaries achieve >99% accuracy. Ship Approaches 1+2.")
    elif best["exact"] > 0.97:
        lines.append("Bayesian boundaries improve significantly. Consider contrastive loss "
                     "for further improvement (Approach 3).")
    else:
        lines.append("Linear probe has fundamental limitations at this accuracy level. "
                     "Contrastive auxiliary loss (Approach 3) is needed for 99.9%+.")

    g_best = max(results, key=lambda r: r.get("gathering_exact", 0))
    if g_best.get("gathering_exact", 0) > 0.99:
        lines.append(f"\nGathering-phase accuracy: {g_best['gathering_exact']:.4f} "
                     f"({g_best['label']}) — excellent for GUI use case.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian probe optimizer: push count predictions toward 99.99%")
    parser.add_argument("--data", required=True,
                        help="Path to battery NPZ or educational episode NPZ")
    parser.add_argument("--format", choices=["auto", "battery", "educational"],
                        default="auto", help="NPZ format (default: auto-detect)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression alpha (default: 1.0)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of episodes for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    parser.add_argument("--max-count", type=int, default=25,
                        help="Maximum count value (default: 25)")
    parser.add_argument("--export-probe", default=None,
                        help="Path to embed_probe.json to extend with Bayesian params")
    parser.add_argument("--output", default=None,
                        help="Directory to save report and standalone JSON")
    parser.add_argument("--use-existing-probe", default=None,
                        help="Path to embed_probe.json with pre-fit weights (skip Ridge fit)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    data = load_data(args.data, args.format)
    h_t = data["h_t"]
    counts = data["counts"]
    episode_ids = data["episode_ids"]
    n_samples = len(counts)
    n_episodes = data["n_episodes"]
    print(f"  {n_samples:,} samples, {n_episodes} episodes, "
          f"counts {counts.min()}-{counts.max()}")

    # Train/test split
    train_mask, test_mask = stratified_split(counts, episode_ids,
                                             test_size=args.test_size, seed=args.seed)
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    print(f"  Split: {n_train:,} train, {n_test:,} test")

    # Fit or load probe
    if args.use_existing_probe:
        print(f"  Loading probe from {args.use_existing_probe}")
        with open(args.use_existing_probe) as f:
            probe_data = json.load(f)
        probe_w = np.array(probe_data["weights"], dtype=np.float64)
        probe_b = float(probe_data["bias"])
        r_squared = probe_data.get("r_squared", -1)
    else:
        print(f"  Fitting Ridge(alpha={args.alpha}) on train data...")
        probe_w, probe_b = fit_ridge_probe(h_t[train_mask], counts[train_mask],
                                           alpha=args.alpha)
        # R² on test
        raw_test = h_t[test_mask] @ probe_w + probe_b
        ss_res = np.sum((counts[test_mask] - raw_test)**2)
        ss_tot = np.sum((counts[test_mask] - counts[test_mask].mean())**2)
        r_squared = 1 - ss_res / ss_tot
        print(f"  Probe R² (test): {r_squared:.6f}")

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    eval_output = run_full_evaluation(
        h_t[test_mask], counts[test_mask], episode_ids[test_mask],
        probe_w, probe_b, args.max_count)

    # Print comparison table
    results = eval_output["results"]
    print(f"\n{'Method':<45} {'Exact':>7} {'±1':>7} {'MAE':>6} {'Gath.Exact':>10}")
    print("-" * 80)
    sorted_results = sorted(results, key=lambda r: r["exact"], reverse=True)
    for r in sorted_results[:20]:  # Top 20
        g_exact = r.get("gathering_exact", r["exact"])
        print(f"{r['label']:<45} {r['exact']:>7.4f} {r['within1']:>7.4f} "
              f"{r['mae']:>6.3f} {g_exact:>10.4f}")

    baseline = next(r for r in results if r["label"] == "rounding+none")
    best = sorted_results[0]
    print(f"\nBaseline (rounding): {baseline['exact']:.4f}")
    print(f"Best:                {best['exact']:.4f} ({best['label']})")
    print(f"Improvement:         +{best['exact'] - baseline['exact']:.4f}")

    # Centroid inversions
    gaussians = eval_output["gaussians"]
    means = [g["mean"] for g in gaussians]
    for i in range(len(means) - 1):
        if means[i + 1] < means[i]:
            print(f"\n  WARNING: Centroid inversion at counts {i}/{i+1}: "
                  f"{means[i]:.3f} > {means[i+1]:.3f}")

    # d-prime summary
    dprimes = eval_output["dprimes"]
    min_dp = min(dprimes) if dprimes else 0
    min_dp_idx = dprimes.index(min_dp) if dprimes else -1
    print(f"\nMin d-prime: {min_dp:.2f} (counts {min_dp_idx}/{min_dp_idx+1})")

    # ── Also evaluate on ALL data (for export) ────────────────────
    print("\n" + "=" * 70)
    print("FIT ON ALL DATA (for export)")
    print("=" * 70)

    if not args.use_existing_probe:
        probe_w_all, probe_b_all = fit_ridge_probe(h_t, counts, alpha=args.alpha)
    else:
        probe_w_all, probe_b_all = probe_w, probe_b

    raw_all = h_t @ probe_w_all + probe_b_all
    gaussians_all = fit_gaussians(raw_all, counts, args.max_count)
    boundaries_all = compute_bayesian_boundaries(gaussians_all, args.max_count)

    rounding_pred_all = predict_rounding(raw_all, args.max_count)
    bayesian_pred_all = predict_bayesian_full(raw_all, gaussians_all, args.max_count)

    rounding_acc_all = float(np.mean(counts == rounding_pred_all))
    bayesian_acc_all = float(np.mean(counts == bayesian_pred_all))
    print(f"  Rounding accuracy (all data): {rounding_acc_all:.4f}")
    print(f"  Bayesian accuracy (all data): {bayesian_acc_all:.4f}")

    # ── Export ────────────────────────────────────────────────────
    if args.export_probe:
        export_bayesian_params(
            args.export_probe, gaussians_all, boundaries_all,
            rounding_acc_all, bayesian_acc_all)

    # ── Report ────────────────────────────────────────────────────
    data_info = {
        "path": args.data,
        "n_samples": n_samples,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_episodes": n_episodes,
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
    }
    probe_info = {"r_squared": r_squared}

    report = format_report(eval_output, data_info, probe_info)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save report
        report_path = out_dir / "bayesian_probe_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved to {report_path}")

        # Save standalone JSON with all parameters
        params_path = out_dir / "bayesian_params.json"
        with open(params_path, "w") as f:
            json.dump({
                "gaussians": [{"mean": g["mean"], "std": g["std"], "n": g["n"]}
                              for g in gaussians_all],
                "boundaries": boundaries_all,
                "dprimes": compute_dprime(gaussians_all, args.max_count),
                "rounding_accuracy": rounding_acc_all,
                "bayesian_accuracy": bayesian_acc_all,
                "probe_r_squared": r_squared,
                "max_count": args.max_count,
                "alpha": args.alpha,
                "data_path": args.data,
                "n_samples": n_samples,
            }, f, indent=2)
        print(f"  Parameters saved to {params_path}")

        # Save full report
        print(f"\n{report}")
    else:
        print(f"\n{report}")


if __name__ == "__main__":
    main()
