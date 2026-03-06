#!/usr/bin/env python3
"""
Transition Detectors — automated pattern flagging in RSSM hidden state dynamics.

Six detectors that scan every count transition and flag interesting patterns:
1. Anticipation: dims that start moving before blob lands
2. Overshoot-correction: dims that reverse direction after landing
3. Coordinated movement: dims that move together during transitions
4. Slow drift: dims that change gradually (spatial tracking, not counting)
5. Saturation: dims pinned near ±1 (tanh boundaries)
6. Surprise: transitions with anomalous dynamics vs same-type transitions

Plus: dimensionality identifier (which dims distinguish D values).

Usage:
    python3 transition_detectors.py --checkpoint-dir DIR --dim 2 --episodes 20 --blob-count 13
    python3 transition_detectors.py --checkpoint-dir DIR --dim 2 3 4 5 --episodes 10
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# RSSM loading (import from anatomy script)
# ---------------------------------------------------------------------------

def _load_and_collect(checkpoint_dir, D, n_episodes, blob_count):
    """Load model, run episodes, return list of episode dicts."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hidden_state_anatomy import load_weights, FastRSSM
    from counting_env_multidim import MultiDimCountingWorldEnv

    weights = load_weights(checkpoint_dir)
    obs_size = weights["enc_linear0_w"].shape[1]

    env = MultiDimCountingWorldEnv(
        blob_count_min=blob_count, blob_count_max=blob_count,
        fixed_dim=D, proj_dim=obs_size,
    )
    model = FastRSSM(weights)

    episodes = []
    for ep in range(n_episodes):
        vec = env.reset()
        model.reset()
        done = False
        ep_data = {"h": [], "count": [], "step": []}
        step_i = 0
        while not done:
            obs = vec[:obs_size].astype(np.float32)
            count = int(env._state.grid.filled_count)
            deter = model.step(obs, 0.0)
            ep_data["h"].append(deter)
            ep_data["count"].append(count)
            ep_data["step"].append(step_i)
            vec, _, done, _ = env.step(-0.995)
            step_i += 1
        ep_data["h"] = np.stack(ep_data["h"])
        ep_data["count"] = np.array(ep_data["count"])
        episodes.append(ep_data)
    env.close()
    return episodes


def _find_transitions(counts):
    """Find timesteps where count changes. Returns list of (step, from_c, to_c)."""
    transitions = []
    for t in range(1, len(counts)):
        if counts[t] != counts[t - 1]:
            transitions.append((t, int(counts[t - 1]), int(counts[t])))
    return transitions


# ---------------------------------------------------------------------------
# Detector 1: Anticipation
# ---------------------------------------------------------------------------

def detect_anticipation(episodes, pre_window=15, sigma_threshold=2.0):
    """Flag dims whose velocity is elevated before blob lands."""
    N_DIMS = 512
    results = []

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]
        counts = ep["count"]
        T = len(counts)
        transitions = _find_transitions(counts)

        # Compute per-step velocity for each dim
        velocity = np.diff(H, axis=0)  # (T-1, 512)

        # Baseline: velocity variance during non-transition periods
        # Mark transition windows
        trans_mask = np.zeros(T - 1, dtype=bool)
        for (t, _, _) in transitions:
            lo = max(0, t - pre_window)
            hi = min(T - 1, t + pre_window)
            trans_mask[lo:hi] = True

        baseline_vel = velocity[~trans_mask]
        if len(baseline_vel) < 20:
            continue
        baseline_std = baseline_vel.std(axis=0) + 1e-10
        baseline_mean = baseline_vel.mean(axis=0)

        for (t, from_c, to_c) in transitions:
            if t < pre_window + 2:
                continue

            # Pre-transition window velocity
            pre_vel = velocity[t - pre_window : t]  # (window, 512)
            pre_mean_vel = pre_vel.mean(axis=0)

            # Z-score: how far above baseline
            z_scores = (np.abs(pre_mean_vel) - np.abs(baseline_mean)) / baseline_std
            anticipating = np.where(z_scores > sigma_threshold)[0]

            if len(anticipating) > 0:
                # Sort by z-score
                order = np.argsort(-z_scores[anticipating])
                anticipating = anticipating[order]

                # For top dims, find how early anticipation starts
                early_starts = []
                for d in anticipating[:10]:
                    # Walk backwards from transition to find when velocity first exceeds threshold
                    for look_back in range(1, pre_window + 1):
                        if t - look_back < 1:
                            break
                        v = abs(velocity[t - look_back, d])
                        if v < abs(baseline_mean[d]) + sigma_threshold * baseline_std[d]:
                            early_starts.append(look_back - 1)
                            break
                    else:
                        early_starts.append(pre_window)

                results.append({
                    "episode": ep_idx,
                    "step": int(t),
                    "transition": f"{from_c}->{to_c}",
                    "n_anticipating": int(len(anticipating)),
                    "top_dims": anticipating[:15].tolist(),
                    "top_z_scores": z_scores[anticipating[:15]].tolist(),
                    "early_starts": early_starts,
                })

    return {"detector": "anticipation", "pre_window": pre_window,
            "sigma_threshold": sigma_threshold, "n_transitions": len(results),
            "transitions": results}


# ---------------------------------------------------------------------------
# Detector 2: Overshoot-correction
# ---------------------------------------------------------------------------

def detect_overshoot(episodes, pre_window=10, post_window=15):
    """Flag dims that reverse direction after blob lands."""
    results = []

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]
        counts = ep["count"]
        T = len(counts)
        transitions = _find_transitions(counts)

        for (t, from_c, to_c) in transitions:
            if t < pre_window + 1 or t + post_window >= T:
                continue

            # Direction during approach (pre) and settle (post)
            delta_pre = H[t] - H[t - pre_window]      # approach direction
            delta_post = H[t + post_window] - H[t]     # settle direction

            # Overshoot: sign reversal AND both have substantial magnitude
            magnitude_threshold = 0.05
            sign_reversal = (delta_pre * delta_post < 0)
            substantial = (np.abs(delta_pre) > magnitude_threshold) & (np.abs(delta_post) > magnitude_threshold)
            overshoot_mask = sign_reversal & substantial

            overshoot_dims = np.where(overshoot_mask)[0]
            if len(overshoot_dims) == 0:
                continue

            # Sort by overshoot magnitude
            overshoot_mag = np.abs(delta_pre[overshoot_dims]) + np.abs(delta_post[overshoot_dims])
            order = np.argsort(-overshoot_mag)
            overshoot_dims = overshoot_dims[order]

            # For top dims, measure settling time
            settling_times = []
            for d in overshoot_dims[:10]:
                settled_val = H[t + post_window, d]
                for s in range(1, post_window):
                    if abs(H[t + s, d] - settled_val) < 0.02:
                        settling_times.append(s)
                        break
                else:
                    settling_times.append(post_window)

            results.append({
                "episode": ep_idx,
                "step": int(t),
                "transition": f"{from_c}->{to_c}",
                "n_overshoot": int(len(overshoot_dims)),
                "top_dims": overshoot_dims[:15].tolist(),
                "top_pre_delta": delta_pre[overshoot_dims[:15]].tolist(),
                "top_post_delta": delta_post[overshoot_dims[:15]].tolist(),
                "settling_times": settling_times,
            })

    return {"detector": "overshoot", "pre_window": pre_window,
            "post_window": post_window, "n_transitions": len(results),
            "transitions": results}


# ---------------------------------------------------------------------------
# Detector 3: Coordinated movement
# ---------------------------------------------------------------------------

def detect_coordinated_movement(episodes, window=15, corr_threshold=0.8, top_n_transitions=5):
    """Find dims that move together during transitions."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    results = []

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]
        counts = ep["count"]
        T = len(counts)
        transitions = _find_transitions(counts)

        for ti, (t, from_c, to_c) in enumerate(transitions):
            if ti >= top_n_transitions:
                break  # Only analyze first N transitions per episode for speed
            if t + window >= T or t < 2:
                continue

            # Velocity time series in the transition window
            vel = np.diff(H[t - 2 : t + window], axis=0)  # (window+1, 512)
            if vel.shape[0] < 5:
                continue

            # Correlation matrix of dim velocities
            vel_std = vel.std(axis=0)
            active_dims = np.where(vel_std > 0.005)[0]
            if len(active_dims) < 10:
                continue

            vel_active = vel[:, active_dims]
            # Standardize
            vel_z = (vel_active - vel_active.mean(axis=0)) / (vel_active.std(axis=0) + 1e-10)
            corr = (vel_z.T @ vel_z) / vel_z.shape[0]
            np.fill_diagonal(corr, 1.0)

            # Cluster using absolute correlation
            dist = 1.0 - np.abs(corr)
            np.fill_diagonal(dist, 0)
            dist = np.clip((dist + dist.T) / 2, 0, 2)
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=0.4, criterion="distance")

            # Find clusters with 5+ members
            clusters = []
            for cl in np.unique(labels):
                members = active_dims[labels == cl]
                if len(members) >= 5:
                    # Mean pairwise correlation within cluster
                    sub_corr = corr[np.ix_(labels == cl, labels == cl)]
                    mask = ~np.eye(len(members), dtype=bool)
                    mean_corr = float(sub_corr[mask].mean()) if mask.sum() > 0 else 1.0
                    clusters.append({
                        "dims": members.tolist(),
                        "n_dims": int(len(members)),
                        "mean_internal_corr": round(mean_corr, 3),
                    })

            if clusters:
                results.append({
                    "episode": ep_idx,
                    "step": int(t),
                    "transition": f"{from_c}->{to_c}",
                    "n_active_dims": int(len(active_dims)),
                    "n_clusters": len(clusters),
                    "clusters": sorted(clusters, key=lambda c: -c["n_dims"]),
                })

    return {"detector": "coordinated_movement", "window": window,
            "n_transitions": len(results), "transitions": results}


# ---------------------------------------------------------------------------
# Detector 4: Slow drift
# ---------------------------------------------------------------------------

def detect_slow_drift(episodes, monotonicity_threshold=0.8):
    """Find dims that change gradually across the episode (spatial tracking)."""
    results = []

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]
        T = H.shape[0]
        if T < 30:
            continue

        # For each dim, compute monotonicity: fraction of steps where sign of diff is consistent
        diffs = np.diff(H, axis=0)  # (T-1, 512)
        pos_frac = (diffs > 0).mean(axis=0)  # fraction of positive steps
        monotonicity = np.maximum(pos_frac, 1 - pos_frac)  # 1.0 = perfectly monotonic

        # Total drift magnitude
        total_drift = np.abs(H[-1] - H[0])

        # Flag dims that are monotonic AND drift substantially
        drift_mask = (monotonicity > monotonicity_threshold) & (total_drift > 0.3)
        drift_dims = np.where(drift_mask)[0]

        if len(drift_dims) > 0:
            order = np.argsort(-total_drift[drift_dims])
            drift_dims = drift_dims[order]

            results.append({
                "episode": ep_idx,
                "n_drifting": int(len(drift_dims)),
                "top_dims": drift_dims[:20].tolist(),
                "top_drift_magnitude": total_drift[drift_dims[:20]].tolist(),
                "top_monotonicity": monotonicity[drift_dims[:20]].tolist(),
                "top_direction": ["+" if H[-1, d] > H[0, d] else "-"
                                  for d in drift_dims[:20]],
            })

    # Aggregate: which dims drift consistently across episodes
    dim_drift_count = np.zeros(512, dtype=int)
    for r in results:
        for d in r["top_dims"]:
            dim_drift_count[d] += 1
    consistent_drifters = np.where(dim_drift_count >= len(episodes) * 0.5)[0]

    return {
        "detector": "slow_drift",
        "monotonicity_threshold": monotonicity_threshold,
        "n_episodes_analyzed": len(results),
        "n_consistent_drifters": int(len(consistent_drifters)),
        "consistent_drifter_dims": consistent_drifters.tolist(),
        "per_episode": results[:5],  # keep first 5 for detail
    }


# ---------------------------------------------------------------------------
# Detector 5: Saturation
# ---------------------------------------------------------------------------

def detect_saturation(episodes, threshold=0.95):
    """Flag dims pinned near ±1 (tanh boundaries)."""
    all_means = []
    all_stds = []

    for ep in episodes:
        H = ep["h"]
        all_means.append(np.abs(H).mean(axis=0))
        all_stds.append(H.std(axis=0))

    mean_abs = np.mean(all_means, axis=0)
    mean_std = np.mean(all_stds, axis=0)

    # Saturated: mean |value| > threshold AND low variance
    saturated_high = np.where((mean_abs > threshold) & (mean_std < 0.05))[0]

    # Near-dead: very low variance regardless of mean
    near_dead = np.where(mean_std < 0.01)[0]

    return {
        "detector": "saturation",
        "threshold": threshold,
        "n_saturated": int(len(saturated_high)),
        "saturated_dims": saturated_high.tolist(),
        "saturated_values": mean_abs[saturated_high].tolist(),
        "n_near_dead": int(len(near_dead)),
        "near_dead_dims": near_dead.tolist(),
        "near_dead_std": mean_std[near_dead].tolist(),
    }


# ---------------------------------------------------------------------------
# Detector 6: Surprise (cross-episode comparison)
# ---------------------------------------------------------------------------

def detect_surprise(episodes, z_threshold=2.5):
    """Flag transitions with anomalous dynamics vs same-type transitions."""
    # Group transitions by type (from_c -> to_c)
    trans_by_type = defaultdict(list)  # type -> list of (ep_idx, step, delta_vector)

    for ep_idx, ep in enumerate(episodes):
        H = ep["h"]
        T = H.shape[0]
        transitions = _find_transitions(ep["count"])
        for (t, from_c, to_c) in transitions:
            if t < 5 or t + 10 >= T:
                continue
            # Characterize transition by hidden state change over ±5 steps
            delta = H[min(t + 5, T - 1)] - H[max(t - 5, 0)]
            trans_by_type[f"{from_c}->{to_c}"].append({
                "ep": ep_idx,
                "step": int(t),
                "delta": delta,
            })

    results = []
    for trans_type, instances in trans_by_type.items():
        if len(instances) < 5:
            continue

        # Stack all deltas for this transition type
        deltas = np.stack([inst["delta"] for inst in instances])
        mean_delta = deltas.mean(axis=0)
        std_delta = deltas.std(axis=0) + 1e-10

        # For each instance, compute z-scores vs the group
        for i, inst in enumerate(instances):
            z = np.abs(inst["delta"] - mean_delta) / std_delta
            anomalous_dims = np.where(z > z_threshold)[0]

            if len(anomalous_dims) > 5:
                order = np.argsort(-z[anomalous_dims])
                anomalous_dims = anomalous_dims[order]
                results.append({
                    "transition_type": trans_type,
                    "episode": inst["ep"],
                    "step": inst["step"],
                    "n_anomalous_dims": int(len(anomalous_dims)),
                    "top_dims": anomalous_dims[:10].tolist(),
                    "top_z_scores": z[anomalous_dims[:10]].tolist(),
                    "group_size": len(instances),
                })

    # Sort by max z-score
    results.sort(key=lambda r: -max(r["top_z_scores"]) if r["top_z_scores"] else 0)

    return {
        "detector": "surprise",
        "z_threshold": z_threshold,
        "n_transition_types": len(trans_by_type),
        "n_anomalous_transitions": len(results),
        "top_anomalies": results[:20],
    }


# ---------------------------------------------------------------------------
# Bonus: Dimensionality identifier (cross-D comparison)
# ---------------------------------------------------------------------------

def detect_dim_identifier(all_episodes_by_D):
    """Find dims that discriminate between spatial dimensionalities."""
    if len(all_episodes_by_D) < 2:
        return {"detector": "dim_identifier", "error": "need 2+ dimensions"}

    # For each D, compute mean hidden state (averaged across all steps and episodes)
    D_means = {}
    for D, episodes in all_episodes_by_D.items():
        all_h = np.concatenate([ep["h"] for ep in episodes])
        D_means[D] = all_h.mean(axis=0)

    # For each dim, compute between-D variance vs within-D variance
    D_values = sorted(D_means.keys())
    grand_mean = np.mean([D_means[d] for d in D_values], axis=0)
    between_var = np.mean([(D_means[d] - grand_mean) ** 2 for d in D_values], axis=0)

    within_vars = []
    for D, episodes in all_episodes_by_D.items():
        all_h = np.concatenate([ep["h"] for ep in episodes])
        within_vars.append(all_h.var(axis=0))
    within_var = np.mean(within_vars, axis=0) + 1e-10

    # F-ratio: between / within
    f_ratio = between_var / within_var
    top_discriminators = np.argsort(-f_ratio)[:30]

    return {
        "detector": "dim_identifier",
        "D_values": D_values,
        "top_30_discriminator_dims": top_discriminators.tolist(),
        "top_30_f_ratios": f_ratio[top_discriminators].tolist(),
        "n_dims_f_above_1": int((f_ratio > 1.0).sum()),
        "n_dims_f_above_0.1": int((f_ratio > 0.1).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Transition Detectors")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dim", type=int, nargs="+", default=[2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--blob-count", type=int, default=13)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.checkpoint_dir) / "transition_detectors.json"
    )

    print("=" * 60)
    print("Transition Detectors")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Dims: {args.dim}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Blob count: {args.blob_count}")
    print()

    all_episodes_by_D = {}
    all_results = {
        "checkpoint": args.checkpoint_dir,
        "blob_count": args.blob_count,
        "episodes_per_dim": args.episodes,
    }

    for D in args.dim:
        print(f"--- D={D}: Collecting {args.episodes} episodes ---", flush=True)
        t0 = time.time()
        episodes = _load_and_collect(args.checkpoint_dir, D, args.episodes, args.blob_count)
        n_steps = sum(len(ep["count"]) for ep in episodes)
        n_transitions = sum(len(_find_transitions(ep["count"])) for ep in episodes)
        print(f"  {n_steps} steps, {n_transitions} transitions in {time.time()-t0:.1f}s")
        all_episodes_by_D[D] = episodes

        print(f"  Running detectors...", flush=True)
        d_results = {}

        t1 = time.time()
        d_results["anticipation"] = detect_anticipation(episodes)
        print(f"    Anticipation: {d_results['anticipation']['n_transitions']} transitions flagged ({time.time()-t1:.1f}s)")

        t1 = time.time()
        d_results["overshoot"] = detect_overshoot(episodes)
        print(f"    Overshoot: {d_results['overshoot']['n_transitions']} transitions flagged ({time.time()-t1:.1f}s)")

        t1 = time.time()
        d_results["coordinated"] = detect_coordinated_movement(episodes)
        print(f"    Coordinated: {d_results['coordinated']['n_transitions']} transitions analyzed ({time.time()-t1:.1f}s)")

        t1 = time.time()
        d_results["slow_drift"] = detect_slow_drift(episodes)
        print(f"    Slow drift: {d_results['slow_drift']['n_consistent_drifters']} consistent drifters ({time.time()-t1:.1f}s)")

        t1 = time.time()
        d_results["saturation"] = detect_saturation(episodes)
        print(f"    Saturation: {d_results['saturation']['n_saturated']} saturated, "
              f"{d_results['saturation']['n_near_dead']} near-dead ({time.time()-t1:.1f}s)")

        t1 = time.time()
        d_results["surprise"] = detect_surprise(episodes)
        print(f"    Surprise: {d_results['surprise']['n_anomalous_transitions']} anomalous ({time.time()-t1:.1f}s)")

        all_results[f"D{D}"] = d_results

    # Cross-D dimensionality identifier
    if len(args.dim) >= 2:
        print(f"\n--- Cross-D dimensionality identifier ---", flush=True)
        dim_id = detect_dim_identifier(all_episodes_by_D)
        all_results["dim_identifier"] = dim_id
        print(f"  Top discriminator dim: {dim_id['top_30_discriminator_dims'][0]}, "
              f"F={dim_id['top_30_f_ratios'][0]:.2f}")
        print(f"  Dims with F>1: {dim_id['n_dims_f_above_1']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for D in args.dim:
        r = all_results[f"D{D}"]
        ant = r["anticipation"]
        ovr = r["overshoot"]
        sat = r["saturation"]
        drft = r["slow_drift"]
        surp = r["surprise"]

        # Typical anticipation count
        if ant["transitions"]:
            mean_ant = np.mean([t["n_anticipating"] for t in ant["transitions"]])
        else:
            mean_ant = 0

        # Typical overshoot count
        if ovr["transitions"]:
            mean_ovr = np.mean([t["n_overshoot"] for t in ovr["transitions"]])
        else:
            mean_ovr = 0

        print(f"D={D}: {mean_ant:.0f} anticipating dims/transition, "
              f"{mean_ovr:.0f} overshoot dims/transition, "
              f"{sat['n_saturated']} saturated, {drft['n_consistent_drifters']} drifters, "
              f"{surp['n_anomalous_transitions']} surprises")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
