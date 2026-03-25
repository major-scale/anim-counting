#!/usr/bin/env python3
"""Critical Slowing Down Analysis — Binary Specialist RSSM.

Measures all three classic CSD indicators (Scheffer et al. 2009) across
all 15 count states:
  1. Increased variance during idle periods
  2. Increased autocorrelation (AR1)
  3. Slower perturbation recovery (half-life)
  4. Jacobian spectral radius (eigenvalue analysis)
  5. Temporal dynamics of instability (quartile analysis at counts 3, 7, 11)

All probes use column-state calibration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge

# ─── Path setup ───────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = Path('/workspace/projects/jamstack-v1/bridge')
sys.path.insert(0, str(BRIDGE_DIR / 'scripts'))

from binary_counting_env import BinaryCountingEnv, OBS_SIZE
from imagination_rollout_binary import (
    FastRSSMWithImagination, load_exported_weights, carry_depth,
    DETER_DIM, STOCH_FLAT, _ln, _silu, _sigmoid, LN_EPS
)

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLUMNS = 4

# Upcoming cascade depth for each count
UPCOMING_DEPTH = {c: carry_depth(c) for c in range(14)}
UPCOMING_DEPTH[14] = -1  # No next transition


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_episodes(rssm_weights, n_episodes=15, max_steps=1200):
    """Collect full episodes with h_t, obs, stoch, col_bits at every step."""
    print(f'  Collecting {n_episodes} episodes...')
    episodes = []

    for ep in range(n_episodes):
        rssm = FastRSSMWithImagination(rssm_weights)
        env = BinaryCountingEnv(seed=42 + ep)
        obs = env.reset()
        rssm.reset()

        h_list, obs_list, stoch_list = [], [], []
        col_bits_list, dc_list = [], []
        state_list = []

        for t in range(max_steps):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            obs_list.append(obs_vec.copy())

            deter = rssm.step(obs_vec, action=0)
            h_list.append(deter)
            stoch_list.append(rssm.stoch.copy())
            state_list.append(rssm.get_state())

            col_bits = [1 if env._state.columns[i].occupied else 0
                        for i in range(NUM_COLUMNS)]
            dc = env._state.decimal_count
            col_bits_list.append(col_bits)
            dc_list.append(dc)

            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']

            if dc >= 14:
                for _ in range(25):
                    t += 1
                    obs_vec = obs[:OBS_SIZE].astype(np.float32)
                    obs_list.append(obs_vec.copy())
                    deter = rssm.step(obs_vec, action=0)
                    h_list.append(deter)
                    stoch_list.append(rssm.stoch.copy())
                    state_list.append(rssm.get_state())
                    col_bits = [1 if env._state.columns[i].occupied else 0
                                for i in range(NUM_COLUMNS)]
                    col_bits_list.append(col_bits)
                    dc_list.append(env._state.decimal_count)
                    result = env.step(0)
                    obs = result[0] if isinstance(result, tuple) else result['obs']
                break

        episodes.append({
            'h': np.array(h_list),
            'obs': np.array(obs_list),
            'stoch': np.array(stoch_list),
            'states': state_list,
            'col_bits': np.array(col_bits_list),
            'dc': np.array(dc_list),
        })
        if (ep + 1) % 5 == 0:
            print(f'    Episode {ep+1}/{n_episodes}: {len(h_list)} steps, max count={max(dc_list)}')

    return episodes


def extract_idle_segments(episodes, trim=5):
    """Extract idle segments for each count, trimming transition edges.

    Returns dict: count -> list of (h_segment, obs_segment, stoch_segment, state_segment)
    """
    idle_by_count = {c: [] for c in range(15)}

    for ep in episodes:
        dc = ep['dc']
        h = ep['h']
        obs = ep['obs']
        stoch = ep['stoch']
        states = ep['states']

        # Find contiguous segments of constant count
        seg_start = 0
        for t in range(1, len(dc)):
            if dc[t] != dc[t-1] or t == len(dc) - 1:
                seg_end = t if dc[t] != dc[t-1] else t + 1
                count_val = dc[seg_start]
                seg_len = seg_end - seg_start

                # Trim edges to avoid transition effects
                if seg_len > 2 * trim + 5:
                    ts = seg_start + trim
                    te = seg_end - trim
                    idle_by_count[count_val].append({
                        'h': h[ts:te],
                        'obs': obs[ts:te],
                        'stoch': stoch[ts:te],
                        'states': states[ts:te],
                        'start': ts,
                        'end': te,
                    })

                seg_start = t

    return idle_by_count


# ─── Analysis 1: Variance Profile ────────────────────────────────────────────

def analysis_1_variance(idle_by_count):
    """Compute per-dimension variance during idle periods for each count."""
    print('\n--- Analysis 1: Variance Profile ---')

    variance_by_count = {}

    for c in range(15):
        segs = idle_by_count[c]
        if not segs:
            continue

        # Concatenate all idle h_t for this count
        all_h = np.concatenate([seg['h'] for seg in segs], axis=0)
        # Per-dimension variance, then mean
        per_dim_var = all_h.var(axis=0)  # (512,)
        mean_var = float(per_dim_var.mean())
        total_var = float(per_dim_var.sum())
        n_steps = len(all_h)

        upcoming = UPCOMING_DEPTH.get(c, -1)
        variance_by_count[c] = {
            'mean_var': mean_var,
            'total_var': total_var,
            'n_steps': n_steps,
            'n_segments': len(segs),
            'upcoming_depth': upcoming,
        }

    print(f'\n{"Count":<8} {"Depth":<7} {"Mean Var":<14} {"Total Var":<14} {"n_steps":<10}')
    print('-' * 55)
    for c in range(15):
        if c not in variance_by_count:
            continue
        v = variance_by_count[c]
        print(f'{c:<8} {v["upcoming_depth"]:<7} {v["mean_var"]:<14.6f} '
              f'{v["total_var"]:<14.3f} {v["n_steps"]:<10}')

    # Correlation with upcoming depth (exclude count 14)
    counts = [c for c in range(14) if c in variance_by_count]
    depths = [variance_by_count[c]['upcoming_depth'] for c in counts]
    vars_ = [variance_by_count[c]['mean_var'] for c in counts]
    r, p = stats.spearmanr(depths, vars_)
    print(f'\n  Spearman correlation (depth vs variance): r={r:.4f}, p={p:.4f}')

    # Test: depth-0 counts only — is variance uniform?
    depth0_counts = [c for c in counts if UPCOMING_DEPTH[c] == 0]
    depth0_vars = [variance_by_count[c]['mean_var'] for c in depth0_counts]
    print(f'  Depth-0 counts variance range: {min(depth0_vars):.6f} - {max(depth0_vars):.6f} '
          f'(ratio {max(depth0_vars)/min(depth0_vars):.2f}x)')

    return variance_by_count, r, p


# ─── Analysis 2: Autocorrelation (AR1) ───────────────────────────────────────

def analysis_2_ar1(idle_by_count):
    """Compute lag-1 autocorrelation during idle periods."""
    print('\n--- Analysis 2: Autocorrelation (AR1) ---')

    ar1_by_count = {}

    for c in range(15):
        segs = idle_by_count[c]
        if not segs:
            continue

        # Compute AR1 per segment, per dimension, then average
        seg_ar1s = []
        for seg in segs:
            h = seg['h']
            if len(h) < 5:
                continue
            # Per-dimension AR1
            h_centered = h - h.mean(axis=0, keepdims=True)
            numerator = (h_centered[:-1] * h_centered[1:]).sum(axis=0)
            denominator = (h_centered ** 2).sum(axis=0)
            # Avoid division by zero
            mask = denominator > 1e-12
            dim_ar1 = np.zeros(DETER_DIM)
            dim_ar1[mask] = numerator[mask] / denominator[mask]
            seg_ar1s.append(dim_ar1.mean())

        if seg_ar1s:
            mean_ar1 = float(np.mean(seg_ar1s))
            std_ar1 = float(np.std(seg_ar1s))
        else:
            mean_ar1 = 0.0
            std_ar1 = 0.0

        upcoming = UPCOMING_DEPTH.get(c, -1)
        ar1_by_count[c] = {
            'mean_ar1': mean_ar1,
            'std_ar1': std_ar1,
            'n_segments': len(seg_ar1s),
            'upcoming_depth': upcoming,
        }

    print(f'\n{"Count":<8} {"Depth":<7} {"AR1":<14} {"Std":<14} {"n_segs":<8}')
    print('-' * 50)
    for c in range(15):
        if c not in ar1_by_count:
            continue
        a = ar1_by_count[c]
        print(f'{c:<8} {a["upcoming_depth"]:<7} {a["mean_ar1"]:<14.4f} '
              f'{a["std_ar1"]:<14.4f} {a["n_segments"]:<8}')

    # Correlation with depth
    counts = [c for c in range(14) if c in ar1_by_count]
    depths = [ar1_by_count[c]['upcoming_depth'] for c in counts]
    ar1s = [ar1_by_count[c]['mean_ar1'] for c in counts]
    r, p = stats.spearmanr(depths, ar1s)
    print(f'\n  Spearman correlation (depth vs AR1): r={r:.4f}, p={p:.4f}')

    return ar1_by_count, r, p


# ─── Analysis 3: Perturbation Recovery ────────────────────────────────────────

def analysis_3_perturbation_recovery(episodes, rssm_weights, idle_by_count,
                                      n_perturbations=5, recovery_horizon=20):
    """Perturb hidden state and measure recovery time."""
    print('\n--- Analysis 3: Perturbation Recovery ---')

    # Compute typical centroid-to-centroid distance for calibrating perturbation size
    centroids = {}
    for c in range(15):
        segs = idle_by_count[c]
        if segs:
            all_h = np.concatenate([s['h'] for s in segs], axis=0)
            centroids[c] = all_h.mean(axis=0)

    centroid_dists = []
    for c in range(14):
        if c in centroids and c + 1 in centroids:
            centroid_dists.append(np.linalg.norm(centroids[c+1] - centroids[c]))
    mean_centroid_dist = np.mean(centroid_dists)
    eps_magnitude = 0.1 * mean_centroid_dist  # 10% of typical centroid distance
    print(f'  Mean centroid distance: {mean_centroid_dist:.4f}')
    print(f'  Perturbation magnitude: {eps_magnitude:.4f}')

    rng = np.random.RandomState(123)
    recovery_by_count = {}
    max_segs_per_count = 3  # Use only 3 longest segments to limit compute

    # Reuse two RSSM instances
    rssm_orig = FastRSSMWithImagination(rssm_weights)
    rssm_pert = FastRSSMWithImagination(rssm_weights)

    for c in range(15):
        segs = idle_by_count[c]
        if not segs:
            continue

        # Pick the longest segments
        valid_segs = [s for s in segs if len(s['h']) >= recovery_horizon + 5]
        valid_segs.sort(key=lambda s: len(s['h']), reverse=True)
        valid_segs = valid_segs[:max_segs_per_count]

        half_lives = []

        for seg in valid_segs:
            h = seg['h']
            obs = seg['obs']
            states = seg['states']

            # Pick a midpoint in the idle period
            mid = len(h) // 2

            for p_idx in range(n_perturbations):
                # Random perturbation direction
                direction = rng.randn(DETER_DIM).astype(np.float32)
                direction /= np.linalg.norm(direction)
                perturbation = direction * eps_magnitude

                # Reset both to the state at midpoint
                rssm_orig.set_state(states[mid])
                rssm_pert.set_state(states[mid])

                # Apply perturbation to deter only
                rssm_pert.deter = rssm_pert.deter + perturbation

                # Run both forward with the same observations
                distances = [np.linalg.norm(rssm_pert.deter - rssm_orig.deter)]
                steps_available = min(recovery_horizon, len(obs) - mid - 1)

                for k in range(steps_available):
                    obs_k = obs[mid + k]
                    rssm_orig.step(obs_k, action=0)
                    rssm_pert.step(obs_k, action=0)
                    d = np.linalg.norm(rssm_pert.deter - rssm_orig.deter)
                    distances.append(d)

                # Compute half-life
                distances = np.array(distances)
                initial_dist = distances[0]
                half_target = initial_dist * 0.5

                half_life = steps_available  # default: didn't recover
                for k in range(1, len(distances)):
                    if distances[k] <= half_target:
                        # Interpolate
                        frac = (distances[k-1] - half_target) / (distances[k-1] - distances[k])
                        half_life = (k - 1) + frac
                        break

                half_lives.append(half_life)

        print(f'  Count {c}: {len(half_lives)} perturbation experiments', flush=True)

        if half_lives:
            mean_hl = float(np.mean(half_lives))
            std_hl = float(np.std(half_lives))
            median_hl = float(np.median(half_lives))
        else:
            mean_hl = std_hl = median_hl = 0.0

        upcoming = UPCOMING_DEPTH.get(c, -1)
        recovery_by_count[c] = {
            'mean_half_life': mean_hl,
            'std_half_life': std_hl,
            'median_half_life': median_hl,
            'n_perturbations': len(half_lives),
            'upcoming_depth': upcoming,
        }

    print(f'\n{"Count":<8} {"Depth":<7} {"Mean HL":<12} {"Median HL":<12} {"Std HL":<12} {"n":<6}')
    print('-' * 58)
    for c in range(15):
        if c not in recovery_by_count:
            continue
        r = recovery_by_count[c]
        print(f'{c:<8} {r["upcoming_depth"]:<7} {r["mean_half_life"]:<12.2f} '
              f'{r["median_half_life"]:<12.2f} {r["std_half_life"]:<12.2f} {r["n_perturbations"]:<6}')

    # Correlation with depth
    counts = [c for c in range(14) if c in recovery_by_count]
    depths = [recovery_by_count[c]['upcoming_depth'] for c in counts]
    hls = [recovery_by_count[c]['mean_half_life'] for c in counts]
    r, p = stats.spearmanr(depths, hls)
    print(f'\n  Spearman correlation (depth vs half-life): r={r:.4f}, p={p:.4f}')

    return recovery_by_count, r, p


# ─── Analysis 4: Jacobian Eigenvalue Analysis ────────────────────────────────

def gru_transition_fn(deter, stoch, action, weights, num_actions):
    """Pure function: compute GRU transition without modifying any state."""
    w = weights

    if num_actions > 1:
        act = np.zeros(num_actions, dtype=np.float32)
        act[int(action)] = 1.0
    else:
        act = np.atleast_1d(np.float32(action))

    cat_in = np.concatenate([stoch, act])
    h_in = _silu(_ln(w["img_in_w"] @ cat_in,
                      w["img_in_norm_w"], w["img_in_norm_b"]))

    combined = np.concatenate([h_in, deter])
    out = w["gru_w"] @ combined
    out = _ln(out, w["gru_norm_w"], w["gru_norm_b"])
    N = DETER_DIM
    reset_gate = _sigmoid(out[:N])
    cand = np.tanh(reset_gate * out[N:2*N])
    update = _sigmoid(out[2*N:] + FastRSSMWithImagination.GRU_UPDATE_BIAS)
    new_deter = (update * cand + (1 - update) * deter).astype(np.float32)
    return new_deter


def compute_jacobian(deter_center, stoch, action, weights, num_actions, eps=1e-4):
    """Compute Jacobian ∂h_{t+1}/∂h_t via finite differences."""
    N = DETER_DIM
    baseline = gru_transition_fn(deter_center, stoch, action, weights, num_actions)

    J = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        deter_plus = deter_center.copy()
        deter_plus[i] += eps
        output_plus = gru_transition_fn(deter_plus, stoch, action, weights, num_actions)
        J[:, i] = (output_plus - baseline) / eps

    return J


def analysis_4_eigenvalues(idle_by_count, rssm_weights):
    """Compute Jacobian spectral radius at each count centroid."""
    print('\n--- Analysis 4: Jacobian Eigenvalue Analysis ---')

    rssm_temp = FastRSSMWithImagination(rssm_weights)
    num_actions = rssm_temp.num_actions

    eigenvalue_by_count = {}

    for c in range(15):
        segs = idle_by_count[c]
        if not segs:
            continue

        # Compute centroid of h_t and median stoch during idle
        all_h = np.concatenate([s['h'] for s in segs], axis=0)
        all_stoch = np.concatenate([s['stoch'] for s in segs], axis=0)
        centroid = all_h.mean(axis=0).astype(np.float32)
        # Use the most common stoch (mode of discrete variable)
        median_stoch = all_stoch[len(all_stoch)//2].copy()

        print(f'  Computing Jacobian at count {c}...', end='', flush=True)
        J = compute_jacobian(centroid, median_stoch, 0, rssm_weights, num_actions)

        # Eigenvalues
        eigs = np.linalg.eigvals(J)
        spectral_radius = float(np.max(np.abs(eigs)))
        mean_abs_eig = float(np.mean(np.abs(eigs)))

        # Top-k eigenvalues
        sorted_eigs = np.sort(np.abs(eigs))[::-1]
        top5 = sorted_eigs[:5].tolist()

        upcoming = UPCOMING_DEPTH.get(c, -1)
        eigenvalue_by_count[c] = {
            'spectral_radius': spectral_radius,
            'mean_abs_eigenvalue': mean_abs_eig,
            'top5_abs_eigenvalues': top5,
            'upcoming_depth': upcoming,
            'eigenvalues_real': eigs.real.tolist(),
            'eigenvalues_imag': eigs.imag.tolist(),
        }
        print(f' spectral_radius={spectral_radius:.4f}')

    print(f'\n{"Count":<8} {"Depth":<7} {"Spec Radius":<14} {"Mean |λ|":<14} {"Top-3 |λ|"}')
    print('-' * 65)
    for c in range(15):
        if c not in eigenvalue_by_count:
            continue
        e = eigenvalue_by_count[c]
        top3_str = ', '.join(f'{v:.4f}' for v in e['top5_abs_eigenvalues'][:3])
        print(f'{c:<8} {e["upcoming_depth"]:<7} {e["spectral_radius"]:<14.4f} '
              f'{e["mean_abs_eigenvalue"]:<14.4f} {top3_str}')

    # Correlation with depth
    counts = [c for c in range(14) if c in eigenvalue_by_count]
    depths = [eigenvalue_by_count[c]['upcoming_depth'] for c in counts]
    srs = [eigenvalue_by_count[c]['spectral_radius'] for c in counts]
    r, p = stats.spearmanr(depths, srs)
    print(f'\n  Spearman correlation (depth vs spectral radius): r={r:.4f}, p={p:.4f}')

    return eigenvalue_by_count, r, p


# ─── Analysis 5: Temporal Dynamics at Counts 3, 7, 11 ────────────────────────

def analysis_5_temporal_dynamics(idle_by_count):
    """Quartile analysis of variance during idle periods at deep-cascade counts."""
    print('\n--- Analysis 5: Temporal Dynamics of Instability ---')

    target_counts = [3, 7, 11]
    temporal_results = {}

    for c in target_counts:
        segs = idle_by_count[c]
        if not segs:
            continue

        quartile_vars = [[] for _ in range(4)]

        for seg in segs:
            h = seg['h']
            if len(h) < 8:
                continue
            q_len = len(h) // 4
            for q in range(4):
                start = q * q_len
                end = start + q_len if q < 3 else len(h)
                segment = h[start:end]
                if len(segment) > 2:
                    quartile_vars[q].append(float(segment.var(axis=0).mean()))

        quartile_means = []
        for q in range(4):
            if quartile_vars[q]:
                quartile_means.append(float(np.mean(quartile_vars[q])))
            else:
                quartile_means.append(0.0)

        trend = 'increasing' if quartile_means[-1] > quartile_means[0] * 1.3 else \
                'decreasing' if quartile_means[-1] < quartile_means[0] * 0.7 else 'flat'

        temporal_results[c] = {
            'quartile_variance': quartile_means,
            'trend': trend,
            'n_segments': len(segs),
        }

        print(f'\n  Count {c} (depth={UPCOMING_DEPTH[c]}):')
        for q in range(4):
            print(f'    Q{q+1}: mean_var={quartile_means[q]:.6f} (n={len(quartile_vars[q])})')
        print(f'    Trend: {trend} (Q4/Q1 ratio: {quartile_means[3]/quartile_means[0]:.2f}x)'
              if quartile_means[0] > 0 else f'    Trend: {trend}')

    return temporal_results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_headline_figure(variance_by_count, ar1_by_count, recovery_by_count,
                          eigenvalue_by_count):
    """The headline figure: 4 panels showing all CSD indicators."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    counts = list(range(15))
    depth_colors = {-1: '#999999', 0: '#55A868', 1: '#4C72B0', 2: '#DD8452', 3: '#C44E52'}
    bar_colors = [depth_colors.get(UPCOMING_DEPTH.get(c, -1), '#999') for c in counts]

    # Panel 1: Variance
    ax = axes[0]
    vals = [variance_by_count.get(c, {}).get('mean_var', 0) for c in counts]
    ax.bar(counts, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean variance\n(per dimension)', fontsize=11)
    ax.set_title('Critical Slowing Down Indicators Across Count States',
                 fontsize=14, fontweight='bold')
    # Annotate depth
    for c in counts:
        d = UPCOMING_DEPTH.get(c, -1)
        if d >= 1:
            ax.text(c, vals[c] * 1.05, f'd={d}', ha='center', fontsize=8, fontweight='bold')

    # Panel 2: AR1
    ax = axes[1]
    vals = [ar1_by_count.get(c, {}).get('mean_ar1', 0) for c in counts]
    ax.bar(counts, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Lag-1 autocorrelation\n(AR1)', fontsize=11)

    # Panel 3: Recovery half-life
    ax = axes[2]
    vals = [recovery_by_count.get(c, {}).get('mean_half_life', 0) for c in counts]
    errs = [recovery_by_count.get(c, {}).get('std_half_life', 0) for c in counts]
    ax.bar(counts, vals, yerr=errs, color=bar_colors, edgecolor='white',
           linewidth=0.5, capsize=2)
    ax.set_ylabel('Perturbation recovery\nhalf-life (steps)', fontsize=11)

    # Panel 4: Spectral radius
    ax = axes[3]
    vals = [eigenvalue_by_count.get(c, {}).get('spectral_radius', 0) for c in counts]
    ax.bar(counts, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='Marginal stability (ρ=1)')
    ax.set_ylabel('Spectral radius\n(max |eigenvalue|)', fontsize=11)
    ax.set_xlabel('Count state', fontsize=11)
    ax.legend(fontsize=9)

    # Shared x-axis labels
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=depth_colors[0], label='Depth 0 (simple flip)'),
        Patch(facecolor=depth_colors[1], label='Depth 1'),
        Patch(facecolor=depth_colors[2], label='Depth 2'),
        Patch(facecolor=depth_colors[3], label='Depth 3 (full cascade)'),
        Patch(facecolor=depth_colors[-1], label='Count 14 (terminal)'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=9, loc='upper left', ncol=3)

    fig.tight_layout()
    path = FIGURES_DIR / 'csd_headline.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_eigenvalue_spectra(eigenvalue_by_count):
    """Eigenvalue spectra at count 6 (depth 0) vs count 7 (depth 3)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, c, label in [(axes[0], 6, 'Count 6 (next: depth-0 flip)'),
                          (axes[1], 7, 'Count 7 (next: depth-3 cascade)')]:
        if c not in eigenvalue_by_count:
            continue
        eigs_r = np.array(eigenvalue_by_count[c]['eigenvalues_real'])
        eigs_i = np.array(eigenvalue_by_count[c]['eigenvalues_imag'])

        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.5, alpha=0.3)

        # Eigenvalues
        sc = ax.scatter(eigs_r, eigs_i, s=8, alpha=0.6, c=np.abs(eigs_r + 1j*eigs_i),
                        cmap='coolwarm', vmin=0, vmax=1.2)

        sr = eigenvalue_by_count[c]['spectral_radius']
        ax.set_title(f'{label}\nSpectral radius = {sr:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Real', fontsize=10)
        ax.set_ylabel('Imaginary', fontsize=10)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.axvline(0, color='gray', linewidth=0.3)

        lim = max(1.3, sr * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle('Jacobian Eigenvalue Spectra: Near vs Far from Critical Transition',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / 'csd_eigenvalue_spectra.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_temporal_dynamics(temporal_results):
    """Quartile variance analysis for counts 3, 7, 11."""
    target_counts = [c for c in [3, 7, 11] if c in temporal_results]
    if not target_counts:
        return

    fig, axes = plt.subplots(1, len(target_counts), figsize=(5*len(target_counts), 5))
    if len(target_counts) == 1:
        axes = [axes]

    for ax, c in zip(axes, target_counts):
        qvars = temporal_results[c]['quartile_variance']
        x = np.arange(4)
        ax.bar(x, qvars, color=['#AECBEB', '#7FB3D8', '#4C72B0', '#2E4A7A'],
               edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(early)', 'Q2', 'Q3', 'Q4\n(late)'])
        ax.set_ylabel('Mean variance', fontsize=10)
        ax.set_title(f'Count {c} (upcoming depth={UPCOMING_DEPTH[c]})\n'
                     f'Trend: {temporal_results[c]["trend"]}',
                     fontsize=11, fontweight='bold')

    fig.suptitle('Does Instability Grow During Idle Period?', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / 'csd_temporal_dynamics.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 90)
    print('CRITICAL SLOWING DOWN ANALYSIS — Binary Specialist RSSM')
    print('=' * 90)

    # Load weights
    print('\nLoading RSSM weights...')
    rssm_weights = load_exported_weights(str(CKPT_DIR))
    print(f'  Loaded {len(rssm_weights)} tensors')

    # Collect episodes
    print('\nCollecting episodes...')
    episodes = collect_episodes(rssm_weights, n_episodes=15)

    # Extract idle segments
    print('\nExtracting idle segments...')
    idle_by_count = extract_idle_segments(episodes, trim=5)
    for c in range(15):
        n_segs = len(idle_by_count[c])
        n_steps = sum(len(s['h']) for s in idle_by_count[c])
        if n_segs > 0:
            print(f'  Count {c}: {n_segs} segments, {n_steps} total steps')

    # Run all 5 analyses
    var_results, var_r, var_p = analysis_1_variance(idle_by_count)
    ar1_results, ar1_r, ar1_p = analysis_2_ar1(idle_by_count)
    recovery_results, rec_r, rec_p = analysis_3_perturbation_recovery(
        episodes, rssm_weights, idle_by_count, n_perturbations=5, recovery_horizon=20)
    eig_results, eig_r, eig_p = analysis_4_eigenvalues(idle_by_count, rssm_weights)
    temporal_results = analysis_5_temporal_dynamics(idle_by_count)

    # ─── Correlation summary ──────────────────────────────────────────────

    print('\n' + '=' * 90)
    print('CORRELATION SUMMARY: Upcoming Cascade Depth vs CSD Indicators')
    print('=' * 90)
    print(f'\n{"Indicator":<30} {"Spearman r":<14} {"p-value":<14} {"Significant?"}')
    print('-' * 70)
    for name, r, p in [
        ('Idle variance', var_r, var_p),
        ('AR1', ar1_r, ar1_p),
        ('Recovery half-life', rec_r, rec_p),
        ('Spectral radius', eig_r, eig_p),
    ]:
        sig = 'YES ***' if p < 0.001 else 'YES **' if p < 0.01 else \
              'YES *' if p < 0.05 else 'marginal' if p < 0.1 else 'no'
        print(f'  {name:<28} {r:<14.4f} {p:<14.4f} {sig}')

    # ─── Generate figures ─────────────────────────────────────────────────

    print('\nGenerating figures...')
    plot_headline_figure(var_results, ar1_results, recovery_results, eig_results)
    plot_eigenvalue_spectra(eig_results)
    plot_temporal_dynamics(temporal_results)

    # ─── Save results ─────────────────────────────────────────────────────

    results = {
        'n_episodes': len(episodes),
        'perturbation_magnitude': 'calibrated to 10% of mean centroid distance',
        'correlations': {
            'variance': {'spearman_r': var_r, 'p_value': var_p},
            'ar1': {'spearman_r': ar1_r, 'p_value': ar1_p},
            'recovery_half_life': {'spearman_r': rec_r, 'p_value': rec_p},
            'spectral_radius': {'spearman_r': eig_r, 'p_value': eig_p},
        },
        'per_count': {},
        'temporal_dynamics': {str(k): v for k, v in temporal_results.items()},
    }

    for c in range(15):
        row = {'count': c, 'upcoming_depth': UPCOMING_DEPTH.get(c, -1)}
        if c in var_results:
            row['mean_variance'] = var_results[c]['mean_var']
        if c in ar1_results:
            row['ar1'] = ar1_results[c]['mean_ar1']
        if c in recovery_results:
            row['recovery_half_life'] = recovery_results[c]['mean_half_life']
        if c in eig_results:
            row['spectral_radius'] = eig_results[c]['spectral_radius']
            row['mean_abs_eigenvalue'] = eig_results[c]['mean_abs_eigenvalue']
        results['per_count'][str(c)] = row

    # Don't save full eigenvalue arrays (too large)
    eig_summary = {}
    for c in range(15):
        if c in eig_results:
            eig_summary[str(c)] = {
                'spectral_radius': eig_results[c]['spectral_radius'],
                'top5_abs_eigenvalues': eig_results[c]['top5_abs_eigenvalues'],
            }
    results['eigenvalue_summary'] = eig_summary

    out_path = ARTIFACTS_DIR / 'critical_slowing_down.json'
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
