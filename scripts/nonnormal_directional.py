#!/usr/bin/env python3
"""Non-Normal Amplification + Directional Variance — Two Targeted Tests.

Test 1: Henrici departure from normality at each count centroid.
  If Henrici number correlates with cascade depth, non-normal amplification
  explains the variance-only CSD signal.

Test 2: Directional variance — does wobble follow the upcoming transition?
  Projects idle-period variance onto bit-probe directions to test whether
  variance is directional (along bits about to flip) or isotropic.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import schur
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
from critical_slowing_down import gru_transition_fn, compute_jacobian

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLUMNS = 4

# Upcoming cascade depth and which bits flip for each count
UPCOMING_DEPTH = {c: carry_depth(c) for c in range(14)}
UPCOMING_DEPTH[14] = -1

def bits_that_flip(count):
    """Return list of bit indices that flip when going from count to count+1."""
    if count >= 14:
        return []
    flipping = []
    for b in range(NUM_COLUMNS):
        if (count >> b) & 1 == 1:
            flipping.append(b)  # this bit resets (1→0 via carry)
        else:
            flipping.append(b)  # this bit flips on
            break  # carry stops here
    return flipping


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_idle_data(rssm_weights, n_episodes=15, max_steps=1000, trim=5):
    """Collect episodes and extract idle segments per count."""
    print(f'  Collecting {n_episodes} episodes...', flush=True)

    all_h, all_dc = [], []
    idle_by_count = {c: [] for c in range(15)}

    for ep in range(n_episodes):
        rssm = FastRSSMWithImagination(rssm_weights)
        env = BinaryCountingEnv(seed=42 + ep)
        obs = env.reset()
        rssm.reset()

        ep_h, ep_dc, ep_stoch = [], [], []

        for t in range(max_steps):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            deter = rssm.step(obs_vec, action=0)
            ep_h.append(deter)
            ep_dc.append(env._state.decimal_count)
            ep_stoch.append(rssm.stoch.copy())

            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']
            if env._state.decimal_count >= 14:
                break

        ep_h = np.array(ep_h)
        ep_dc = np.array(ep_dc)
        ep_stoch = np.array(ep_stoch)
        all_h.append(ep_h)
        all_dc.append(ep_dc)

        # Extract idle segments
        seg_start = 0
        for t in range(1, len(ep_dc)):
            if ep_dc[t] != ep_dc[t-1] or t == len(ep_dc) - 1:
                seg_end = t if ep_dc[t] != ep_dc[t-1] else t + 1
                count_val = ep_dc[seg_start]
                seg_len = seg_end - seg_start
                if seg_len > 2 * trim + 5:
                    ts = seg_start + trim
                    te = seg_end - trim
                    idle_by_count[count_val].append({
                        'h': ep_h[ts:te],
                        'stoch': ep_stoch[ts:te],
                    })
                seg_start = t

        if (ep + 1) % 5 == 0:
            print(f'    Episode {ep+1}/{n_episodes}: {len(ep_h)} steps', flush=True)

    all_h = np.concatenate(all_h)
    all_dc = np.concatenate(all_dc)
    return idle_by_count, all_h, all_dc


def train_bit_probes(battery_path, all_h, all_dc):
    """Train Ridge probes for each bit using battery data, return weight vectors."""
    print('  Training bit probes...', flush=True)

    data = np.load(battery_path)
    bat_h = data['h_t']
    bat_bits = data['bits']  # (T, 4)

    probes = []
    for b in range(NUM_COLUMNS):
        clf = Ridge(alpha=1.0)
        clf.fit(bat_h, bat_bits[:, b])
        acc = (clf.predict(bat_h).round() == bat_bits[:, b]).mean()
        probes.append(clf)
        print(f'    Bit {b}: acc={acc:.4f}', flush=True)

    # Normalize weight vectors to unit length for direction computation
    probe_directions = []
    for clf in probes:
        w = clf.coef_.copy()
        w_norm = w / np.linalg.norm(w)
        probe_directions.append(w_norm)

    return probes, np.array(probe_directions)


# ─── Test 1: Henrici Number ─────────────────────────────────────────────────

def henrici_number(A):
    """Compute Henrici departure from normality.

    For matrix A: H = sqrt(||A||_F^2 - sum(|eigenvalues|^2)) / ||A||_F
    Equivalently: measures the Frobenius norm of the off-diagonal part
    of the Schur decomposition.
    """
    T, Q = schur(A, output='complex')
    norm_A = np.linalg.norm(A, 'fro')
    eig_sum = np.sum(np.abs(np.diag(T))**2)
    return np.sqrt(max(0, norm_A**2 - eig_sum)) / max(norm_A, 1e-10)


def test_1_henrici(idle_by_count, rssm_weights):
    """Compute Henrici number at each count centroid's Jacobian."""
    print('\n' + '=' * 70)
    print('TEST 1: Henrici Departure from Normality')
    print('=' * 70, flush=True)

    rssm_temp = FastRSSMWithImagination(rssm_weights)
    num_actions = rssm_temp.num_actions

    results = {}

    for c in range(15):
        segs = idle_by_count[c]
        if not segs:
            continue

        all_h = np.concatenate([s['h'] for s in segs], axis=0)
        all_stoch = np.concatenate([s['stoch'] for s in segs], axis=0)
        centroid = all_h.mean(axis=0).astype(np.float32)
        median_stoch = all_stoch[len(all_stoch)//2].copy()

        print(f'  Count {c}: computing Jacobian...', end='', flush=True)
        J = compute_jacobian(centroid, median_stoch, 0, rssm_weights, num_actions)

        eigs = np.linalg.eigvals(J)
        spec_radius = float(np.max(np.abs(eigs)))
        h_num = henrici_number(J)

        depth = UPCOMING_DEPTH.get(c, -1)
        results[c] = {
            'upcoming_depth': depth,
            'spectral_radius': spec_radius,
            'henrici_number': float(h_num),
        }
        print(f' spec_rad={spec_radius:.4f}, henrici={h_num:.4f}')

    # Correlation
    counts = [c for c in range(14) if c in results]
    depths = [results[c]['upcoming_depth'] for c in counts]
    henricis = [results[c]['henrici_number'] for c in counts]
    spec_rads = [results[c]['spectral_radius'] for c in counts]

    r_depth, p_depth = stats.spearmanr(depths, henricis)
    r_spec, p_spec = stats.spearmanr(spec_rads, henricis)

    print(f'\n  Spearman (depth vs Henrici): r={r_depth:.4f}, p={p_depth:.4f}')
    print(f'  Spearman (spec_radius vs Henrici): r={r_spec:.4f}, p={p_spec:.4f}')

    # Table
    print(f'\n  {"Count":<8} {"Depth":<8} {"Spec Rad":<12} {"Henrici":<12}')
    print('  ' + '-' * 40)
    for c in range(15):
        if c not in results:
            continue
        r = results[c]
        print(f'  {c:<8} {r["upcoming_depth"]:<8} {r["spectral_radius"]:<12.4f} '
              f'{r["henrici_number"]:<12.4f}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Henrici vs depth
    depths_plot = [results[c]['upcoming_depth'] for c in counts]
    henricis_plot = [results[c]['henrici_number'] for c in counts]
    colors = ['#F44336' if d >= 2 else '#FF9800' if d == 1 else '#2196F3'
              for d in depths_plot]

    axes[0].scatter(depths_plot, henricis_plot, c=colors, s=80, zorder=3)
    for c in counts:
        axes[0].annotate(str(c), (results[c]['upcoming_depth'],
                         results[c]['henrici_number']),
                         fontsize=7, ha='center', va='bottom')
    axes[0].set_xlabel('Upcoming Cascade Depth')
    axes[0].set_ylabel('Henrici Number')
    axes[0].set_title(f'Non-Normality vs Cascade Depth\n(r={r_depth:.3f}, p={p_depth:.4f})')

    # Henrici bar chart by count (matching CSD variance format)
    all_counts = [c for c in range(15) if c in results]
    all_henricis = [results[c]['henrici_number'] for c in all_counts]
    all_depths = [results[c]['upcoming_depth'] for c in all_counts]
    bar_colors = ['#F44336' if d >= 2 else '#FF9800' if d == 1 else '#2196F3'
                  if d == 0 else '#999' for d in all_depths]

    axes[1].bar(all_counts, all_henricis, color=bar_colors)
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Henrici Number')
    axes[1].set_title('Non-Normality by Count')

    fig.suptitle('Test 1: Henrici Departure from Normality', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nonnormal_henrici.png', dpi=150)
    plt.close()
    print(f'  Saved {FIGURES_DIR / "nonnormal_henrici.png"}', flush=True)

    return results, r_depth, p_depth


# ─── Test 2: Directional Variance ───────────────────────────────────────────

def test_2_directional_variance(idle_by_count, probe_directions):
    """Test whether idle-period variance is directional along upcoming-flip bits."""
    print('\n' + '=' * 70)
    print('TEST 2: Directional Variance — Does Wobble Follow the Transition?')
    print('=' * 70, flush=True)

    results = {}

    for c in range(14):
        segs = idle_by_count[c]
        if not segs:
            continue

        all_h = np.concatenate([s['h'] for s in segs], axis=0)
        if len(all_h) < 10:
            continue

        # Center the data
        h_centered = all_h - all_h.mean(axis=0)

        # Which bits flip?
        flipping = bits_that_flip(c)
        not_flipping = [b for b in range(NUM_COLUMNS) if b not in flipping]

        # Variance along each bit-probe direction
        per_bit_variance = {}
        for b in range(NUM_COLUMNS):
            proj = h_centered @ probe_directions[b]  # project onto bit direction
            per_bit_variance[b] = float(proj.var())

        # Relevant variance: mean over flipping bits
        relevant_var = np.mean([per_bit_variance[b] for b in flipping])
        # Irrelevant variance: mean over non-flipping bits (if any)
        if not_flipping:
            irrelevant_var = np.mean([per_bit_variance[b] for b in not_flipping])
        else:
            # All bits flip (count 7, 15) — use random orthogonal directions
            rng = np.random.RandomState(c)
            random_dirs = []
            for _ in range(4):
                d = rng.randn(DETER_DIM).astype(np.float64)
                # Orthogonalize against all probe directions
                for pd in probe_directions:
                    d -= np.dot(d, pd) * pd
                d /= np.linalg.norm(d)
                random_dirs.append(d)
            irrelevant_var = np.mean([(h_centered @ d).var() for d in random_dirs])

        ratio = relevant_var / max(irrelevant_var, 1e-10)

        depth = UPCOMING_DEPTH[c]
        results[c] = {
            'upcoming_depth': depth,
            'bits_flipping': flipping,
            'bits_not_flipping': not_flipping,
            'per_bit_variance': per_bit_variance,
            'relevant_variance': float(relevant_var),
            'irrelevant_variance': float(irrelevant_var),
            'ratio': float(ratio),
            'n_steps': len(all_h),
        }

    # Print main table
    print(f'\n  {"Count":<8} {"Depth":<8} {"Bits flip":<14} '
          f'{"Relev var":<12} {"Irrelev var":<12} {"Ratio":<8}')
    print('  ' + '-' * 62)
    for c in range(14):
        if c not in results:
            continue
        r = results[c]
        bits_str = ','.join(str(b) for b in r['bits_flipping'])
        print(f'  {c:<8} {r["upcoming_depth"]:<8} {bits_str:<14} '
              f'{r["relevant_variance"]:<12.6f} {r["irrelevant_variance"]:<12.6f} '
              f'{r["ratio"]:<8.3f}')

    # Correlation: ratio vs depth
    counts = [c for c in range(14) if c in results]
    depths = [results[c]['upcoming_depth'] for c in counts]
    ratios = [results[c]['ratio'] for c in counts]
    r_corr, p_corr = stats.spearmanr(depths, ratios)
    print(f'\n  Spearman (depth vs variance ratio): r={r_corr:.4f}, p={p_corr:.4f}')

    # Per-bit breakdown for key counts
    for target_count in [3, 7, 6]:
        if target_count not in results:
            continue
        r = results[target_count]
        print(f'\n  Count {target_count} (depth={r["upcoming_depth"]}) per-bit breakdown:')
        for b in range(NUM_COLUMNS):
            flips = 'YES' if b in r['bits_flipping'] else 'no'
            print(f'    Bit {b}: variance={r["per_bit_variance"][b]:.6f}  '
                  f'(flips={flips})')

    # ─── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Variance ratio vs depth
    colors = ['#F44336' if d >= 2 else '#FF9800' if d == 1 else '#2196F3'
              for d in depths]
    axes[0].scatter(depths, ratios, c=colors, s=80, zorder=3)
    for c in counts:
        axes[0].annotate(str(c), (results[c]['upcoming_depth'], results[c]['ratio']),
                         fontsize=7, ha='center', va='bottom')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                    label='Isotropic (ratio=1)')
    axes[0].set_xlabel('Upcoming Cascade Depth')
    axes[0].set_ylabel('Relevant / Irrelevant Variance Ratio')
    axes[0].set_title(f'Directional Variance vs Depth\n(r={r_corr:.3f}, p={p_corr:.4f})')
    axes[0].legend(fontsize=8)

    # Panel 2: Per-bit variance at counts 3, 7, and control (6)
    target_counts = [6, 3, 7]
    target_labels = ['Count 6\n(depth=0)', 'Count 3\n(depth=2)', 'Count 7\n(depth=3)']
    x = np.arange(NUM_COLUMNS)
    width = 0.25
    for idx, (tc, label) in enumerate(zip(target_counts, target_labels)):
        if tc not in results:
            continue
        r = results[tc]
        vars_per_bit = [r['per_bit_variance'][b] for b in range(NUM_COLUMNS)]
        flips = r['bits_flipping']
        bar_colors = ['#F44336' if b in flips else '#2196F3' for b in range(NUM_COLUMNS)]
        bars = axes[1].bar(x + idx * width, vars_per_bit, width,
                           label=label, alpha=0.8)
        for bar, b in zip(bars, range(NUM_COLUMNS)):
            bar.set_color('#F44336' if b in flips else '#2196F3')

    axes[1].set_xlabel('Bit Index')
    axes[1].set_ylabel('Variance along bit-probe direction')
    axes[1].set_title('Per-Bit Variance\n(red=flips in upcoming transition)')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f'Bit {b}' for b in range(NUM_COLUMNS)])
    axes[1].legend(fontsize=8)

    # Panel 3: Count 3 detailed (the key test)
    if 3 in results:
        r3 = results[3]
        bit_vars = [r3['per_bit_variance'][b] for b in range(NUM_COLUMNS)]
        bit_colors = ['#F44336' if b in r3['bits_flipping'] else '#2196F3'
                      for b in range(NUM_COLUMNS)]
        axes[2].bar(range(NUM_COLUMNS), bit_vars, color=bit_colors)
        axes[2].set_xlabel('Bit Index')
        axes[2].set_ylabel('Idle-Period Variance')
        axes[2].set_title('Count 3: Bits 0,1,2 flip → bit 3 stays\n'
                          '(Key test: bit 3 should be LOW)')
        axes[2].set_xticks(range(NUM_COLUMNS))
        axes[2].set_xticklabels([f'Bit {b}\n({"flips" if b in r3["bits_flipping"] else "stays"})'
                                  for b in range(NUM_COLUMNS)])

    fig.suptitle('Test 2: Directional Variance — Wobble Follows the Transition', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nonnormal_directional.png', dpi=150)
    plt.close()
    print(f'  Saved {FIGURES_DIR / "nonnormal_directional.png"}', flush=True)

    return results, r_corr, p_corr


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('NON-NORMAL AMPLIFICATION + DIRECTIONAL VARIANCE')
    print('=' * 80)

    print('\nLoading RSSM weights...')
    rssm_weights = load_exported_weights(CKPT_DIR)
    print(f'  Loaded {len(rssm_weights)} tensors')

    # Collect data
    print('\nCollecting data...')
    idle_by_count, all_h, all_dc = collect_idle_data(rssm_weights, n_episodes=15)

    # Train bit probes
    probes, probe_directions = train_bit_probes(BATTERY_PATH, all_h, all_dc)

    # Test 2 first (priority, fast)
    dir_results, dir_r, dir_p = test_2_directional_variance(idle_by_count, probe_directions)

    # Test 1 (Jacobians needed — slower)
    henrici_results, hen_r, hen_p = test_1_henrici(idle_by_count, rssm_weights)

    # ─── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'\nTest 1 — Henrici (non-normality vs depth):')
    print(f'  Spearman r = {hen_r:.4f}, p = {hen_p:.4f}')
    if abs(hen_r) > 0.5 and hen_p < 0.05:
        print('  → CONFIRMED: Non-normal amplification correlates with cascade depth')
    else:
        print('  → NOT confirmed: Non-normality does not predict cascade depth')

    print(f'\nTest 2 — Directional variance (ratio vs depth):')
    print(f'  Spearman r = {dir_r:.4f}, p = {dir_p:.4f}')
    if dir_r > 0.3 and dir_p < 0.1:
        print('  → CONFIRMED: Variance is directional along upcoming-flip bits')
    else:
        print('  → NOT confirmed: Variance is isotropic or depth-independent')

    # Count 3 key test
    if 3 in dir_results:
        r3 = dir_results[3]
        flip_vars = [r3['per_bit_variance'][b] for b in r3['bits_flipping']]
        stay_vars = [r3['per_bit_variance'][b] for b in r3['bits_not_flipping']]
        if stay_vars:
            mean_flip = np.mean(flip_vars)
            mean_stay = np.mean(stay_vars)
            print(f'\n  Count 3 key test:')
            print(f'    Flipping bits (0,1,2) mean variance: {mean_flip:.6f}')
            print(f'    Staying bit (3) variance: {mean_stay:.6f}')
            print(f'    Ratio: {mean_flip/max(mean_stay, 1e-10):.2f}x')
            if mean_flip > 2 * mean_stay:
                print('    → CLEAN: Wobble is specifically along bits about to change')
            elif mean_flip > mean_stay:
                print('    → Modest directional signal')
            else:
                print('    → No directional signal')

    # Save
    save_data = {
        'henrici': {str(k): v for k, v in henrici_results.items()},
        'henrici_correlation': {'r': float(hen_r), 'p': float(hen_p)},
        'directional': {str(k): {kk: (vv if not isinstance(vv, dict) else
                                       {str(kkk): vvv for kkk, vvv in vv.items()})
                                  for kk, vv in v.items()}
                        for k, v in dir_results.items()},
        'directional_correlation': {'r': float(dir_r), 'p': float(dir_p)},
    }

    out_path = ARTIFACTS_DIR / 'nonnormal_directional.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
