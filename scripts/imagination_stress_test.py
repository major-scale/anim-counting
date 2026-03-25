#!/usr/bin/env python3
"""Task B: Extended Imagination Stress Test.

Four experiments testing the limits of the RSSM's imagination (prior-only)
mode for binary counting:

1. Multi-start imagination — correct transitions from various starting points
2. Degradation tracking — bit accuracy vs transition number from count=0
3. Periodic peeks — observation refreshes every N steps or at transitions
4. Error characterization — how and where imagination fails

All probes use column-state calibration (matching carry propagation methodology).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ─── Path setup ───────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = Path('/workspace/projects/jamstack-v1/bridge')
sys.path.insert(0, str(BRIDGE_DIR / 'scripts'))

from binary_counting_env import BinaryCountingEnv, OBS_SIZE
from imagination_rollout_binary import (
    FastRSSMWithImagination, load_exported_weights, carry_depth
)

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLUMNS = 4


# ─── Probe training ─────────────────────────────────────────────────────────

def train_colstate_probes():
    """Train Ridge probes on actual column states from battery.npz."""
    data = np.load(str(BATTERY_PATH), allow_pickle=True)
    h_t = data['h_t']
    bits = data['bits']
    print(f'  Battery: {h_t.shape[0]} timesteps')

    probes = []
    bit_means = {'off': np.zeros(4), 'on': np.zeros(4)}
    for b in range(4):
        p = Ridge(alpha=1.0)
        p.fit(h_t, bits[:, b])
        probes.append(p)
        preds = p.predict(h_t)
        bit_means['off'][b] = preds[bits[:, b] == 0].mean()
        bit_means['on'][b] = preds[bits[:, b] == 1].mean()
        acc = (np.round(preds).astype(int) == bits[:, b]).mean()
        print(f'  Bit {b}: acc={acc:.4f}')

    return probes, bit_means


def project_and_normalize(h_arr, probes, bit_means):
    """Project hidden states through probes, normalize to [0,1]."""
    proj = np.zeros((len(h_arr), 4))
    for b in range(4):
        preds = probes[b].predict(h_arr)
        off_val = bit_means['off'][b]
        on_val = bit_means['on'][b]
        rng = on_val - off_val
        if abs(rng) > 1e-8:
            proj[:, b] = (preds - off_val) / rng
        else:
            proj[:, b] = preds
    return proj


def decode_count_from_proj(proj_row):
    """Decode a count from normalized bit-probe activations."""
    bits = (proj_row > 0.5).astype(int)
    return sum(bits[b] * (2**b) for b in range(4))


def bits_from_count(c):
    """Get 4-bit array from count."""
    return np.array([(c >> b) & 1 for b in range(4)])


# ─── Helper: run posterior up to a target count ──────────────────────────────

def run_posterior_to_count(rssm, env, obs, target_count, max_steps=1200):
    """Run posterior until env reaches target_count, return (state, h_history, col_history)."""
    h_history = []
    col_history = []

    for t in range(max_steps):
        obs_vec = obs[:OBS_SIZE].astype(np.float32)
        deter = rssm.step(obs_vec, action=0)

        col_bits = [1 if env._state.columns[i].occupied else 0
                    for i in range(NUM_COLUMNS)]
        dc = env._state.decimal_count

        h_history.append(deter)
        col_history.append(col_bits)

        if dc >= target_count:
            return rssm.get_state(), np.array(h_history), np.array(col_history), t

        result = env.step(0)
        obs = result[0] if isinstance(result, tuple) else result['obs']

    return rssm.get_state(), np.array(h_history), np.array(col_history), max_steps


# ─── Experiment 1: Multi-Start Imagination ───────────────────────────────────

def experiment_1_multi_start(rssm_weights, probes, bit_means, n_seeds=10):
    """Fork imagination at various starting counts, measure correct transitions."""
    print('\n--- Experiment 1: Multi-Start Imagination ---')

    start_counts = [0, 3, 5, 7, 10]
    imagination_horizon = 800  # Enough for ~14 transitions

    results = {}

    for start_c in start_counts:
        print(f'\n  Starting from count={start_c}...')
        correct_transitions_list = []
        total_transitions_list = []

        for seed in range(n_seeds):
            rssm = FastRSSMWithImagination(rssm_weights)
            env = BinaryCountingEnv(seed=seed + 100)
            obs = env.reset()
            rssm.reset()

            # Run posterior to target count
            if start_c > 0:
                state, _, _, _ = run_posterior_to_count(rssm, env, obs, start_c)
            else:
                # Just after reset, run a few posterior steps to initialize
                for warmup in range(5):
                    obs_vec = obs[:OBS_SIZE].astype(np.float32)
                    rssm.step(obs_vec, action=0)
                    result = env.step(0)
                    obs = result[0] if isinstance(result, tuple) else result['obs']

            # Save state and fork to imagination
            saved_state = rssm.get_state()
            fork_h = rssm.deter.copy()

            # Run imagination
            rssm.set_state(saved_state)
            imag_h = [fork_h.copy()]
            for s in range(imagination_horizon):
                d = rssm.imagine_step(action=0)
                imag_h.append(d)
            imag_h = np.array(imag_h)

            # Project and decode
            proj = project_and_normalize(imag_h, probes, bit_means)

            # Detect transitions in imagination
            prev_count = decode_count_from_proj(proj[0])
            correct = 0
            total = 0
            expected_next = start_c + 1

            for t in range(1, len(proj)):
                cur_count = decode_count_from_proj(proj[t])
                if cur_count != prev_count:
                    total += 1
                    if cur_count == expected_next and cur_count <= 14:
                        correct += 1
                        expected_next = cur_count + 1
                    prev_count = cur_count

            correct_transitions_list.append(correct)
            total_transitions_list.append(total)

        mean_correct = np.mean(correct_transitions_list)
        mean_total = np.mean(total_transitions_list)
        max_possible = 14 - start_c

        print(f'    Correct transitions: {mean_correct:.1f} ± {np.std(correct_transitions_list):.1f} '
              f'(max possible: {max_possible})')
        print(f'    Total detected: {mean_total:.1f} ± {np.std(total_transitions_list):.1f}')

        results[start_c] = {
            'start_count': start_c,
            'max_possible': max_possible,
            'mean_correct': float(mean_correct),
            'std_correct': float(np.std(correct_transitions_list)),
            'mean_total_detected': float(mean_total),
            'per_seed_correct': [int(x) for x in correct_transitions_list],
        }

    # Summary table
    print(f'\n{"Start":<8} {"Max":<6} {"Correct":<14} {"Detected":<14} {"Accuracy"}')
    print('-' * 55)
    for sc in start_counts:
        r = results[sc]
        acc = r['mean_correct'] / r['max_possible'] * 100 if r['max_possible'] > 0 else 0
        print(f'{sc:<8} {r["max_possible"]:<6} '
              f'{r["mean_correct"]:.1f}±{r["std_correct"]:.1f}{"":>4} '
              f'{r["mean_total_detected"]:.1f}{"":>8} '
              f'{acc:.0f}%')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(start_counts))
    correct = [results[sc]['mean_correct'] for sc in start_counts]
    max_pos = [results[sc]['max_possible'] for sc in start_counts]
    errs = [results[sc]['std_correct'] for sc in start_counts]

    ax.bar(x - 0.15, max_pos, 0.3, label='Max possible', color='#CCCCCC', edgecolor='white')
    ax.bar(x + 0.15, correct, 0.3, yerr=errs, label='Correct', color='#4C72B0',
           edgecolor='white', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Count={sc}' for sc in start_counts], fontsize=10)
    ax.set_ylabel('Number of correct transitions', fontsize=11)
    ax.set_title('Imagination: Correct Transitions from Various Starting Points',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = FIGURES_DIR / 'stress_multi_start.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return results


# ─── Experiment 2: Degradation Tracking ──────────────────────────────────────

def experiment_2_degradation(rssm_weights, probes, bit_means, n_seeds=10):
    """Track bit accuracy vs imagination step from count=0."""
    print('\n--- Experiment 2: Degradation Tracking ---')

    imagination_horizon = 900
    all_proj = []
    all_ground_truth = []

    for seed in range(n_seeds):
        rssm = FastRSSMWithImagination(rssm_weights)
        env = BinaryCountingEnv(seed=seed + 200)
        obs = env.reset()
        rssm.reset()

        # Collect ground truth from posterior run
        gt_h = []
        gt_colbits = []
        gt_dc = []
        for t in range(imagination_horizon + 50):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            deter = rssm.step(obs_vec, action=0)
            col_bits = [1 if env._state.columns[i].occupied else 0
                        for i in range(NUM_COLUMNS)]
            gt_h.append(deter)
            gt_colbits.append(col_bits)
            gt_dc.append(env._state.decimal_count)

            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']
            if env._state.decimal_count >= 14:
                for _ in range(30):
                    obs_vec = obs[:OBS_SIZE].astype(np.float32)
                    deter = rssm.step(obs_vec, action=0)
                    col_bits = [1 if env._state.columns[i].occupied else 0
                                for i in range(NUM_COLUMNS)]
                    gt_h.append(deter)
                    gt_colbits.append(col_bits)
                    gt_dc.append(env._state.decimal_count)
                    result = env.step(0)
                    obs = result[0] if isinstance(result, tuple) else result['obs']
                break

        gt_colbits = np.array(gt_colbits)

        # Now fork at step 5 (after warmup) and run imagination
        rssm2 = FastRSSMWithImagination(rssm_weights)
        env2 = BinaryCountingEnv(seed=seed + 200)
        obs2 = env2.reset()
        rssm2.reset()

        # Warmup with 5 posterior steps
        for warmup in range(5):
            obs_vec = obs2[:OBS_SIZE].astype(np.float32)
            rssm2.step(obs_vec, action=0)
            result = env2.step(0)
            obs2 = result[0] if isinstance(result, tuple) else result['obs']

        # Fork to imagination
        imag_h = []
        for s in range(min(imagination_horizon, len(gt_colbits) - 5)):
            d = rssm2.imagine_step(action=0)
            imag_h.append(d)
        imag_h = np.array(imag_h)

        # Project imagination through probes
        imag_proj = project_and_normalize(imag_h, probes, bit_means)

        # Ground truth from step 5 onwards
        gt_for_comparison = gt_colbits[5:5 + len(imag_proj)]

        all_proj.append(imag_proj[:len(gt_for_comparison)])
        all_ground_truth.append(gt_for_comparison.astype(float))

    # Compute per-step bit accuracy
    min_len = min(len(p) for p in all_proj)
    proj_arr = np.array([p[:min_len] for p in all_proj])  # (n_seeds, T, 4)
    gt_arr = np.array([g[:min_len] for g in all_ground_truth])  # (n_seeds, T, 4)

    # Bit accuracy: |round(proj) - gt| for each bit
    decoded_bits = (proj_arr > 0.5).astype(float)
    per_bit_acc = (decoded_bits == gt_arr).mean(axis=0)  # (T, 4)
    all_bit_acc = per_bit_acc.mean(axis=1)  # (T,)

    # Count accuracy: all 4 bits correct
    count_correct = (decoded_bits == gt_arr).all(axis=2).mean(axis=0)  # (T,)

    # Decoded count vs true count
    decoded_counts = np.array([
        [sum(decoded_bits[s, t, b] * (2**b) for b in range(4))
         for t in range(min_len)]
        for s in range(n_seeds)
    ])
    true_counts = np.array([
        [sum(gt_arr[s, t, b] * (2**b) for b in range(4))
         for t in range(min_len)]
        for s in range(n_seeds)
    ])
    count_exact_acc = (decoded_counts == true_counts).mean(axis=0)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    bit_labels = ['Bit0 (1s)', 'Bit1 (2s)', 'Bit2 (4s)', 'Bit3 (8s)']

    # Per-bit accuracy
    ax = axes[0]
    for b in range(4):
        # Smooth with rolling window
        window = 20
        smoothed = np.convolve(per_bit_acc[:, b], np.ones(window)/window, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, color=colors[b], linewidth=1.5,
                label=bit_labels[b])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Bit accuracy (rolling avg)', fontsize=11)
    ax.set_title('Imagination Degradation: Bit Accuracy vs Steps (from count=0)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Count exact accuracy
    ax2 = axes[1]
    window = 20
    smoothed_count = np.convolve(count_exact_acc, np.ones(window)/window, mode='valid')
    ax2.plot(range(len(smoothed_count)), smoothed_count, color='#333', linewidth=1.5)
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax2.set_ylabel('Exact count accuracy\n(all 4 bits correct)', fontsize=11)
    ax2.set_ylim(0, 1.05)

    # Mean absolute count error
    ax3 = axes[2]
    count_error = np.abs(decoded_counts - true_counts).mean(axis=0)
    smoothed_err = np.convolve(count_error, np.ones(window)/window, mode='valid')
    ax3.plot(range(len(smoothed_err)), smoothed_err, color='#C44E52', linewidth=1.5)
    ax3.set_ylabel('Mean |count error|', fontsize=11)
    ax3.set_xlabel('Imagination step (from count=0)', fontsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / 'stress_degradation.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    # Find where accuracy drops below thresholds
    thresholds = [0.9, 0.8, 0.5]
    for thresh in thresholds:
        below = np.where(count_exact_acc < thresh)[0]
        if len(below) > 0:
            print(f'  Count accuracy drops below {thresh*100:.0f}% at step {below[0]}')
        else:
            print(f'  Count accuracy stays above {thresh*100:.0f}% for all {min_len} steps')

    return {
        'min_len': int(min_len),
        'final_bit_accuracy': per_bit_acc[-1].tolist(),
        'final_count_accuracy': float(count_exact_acc[-1]),
        'n_seeds': n_seeds,
    }


# ─── Experiment 3: Periodic Peeks ───────────────────────────────────────────

def experiment_3_periodic_peeks(rssm_weights, probes, bit_means, n_seeds=10):
    """Test observation refreshes at various intervals."""
    print('\n--- Experiment 3: Periodic Peeks ---')

    peek_intervals = [0, 10, 25, 50, 100, 'transitions']  # 0 = pure posterior (baseline)
    max_steps = 800

    results = {}

    for interval in peek_intervals:
        interval_label = str(interval) if isinstance(interval, int) else interval
        print(f'\n  Peek interval: {interval_label}...')

        all_correct_counts = []
        all_total_transitions = []

        for seed in range(n_seeds):
            rssm = FastRSSMWithImagination(rssm_weights)
            env = BinaryCountingEnv(seed=seed + 300)
            obs = env.reset()
            rssm.reset()

            h_list = []
            dc_list = []
            steps_since_peek = 0
            prev_count = 0
            last_peek_count = 0

            for t in range(max_steps):
                obs_vec = obs[:OBS_SIZE].astype(np.float32)
                cur_count = env._state.decimal_count

                if interval == 0:
                    # Pure posterior — always use observation
                    deter = rssm.step(obs_vec, action=0)
                elif interval == 'transitions':
                    # Peek only at transitions
                    if cur_count != last_peek_count or t == 0:
                        deter = rssm.step(obs_vec, action=0)
                        last_peek_count = cur_count
                        steps_since_peek = 0
                    else:
                        deter = rssm.imagine_step(action=0)
                        steps_since_peek += 1
                else:
                    # Periodic peeks
                    if steps_since_peek >= interval or t == 0:
                        deter = rssm.step(obs_vec, action=0)
                        steps_since_peek = 0
                    else:
                        deter = rssm.imagine_step(action=0)
                        steps_since_peek += 1

                h_list.append(deter)
                dc_list.append(cur_count)

                result = env.step(0)
                obs = result[0] if isinstance(result, tuple) else result['obs']
                prev_count = cur_count

                if env._state.decimal_count >= 14:
                    break

            # Evaluate: decode counts from probes
            h_arr = np.array(h_list)
            proj = project_and_normalize(h_arr, probes, bit_means)
            dc_arr = np.array(dc_list)

            decoded = np.array([decode_count_from_proj(proj[t]) for t in range(len(proj))])
            correct = (decoded == dc_arr).sum()
            all_correct_counts.append(correct / len(dc_arr))

            # Count correct transitions
            gt_transitions = 0
            decoded_correct_transitions = 0
            for t in range(1, len(dc_arr)):
                if dc_arr[t] != dc_arr[t-1]:
                    gt_transitions += 1
                    if decoded[t] == dc_arr[t]:
                        decoded_correct_transitions += 1
            all_total_transitions.append(decoded_correct_transitions / max(gt_transitions, 1))

        mean_acc = np.mean(all_correct_counts)
        mean_trans = np.mean(all_total_transitions)
        print(f'    Step accuracy: {mean_acc:.3f} ± {np.std(all_correct_counts):.3f}')
        print(f'    Transition accuracy: {mean_trans:.3f} ± {np.std(all_total_transitions):.3f}')

        results[interval_label] = {
            'interval': interval_label,
            'mean_step_accuracy': float(mean_acc),
            'std_step_accuracy': float(np.std(all_correct_counts)),
            'mean_transition_accuracy': float(mean_trans),
            'std_transition_accuracy': float(np.std(all_total_transitions)),
            'n_seeds': n_seeds,
        }

    # Summary table
    print(f'\n{"Interval":<14} {"Step Acc":<14} {"Trans Acc":<14}')
    print('-' * 42)
    for key in ['0', '10', '25', '50', '100', 'transitions']:
        if key in results:
            r = results[key]
            print(f'{key:<14} {r["mean_step_accuracy"]:.3f}±{r["std_step_accuracy"]:.3f}{"":>3} '
                  f'{r["mean_transition_accuracy"]:.3f}±{r["std_transition_accuracy"]:.3f}')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_intervals = [0, 10, 25, 50, 100]
    step_accs = [results[str(i)]['mean_step_accuracy'] for i in numeric_intervals]
    step_errs = [results[str(i)]['std_step_accuracy'] for i in numeric_intervals]
    trans_accs = [results[str(i)]['mean_transition_accuracy'] for i in numeric_intervals]
    trans_errs = [results[str(i)]['std_transition_accuracy'] for i in numeric_intervals]

    x = np.arange(len(numeric_intervals))
    ax.errorbar(x, step_accs, yerr=step_errs, marker='o', linewidth=2,
                label='Step accuracy', capsize=4)
    ax.errorbar(x, trans_accs, yerr=trans_errs, marker='s', linewidth=2,
                label='Transition accuracy', capsize=4)

    # Add transitions-only as dashed line
    if 'transitions' in results:
        for metric, style, label in [
            ('mean_step_accuracy', '--', 'Transitions-only (step)'),
            ('mean_transition_accuracy', ':', 'Transitions-only (trans)')
        ]:
            val = results['transitions'][metric]
            ax.axhline(val, linestyle=style, color='gray', alpha=0.7, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in numeric_intervals])
    ax.set_xlabel('Peek interval (steps between observations)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Effect of Observation Frequency on Count Tracking',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = FIGURES_DIR / 'stress_periodic_peeks.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return results


# ─── Experiment 4: Error Characterization ────────────────────────────────────

def experiment_4_error_analysis(rssm_weights, probes, bit_means, n_seeds=15):
    """Characterize how and where imagination fails."""
    print('\n--- Experiment 4: Error Characterization ---')

    error_types = defaultdict(int)
    error_by_depth = defaultdict(lambda: defaultdict(int))
    error_by_count = defaultdict(lambda: defaultdict(int))
    total_transitions_by_depth = defaultdict(int)

    imagination_horizon = 800

    for seed in range(n_seeds):
        rssm = FastRSSMWithImagination(rssm_weights)
        env = BinaryCountingEnv(seed=seed + 400)
        obs = env.reset()
        rssm.reset()

        # Run posterior first to get ground truth
        gt_h = []
        gt_colbits = []
        gt_dc = []
        rssm_gt = FastRSSMWithImagination(rssm_weights)
        env_gt = BinaryCountingEnv(seed=seed + 400)
        obs_gt = env_gt.reset()
        rssm_gt.reset()

        for t in range(imagination_horizon + 50):
            obs_vec = obs_gt[:OBS_SIZE].astype(np.float32)
            deter = rssm_gt.step(obs_vec, action=0)
            col_bits = [1 if env_gt._state.columns[i].occupied else 0
                        for i in range(NUM_COLUMNS)]
            gt_h.append(deter)
            gt_colbits.append(col_bits)
            gt_dc.append(env_gt._state.decimal_count)
            result = env_gt.step(0)
            obs_gt = result[0] if isinstance(result, tuple) else result['obs']
            if env_gt._state.decimal_count >= 14:
                for _ in range(30):
                    obs_vec = obs_gt[:OBS_SIZE].astype(np.float32)
                    deter = rssm_gt.step(obs_vec, action=0)
                    gt_h.append(deter)
                    gt_dc.append(env_gt._state.decimal_count)
                    gt_colbits.append([1 if env_gt._state.columns[i].occupied else 0
                                       for i in range(NUM_COLUMNS)])
                    result = env_gt.step(0)
                    obs_gt = result[0] if isinstance(result, tuple) else result['obs']
                break

        gt_dc = np.array(gt_dc)

        # Fork at step 5 and run imagination
        rssm2 = FastRSSMWithImagination(rssm_weights)
        env2 = BinaryCountingEnv(seed=seed + 400)
        obs2 = env2.reset()
        rssm2.reset()

        for warmup in range(5):
            obs_vec = obs2[:OBS_SIZE].astype(np.float32)
            rssm2.step(obs_vec, action=0)
            result = env2.step(0)
            obs2 = result[0] if isinstance(result, tuple) else result['obs']

        imag_h = []
        for s in range(min(imagination_horizon, len(gt_dc) - 5)):
            d = rssm2.imagine_step(action=0)
            imag_h.append(d)
        imag_h = np.array(imag_h)

        # Project and decode
        imag_proj = project_and_normalize(imag_h, probes, bit_means)
        gt_subset = gt_dc[5:5 + len(imag_proj)]

        # Detect transitions in ground truth and check imagination
        for t in range(1, len(gt_subset)):
            if gt_subset[t] != gt_subset[t-1] and gt_subset[t] == gt_subset[t-1] + 1:
                c_from = gt_subset[t-1]
                c_to = gt_subset[t]
                depth = carry_depth(c_from)

                total_transitions_by_depth[depth] += 1

                # Check imagination's decoded count in a window around the transition
                imag_count_at = decode_count_from_proj(imag_proj[t])
                imag_count_before = decode_count_from_proj(imag_proj[max(0, t-3)])

                # Check each bit
                expected_bits = bits_from_count(c_to)
                actual_bits = (imag_proj[t] > 0.5).astype(int)

                if imag_count_at == c_to:
                    error_types['correct'] += 1
                    error_by_depth[depth]['correct'] += 1
                    error_by_count[c_from]['correct'] += 1
                elif imag_count_at == c_from:
                    # Stuck at old count
                    error_types['missed_flip'] += 1
                    error_by_depth[depth]['missed_flip'] += 1
                    error_by_count[c_from]['missed_flip'] += 1
                elif imag_count_before == c_from and imag_count_at != c_to:
                    # Started flipping but wrong result
                    wrong_bits = np.where(actual_bits != expected_bits)[0]
                    if len(wrong_bits) > 0 and max(wrong_bits) >= depth:
                        error_types['partial_cascade'] += 1
                        error_by_depth[depth]['partial_cascade'] += 1
                        error_by_count[c_from]['partial_cascade'] += 1
                    else:
                        error_types['wrong_flip'] += 1
                        error_by_depth[depth]['wrong_flip'] += 1
                        error_by_count[c_from]['wrong_flip'] += 1
                else:
                    # Already drifted — compounding error
                    error_types['compounding'] += 1
                    error_by_depth[depth]['compounding'] += 1
                    error_by_count[c_from]['compounding'] += 1

    # Print results
    total = sum(error_types.values())
    print(f'\n  Total transitions analyzed: {total}')
    print(f'\n  Error Type Breakdown:')
    print(f'  {"Type":<20} {"Count":<8} {"Fraction"}')
    print(f'  {"-"*40}')
    for etype in ['correct', 'missed_flip', 'partial_cascade', 'wrong_flip', 'compounding']:
        n = error_types.get(etype, 0)
        print(f'  {etype:<20} {n:<8} {n/total:.3f}' if total > 0 else f'  {etype:<20} 0')

    # By depth
    print(f'\n  By Carry Depth:')
    print(f'  {"Depth":<8} {"Total":<8} {"Correct":<10} {"Missed":<10} '
          f'{"Partial":<10} {"Wrong":<10} {"Compound":<10}')
    print(f'  {"-"*65}')
    depth_summary = {}
    for d in sorted(total_transitions_by_depth.keys()):
        total_d = total_transitions_by_depth[d]
        correct_d = error_by_depth[d].get('correct', 0)
        missed_d = error_by_depth[d].get('missed_flip', 0)
        partial_d = error_by_depth[d].get('partial_cascade', 0)
        wrong_d = error_by_depth[d].get('wrong_flip', 0)
        compound_d = error_by_depth[d].get('compounding', 0)
        print(f'  {d:<8} {total_d:<8} {correct_d:<10} {missed_d:<10} '
              f'{partial_d:<10} {wrong_d:<10} {compound_d:<10}')
        depth_summary[d] = {
            'total': total_d, 'correct': correct_d,
            'missed': missed_d, 'partial': partial_d,
            'wrong': wrong_d, 'compounding': compound_d,
            'accuracy': correct_d / total_d if total_d > 0 else 0,
        }

    # By count
    print(f'\n  By Starting Count:')
    count_summary = {}
    for c in sorted(error_by_count.keys()):
        total_c = sum(error_by_count[c].values())
        correct_c = error_by_count[c].get('correct', 0)
        acc = correct_c / total_c if total_c > 0 else 0
        print(f'    {c}→{c+1}: {correct_c}/{total_c} correct ({acc:.0%})')
        count_summary[c] = {
            'total': total_c, 'correct': correct_c, 'accuracy': float(acc)
        }

    # Plot: error type by depth
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: stacked bar by depth
    ax = axes[0]
    depths = sorted(total_transitions_by_depth.keys())
    x = np.arange(len(depths))
    categories = ['correct', 'missed_flip', 'partial_cascade', 'wrong_flip', 'compounding']
    cat_colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52', '#8172B2']
    cat_labels = ['Correct', 'Missed flip', 'Partial cascade', 'Wrong flip', 'Compounding']

    bottom = np.zeros(len(depths))
    for cat, color, label in zip(categories, cat_colors, cat_labels):
        vals = [error_by_depth[d].get(cat, 0) for d in depths]
        ax.bar(x, vals, bottom=bottom, color=color, label=label, edgecolor='white')
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Depth {d}' for d in depths])
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Types by Carry Depth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Right: accuracy by starting count
    ax2 = axes[1]
    counts_sorted = sorted(count_summary.keys())
    accs = [count_summary[c]['accuracy'] for c in counts_sorted]
    colors_bar = ['#C44E52' if carry_depth(c) >= 2 else '#4C72B0' if carry_depth(c) == 1
                  else '#55A868' for c in counts_sorted]
    ax2.bar(range(len(counts_sorted)), accs, color=colors_bar, edgecolor='white')
    ax2.set_xticks(range(len(counts_sorted)))
    ax2.set_xticklabels([f'{c}→{c+1}' for c in counts_sorted], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Imagination accuracy', fontsize=11)
    ax2.set_title('Accuracy by Transition (color=carry depth)', fontsize=13, fontweight='bold')
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)

    fig.tight_layout()
    path = FIGURES_DIR / 'stress_error_analysis.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

    return {
        'error_types': dict(error_types),
        'depth_summary': {str(k): v for k, v in depth_summary.items()},
        'count_summary': {str(k): v for k, v in count_summary.items()},
        'n_seeds': n_seeds,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 90)
    print('TASK B: Extended Imagination Stress Test')
    print('=' * 90)

    # 1. Train probes
    print('\nTraining column-state probes...')
    probes, bit_means = train_colstate_probes()

    # 2. Load RSSM
    print('\nLoading RSSM weights...')
    rssm_weights = load_exported_weights(str(CKPT_DIR))
    print(f'  Loaded {len(rssm_weights)} tensors')

    # 3. Run all 4 experiments
    results = {}
    results['exp1_multi_start'] = experiment_1_multi_start(rssm_weights, probes, bit_means)
    results['exp2_degradation'] = experiment_2_degradation(rssm_weights, probes, bit_means)
    results['exp3_periodic_peeks'] = experiment_3_periodic_peeks(rssm_weights, probes, bit_means)
    results['exp4_error_analysis'] = experiment_4_error_analysis(rssm_weights, probes, bit_means)

    # 4. Save combined results
    out_path = ARTIFACTS_DIR / 'imagination_stress_test.json'
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
