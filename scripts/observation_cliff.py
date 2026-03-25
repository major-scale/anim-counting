#!/usr/bin/env python3
"""Observation Cliff Deep Dive — Why Can't the Model Recover?

Investigates the mechanism behind the binary observation cliff:
continuous observations → 96% accuracy, ANY interruption → ~10%.

7 analyses:
  1. Hidden state drift during blind steps
  2. What happens at the peek (posterior correction mechanics)
  3. GRU gate values during peek recovery
  4. Multi-peek recovery curve
  5. Where does it go? (PCA visualization of blind states)
  6. Count-specific cliff profiles
  7. Observation encoding test (centroid replacement surgery)
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# ─── Path setup ───────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = Path('/workspace/projects/jamstack-v1/bridge')
sys.path.insert(0, str(BRIDGE_DIR / 'scripts'))

from binary_counting_env import BinaryCountingEnv, OBS_SIZE
from imagination_rollout_binary import (
    FastRSSMWithImagination, load_exported_weights, carry_depth,
    DETER_DIM, STOCH_DIM, STOCH_CLASSES, STOCH_FLAT,
    _ln, _silu, _sigmoid, _symlog, _argmax_one_hot, LN_EPS
)

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
BATTERY_PATH = BRIDGE_DIR / 'artifacts' / 'battery' / 'binary_baseline_s0' / 'battery.npz'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLUMNS = 4


# ─── Instrumented RSSM (exposes GRU gates) ──────────────────────────────────

class InstrumentedRSSM(FastRSSMWithImagination):
    """RSSM that records GRU gate activations on each transition."""

    def __init__(self, weights):
        super().__init__(weights)
        self.last_update_gate = None
        self.last_reset_gate = None

    def _transition(self, action=None):
        """GRU transition with gate recording."""
        w = self.w
        if self.is_first or action is None:
            act = np.zeros(self.num_actions, dtype=np.float32)
        elif self.num_actions > 1:
            act = np.zeros(self.num_actions, dtype=np.float32)
            act[int(action)] = 1.0
        else:
            act = np.atleast_1d(np.float32(action))

        cat_in = np.concatenate([self.stoch.copy(), act])
        h_in = _silu(_ln(w["img_in_w"] @ cat_in,
                         w["img_in_norm_w"], w["img_in_norm_b"]))

        combined = np.concatenate([h_in, self.deter])
        out = w["gru_w"] @ combined
        out = _ln(out, w["gru_norm_w"], w["gru_norm_b"])
        N = DETER_DIM
        self.last_reset_gate = _sigmoid(out[:N])
        cand = np.tanh(self.last_reset_gate * out[N:2*N])
        self.last_update_gate = _sigmoid(out[2*N:] + self.GRU_UPDATE_BIAS)
        self.deter = (self.last_update_gate * cand +
                      (1 - self.last_update_gate) * self.deter).astype(np.float32)
        return self.deter.copy()


# ─── Data collection ────────────────────────────────────────────────────────

def collect_baseline_episodes(rssm_weights, n_episodes=15, max_steps=1000):
    """Collect normal posterior episodes for centroid computation."""
    print(f'  Collecting {n_episodes} baseline episodes...', flush=True)
    all_h = []
    all_dc = []
    episodes = []

    for ep in range(n_episodes):
        rssm = InstrumentedRSSM(rssm_weights)
        env = BinaryCountingEnv(seed=42 + ep)
        obs = env.reset()
        rssm.reset()

        ep_h, ep_dc, ep_obs, ep_states, ep_stoch = [], [], [], [], []
        ep_update_gates = []

        for t in range(max_steps):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            ep_obs.append(obs_vec.copy())

            deter = rssm.step(obs_vec, action=0)
            ep_h.append(deter)
            ep_dc.append(env._state.decimal_count)
            ep_states.append(rssm.get_state())
            ep_stoch.append(rssm.stoch.copy())
            ep_update_gates.append(rssm.last_update_gate.copy())

            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']

            if env._state.decimal_count >= 14:
                break

        episodes.append({
            'h': np.array(ep_h),
            'dc': np.array(ep_dc),
            'obs': np.array(ep_obs),
            'states': ep_states,
            'stoch': np.array(ep_stoch),
            'update_gates': np.array(ep_update_gates),
        })
        all_h.append(np.array(ep_h))
        all_dc.append(np.array(ep_dc))

        if (ep + 1) % 5 == 0:
            print(f'    Episode {ep+1}/{n_episodes}: {len(ep_h)} steps', flush=True)

    return episodes, np.concatenate(all_h), np.concatenate(all_dc)


def compute_centroids(all_h, all_dc):
    """Compute count centroids from posterior hidden states."""
    centroids = {}
    for c in range(15):
        mask = all_dc == c
        if mask.sum() > 0:
            centroids[c] = all_h[mask].mean(axis=0)
    return centroids


# ─── Analysis 1: Hidden State Drift During Blind Steps ──────────────────────

def analysis_1_drift(episodes, rssm_weights, centroids, all_h):
    """Measure drift from centroid during blind periods."""
    print('\n--- Analysis 1: Hidden State Drift During Blind Steps ---', flush=True)

    blind_durations = [5, 10, 20, 50]

    # Fit PCA on all posterior states for manifold distance
    pca = PCA(n_components=50)
    pca.fit(all_h)
    explained = pca.explained_variance_ratio_.sum()
    print(f'  PCA (50 components) explains {explained:.3f} of variance', flush=True)

    results = {}
    for dur in blind_durations:
        drift_correct = []   # distance from correct centroid
        drift_nearest = []   # distance from nearest centroid
        manifold_residual = []  # reconstruction error from PCA
        start_counts = []

        for ep in episodes:
            dc = ep['dc']
            states = ep['states']
            obs_arr = ep['obs']

            # Find stable positions (well into an idle period)
            for t in range(20, len(dc) - dur - 5):
                # Must be in a stable count region
                if dc[t] != dc[t-1] or dc[t] != dc[min(t + dur + 5, len(dc)-1)]:
                    continue
                if t % 40 != 0:  # subsample to limit compute
                    continue

                count = dc[t]
                if count not in centroids:
                    continue

                # Fork: run blind from this point
                rssm_blind = InstrumentedRSSM(rssm_weights)
                rssm_blind.set_state(states[t])

                trajectory_correct = []
                trajectory_nearest = []
                trajectory_residual = []

                for s in range(dur):
                    rssm_blind.imagine_step(action=0)  # prior-only
                    h_blind = rssm_blind.deter.copy()

                    # Distance from correct centroid
                    d_correct = np.linalg.norm(h_blind - centroids[count])
                    trajectory_correct.append(d_correct)

                    # Distance from nearest centroid
                    d_nearest = min(np.linalg.norm(h_blind - centroids[c])
                                    for c in centroids)
                    trajectory_nearest.append(d_nearest)

                    # Manifold distance: reconstruction residual
                    projected = pca.transform(h_blind.reshape(1, -1))
                    reconstructed = pca.inverse_transform(projected)
                    residual = np.linalg.norm(h_blind - reconstructed.flatten())
                    trajectory_residual.append(residual)

                drift_correct.append(trajectory_correct)
                drift_nearest.append(trajectory_nearest)
                manifold_residual.append(trajectory_residual)
                start_counts.append(count)

        if not drift_correct:
            continue

        drift_correct = np.array(drift_correct)
        drift_nearest = np.array(drift_nearest)
        manifold_residual = np.array(manifold_residual)

        results[dur] = {
            'mean_correct': drift_correct.mean(axis=0).tolist(),
            'std_correct': drift_correct.std(axis=0).tolist(),
            'mean_nearest': drift_nearest.mean(axis=0).tolist(),
            'mean_residual': manifold_residual.mean(axis=0).tolist(),
            'n_trials': len(drift_correct),
        }

        print(f'  Blind {dur} steps (n={len(drift_correct)}): '
              f'final correct dist={drift_correct[:, -1].mean():.3f}, '
              f'final nearest dist={drift_nearest[:, -1].mean():.3f}, '
              f'final residual={manifold_residual[:, -1].mean():.3f}', flush=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for dur in blind_durations:
        if dur not in results:
            continue
        r = results[dur]
        steps = np.arange(1, dur + 1)
        axes[0].plot(steps, r['mean_correct'], label=f'{dur} blind steps')
        axes[1].plot(steps, r['mean_nearest'], label=f'{dur} blind steps')
        axes[2].plot(steps, r['mean_residual'], label=f'{dur} blind steps')

    axes[0].set_xlabel('Blind steps')
    axes[0].set_ylabel('Distance from correct centroid')
    axes[0].set_title('Drift from Correct Count Centroid')
    axes[0].legend()

    axes[1].set_xlabel('Blind steps')
    axes[1].set_ylabel('Distance from nearest centroid')
    axes[1].set_title('Drift from Nearest Count Centroid')
    axes[1].legend()

    axes[2].set_xlabel('Blind steps')
    axes[2].set_ylabel('PCA reconstruction residual')
    axes[2].set_title('Off-Manifold Distance')
    axes[2].legend()

    fig.suptitle('Hidden State Drift During Blind Steps', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cliff_drift_trajectory.png', dpi=150)
    plt.close()
    print(f'  Saved {FIGURES_DIR / "cliff_drift_trajectory.png"}', flush=True)

    return results, pca


# ─── Analysis 2: What Happens at the Peek ───────────────────────────────────

def analysis_2_peek(episodes, rssm_weights, centroids):
    """Trace hidden state before/after a peek observation."""
    print('\n--- Analysis 2: What Happens at the Peek ---', flush=True)

    blind_durations = [5, 10, 20, 50]
    results = {}

    for dur in blind_durations:
        peek_data = []

        for ep in episodes:
            dc = ep['dc']
            states = ep['states']
            obs_arr = ep['obs']

            for t in range(20, len(dc) - dur - 5):
                if dc[t] != dc[t-1] or dc[t] != dc[min(t + dur + 2, len(dc)-1)]:
                    continue
                if t % 40 != 0:
                    continue

                count = dc[t]
                if count not in centroids:
                    continue

                # Run blind
                rssm = InstrumentedRSSM(rssm_weights)
                rssm.set_state(states[t])

                for s in range(dur):
                    rssm.imagine_step(action=0)

                h_before = rssm.deter.copy()

                # Now give it the peek observation
                obs_peek = obs_arr[min(t + dur, len(obs_arr) - 1)]
                h_after = rssm.step(obs_peek, action=0)
                update_gate = rssm.last_update_gate.copy()

                # Measurements
                move_dist = np.linalg.norm(h_after - h_before)
                dist_correct_before = np.linalg.norm(h_before - centroids[count])
                dist_correct_after = np.linalg.norm(h_after - centroids[count])

                # Nearest centroid after peek
                nearest_after = min(centroids.keys(),
                                    key=lambda c: np.linalg.norm(h_after - centroids[c]))
                dist_nearest_after = np.linalg.norm(h_after - centroids[nearest_after])

                peek_data.append({
                    'count': count,
                    'move_dist': float(move_dist),
                    'dist_correct_before': float(dist_correct_before),
                    'dist_correct_after': float(dist_correct_after),
                    'nearest_after': int(nearest_after),
                    'dist_nearest_after': float(dist_nearest_after),
                    'mean_update_gate': float(update_gate.mean()),
                    'max_update_gate': float(update_gate.max()),
                })

        if not peek_data:
            continue

        results[dur] = peek_data
        move_dists = [d['move_dist'] for d in peek_data]
        correct_before = [d['dist_correct_before'] for d in peek_data]
        correct_after = [d['dist_correct_after'] for d in peek_data]
        nearest_match = sum(1 for d in peek_data if d['nearest_after'] == d['count'])

        print(f'  Blind {dur} steps (n={len(peek_data)}):')
        print(f'    Peek move: {np.mean(move_dists):.3f} ± {np.std(move_dists):.3f}')
        print(f'    Dist correct before: {np.mean(correct_before):.3f}')
        print(f'    Dist correct after:  {np.mean(correct_after):.3f}')
        print(f'    Nearest=correct after: {nearest_match}/{len(peek_data)} '
              f'({100*nearest_match/len(peek_data):.1f}%)')
        print(f'    Mean update gate: '
              f'{np.mean([d["mean_update_gate"] for d in peek_data]):.4f}')

    return results


# ─── Analysis 3: GRU Gate Values ────────────────────────────────────────────

def analysis_3_gates(episodes, rssm_weights):
    """Compare GRU update gate values across conditions."""
    print('\n--- Analysis 3: GRU Gate Values ---', flush=True)

    # Baseline: normal posterior gates
    all_gates_normal = np.concatenate([ep['update_gates'] for ep in episodes])
    normal_mean = all_gates_normal.mean()
    normal_dim_means = all_gates_normal.mean(axis=0)  # per-dim
    print(f'  Normal posterior: mean gate = {normal_mean:.4f} '
          f'(min dim={normal_dim_means.min():.4f}, max dim={normal_dim_means.max():.4f})')

    # Blind steps and peek steps at various durations
    blind_durations = [5, 10, 20, 50]
    gate_results = {'normal_mean': float(normal_mean)}

    for dur in blind_durations:
        blind_gates = []
        peek_gates = []

        for ep in episodes:
            dc = ep['dc']
            states = ep['states']
            obs_arr = ep['obs']

            for t in range(20, len(dc) - dur - 5):
                if dc[t] != dc[t-1] or dc[t] != dc[min(t + dur + 2, len(dc)-1)]:
                    continue
                if t % 40 != 0:
                    continue

                rssm = InstrumentedRSSM(rssm_weights)
                rssm.set_state(states[t])

                # Blind steps
                for s in range(dur):
                    rssm.imagine_step(action=0)
                    blind_gates.append(rssm.last_update_gate.mean())

                # Peek step
                obs_peek = obs_arr[min(t + dur, len(obs_arr) - 1)]
                rssm.step(obs_peek, action=0)
                peek_gates.append(rssm.last_update_gate.mean())

        if not peek_gates:
            continue

        blind_mean = float(np.mean(blind_gates))
        peek_mean = float(np.mean(peek_gates))

        gate_results[f'blind_{dur}'] = blind_mean
        gate_results[f'peek_after_{dur}'] = peek_mean

        print(f'  Blind {dur} steps: blind gate = {blind_mean:.4f}, '
              f'peek gate = {peek_mean:.4f} '
              f'(ratio peek/normal = {peek_mean/normal_mean:.3f})')

    return gate_results


# ─── Analysis 4: Multi-Peek Recovery ────────────────────────────────────────

def analysis_4_multi_peek(episodes, rssm_weights, centroids):
    """Test recovery with multiple consecutive peeks after blind period."""
    print('\n--- Analysis 4: Multi-Peek Recovery ---', flush=True)

    blind_dur = 10
    peek_counts = [1, 2, 5, 10, 20]
    results = {}

    for n_peeks in peek_counts:
        recoveries = []

        for ep in episodes:
            dc = ep['dc']
            states = ep['states']
            obs_arr = ep['obs']

            for t in range(20, len(dc) - blind_dur - n_peeks - 5):
                if dc[t] != dc[t-1]:
                    continue
                # Verify count is stable through entire window
                end_t = min(t + blind_dur + n_peeks + 2, len(dc) - 1)
                if dc[end_t] != dc[t]:
                    continue
                if t % 40 != 0:
                    continue

                count = dc[t]
                if count not in centroids:
                    continue

                rssm = InstrumentedRSSM(rssm_weights)
                rssm.set_state(states[t])

                # Blind period
                for s in range(blind_dur):
                    rssm.imagine_step(action=0)

                # Recovery peeks
                for p in range(n_peeks):
                    obs_idx = min(t + blind_dur + p, len(obs_arr) - 1)
                    rssm.step(obs_arr[obs_idx], action=0)

                # Check: nearest centroid
                h_final = rssm.deter.copy()
                nearest = min(centroids.keys(),
                              key=lambda c: np.linalg.norm(h_final - centroids[c]))
                dist_correct = np.linalg.norm(h_final - centroids[count])

                recoveries.append({
                    'count': count,
                    'correct': nearest == count,
                    'dist_correct': float(dist_correct),
                })

        if not recoveries:
            continue

        n_correct = sum(1 for r in recoveries if r['correct'])
        acc = n_correct / len(recoveries)
        mean_dist = np.mean([r['dist_correct'] for r in recoveries])

        results[n_peeks] = {
            'accuracy': acc,
            'n_trials': len(recoveries),
            'mean_dist_correct': float(mean_dist),
        }

        print(f'  10 blind → {n_peeks} peeks: {acc:.1%} recovery '
              f'({n_correct}/{len(recoveries)}), '
              f'dist_correct={mean_dist:.3f}', flush=True)

    # Plot
    if results:
        fig, ax = plt.subplots(figsize=(8, 5))
        peeks = sorted(results.keys())
        accs = [results[p]['accuracy'] for p in peeks]
        ax.plot(peeks, accs, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of consecutive observation peeks')
        ax.set_ylabel('Recovery accuracy (nearest centroid = correct)')
        ax.set_title('Multi-Peek Recovery After 10 Blind Steps')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Perfect')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cliff_multi_peek_recovery.png', dpi=150)
        plt.close()
        print(f'  Saved {FIGURES_DIR / "cliff_multi_peek_recovery.png"}', flush=True)

    return results


# ─── Analysis 5: Where Does It Go? ──────────────────────────────────────────

def analysis_5_visualization(episodes, rssm_weights, all_h, all_dc, pca_model):
    """PCA visualization of blind states vs posterior states."""
    print('\n--- Analysis 5: Where Does It Go? ---', flush=True)

    # Collect blind states at various durations
    blind_states = {10: [], 20: [], 50: []}
    blind_stoch_entropy = {10: [], 20: [], 50: []}

    for ep in episodes:
        dc = ep['dc']
        states = ep['states']

        for t in range(20, len(dc) - 55):
            if dc[t] != dc[t-1]:
                continue
            if t % 60 != 0:
                continue

            for dur in blind_states.keys():
                rssm = InstrumentedRSSM(rssm_weights)
                rssm.set_state(states[t])

                for s in range(dur):
                    rssm.imagine_step(action=0)

                blind_states[dur].append(rssm.deter.copy())

                # Stochastic state diversity: count unique active indices
                stoch_reshaped = rssm.stoch.reshape(STOCH_DIM, STOCH_CLASSES)
                active = stoch_reshaped.argmax(axis=1)
                n_unique = len(np.unique(active))
                blind_stoch_entropy[dur].append(n_unique)

    # Normal posterior stochastic diversity
    normal_stoch_diversity = []
    for ep in episodes:
        for stoch in ep['stoch'][::20]:  # subsample
            reshaped = stoch.reshape(STOCH_DIM, STOCH_CLASSES)
            active = reshaped.argmax(axis=1)
            normal_stoch_diversity.append(len(np.unique(active)))

    print(f'  Stochastic diversity (unique categories across 32 dims):')
    print(f'    Normal posterior: {np.mean(normal_stoch_diversity):.1f} ± '
          f'{np.std(normal_stoch_diversity):.1f}')
    for dur in sorted(blind_stoch_entropy.keys()):
        vals = blind_stoch_entropy[dur]
        if vals:
            print(f'    After {dur} blind steps: {np.mean(vals):.1f} ± '
                  f'{np.std(vals):.1f}')

    # PCA projection
    pca3 = PCA(n_components=3)
    pca3.fit(all_h)

    posterior_proj = pca3.transform(all_h[::10])  # subsample
    posterior_dc = all_dc[::10]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, dur in enumerate(sorted(blind_states.keys())):
        if not blind_states[dur]:
            continue
        blind_arr = np.array(blind_states[dur])
        blind_proj = pca3.transform(blind_arr)

        ax = axes[idx]
        # Plot posterior states colored by count
        scatter = ax.scatter(posterior_proj[:, 0], posterior_proj[:, 1],
                             c=posterior_dc, cmap='viridis', s=2, alpha=0.3,
                             label='Posterior')
        # Plot blind states in red
        ax.scatter(blind_proj[:, 0], blind_proj[:, 1],
                   c='red', s=15, alpha=0.7, marker='x',
                   label=f'After {dur} blind steps')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'After {dur} Blind Steps (n={len(blind_arr)})')
        ax.legend(fontsize=8)

    fig.suptitle('Where Do Blind States End Up? (PCA Projection)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cliff_blind_pca.png', dpi=150)
    plt.close()
    print(f'  Saved {FIGURES_DIR / "cliff_blind_pca.png"}', flush=True)

    return {dur: len(blind_states[dur]) for dur in blind_states}, blind_stoch_entropy


# ─── Analysis 6: Count-Specific Cliff Profiles ──────────────────────────────

def analysis_6_count_specific(episodes, rssm_weights, centroids):
    """Test cliff severity per count."""
    print('\n--- Analysis 6: Count-Specific Cliff Profiles ---', flush=True)

    blind_dur = 10
    results = {c: {'correct': 0, 'total': 0} for c in range(15)}

    for ep in episodes:
        dc = ep['dc']
        states = ep['states']
        obs_arr = ep['obs']

        for t in range(20, len(dc) - blind_dur - 5):
            if dc[t] != dc[t-1]:
                continue
            end_t = min(t + blind_dur + 2, len(dc) - 1)
            if dc[end_t] != dc[t]:
                continue
            if t % 20 != 0:
                continue

            count = dc[t]
            if count not in centroids:
                continue

            rssm = InstrumentedRSSM(rssm_weights)
            rssm.set_state(states[t])

            for s in range(blind_dur):
                rssm.imagine_step(action=0)

            # Single peek
            obs_peek = obs_arr[min(t + blind_dur, len(obs_arr) - 1)]
            rssm.step(obs_peek, action=0)

            nearest = min(centroids.keys(),
                          key=lambda c: np.linalg.norm(rssm.deter - centroids[c]))

            results[count]['total'] += 1
            if nearest == count:
                results[count]['correct'] += 1

    print(f'\n  {"Count":<8} {"Depth":<7} {"Recovery":<12} {"n":<6}')
    print('  ' + '-' * 35)
    for c in range(15):
        if results[c]['total'] == 0:
            continue
        acc = results[c]['correct'] / results[c]['total']
        depth = carry_depth(c) if c < 14 else -1
        print(f'  {c:<8} {depth:<7} {acc:.1%} ({results[c]["correct"]}/{results[c]["total"]})')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    counts_plot = [c for c in range(15) if results[c]['total'] > 0]
    accs_plot = [results[c]['correct'] / results[c]['total'] for c in counts_plot]
    depths_plot = [carry_depth(c) if c < 14 else -1 for c in counts_plot]

    colors = ['#2196F3' if d == 0 else '#FF9800' if d == 1 else
              '#F44336' if d == 2 else '#9C27B0' if d == 3 else '#999'
              for d in depths_plot]

    bars = ax.bar(counts_plot, accs_plot, color=colors)
    ax.set_xlabel('Count')
    ax.set_ylabel('1-Peek Recovery Accuracy')
    ax.set_title('Count-Specific Recovery After 10 Blind Steps')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(15))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='Depth 0'),
                       Patch(facecolor='#FF9800', label='Depth 1'),
                       Patch(facecolor='#F44336', label='Depth 2'),
                       Patch(facecolor='#9C27B0', label='Depth 3')]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cliff_count_specific.png', dpi=150)
    plt.close()
    print(f'  Saved {FIGURES_DIR / "cliff_count_specific.png"}', flush=True)

    return results


# ─── Analysis 7: Observation Encoding Test (Centroid Surgery) ────────────────

def analysis_7_surgery(episodes, rssm_weights, centroids):
    """Replace drifted hidden state with correct centroid, then process obs."""
    print('\n--- Analysis 7: Centroid Replacement Surgery ---', flush=True)

    blind_dur = 20
    results = {'with_surgery': [], 'without_surgery': []}

    for ep in episodes:
        dc = ep['dc']
        states = ep['states']
        obs_arr = ep['obs']

        for t in range(20, len(dc) - blind_dur - 10):
            if dc[t] != dc[t-1]:
                continue
            end_t = min(t + blind_dur + 5, len(dc) - 1)
            if dc[end_t] != dc[t]:
                continue
            if t % 40 != 0:
                continue

            count = dc[t]
            if count not in centroids:
                continue

            # Without surgery: blind → peek
            rssm_no = InstrumentedRSSM(rssm_weights)
            rssm_no.set_state(states[t])
            for s in range(blind_dur):
                rssm_no.imagine_step(action=0)
            drifted_h = rssm_no.deter.copy()
            obs_peek = obs_arr[min(t + blind_dur, len(obs_arr) - 1)]
            rssm_no.step(obs_peek, action=0)
            nearest_no = min(centroids.keys(),
                             key=lambda c: np.linalg.norm(rssm_no.deter - centroids[c]))

            # With surgery: blind → replace deter with correct centroid → peek
            rssm_yes = InstrumentedRSSM(rssm_weights)
            rssm_yes.set_state(states[t])
            for s in range(blind_dur):
                rssm_yes.imagine_step(action=0)
            # SURGERY: replace deter with correct centroid
            rssm_yes.deter = centroids[count].copy()
            rssm_yes.step(obs_peek, action=0)
            nearest_yes = min(centroids.keys(),
                              key=lambda c: np.linalg.norm(rssm_yes.deter - centroids[c]))

            results['without_surgery'].append({
                'count': count,
                'correct': nearest_no == count,
                'nearest': int(nearest_no),
            })
            results['with_surgery'].append({
                'count': count,
                'correct': nearest_yes == count,
                'nearest': int(nearest_yes),
            })

    n_no = len(results['without_surgery'])
    n_yes = len(results['with_surgery'])
    acc_no = sum(1 for r in results['without_surgery'] if r['correct']) / max(n_no, 1)
    acc_yes = sum(1 for r in results['with_surgery'] if r['correct']) / max(n_yes, 1)

    print(f'  Without surgery (drifted h → peek): '
          f'{acc_no:.1%} ({sum(1 for r in results["without_surgery"] if r["correct"])}/{n_no})')
    print(f'  With surgery (centroid h → peek):    '
          f'{acc_yes:.1%} ({sum(1 for r in results["with_surgery"] if r["correct"])}/{n_yes})')

    if acc_yes > acc_no + 0.1:
        print('  → Surgery helps significantly: the drifted hidden state is the problem,')
        print('    not the encoder or GRU architecture.')
    elif acc_yes > acc_no:
        print('  → Surgery helps modestly: hidden state drift is part of the problem.')
    else:
        print('  → Surgery does NOT help: the problem is not just hidden state position.')

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('OBSERVATION CLIFF DEEP DIVE — Why Can\'t the Model Recover?')
    print('=' * 80)

    print('\nLoading RSSM weights...')
    rssm_weights = load_exported_weights(CKPT_DIR)
    print(f'  Loaded {len(rssm_weights)} tensors')

    # Collect baseline data
    print('\nCollecting baseline episodes...')
    episodes, all_h, all_dc = collect_baseline_episodes(rssm_weights, n_episodes=15)

    # Compute centroids
    centroids = compute_centroids(all_h, all_dc)
    centroid_dists = []
    for c in range(14):
        if c in centroids and c+1 in centroids:
            centroid_dists.append(np.linalg.norm(centroids[c+1] - centroids[c]))
    print(f'  Mean centroid distance: {np.mean(centroid_dists):.3f}')

    # Run analyses (priority order)
    drift_results, pca_model = analysis_1_drift(episodes, rssm_weights, centroids, all_h)
    peek_results = analysis_2_peek(episodes, rssm_weights, centroids)
    gate_results = analysis_3_gates(episodes, rssm_weights)
    multi_peek_results = analysis_4_multi_peek(episodes, rssm_weights, centroids)
    viz_counts, stoch_entropy = analysis_5_visualization(
        episodes, rssm_weights, all_h, all_dc, pca_model)
    count_results = analysis_6_count_specific(episodes, rssm_weights, centroids)
    surgery_results = analysis_7_surgery(episodes, rssm_weights, centroids)

    # ─── Summary table ──────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('SUMMARY TABLE')
    print('=' * 80)

    print(f'\n{"Blind dur":<12} {"Drift correct":<16} {"Drift nearest":<16} '
          f'{"On-manifold?":<14} {"1-peek?":<10} {"5-peek?":<10}')
    print('-' * 78)
    for dur in [5, 10, 20, 50]:
        drift_c = drift_results.get(dur, {}).get('mean_correct', [0])[-1] if dur in drift_results else '?'
        drift_n = drift_results.get(dur, {}).get('mean_nearest', [0])[-1] if dur in drift_results else '?'
        residual = drift_results.get(dur, {}).get('mean_residual', [0])[-1] if dur in drift_results else '?'
        on_manifold = f'{residual:.2f}' if isinstance(residual, float) else '?'

        # 1-peek: from analysis 6 (aggregate) or analysis 2
        peek_1 = '?'
        if dur == 10 and count_results:
            total = sum(r['total'] for r in count_results.values())
            correct = sum(r['correct'] for r in count_results.values())
            peek_1 = f'{correct/max(total,1):.1%}'
        elif dur in peek_results:
            n_correct = sum(1 for d in peek_results[dur] if d['nearest_after'] == d['count'])
            peek_1 = f'{n_correct/max(len(peek_results[dur]),1):.1%}'

        # 5-peek: from analysis 4
        peek_5 = '?'
        if dur == 10 and 5 in multi_peek_results:
            peek_5 = f'{multi_peek_results[5]["accuracy"]:.1%}'

        print(f'{dur:<12} {drift_c if isinstance(drift_c, str) else f"{drift_c:.3f}":<16} '
              f'{drift_n if isinstance(drift_n, str) else f"{drift_n:.3f}":<16} '
              f'{on_manifold:<14} {peek_1:<10} {peek_5:<10}')

    # ─── Save results ───────────────────────────────────────────────────────
    save_data = {
        'drift': {str(k): v for k, v in drift_results.items()},
        'peek': {str(k): v for k, v in peek_results.items()},
        'gates': gate_results,
        'multi_peek': {str(k): v for k, v in multi_peek_results.items()},
        'stoch_diversity': {
            str(k): {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in stoch_entropy.items() if v
        },
        'count_specific': {str(k): v for k, v in count_results.items()},
        'surgery': {
            'without': {'accuracy': sum(1 for r in surgery_results['without_surgery'] if r['correct']) /
                        max(len(surgery_results['without_surgery']), 1),
                        'n': len(surgery_results['without_surgery'])},
            'with': {'accuracy': sum(1 for r in surgery_results['with_surgery'] if r['correct']) /
                     max(len(surgery_results['with_surgery']), 1),
                     'n': len(surgery_results['with_surgery'])},
        },
    }

    # Convert numpy types for JSON serialization
    def numpy_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {str(k): deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return numpy_safe(d)

    save_data = deep_convert(save_data)

    out_path = ARTIFACTS_DIR / 'observation_cliff.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
