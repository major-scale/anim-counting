#!/usr/bin/env python3
"""Imagination Rollout — Does the RSSM simulate carries without observations?

Runs the binary specialist's RSSM in two modes:
  1. POSTERIOR: normal operation, conditioned on observations at every step
  2. IMAGINATION: prior-only, no observations after the fork point

If the imagination mode shows sequential LSB→MSB bit-probe cascades matching
the posterior, the model is actively simulating the carry mechanism from
internal dynamics alone — not just reactively tracking observations.

Outcomes:
  A) Sequential cascade in imagination → active simulation
  B) Simultaneous flip → endpoint prediction but no process simulation
  C) Partial/noisy cascade → Kalman-filter hybrid
  D) No cascade → purely reactive tracking
"""

import json
import os
import sys
from pathlib import Path

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

CKPT_DIR = BRIDGE_DIR / 'artifacts' / 'checkpoints' / 'binary_baseline_s0' / 'exported'
FIGURES_DIR = SCRIPTS_DIR.parent / 'figures'
ARTIFACTS_DIR = SCRIPTS_DIR.parent / 'artifacts' / 'binary_successor'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── RSSM constants ──────────────────────────────────────────────────────────

STOCH_DIM = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_DIM * STOCH_CLASSES  # 1024
DETER_DIM = 512
LN_EPS = 1e-3


def _ln(x, w, b, eps=LN_EPS):
    mu = x.mean()
    var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _argmax_one_hot(logits):
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    indices = logits.argmax(axis=-1)
    for i, j in enumerate(indices):
        out[i * STOCH_CLASSES + j] = 1.0
    return out


# ─── Weight loading ──────────────────────────────────────────────────────────

def load_exported_weights(weights_dir):
    p = Path(weights_dir)
    with open(p / "dreamer_manifest.json") as f:
        manifest = json.load(f)
    with open(p / "dreamer_weights.bin", "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"],
                            offset=entry["offset"]).copy()
        weights[name] = arr.reshape(entry["shape"])
    return weights


# ─── RSSM with imagination ───────────────────────────────────────────────────

class FastRSSMWithImagination:
    """RSSM with posterior (obs-conditioned) and prior (imagination) stepping."""
    GRU_UPDATE_BIAS = -1.0

    def __init__(self, weights):
        self.w = weights
        self.obs_size = weights["enc_linear0_w"].shape[1]
        self.num_actions = weights["img_in_w"].shape[1] - STOCH_FLAT
        self.reset()

    def reset(self):
        if "deter_init_w" in self.w:
            self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        else:
            self.deter = np.zeros(DETER_DIM, dtype=np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        h = _silu(_ln(self.w["img_out_w"] @ self.deter,
                       self.w["img_out_norm_w"], self.w["img_out_norm_b"]))
        logits = (self.w["imgs_stat_w"] @ h + self.w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        self.stoch = _argmax_one_hot(logits)

    def _transition(self, action=None):
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
        reset_gate = _sigmoid(out[:N])
        cand = np.tanh(reset_gate * out[N:2*N])
        update = _sigmoid(out[2*N:] + self.GRU_UPDATE_BIAS)
        self.deter = (update * cand + (1 - update) * self.deter).astype(np.float32)
        return self.deter.copy()

    def _prior(self):
        w = self.w
        h = _silu(_ln(w["img_out_w"] @ self.deter,
                       w["img_out_norm_w"], w["img_out_norm_b"]))
        logits = (w["imgs_stat_w"] @ h + w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        self.stoch = _argmax_one_hot(logits)
        return self.stoch.copy()

    def _posterior(self, embed):
        w = self.w
        cat_post = np.concatenate([self.deter, embed])
        h_post = _silu(_ln(w["obs_out_w"] @ cat_post,
                            w["obs_out_norm_w"], w["obs_out_norm_b"]))
        post_logits = (w["obs_stat_w"] @ h_post + w["obs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        self.stoch = _argmax_one_hot(post_logits)
        return self.stoch.copy()

    def _encode(self, obs_raw):
        w = self.w
        x = _symlog(obs_raw)
        h = _silu(_ln(w["enc_linear0_w"] @ x, w["enc_norm0_w"], w["enc_norm0_b"]))
        h = _silu(_ln(w["enc_linear1_w"] @ h, w["enc_norm1_w"], w["enc_norm1_b"]))
        embed = _silu(_ln(w["enc_linear2_w"] @ h, w["enc_norm2_w"], w["enc_norm2_b"]))
        return embed

    def step(self, obs_raw, action=None):
        """Full posterior step (observation-conditioned)."""
        self.is_first = False
        embed = self._encode(obs_raw)
        self._transition(action)
        self._posterior(embed)
        return self.deter.copy()

    def imagine_step(self, action=None):
        """Prior-only step (imagination — no observation)."""
        self._transition(action)
        self._prior()
        return self.deter.copy()

    def get_state(self):
        return {
            "deter": self.deter.copy(),
            "stoch": self.stoch.copy(),
            "is_first": self.is_first,
        }

    def set_state(self, state):
        self.deter = state["deter"].copy()
        self.stoch = state["stoch"].copy()
        self.is_first = state["is_first"]


# ─── Carry depth table ───────────────────────────────────────────────────────

def carry_depth(n):
    """Number of consecutive carries from bit 0 when going from n to n+1."""
    d = 0
    for b in range(4):
        if (n >> b) & 1 == 1:
            d += 1
        else:
            break
    return d


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_imagination_data(weights, n_episodes=15, max_steps=900,
                             fork_offset=20, imagination_horizon=35,
                             state_buffer_size=25, seed=42):
    """Run episodes, fork to imagination at carry transitions.

    Uses a circular state buffer (last N states) to avoid saving state
    at every step. Detects transitions in real-time and forks to imagination
    immediately.

    Returns dict of carry transition data organized by carry depth.
    """
    rng = np.random.RandomState(seed)
    rssm = FastRSSMWithImagination(weights)

    # Collect all transitions
    transitions_by_depth = {0: [], 1: [], 2: [], 3: []}

    for ep in range(n_episodes):
        env = BinaryCountingEnv(seed=int(rng.randint(0, 100000)))
        obs = env.reset()
        rssm.reset()

        # Circular state buffer: keep last N states for forking
        state_buffer = []  # (state, deter, count, obs)
        posterior_h_list = []
        posterior_counts_list = []
        prev_count = 0

        for t in range(max_steps):
            obs_vec = obs[:OBS_SIZE].astype(np.float32)
            deter = rssm.step(obs_vec, action=0)
            cur_count = env._state.decimal_count

            # Save to circular buffer
            state_buffer.append({
                'state': rssm.get_state(),
                'deter': deter.copy(),
                'count': cur_count,
                't': t,
            })
            if len(state_buffer) > state_buffer_size:
                state_buffer.pop(0)

            posterior_h_list.append(deter)
            posterior_counts_list.append(cur_count)

            # Detect transition
            if cur_count > prev_count and cur_count == prev_count + 1 and prev_count < 14:
                c_from = prev_count
                c_to = cur_count
                depth = carry_depth(c_from)

                # Fork from buffer (fork_offset steps back)
                buf_idx = max(0, len(state_buffer) - 1 - fork_offset)
                fork_state = state_buffer[buf_idx]
                actual_fork_offset = t - fork_state['t']

                # Posterior trajectory: last 15 + next 10 steps (we'll extend)
                post_start = max(0, len(posterior_h_list) - 15)
                post_traj_so_far = np.array(posterior_h_list[post_start:])
                post_counts_so_far = np.array(posterior_counts_list[post_start:])
                transition_idx = len(post_traj_so_far) - 1

                # Fork to imagination
                saved_current = rssm.get_state()
                rssm.set_state(fork_state['state'])
                imag_h = [fork_state['deter'].copy()]
                for s in range(imagination_horizon):
                    d = rssm.imagine_step(action=0)
                    imag_h.append(d)
                imag_h = np.array(imag_h)

                # Restore RSSM to continue posterior
                rssm.set_state(saved_current)

                transitions_by_depth[depth].append({
                    'episode': ep,
                    'transition_t': t,
                    'c_from': c_from,
                    'c_to': c_to,
                    'depth': depth,
                    'posterior_traj': post_traj_so_far,
                    'posterior_counts': post_counts_so_far,
                    'transition_idx': transition_idx,
                    'imagination_traj': imag_h,
                    'fork_offset': actual_fork_offset,
                })

            prev_count = cur_count

            result = env.step(0)
            obs = result[0] if isinstance(result, tuple) else result['obs']

            if env._state.decimal_count >= 14:
                # Continue a few more steps for posterior data after last transition
                for extra in range(15):
                    obs_vec = obs[:OBS_SIZE].astype(np.float32)
                    deter = rssm.step(obs_vec, action=0)
                    posterior_h_list.append(deter)
                    posterior_counts_list.append(env._state.decimal_count)
                    result = env.step(0)
                    obs = result[0] if isinstance(result, tuple) else result['obs']
                break

        # Extend posterior trajectories for transitions near end of episode
        all_post_h = np.array(posterior_h_list)
        all_post_c = np.array(posterior_counts_list)
        for ev in transitions_by_depth[carry_depth(0)]:  # dummy - extend all
            pass
        # Extend each transition's posterior trajectory with remaining data
        for depth_list in transitions_by_depth.values():
            for ev in depth_list:
                if ev['episode'] == ep:
                    t_idx = ev['transition_t']
                    post_start = max(0, t_idx - 15)
                    post_end = min(len(all_post_h), t_idx + imagination_horizon)
                    ev['posterior_traj'] = all_post_h[post_start:post_end]
                    ev['posterior_counts'] = all_post_c[post_start:post_end]
                    ev['transition_idx'] = t_idx - post_start

        elapsed_ep = ep + 1
        if elapsed_ep % 5 == 0 or elapsed_ep == n_episodes:
            counts = {d: len(v) for d, v in transitions_by_depth.items()}
            print(f'  Episode {elapsed_ep}/{n_episodes}: {counts}')

    return transitions_by_depth


# ─── Analysis ─────────────────────────────────────────────────────────────────

def train_bit_probes(all_posterior_h, all_posterior_counts):
    """Train 4 bit probes on posterior hidden states."""
    h = np.concatenate(all_posterior_h, axis=0)
    c = np.concatenate(all_posterior_counts, axis=0).astype(int)
    bits = np.array([[(ci >> b) & 1 for b in range(4)] for ci in c])

    probe_weights = []
    probe_biases = []
    probe_accs = []
    for b in range(4):
        probe = Ridge(alpha=1.0)
        probe.fit(h, bits[:, b])
        w = probe.coef_
        w_unit = w / np.linalg.norm(w)
        probe_weights.append(w_unit)
        probe_biases.append(float(probe.intercept_))
        acc = (np.round(probe.predict(h)).astype(int) == bits[:, b]).mean()
        probe_accs.append(float(acc))
        print(f'  Bit {b} probe accuracy: {acc:.4f}')

    return probe_weights, probe_biases, probe_accs


def project_trajectories(transitions_by_depth, probe_weights):
    """Project posterior and imagination trajectories onto bit-probe directions."""
    results_by_depth = {}

    for depth in sorted(transitions_by_depth.keys()):
        events = transitions_by_depth[depth]
        if not events:
            continue

        posterior_projs = []
        imagination_projs = []

        for ev in events:
            # Project posterior trajectory
            post_proj = np.array([
                [np.dot(h, w) for w in probe_weights]
                for h in ev['posterior_traj']
            ])  # (T_post, 4)
            posterior_projs.append({
                'proj': post_proj,
                'transition_idx': ev['transition_idx'],
                'counts': ev['posterior_counts'],
            })

            # Project imagination trajectory
            imag_proj = np.array([
                [np.dot(h, w) for w in probe_weights]
                for h in ev['imagination_traj']
            ])  # (T_imag, 4)
            imagination_projs.append({
                'proj': imag_proj,
                'fork_offset': ev['fork_offset'],
            })

        results_by_depth[depth] = {
            'posterior': posterior_projs,
            'imagination': imagination_projs,
            'c_from': events[0]['c_from'],
            'c_to': events[0]['c_to'],
            'n_events': len(events),
        }

    return results_by_depth


def measure_cascade_timing(projections, transitions_by_depth, probe_weights):
    """Measure when each bit changes in posterior vs imagination."""
    timing_results = {}

    for depth in sorted(projections.keys()):
        info = projections[depth]
        events = transitions_by_depth[depth]

        # Determine which bits change
        c_from = events[0]['c_from']
        c_to = events[0]['c_to']
        bits_that_change = []
        bits_that_stay = []
        for b in range(4):
            bf = (c_from >> b) & 1
            bt = (c_to >> b) & 1
            if bf != bt:
                bits_that_change.append(b)
            else:
                bits_that_stay.append(b)

        # For posterior: measure transition timing for each bit
        post_timings = {b: [] for b in bits_that_change}
        post_stay_devs = {b: [] for b in bits_that_stay}

        for pev in info['posterior']:
            proj = pev['proj']  # (T, 4)
            tidx = pev['transition_idx']

            for b in bits_that_change:
                # Find the timestep where this bit's projection crosses midpoint
                bf_val = (c_from >> b) & 1
                bt_val = (c_to >> b) & 1
                midpoint = (proj[0, b] + proj[-1, b]) / 2

                for t in range(len(proj) - 1):
                    if bt_val > bf_val:  # Rising
                        if proj[t, b] < midpoint <= proj[t+1, b]:
                            post_timings[b].append(t - tidx)
                            break
                    else:  # Falling
                        if proj[t, b] > midpoint >= proj[t+1, b]:
                            post_timings[b].append(t - tidx)
                            break

            for b in bits_that_stay:
                # Max deviation of non-participating bits around transition
                w_start = max(0, tidx-15)
                w_end = min(len(proj), tidx+20)
                window_vals = proj[w_start:w_end, b]
                if len(window_vals) > 0:
                    post_stay_devs[b].append(float(np.abs(window_vals - window_vals[0]).max()))

        # For imagination: same but aligned to fork point
        imag_timings = {b: [] for b in bits_that_change}
        imag_stay_devs = {b: [] for b in bits_that_stay}

        for iev in info['imagination']:
            proj = iev['proj']  # (T, 4)
            fork_off = iev['fork_offset']

            for b in bits_that_change:
                bf_val = (c_from >> b) & 1
                bt_val = (c_to >> b) & 1
                midpoint = (proj[0, b] + proj[-1, b]) / 2

                for t in range(len(proj) - 1):
                    if bt_val > bf_val:
                        if proj[t, b] < midpoint <= proj[t+1, b]:
                            imag_timings[b].append(t - fork_off)
                            break
                    else:
                        if proj[t, b] > midpoint >= proj[t+1, b]:
                            imag_timings[b].append(t - fork_off)
                            break

            for b in bits_that_stay:
                window_vals = proj[fork_off:min(len(proj), fork_off+25), b]
                if len(window_vals) > 0:
                    imag_stay_devs[b].append(float(np.abs(window_vals - window_vals[0]).max()))

        timing_results[depth] = {
            'c_from': c_from, 'c_to': c_to,
            'bits_that_change': bits_that_change,
            'bits_that_stay': bits_that_stay,
            'posterior_timings': {b: post_timings[b] for b in bits_that_change},
            'imagination_timings': {b: imag_timings[b] for b in bits_that_change},
            'posterior_stay_devs': {b: post_stay_devs[b] for b in bits_that_stay},
            'imagination_stay_devs': {b: imag_stay_devs[b] for b in bits_that_stay},
        }

    return timing_results


def is_sequential(timings, bits_that_change):
    """Check if bit changes are sequential (LSB first)."""
    if len(bits_that_change) <= 1:
        return True, 0.0
    mean_times = []
    for b in sorted(bits_that_change):
        if timings[b]:
            mean_times.append((b, np.mean(timings[b])))
    if len(mean_times) < 2:
        return True, 0.0
    # Check if higher bits change later
    ordered = all(mean_times[i][1] <= mean_times[i+1][1]
                  for i in range(len(mean_times)-1))
    span = mean_times[-1][1] - mean_times[0][1] if len(mean_times) >= 2 else 0.0
    return ordered, float(span)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(projections, transitions_by_depth):
    """2×4 comparison figure: posterior (top) vs imagination (bottom)."""
    target_depths = [0, 1, 2, 3]
    # Pick representative transitions for each depth
    depth_examples = {}
    for d in target_depths:
        events = transitions_by_depth.get(d, [])
        if events:
            # Find most common (c_from, c_to) pair
            pairs = [(e['c_from'], e['c_to']) for e in events]
            from collections import Counter
            most_common = Counter(pairs).most_common(1)[0][0]
            depth_examples[d] = most_common

    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    bit_colors = ['#55A868', '#4C72B0', '#DD8452', '#C44E52']
    bit_labels = ['Bit0 (1s)', 'Bit1 (2s)', 'Bit2 (4s)', 'Bit3 (8s)']

    for col, d in enumerate(target_depths):
        if d not in projections:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        info = projections[d]
        n_ev = info['n_events']
        c_from, c_to = info['c_from'], info['c_to']

        # Which bits change
        bits_change = set()
        for b in range(4):
            if ((c_from >> b) & 1) != ((c_to >> b) & 1):
                bits_change.add(b)

        # --- TOP ROW: Posterior ---
        ax_post = axes[0, col]
        for pev in info['posterior'][:15]:  # Plot up to 15 traces
            proj = pev['proj']
            tidx = pev['transition_idx']
            x = np.arange(len(proj)) - tidx
            for b in range(4):
                alpha = 0.15 if b not in bits_change else 0.3
                ax_post.plot(x, proj[:, b], color=bit_colors[b], alpha=alpha, linewidth=0.8)

        # Average trajectory
        # Align all posterior traces by transition index
        window = 25
        aligned_post = []
        for pev in info['posterior']:
            proj = pev['proj']
            tidx = pev['transition_idx']
            start = tidx - window
            end = tidx + window
            if start >= 0 and end <= len(proj):
                aligned_post.append(proj[start:end])

        if aligned_post:
            mean_post = np.mean(aligned_post, axis=0)
            x_aligned = np.arange(len(mean_post)) - window
            for b in range(4):
                lw = 2.5 if b in bits_change else 1.5
                ls = '-' if b in bits_change else ':'
                ax_post.plot(x_aligned, mean_post[:, b], color=bit_colors[b],
                             linewidth=lw, linestyle=ls, label=bit_labels[b])

        ax_post.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_post.set_title(f'{c_from}→{c_to} (depth {d})\nPOSTERIOR (with obs)',
                          fontsize=10, fontweight='bold')
        if col == 0:
            ax_post.set_ylabel('Bit probe projection', fontsize=10)
        ax_post.set_xlim(-20, 20)

        # --- BOTTOM ROW: Imagination ---
        ax_imag = axes[1, col]
        for iev in info['imagination'][:15]:
            proj = iev['proj']
            fork_off = iev['fork_offset']
            x = np.arange(len(proj)) - fork_off
            for b in range(4):
                alpha = 0.15 if b not in bits_change else 0.3
                ax_imag.plot(x, proj[:, b], color=bit_colors[b], alpha=alpha, linewidth=0.8)

        # Average imagination trajectory
        aligned_imag = []
        for iev in info['imagination']:
            proj = iev['proj']
            fork_off = iev['fork_offset']
            # All imagination traces start at the fork point
            if len(proj) >= fork_off + window:
                start = max(0, fork_off - window)
                end = fork_off + window
                aligned_imag.append(proj[start:end])

        if aligned_imag:
            # Pad shorter traces
            max_len = max(len(a) for a in aligned_imag)
            padded = []
            for a in aligned_imag:
                if len(a) == max_len:
                    padded.append(a)
            if padded:
                mean_imag = np.mean(padded, axis=0)
                x_aligned = np.arange(len(mean_imag)) - window
                for b in range(4):
                    lw = 2.5 if b in bits_change else 1.5
                    ls = '-' if b in bits_change else ':'
                    ax_imag.plot(x_aligned, mean_imag[:, b], color=bit_colors[b],
                                 linewidth=lw, linestyle=ls, label=bit_labels[b])

        ax_imag.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax_imag.set_title(f'{c_from}→{c_to} (depth {d})\nIMAGINATION (no obs)',
                          fontsize=10, fontweight='bold')
        ax_imag.set_xlabel('Timesteps relative to transition', fontsize=9)
        if col == 0:
            ax_imag.set_ylabel('Bit probe projection', fontsize=10)
        ax_imag.set_xlim(-20, 20)

    # Shared legend
    handles = [plt.Line2D([0], [0], color=bit_colors[b], linewidth=2, label=bit_labels[b])
               for b in range(4)]
    fig.legend(handles=handles, loc='upper right', fontsize=10, ncol=4,
               bbox_to_anchor=(0.98, 1.0))

    fig.suptitle('Does the RSSM Simulate Carries Without Observations?',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = FIGURES_DIR / 'imagination_vs_posterior.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 90)
    print('IMAGINATION ROLLOUT — Does the RSSM simulate carries without observations?')
    print('=' * 90)

    # Load weights
    print('\nLoading binary specialist weights...')
    weights = load_exported_weights(str(CKPT_DIR))
    print(f'  Loaded {len(weights)} weight tensors')
    print(f'  obs_size={weights["enc_linear0_w"].shape[1]}, '
          f'num_actions={weights["img_in_w"].shape[1] - STOCH_FLAT}')

    # Collect data
    print('\nCollecting imagination rollout data (30 episodes)...')
    transitions_by_depth = collect_imagination_data(
        weights, n_episodes=20, max_steps=900,
        fork_offset=20, imagination_horizon=35,
        state_buffer_size=25, seed=42
    )

    for d in sorted(transitions_by_depth.keys()):
        events = transitions_by_depth[d]
        if events:
            pairs = set((e['c_from'], e['c_to']) for e in events)
            print(f'  Depth {d}: {len(events)} transitions '
                  f'({", ".join(f"{a}→{b}" for a, b in sorted(pairs))})')

    # Train bit probes on all posterior data
    print('\nTraining bit probes on posterior data...')
    all_post_h = [ev['posterior_traj'] for d in transitions_by_depth.values()
                  for ev in d]
    all_post_c = [ev['posterior_counts'] for d in transitions_by_depth.values()
                  for ev in d]
    probe_weights, probe_biases, probe_accs = train_bit_probes(all_post_h, all_post_c)

    # Project trajectories
    print('\nProjecting trajectories onto bit-probe directions...')
    projections = project_trajectories(transitions_by_depth, probe_weights)

    # Measure cascade timing
    print('\nMeasuring cascade timing...')
    timing = measure_cascade_timing(projections, transitions_by_depth, probe_weights)

    # ─── Print results ────────────────────────────────────────────────────

    print('\n' + '=' * 100)
    print('RESULTS: Posterior vs Imagination Cascade Timing')
    print('=' * 100)
    print(f'{"Trans":<10} {"Depth":<7} {"Post sequential?":<18} {"Post span":<12} '
          f'{"Imag sequential?":<18} {"Imag span":<12} {"Carry stops?"}')
    print('-' * 100)

    all_results = {}
    for d in sorted(timing.keys()):
        t = timing[d]
        c_from, c_to = t['c_from'], t['c_to']
        bits_change = t['bits_that_change']
        bits_stay = t['bits_that_stay']

        post_seq, post_span = is_sequential(t['posterior_timings'], bits_change)
        imag_seq, imag_span = is_sequential(t['imagination_timings'], bits_change)

        # "Carry stops here" — max deviation of non-participating bits
        post_stay_max = max(
            (np.mean(v) if v else 0.0)
            for v in t['posterior_stay_devs'].values()
        ) if t['posterior_stay_devs'] else 0.0
        imag_stay_max = max(
            (np.mean(v) if v else 0.0)
            for v in t['imagination_stay_devs'].values()
        ) if t['imagination_stay_devs'] else 0.0

        print(f'{c_from}→{c_to:<6} {d:<7} '
              f'{"Yes" if post_seq else "No":<18} {post_span:<12.1f} '
              f'{"Yes" if imag_seq else "No":<18} {imag_span:<12.1f} '
              f'post={post_stay_max:.3f}, imag={imag_stay_max:.3f}')

        # Per-bit timing details
        for b in sorted(bits_change):
            pt = t['posterior_timings'].get(b, [])
            it = t['imagination_timings'].get(b, [])
            pt_str = f'{np.mean(pt):.1f}±{np.std(pt):.1f}' if pt else 'N/A'
            it_str = f'{np.mean(it):.1f}±{np.std(it):.1f}' if it else 'N/A'
            bf = (c_from >> b) & 1
            bt = (c_to >> b) & 1
            direction = '↑' if bt > bf else '↓'
            print(f'    bit{b} ({direction}): posterior t={pt_str}, imagination t={it_str}')

        all_results[d] = {
            'c_from': c_from, 'c_to': c_to,
            'bits_that_change': bits_change,
            'bits_that_stay': bits_stay,
            'posterior_sequential': post_seq,
            'posterior_span': post_span,
            'imagination_sequential': imag_seq,
            'imagination_span': imag_span,
            'posterior_stay_max': post_stay_max,
            'imagination_stay_max': imag_stay_max,
            'posterior_timings': {str(b): t['posterior_timings'][b] for b in bits_change},
            'imagination_timings': {str(b): t['imagination_timings'][b] for b in bits_change},
        }

    # ─── Outcome classification ───────────────────────────────────────────

    print('\n' + '=' * 90)
    print('OUTCOME CLASSIFICATION')
    print('=' * 90)

    # Check if depth-2 and depth-3 show sequential cascades in imagination
    deep_depths = [d for d in [2, 3] if d in timing]
    if deep_depths:
        all_imag_seq = all(
            is_sequential(timing[d]['imagination_timings'],
                          timing[d]['bits_that_change'])[0]
            for d in deep_depths
        )
        any_imag_span = any(
            is_sequential(timing[d]['imagination_timings'],
                          timing[d]['bits_that_change'])[1] > 1.0
            for d in deep_depths
        )

        if all_imag_seq and any_imag_span:
            outcome = 'A'
            print('  OUTCOME A: Sequential cascade in imagination')
            print('  The model generates the carry cascade from internal dynamics alone.')
            print('  This is active simulation, not reactive tracking.')
        elif all_imag_seq and not any_imag_span:
            outcome = 'B'
            print('  OUTCOME B: Simultaneous flip in imagination')
            print('  The model predicts WHAT happens (correct endpoint) but not HOW')
            print('  it unfolds (all bits flip at once, not sequentially).')
        else:
            # Check if there's any structure
            any_imag_seq = any(
                is_sequential(timing[d]['imagination_timings'],
                              timing[d]['bits_that_change'])[0]
                for d in deep_depths
            )
            if any_imag_seq:
                outcome = 'C'
                print('  OUTCOME C: Partial cascade in imagination')
                print('  Some sequential structure but degraded vs posterior.')
            else:
                outcome = 'D'
                print('  OUTCOME D: No meaningful cascade in imagination')
                print('  The model depends entirely on observations for carry tracking.')
    else:
        outcome = 'X'
        print('  Insufficient data — no depth-2 or depth-3 transitions found')

    # ─── Carry stops here ─────────────────────────────────────────────────

    print('\n  "Carry stops here" in imagination:')
    for d in sorted(timing.keys()):
        t = timing[d]
        if t['bits_that_stay']:
            devs = {b: np.mean(v) if v else 0.0
                    for b, v in t['imagination_stay_devs'].items()}
            stay_str = ', '.join(f'bit{b}={v:.3f}' for b, v in sorted(devs.items()))
            all_low = all(v < 0.05 for v in devs.values())
            print(f'    {t["c_from"]}→{t["c_to"]} (d={d}): {stay_str}'
                  f' {"✓ clean" if all_low else "✗ bleed"}')

    # ─── Generate figures ─────────────────────────────────────────────────

    print('\nGenerating comparison figure...')
    plot_comparison(projections, transitions_by_depth)

    # ─── Save results ─────────────────────────────────────────────────────

    save_data = {
        'outcome': outcome,
        'n_episodes': 30,
        'probe_accs': probe_accs,
        'timing_results': all_results,
    }
    out_path = ARTIFACTS_DIR / 'imagination_rollout.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'  Results saved to {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
