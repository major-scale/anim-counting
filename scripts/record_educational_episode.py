#!/usr/bin/env python3
"""
Record episode data for the educational visualization.

Runs the randproj model through the counting environment, records all states
(env state, hidden state, prior/posterior, probe output, transitions),
and saves as a .npz file.

Usage:
    python3 record_educational_episode.py --out episode_data.npz
    python3 record_educational_episode.py --out episode_data.npz --n-blobs 13 --seed 7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# RSSM (from advanced_manifold_analysis.py)
# ---------------------------------------------------------------------------

STOCH_DIM = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_DIM * STOCH_CLASSES
DETER_DIM = 512


def _ln(x, w, b, eps=1e-5):
    mu = x.mean()
    var = x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _gumbel_sample(logits):
    uniform = np.random.uniform(1e-5, 1 - 1e-5, logits.shape).astype(np.float32)
    gumbel = -np.log(-np.log(uniform))
    sample = np.zeros_like(logits)
    idx = (logits + gumbel).argmax(axis=-1)
    for i, j in enumerate(idx):
        sample[i, j] = 1.0
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return (sample + probs - probs).flatten().astype(np.float32)


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


class FastRSSMWithPrior:
    GRU_UPDATE_BIAS = -1.0

    def __init__(self, weights):
        self.w = weights
        self.obs_size = weights["enc_linear0_w"].shape[1]
        self.reset()

    def reset(self):
        self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        self._compute_prior()

    def _compute_prior(self):
        w = self.w
        h = _silu(_ln(w["img_out_w"] @ self.deter,
                       w["img_out_norm_w"], w["img_out_norm_b"]))
        logits = (w["imgs_stat_w"] @ h + w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        self.stoch = _gumbel_sample(logits)

    def step(self, obs_raw, action=0.0):
        w = self.w
        # Encode
        x = np.sign(obs_raw) * np.log(np.abs(obs_raw) + 1)
        h = _silu(_ln(w["enc_linear0_w"] @ x, w["enc_norm0_w"], w["enc_norm0_b"]))
        h = _silu(_ln(w["enc_linear1_w"] @ h, w["enc_norm1_w"], w["enc_norm1_b"]))
        embed = _silu(_ln(w["enc_linear2_w"] @ h, w["enc_norm2_w"], w["enc_norm2_b"]))

        # img_step
        act = np.zeros(1, dtype=np.float32) if self.is_first else np.atleast_1d(np.float32(action))
        self.is_first = False
        cat_in = np.concatenate([self.stoch.copy(), act])
        h_in = _silu(_ln(w["img_in_w"] @ cat_in, w["img_in_norm_w"], w["img_in_norm_b"]))

        # GRU
        combined = np.concatenate([h_in, self.deter])
        out = w["gru_w"] @ combined
        out = _ln(out, w["gru_norm_w"], w["gru_norm_b"])
        N = DETER_DIM
        reset = _sigmoid(out[:N])
        cand = np.tanh(reset * out[N:2*N])
        update = _sigmoid(out[2*N:] + self.GRU_UPDATE_BIAS)
        self.deter = (update * cand + (1 - update) * self.deter).astype(np.float32)

        # Prior
        h_prior = _silu(_ln(w["img_out_w"] @ self.deter,
                             w["img_out_norm_w"], w["img_out_norm_b"]))
        prior_logits = (w["imgs_stat_w"] @ h_prior + w["imgs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        prior_probs = np.exp(prior_logits - prior_logits.max(axis=-1, keepdims=True))
        prior_probs /= prior_probs.sum(axis=-1, keepdims=True)
        prior_stoch = prior_probs.flatten().copy()

        # Posterior
        cat_post = np.concatenate([self.deter, embed])
        h_post = _silu(_ln(w["obs_out_w"] @ cat_post,
                            w["obs_out_norm_w"], w["obs_out_norm_b"]))
        post_logits = (w["obs_stat_w"] @ h_post + w["obs_stat_b"]).reshape(
            STOCH_DIM, STOCH_CLASSES)
        post_probs = np.exp(post_logits - post_logits.max(axis=-1, keepdims=True))
        post_probs /= post_probs.sum(axis=-1, keepdims=True)

        self.stoch = _gumbel_sample(post_logits)
        return self.deter.copy(), prior_stoch, post_stoch.flatten().copy() if False else post_probs.flatten().copy(), embed.copy()


# ---------------------------------------------------------------------------
# Episode recording
# ---------------------------------------------------------------------------

def record_episode(weights_dir, seed=0, n_blobs=25):
    """Run one episode and record everything."""
    from counting_env_pure import CountingWorldEnv

    weights = load_exported_weights(weights_dir)
    obs_size = weights["enc_linear0_w"].shape[1]
    model = FastRSSMWithPrior(weights)

    # Load probe
    probe_path = Path(weights_dir) / "embed_probe.json"
    with open(probe_path) as f:
        probe_data = json.load(f)
    probe_w = np.array(probe_data["weights"], dtype=np.float32)
    probe_b = float(probe_data["bias"])

    # Random projection matrix (same seed as training)
    from scipy.stats import ortho_group
    proj_matrix = ortho_group.rvs(obs_size,
                                  random_state=np.random.RandomState(42_000)).astype(np.float32)

    env = CountingWorldEnv(blob_count=n_blobs, seed=seed)
    state = env.reset()
    model.reset()

    # Storage
    frames = {
        "deter": [],         # (T, 512)
        "prior_stoch": [],   # (T, 1024)
        "post_stoch": [],    # (T, 1024)
        "embed": [],         # (T, 512) encoder embed
        "gt_count": [],      # (T,)
        "probe_pred": [],    # (T,) continuous probe output
        "prior_probe": [],   # (T,) probe applied to prior-derived deter (same deter, different intent)
        "bot_x": [],         # (T,)
        "bot_y": [],         # (T,)
        "bot_dx": [],        # (T,) facing direction
        "bot_dy": [],        # (T,)
        "blob_x": [],        # (T, n_blobs)
        "blob_y": [],        # (T, n_blobs)
        "blob_on_grid": [],  # (T, n_blobs) bool
        "blob_animating": [],  # (T, n_blobs) bool
        "grid_filled": [],   # (T, 25) bool per slot
        "phase": [],         # (T,) 0=counting, 1=unmarking, 2=predict
        "obs_raw": [],       # (T, obs_size) raw obs before projection
        "transition": [],    # (T,) bool - did count change this frame?
    }

    done = False
    prev_count = 0
    step_i = 0

    while not done:
        es = env._state  # episode state

        # Environment state
        bx, by = es.bot.pos_x, es.bot.pos_y
        # Facing direction from velocity
        vx, vy = es.bot.vel_x, es.bot.vel_y
        speed = max(np.sqrt(vx**2 + vy**2), 1e-8)
        dx, dy = vx / speed, vy / speed

        blob_xs = [b.pos_x for b in es.blobs]
        blob_ys = [b.pos_y for b in es.blobs]
        blob_grids = [b.grid_slot is not None for b in es.blobs]
        blob_anims = [b.animating for b in es.blobs]
        grid_occ = [s >= 0 for s in es.grid.occupancy]
        gt_count = es.grid.filled_count

        phase_val = 0
        if es.phase == "unmarking":
            phase_val = 1
        elif es.phase == "predict":
            phase_val = 2

        is_transition = (gt_count != prev_count)

        # RSSM step
        obs = state[:obs_size].astype(np.float32)
        obs_raw_copy = obs.copy()
        obs_proj = (proj_matrix @ obs)[:obs_size]
        deter, prior_s, post_s, embed = model.step(obs_proj, 0.0)

        # Probe predictions
        probe_val = float(deter @ probe_w + probe_b)
        # For prior probe: we use the same deter (prior doesn't change deter,
        # it's posterior that feeds back via stoch). So prior_probe = probe_val.
        # The INFORMATIVE comparison is prior_stoch vs post_stoch applied to obs_stat.
        # But for the viz, we show probe on deter which is the main signal.
        prior_probe_val = probe_val  # Same deter, prior only affects stoch

        frames["deter"].append(deter)
        frames["prior_stoch"].append(prior_s)
        frames["post_stoch"].append(post_s)
        frames["embed"].append(embed)
        frames["gt_count"].append(gt_count)
        frames["probe_pred"].append(probe_val)
        frames["prior_probe"].append(prior_probe_val)
        frames["bot_x"].append(bx)
        frames["bot_y"].append(by)
        frames["bot_dx"].append(dx)
        frames["bot_dy"].append(dy)
        frames["blob_x"].append(blob_xs + [0.0] * (n_blobs - len(blob_xs)))
        frames["blob_y"].append(blob_ys + [0.0] * (n_blobs - len(blob_ys)))
        frames["blob_on_grid"].append(blob_grids + [False] * (n_blobs - len(blob_grids)))
        frames["blob_animating"].append(blob_anims + [False] * (n_blobs - len(blob_anims)))
        frames["grid_filled"].append(grid_occ + [False] * (25 - len(grid_occ)))
        frames["phase"].append(phase_val)
        frames["obs_raw"].append(obs_raw_copy)
        frames["transition"].append(is_transition)

        prev_count = gt_count
        state, reward, done, info = env.step(0)  # passive agent
        step_i += 1

    # Convert to arrays
    result = {}
    for k, v in frames.items():
        result[k] = np.array(v)

    # Add metadata
    result["n_blobs"] = np.array([n_blobs])
    result["seed"] = np.array([seed])
    result["n_frames"] = np.array([step_i])
    result["world_width"] = np.array([1400.0])
    result["world_height"] = np.array([1000.0])

    # Add probe weights for the viz to use
    result["probe_weights"] = probe_w
    result["probe_bias"] = np.array([probe_b])

    return result


def main():
    parser = argparse.ArgumentParser(description="Record educational episode data")
    parser.add_argument("--weights-dir", default="/workspace/bridge/models/randproj_clean",
                        help="Path to model weights directory")
    parser.add_argument("--out", default="/workspace/bridge/artifacts/episodes/randproj_episode.npz",
                        help="Output .npz file path")
    parser.add_argument("--seed", type=int, default=7,
                        help="Environment seed")
    parser.add_argument("--n-blobs", type=int, default=25,
                        help="Number of blobs")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording episode: seed={args.seed}, blobs={args.n_blobs}")
    print(f"  Weights: {args.weights_dir}")
    print(f"  Output:  {args.out}")

    data = record_episode(args.weights_dir, seed=args.seed, n_blobs=args.n_blobs)

    n = int(data["n_frames"][0])
    n_transitions = int(data["transition"].sum())
    final_count = int(data["gt_count"][-1])
    print(f"  Recorded {n} frames, {n_transitions} transitions, final count={final_count}")

    np.savez_compressed(args.out, **data)
    print(f"  Saved to {args.out} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
