"""
Regenerate the binary-baseline battery with the 14→15 transition included.

Root cause of the original gap: the collector in
jamstack-v1/bridge/scripts/quick_ghe_binary.py:collect_episodes (L248-272) reads
env state BEFORE stepping, inside a `while not done` loop. The step that
transitions count 14→15 simultaneously sets done=True, so count=15 is never
appended. This one-line ordering fix captures the terminal state.

This script regenerates ONLY the raw data fields (h_t, counts, bits,
carry_active, episode_ids, timesteps, n_episodes). The result_* metric fields
are copied verbatim from the prior battery — they were computed on 14-transition
data and are slightly stale relative to the new raw fields. Downstream analyses
that depend on result_* should be re-run separately.
"""

import sys
from pathlib import Path
import numpy as np

JAMSTACK_SCRIPTS = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/scripts")
CHECKPOINT_DIR = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/artifacts/checkpoints/binary_baseline_s0/exported")
OLD_BATTERY = Path("/Users/petermurphy/anima-bridge/artifacts/battery/binary_baseline_s0/battery_v1_14transitions.npz")
OUT_PATH = Path("/Users/petermurphy/anima-bridge/artifacts/battery/binary_baseline_s0/battery.npz")

sys.path.insert(0, str(JAMSTACK_SCRIPTS))
from binary_counting_env import BinaryCountingEnv, NUM_COLUMNS, PHASE_IDLE
from quick_ghe_binary import load_exported_weights, FastRSSM

N_EPISODES = 15
MAX_STEPS = 2000
BASE_SEED = 0


def collect_episodes_fixed(weights, n_episodes, max_steps, base_seed):
    model = FastRSSM(weights)

    all_h_t, all_counts, all_bits = [], [], []
    all_carry_active, all_episode_ids, all_timesteps = [], [], []

    for ep in range(n_episodes):
        env = BinaryCountingEnv(
            n_blobs=15, max_steps=max_steps, seed=base_seed + ep * 1000)
        obs = env.reset()
        model.reset()

        done = False
        step = 0
        while True:
            state = env._state
            decimal_count = state.decimal_count
            bits = [1 if state.columns[i].occupied else 0 for i in range(NUM_COLUMNS)]
            carry_active = 1 if state.carry.phase != PHASE_IDLE else 0

            obs_for_model = obs[:model.obs_size].astype(np.float32)
            deter, _ = model.step(obs_for_model, action=0)

            all_h_t.append(deter)
            all_counts.append(decimal_count)
            all_bits.append(bits)
            all_carry_active.append(carry_active)
            all_episode_ids.append(ep)
            all_timesteps.append(step)

            if done:
                break
            obs, _, done, info = env.step(0)
            step += 1

        env.close()
        print(f"  Episode {ep}: {step+1} steps, final_count={info['decimal_count']}, "
              f"last_captured_count={decimal_count}")

    return {
        "h_t": np.array(all_h_t, dtype=np.float32),
        "counts": np.array(all_counts, dtype=np.int32),
        "bits": np.array(all_bits, dtype=np.int32),
        "carry_active": np.array(all_carry_active, dtype=np.int32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
        "n_episodes": np.array([n_episodes], dtype=np.int64),
    }


def main():
    print(f"Loading weights from {CHECKPOINT_DIR}")
    weights = load_exported_weights(CHECKPOINT_DIR)

    print(f"Running {N_EPISODES} episodes (max_steps={MAX_STEPS})")
    raw = collect_episodes_fixed(weights, N_EPISODES, MAX_STEPS, BASE_SEED)

    print(f"\nSummary:")
    print(f"  total samples: {raw['h_t'].shape[0]}")
    print(f"  unique counts: {sorted(set(raw['counts'].tolist()))}")
    stable_mask = raw['carry_active'] == 0
    print(f"  stable samples: {int(stable_mask.sum())}")
    print(f"  stable samples per count: "
          f"{[(int(c), int(((raw['counts']==c) & stable_mask).sum())) for c in sorted(set(raw['counts'].tolist()))]}")

    print(f"\nCopying result_* metric fields verbatim from old battery (flagged as stale)")
    old = np.load(OLD_BATTERY)
    save_data = dict(raw)
    for k in old.files:
        if k.startswith("result_"):
            save_data[k] = old[k]

    print(f"Writing new battery: {OUT_PATH}")
    np.savez_compressed(OUT_PATH, **save_data)
    print(f"  size: {OUT_PATH.stat().st_size/1e6:.1f} MB")
    print(f"  backup preserved at: {OLD_BATTERY}")


if __name__ == "__main__":
    main()
