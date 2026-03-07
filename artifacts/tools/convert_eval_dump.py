"""
Convert eval_dump .npy output files to trajectories.npz format
used by successor_battery.py, fit_umap.py, etc.

Usage:
    python convert_eval_dump.py <eval_dump_dir> [output_path]

If output_path is omitted, writes to data/trajectories.npz (overwriting existing).
"""

import sys
import pathlib
import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"


def convert(eval_dir, output_path=None):
    eval_dir = pathlib.Path(eval_dir)
    if output_path is None:
        output_path = DATA_DIR / "trajectories.npz"
    else:
        output_path = pathlib.Path(output_path)

    print(f"Converting eval_dump from {eval_dir}")

    h_t = np.load(eval_dir / "h_t.npy")
    z_t = np.load(eval_dir / "z_t.npy")
    counts = np.load(eval_dir / "counts.npy")
    actor_predictions = np.load(eval_dir / "actor_predictions.npy")
    rewards = np.load(eval_dir / "rewards.npy")
    bot_positions = np.load(eval_dir / "bot_positions.npy")
    timesteps = np.load(eval_dir / "timesteps.npy")
    episodes = np.load(eval_dir / "episodes.npy")
    total_blobs = np.load(eval_dir / "total_blobs.npy")

    # Detect marking events: count increases from one step to the next
    # within the same episode
    is_marking = np.zeros(len(counts), dtype=bool)
    for i in range(1, len(counts)):
        if episodes[i] == episodes[i - 1] and counts[i] > counts[i - 1]:
            is_marking[i] = True

    n = len(counts)
    print(f"  Timesteps: {n}")
    print(f"  Episodes: {len(np.unique(episodes))}")
    print(f"  h_t shape: {h_t.shape}")
    print(f"  Count range: {counts.min()} - {counts.max()}")
    print(f"  Total blobs range: {total_blobs.min()} - {total_blobs.max()}")
    print(f"  Marking events: {is_marking.sum()}")

    np.savez_compressed(
        output_path,
        h_t=h_t,
        z_t=z_t,
        true_count=counts,
        actor_prediction=actor_predictions,
        reward=rewards,
        bot_x=bot_positions[:, 0],
        bot_y=bot_positions[:, 1],
        is_marking_event=is_marking,
        episode_id=episodes,
        timestep=timesteps,
        total_blobs=total_blobs,
    )
    print(f"  Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_eval_dump.py <eval_dump_dir> [output_path]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    convert(sys.argv[1], out)
