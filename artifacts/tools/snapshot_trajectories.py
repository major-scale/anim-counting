"""
Generate trajectory progression snapshots — PCA and UMAP default.
Shows cursor + trail at every 50 steps for selected episodes.
"""
import os, pathlib, numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"
FIG_DIR = _SCRIPT_DIR / "figures"

COUNT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

def load():
    with open(DATA_DIR / "brain_explorer_data.json") as f:
        return json.load(f)

def make_snapshot_grid(data, ep_idx, config, step_interval=50):
    """Generate a grid of snapshots for one episode in one embedding."""
    ep = data["episodes"][ep_idx]
    bg = data["background"]
    centers = data["cluster_centers"][config]

    ep_x = ep[f"x_{config}"]
    ep_y = ep[f"y_{config}"]
    bg_x = bg[f"x_{config}"]
    bg_y = bg[f"y_{config}"]

    steps = list(range(0, ep["length"], step_interval))
    if steps[-1] != ep["length"] - 1:
        steps.append(ep["length"] - 1)

    n = len(steps)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    config_label = "PCA" if config == "pca" else f"UMAP ({config})"

    for idx, step in enumerate(steps):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        # Background
        bg_colors = [COUNT_COLORS[c_] for c_ in bg["count"]]
        ax.scatter(bg_x, bg_y, c=bg_colors, s=1, alpha=0.15, rasterized=True)

        # Cluster centers
        for k, pos in centers.items():
            ax.scatter(pos[0], pos[1], c=COUNT_COLORS[int(k)], s=80,
                       edgecolors="white", linewidths=1, zorder=8)
            ax.text(pos[0], pos[1], k, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=9)

        # Trail (last 30 steps)
        trail_start = max(0, step - 30)
        tx = ep_x[trail_start:step + 1]
        ty = ep_y[trail_start:step + 1]
        tc = [ep["count"][i] for i in range(trail_start, step + 1)]
        for i in range(len(tx) - 1):
            alpha = 0.2 + 0.8 * (i / len(tx))
            ax.plot([tx[i], tx[i+1]], [ty[i], ty[i+1]],
                    color=COUNT_COLORS[tc[i+1]], alpha=alpha, linewidth=1.5)

        # Mark events in trail
        for i in range(trail_start, step + 1):
            if ep["is_marking"][i]:
                ax.scatter(ep_x[i], ep_y[i], s=60, facecolors="none",
                           edgecolors="#ffdd57", linewidths=2, zorder=9)

        # Cursor
        cx, cy = ep_x[step], ep_y[step]
        cc = COUNT_COLORS[ep["count"][step]]
        ax.scatter(cx, cy, c=cc, s=120, edgecolors="white", linewidths=2, zorder=10)

        count = ep["count"][step]
        pred = ep["prediction"][step]
        err = abs(pred - count)
        ax.set_title(f"t={step}  count={count}  pred={pred:.1f}  err={err:.2f}",
                     fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"{config_label} — Episode {ep['id']} ({ep['blob_count']} blobs, {ep['length']} steps)\n"
        f"Tags: {', '.join(ep['tags']) if ep['tags'] else 'none'} | "
        f"Mean error: {ep['mean_error']:.3f}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    return fig


def main():
    data = load()
    episodes = data["episodes"]

    # Pick interesting episodes: one with high blob count, one clean, one reaches-8
    picks = []
    # Find a clean 5-blob episode
    for i, ep in enumerate(episodes):
        if ep["blob_count"] == 5 and "clean" in ep["tags"]:
            picks.append(("clean_5blob", i))
            break
    # Find a reaches-8 episode
    for i, ep in enumerate(episodes):
        if "reaches-8" in ep["tags"]:
            picks.append(("reaches8", i))
            break
    # Find a 7-blob episode
    for i, ep in enumerate(episodes):
        if ep["blob_count"] == 7:
            picks.append(("7blob", i))
            break

    for config in ["pca", "default"]:
        for label, ep_idx in picks:
            ep = episodes[ep_idx]
            print(f"Generating {config} snapshots for ep {ep['id']} ({label}, {ep['blob_count']} blobs, {ep['length']} steps)...")
            fig = make_snapshot_grid(data, ep_idx, config, step_interval=50)
            fname = f"trajectory_{config}_{label}_ep{ep['id']}.png"
            fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
