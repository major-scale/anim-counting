"""
Generate zoomed trajectory snapshots — key moments around marking events.
4 panels per row, bigger, focusing on transitions.
"""
import os, pathlib, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def make_zoomed_snapshots(data, ep_idx, config, n_panels=12):
    ep = data["episodes"][ep_idx]
    bg = data["background"]
    centers = data["cluster_centers"][config]

    ep_x = ep[f"x_{config}"]
    ep_y = ep[f"y_{config}"]
    bg_x = bg[f"x_{config}"]
    bg_y = bg[f"y_{config}"]

    # Pick key moments: start, marking events ± some buffer, end
    marks = [i for i, m in enumerate(ep["is_marking"]) if m]
    key_steps = [0]
    for m in marks:
        key_steps.extend([max(0, m - 5), m, min(ep["length"] - 1, m + 5)])
    key_steps.append(ep["length"] - 1)
    # Add some evenly-spaced filler
    filler = list(range(0, ep["length"], ep["length"] // 8))
    key_steps = sorted(set(key_steps + filler))

    # Thin to n_panels
    if len(key_steps) > n_panels:
        indices = np.linspace(0, len(key_steps) - 1, n_panels, dtype=int)
        key_steps = [key_steps[i] for i in indices]

    cols = 4
    rows = (len(key_steps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    config_label = "PCA" if config == "pca" else f"UMAP ({config})"

    for idx, step in enumerate(key_steps):
        r, c_ = divmod(idx, cols)
        ax = axes[r, c_]

        # Background
        bg_colors = [COUNT_COLORS[c] for c in bg["count"]]
        ax.scatter(bg_x, bg_y, c=bg_colors, s=2, alpha=0.2, rasterized=True)

        # Centers
        for k, pos in centers.items():
            ax.scatter(pos[0], pos[1], c=COUNT_COLORS[int(k)], s=150,
                       edgecolors="white", linewidths=1.5, zorder=8)
            ax.text(pos[0], pos[1], k, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=9)

        # Connecting lines between centers
        sorted_keys = sorted(centers.keys(), key=int)
        for i in range(len(sorted_keys) - 1):
            k1, k2 = sorted_keys[i], sorted_keys[i + 1]
            ax.plot([centers[k1][0], centers[k2][0]],
                    [centers[k1][1], centers[k2][1]],
                    "w--", alpha=0.3, linewidth=1)

        # Full path so far (faded)
        if step > 0:
            ax.plot(ep_x[:step+1], ep_y[:step+1], color="white", alpha=0.1, linewidth=0.5)

        # Trail (last 40 steps)
        trail_start = max(0, step - 40)
        for i in range(trail_start, step):
            alpha = 0.15 + 0.85 * ((i - trail_start) / max(1, step - trail_start))
            lw = 1 + 2 * ((i - trail_start) / max(1, step - trail_start))
            ax.plot([ep_x[i], ep_x[i+1]], [ep_y[i], ep_y[i+1]],
                    color=COUNT_COLORS[ep["count"][i+1]], alpha=alpha, linewidth=lw)

        # Marking events in trail
        for i in range(trail_start, step + 1):
            if ep["is_marking"][i]:
                ax.scatter(ep_x[i], ep_y[i], s=100, facecolors="none",
                           edgecolors="#ffdd57", linewidths=2.5, zorder=9)

        # Cursor
        cx, cy = ep_x[step], ep_y[step]
        cc = COUNT_COLORS[ep["count"][step]]
        ax.scatter(cx, cy, c=cc, s=200, edgecolors="white", linewidths=2.5, zorder=10)

        count = ep["count"][step]
        pred = ep["prediction"][step]
        err = abs(pred - count)
        is_mark = " [MARK]" if ep["is_marking"][step] else ""
        ax.set_title(f"t={step}  count={count}  pred={pred:.2f}{is_mark}",
                     fontsize=10, fontweight="bold" if is_mark else "normal", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#1a1a2e")

    for idx in range(len(key_steps), rows * cols):
        r, c_ = divmod(idx, cols)
        axes[r, c_].set_visible(False)

    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"{config_label} — Episode {ep['id']} ({ep['blob_count']} blobs, {ep['length']} steps)\n"
        f"Marking events: {len(marks)} | Mean error: {ep['mean_error']:.3f}",
        fontsize=14, color="white", y=1.01
    )
    plt.tight_layout()
    return fig

def main():
    data = load()
    episodes = data["episodes"]

    # Pick episodes
    picks = []
    for i, ep in enumerate(episodes):
        if "reaches-8" in ep["tags"]:
            picks.append(("reaches8", i))
            break
    for i, ep in enumerate(episodes):
        if ep["blob_count"] == 7 and i not in [p[1] for p in picks]:
            picks.append(("7blob", i))
            break
    for i, ep in enumerate(episodes):
        if ep["blob_count"] == 5 and i not in [p[1] for p in picks]:
            picks.append(("5blob", i))
            break

    for config in ["pca", "default"]:
        for label, ep_idx in picks:
            ep = episodes[ep_idx]
            print(f"  {config} / {label} (ep {ep['id']}, {ep['blob_count']} blobs)...")
            fig = make_zoomed_snapshots(data, ep_idx, config, n_panels=12)
            fname = f"zoomed_{config}_{label}_ep{ep['id']}.png"
            fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
            plt.close(fig)
            print(f"    Saved {fname}")

    print("Done!")

if __name__ == "__main__":
    main()
