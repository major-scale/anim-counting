#!/usr/bin/env python3
"""
Generate all publication-quality figures for CCN paper.
Output: bridge/figures/*.png (300 DPI, poster-readable fonts)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Paths
ART_DIR = Path("/workspace/bridge/artifacts/binary_successor")
FIG_DIR = Path("/workspace/bridge/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
BATTERY_PATH = Path("/workspace/projects/jamstack-v1/bridge/artifacts/battery/binary_baseline_s0/battery.npz")

# Global style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

BIT_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']  # blue, orange, green, red
BIT_LABELS = ['Bit 0 (LSB)', 'Bit 1', 'Bit 2', 'Bit 3 (MSB)']


def load_json(name):
    with open(ART_DIR / name) as f:
        return json.load(f)


# ============================================================
# FIGURE 2: Imagination vs Posterior (PRIORITY 1)
# ============================================================
def figure2_imagination():
    """Side-by-side 7->8 cascade: posterior vs imagination."""
    data = load_json("imagination_rollout_colstate.json")
    d3 = data["depth_3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, mode, title, span_val in [
        (ax1, "post", "Posterior (with observations)", d3["post_span"]),
        (ax2, "imag", "Imagination (no observations)", d3["imag_span"]),
    ]:
        crossings = d3[f"{mode}_crossings"]

        # Plot each bit's crossing times as a distribution
        for bit_idx in range(4):
            bit_key = str(bit_idx)
            if bit_key in crossings:
                times = np.array(crossings[bit_key])
                mean_t = times.mean()
                std_t = times.std()

                # Horizontal bar showing mean +/- std
                ax.barh(bit_idx, std_t * 2, left=mean_t - std_t,
                        height=0.6, color=BIT_COLORS[bit_idx], alpha=0.3,
                        edgecolor=BIT_COLORS[bit_idx], linewidth=1.5)
                # Mean marker
                ax.plot(mean_t, bit_idx, 'o', color=BIT_COLORS[bit_idx],
                        markersize=8, zorder=5)
                # Individual points
                ax.plot(times, np.full_like(times, bit_idx), '|',
                        color=BIT_COLORS[bit_idx], markersize=6, alpha=0.3)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Steps relative to cascade completion")
        ax.set_yticks(range(4))
        ax.set_yticklabels(BIT_LABELS)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlim(-14, 2)

        # Annotate span
        ax.text(0.97, 0.03, f"Span: {span_val:.1f} steps",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Gray background for imagination panel
    ax2.set_facecolor('#f5f5f5')

    # Arrow showing LSB->MSB direction
    ax1.annotate('LSB then MSB\nsequence', xy=(-10, 0.3), xytext=(-6, 3.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    fig.suptitle("7 to 8 Carry Cascade: Physical Process vs Mental Simulation",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_imagination_vs_posterior.png")
    plt.close()
    print("  Saved fig2_imagination_vs_posterior.png")


# ============================================================
# FIGURE 4: ESN Control (PRIORITY 2)
# ============================================================
def figure4_esn():
    """Grouped bar chart: RSSM vs ESN."""
    esn = load_json("esn_control.json")
    agg = esn["aggregate"]

    metrics = ['Probe\nAccuracy', 'RSA\nOrdinal', 'RSA\nHamming', 'Imagination\n(from count 7)']
    rssm_vals = [1.000, 0.713, 0.811, 1.000]  # RSSM values
    esn_vals = [
        agg["count_accuracy"]["mean"],
        agg["rsa_ordinal"]["mean"],
        agg["rsa_hamming"]["mean"],
        0.0,  # ESN imagination fails completely
    ]
    esn_errs = [
        agg["count_accuracy"]["std"],
        agg["rsa_ordinal"]["std"],
        agg["rsa_hamming"]["std"],
        0.0,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, rssm_vals, width, label='Trained RSSM',
                   color='#2196F3', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, esn_vals, width, yerr=esn_errs,
                   label='Echo State Network', color='#9E9E9E',
                   edgecolor='white', linewidth=1, capsize=4)

    ax.set_ylabel('Score')
    ax.set_title('Trained RSSM vs Random Reservoir (ESN)', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars1, rssm_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, esn_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Highlight imagination failure
    ax.annotate('Complete\nfailure', xy=(3 + width/2, 0.05),
                fontsize=11, ha='center', color='#D32F2F', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_esn_control.png")
    plt.close()
    print("  Saved fig4_esn_control.png")


# ============================================================
# FIGURE 1: Bit-Axis Heatmap (PRIORITY 3)
# ============================================================
def figure1_heatmap():
    """4-row heatmap of bit activations across an episode."""
    battery = np.load(BATTERY_PATH, allow_pickle=True)
    bits = battery['bits']  # (13280, 4)
    counts = battery['counts']  # (13280,)

    # Use first episode (~885 steps)
    ep_size = len(bits) // 15
    ep_bits = bits[:ep_size]
    ep_counts = counts[:ep_size]

    # Find transition points
    transitions = np.where(np.diff(ep_counts) != 0)[0]

    # Build heatmap: 4 rows (bit 3 top, bit 0 bottom), columns = timesteps
    heatmap_data = ep_bits.T[::-1]  # Reverse so MSB is on top

    # Custom colormap: red (0) to green (1)
    cmap = LinearSegmentedColormap.from_list('bit', ['#EF5350', '#FFEE58', '#66BB6A'], N=256)

    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(heatmap_data.astype(float), aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0, vmax=1)

    # Transition lines
    for t in transitions:
        ax.axvline(t, color='white', linewidth=0.3, alpha=0.5)

    # Count labels (place at midpoint between transitions)
    all_boundaries = np.concatenate([[0], transitions, [len(ep_counts)-1]])
    for i in range(len(all_boundaries) - 1):
        mid = (all_boundaries[i] + all_boundaries[i+1]) / 2
        count_val = ep_counts[int(all_boundaries[i]) + 1] if i > 0 else ep_counts[0]
        if all_boundaries[i+1] - all_boundaries[i] > 15:  # Only label if enough space
            ax.text(mid, -0.7, str(int(count_val)), ha='center', va='top',
                    fontsize=7, color='black')

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Bit 3 (MSB)', 'Bit 2', 'Bit 1', 'Bit 0 (LSB)'])
    ax.set_xlabel('Timestep')
    ax.set_title('Binary Counter in Neural Dynamics: One Full Episode', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['OFF', 'ON'])

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bit_heatmap.png")
    plt.close()
    print("  Saved fig1_bit_heatmap.png")


# ============================================================
# FIGURE 3: Observation Cliff
# ============================================================
def figure3_cliff():
    """The dramatic cliff from continuous to interrupted observation."""
    stress = load_json("imagination_stress_test.json")
    cliff = load_json("observation_cliff.json")

    # Extract periodic peek data
    peeks = stress["exp3_periodic_peeks"]
    conditions = ['0', '10', '25', '50', '100']
    labels = ['Continuous', 'Every 10', 'Every 25', 'Every 50', 'Every 100']
    accs = [peeks[c]["mean_step_accuracy"] for c in conditions]
    stds = [peeks[c]["std_step_accuracy"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: The cliff
    colors = ['#4CAF50'] + ['#F44336'] * 4
    bars = ax1.bar(range(len(labels)), accs, yerr=stds, capsize=4,
                   color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_ylabel('Step Accuracy')
    ax1.set_title('Observation Cliff', fontweight='bold')
    ax1.set_ylim(0, 1.05)

    # Annotate the cliff
    ax1.annotate('', xy=(1, accs[1] + 0.05), xytext=(0, accs[0] - 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(0.5, 0.6, f'{accs[0]:.1%} -> {accs[1]:.1%}',
            ha='center', fontsize=12, color='red', fontweight='bold')

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, accs)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + stds[i] + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    # Panel 2: Multi-peek recovery plateau
    mp = cliff["multi_peek"]
    peek_counts = sorted(mp.keys(), key=int)
    peek_dists = [mp[k]["dist_correct"] for k in peek_counts]

    ax2.plot([int(k) for k in peek_counts], peek_dists, 'o-',
             color='#F44336', linewidth=2, markersize=8)
    ax2.axhline(6.77, color='gray', linestyle='--', alpha=0.5, label='Mean inter-centroid dist')
    ax2.axhline(6.45, color='red', linestyle=':', alpha=0.7, label='Plateau (~6.4)')
    ax2.set_xlabel('Number of consecutive peeks')
    ax2.set_ylabel('Distance to correct centroid')
    ax2.set_title('Multi-Peek Recovery: Permanent Off-Manifold', fontweight='bold')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_observation_cliff.png")
    plt.close()
    print("  Saved fig3_observation_cliff.png")


# ============================================================
# FIGURE 5: Candidate Explanation Scorecard
# ============================================================
def figure5_scorecard():
    """Visual scorecard of 13 claims tested."""
    claims = [
        ("RSSM simulates carry cascades", "Column-state probes, prior-only rollout", True),
        ("Learned dynamics required", "ESN control (random dynamics, trained readout)", True),
        ("Bit axes encode disjointly", "MI analysis (512-d x 4-bit)", True),
        ("Variance scales with depth", "Idle variance vs carry depth", True),
        ("Count 14 is terminal attractor", "Jacobian eigenvalue analysis", True),
        ("Off-manifold drift causes cliff", "Drift + multi-peek + gate analysis", True),
        ("AR1 tracks cascade depth", "AR1 vs carry depth correlation", False),
        ("Spectral radius predicts depth", "Jacobian at all 15 centroids", False),
        ("GRU gates close when blind", "Gate comparison across 3 conditions", False),
        ("Multi-peek recovers state", "Sequential peek recovery (1-20)", False),
        ("Non-normal amplification", "Henrici departure from normality", False),
        ("Variance aligns with all bits", "Per-bit directional projection", False),
        ("Variance-depth = timing artifact", "Partial correlation (idle duration)", False),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(claims) - 0.5)
    ax.invert_yaxis()
    ax.axis('off')

    # Header
    ax.text(0.05, -0.8, 'Claim', fontweight='bold', fontsize=13, va='center')
    ax.text(5.5, -0.8, 'Method', fontweight='bold', fontsize=13, va='center')
    ax.text(9.5, -0.8, '', fontweight='bold', fontsize=13, va='center', ha='center')

    for i, (claim, method, confirmed) in enumerate(claims):
        # Alternating background
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i - 0.4), 10, 0.8,
                        facecolor='#f5f5f5', edgecolor='none'))

        # Separator between confirmed and killed
        if i == 6:
            ax.axhline(i - 0.45, color='gray', linewidth=1.5, linestyle='-')

        # Claim text
        ax.text(0.05, i, claim, fontsize=11, va='center',
                color='#1B5E20' if confirmed else '#B71C1C')

        # Method text
        ax.text(5.5, i, method, fontsize=9.5, va='center', color='#555')

        # Icon
        if confirmed:
            ax.text(9.5, i, 'CONFIRMED', fontsize=10, va='center', ha='center',
                    color='#4CAF50', fontweight='bold')
        else:
            ax.text(9.5, i, 'KILLED', fontsize=10, va='center', ha='center',
                    color='#F44336', fontweight='bold')

    ax.set_title('Systematic Evaluation: 6 Confirmed, 7 Eliminated',
                 fontweight='bold', fontsize=15, pad=20)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_scorecard.png")
    plt.close()
    print("  Saved fig5_scorecard.png")


# ============================================================
# FIGURE 6: PCA Trajectory
# ============================================================
def figure6_pca():
    """2D PCA trajectory through count states."""
    data = load_json("full_pipeline_trace.json")
    centroids = data["pca"]["centroids"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot centroids
    xs = [centroids[str(c)][0] for c in range(15)]
    ys = [centroids[str(c)][1] for c in range(15)]

    # Color by count
    colors_map = plt.cm.viridis(np.linspace(0, 1, 15))

    # Draw trajectory arrows
    for i in range(14):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

    # Plot points
    for c in range(15):
        marker = '*' if c == 14 else 'o'
        size = 200 if c == 14 else 120
        edgecolor = 'red' if c == 14 else 'black'
        lw = 2 if c == 14 else 0.5
        ax.scatter(xs[c], ys[c], c=[colors_map[c]], s=size, marker=marker,
                  edgecolors=edgecolor, linewidth=lw, zorder=5)
        # Label
        ax.annotate(str(c), (xs[c], ys[c]), textcoords="offset points",
                   xytext=(8, 8), fontsize=10, fontweight='bold')

    # Highlight 7->8 (longest path, all bits flip)
    ax.annotate('', xy=(xs[8], ys[8]), xytext=(xs[7], ys[7]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    mid_x = (xs[7] + xs[8]) / 2
    mid_y = (ys[7] + ys[8]) / 2
    ax.text(mid_x + 0.3, mid_y + 0.3, '7->8\n(full cascade)',
            fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('PC1 (18.0%)')
    ax.set_ylabel('PC2 (13.0%)')
    ax.set_title('Hidden State Trajectory Through Count Space (PCA)',
                 fontweight='bold')

    # Legend for count 14
    ax.plot([], [], '*', color='gray', markersize=15, markeredgecolor='red',
            markeredgewidth=2, label='Count 14 (terminal attractor)')
    ax.legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_pca_trajectory.png")
    plt.close()
    print("  Saved fig6_pca_trajectory.png")


# ============================================================
# FIGURE 7: Anticipatory Destabilization
# ============================================================
def figure7_destabilization():
    """Variance vs carry depth across 15 counts."""
    csd = load_json("critical_slowing_down.json")

    counts_list = list(range(15))
    variances = [csd["per_count"][str(c)]["mean_variance"] for c in counts_list]

    def carry_depth(c):
        if c >= 14: return -1
        xor = c ^ (c + 1)
        d = 0
        while xor > 0:
            d += 1
            xor >>= 1
        return d - 1

    depths = [carry_depth(c) for c in counts_list]
    depth_colors = {-1: '#9E9E9E', 0: '#2196F3', 1: '#FF9800', 2: '#4CAF50', 3: '#F44336'}

    fig, ax = plt.subplots(figsize=(10, 5))

    for c in counts_list:
        color = depth_colors[depths[c]]
        marker = 's' if c == 14 else 'o'
        size = 120 if c == 14 else 80
        ax.scatter(c, variances[c], c=color, s=size, marker=marker,
                  edgecolors='black', linewidth=0.5, zorder=5)
        ax.text(c, variances[c] + 0.003, str(c), ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Count state')
    ax.set_ylabel('Idle-period variance')
    ax.set_title('Anticipatory Destabilization: Variance Scales with Carry Depth',
                 fontweight='bold')

    # Legend
    for d, label in [(-1, 'Terminal (14)'), (0, 'Depth 0'), (1, 'Depth 1'),
                     (2, 'Depth 2'), (3, 'Depth 3')]:
        marker = 's' if d == -1 else 'o'
        ax.scatter([], [], c=depth_colors[d], s=80, marker=marker,
                  label=label, edgecolors='black', linewidth=0.5)
    ax.legend(title='Upcoming cascade depth', loc='upper left')

    # Annotate correlation
    ax.text(0.97, 0.97, 'Spearman r = 0.923\np < 0.0001',
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_destabilization.png")
    plt.close()
    print("  Saved fig7_destabilization.png")


# ============================================================
# FIGURE 8: Spectral Radius Profile
# ============================================================
def figure8_spectral():
    """Spectral radius at each count state."""
    csd = load_json("critical_slowing_down.json")

    counts_list = list(range(15))
    spectral = [csd["eigenvalue_summary"][str(c)]["spectral_radius"] for c in counts_list]

    def carry_depth(c):
        if c >= 14: return -1
        xor = c ^ (c + 1)
        d = 0
        while xor > 0:
            d += 1
            xor >>= 1
        return d - 1

    depths = [carry_depth(c) for c in counts_list]
    depth_colors = {-1: '#9E9E9E', 0: '#2196F3', 1: '#FF9800', 2: '#4CAF50', 3: '#F44336'}

    fig, ax = plt.subplots(figsize=(10, 5))

    # Stability boundary
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Stability boundary (rho = 1.0)')
    ax.fill_between([-0.5, 14.5], 0.9, 1.0, alpha=0.1, color='green')
    ax.fill_between([-0.5, 14.5], 1.0, 1.35, alpha=0.1, color='red')

    for c in counts_list:
        color = depth_colors[depths[c]]
        marker = 's' if c == 14 else 'o'
        size = 120 if c == 14 else 80
        ax.scatter(c, spectral[c], c=color, s=size, marker=marker,
                  edgecolors='black', linewidth=0.5, zorder=5)
        ax.text(c, spectral[c] + 0.015, str(c), ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Count state')
    ax.set_ylabel('Spectral radius (rho)')
    ax.set_title('The Controlled Explosion: Mildly Expansive Everywhere Except Terminal State',
                 fontweight='bold')
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0.95, 1.35)

    # Annotations
    ax.annotate('Sole attractor\n(rho = 0.998)', xy=(14, spectral[14]),
                xytext=(11.5, 1.02), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(0.5, 1.32, 'Mildly expansive', fontsize=11, color='#D32F2F', alpha=0.7)
    ax.text(0.5, 0.96, 'Contractive (stable)', fontsize=11, color='#2E7D32', alpha=0.7)

    # Legend for depth colors
    for d, label in [(-1, 'Terminal'), (0, 'Depth 0'), (1, 'Depth 1'),
                     (2, 'Depth 2'), (3, 'Depth 3')]:
        marker = 's' if d == -1 else 'o'
        ax.scatter([], [], c=depth_colors[d], s=80, marker=marker,
                  label=label, edgecolors='black', linewidth=0.5)
    ax.legend(title='Upcoming cascade depth', loc='upper right', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_spectral_radius.png")
    plt.close()
    print("  Saved fig8_spectral_radius.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Generating publication figures...")
    print()

    print("[Priority 1] Figure 2: Imagination vs Posterior...")
    figure2_imagination()

    print("[Priority 2] Figure 4: ESN Control...")
    figure4_esn()

    print("[Priority 3] Figure 1: Bit Heatmap...")
    figure1_heatmap()

    print("[Poster] Figure 3: Observation Cliff...")
    figure3_cliff()

    print("[Poster] Figure 5: Scorecard...")
    figure5_scorecard()

    print("[Poster] Figure 6: PCA Trajectory...")
    figure6_pca()

    print("[Poster] Figure 7: Destabilization...")
    figure7_destabilization()

    print("[Poster] Figure 8: Spectral Radius...")
    figure8_spectral()

    print()
    print(f"All figures saved to {FIG_DIR}/")
