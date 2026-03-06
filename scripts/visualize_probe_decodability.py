#!/usr/bin/env python3
"""
Visualize the probe decodability difference between baseline and randproj models.

Generates three publication-ready figures:
  1. Probe-direction histograms (KDE) — shows peak separation vs overlap
  2. Probe vs noise 2D scatter — shows factorized vs entangled representations
  3. Probe trace time-series — shows clean steps vs wobbly predictions

Usage:
    python visualize_probe_decodability.py [--output-dir bridge/artifacts/figures]

Requires: clean_baseline_s0.npz and clean_randproj_s0.npz in battery dirs.
"""

import os, sys, argparse, pathlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
BATTERY_DIR = pathlib.Path("/workspace/bridge/artifacts/battery")
BASELINE_NPZ = BATTERY_DIR / "clean" / "clean_baseline_s0" / "clean_baseline_s0.npz"
RANDPROJ_NPZ = BATTERY_DIR / "clean" / "clean_randproj_s0" / "clean_randproj_s0.npz"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,  # editable text in PDFs
    'ps.fonttype': 42,
})

VIRIDIS = plt.cm.viridis


def load_model_data(npz_path):
    """Load h_t and counts, fit probe, return everything needed."""
    data = np.load(npz_path)
    h_t = data['h_t']        # (N, 512)
    counts = data['counts']   # (N,)
    ep_ids = data.get('episode_ids', None)
    timesteps = data.get('timesteps', None)
    if ep_ids is not None:
        ep_ids = np.array(ep_ids)
    if timesteps is not None:
        timesteps = np.array(timesteps)

    # Fit probe
    probe = Ridge(alpha=1.0)
    probe.fit(h_t, counts)
    w = probe.coef_
    b = probe.intercept_
    w_norm = w / (np.linalg.norm(w) + 1e-12)

    # Projections onto probe direction
    proj = h_t @ w_norm  # scalar per sample

    # Raw probe predictions (continuous)
    raw_pred = h_t @ w + b

    # Compute noise direction: PC1 of pooled within-count residuals
    residuals = []
    for c in np.unique(counts):
        mask = counts == c
        h_c = h_t[mask]
        mean_c = h_c.mean(axis=0)
        residuals.append(h_c - mean_c)
    residuals = np.vstack(residuals)
    pca_resid = PCA(n_components=1)
    pca_resid.fit(residuals)
    noise_dir = pca_resid.components_[0]
    noise_dir = noise_dir / (np.linalg.norm(noise_dir) + 1e-12)

    # Project onto noise direction
    noise_proj = h_t @ noise_dir

    # Probe SNR
    unique_counts = np.unique(counts)
    per_count_means, per_count_vars = [], []
    for c in unique_counts:
        mask = counts == c
        p_c = proj[mask]
        if len(p_c) > 1:
            per_count_means.append(np.mean(p_c))
            per_count_vars.append(np.var(p_c))
    snr = np.var(per_count_means) / np.mean(per_count_vars)

    # Exact accuracy
    pred_round = np.clip(np.round(raw_pred), 0, 25).astype(int)
    exact_acc = np.mean(pred_round == counts)

    return {
        'h_t': h_t,
        'counts': counts,
        'ep_ids': ep_ids,
        'timesteps': timesteps,
        'proj': proj,
        'noise_proj': noise_proj,
        'raw_pred': raw_pred,
        'snr': snr,
        'exact_acc': exact_acc,
        'probe_w': w,
        'probe_b': b,
        'w_norm': w_norm,
        'noise_dir': noise_dir,
    }


def fig1_probe_histograms(baseline, randproj, output_dir):
    """Side-by-side KDE histograms of probe-direction projections per count."""
    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    max_count = 25
    # Subsample counts for cleaner viz: show every count but only a selection prominently
    show_counts = list(range(0, 26))

    for ax, data, title_suffix in [
        (ax_b, baseline, f"Baseline (SNR={baseline['snr']:.0f})"),
        (ax_r, randproj, f"Random Projection (SNR={randproj['snr']:.0f})")
    ]:
        for c in show_counts:
            mask = data['counts'] == c
            if mask.sum() < 5:
                continue
            vals = data['proj'][mask]
            color = VIRIDIS(c / max_count)

            # KDE via histogram with density
            ax.hist(vals, bins=60, density=True, alpha=0.35, color=color,
                    edgecolor='none', linewidth=0)

            # Add thin KDE line for clarity
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(vals, bw_method=0.15)
                x_range = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=1.0, alpha=0.8)
            except Exception:
                pass

        ax.set_xlabel('Projection onto probe direction', fontsize=11)
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_b.set_ylabel('Density', fontsize=11)

    # Shared x limits
    all_proj = np.concatenate([baseline['proj'], randproj['proj']])
    margin = (all_proj.max() - all_proj.min()) * 0.05
    for ax in [ax_b, ax_r]:
        ax.set_xlim(all_proj.min() - margin, all_proj.max() + margin)

    fig.suptitle('Linear Probe Decodability: Where Counts Overlap Along the Probe Direction',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Colorbar — placed after tight_layout, with enough pad to sit outside plots
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(0, 25))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_b, ax_r], shrink=0.7, pad=0.05, aspect=30)
    cbar.set_label('Count', fontsize=10)

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'probe_histograms.{ext}',
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [1/3] Probe histograms saved")


def fig2_probe_vs_noise(baseline, randproj, output_dir):
    """Side-by-side scatter: probe direction (x) vs noise direction (y)."""
    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(12, 5.5))

    max_count = 25

    # Subsample for performance (>50K points each is too dense)
    rng = np.random.RandomState(42)
    max_points = 15000

    for ax, data, title_suffix in [
        (ax_b, baseline, f"Baseline (SNR={baseline['snr']:.0f})"),
        (ax_r, randproj, f"Random Projection (SNR={randproj['snr']:.0f})")
    ]:
        n = len(data['counts'])
        if n > max_points:
            idx = rng.choice(n, max_points, replace=False)
        else:
            idx = np.arange(n)

        sc = ax.scatter(
            data['proj'][idx], data['noise_proj'][idx],
            c=data['counts'][idx], cmap='viridis',
            vmin=0, vmax=max_count,
            s=1.5, alpha=0.4, edgecolors='none', rasterized=True
        )

        ax.set_xlabel('Projection onto probe direction (count signal)', fontsize=10)
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_b.set_ylabel('Projection onto noise direction\n(max within-count variance)', fontsize=10)

    # Shared axis limits
    all_x = np.concatenate([baseline['proj'], randproj['proj']])
    all_y = np.concatenate([baseline['noise_proj'], randproj['noise_proj']])
    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05
    for ax in [ax_b, ax_r]:
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    fig.suptitle('Count Signal vs Noise: Factorized Columns (right) vs Entangled Clouds (left)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Colorbar — placed after tight_layout, with enough pad to sit outside plots
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(0, 25))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_b, ax_r], shrink=0.7, pad=0.05, aspect=30)
    cbar.set_label('Count', fontsize=10)

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'probe_vs_noise.{ext}',
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [2/3] Probe vs noise scatter saved")


def fig3_probe_trace(baseline, randproj, output_dir):
    """Time-series: ground truth staircase + raw probe output for one episode."""
    fig, (ax_b, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, data, title_suffix in [
        (ax_b, baseline, f"Baseline (SNR={baseline['snr']:.0f})"),
        (ax_r, randproj, f"Random Projection (SNR={randproj['snr']:.0f})")
    ]:
        ep_ids = data['ep_ids']
        if ep_ids is None:
            ax.text(0.5, 0.5, 'No episode data available', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        # Pick an episode that goes through a full counting cycle (0 → 25)
        unique_eps = np.unique(ep_ids)
        best_ep = None
        best_range = 0
        for ep in unique_eps:
            mask = ep_ids == ep
            c = data['counts'][mask]
            count_range = c.max() - c.min()
            if count_range > best_range:
                best_range = count_range
                best_ep = ep

        if best_ep is None:
            continue

        mask = ep_ids == best_ep
        steps = np.arange(mask.sum())
        gt = data['counts'][mask]
        raw = data['raw_pred'][mask]

        # Ground truth staircase
        ax.step(steps, gt, where='post', color='#888888', linewidth=2.0,
                alpha=0.6, label='Ground truth', zorder=1)

        # Raw probe output — color segments by accuracy
        pred_round = np.clip(np.round(raw), 0, 25).astype(int)
        exact = pred_round == gt

        # Draw line colored by error magnitude
        for i in range(len(steps) - 1):
            err = abs(raw[i] - gt[i])
            if err < 0.5:
                color = '#22c55e'  # green — exact
            elif err < 1.5:
                color = '#eab308'  # yellow — ±1
            else:
                color = '#ef4444'  # red — worse
            ax.plot([steps[i], steps[i+1]], [raw[i], raw[i+1]],
                    color=color, linewidth=1.2, alpha=0.9, zorder=2)

        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title_suffix, fontsize=11, fontweight='bold')
        ax.set_ylim(-1, 27)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Accuracy annotation
        exact_pct = np.mean(exact) * 100
        ax.text(0.98, 0.95, f'{exact_pct:.0f}% exact',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax_r.set_xlabel('Timestep', fontsize=10)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#888888', linewidth=2, alpha=0.6, label='Ground truth'),
        Line2D([0], [0], color='#22c55e', linewidth=1.5, label='Exact'),
        Line2D([0], [0], color='#eab308', linewidth=1.5, label='Within ±1'),
        Line2D([0], [0], color='#ef4444', linewidth=1.5, label='Error > ±1'),
    ]
    ax_b.legend(handles=legend_elements, loc='upper left', fontsize=8,
                framealpha=0.8, edgecolor='gray')

    fig.suptitle('Real-Time Probe Predictions: Clean Steps vs Wobbly Tracking',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'probe_trace.{ext}',
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [3/3] Probe trace saved")


def main():
    parser = argparse.ArgumentParser(description="Probe decodability visualizations")
    parser.add_argument("--output-dir", type=str,
                        default="/workspace/bridge/artifacts/figures")
    parser.add_argument("--baseline", type=str, default=str(BASELINE_NPZ))
    parser.add_argument("--randproj", type=str, default=str(RANDPROJ_NPZ))
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading baseline data...")
    baseline = load_model_data(args.baseline)
    print(f"  {len(baseline['counts'])} samples, SNR={baseline['snr']:.0f}, "
          f"exact={baseline['exact_acc']:.1%}")

    print("Loading randproj data...")
    randproj = load_model_data(args.randproj)
    print(f"  {len(randproj['counts'])} samples, SNR={randproj['snr']:.0f}, "
          f"exact={randproj['exact_acc']:.1%}")

    print(f"\nGenerating figures in {output_dir}...")
    fig1_probe_histograms(baseline, randproj, output_dir)
    fig2_probe_vs_noise(baseline, randproj, output_dir)
    fig3_probe_trace(baseline, randproj, output_dir)

    print(f"\nAll figures saved to {output_dir}")
    print(f"  probe_histograms.png/.pdf")
    print(f"  probe_vs_noise.png/.pdf")
    print(f"  probe_trace.png/.pdf")


if __name__ == "__main__":
    main()
