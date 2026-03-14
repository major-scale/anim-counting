#!/usr/bin/env python3
"""Generate figures for UNIFIER.md write-up."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
ARTIFACTS_DIR = '/workspace/projects/jamstack-v1/bridge/artifacts'
SCRIPTS_DIR = '/workspace/projects/jamstack-v1/bridge/scripts'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Figure 1: GHE Comparison Bar Chart ────────────────────────────────────────

def fig_ghe_comparison():
    labels = ['Grid-only\n(Type A)', 'Binary-only\n(Type B)', 'Combined\n(Grid + Binary)']
    values = [0.33, 4.91, 12.087]
    colors = ['#4C72B0', '#DD8452', '#C44E52']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add threshold line
    ax.axhline(y=0.5, color='#55A868', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2.35, 0.7, 'GHE < 0.5 = smooth\ndecimal manifold',
            fontsize=9, color='#55A868', ha='right', style='italic')

    ax.set_ylabel('Geodesic Homogeneity Error (decimal)', fontsize=12)
    ax.set_title('Same Concept, Different Geometry', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 14.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'ghe_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Figure 2: Lambda Phase Transition ─────────────────────────────────────────

def fig_lambda_phase_transition():
    sweep_path = os.path.join(ARTIFACTS_DIR, 'sweep_results', 'results.json')
    with open(sweep_path) as f:
        sweep = json.load(f)

    lambdas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    pr2_ham = [sweep['results'][str(l)]['pr2_hamming'] for l in lambdas]
    # Per-bit: average of 4 bits
    perbit = []
    for l in lambdas:
        r = sweep['results'][str(l)]
        pb = np.mean([r['bit0_acc'], r['bit1_acc'], r['bit2_acc'], r['bit3_acc']])
        perbit.append(pb)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_blue = '#4C72B0'
    color_orange = '#DD8452'

    # pR² Hamming (left axis)
    ax1.plot(range(len(lambdas)), pr2_ham, 'o-', color=color_blue, linewidth=2.5,
             markersize=8, label='pR² Hamming (geometric structure)', zorder=3)
    ax1.set_ylabel('pR² Hamming', fontsize=12, color=color_blue)
    ax1.tick_params(axis='y', labelcolor=color_blue)
    ax1.set_ylim(-0.05, 0.65)

    # Shade the phase transition region
    ax1.axvspan(0.5, 1.5, alpha=0.1, color='red')
    ax1.annotate('Phase\ntransition', xy=(1, 0.008), xytext=(1.8, 0.35),
                 fontsize=10, ha='center', color='#C44E52',
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    # Per-bit accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.plot(range(len(lambdas)), [p * 100 for p in perbit], 's--', color=color_orange,
             linewidth=2.5, markersize=8, label='Per-bit accuracy (%)', zorder=2)
    ax2.set_ylabel('Per-bit accuracy (%)', fontsize=12, color=color_orange)
    ax2.tick_params(axis='y', labelcolor=color_orange)
    ax2.set_ylim(98.5, 100.5)

    ax1.set_xticks(range(len(lambdas)))
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.set_xlabel('Contrastive alignment weight (λ)', fontsize=12)
    ax1.set_title('Information ≠ Geometry', fontsize=14, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'lambda_phase_transition.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Figure 3: eRank Trajectory ────────────────────────────────────────────────

def fig_erank_trajectory():
    traj_path = os.path.join(SCRIPTS_DIR, 'subspace_results', 'erank_trajectory.json')
    with open(traj_path) as f:
        traj = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = [
        ('onset_5000', 'Late VICReg (onset 5000)', '#4C72B0', '-', 'o'),
        ('gru_frozen_5000', 'GRU Frozen (onset 5000)', '#C44E52', '--', 's'),
        ('N400_onset0', 'Early VICReg (onset 0)', '#8C8C8C', ':', 'D'),
    ]

    for key, label, color, ls, marker in conditions:
        steps = [d['step'] for d in traj[key]]
        eranks = [d['erank'] for d in traj[key]]
        ax.plot(steps, eranks, ls, color=color, linewidth=2.5,
                marker=marker, markersize=6, label=label, zorder=3)

    # Shade VICReg windows
    ax.axvspan(0, 400, alpha=0.06, color='#8C8C8C')
    ax.text(200, 25.5, 'VICReg\n(onset-0)', fontsize=8, ha='center',
            color='#8C8C8C', style='italic')
    ax.axvspan(5000, 5400, alpha=0.12, color='#4C72B0')
    ax.text(5200, 25.5, 'VICReg\n(onset-5000)', fontsize=8, ha='center',
            color='#4C72B0', style='italic')

    # Annotate key divergence
    ax.annotate('GRU expands →',
                xy=(10000, 23.9), xytext=(14000, 25),
                fontsize=9, color='#4C72B0',
                arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=1.2))
    ax.annotate('← GRU collapses',
                xy=(10000, 16.4), xytext=(14000, 14.5),
                fontsize=9, color='#C44E52',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.2))

    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Effective rank (eRank)', fontsize=12)
    ax.set_title('Cooperative Plasticity: GRU Must Respond', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(-500, 26000)
    ax.set_ylim(0, 27)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'erank_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating UNIFIER.md figures...')
    fig_ghe_comparison()
    fig_lambda_phase_transition()
    fig_erank_trajectory()
    print('Done.')
