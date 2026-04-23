"""Interactive visualization of binary world representation geometry.

Core simulator + matplotlib plots for Option A (carry cascade + intervention)
and Option B (real-time decomposition onto bit-flip directions).

Launch (batch screenshots):
    python3 scripts/viz_binary_world.py --mode batch

Launch (Streamlit interactive):
    streamlit run scripts/viz_binary_world.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge

BRIDGE_SCRIPTS = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/scripts")
CHECKPOINT_DIR = Path("/Users/petermurphy/cc-sandbox/projects/jamstack-v1/bridge/artifacts/checkpoints/binary_baseline_s0/exported")
BATTERY_PATH = Path("/Users/petermurphy/anima-bridge/artifacts/battery/binary_baseline_s0/battery.npz")
SCREENSHOTS_DIR = Path("/Users/petermurphy/anima-bridge/artifacts/screenshots/viz_binary_20260423")

sys.path.insert(0, str(BRIDGE_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_ghe_binary import load_exported_weights, STOCH_FLAT, STOCH_DIM, STOCH_CLASSES, DETER_DIM

LN_EPS = 1e-3
BIT_OBS_INDICES = [49, 53, 57, 61]
BIT_COLORS = ["#E63946", "#F4A261", "#2A9D8F", "#264653"]  # warm→cool for bits 0→3


# ─── RSSM math (copied/merged from quick_ghe_binary + imagination_rollout_binary) ──

def _ln(x, w, b, eps=LN_EPS):
    mu, var = x.mean(), x.var()
    return w * (x - mu) / np.sqrt(var + eps) + b

def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _argmax_one_hot(logits):
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    idx = logits.argmax(axis=-1)
    for i, j in enumerate(idx):
        out[i * STOCH_CLASSES + j] = 1.0
    return out


class Simulator:
    """RSSM forward dynamics: imagine_step() + decode() + intervention patching."""
    GRU_UPDATE_BIAS = -1.0

    def __init__(self, weights):
        self.w = weights

    def prior_stoch_from_deter(self, deter):
        """Compute prior stoch deterministically from h_t alone (no action, no obs)."""
        h = _silu(_ln(self.w["img_out_w"] @ deter, self.w["img_out_norm_w"], self.w["img_out_norm_b"]))
        logits = (self.w["imgs_stat_w"] @ h + self.w["imgs_stat_b"]).reshape(STOCH_DIM, STOCH_CLASSES)
        return _argmax_one_hot(logits)

    def _transition(self, deter, stoch, action=0):
        """One GRU step: (deter, stoch, action) -> new deter."""
        w = self.w
        num_actions = w["img_in_w"].shape[1] - STOCH_FLAT
        act = np.zeros(num_actions, dtype=np.float32)
        if num_actions > 0:
            act[0] = 1.0 if num_actions > 1 else float(action)
        cat_in = np.concatenate([stoch, act])
        h_in = _silu(_ln(w["img_in_w"] @ cat_in, w["img_in_norm_w"], w["img_in_norm_b"]))
        combined = np.concatenate([h_in, deter])
        out = w["gru_w"] @ combined
        out = _ln(out, w["gru_norm_w"], w["gru_norm_b"])
        N = DETER_DIM
        reset_gate = _sigmoid(out[:N])
        cand = np.tanh(reset_gate * out[N:2*N])
        update = _sigmoid(out[2*N:] + self.GRU_UPDATE_BIAS)
        new_deter = (update * cand + (1 - update) * deter).astype(np.float32)
        return new_deter

    def imagine_step(self, deter, stoch, action=0):
        new_deter = self._transition(deter, stoch, action)
        new_stoch = self.prior_stoch_from_deter(new_deter)
        return new_deter, new_stoch

    def imagine_forward(self, deter0, stoch0, horizon, intervention=None):
        """Run imagination forward `horizon` steps.

        intervention: dict with keys {step, direction, alpha} or None.
            At imagination step `step`, add alpha * direction to deter before continuing.
        """
        deter_traj = [deter0.copy()]
        stoch_traj = [stoch0.copy()]
        d, s = deter0.copy(), stoch0.copy()
        for t in range(horizon):
            if intervention is not None and t == intervention["step"]:
                d = d + intervention["alpha"] * intervention["direction"]
                s = self.prior_stoch_from_deter(d)  # refresh stoch after patch
            d, s = self.imagine_step(d, s)
            deter_traj.append(d)
            stoch_traj.append(s)
        return np.array(deter_traj), np.array(stoch_traj)

    def decode(self, stoch, deter):
        """Forward through decoder. Returns symlog'd obs prediction."""
        w = self.w
        x = np.concatenate([stoch, deter])
        h = _silu(_ln(w["dec_linear0_w"] @ x, w["dec_norm0_w"], w["dec_norm0_b"]))
        h = _silu(_ln(w["dec_linear1_w"] @ h, w["dec_norm1_w"], w["dec_norm1_b"]))
        h = _silu(_ln(w["dec_linear2_w"] @ h, w["dec_norm2_w"], w["dec_norm2_b"]))
        return w["dec_out_w"] @ h + w["dec_out_b"]

    def decode_bits(self, stoch, deter):
        obs_symlog = self.decode(stoch, deter)
        obs = np.sign(obs_symlog) * (np.exp(np.abs(obs_symlog)) - 1.0)
        return np.array([1 if obs[i] > 0.5 else 0 for i in BIT_OBS_INDICES], dtype=np.int32)

    def decode_bits_soft(self, stoch, deter):
        """Return soft (continuous) bit predictions for heatmap display."""
        obs_symlog = self.decode(stoch, deter)
        obs = np.sign(obs_symlog) * (np.exp(np.abs(obs_symlog)) - 1.0)
        return np.array([obs[i] for i in BIT_OBS_INDICES])


# ─── Data helpers ─────────────────────────────────────────────────────────────

def fit_probes(h_stable, bits_stable):
    """Returns unit-normalized probe directions (4, 512) and raw probes (4, 512)."""
    raw, unit = [], []
    for b in range(4):
        p = Ridge(alpha=1.0).fit(h_stable, bits_stable[:, b])
        w_raw = p.coef_.astype(np.float64)
        raw.append(w_raw)
        unit.append(w_raw / np.linalg.norm(w_raw))
    return np.stack(unit), np.stack(raw)


def sample_initial_state(battery_h, battery_counts, battery_carry, source_count, idx=0, mode="mid"):
    """Get a stable h_t sample at source_count.

    mode="mid": generic stable h_t (middle of count=c period)
    mode="pre-transition": last stable h_t before the natural transition to c+1
        (this is the fork point that contains anticipation signal; matches the
        §6.4 imagination-rollout protocol in the paper.)
    """
    stable_mask = (battery_counts == source_count) & (battery_carry == 0)
    stable_indices = np.where(stable_mask)[0]
    if len(stable_indices) == 0:
        raise ValueError(f"No stable samples for count={source_count}")

    if mode == "mid":
        return battery_h[stable_indices[idx]].astype(np.float64)

    # Pre-transition: for each contiguous run of count=c, take the LAST index.
    # These are the states at/just before the natural transition point.
    diffs = np.diff(stable_indices)
    run_ends = np.where(diffs > 1)[0]  # ends of contiguous runs
    last_in_runs = np.concatenate([stable_indices[run_ends], [stable_indices[-1]]])
    if len(last_in_runs) == 0:
        return battery_h[stable_indices[-1]].astype(np.float64)
    return battery_h[last_in_runs[idx % len(last_in_runs)]].astype(np.float64)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_cascade(deter_traj, stoch_traj, simulator, probes_unit, title, ax_bits=None, ax_probes=None,
                 intervention_step=None):
    """Two-panel cascade plot. Top: decoded bit states heatmap. Bottom: probe signals over time.

    If intervention_step is provided, mark it with a vertical line.
    """
    T = len(deter_traj)
    # Decoded bits (continuous + thresholded)
    soft_bits = np.zeros((T, 4))
    hard_bits = np.zeros((T, 4), dtype=int)
    probe_sigs = np.zeros((T, 4))
    for t in range(T):
        soft_bits[t] = simulator.decode_bits_soft(stoch_traj[t], deter_traj[t])
        hard_bits[t] = simulator.decode_bits(stoch_traj[t], deter_traj[t])
        probe_sigs[t] = [deter_traj[t] @ probes_unit[b] for b in range(4)]

    time_axis = np.arange(T)

    if ax_bits is None:
        fig, (ax_bits, ax_probes) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                                  gridspec_kw={"height_ratios": [1, 1.2]})
    else:
        fig = ax_bits.figure

    # Top: bit-state heatmap (continuous, to show the smooth cascade)
    im = ax_bits.imshow(soft_bits.T, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=1.2,
                        extent=[0, T-1, 3.5, -0.5], interpolation="nearest")
    for b in range(4):
        for t in range(T):
            ax_bits.text(t, b, str(hard_bits[t, b]), ha="center", va="center",
                         color="white" if 0.2 < soft_bits[t, b] < 0.8 else "black",
                         fontsize=7, fontweight="bold")
    ax_bits.set_yticks(range(4))
    ax_bits.set_yticklabels([f"bit {b} ({2**b})" for b in range(4)])
    ax_bits.set_ylabel("Decoded bit (hard 0/1, color = soft prob)")
    ax_bits.set_title(title, fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_bits, fraction=0.025, pad=0.01)
    cbar.set_label("soft bit value", fontsize=9)

    # Bottom: probe signals (h @ ŵ_i for each bit)
    for b in range(4):
        ax_probes.plot(time_axis, probe_sigs[:, b], color=BIT_COLORS[b],
                       label=f"bit {b}", linewidth=2)
    ax_probes.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_probes.set_xlabel("imagination step (t=0 is fork point)")
    ax_probes.set_ylabel(r"probe signal  $h_t \cdot \hat{w}_b$")
    ax_probes.legend(loc="upper right", fontsize=9)
    ax_probes.grid(True, alpha=0.3)

    if intervention_step is not None:
        ax_bits.axvline(x=intervention_step, color="magenta", linewidth=1.5, linestyle=":", alpha=0.9)
        ax_probes.axvline(x=intervention_step, color="magenta", linewidth=1.5, linestyle=":", alpha=0.9,
                          label=f"intervention at t={intervention_step}")
        ax_probes.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig, {"soft_bits": soft_bits, "hard_bits": hard_bits, "probe_sigs": probe_sigs}


def plot_intervention_compare(deter_base, stoch_base, deter_int, stoch_int,
                               simulator, probes_unit, intervention_info, title):
    """Compare natural cascade vs intervened cascade side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=False,
                              gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.25, "wspace": 0.12})
    plot_cascade(deter_base, stoch_base, simulator, probes_unit,
                 f"NATURAL imagination (no intervention)",
                 ax_bits=axes[0, 0], ax_probes=axes[1, 0])
    title_int = (f"INTERVENED: patch bit-{intervention_info['bit']} direction, "
                 f"α={intervention_info['alpha']:+.2f}, at t={intervention_info['step']}")
    plot_cascade(deter_int, stoch_int, simulator, probes_unit, title_int,
                 ax_bits=axes[0, 1], ax_probes=axes[1, 1],
                 intervention_step=intervention_info["step"])
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    return fig


def plot_alpha_sweep(deter0, stoch0, simulator, probes_unit, bit_idx, intervention_step,
                     alphas, horizon, title):
    """Show how outcome changes as α varies for a given bit's intervention.

    Produces a heatmap: rows = α values, cols = imagination time, color = decoded bit `bit_idx`.
    Plus a column indicating whether the cascade reached its final state.
    """
    outcomes = np.zeros((len(alphas), horizon + 1, 4))  # α × t × bit
    for i, alpha in enumerate(alphas):
        intervention = {"step": intervention_step, "direction": probes_unit[bit_idx].astype(np.float32),
                        "alpha": float(alpha), "bit": bit_idx}
        dtraj, straj = simulator.imagine_forward(deter0.astype(np.float32), stoch0, horizon=horizon,
                                                  intervention=intervention)
        for t in range(horizon + 1):
            outcomes[i, t] = simulator.decode_bits_soft(straj[t], dtraj[t])

    fig, axes = plt.subplots(1, 4, figsize=(18, max(4, len(alphas) * 0.35)), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for b in range(4):
        im = axes[b].imshow(outcomes[:, :, b], aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=1.2,
                            extent=[0, horizon, len(alphas) - 0.5, -0.5], interpolation="nearest")
        axes[b].set_title(f"bit {b} ({2**b})", fontsize=11, fontweight="bold")
        axes[b].axvline(x=intervention_step, color="magenta", linestyle=":", linewidth=1, alpha=0.8)
        axes[b].set_xlabel("imagination step")
        if b == 0:
            axes[b].set_ylabel(f"α (intervention magnitude on bit {bit_idx})")
            axes[b].set_yticks(range(len(alphas)))
            axes[b].set_yticklabels([f"{a:+.1f}" for a in alphas], fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label("soft bit value", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    return fig


def plot_step_decomposition(deter_traj, probes_unit, title):
    """Option B: step vector decomposition onto the four bit-flip directions.

    For each t, compute step = deter[t+1] - deter[t], project onto each ŵ_b.
    Shows the four colored components summing to the total step magnitude.
    """
    T = len(deter_traj) - 1
    step_vecs = np.diff(deter_traj, axis=0)  # (T, 512)
    projections = np.zeros((T, 4))
    for t in range(T):
        for b in range(4):
            projections[t, b] = step_vecs[t] @ probes_unit[b]
    step_mag = np.linalg.norm(step_vecs, axis=1)
    bitflip_mag = np.linalg.norm(projections, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1]})
    t_axis = np.arange(T)
    # Top: individual projections onto each bit direction
    for b in range(4):
        ax1.plot(t_axis, projections[:, b], color=BIT_COLORS[b], linewidth=2, label=f"bit {b}")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel(r"$(h_{t+1}-h_t)\cdot \hat{w}_b$")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: total step magnitude vs bit-flip-subspace magnitude
    ax2.plot(t_axis, step_mag, color="black", linewidth=2, label=r"$\|h_{t+1}-h_t\|$ (total)")
    ax2.plot(t_axis, bitflip_mag, color="#888", linewidth=2, linestyle="--",
             label=r"$\|$projection onto bit-flip 4-subspace$\|$")
    ax2.fill_between(t_axis, 0, bitflip_mag, color="gray", alpha=0.2)
    ax2.set_xlabel("imagination step")
    ax2.set_ylabel("step magnitude")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─── Batch mode: generate the key screenshots ─────────────────────────────────

def batch_generate():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading weights + battery...")
    w = load_exported_weights(CHECKPOINT_DIR)
    sim = Simulator(w)
    d = np.load(BATTERY_PATH)
    h_t, counts, bits_, carry_ = d["h_t"], d["counts"], d["bits"], d["carry_active"]
    stable = carry_ == 0
    probes_unit, probes_raw = fit_probes(h_t[stable], bits_[stable])

    # ── 1. Natural cascade for 7→8 full cascade ───────────────────────────────
    print("Rendering cascade: 7→8 full cascade, natural imagination")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=7, idx=3, mode="pre-transition")
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    dtraj, straj = sim.imagine_forward(h0.astype(np.float32), s0, horizon=40)
    fig, _ = plot_cascade(dtraj, straj, sim, probes_unit,
                          "Carry cascade in imagination — source count=7, natural dynamics")
    fig.savefig(SCREENSHOTS_DIR / "01_cascade_7to8_natural.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Natural cascade for various sources (depths 0, 1, 2, 3) ────────────
    print("Rendering cascades: all 4 depths")
    fig, axes = plt.subplots(4, 2, figsize=(18, 16),
                              gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.55, "wspace": 0.12})
    for row, c in enumerate([0, 1, 3, 7]):  # depth 0, 1, 2, 3
        depth = bin(c ^ (c+1)).count('1') - 1
        h0 = sample_initial_state(h_t, counts, carry_, source_count=c, idx=2, mode="pre-transition")
        s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
        dtraj, straj = sim.imagine_forward(h0.astype(np.float32), s0, horizon=35)
        plot_cascade(dtraj, straj, sim, probes_unit,
                     f"Source count = {c} → {c+1}  (carry depth {depth})",
                     ax_bits=axes[row, 0], ax_probes=axes[row, 1])
    fig.suptitle("Cascade by carry depth — natural imagination dynamics", fontsize=15, fontweight="bold", y=0.995)
    fig.savefig(SCREENSHOTS_DIR / "02_cascade_depth_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Intervention: push bit 2 during a non-cascade moment for source=0 ──
    print("Rendering intervention: push bit 2 at t=10, source count=0")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=0, idx=3)
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    # Baseline
    dtraj_base, straj_base = sim.imagine_forward(h0.astype(np.float32), s0, horizon=35)
    # Intervention: strong push along bit-2 direction at t=10
    alpha_push = 5.0
    intervention = {"step": 10, "direction": probes_unit[2].astype(np.float32), "alpha": alpha_push, "bit": 2}
    dtraj_int, straj_int = sim.imagine_forward(h0.astype(np.float32), s0, horizon=35, intervention=intervention)
    fig = plot_intervention_compare(dtraj_base, straj_base, dtraj_int, straj_int,
                                      sim, probes_unit, intervention,
                                      "Intervention probe: can we force bit 2 to flip from a low-count state?")
    fig.savefig(SCREENSHOTS_DIR / "03_intervention_bit2_push.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Intervention mid-cascade: disrupt the 7→8 full cascade ─────────────
    print("Rendering intervention: disrupt 7→8 cascade mid-flight")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=7, idx=3, mode="pre-transition")
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    dtraj_base, straj_base = sim.imagine_forward(h0.astype(np.float32), s0, horizon=35)
    # Intervention: cancel bit 0's flip mid-cascade (anti-push)
    intervention = {"step": 8, "direction": probes_unit[0].astype(np.float32), "alpha": -4.0, "bit": 0}
    dtraj_int, straj_int = sim.imagine_forward(h0.astype(np.float32), s0, horizon=35, intervention=intervention)
    fig = plot_intervention_compare(dtraj_base, straj_base, dtraj_int, straj_int,
                                      sim, probes_unit, intervention,
                                      "Disrupting a full cascade — push AGAINST bit 0 mid-flight")
    fig.savefig(SCREENSHOTS_DIR / "04_intervention_cascade_disrupt.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Option B: step-vector decomposition during natural cascade ─────────
    print("Rendering Option B: step-vector decomposition")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=7, idx=3, mode="pre-transition")
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    dtraj, straj = sim.imagine_forward(h0.astype(np.float32), s0, horizon=40)
    fig = plot_step_decomposition(dtraj, probes_unit,
                                   "Step-vector decomposition onto the four bit-flip directions (source=7, full cascade)")
    fig.savefig(SCREENSHOTS_DIR / "05_step_decomposition_cascade.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 6. Alpha sweep: how intervention strength shapes outcome ──────────────
    print("Rendering alpha sweep on bit 2 from count=0")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=0, idx=2, mode="pre-transition")
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    alphas = np.array([-6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10])
    fig = plot_alpha_sweep(h0, s0, sim, probes_unit, bit_idx=2, intervention_step=8,
                            alphas=alphas, horizon=30,
                            title="α sweep on bit-2 direction from source count=0 (α=0 row is natural)")
    fig.savefig(SCREENSHOTS_DIR / "06_alpha_sweep_bit2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── 7. Alpha sweep: ANTI-push during 7→8 cascade ──────────────────────────
    print("Rendering alpha sweep on bit 0 during 7→8 cascade (anti-push strength test)")
    h0 = sample_initial_state(h_t, counts, carry_, source_count=7, idx=3, mode="pre-transition")
    s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))
    alphas = np.array([-10, -8, -6, -4, -2, 0, 2, 4])
    fig = plot_alpha_sweep(h0, s0, sim, probes_unit, bit_idx=0, intervention_step=2,
                            alphas=alphas, horizon=30,
                            title="α sweep: how much anti-push on bit 0 does it take to break the 7→8 cascade?")
    fig.savefig(SCREENSHOTS_DIR / "07_alpha_sweep_disrupt.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote 7 screenshots to {SCREENSHOTS_DIR}")
    for f in sorted(SCREENSHOTS_DIR.glob("*.png")):
        print(f"  {f.name}")


# ─── Streamlit mode ───────────────────────────────────────────────────────────

def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Binary World Viz", layout="wide")
    st.title("Binary world — representation geometry viewer")

    @st.cache_resource
    def load_all():
        w = load_exported_weights(CHECKPOINT_DIR)
        sim = Simulator(w)
        d = np.load(BATTERY_PATH)
        stable = d["carry_active"] == 0
        probes_unit, _ = fit_probes(d["h_t"][stable], d["bits"][stable])
        return sim, d, probes_unit

    sim, d, probes_unit = load_all()
    h_t, counts, carry_ = d["h_t"], d["counts"], d["carry_active"]

    col_left, col_right = st.columns([1, 3])
    with col_left:
        st.header("Controls")
        source_c = st.slider("source count c", 0, 14, 7)
        horizon = st.slider("imagination horizon", 15, 60, 35)
        sample_idx = st.slider("initial h_t sample index", 0, 20, 3)

        st.markdown("---")
        do_intervene = st.checkbox("Apply intervention", value=False)
        bit_patch = st.selectbox("bit direction", [0, 1, 2, 3], index=0)
        alpha_patch = st.slider("α (patch magnitude)", -10.0, 10.0, 0.0, step=0.5)
        step_patch = st.slider("intervention at step", 0, horizon - 1, 10)

        st.markdown("---")
        view_mode = st.radio("View", ["Cascade (Option A)", "Step decomposition (Option B)", "Compare natural vs intervened"])

    with col_right:
        try:
            h0 = sample_initial_state(h_t, counts, carry_, source_count=source_c, idx=sample_idx)
        except Exception as e:
            st.error(str(e)); return
        s0 = sim.prior_stoch_from_deter(h0.astype(np.float32))

        if view_mode == "Cascade (Option A)":
            intervention = None
            if do_intervene and alpha_patch != 0:
                intervention = {"step": step_patch, "direction": probes_unit[bit_patch].astype(np.float32),
                                "alpha": alpha_patch, "bit": bit_patch}
            dtraj, straj = sim.imagine_forward(h0.astype(np.float32), s0, horizon=horizon, intervention=intervention)
            title = f"Cascade from source count={source_c}, horizon={horizon}"
            if intervention:
                title += f" | intervention: bit {bit_patch}, α={alpha_patch:+.1f} at t={step_patch}"
            fig, _ = plot_cascade(dtraj, straj, sim, probes_unit, title,
                                  intervention_step=step_patch if intervention else None)
            st.pyplot(fig)
            plt.close(fig)

        elif view_mode == "Step decomposition (Option B)":
            dtraj, straj = sim.imagine_forward(h0.astype(np.float32), s0, horizon=horizon)
            fig = plot_step_decomposition(dtraj, probes_unit,
                                           f"Step decomposition onto bit-flip directions — source={source_c}")
            st.pyplot(fig)
            plt.close(fig)

        else:  # compare
            dtraj_base, straj_base = sim.imagine_forward(h0.astype(np.float32), s0, horizon=horizon)
            intervention = {"step": step_patch, "direction": probes_unit[bit_patch].astype(np.float32),
                            "alpha": alpha_patch, "bit": bit_patch}
            dtraj_int, straj_int = sim.imagine_forward(h0.astype(np.float32), s0, horizon=horizon, intervention=intervention)
            fig = plot_intervention_compare(dtraj_base, straj_base, dtraj_int, straj_int,
                                             sim, probes_unit, intervention,
                                             f"Source count={source_c} — natural vs intervened")
            st.pyplot(fig)
            plt.close(fig)


# ─── Entry point ──────────────────────────────────────────────────────────────

def _running_under_streamlit():
    try:
        import streamlit.runtime.scriptrunner as _sr
        return _sr.get_script_run_ctx() is not None
    except Exception:
        return False


if _running_under_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "streamlit"], default="batch")
    args = parser.parse_args()
    if args.mode == "batch":
        batch_generate()
    else:
        print("Use: streamlit run scripts/viz_binary_world.py")
