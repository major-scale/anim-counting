# Binary World Representation Geometry — Visualization Screenshots

Captured 2026-04-23 during the build of `scripts/viz_binary_world.py`. Option A (carry cascade + intervention) is the primary focus; Option B (step-vector decomposition) is included as a second view because it turned out to be the single most insightful plot in the batch. All seven PNGs are static matplotlib renders generated in `batch` mode; the same plotting functions back the Streamlit interactive tool.

## Launch

**Interactive (recommended for intuition-building)**:
```bash
streamlit run scripts/viz_binary_world.py
```
Streamlit will open a browser tab. Controls: source count slider, horizon, sample index, intervention checkbox, bit/α/step sliders, view selector (cascade / step-decomposition / compare).

**Static (regenerate these screenshots)**:
```bash
python3 scripts/viz_binary_world.py --mode batch
```

## How to read the visualizations

Two sampling modes for the initial `h_t`:

- **mid**: a generic stable sample at count=c from the middle of a count run. The RSSM sits at its stable attractor and imagination just holds count=c indefinitely. Shows the **resting** geometry.
- **pre-transition** (used in all screenshots below): the last stable sample at count=c in each episode, just before the natural transition to c+1. This h_t already contains **anticipation signal** that the paper's §6.4 shows begins up to 8 steps before the natural transition for full cascades. Imagination from this fork point reproduces the cascade autonomously.

The "natural cascade in imagination" finding (paper §6.4, contribution C5) is **conditional on forking from a pre-transition state**. From a generic mid-stable h_t the RSSM sits at its stable attractor and the cascade never fires. The Streamlit tool exposes this as a toggle (mode slider).

Bit values shown two ways simultaneously: the heatmap color is the **soft** (continuous) decoded bit value from the decoder's `obs[49,53,57,61]` outputs; the overlaid "1" / "0" text is the **hard** threshold at 0.5. Probe signals (bottom panels) are `h_t · ŵ_b` — the raw inner product of the current hidden state with the unit-normalized bit-flip probe direction.

---

## 01_cascade_7to8_natural.png

**What it shows**: Natural imagination from a pre-transition h_t at source count=7 (bits 0111), 40 imagined steps forward. The cascade fires autonomously.

**Why it matters**: This is the paper's core C5 finding as a picture. Bit 0 flips at t≈3, bit 1 at t≈7, bit 2 at t≈11, bit 3 at t≈13–14. Order is cleanly LSB→MSB, spacing is ~4 steps per bit. Matches §6.4's reported posterior-vs-imagination agreement within one timestep. Final state is 1000 (count=8).

Best single image for the "the model internally simulates the carry cascade" claim.

## 02_cascade_depth_comparison.png

**What it shows**: Cascade in imagination for source counts 0→1 (depth 0), 1→2 (depth 1), 3→4 (depth 2), 7→8 (depth 3). Each row a different cascade.

**Why it matters**: Shows the timing scales with depth. Depth-0 (0→1) shows a single bit flip plus a second natural increment later in the rollout (the RSSM keeps going). Depth-1 (1→2) shows bit 0 dropping and bit 1 rising. Depth-2 (3→4) shows the two-bit cascade: bits 0 and 1 drop, bit 2 rises. Depth-3 (7→8) is the full four-bit cascade. Compositional structure is visible across depths.

## 03_intervention_bit2_push.png

**What it shows**: Source count=0, two-panel comparison. Left: natural imagination. Right: intervention pushing bit 2's probe direction with α=+5.0 at t=10.

**Why it matters**: Tests whether a single-direction patch can induce a bit flip that wasn't going to happen otherwise. The intervention clearly pushes bit 2 up immediately (visible heatmap spike), but the downstream effect is more complex than "bit 2 = 1 for the rest of the rollout" — the RSSM's dynamics reassert themselves. Suggests the bit-flip directions are causally effective but the decoder-level outcome is governed by the RSSM attractor structure rather than by any single-direction injection alone. Good data for the "what does intervention actually do" question in the coming causal-intervention protocol.

## 04_intervention_cascade_disrupt.png

**What it shows**: Source count=7 full cascade. Left: natural cascade. Right: anti-push on bit 0 (α=−4.0) applied at t=8, after bit 0 has already flipped.

**Why it matters**: Tests cascade resilience. The anti-push creates a sharp transient (bit 0 probe signal dives to ~−1.4 and recovers), but the cascade otherwise completes normally — bit 1 flips, bit 2 flips, bit 3 rises on schedule. The cascade is not dependent on bit 0's probe value at t=8; by that point the "count is going to 8" information is distributed across h_t and the prior dynamics carry through.

## 05_step_decomposition_cascade.png

**The single most insightful plot in the batch.** Option B rendered.

**What it shows**: During the 7→8 cascade, the step vector (h_{t+1} − h_t) decomposed onto the four unit bit-flip directions. Top panel: per-bit projection. Bottom panel: total step magnitude vs the magnitude of the step's projection onto the four-dimensional bit-flip subspace.

**Why it matters**:
- **Top panel** makes the compositional-decomposition claim visceral. You can see each bit's flip as a single coherent spike in the corresponding direction: bit 0 dives at t=2, bit 1 dives at t=7, bit 2 dives at t=11, bit 3 rises at t=11–13. The order, timing, and magnitude of each bit's contribution are all visible in one glance.
- **Bottom panel** makes the 16% subspace-fraction finding visible. The black curve (total step magnitude) is ~3–4 at the cascade peaks; the grey curve (bit-flip-subspace magnitude) is ~1. The ratio tracks the measured subspace fraction. The step vector is doing a lot of other work besides the bit flips — magnitude outside the bit-flip directions is larger than magnitude inside.

If there's one plot to use for "what does the coordinate system look like," this is it.

## 06_alpha_sweep_bit2.png

**What it shows**: Heatmap sweep. Source count=0, intervention on bit-2 direction at t=8, α ∈ {−6, −4, −2, −1, 0, +1, +2, +4, +6, +8, +10}. Four side-by-side panels, one per decoded bit.

**Why it matters**: Shows how outcome space depends continuously on intervention magnitude. At α=0 (middle row) the natural trajectory is visible. Positive α pushes bit 2 up and has visible knock-on effects on bits 0 and 1's trajectories — the bits are not independent under intervention. Strong negative α suppresses bit 2 deeply but leaves the other bits' natural dynamics mostly intact. This is the kind of view that pays off rich interactive exploration — the Streamlit version lets you vary source count and step interactively.

## 07_alpha_sweep_disrupt.png

**What it shows**: Same heatmap sweep as #6 but testing **how much anti-push on bit 0 at t=2 it takes to break the full 7→8 cascade**. α ∈ {−10, −8, ..., +4}.

**Why it matters**: The cascade survives at all tested α values. Even at α=−10 (strong enough to drive bit 0's decoded value deep into the negative soft-bit region for the full rollout), bits 1, 2, 3 still complete their cascade on schedule and the final state still reaches 1000-ish. This is a robustness finding: the "count is going to 8" information is distributed across h_t widely enough that removing any one bit's contribution doesn't derail the trajectory. It also frames a design input for the main causal-intervention protocol — the "flip bit i" test needs to separate "decoder reads bit i as flipped" from "cascade dynamics respect bit i's flipping."

---

## What worked, what didn't, honest assessment

**Clearly useful**:
- **05 (step decomposition)** — the best single image. Both the compositional and the subspace claims in one plot.
- **01 (7→8 cascade)** — the canonical picture of the paper's C5 imagination-rollout finding.
- **02 (depth comparison)** — a natural follow-up that shows depth-scaling cleanly.

**Useful but requires some interpretation effort**:
- **06, 07 (α sweeps)** — rich, but the reader has to know what they're looking at. Reward interactive exploration more than standalone reading. Good for Streamlit, slightly too dense for static.

**Less immediately striking**:
- **03, 04 (single-α intervention comparisons)** — scientifically valid but the α-sweep heatmap subsumes them. If I were cutting to three screenshots, I'd keep 01, 02, 05 and drop 03/04 in favor of 07.

**One finding that surfaced during the build**:
The imagination cascade fires only from pre-transition fork states (late-stable h_t), not from generic mid-stable h_t. The paper's §6.4 is explicit that forks were 20 steps before transitions; my initial attempt at a "cascade from generic count=7" view produced a flat trajectory (the RSSM sits at its attractor). The Streamlit tool now exposes both modes as a switch so the user can explore both regimes.

## Known limitations

- **Single-seed only.** All visualizations are from the one binary specialist checkpoint (`binary_baseline_s0`). Whether the cascade timing or intervention responses are consistent across seeds is not tested (a paper §9.4 limitation).
- **Count=15 excluded.** The split-convention from the terminal-state sampling asymmetry carries over here; the visualizer caps source_count at 14.
- **Probe signal baselines aren't calibrated to (0, 1).** The Ridge-fit probes give h·ŵ ≈ 1.5–2.0 when the bit is 1 and ≈ −0.5 when the bit is 0. This is a property of L2-regularized regression on binary targets, not a bug. The sign is what matters for the cascade story; the absolute values are for relative comparison.
- **Stoch is computed from h_t via prior dynamics**, not recovered from the actual posterior that accompanied the h_t during battery collection. Valid for A/B-comparison purposes (same stoch per sample across conditions) but means decoder outputs may not exactly match what a fully-wired posterior-path inference would produce.

## Files

- `scripts/viz_binary_world.py` — source. Contains Simulator class, plotting functions, batch mode, and Streamlit wrapper.
- `artifacts/screenshots/viz_binary_20260423/01..07_*.png` — the seven generated views.
