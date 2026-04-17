# Causal Intervention Protocol — Design Inputs

## What this document is

Three concrete findings that surfaced during the Makelov subspace-illusion test (April 2026) and that should feed into the main causal intervention protocol when it gets drafted. Each finding has implications for specific protocol decisions — α calibration, threshold design, null-control construction, pass/fail criteria. Capturing them here so they don't decay between now and when the protocol is written.

## What this document is not

This is not the causal intervention protocol. The protocol itself will be drafted separately and will synthesize Opus 4.7's turn 1-2 design collaboration, the Makelov test results, the Sobotka/Makelov literature reads, and the findings below into a coherent standalone document. This file is feed-forward inputs, not a specification.

## How to read this note

For each finding: what was observed during the Makelov pilot, what it implies for protocol design, which protocol sections or decisions it bears on. The section references below (§1.1 / §1.3 / §2.6 / §5.1) point into the Opus 4.7 turn-2 design draft structure; the real protocol may renumber.

---

## Finding 1 — α calibration: per-bit sweep necessary, natural-scale as lower bound

### What was observed

α_nat defined as |(μ_{c+1} − μ_c) · ŵ_i| — the projection of the natural step vector onto the bit-flip probe direction — drives decoder-level bit flips cleanly for bits 0 and 1 (M1_A = 0.13 and 0.39 respectively, statistically nonzero) but produces **zero decoder flips** for bits 2 and 3 (M1_A = 0.00 at α_nat = 1.96 and 2.67). The degeneracy is one-sided: the intervention is too weak to drive the decoder, not wrong in direction.

The mechanism is geometric. α_nat captures only the component of the natural step vector aligned with the bit-subspace — roughly the 16% subspace fraction already measured. For transitions where the non-bit-flip variance is large (deeper carry depths, more complex transitions, or in this case terminal transitions), the bit-subspace projection is a small fraction of what the full step delivers to the decoder. Bits that flip rarely (bit 3 flips on exactly one transition; bit 2 on three) also have less data to calibrate against.

### What it implies for protocol design

1. **The single-α_nat operating point is insufficient.** Any protocol that specifies "intervene at α_nat and measure M1" will produce false-negative results for deeper bits. The natural-scale anchor should be treated as a **lower bound** on the useful α range, not the operating point.

2. **Per-bit calibration is necessary.** Sweeping α until each bit's intervention saturates (or reaches a defensible positive threshold, e.g. M1_A ≥ 0.7) gives per-bit operating points that reflect the actual scale needed to drive that bit's flip through the decoder nonlinearity. Pooled across-bit α values will under- or over-shoot per-bit optima.

3. **The sweep range needs to be wide enough to cover the deeper-bit scale.** Based on the Makelov pilot: α_nat ≈ 2.0 produces zero flips for bit 2; a sweep that tops out at 2× α_nat won't resolve the issue. Recommend sweep range at minimum [0.5 × α_nat, 10 × α_nat] for each bit, with the option to extend further if saturation isn't reached.

4. **Reconstruction-cosine-style caveats apply.** The main protocol should distinguish between two related but different α definitions: (a) the projection α_nat used in the Makelov test, and (b) the pooled step-magnitude m_c used in Opus 4.7's turn 2 (see §1.1). These measure different things; both have failure modes. A per-bit sweep sidesteps the need to commit to either.

### Protocol sections affected

- **§1.1 (α sweep specification)**: strengthen from "quantile-anchored plus natural-scale" to "per-bit calibration sweep, with natural-scale treated as a lower bound on the range." Add rationale pointing to the bit 2/3 degeneracy in the Makelov pilot.
- **§5.1 (defensible positive result criteria)**: ensure criteria are stated per-bit so a protocol that fails on one bit at α_nat can still succeed at a higher α for that bit without poisoning the aggregate verdict.

---

## Finding 2 — 0→1 vs 1→0 asymmetry: large and material at α_nat

### What was observed

At α_nat, flips that push "bit off" (current bit = 1, target = 0, patch along −ŵ_i) succeed at rates 0.27–1.00 depending on source count. Flips that push "bit on" (current = 0, target = 1, patch along +ŵ_i) succeed at essentially **zero** rate across every source count tested. Pattern holds for both bit 0 and bit 1.

Per-cell examples at α_nat:
- Bit 0 odd-source (1→0 flip): c=1 → 0.60, c=3 → 0.36, c=5 → 0.27, c=9 → 0.46
- Bit 0 even-source (0→1 flip): c=0 → 0.04, c=2 → 0.00, c=4 → 0.00, c=10 → 0.00
- Bit 1 odd-bit source (1→0 flip): c=3 → 1.00, c=7 → 0.96, c=11 → 0.74
- Bit 1 even-bit source (0→1 flip): c=1 → 0.00, c=5 → 0.00, c=9 → 0.00

### Plausible mechanisms

Multiple candidate explanations — protocol doesn't need to commit to which, but should stratify:

1. **Bias-term asymmetry**: the decoder's output bias may favor "bit off" as the default readout, so pushing toward 0 goes with the gradient while pushing toward 1 must overcome the bias.

2. **Learned threshold asymmetry**: the natural training data has asymmetric bit-state distributions (low counts have more 0 bits than 1 bits; terminal counts invert this), and the decoder may have learned asymmetric decision thresholds as a result.

3. **Decoder-geometry-specific α threshold**: the intervention at α_nat happens to be below the 0→1 threshold but above the 1→0 threshold for reasons specific to the decoder's nonlinear geometry at natural h_t values.

### What it implies for protocol design

1. **All M1 metrics must be reported stratified by flip direction.** Aggregating across 0→1 and 1→0 hides asymmetries this large. Opus 4.7's turn 2 already included per-direction reporting as a mitigation (§2.6); this finding escalates it from "nice to have" to "load-bearing."

2. **Null controls should match the sign distribution of their corresponding bit-flip patches.** A random-direction null that adds equal amounts of +α and −α across samples doesn't control for a decoder that responds asymmetrically to sign. If the real bit-flip intervention is +α-heavy for some cells and −α-heavy for others, the matched null must mirror that distribution.

3. **Pre-registered success thresholds must be per-direction, not averaged.** A threshold like "M1 ≥ 0.70 with symmetry within 0.15" (from Opus 4.7 §5.1) needs per-direction rates as the primary and symmetry as a secondary check — averaging hides the failure mode where one direction succeeds and the other doesn't.

4. **The asymmetry itself is a finding worth reporting in the paper, regardless of the protocol's primary results.** A decoder that responds asymmetrically to patches along a symmetric-looking probe direction tells us something about the representation's compositional structure — the "coordinate system for binary arithmetic" framing may need to accommodate the asymmetry, or the asymmetry may itself be evidence about how bit-flip directions compose with decoder state.

### Protocol sections affected

- **§2.6 (bit-flip asymmetry)**: strengthen from "report separately" to "the main protocol assumes large asymmetry at natural-scale α based on pilot evidence (Makelov test, bits 0 and 1); all thresholds are per-direction; null controls match sign distributions."
- **§5.1 (success criteria)**: require per-direction rates to meet threshold independently, not averaged.
- **Discussion / findings**: flag asymmetry as a substantive finding, not a nuisance.

---

## Finding 3 — ratio-metric edge cases: ε = 0.05 minimum denominator, Test 2 + decomposition fallback

### What was observed

The Makelov test's pre-registered thresholds (ratio M1_B / M1_A ≥ 0.80 for pass, < 0.50 for fail) implicitly assumed M1_A > 0. When M1_A = 0 — as happened for bits 2 and 3 at α_nat — the ratio is undefined (NaN). The task-level verdict classifier produced "degenerate" for these bits rather than a clean pass/fail, and the overall verdict had to be derived from the decomposition and Test 2 lines.

This wasn't a failure of the Makelov test — the other two lines cleanly supported faithfulness for bits 2 and 3 — but it was a gap in the pre-registration that required on-the-fly interpretation.

### What it implies for protocol design

1. **Ratio-based success metrics need a minimum denominator.** The fix is specifying: if M1_A < ε (recommended ε = 0.05, i.e. 5% baseline flip rate), the ratio test is declared inconclusive at that α and the protocol falls back on Test 2 (class separation) and decomposition fractions. The inconclusive verdict at low α doesn't preclude retrying at higher α — it just signals that Step 3 alone can't yield a ratio-based verdict at that operating point.

2. **The fallback evidence lines should be pre-registered alongside the primary.** Test 2 class separation (AUC, Cohen's d on w_i^null vs w_i^row projections) and decomposition fractions (‖w_null‖/‖w‖ vs random baseline) are both well-defined independent of α. Protocol should specify upfront that these are secondary evidence sources that become primary when the intervention-based evidence is undefined.

3. **The ε threshold should be calibrated to the protocol's sample size.** With 100 samples per cell, M1 = 0.05 corresponds to 5 flips expected — below this, bootstrap CIs on M1_B / M1_A become very wide and the ratio becomes numerically unstable even before the zero-denominator case. Larger sample sizes could justify lower ε.

### Protocol sections affected

- **§1.3 (success thresholds)**: add: "ratio-based metrics require M1_A ≥ ε (ε = 0.05) to be well-defined. When M1_A < ε, the intervention test is declared inconclusive at that α; verdict derives from Test 2 and decomposition fractions alone until a higher α is swept."
- **§5.1 (defensible positive result criteria)**: clarify the fallback hierarchy — primary evidence is intervention-based M1 ratio; secondary is row-space class separation (AUC ≥ 0.95 under null); tertiary is decomposition fraction within random-direction baseline.

---

## Pointers

- **Makelov test results**: `results/makelov_subspace_test.md` (human-readable writeup with verdict, per-bit tables, methodology) and `results/makelov_subspace_test.json` (full numerics including per-cell M1 values, BCa CIs, SV spectrum, class-separation Cohen's d / AUC per bit).
- **Paper methods reference**: `writeups/03-paper.md` Appendix C, paragraph "Subspace-illusion test (Makelov et al. 2023)" — the one-paragraph writeup citing the test and pointing to the full results.
- **Planner audit trail**: `artifacts/reports/planner_update_binary_heldout.md` documents the held-out decomposition work that preceded this; the Makelov test was a follow-up diagnostic motivated by the same skeptical read.
- **Held-out results**: `results/binary_heldout_decomposition.md` — the earlier test that established the sign-agreement / orthogonality baseline against which the Makelov test added the decoder-faithfulness dimension.
- **Design-collaboration history**: the full context for these findings lives in the conversation thread where the Makelov test was commissioned and interpreted. When the main protocol gets drafted, consult that thread for the reasoning behind the three findings beyond what's captured here — particularly the discussion of why per-bit calibration replaces pooled α_nat, and why the 1→0 / 0→1 asymmetry is worth reporting as a substantive finding rather than absorbing as a nuisance.

---

## Scope note for the future protocol drafter

These three inputs are the Makelov-test-derived subset of the design inputs. The full protocol will also need to incorporate:

- Opus 4.7 turn 1-2 α-sweep design (quantile anchoring, pooled step-magnitude α)
- Sobotka/Makelov literature review findings beyond the illusion test itself
- The full null-control battery (random directions, rotations, matched-magnitude noise, non-bit-flip directions within the count-correlated subspace)
- Compositional intervention design (multi-bit patches, ordering effects, cascade predictions)
- Multi-seed replication plan (currently a known gap — single seed `binary_baseline_s0`)

Those live elsewhere and aren't reproduced here. This note captures only what the Makelov test taught us that feeds forward.
