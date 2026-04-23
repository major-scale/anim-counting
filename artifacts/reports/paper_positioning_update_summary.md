# Paper Positioning Update: Summary of Edits

Applied 2026-04-23 to `writeups/03-paper.md` based on the positioning-and-novelty DRQ, reference-read verification, and the edit plan resolved with the planner. Complements `artifacts/reports/positioning_reference_reads.md` (the verification record) and `artifacts/reports/causal_intervention_protocol_design_inputs.md` (the earlier Makelov design inputs).

## What changed

### 1. Introduction reframed (§1)

The central empirical headline shifted from the task-physics thesis to the compositional coordinate system. New paragraph (line 29) explicitly frames the result as "validated at a methodological bar (held-out generalization across all transitions + Makelov-faithful against subspace-illusion alternatives) that is unusually rigorous for compositional-linearity claims in learned world models." Task-physics demoted to an interpretive claim with published lineage (Alleman-Lindsey-Fusi 2024; Driscoll-Shenoy-Sussillo 2024; Zhong et al. 2023); our contribution framed as the minimal-pair controlled instantiation in a deep recurrent world model.

### 2. Anti-"orthogonality trivially task-imposed" rebuttal (§1, new subsection)

New subsection "Why the compositional geometry is not trivially task-imposed" placed at end of introduction before Contributions. Three-part defense: (a) orthogonality not forced by encoder (same architecture produces 1D manifold in decimal environment); (b) held-out generalization rules out probes-fit-their-own-direction tautology, with the maximally adversarial 7→8 full cascade as the key evidence; (c) imagination-rollout dynamics use the basis causally (Section 6.4) plus Makelov-faithful intervention (Appendix C).

### 3. Contributions split and renumbered (§1)

C4 previously bundled the decomposition and imagination results. Now split:
- **C4** = compositional decomposition (with methodological-bar language: 100% sign agreement × 15 transitions, held-out generalization, Makelov-faithful single-direction intervention).
- **C5** = internal simulation of carry cascades via imagination rollout (previously a sub-claim of C4).

Renumbered: old C5 (FP Unifier) → C6; old C6 (three integration findings) → C7. No body-text references to contribution labels outside the list, so the renumber is clean.

### 4. Related Work additions (§2.2, §2.3, §2.4, §2.7)

- **§2.2** (emergent numerical representations): added Kantamneni & Tegmark (2025) helical-integer differentiation and Chawla et al. (2026) MetaOthello contrast. Framed MetaOthello as the inverse-direction bracket: they show rotation-equivalence under token remapping, we show rotation-nonequivalence under task-physics change.
- **§2.3** (world models): added Sobotka et al. (NeurIPS 2025 MI Workshop) as the only prior DreamerV3 interpretability work; differentiated on coordinate-level vs compositional-claim methodology.
- **§2.4** (representational geometry): added Gröger, Wen & Brbić (2026) (Aristotelian RH defensive framing), Alleman-Lindsey-Fusi (ICLR 2024) with precise mechanistic differentiation (feedforward Tanh/ReLU dichotomy vs deep RSSM task-physics), Driscoll-Shenoy-Sussillo (Nat. Neurosci. 2024) differentiated at coordinate-vs-motif granularity, and Zhong et al. (2023) Clock-and-Pizza.
- **§2.7** (new subsection): "Causal intervention and compositional features." Situates our Makelov test within the lineage: Geiger et al. DAS, Makelov-Lange-Nanda 2023, Wu et al. 2024 reply, Engels et al. ICLR 2025 (nearest methodological neighbor), RAVEL (closest disentanglement benchmark), Quirke & Barez ICLR 2024 (feedforward carry-circuit differentiation), Nanda 2023 grokking (lineage).

### 5. Imagination rollout promoted (§6.4)

Added a lead-in paragraph to §6.4 framing the imagination result as a primary contribution rather than supporting evidence: "The bit-decomposition basis is not only a static characterization of the posterior representation — it is used by the RSSM's generative dynamics." Existing content preserved.

### 6. Discussion §9.1 reframed

Replaced the previous task-physics-first framing ("The counting manifold is the strongest result. Same architecture, same training objective, same concept, yet the representation geometry follows the physical implementation...") with compositional-coordinate-system-first framing ("The primary empirical result is the compositional decomposition of the binary successor..."). Explicitly acknowledged published lineage for the task-physics interpretation. Added negative-result-against-strong-PRH paragraph with Gröger et al. Aristotelian framing as the compatible-with reading.

### 7. Conclusion §10 reframed

Rewrote opening to lead with the compositional coordinate system (binary world), then the complementary counting-grid result, then task-physics as interpretive payoff with lineage attribution. Removed the simple-thesis closer ("same concept, different physics, different geometry") that presumed task-physics was the de novo claim.

### 8. Appendix C Makelov paragraph sharpened

Added the Wu et al. 2024 counter-context and our defense: "Wu et al. (2024) have argued that null-space effects arise naturally from the data-induced submanifold of activation space and are not by themselves diagnostic of illusion; our test sidesteps this objection by making a strictly stronger claim — rather than defending null-space components as benign, we show that the rowspace-only intervention alone reproduces the full-direction effect, which exceeds what the Makelov test requires and is immune to the Wu et al. counterpoint." Also added the concrete SV ratio (5.4×10⁻⁴) as evidence for decoder full-rank structure.

### 9. Limitations addition (§9.4)

Added the cross-seed basis alignment limitation: the counting-manifold topology is 5-seed, the binary specialist is single-seed (`binary_baseline_s0`), so whether the specific bit-flip basis aligns across independently trained instances — or only the span — is a future analysis. Scoped as current limitation, not silent omission.

### 10. References

Added 13 new bibliography entries in alphabetical order: Alleman-Lindsey-Fusi; Chawla et al. MetaOthello; Driscoll-Shenoy-Sussillo; Engels et al.; Geiger et al. DAS; Gröger-Wen-Brbić Aristotelian; Huang et al. RAVEL; Makelov-Lange-Nanda; Quirke-Barez; Sobotka et al.; Wu et al. reply; Zhong et al. Clock-and-Pizza.

## Flagged issues resolved during edit pass

### (1) Rotation-within-span identifiability — NOT RUN, NOT CLAIMED

The DR-derived task spec asked for framing around "rotation-within-span identifiability + Makelov-faithful" as the methodological bar. **This test has not been run.** Rotation-within-span / DAS-style identifiability was scoped as part of the causal intervention protocol, which is not yet drafted. Under user direction we went with path (B): reframe around the tests we have actually performed (held-out generalization + Makelov), dropped the rotation-within-span language from both introduction and rebuttal, and replaced defense (b) in the rebuttal paragraph with a held-out-generalization argument. Rotation-within-span and the DAS framework are still acknowledged in §2.7 as methodological lineage, without being claimed as tests we have executed.

### (2) "Compositional across all 15 subsets of {0,1,2,3}" — NOT SUPPORTED, NOT CLAIMED

The DR-derived framing gestured at compositional subset-patching across the 15 non-empty subsets of bit positions. **This test has not been run either.** The natural step vectors cover the 15 naturally occurring transitions (0→1 through 14→15); those are all single, double, triple, or quadruple bit-flips that occur in binary counting. Forcing arbitrary non-natural subsets (e.g., flip bits 0 and 2 only) would require compositional intervention patches — separate work. Edits use "across all 15 transitions (0→1 through 14→15)" throughout rather than the subset language.

### (3) DR mischaracterizations from the reference reads

- **Alleman et al. 2024**: DR attributed third authorship to Chung; actual is Fusi. More substantively, the paper's mechanism is Tanh-vs-ReLU activation geometry in one-hidden-layer feedforward networks, not a general task-structure claim. The title is near-verbatim to ours; the mechanistic regimes are disjoint. Under planner direction, front-loaded as must-cite-and-differentiate with explicit mechanistic disjointness rather than softening the differentiation.
- **MetaOthello**: DR said "different organizations for rule-conflict variants." Actual finding is layer-depth-indexed divergence (early layers stay game-agnostic, middle layer identifies game, later specialize). Edit uses the more precise characterization.
- **Quirke & Barez**: minor citation caveat — ICLR 2024 poster is 2-author (Quirke & Barez, arXiv:2310.13121); extended arxiv 2402.02619 adds Clement Neo as third author. Cited the 2-author ICLR version as DR intended.
- **Driscoll-Shenoy-Sussillo**: DR framing was "canonical shared dynamical motifs result"; reference read surfaced a sharper compositional-reuse angle (reusable dynamical primitives for rapid transfer). Under planner direction, adopted the sharper framing — differentiated on "compositional-reuse at linear-coordinate level vs compositional-reuse at dynamical-motif level" — which aligns with the new compositional-coordinate-system headline.

### (4) Hallucinated reference discarded

The DR report cited a third unverified paper "What Do World Models Learn in RL?" (arXiv:2603.21546). Reference-read verification confirmed it cannot be found. Not cited in any edit.

## Self-correction note

The original task spec included language ("rotation-within-span identifiability + Makelov-faithful") drawn from the causal-intervention protocol draft that was mapped forward into paper-edit material without checking which components had actually been run. I flagged this before editing and the planner confirmed path (B): reframe around completed tests rather than running new experiments to backfill claimed framing. The edits preserve the "coordinate system as headline, task-physics as interpretive payoff" spirit of the reframe while staying faithful to what has actually been shown.

## Not touched

Per scope guardrails:
- Abstract (not in task spec; current version already presents binary compositional result prominently in second paragraph)
- Existing self-correction notes in Appendix C (generator bug, terminal-state sampling, split-convention) — preserved as-is
- Section 3 (environments), 4 (measurement battery), 5 (counting manifold results), 7 (FP Unifier), 8 (mechanism of integration) — no framing changes; empirical content unchanged
- Paper title: "When Physics Determines Geometry" leads with task-physics, which is now the interpretive payoff rather than headline thesis. Under the reframe a coordinate-system-forward title would fit better, but the title is also the blog-post title (already shared), and the NeurIPS submission pass is weeks out. Flagged for revisit as part of the final pre-submission pass, when causal-intervention results are in and the final positioning is settled. Not changed in this pass.

## Pointers

- Verification record: `artifacts/reports/positioning_reference_reads.md`
- Held-out test results: `results/binary_heldout_decomposition.md`
- Makelov test results: `results/makelov_subspace_test.md`
- Earlier design inputs for future causal-intervention protocol: `artifacts/reports/causal_intervention_protocol_design_inputs.md`
- Previous edit summaries: `artifacts/reports/planner_update_binary_heldout.md`
