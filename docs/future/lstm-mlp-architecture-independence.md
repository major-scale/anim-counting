# LSTM Predictor and Feedforward MLP — Architecture Independence Test

**Status:** Ready to implement. Waiting on random projection seed 0 results.
**Priority:** High — CPU experiment, can run alongside GPU training.
**Estimated time:** MLP < 1 hour, LSTM 3-6 hours (both CPU).

---

## Motivation

Our counting manifold finding rests on a single architecture (DreamerV3's RSSM). A reviewer can always say "maybe this is a quirk of the GRU-based RSSM." Two simple experiments — one positive, one negative — dramatically strengthen the claim:

- **LSTM succeeds** → any recurrent predictor finds the manifold
- **MLP fails** → temporal memory is necessary

Together: "recurrent prediction finds this, and memory is the necessary ingredient."

---

## Shared Setup

Minimal next-observation predictors. No RL. No reward. No policy. No encoder-decoder separation. No stochastic latent variables. Just `obs_t → predict obs_{t+1}` trained with MSE loss.

**Training data:** Collect 200K-300K transitions from the existing counting environment using the trained DreamerV3's policy (or random policy — just needs to cover count range 0-25). Store as `(obs_t, obs_{t+1})` pairs, preserving episode boundaries for LSTM hidden state resets. Standard 82-dim observation vector, grid baseline, full observation channels.

---

## Model 1: LSTM Predictor

```
obs_t (82-dim) → Linear(82, 256) → LSTM(256, hidden_size=256) → Linear(256, 82) → pred_obs_{t+1}
```

- Loss: MSE between predicted and actual next observation
- Optimizer: Adam, lr=1e-3
- Train for 300K steps (or 100 epochs over dataset)
- Reset LSTM hidden state at episode boundaries
- Batch size: 32 sequences of length 64
- ~50-80 lines of PyTorch, CPU, 3-6 hours

---

## Model 2: Feedforward MLP

```
obs_t (82-dim) → Linear(82, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 82) → pred_obs_{t+1}
```

- Same MSE loss, same optimizer, same data
- No hidden state, no memory — each prediction is independent
- Batch size: 256 (no sequential dependency)
- Train on shuffled individual transitions (correct protocol for memoryless model)
- ~30-50 lines of PyTorch, CPU, < 1 hour

---

## Analysis (identical for both)

**Target representations:**
- LSTM: hidden state `h_t` (256-dim) at each timestep
- MLP: second hidden layer activations (256-dim) at each timestep

**Evaluation procedure:** Run evaluation episodes with trained DreamerV3 policy driving the environment. At each timestep, feed observation through both LSTM and MLP, record internal representations + ground truth count.

**Measurement battery:**
- Linear probe: regress count from representation → R²
- RSA: RDM from representations vs count-based ordinal RDM
- PCA: variance explained by PC1
- Nearest neighbor accuracy: is nearest representation from adjacent count?
- If LSTM shows strong results: full battery including persistent homology

---

## Expected Results

| Model | R² | RSA | PCA PC1 | Interpretation |
|-------|-----|-----|---------|----------------|
| DreamerV3 (baseline) | 0.998 | 0.982 | 73% | Reference |
| LSTM | >0.95 expected | >0.9 expected | High | Recurrent prediction suffices |
| MLP | Low | Low | Low | Memory is necessary |

**If LSTM succeeds and MLP fails:** Paper gains: "A vanilla LSTM predictor produces comparable manifold quality (R²=X, RSA=Y), while a feedforward MLP without temporal memory does not (R²=Z, RSA=W), confirming that the counting manifold requires recurrent prediction but not RSSM-specific engineering."

**If LSTM fails:** DreamerV3's innovations (KL balancing, categorical states, symlog) actually matter. Less dramatic but still worth reporting.

---

## Implementation Notes

- LSTM MUST process observations sequentially within episodes, maintaining hidden state across timesteps, resetting between episodes. Breaking temporal order destroys the sequential counting signal.
- MLP SHOULD be trained on shuffled transitions — correct protocol for memoryless model.
- Both models trained on same underlying data for fair comparison.
- LSTM hidden state extraction: run forward through complete episodes, save `h_t` at every timestep with ground truth count. Produces same `(representation, count)` pairs as DreamerV3 evaluation.

---

## Bonus Variant (if time permits)

Stack last k observations (k=2, 4, 8) as input to MLP, creating fixed-window pseudo-memory. This titrates how much temporal context is needed. If k=8 produces a weak manifold but k=2 doesn't, reveals roughly how much history matters.

---

## Relationship to Other Experiments

| Experiment | Tests | Axis |
|-----------|-------|------|
| Random projection (running now) | Does spatial semantics matter? | Observation structure |
| LSTM/MLP (this) | Does the specific architecture matter? | Model architecture |
| Ablations (done) | Do specific observation channels matter? | Input features |

If random projection AND LSTM both succeed: the manifold emerges from any recurrent predictor applied to any distance-preserving encoding of sequential counting. That's a very general claim.
