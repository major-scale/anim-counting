#!/usr/bin/env python3
"""
Test: Does the RSSM produce valid manifold geometry when fed correct observations
but zero actions?

Hypothesis: The RSSM hidden state is overwhelmingly driven by observations (512-dim
encoded), not actions (1-dim). If true, feeding action=0 every step should still
produce deter states with GHE < 0.5, high RSA, and correct topology.

Uses vectorized numpy — no Python for-loops in the forward pass.
"""

import sys, os, json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/workspace/bridge/scripts")

OBS_SIZE = 80
DETER = 512
HIDDEN = 512
STOCH_CATS = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_CATS * STOCH_CLASSES  # 1024
LN_EPS = 1e-3

MODELS_DIR = Path("/workspace/projects/jamstack-v1/packages/signal-app/public/models")
BIN_PATH = MODELS_DIR / "dreamer_weights.bin"
MANIFEST_PATH = MODELS_DIR / "dreamer_manifest.json"


# ─── Weight loading ───────────────────────────────────────────────────
def load_weights():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(BIN_PATH, "rb") as f:
        raw = f.read()
    weights = {}
    for name, entry in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype="<f4", count=entry["length"], offset=entry["offset"]).copy()
        arr = arr.reshape(entry["shape"])
        weights[name] = arr
    return weights


# ─── Vectorized RSSM ─────────────────────────────────────────────────
def ln(x, w, b):
    m = x.mean()
    v = ((x - m) ** 2).mean()
    return (x - m) / np.sqrt(v + LN_EPS) * w + b


def silu_v(x):
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sigmoid_v(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def argmax_one_hot(logits):
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    reshaped = logits.reshape(STOCH_CATS, STOCH_CLASSES)
    indices = reshaped.argmax(axis=1)
    for c in range(STOCH_CATS):
        out[c * STOCH_CLASSES + indices[c]] = 1.0
    return out


class FastRSSM:
    def __init__(self, w):
        self.w = w
        self.deter = np.tanh(w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True

        # Compute initial prior
        h = silu_v(ln(w["img_out_w"] @ self.deter, w["img_out_norm_w"], w["img_out_norm_b"]))
        logits = w["imgs_stat_w"] @ h + w["imgs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)

    def reset(self):
        self.deter = np.tanh(self.w["deter_init_w"].flatten()).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        h = silu_v(ln(self.w["img_out_w"] @ self.deter, self.w["img_out_norm_w"], self.w["img_out_norm_b"]))
        logits = self.w["imgs_stat_w"] @ h + self.w["imgs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)
        self.is_first = True

    def step(self, obs_raw, action=0.0):
        """Full obs_step: encode → img_step → posterior. Returns deter copy."""
        w = self.w

        # Symlog + encode
        x = np.sign(obs_raw) * np.log(np.abs(obs_raw) + 1)
        h = silu_v(ln(w["enc_linear0_w"] @ x, w["enc_norm0_w"], w["enc_norm0_b"]))
        h = silu_v(ln(w["enc_linear1_w"] @ h, w["enc_norm1_w"], w["enc_norm1_b"]))
        embed = silu_v(ln(w["enc_linear2_w"] @ h, w["enc_norm2_w"], w["enc_norm2_b"]))

        # img_step
        act = 0.0 if self.is_first else action
        self.is_first = False
        cat = np.concatenate([self.stoch, [act]]).astype(np.float32)
        img_h = silu_v(ln(w["img_in_w"] @ cat, w["img_in_norm_w"], w["img_in_norm_b"]))

        gru_in = np.concatenate([img_h, self.deter]).astype(np.float32)
        gru_ln = ln(w["gru_w"] @ gru_in, w["gru_norm_w"], w["gru_norm_b"])

        # Vectorized GRU gates
        reset = sigmoid_v(gru_ln[:DETER])
        cand = np.tanh(gru_ln[DETER:2*DETER] * reset)
        update = sigmoid_v(gru_ln[2*DETER:] - 1.0)
        self.deter = (update * cand + (1.0 - update) * self.deter).astype(np.float32)

        # Posterior
        inp = np.concatenate([self.deter, embed]).astype(np.float32)
        h = silu_v(ln(w["obs_out_w"] @ inp, w["obs_out_norm_w"], w["obs_out_norm_b"]))
        logits = w["obs_stat_w"] @ h + w["obs_stat_b"].flatten()
        self.stoch = argmax_one_hot(logits)

        return self.deter.copy()


# ─── Data collection ──────────────────────────────────────────────────
def collect_deter_states(w, condition="zero", n_episodes=3, blob_counts=None):
    from counting_env_pure import CountingWorldEnv
    if blob_counts is None:
        blob_counts = [8, 15, 25]

    deter_by_count = defaultdict(list)
    total = 0
    t0 = time.time()

    for ep in range(n_episodes):
        for bc in blob_counts:
            env = CountingWorldEnv(blob_count_min=bc, blob_count_max=bc)
            model = FastRSSM(w)
            vec = env.reset()
            done = False

            while not done:
                obs = vec[:OBS_SIZE].astype(np.float32)
                count = int(vec[81])

                if condition == "zero":
                    action = 0.0
                elif condition == "random":
                    action = np.random.uniform(-1, 1)
                else:
                    action = 0.0

                deter = model.step(obs, action)
                deter_by_count[count].append(deter)
                total += 1
                vec, reward, done, info = env.step(-0.995)
            env.close()

        elapsed = time.time() - t0
        fps = total / elapsed if elapsed > 0 else 0
        print(f"    Episode {ep+1}/{n_episodes}: {total} samples ({fps:.0f} steps/s)")

    return deter_by_count


# ─── Metrics ──────────────────────────────────────────────────────────
def compute_metrics(deter_by_count):
    counts = sorted(deter_by_count.keys())
    if len(counts) < 3:
        return {"error": "too few counts"}

    centroids = {c: np.stack(deter_by_count[c]).mean(axis=0) for c in counts}
    centroid_matrix = np.stack([centroids[c] for c in counts])
    count_array = np.array(counts, dtype=np.float64)

    # PCA
    mean = centroid_matrix.mean(axis=0)
    centered = centroid_matrix - mean
    cov = np.cov(centered.T)
    evals = np.linalg.eigvalsh(cov)[::-1]
    pc1_var = evals[0] / evals.sum()

    # Linear probe
    X_all = np.vstack([np.stack(deter_by_count[c]) for c in counts])
    y_all = np.concatenate([[c] * len(deter_by_count[c]) for c in counts]).astype(np.float64)
    alpha = 1.0
    w = np.linalg.solve(X_all.T @ X_all + alpha * np.eye(DETER), X_all.T @ y_all)
    bias = y_all.mean() - w @ X_all.mean(axis=0)
    preds = X_all @ w + bias
    r_squared = 1 - ((y_all - preds)**2).sum() / ((y_all - y_all.mean())**2).sum()
    exact_acc = (np.clip(np.round(preds), 0, 25).astype(int) == y_all.astype(int)).mean()

    # RSA
    n_c = len(counts)
    neural_dists = np.array([[np.linalg.norm(centroids[counts[i]] - centroids[counts[j]])
                              for j in range(n_c)] for i in range(n_c)])
    count_dists = np.abs(count_array[:, None] - count_array[None, :])
    triu = np.triu_indices(n_c, k=1)
    rsa = np.corrcoef(neural_dists[triu], count_dists[triu])[0, 1]

    # Step sizes and GHE
    step_sizes = np.array([np.linalg.norm(centroids[counts[i+1]] - centroids[counts[i]])
                           for i in range(n_c - 1)])
    mean_step = step_sizes.mean()
    step_cv = step_sizes.std() / mean_step if mean_step > 0 else float('inf')
    arc_lengths = np.concatenate([[0], np.cumsum(step_sizes)])
    arc_r2 = np.corrcoef(count_array, arc_lengths)[0, 1] ** 2 if arc_lengths[-1] > 0 else 0

    geo_errors = []
    for i in range(n_c):
        for j in range(i+1, n_c):
            actual = step_sizes[i:j].sum()
            expected = abs(counts[j] - counts[i]) * mean_step
            if expected > 0:
                geo_errors.append(abs(actual - expected) / expected)
    ghe = np.mean(geo_errors)

    # NN accuracy
    nn_ok = 0
    for i in range(n_c):
        dists = [np.linalg.norm(centroids[counts[i]] - centroids[counts[j]])
                 for j in range(n_c) if j != i]
        idxs = [j for j in range(n_c) if j != i]
        nearest = idxs[np.argmin(dists)]
        if abs(counts[nearest] - counts[i]) <= 1:
            nn_ok += 1
    nn_acc = nn_ok / n_c

    return {
        "ghe": ghe, "arc_r2": arc_r2, "rsa": rsa, "r_squared": r_squared,
        "exact_acc": exact_acc, "nn_acc": nn_acc, "pc1_var": pc1_var,
        "step_cv": step_cv, "n_counts": len(counts), "n_samples": len(y_all),
    }


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=== RSSM Zero-Action Experiment ===\n")
    print("Loading weights...")
    w = load_weights()
    print(f"  {len(w)} tensors\n")

    results = {}
    for cond in ["zero", "random"]:
        print(f"--- Condition: action={cond} ---")
        deter_by_count = collect_deter_states(w, condition=cond, n_episodes=3)
        m = compute_metrics(deter_by_count)
        results[cond] = m
        print(f"  GHE:       {m['ghe']:.4f}  {'PASS' if m['ghe'] < 0.5 else 'FAIL'}")
        print(f"  Arc R²:    {m['arc_r2']:.4f}")
        print(f"  RSA:       {m['rsa']:.4f}")
        print(f"  Probe R²:  {m['r_squared']:.6f}")
        print(f"  Exact acc: {m['exact_acc']:.1%}")
        print(f"  NN acc:    {m['nn_acc']:.1%}")
        print(f"  PCA PC1:   {m['pc1_var']:.1%}")
        print(f"  Step CV:   {m['step_cv']:.4f}")
        print(f"  Samples:   {m['n_samples']}\n")

    # Summary
    print("=" * 72)
    print(f"{'Metric':<12} | {'action=zero':>14} | {'action=random':>14} | {'baseline*':>14}")
    print("-" * 72)
    base = {"ghe": 0.329, "arc_r2": 0.998, "rsa": 0.982, "r_squared": 0.998,
            "exact_acc": 0.96, "pc1_var": 0.73, "step_cv": 0.39}
    for key, lab in [("ghe","GHE"),("arc_r2","Arc R²"),("rsa","RSA"),
                     ("r_squared","Probe R²"),("exact_acc","Exact"),
                     ("pc1_var","PCA PC1"),("step_cv","Step CV")]:
        fmt = ".4f" if key not in ("exact_acc","pc1_var") else ".1%"
        z = f"{results['zero'][key]:{fmt}}"
        r = f"{results['random'][key]:{fmt}}"
        b = f"{base[key]:{fmt}}"
        print(f"{lab:<12} | {z:>14} | {r:>14} | {b:>14}")
    print("-" * 72)
    print("* baseline = DreamerV3 with policy actions (6-seed mean from training)\n")

    z = results["zero"]
    if z["ghe"] < 0.5 and z["rsa"] > 0.9:
        print("VERDICT: RSSM WORKS with zero actions!")
        print("→ Port full RSSM to browser, run from frame 0, feed obs each frame.")
        print("→ The deter state IS the manifold. No probe approximation needed.")
    else:
        print("VERDICT: RSSM does NOT work with zero actions.")
        print(f"  GHE={z['ghe']:.4f} (need <0.5), RSA={z['rsa']:.4f} (need >0.9)")


if __name__ == "__main__":
    main()
