#!/usr/bin/env python3
"""
Simulate the browser DreamerV3 forward pass in Python.

Reimplements the exact forward pass from dreamerInference.ts:
  - LN_EPS = 1e-3
  - Encoder: 3x (Linear -> LayerNorm -> SiLU)
  - img_step: concat(stoch[1024], action[1]) -> Linear(1025->512) -> LN -> SiLU
              -> concat(hidden, deter) -> Linear(1024->1536) -> LN
              -> split 3x512: reset=sigmoid, cand=tanh(ln*reset), update=sigmoid(ln-1.0)
              -> new_deter = update*cand + (1-update)*deter
  - posterior: concat(deter[512], embed[512]) -> Linear(1024->512) -> LN -> SiLU
              -> Linear(512->1024) + bias -> argmax one-hot
  - Initial state: deter=tanh(W), stoch from prior(argmax)
  - action=0.0 always (matching browser)
  - is_first=True for step 0 (action forced to 0)
"""

import json
import struct
import numpy as np
from pathlib import Path

# ─── Constants (matching TS) ────────────────────────────────────────────
OBS_SIZE = 80
DETER = 512
HIDDEN = 512
STOCH_CATS = 32
STOCH_CLASSES = 32
STOCH_FLAT = STOCH_CATS * STOCH_CLASSES  # 1024
IMG_IN_DIM = STOCH_FLAT + 1  # 1025
GRU_IN_DIM = HIDDEN + DETER  # 1024
GRU_OUT_DIM = 3 * DETER  # 1536
OBS_OUT_IN = DETER + HIDDEN  # 1024
LN_EPS = 1e-3

# ─── File paths ─────────────────────────────────────────────────────────
MODELS_DIR = Path("/workspace/projects/jamstack-v1/packages/signal-app/public/models")
BIN_PATH = MODELS_DIR / "dreamer_weights.bin"
MANIFEST_PATH = MODELS_DIR / "dreamer_manifest.json"
PCA_PATH = MODELS_DIR / "manifold_pca.json"
PARITY_PATH = MODELS_DIR / "parity_test.json"


# ─── Weight loading ────────────────────────────────────────────────────
def load_weights(bin_path: Path, manifest_path: Path) -> dict[str, np.ndarray]:
    """Load binary weights using the manifest, returning named numpy arrays."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    with open(bin_path, "rb") as f:
        raw = f.read()

    weights = {}
    for name, entry in manifest["tensors"].items():
        offset = entry["offset"]
        length = entry["length"]
        shape = entry["shape"]
        # float32 little-endian
        arr = np.frombuffer(raw, dtype="<f4", count=length, offset=offset).copy()
        arr = arr.reshape(shape)
        weights[name] = arr

    return weights


# ─── Math primitives (matching TS exactly) ──────────────────────────────
def matmul(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """W @ x: W is [rows, cols], x is [cols]. Returns [rows]."""
    return W @ x


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """LayerNorm with LN_EPS=1e-3, matching TS implementation."""
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    inv_std = 1.0 / np.sqrt(var + LN_EPS)
    return (x - mean) * inv_std * w + b


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def argmax_one_hot(logits: np.ndarray) -> np.ndarray:
    """Argmax one-hot: [STOCH_FLAT] -> [STOCH_FLAT], per-categorical argmax."""
    out = np.zeros(STOCH_FLAT, dtype=np.float32)
    for c in range(STOCH_CATS):
        base = c * STOCH_CLASSES
        chunk = logits[base : base + STOCH_CLASSES]
        max_idx = np.argmax(chunk)
        out[base + max_idx] = 1.0
    return out


# ─── DreamerWorldModel ────────────────────────────────────────────────
class DreamerWorldModel:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights
        self.deter = np.zeros(DETER, dtype=np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self.is_first = True
        self.step_count = 0
        self.reset()

    def reset(self):
        """Match TS reset(): deter = tanh(W), stoch from prior."""
        init_w = self.w["deter_init_w"].flatten()
        self.deter = np.tanh(init_w).astype(np.float32)
        self.stoch = np.zeros(STOCH_FLAT, dtype=np.float32)
        self._compute_prior()
        self.is_first = True
        self.step_count = 0

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        """Encoder: 3x (Linear -> LN -> SiLU). obs[80] -> embed[512]."""
        x = obs[:OBS_SIZE].astype(np.float32)

        # Layer 0
        h = matmul(self.w["enc_linear0_w"], x)
        h = layer_norm(h, self.w["enc_norm0_w"], self.w["enc_norm0_b"])
        h = silu(h)

        # Layer 1
        h = matmul(self.w["enc_linear1_w"], h)
        h = layer_norm(h, self.w["enc_norm1_w"], self.w["enc_norm1_b"])
        h = silu(h)

        # Layer 2
        h = matmul(self.w["enc_linear2_w"], h)
        h = layer_norm(h, self.w["enc_norm2_w"], self.w["enc_norm2_b"])
        h = silu(h)

        return h.copy()

    def _img_step(self, action: float):
        """RSSM img_step: updates deter in place."""
        # concat(stoch[1024], action[1]) -> [1025]
        cat = np.concatenate([self.stoch, [action]]).astype(np.float32)

        # img_in: Linear(1025->512) -> LN -> SiLU
        img_in = matmul(self.w["img_in_w"], cat)
        img_in_ln = layer_norm(img_in, self.w["img_in_norm_w"], self.w["img_in_norm_b"])
        img_in_ln = silu(img_in_ln)

        # GRU input: concat(hidden[512], deter[512]) = [1024]
        gru_in = np.concatenate([img_in_ln, self.deter]).astype(np.float32)

        # GRU: Linear(1024->1536) -> LN -> split 3x512
        gru_raw = matmul(self.w["gru_w"], gru_in)
        gru_ln = layer_norm(gru_raw, self.w["gru_norm_w"], self.w["gru_norm_b"])

        # Split and apply gates
        for i in range(DETER):
            reset_val = sigmoid(gru_ln[i])
            cand_val = np.tanh(gru_ln[DETER + i] * reset_val)
            update_val = sigmoid(gru_ln[2 * DETER + i] - 1.0)
            self.deter[i] = update_val * cand_val + (1.0 - update_val) * self.deter[i]

    def _compute_prior(self):
        """Prior: deter -> img_out -> imgs_stat -> argmax one-hot stoch."""
        h = matmul(self.w["img_out_w"], self.deter)
        h = layer_norm(h, self.w["img_out_norm_w"], self.w["img_out_norm_b"])
        h = silu(h)

        logits = matmul(self.w["imgs_stat_w"], h)
        logits = logits + self.w["imgs_stat_b"].flatten()

        self.stoch = argmax_one_hot(logits)

    def _compute_posterior(self, embed: np.ndarray):
        """Posterior: concat(deter, embed) -> obs_out -> obs_stat -> argmax one-hot stoch."""
        inp = np.concatenate([self.deter, embed]).astype(np.float32)

        h = matmul(self.w["obs_out_w"], inp)
        h = layer_norm(h, self.w["obs_out_norm_w"], self.w["obs_out_norm_b"])
        h = silu(h)

        logits = matmul(self.w["obs_stat_w"], h)
        logits = logits + self.w["obs_stat_b"].flatten()

        self.stoch = argmax_one_hot(logits)

    def obs_step(self, obs: np.ndarray, action: float) -> np.ndarray:
        """
        Full obs_step: encode obs -> img_step -> posterior.
        Returns reference to internal deter[512].
        """
        # 1. Encode observation
        embed = self._encode(obs)

        # 2. img_step (action forced to 0 on first step)
        if self.is_first:
            self.is_first = False
            self._img_step(0.0)
        else:
            self._img_step(action)

        # 3. Compute posterior stoch from (deter, embed)
        self._compute_posterior(embed)

        self.step_count += 1
        return self.deter.copy()


# ─── PCA projection and centroid matching ──────────────────────────────
def load_pca(pca_path: Path) -> dict:
    with open(pca_path) as f:
        data = json.load(f)
    return {
        "components": np.array(data["pca_components"], dtype=np.float64),  # [2, 512]
        "mean": np.array(data["pca_mean"], dtype=np.float64),  # [512]
        "centroids_2d": np.array(data["centroids_2d"], dtype=np.float64),  # [26, 2]
        "centroid_counts": data["centroid_counts"],  # list of ints
    }


def project_pca(deter: np.ndarray, pca: dict) -> np.ndarray:
    """Project deter[512] to 2D using PCA."""
    centered = deter.astype(np.float64) - pca["mean"]
    return pca["components"] @ centered  # [2]


def nearest_centroid(point_2d: np.ndarray, pca: dict) -> tuple[int, float]:
    """Find nearest centroid, return (count, distance)."""
    dists = np.linalg.norm(pca["centroids_2d"] - point_2d, axis=1)
    idx = np.argmin(dists)
    return pca["centroid_counts"][idx], dists[idx]


# ─── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading weights...")
    weights = load_weights(BIN_PATH, MANIFEST_PATH)
    print(f"  Loaded {len(weights)} tensors")

    print("Loading PCA data...")
    pca = load_pca(PCA_PATH)
    print(f"  Components shape: {pca['components'].shape}")
    print(f"  Centroids: {len(pca['centroid_counts'])} counts")

    print("Loading parity test data...")
    with open(PARITY_PATH) as f:
        parity = json.load(f)
    n_steps = parity["n_steps"]
    obs_list = parity["obs"]
    ref_deter = parity["deter"]  # reference deter from TS
    ref_counts = parity["counts"]
    print(f"  {n_steps} steps, obs_size={parity['obs_size']}")

    print("\nCreating model and running forward pass...")
    model = DreamerWorldModel(weights)

    # Print reference deter[0] for comparison
    ref_d0 = np.array(ref_deter[0], dtype=np.float32)
    print(f"  Reference deter[0] norm: {np.linalg.norm(ref_d0):.6f}")
    print(f"  Reference deter[0][:3]: [{ref_d0[0]:.6f}, {ref_d0[1]:.6f}, {ref_d0[2]:.6f}]")

    print()
    print(f"{'Step':>4}  {'PCA_X':>9}  {'PCA_Y':>9}  {'Nearest':>7}  {'Dist':>7}  {'Deter_Norm':>10}  {'RefCount':>8}  {'Match':>5}")
    print("-" * 80)

    for step in range(n_steps):
        obs = np.array(obs_list[step], dtype=np.float32)

        # Action = 0.0 always (matching browser)
        deter = model.obs_step(obs, action=0.0)

        # PCA projection
        pca_2d = project_pca(deter, pca)
        pca_x, pca_y = pca_2d[0], pca_2d[1]

        # Nearest centroid
        nearest_count, nearest_dist = nearest_centroid(pca_2d, pca)

        # Deter norm
        deter_norm = np.linalg.norm(deter)

        # Compare with reference
        ref_d = np.array(ref_deter[step], dtype=np.float32)
        ref_norm = np.linalg.norm(ref_d)
        diff_norm = np.linalg.norm(deter - ref_d)
        match = "OK" if diff_norm < 0.01 else f"DIFF={diff_norm:.4f}"

        print(
            f"{step:4d}  {pca_x:9.4f}  {pca_y:9.4f}  "
            f"{nearest_count:7d}  {nearest_dist:7.4f}  "
            f"{deter_norm:10.6f}  {ref_counts[step]:8d}  {match}"
        )

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    all_diffs = []
    for step in range(n_steps):
        obs = np.array(obs_list[step], dtype=np.float32)
        ref_d = np.array(ref_deter[step], dtype=np.float32)
        # We already ran the model, so just compute diffs from stored refs
        # (We need to re-run to get our deter at each step - let's use stored)

    # Re-run to collect all deters
    model.reset()
    py_deters = []
    py_nearest = []
    for step in range(n_steps):
        obs = np.array(obs_list[step], dtype=np.float32)
        deter = model.obs_step(obs, action=0.0)
        py_deters.append(deter.copy())
        pca_2d = project_pca(deter, pca)
        nc, _ = nearest_centroid(pca_2d, pca)
        py_nearest.append(nc)

    diffs = [np.linalg.norm(py_deters[i] - np.array(ref_deter[i], dtype=np.float32)) for i in range(n_steps)]
    print(f"  Max deter diff from reference: {max(diffs):.8f}")
    print(f"  Mean deter diff from reference: {np.mean(diffs):.8f}")
    print(f"  Nearest counts (unique): {sorted(set(py_nearest))}")
    print(f"  Count distribution: ", end="")
    from collections import Counter
    ctr = Counter(py_nearest)
    for c in sorted(ctr.keys()):
        print(f"  count={c}: {ctr[c]} steps", end="")
    print()


if __name__ == "__main__":
    main()
