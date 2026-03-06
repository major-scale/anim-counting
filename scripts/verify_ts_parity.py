#!/usr/bin/env python3
"""
Verify the TypeScript DreamerV3 inference matches Python.

Reimplements the TS forward pass in numpy and compares against
the parity_test.json ground truth from the Python model.

This catches bugs in: matmul order, LayerNorm, GRU gates, prior/posterior.
"""

import json, struct
import numpy as np

# Load binary weights
BIN_PATH = "/workspace/projects/jamstack-v1/packages/signal-app/public/models/dreamer_weights.bin"
MANIFEST_PATH = "/workspace/projects/jamstack-v1/packages/signal-app/public/models/dreamer_manifest.json"
PARITY_PATH = "/workspace/projects/jamstack-v1/packages/signal-app/public/models/parity_test.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

with open(BIN_PATH, "rb") as f:
    buf = f.read()

def load_tensor(name):
    entry = manifest["tensors"][name]
    offset = entry["offset"]
    length = entry["length"]
    arr = np.frombuffer(buf, dtype=np.float32, count=length, offset=offset)
    return arr.reshape(entry["shape"]) if len(entry["shape"]) > 1 else arr.copy()

# Load weights
enc_w = [load_tensor(f"enc_linear{i}_w") for i in range(3)]
enc_nw = [load_tensor(f"enc_norm{i}_w") for i in range(3)]
enc_nb = [load_tensor(f"enc_norm{i}_b") for i in range(3)]
img_in_w = load_tensor("img_in_w")
img_in_nw = load_tensor("img_in_norm_w")
img_in_nb = load_tensor("img_in_norm_b")
gru_w = load_tensor("gru_w")
gru_nw = load_tensor("gru_norm_w")
gru_nb = load_tensor("gru_norm_b")
img_out_w = load_tensor("img_out_w")
img_out_nw = load_tensor("img_out_norm_w")
img_out_nb = load_tensor("img_out_norm_b")
imgs_stat_w = load_tensor("imgs_stat_w")
imgs_stat_b = load_tensor("imgs_stat_b")
obs_out_w = load_tensor("obs_out_w")
obs_out_nw = load_tensor("obs_out_norm_w")
obs_out_nb = load_tensor("obs_out_norm_b")
obs_stat_w = load_tensor("obs_stat_w")
obs_stat_b = load_tensor("obs_stat_b")
deter_init_w = load_tensor("deter_init_w")  # [1, 512]

print(f"enc_w[0]: {enc_w[0].shape}")  # [512, 80]
print(f"img_in_w: {img_in_w.shape}")   # [512, 1025]
print(f"gru_w: {gru_w.shape}")         # [1536, 1024]
print(f"obs_out_w: {obs_out_w.shape}") # [512, 1024]

# Math primitives matching TS
LN_EPS = 1e-3

def layer_norm(x, w, b):
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b

def silu(x):
    return x / (1.0 + np.exp(-x))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def argmax_onehot(logits, n_cats=32, n_classes=32):
    """logits[1024] → onehot[1024]"""
    logits = logits.reshape(n_cats, n_classes)
    out = np.zeros_like(logits)
    for c in range(n_cats):
        idx = np.argmax(logits[c])
        out[c, idx] = 1.0
    return out.flatten()

# Forward pass functions
def encode(obs):
    x = obs  # [80]
    for i in range(3):
        x = enc_w[i] @ x  # [512, in] @ [in] → [512]
        x = layer_norm(x, enc_nw[i], enc_nb[i])
        x = silu(x)
    return x  # [512]

def img_step(stoch, action, deter):
    # concat stoch[1024] + action[1] → [1025]
    cat = np.concatenate([stoch, [action]])
    # img_in: Linear(1025→512) + LN + SiLU
    x = img_in_w @ cat
    x = layer_norm(x, img_in_nw, img_in_nb)
    x = silu(x)
    # GRU: concat(hidden[512], deter[512]) → [1024]
    gru_in = np.concatenate([x, deter])
    # Linear(1024→1536) + LN
    raw = gru_w @ gru_in
    ln = layer_norm(raw, gru_nw, gru_nb)
    # Split 3×512
    reset = sigmoid(ln[:512])
    cand = np.tanh(ln[512:1024] * reset)
    update = sigmoid(ln[1024:] - 1.0)
    new_deter = update * cand + (1.0 - update) * deter
    return new_deter

def compute_prior(deter):
    x = img_out_w @ deter
    x = layer_norm(x, img_out_nw, img_out_nb)
    x = silu(x)
    logits = imgs_stat_w @ x + imgs_stat_b
    return argmax_onehot(logits)

def compute_posterior(deter, embed):
    inp = np.concatenate([deter, embed])
    x = obs_out_w @ inp
    x = layer_norm(x, obs_out_nw, obs_out_nb)
    x = silu(x)
    logits = obs_stat_w @ x + obs_stat_b
    return argmax_onehot(logits)

# Load parity test
with open(PARITY_PATH) as f:
    parity = json.load(f)

obs_seq = [np.array(o, dtype=np.float32) for o in parity["obs"]]
deter_expected = [np.array(d, dtype=np.float32) for d in parity["deter"]]
stoch_expected = [np.array(s, dtype=np.float32) for s in parity["stoch"]]
actions = [np.array(a, dtype=np.float32) for a in parity["actions"]]

print(f"\nParity test: {len(obs_seq)} steps")
print(f"Expected deter[0][:5] = {deter_expected[0][:5]}")

# Run our forward pass
# Initialize
deter = np.tanh(deter_init_w[0])  # [512]
stoch = np.zeros(1024, dtype=np.float32)
stoch = compute_prior(deter)

print(f"\nInitial deter[:5] = {deter[:5]}")
print(f"Initial stoch nonzero = {(stoch > 0).sum()}")

# Step 0: is_first → img_step with zero action + posterior
embed = encode(obs_seq[0])
print(f"\nStep 0: obs[:5] = {obs_seq[0][:5]}")
print(f"  embed[:5] = {embed[:5]}")

deter = img_step(stoch, 0.0, deter)
print(f"  deter after img_step[:5] = {deter[:5]}")
print(f"  expected deter[0][:5]    = {deter_expected[0][:5]}")
err0 = np.abs(deter - deter_expected[0]).max()
print(f"  max abs error (deter) = {err0:.6f}")

stoch = compute_posterior(deter, embed)
print(f"  stoch nonzero = {(stoch > 0).sum()}")
print(f"  expected stoch nonzero = {(np.array(stoch_expected[0]) > 0).sum()}")
stoch_err = np.abs(stoch - np.array(stoch_expected[0])).max()
print(f"  max abs error (stoch) = {stoch_err:.6f}")

# Steps 1-9
for t in range(1, min(10, len(obs_seq))):
    action = actions[t-1][0]  # previous action
    embed = encode(obs_seq[t])
    deter = img_step(stoch, action, deter)
    stoch = compute_posterior(deter, embed)

    err = np.abs(deter - deter_expected[t]).max()
    stoch_match = np.array_equal(stoch, np.array(stoch_expected[t]))
    print(f"  Step {t}: deter_err={err:.6f}, stoch_match={stoch_match}, "
          f"deter[:3]=[{deter[0]:.4f},{deter[1]:.4f},{deter[2]:.4f}]")

# Check: what if we use action=0 for all steps (like the browser)?
print("\n=== Rerun with action=0.0 everywhere (browser mode) ===")
deter = np.tanh(deter_init_w[0])
stoch = compute_prior(deter)

embed = encode(obs_seq[0])
deter = img_step(stoch, 0.0, deter)
stoch = compute_posterior(deter, embed)
print(f"Step 0: deter[:3]=[{deter[0]:.4f},{deter[1]:.4f},{deter[2]:.4f}] norm={np.linalg.norm(deter):.4f}")

for t in range(1, min(20, len(obs_seq))):
    embed = encode(obs_seq[t])
    deter = img_step(stoch, 0.0, deter)  # action=0 always
    stoch = compute_posterior(deter, embed)
    if t < 10 or t % 5 == 0:
        print(f"Step {t}: deter[:3]=[{deter[0]:.4f},{deter[1]:.4f},{deter[2]:.4f}] "
              f"norm={np.linalg.norm(deter):.4f} stoch_nz={int((stoch>0).sum())}")
