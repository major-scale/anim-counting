# Brain Viz Analysis Tools

Analysis tools for DreamerV3 latent state visualization. No GPU needed — all operate on saved numpy arrays.

## Location in container

```
/workspace/bridge/artifacts/tools/
├── data/                          # Baseline data (frozen model, counts 0-8)
│   ├── trajectories.npz           # 68,803 timesteps × 512-dim h_t (130 MB)
│   ├── umap_embeddings.npz        # UMAP + PCA coords (1.9 MB)
│   ├── brain_explorer_data.json   # Pre-built app data (8 MB)
│   ├── analysis_results.json      # Transition analysis metrics
│   └── successor_battery_results.json  # Successor structure metrics
├── figures/                       # Output PNGs go here (auto-created)
├── fit_umap.py
├── export_for_app.py
├── brain_explorer.html
├── analyze_transitions.py
├── successor_battery.py
├── snapshot_trajectories.py
└── snapshot_zoomed.py
```

## Install deps

```bash
pip install numpy scipy scikit-learn matplotlib umap-learn
```

## The tools

### fit_umap.py
Fits 3 UMAP configs (default/local/global) + PCA on 512-dim h_t. Computes per-count cluster centers.
- **In:** `data/trajectories.npz`
- **Out:** `data/umap_embeddings.npz`, `figures/brain_map_*.png`

### export_for_app.py
Combines embeddings + trajectories into JSON for the interactive browser app.
- **In:** `data/trajectories.npz` + `data/umap_embeddings.npz`
- **Out:** `data/brain_explorer_data.json`

### brain_explorer.html
Interactive Plotly.js app. Episode playback, UMAP/PCA toggle, timeline, tagging.
- **In:** `data/brain_explorer_data.json` (loads via fetch, needs a local server)
- **Run:** `python -m http.server 8080` in the tools dir, open browser

### analyze_transitions.py
Transition dynamics, stability/drift per count, count=8 deep dive, dimensionality, anticipation.
- **In:** `data/trajectories.npz` + `data/umap_embeddings.npz`
- **Out:** `data/analysis_results.json`, `figures/transition_dynamics.png` etc.

### successor_battery.py
The big one. 5 independent analyses for successor structure:
1. RSA (Spearman/Pearson on pairwise count distances)
2. Geodesic distance linearity (k-NN graph shortest paths)
3. Homomorphism error (deviation from linear mapping)
4. Anisotropy-corrected cosine similarity (PCA whitening)
5. Persistent homology (topological analysis)
- **In:** `data/trajectories.npz` (h_t + true_count only)
- **Out:** `data/successor_battery_results.json`, `figures/successor_*.png`

### snapshot_trajectories.py / snapshot_zoomed.py
Grid PNGs of trajectory progression through UMAP/PCA space.
- **In:** `data/brain_explorer_data.json`
- **Out:** `figures/ep_*_snapshots.png`

## Using with new eval_dump data

When the Mac GPU server produces new tensor dumps via eval_dump jobs:

```python
import shutil
# Copy new data into the tools data dir
shutil.copy("/workspace/bridge/artifacts/tensors/my_job/h_t.npy", ...)
# Or modify the scripts to point at the new data dir
```

The tools expect `trajectories.npz` format. To convert from eval_dump .npy files:

```python
import numpy as np
data = {
    "h_t": np.load("h_t.npy"),
    "true_count": np.load("counts.npy"),
    "episode_id": np.load("episodes.npy"),
    "timestep": np.load("timesteps.npy"),
    "bot_x": np.load("bot_positions.npy")[:, 0],
    "bot_y": np.load("bot_positions.npy")[:, 1],
    "actor_prediction": np.load("actor_predictions.npy"),
    "reward": np.load("rewards.npy"),
    "total_blobs": np.load("total_blobs.npy"),
    "is_marking_event": np.diff(np.load("counts.npy"), prepend=0) > 0,
}
np.savez_compressed("data/trajectories.npz", **data)
```

## Baseline results (frozen model, trained 0-8)

- Ordinal structure present for counts 1-4 (cosine displacements 0.68-0.87)
- Structure degrades at higher counts (7-8 boundaries blur)
- RSA Spearman = 0.93, geodesic R² = 0.97
- Probe MAE = 0.060 on training data
- "Ordinal structure without operational successor" — the key finding
