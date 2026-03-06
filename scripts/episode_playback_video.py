#!/usr/bin/env python3
"""
Episode Playback Video — Real-Time Counting Visualization Demo.

Two-phase design:
  1. Collect: Run episodes with DreamerV3 driving + LSTM/MLP observing → NPZ
  2. Render: Load NPZ, compute PCA manifold backdrops, render frames → MP4

Layout (1920×1080):
  +-----------------------------------+---------------------------+---------------------------+
  |                                   |   DreamerV3 (blue)        |   LSTM (green)            |
  |   Environment View                |   PCA manifold +          |   PCA manifold +          |
  |   (bot, blobs, grid)              |   moving h_t dot          |   moving h_t dot          |
  |                                   +---------------------------+---------------------------+
  |   ~720 × 900 px                   |   MLP (red)               |   MLP-nocount (orange)    |
  |                                   |   PCA manifold +          |   PCA manifold +          |
  |                                   |   moving h_t dot          |   moving h_t dot          |
  +-----------------------------------+---------------------------+---------------------------+
  |  Step: 1240/5500   GT Count: 14   |  DV3: 14  LSTM: 13  MLP: 14  MLP-nc: 12              |
  +-----------------------------------------------------------------------------------+

Usage:
    python episode_playback_video.py --phase all
    python episode_playback_video.py --phase collect
    python episode_playback_video.py --phase render --subsample 50 --fps 10   # quick preview
    python episode_playback_video.py --phase render --subsample 8             # full quality
"""

import os, sys, time, pathlib, argparse, subprocess, struct
import numpy as np
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_SIZE = 82
HIDDEN_DIM = 256
GRID_FILLED_RAW_IDX = 81
MAX_BLOB_COUNT = 25

# World geometry (from counting_env_pure.py)
WORLD_WIDTH = 1400
WORLD_HEIGHT = 1000
GRID_COLS = 5
GRID_ROWS = 5
GRID_CELL_SIZE = 50
GRID_RIGHT_MARGIN = 60

# Paths
DEFAULT_CHECKPOINT = "/workspace/bridge/artifacts/checkpoints/train_25blobs_v3"
LSTM_CHECKPOINT = "/workspace/bridge/artifacts/battery/lstm_mlp/lstm_predictor.pt"
MLP_CHECKPOINT = "/workspace/bridge/artifacts/battery/lstm_mlp/mlp_predictor.pt"
MLP_NC_CHECKPOINT = "/workspace/bridge/artifacts/battery/lstm_mlp/mlp_nocount_predictor.pt"
LSTM_EVAL_NPZ = "/workspace/bridge/artifacts/battery/lstm_mlp/lstm/lstm_eval.npz"
MLP_EVAL_NPZ = "/workspace/bridge/artifacts/battery/lstm_mlp/mlp/mlp_eval.npz"
MLP_NC_EVAL_NPZ = "/workspace/bridge/artifacts/battery/lstm_mlp/mlp_nocount/mlp_nocount_eval.npz"
OUTPUT_DIR = "/workspace/bridge/artifacts/video"

# Colors
COLOR_BG = '#1a1a2e'
COLOR_FIELD = '#16213e'
COLOR_GRID_BG = '#0f3460'
COLOR_GRID_LINE = '#53586680'
COLOR_BLOB_FIELD = '#4CAF50'
COLOR_BLOB_GRID = '#2196F3'
COLOR_BOT = '#FF9800'
COLOR_DV3 = '#42A5F5'      # blue
COLOR_LSTM = '#66BB6A'      # green
COLOR_MLP = '#EF5350'       # red
COLOR_MLP_NC = '#FFA726'    # orange
COLOR_TEXT = '#E0E0E0'
COLOR_GAUGE_BG = '#2a2a3e'
COLOR_GAUGE_FILL = '#4CAF50'

MODEL_COLORS = {
    'dreamer': COLOR_DV3,
    'lstm': COLOR_LSTM,
    'mlp': COLOR_MLP,
    'mlp_nocount': COLOR_MLP_NC,
}
MODEL_LABELS = {
    'dreamer': 'DreamerV3',
    'lstm': 'LSTM',
    'mlp': 'MLP',
    'mlp_nocount': 'MLP-nocount',
}

# Trail length (frames)
TRAIL_LEN = 30


# ---------------------------------------------------------------------------
# Models (copied from lstm_mlp_experiment.py to keep self-contained)
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, obs_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        encoded = torch.relu(self.encoder(x))
        lstm_out, hidden = self.lstm(encoded, hidden)
        pred = self.decoder(lstm_out)
        return pred, hidden, lstm_out

    def forward_step(self, x, hidden=None):
        encoded = torch.relu(self.encoder(x)).unsqueeze(1)
        lstm_out, hidden = self.lstm(encoded, hidden)
        h_t = lstm_out.squeeze(1)
        pred = self.decoder(h_t)
        return pred, hidden, h_t


class MLPPredictor(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, obs_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h1 = torch.relu(self.layer1(x))
        h2 = torch.relu(self.layer2(h1))
        pred = self.decoder(h2)
        return pred, h2


# ---------------------------------------------------------------------------
# DreamerV3 loading (from lstm_mlp_experiment.py)
# ---------------------------------------------------------------------------

def _setup_dreamer_imports():
    dreamer_dir = os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")
    if dreamer_dir not in sys.path:
        sys.path.insert(0, dreamer_dir)
    scripts_dir = "/workspace/bridge/scripts"
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from ruamel.yaml import YAML as _YAML
    class _YAMLCompat:
        @staticmethod
        def safe_load(text):
            y = _YAML(typ='safe', pure=True)
            return y.load(text)
    return _YAMLCompat(), dreamer_dir


def _load_dreamer_config(checkpoint_dir):
    yaml, configs_dir = _setup_dreamer_imports()
    import tools
    configs = yaml.safe_load(pathlib.Path(configs_dir, "configs.yaml").read_text())
    defaults = {}
    for name in ["defaults", "counting_continuous"]:
        for k, v in configs[name].items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v
    defaults["logdir"] = str(checkpoint_dir)
    defaults["device"] = "cpu"
    defaults["compile"] = False
    defaults["steps"] = 100000
    defaults["action_repeat"] = 1
    defaults["eval_every"] = 10000
    defaults["time_limit"] = 6000
    defaults["log_every"] = 1000
    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value),
                            default=tools.args_type(value)(value))
    config = parser.parse_args([])
    config.num_actions = 1
    return config


def _detect_checkpoint_obs_size(checkpoint_dir):
    ckpt = torch.load(pathlib.Path(checkpoint_dir) / "latest.pt", map_location="cpu")
    for k, v in ckpt["agent_state_dict"].items():
        if "Encoder_linear0.weight" in k:
            return v.shape[1]
    return OBS_SIZE


def _create_dreamer_agent(config):
    import gym, gym.spaces
    from dreamer import make_dataset, Dreamer
    import tools

    ckpt_obs_size = _detect_checkpoint_obs_size(config.logdir)
    if ckpt_obs_size != OBS_SIZE:
        print(f"  Note: checkpoint OBS_SIZE={ckpt_obs_size}, env emits {OBS_SIZE}. Truncating.")

    obs_space = gym.spaces.Dict({
        "vector": gym.spaces.Box(0.0, 25.0, (ckpt_obs_size,), dtype=np.float32),
        "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
    })
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    logdir = pathlib.Path(config.logdir)
    logger = tools.Logger(logdir, 0)
    train_eps = tools.load_episodes(logdir / "train_eps", limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(obs_space, act_space, config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    agent.eval()
    agent._ckpt_obs_size = ckpt_obs_size
    return agent


# ---------------------------------------------------------------------------
# Grid slot positions (precomputed, matching _create_grid() in counting_env_pure.py)
# ---------------------------------------------------------------------------

def _compute_grid_slot_positions():
    """Return (25, 2) array of grid slot center positions in world coords."""
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - GRID_COLS * GRID_CELL_SIZE
    grid_height = GRID_ROWS * GRID_CELL_SIZE
    grid_top = (WORLD_HEIGHT - grid_height) / 2
    slots = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = grid_left + col * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
            y = grid_top + (GRID_ROWS - 1 - row) * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
            slots.append((x, y))
    return np.array(slots, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_episode_data(output_path, n_blobs=25, n_episodes=5, seed=42):
    """
    Run episodes with DreamerV3 driving. At each step, extract:
    - Environment state (bot pos, blob positions, grid occupancy)
    - DreamerV3 h_t (RSSM deter)
    - LSTM h_t
    - MLP h_t
    - MLP-nocount h_t (obs[80:82] zeroed)
    Save all to compressed NPZ.
    """
    from envs.counting import CountingWorld

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_MAX_STEPS'] = '10000'
    os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
    os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)

    print("Loading DreamerV3 agent...")
    config = _load_dreamer_config(DEFAULT_CHECKPOINT)
    agent = _create_dreamer_agent(config)
    device = config.device
    ckpt_obs = agent._ckpt_obs_size

    print("Loading LSTM predictor...")
    lstm = LSTMPredictor()
    lstm.load_state_dict(torch.load(LSTM_CHECKPOINT, map_location="cpu"))
    lstm.eval()

    print("Loading MLP predictor...")
    mlp = MLPPredictor()
    mlp.load_state_dict(torch.load(MLP_CHECKPOINT, map_location="cpu"))
    mlp.eval()

    print("Loading MLP-nocount predictor...", flush=True)
    mlp_nc = MLPPredictor()
    mlp_nc.load_state_dict(torch.load(MLP_NC_CHECKPOINT, map_location="cpu"))
    mlp_nc.eval()
    print("All models loaded.", flush=True)

    # Pre-allocate lists
    all_bot_x = []
    all_bot_y = []
    all_blob_x = []
    all_blob_y = []
    all_blob_grid_slot = []
    all_grid_filled = []
    all_episode_ids = []
    all_dreamer_h_t = []
    all_lstm_h_t = []
    all_mlp_h_t = []
    all_mlp_nc_h_t = []

    print("Creating environment...", flush=True)
    env = CountingWorld("counting_world", seed=seed)
    grid_slots = _compute_grid_slot_positions()
    print(f"Environment ready. Running {n_episodes} episodes with {n_blobs} blobs...", flush=True)
    t0 = time.time()

    for ep in range(n_episodes):
        print(f"  Episode {ep+1}/{n_episodes}: resetting...", end="", flush=True)
        obs_raw = env.reset()
        print(f" started.", flush=True)
        dreamer_state = None
        lstm_hidden = None
        done = False
        step = 0

        while not done:
            vec = obs_raw["vector"]
            state_obj = env._env._state

            # --- Ground truth from env state ---
            grid_filled = state_obj.grid.filled_count if state_obj else int(vec[GRID_FILLED_RAW_IDX])

            # Bot position (denormalized)
            bot_x = vec[0] * WORLD_WIDTH
            bot_y = vec[1] * WORLD_HEIGHT

            # Blob positions (denormalized) and grid slot assignments
            blob_x = np.zeros(MAX_BLOB_COUNT, dtype=np.float32)
            blob_y = np.zeros(MAX_BLOB_COUNT, dtype=np.float32)
            blob_gs = np.full(MAX_BLOB_COUNT, -1, dtype=np.int32)
            n_actual = len(state_obj.blobs) if state_obj else 0
            for i in range(min(n_actual, MAX_BLOB_COUNT)):
                blob_x[i] = vec[3 + i * 2] * WORLD_WIDTH
                blob_y[i] = vec[3 + i * 2 + 1] * WORLD_HEIGHT
                if state_obj and state_obj.blobs[i].grid_slot is not None:
                    blob_gs[i] = state_obj.blobs[i].grid_slot

            all_bot_x.append(bot_x)
            all_bot_y.append(bot_y)
            all_blob_x.append(blob_x)
            all_blob_y.append(blob_y)
            all_blob_grid_slot.append(blob_gs)
            all_grid_filled.append(grid_filled)
            all_episode_ids.append(ep)

            # --- DreamerV3 forward (policy + h_t) ---
            obs_dict = {
                "vector": torch.tensor(vec[:ckpt_obs], dtype=torch.float32).unsqueeze(0).to(device),
                "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]],
                                         dtype=torch.float32).to(device),
                "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]],
                                        dtype=torch.float32).to(device),
                "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]],
                                            dtype=torch.float32).to(device),
            }
            reset = np.array([obs_raw["is_first"]])
            with torch.no_grad():
                policy_output, dreamer_state = agent(obs_dict, reset, dreamer_state, training=False)
            latent, _ = dreamer_state
            dreamer_h = latent["deter"].cpu().numpy().squeeze(0)
            all_dreamer_h_t.append(dreamer_h)

            # --- LSTM forward ---
            obs_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, lstm_hidden, lstm_h = lstm.forward_step(obs_tensor, lstm_hidden)
                all_lstm_h_t.append(lstm_h.squeeze(0).numpy())

            # --- MLP forward ---
            with torch.no_grad():
                _, mlp_h = mlp(obs_tensor)
                all_mlp_h_t.append(mlp_h.squeeze(0).numpy())

            # --- MLP-nocount forward (zero out count scalars) ---
            obs_nc = vec.copy()
            obs_nc[80] = 0.0
            obs_nc[81] = 0.0
            obs_nc_tensor = torch.tensor(obs_nc, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, mlp_nc_h = mlp_nc(obs_nc_tensor)
                all_mlp_nc_h_t.append(mlp_nc_h.squeeze(0).numpy())

            # Step env
            obs_raw, reward, done, info = env.step(policy_output["action"].cpu().numpy().flatten())
            step += 1
            if step % 500 == 0:
                print(f"    step {step}, grid_filled={grid_filled}", flush=True)

        elapsed = time.time() - t0
        print(f"  Episode {ep+1}/{n_episodes} done: {step} steps, "
              f"grid_filled={grid_filled}, {elapsed:.0f}s", flush=True)

    env.close()
    elapsed = time.time() - t0
    total_steps = len(all_bot_x)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        bot_x=np.array(all_bot_x, dtype=np.float32),
        bot_y=np.array(all_bot_y, dtype=np.float32),
        blob_x=np.stack(all_blob_x),
        blob_y=np.stack(all_blob_y),
        blob_grid_slot=np.stack(all_blob_grid_slot),
        grid_filled=np.array(all_grid_filled, dtype=np.int32),
        episode_ids=np.array(all_episode_ids, dtype=np.int32),
        dreamer_h_t=np.stack(all_dreamer_h_t),
        lstm_h_t=np.stack(all_lstm_h_t),
        mlp_h_t=np.stack(all_mlp_h_t),
        mlp_nc_h_t=np.stack(all_mlp_nc_h_t),
        grid_slot_positions=grid_slots,
        n_blobs=np.int32(n_blobs),
    )
    print(f"\nCollected {total_steps} steps across {n_episodes} episodes in {elapsed:.0f}s")
    print(f"Saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Manifold Backdrop
# ---------------------------------------------------------------------------

class ManifoldBackdrop:
    """PCA projection of model hidden states, with centroid backbone."""

    def __init__(self, h_t, counts, label=""):
        from sklearn.decomposition import PCA
        self.label = label
        # Compute centroids per count
        max_count = int(counts.max())
        centroids, valid_counts = [], []
        for c in range(max_count + 1):
            mask = counts == c
            if mask.sum() > 0:
                centroids.append(h_t[mask].mean(axis=0))
                valid_counts.append(c)
        self.centroids = np.stack(centroids)
        self.valid_counts = np.array(valid_counts)

        # Fit PCA on centroids
        self.pca = PCA(n_components=2)
        self.centroid_2d = self.pca.fit_transform(self.centroids)

        # Precompute centroid norms for nearest-count lookup (in HD space)
        self._centroid_norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)

    def project(self, h_t):
        """Project h_t (D,) or (N, D) to 2D."""
        if h_t.ndim == 1:
            return self.pca.transform(h_t.reshape(1, -1))[0]
        return self.pca.transform(h_t)

    def nearest_count(self, h_t):
        """Return predicted count (nearest centroid in HD space)."""
        if h_t.ndim == 1:
            h_t = h_t.reshape(1, -1)
        dists = np.linalg.norm(self.centroids - h_t, axis=1)
        return self.valid_counts[np.argmin(dists)]


# ---------------------------------------------------------------------------
# Figure Setup
# ---------------------------------------------------------------------------

def setup_figure(dpi=150):
    """Create the 1920×1080 figure with all axes."""
    fig = plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi, facecolor=COLOR_BG)

    # Layout: left panel (env view), right 2×2 (manifolds), bottom gauge
    # Using gridspec for precise control
    gs = fig.add_gridspec(
        3, 4,
        left=0.02, right=0.98, top=0.97, bottom=0.02,
        hspace=0.08, wspace=0.06,
        height_ratios=[1, 1, 0.12],
        width_ratios=[1.4, 1, 1, 0.001],  # tiny rightmost col for padding
    )

    axes = {}
    # Environment view spans left 2 rows
    axes['env'] = fig.add_subplot(gs[0:2, 0])
    # Manifold panels (2×2 on the right)
    axes['dreamer'] = fig.add_subplot(gs[0, 1])
    axes['lstm'] = fig.add_subplot(gs[0, 2])
    axes['mlp'] = fig.add_subplot(gs[1, 1])
    axes['mlp_nocount'] = fig.add_subplot(gs[1, 2])
    # Gauge bar spans bottom
    axes['gauge'] = fig.add_subplot(gs[2, :3])

    # Style all axes
    for name, ax in axes.items():
        ax.set_facecolor(COLOR_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=COLOR_TEXT, labelsize=6)

    return fig, axes


# ---------------------------------------------------------------------------
# Static Drawing
# ---------------------------------------------------------------------------

def draw_env_static(ax, n_blobs, grid_slots):
    """Draw the static environment elements: grid outline, field boundary."""
    ax.set_xlim(0, WORLD_WIDTH)
    ax.set_ylim(WORLD_HEIGHT, 0)  # inverted Y
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Field zone background
    field_right = WORLD_WIDTH - GRID_RIGHT_MARGIN - GRID_COLS * GRID_CELL_SIZE - 30
    ax.axvspan(0, field_right, color=COLOR_FIELD, alpha=0.3, zorder=0)

    # Grid zone
    grid_left = WORLD_WIDTH - GRID_RIGHT_MARGIN - GRID_COLS * GRID_CELL_SIZE
    grid_height = GRID_ROWS * GRID_CELL_SIZE
    grid_top = (WORLD_HEIGHT - grid_height) / 2

    # Grid cells
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = grid_left + col * GRID_CELL_SIZE
            y = grid_top + (GRID_ROWS - 1 - row) * GRID_CELL_SIZE
            rect = Rectangle((x, y), GRID_CELL_SIZE, GRID_CELL_SIZE,
                            linewidth=0.5, edgecolor=COLOR_GRID_LINE,
                            facecolor=COLOR_GRID_BG, alpha=0.4, zorder=1)
            ax.add_patch(rect)

    # Separator line
    ax.axvline(field_right, color='#ffffff20', linewidth=1, linestyle='--', zorder=1)

    # Title
    ax.set_title(f'Counting World  ({n_blobs} blobs)',
                 color=COLOR_TEXT, fontsize=10, fontweight='bold', pad=5)


def draw_manifold_static(ax, backdrop, name, color):
    """Draw the static manifold backbone (centroid dots + connecting lines)."""
    c2d = backdrop.centroid_2d
    counts = backdrop.valid_counts

    # Pad axes
    x_range = c2d[:, 0].max() - c2d[:, 0].min()
    y_range = c2d[:, 1].max() - c2d[:, 1].min()
    pad = max(x_range, y_range) * 0.15
    ax.set_xlim(c2d[:, 0].min() - pad, c2d[:, 0].max() + pad)
    ax.set_ylim(c2d[:, 1].min() - pad, c2d[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Connecting lines (thin gray)
    ax.plot(c2d[:, 0], c2d[:, 1], '-', color='#ffffff30', linewidth=0.8, zorder=1)

    # Centroid dots colored by count (viridis)
    norm = plt.Normalize(counts.min(), counts.max())
    cmap = plt.cm.viridis
    scatter = ax.scatter(c2d[:, 0], c2d[:, 1], c=counts, cmap=cmap, norm=norm,
                         s=20, edgecolors='#ffffff40', linewidths=0.5, zorder=2)

    # Count labels on select centroids
    label_every = max(1, len(counts) // 8)
    for i in range(0, len(counts), label_every):
        ax.annotate(str(counts[i]), (c2d[i, 0], c2d[i, 1]),
                    fontsize=5, color='#ffffff80',
                    ha='center', va='bottom', xytext=(0, 4),
                    textcoords='offset points')

    ax.set_title(MODEL_LABELS[name], color=color, fontsize=9, fontweight='bold', pad=3)


def draw_gauge_static(ax):
    """Draw the static gauge bar background."""
    ax.set_xlim(-0.5, MAX_BLOB_COUNT + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, MAX_BLOB_COUNT + 1, 5))
    ax.tick_params(axis='x', colors=COLOR_TEXT, labelsize=6)
    ax.tick_params(axis='y', left=False, labelleft=False)

    # Background bar
    ax.barh(0.5, MAX_BLOB_COUNT, height=0.6, left=0, color=COLOR_GAUGE_BG,
            edgecolor='#ffffff20', linewidth=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Dynamic Update (artist-based for speed)
# ---------------------------------------------------------------------------

def create_dynamic_artists(axes, model_names):
    """Create all mutable artists. Returns dict of artist references."""
    artists = {}

    # Env panel
    ax = axes['env']
    artists['field_blobs'] = ax.scatter([], [], s=40, c=COLOR_BLOB_FIELD,
                                         edgecolors='#ffffff40', linewidths=0.3, zorder=3)
    artists['grid_blobs'] = ax.scatter([], [], s=50, c=COLOR_BLOB_GRID,
                                        edgecolors='#ffffff60', linewidths=0.4, zorder=3)
    artists['bot'] = ax.scatter([], [], s=100, c=COLOR_BOT,
                                 edgecolors='white', linewidths=0.8, zorder=5, marker='o')
    artists['count_text'] = ax.text(
        70, 50, '', color=COLOR_TEXT, fontsize=11, fontweight='bold',
        fontfamily='monospace', zorder=10)

    # Manifold panels — current dot + trail
    for name in model_names:
        ax = axes[name]
        color = MODEL_COLORS[name]
        # Trail (fading dots)
        artists[f'{name}_trail'] = ax.scatter([], [], s=8, c=color, alpha=0.3, zorder=3)
        # Current h_t dot
        artists[f'{name}_dot'] = ax.scatter([], [], s=80, c=color,
                                              edgecolors='white', linewidths=1.2, zorder=5)
        # Prediction highlight ring on nearest centroid
        artists[f'{name}_pred_ring'] = ax.scatter([], [], s=120, facecolors='none',
                                                    edgecolors='#FFD700', linewidths=1.5, zorder=4)
        # Prediction label in title (will update title text)
        artists[f'{name}_title'] = ax.title

    # Gauge panel
    ax = axes['gauge']
    artists['gauge_fill'] = ax.barh(0.5, 0, height=0.6, left=0,
                                     color=COLOR_GAUGE_FILL, alpha=0.7, zorder=1)[0]
    # Model prediction ticks on gauge
    for name in model_names:
        color = MODEL_COLORS[name]
        artists[f'{name}_gauge_tick'] = ax.plot([0], [0.5], marker='v', color=color,
                                                  markersize=6, zorder=3, visible=False)[0]
    artists['gauge_text'] = ax.text(
        0.5, 0.08, '', transform=ax.transAxes, color=COLOR_TEXT,
        fontsize=7, fontweight='bold', fontfamily='monospace',
        ha='center', va='bottom', zorder=10)

    return artists


def update_frame(artists, axes, data, backdrops, frame_idx, trails,
                 model_names, total_frames):
    """Update all dynamic artists for a single frame."""
    i = frame_idx
    n_blobs = int(data['n_blobs'])

    # --- Environment panel ---
    bot_x, bot_y = data['bot_x'][i], data['bot_y'][i]
    bx, by = data['blob_x'][i], data['blob_y'][i]
    bgs = data['blob_grid_slot'][i]
    gt_count = int(data['grid_filled'][i])

    # Split blobs into field vs grid
    field_mask = np.zeros(MAX_BLOB_COUNT, dtype=bool)
    grid_mask = np.zeros(MAX_BLOB_COUNT, dtype=bool)
    for j in range(n_blobs):
        if bgs[j] >= 0:
            grid_mask[j] = True
        else:
            field_mask[j] = True

    if field_mask.any():
        artists['field_blobs'].set_offsets(np.column_stack([bx[field_mask], by[field_mask]]))
    else:
        artists['field_blobs'].set_offsets(np.empty((0, 2)))

    if grid_mask.any():
        artists['grid_blobs'].set_offsets(np.column_stack([bx[grid_mask], by[grid_mask]]))
    else:
        artists['grid_blobs'].set_offsets(np.empty((0, 2)))

    artists['bot'].set_offsets([[bot_x, bot_y]])
    artists['count_text'].set_text(f'Count: {gt_count} / {n_blobs}')

    # --- Manifold panels ---
    predictions = {}
    h_t_keys = {
        'dreamer': 'dreamer_h_t',
        'lstm': 'lstm_h_t',
        'mlp': 'mlp_h_t',
        'mlp_nocount': 'mlp_nc_h_t',
    }

    for name in model_names:
        bd = backdrops[name]
        h_t = data[h_t_keys[name]][i]
        pt = bd.project(h_t)
        pred = bd.nearest_count(h_t)
        predictions[name] = pred

        # Update trail
        trails[name].append(pt)
        if len(trails[name]) > TRAIL_LEN:
            trails[name] = trails[name][-TRAIL_LEN:]

        trail_pts = np.array(trails[name])
        # Trail with fading alpha
        n_trail = len(trail_pts)
        alphas = np.linspace(0.05, 0.4, n_trail)
        sizes = np.linspace(3, 8, n_trail)
        artists[f'{name}_trail'].set_offsets(trail_pts)
        artists[f'{name}_trail'].set_sizes(sizes)
        artists[f'{name}_trail'].set_alpha(None)  # per-point alpha via color array
        color_rgba = np.array(mcolors.to_rgba(MODEL_COLORS[name]))
        trail_colors = np.tile(color_rgba, (n_trail, 1))
        trail_colors[:, 3] = alphas
        artists[f'{name}_trail'].set_facecolors(trail_colors)

        # Current dot
        artists[f'{name}_dot'].set_offsets([pt])

        # Prediction ring on nearest centroid
        pred_idx = np.where(bd.valid_counts == pred)[0]
        if len(pred_idx) > 0:
            pred_pt = bd.centroid_2d[pred_idx[0]]
            artists[f'{name}_pred_ring'].set_offsets([pred_pt])
        else:
            artists[f'{name}_pred_ring'].set_offsets(np.empty((0, 2)))

        # Update title with prediction
        color = MODEL_COLORS[name]
        axes[name].set_title(f'{MODEL_LABELS[name]} — Pred: {pred}',
                             color=color, fontsize=9, fontweight='bold', pad=3)

    # --- Gauge ---
    artists['gauge_fill'].set_width(gt_count)

    for name in model_names:
        pred = predictions[name]
        artists[f'{name}_gauge_tick'].set_data([pred], [0.5])
        artists[f'{name}_gauge_tick'].set_visible(True)

    # Step counter + predictions text
    step_in_ep = i  # global frame index
    gauge_str = (f'Step: {step_in_ep}/{total_frames}    '
                 f'GT: {gt_count}    '
                 f'DV3: {predictions["dreamer"]}  '
                 f'LSTM: {predictions["lstm"]}  '
                 f'MLP: {predictions["mlp"]}  '
                 f'MLP-nc: {predictions["mlp_nocount"]}')
    artists['gauge_text'].set_text(gauge_str)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_video(episode_path, output_path, subsample=8, fps=30, dpi=150):
    """Load collected data, compute manifold backdrops, render to MP4."""
    print(f"Loading episode data from {episode_path}...")
    data = dict(np.load(episode_path, allow_pickle=True))
    total_raw = len(data['bot_x'])
    n_blobs = int(data['n_blobs'])
    print(f"  {total_raw} total steps, {n_blobs} blobs")

    # Frame indices after subsampling
    frame_indices = np.arange(0, total_raw, subsample)
    n_frames = len(frame_indices)
    duration = n_frames / fps
    print(f"  Subsampling every {subsample} → {n_frames} frames → {duration:.1f}s at {fps}fps")

    # --- Build manifold backdrops ---
    print("Building manifold backdrops...")
    model_names = ['dreamer', 'lstm', 'mlp', 'mlp_nocount']
    h_t_keys = {
        'dreamer': 'dreamer_h_t',
        'lstm': 'lstm_h_t',
        'mlp': 'mlp_h_t',
        'mlp_nocount': 'mlp_nc_h_t',
    }

    backdrops = {}
    for name in model_names:
        h_t = data[h_t_keys[name]]
        counts = data['grid_filled']
        backdrops[name] = ManifoldBackdrop(h_t, counts, label=name)
        print(f"  {MODEL_LABELS[name]}: {len(backdrops[name].valid_counts)} count centroids, "
              f"PCA explained var: {backdrops[name].pca.explained_variance_ratio_[:2].sum():.1%}")

    # --- Setup figure ---
    print("Setting up figure...")
    fig, axes = setup_figure(dpi=dpi)
    grid_slots = _compute_grid_slot_positions()

    # Draw static elements
    draw_env_static(axes['env'], n_blobs, grid_slots)
    for name in model_names:
        draw_manifold_static(axes[name], backdrops[name], name, MODEL_COLORS[name])
    draw_gauge_static(axes['gauge'])

    # Create dynamic artists
    artists = create_dynamic_artists(axes, model_names)
    trails = {name: [] for name in model_names}

    # --- Detect episode boundaries for trail resets ---
    episode_ids = data['episode_ids']

    # --- Render via ffmpeg pipe ---
    use_ffmpeg = True
    try:
        probe = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if probe.returncode != 0:
            use_ffmpeg = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        use_ffmpeg = False

    if use_ffmpeg:
        print(f"Rendering {n_frames} frames via ffmpeg → {output_path}")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{int(1920)}x{int(1080)}',
            '-pix_fmt', 'rgba',
            '-r', str(fps),
            '-i', '-',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            output_path,
        ]
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        print("ffmpeg not available, falling back to GIF via PIL")
        from PIL import Image
        gif_frames = []
        pipe = None

    t0 = time.time()
    prev_ep = -1

    for fi, raw_idx in enumerate(frame_indices):
        # Reset trails at episode boundaries
        cur_ep = int(episode_ids[raw_idx])
        if cur_ep != prev_ep:
            trails = {name: [] for name in model_names}
            prev_ep = cur_ep

        update_frame(artists, axes, data, backdrops, raw_idx, trails,
                     model_names, total_raw)

        # Render to buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_data = np.asarray(buf)

        if use_ffmpeg:
            pipe.stdin.write(frame_data.tobytes())
        else:
            # PIL fallback
            img = Image.fromarray(frame_data[:, :, :3])  # drop alpha
            gif_frames.append(img)

        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (n_frames - fi - 1) / rate if rate > 0 else 0
            print(f"  Frame {fi+1}/{n_frames} ({rate:.1f} fps, ETA {eta:.0f}s)", end="\r")

    print()

    if use_ffmpeg:
        pipe.stdin.close()
        stderr = pipe.stderr.read()
        pipe.wait()
        if pipe.returncode != 0:
            print(f"ffmpeg error: {stderr.decode()[-500:]}")
        else:
            # Check file size
            size = os.path.getsize(output_path)
            print(f"Saved: {output_path} ({size / 1e6:.1f} MB, {duration:.1f}s)")
    else:
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000/fps), loop=0, optimize=True)
        size = os.path.getsize(gif_path)
        print(f"Saved: {gif_path} ({size / 1e6:.1f} MB)")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Episode Playback Video Generator")
    parser.add_argument('--phase', choices=['collect', 'render', 'all'], default='all')
    parser.add_argument('--n-blobs', type=int, default=25)
    parser.add_argument('--n-episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample', type=int, default=8)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    npz_path = os.path.join(args.output_dir, 'episode_data.npz')
    mp4_path = os.path.join(args.output_dir, 'episode_playback.mp4')

    if args.phase in ('collect', 'all'):
        collect_episode_data(npz_path, n_blobs=args.n_blobs,
                            n_episodes=args.n_episodes, seed=args.seed)

    if args.phase in ('render', 'all'):
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found. Run --phase collect first.")
            sys.exit(1)
        render_video(npz_path, mp4_path, subsample=args.subsample,
                    fps=args.fps, dpi=args.dpi)


if __name__ == '__main__':
    main()
