#!/usr/bin/env python3
"""
LSTM/MLP Architecture Independence Test.

Tests whether the counting manifold requires a specific architecture (DreamerV3's RSSM)
or emerges from any recurrent predictor. Two minimal next-obs predictors:
  - LSTM (recurrent, 256-dim hidden): expected to succeed
  - MLP (feedforward, 256-dim bottleneck): expected to fail

Both are trained on obs_t -> obs_{t+1} with MSE loss, no RL, no reward.
DreamerV3's trained policy drives the env for data collection and evaluation.

Usage:
    python lstm_mlp_experiment.py --phase all
    python lstm_mlp_experiment.py --phase collect
    python lstm_mlp_experiment.py --phase train
    python lstm_mlp_experiment.py --phase eval
    python lstm_mlp_experiment.py --phase compare

    # Resume from existing data:
    python lstm_mlp_experiment.py --phase train  # skips collect if training_data.npz exists
    python lstm_mlp_experiment.py --phase eval   # skips train if checkpoints exist
"""

import os, sys, time, pathlib, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 8

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_SIZE = 82
HIDDEN_DIM = 256
SEQ_LEN = 64
GRID_FILLED_RAW_IDX = 81
COUNT_SCALAR_INDICES = [80, 81]  # grid_filled_norm, grid_filled_raw — direct count leakage
BLOB_COUNTS_TRAIN = [3, 5, 8, 10, 12, 15, 20, 25]
BLOB_COUNTS_EVAL = [3, 5, 8, 10, 12, 15, 20, 25]
EPISODES_TRAIN = 25   # per blob count
EPISODES_EVAL = 5     # per blob count

DEFAULT_CHECKPOINT = "/workspace/bridge/artifacts/checkpoints/train_25blobs_v3"
DEFAULT_OUTPUT = "/workspace/bridge/artifacts/battery/lstm_mlp"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """Minimal LSTM next-observation predictor."""

    def __init__(self, obs_dim=OBS_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, obs_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, obs_dim)
        hidden: (h_0, c_0) each (1, batch, hidden_dim) or None
        Returns: pred (batch, seq_len, obs_dim), hidden, h_t (batch, seq_len, hidden_dim)
        """
        encoded = torch.relu(self.encoder(x))
        lstm_out, hidden = self.lstm(encoded, hidden)
        pred = self.decoder(lstm_out)
        return pred, hidden, lstm_out

    def forward_step(self, x, hidden=None):
        """
        Single-step forward for evaluation.
        x: (batch, obs_dim)
        Returns: pred (batch, obs_dim), hidden, h_t (batch, hidden_dim)
        """
        encoded = torch.relu(self.encoder(x)).unsqueeze(1)  # (batch, 1, hidden)
        lstm_out, hidden = self.lstm(encoded, hidden)
        h_t = lstm_out.squeeze(1)  # (batch, hidden_dim)
        pred = self.decoder(h_t)
        return pred, hidden, h_t


class MLPPredictor(nn.Module):
    """Minimal feedforward next-observation predictor."""

    def __init__(self, obs_dim=OBS_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, obs_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: (batch, obs_dim)
        Returns: pred (batch, obs_dim), h_t (batch, hidden_dim)
        """
        h1 = torch.relu(self.layer1(x))
        h2 = torch.relu(self.layer2(h1))
        pred = self.decoder(h2)
        return pred, h2


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Chunked sequential data for LSTM training."""

    def __init__(self, obs_data, next_obs_data, episode_ids, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.chunks_obs = []
        self.chunks_next = []

        # Split by episode, chop into chunks
        unique_eps = np.unique(episode_ids)
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_obs = obs_data[mask]
            ep_next = next_obs_data[mask]
            n_steps = len(ep_obs)

            # Chop into non-overlapping chunks, drop short remainder
            for start in range(0, n_steps - seq_len + 1, seq_len):
                self.chunks_obs.append(ep_obs[start:start + seq_len])
                self.chunks_next.append(ep_next[start:start + seq_len])

        self.chunks_obs = np.array(self.chunks_obs, dtype=np.float32)
        self.chunks_next = np.array(self.chunks_next, dtype=np.float32)
        print(f"  SequenceDataset: {len(self.chunks_obs)} chunks of length {seq_len} "
              f"from {len(unique_eps)} episodes")

    def __len__(self):
        return len(self.chunks_obs)

    def __getitem__(self, idx):
        return (torch.tensor(self.chunks_obs[idx]),
                torch.tensor(self.chunks_next[idx]))


class TransitionDataset(Dataset):
    """Shuffled individual transitions for MLP training."""

    def __init__(self, obs_data, next_obs_data, mask_indices=None):
        self.obs = obs_data.astype(np.float32)
        self.next_obs = next_obs_data.astype(np.float32)
        if mask_indices:
            self.obs[:, mask_indices] = 0.0
            self.next_obs[:, mask_indices] = 0.0
            print(f"  TransitionDataset: {len(self.obs)} transitions (masked indices {mask_indices})")
        else:
            print(f"  TransitionDataset: {len(self.obs)} transitions")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (torch.tensor(self.obs[idx]),
                torch.tensor(self.next_obs[idx]))


# ---------------------------------------------------------------------------
# DreamerV3 Loading (gated behind functions that need it)
# ---------------------------------------------------------------------------

def _setup_dreamer_imports():
    """Add DreamerV3 to path. Only called when needed."""
    dreamer_dir = os.environ.get("DREAMER_DIR", "/workspace/dreamerv3-torch")
    sys.path.insert(0, dreamer_dir)
    sys.path.insert(0, "/workspace/bridge/scripts")

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
    """Detect OBS_SIZE from checkpoint encoder weight shape."""
    ckpt = torch.load(pathlib.Path(checkpoint_dir) / "latest.pt", map_location="cpu")
    for k, v in ckpt["agent_state_dict"].items():
        if "Encoder_linear0.weight" in k:
            return v.shape[1]  # (hidden, obs_dim)
    return OBS_SIZE  # fallback


def _create_dreamer_agent(config):
    import gym, gym.spaces
    from dreamer import make_dataset, Dreamer
    import tools

    # Use checkpoint's OBS_SIZE — may differ from current env (e.g. 80 vs 82)
    ckpt_obs_size = _detect_checkpoint_obs_size(config.logdir)
    if ckpt_obs_size != OBS_SIZE:
        print(f"  Note: checkpoint trained with OBS_SIZE={ckpt_obs_size}, "
              f"env now emits {OBS_SIZE}. Truncating obs for DreamerV3.")

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
    agent._ckpt_obs_size = ckpt_obs_size  # stash for callers
    return agent


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_training_data(checkpoint_dir, output_dir, seed=42):
    """Use trained DreamerV3 policy to collect (obs_t, obs_{t+1}) pairs."""
    from envs.counting import CountingWorld

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_MAX_STEPS'] = '10000'

    print("Loading DreamerV3 agent for data collection...")
    config = _load_dreamer_config(checkpoint_dir)
    agent = _create_dreamer_agent(config)
    device = config.device
    ckpt_obs = agent._ckpt_obs_size

    all_obs = []
    all_next_obs = []
    all_counts = []
    all_episode_ids = []
    ep_num = 0
    t0 = time.time()

    print(f"Collecting data: {EPISODES_TRAIN} episodes x {len(BLOB_COUNTS_TRAIN)} blob counts...")
    for n_blobs in BLOB_COUNTS_TRAIN:
        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        env = CountingWorld("counting_world", seed=seed)

        for ep in range(EPISODES_TRAIN):
            obs_raw = env.reset()
            state = None
            done = False
            ep_obs_list = []
            ep_count_list = []

            while not done:
                vec = obs_raw["vector"]
                marked_count = env._env._state.grid.filled_count if env._env._state else int(vec[GRID_FILLED_RAW_IDX])

                # Truncate to checkpoint's obs size for DreamerV3
                obs_dict = {
                    "vector": torch.tensor(vec[:ckpt_obs], dtype=torch.float32).unsqueeze(0).to(device),
                    "is_first": torch.tensor([[1.0 if obs_raw["is_first"] else 0.0]], dtype=torch.float32).to(device),
                    "is_last": torch.tensor([[1.0 if obs_raw["is_last"] else 0.0]], dtype=torch.float32).to(device),
                    "is_terminal": torch.tensor([[1.0 if obs_raw["is_terminal"] else 0.0]], dtype=torch.float32).to(device),
                }
                reset = np.array([obs_raw["is_first"]])

                with torch.no_grad():
                    policy_output, state = agent(obs_dict, reset, state, training=False)

                # Save full obs vector (82-dim) for LSTM/MLP training
                ep_obs_list.append(vec.copy())
                ep_count_list.append(marked_count)

                obs_raw, reward, done, info = env.step(policy_output["action"].cpu().numpy().flatten())

            # Build (obs_t, obs_{t+1}) pairs from this episode
            for i in range(len(ep_obs_list) - 1):
                all_obs.append(ep_obs_list[i])
                all_next_obs.append(ep_obs_list[i + 1])
                all_counts.append(ep_count_list[i])
                all_episode_ids.append(ep_num)

            ep_num += 1
            elapsed = time.time() - t0
            total_transitions = len(all_obs)
            print(f"  Episode {ep_num}/{EPISODES_TRAIN * len(BLOB_COUNTS_TRAIN)} "
                  f"(blobs={n_blobs}, transitions={total_transitions}): {elapsed:.0f}s",
                  end="\r")

        env.close()

    print()
    elapsed = time.time() - t0

    all_obs = np.array(all_obs, dtype=np.float32)
    all_next_obs = np.array(all_next_obs, dtype=np.float32)
    all_counts = np.array(all_counts, dtype=np.int32)
    all_episode_ids = np.array(all_episode_ids, dtype=np.int32)

    # Save
    save_path = pathlib.Path(output_dir) / "training_data.npz"
    np.savez_compressed(save_path,
                        obs=all_obs, next_obs=all_next_obs,
                        counts=all_counts, episode_ids=all_episode_ids)

    print(f"Collected {len(all_obs)} transitions in {elapsed:.0f}s")
    print(f"  Count coverage: {sorted(np.unique(all_counts).tolist())}")
    print(f"  Saved to: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(model, dataset, epochs, output_dir, lr=1e-3):
    """Train LSTM predictor on sequential chunks."""
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    best_loss = float('inf')
    t0 = time.time()

    print(f"Training LSTM: {epochs} epochs, {len(dataset)} chunks, batch_size=32")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for obs_chunk, next_chunk in loader:
            # obs_chunk: (batch, seq_len, obs_dim)
            # Initialize hidden to zeros for each chunk
            pred, _, _ = model(obs_chunk)
            loss = criterion(pred, next_chunk)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), pathlib.Path(output_dir) / "lstm_predictor.pt")

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f} "
                  f"best={best_loss:.6f} ({elapsed:.0f}s)")

    print(f"LSTM training done in {time.time() - t0:.0f}s, best loss={best_loss:.6f}")
    return losses


def train_mlp(model, dataset, epochs, output_dir, lr=1e-3, ckpt_name="mlp_predictor.pt"):
    """Train MLP predictor on shuffled transitions."""
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    best_loss = float('inf')
    t0 = time.time()

    print(f"Training MLP: {epochs} epochs, {len(dataset)} transitions, batch_size=256")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for obs_batch, next_batch in loader:
            pred, _ = model(obs_batch)
            loss = criterion(pred, next_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), pathlib.Path(output_dir) / ckpt_name)

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.6f} "
                  f"best={best_loss:.6f} ({elapsed:.0f}s)")

    print(f"MLP training done in {time.time() - t0:.0f}s, best loss={best_loss:.6f}")
    return losses


def plot_training_curves(loss_dict, output_dir):
    """Save training loss curves. loss_dict: {name: [losses]}."""
    colors = {"LSTM": "#2ca02c", "MLP": "#d62728", "MLP-nocount": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, losses in loss_dict.items():
        ax.plot(losses, label=name, linewidth=1, color=colors.get(name, None))
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('MSE Loss', fontsize=8)
    ax.set_title('Training Curves', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(pathlib.Path(output_dir) / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {output_dir}/training_curves.png")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def extract_representations(model, model_type, checkpoint_dir, output_dir, seed=42,
                            mask_indices=None):
    """
    DreamerV3 drives env, LSTM/MLP observes, extract representations at every step.
    Parallels eval_untrained_v2.py dual-eval pattern.
    mask_indices: list of obs indices to zero out before feeding to the model.
    """
    from envs.counting import CountingWorld

    os.environ['COUNTING_ACTION_SPACE'] = 'continuous'
    os.environ['COUNTING_BIDIRECTIONAL'] = 'true'
    os.environ['COUNTING_ARRANGEMENT'] = 'grid'
    os.environ['COUNTING_MAX_STEPS'] = '10000'

    print(f"Loading DreamerV3 agent for {model_type} evaluation...")
    config = _load_dreamer_config(checkpoint_dir)
    agent = _create_dreamer_agent(config)
    device = config.device
    ckpt_obs = agent._ckpt_obs_size

    model.eval()
    all_h_t = []
    all_counts = []
    all_episode_ids = []
    all_timesteps = []
    ep_num = 0
    t0 = time.time()

    print(f"Extracting {model_type} representations: {EPISODES_EVAL} episodes x "
          f"{len(BLOB_COUNTS_EVAL)} blob counts...")

    for n_blobs in BLOB_COUNTS_EVAL:
        os.environ['COUNTING_BLOB_MIN'] = str(n_blobs)
        os.environ['COUNTING_BLOB_MAX'] = str(n_blobs)
        env = CountingWorld("counting_world", seed=seed)

        for ep in range(EPISODES_EVAL):
            obs_raw = env.reset()
            dreamer_state = None
            lstm_hidden = None  # Reset LSTM hidden at episode boundary
            done = False
            step = 0

            while not done:
                vec = obs_raw["vector"]
                marked_count = (env._env._state.grid.filled_count
                                if env._env._state
                                else int(vec[GRID_FILLED_RAW_IDX]))

                # DreamerV3 forward for policy (truncate to checkpoint obs size)
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
                    policy_output, dreamer_state = agent(obs_dict, reset, dreamer_state,
                                                         training=False)

                # Extract representation from LSTM or MLP
                obs_vec = vec.copy()
                if mask_indices:
                    obs_vec[mask_indices] = 0.0
                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if model_type == "lstm":
                        _, lstm_hidden, h_t = model.forward_step(obs_tensor, lstm_hidden)
                        h_t = h_t.squeeze(0).numpy()
                    else:  # mlp
                        _, h_t = model(obs_tensor)
                        h_t = h_t.squeeze(0).numpy()

                all_h_t.append(h_t)
                all_counts.append(marked_count)
                all_episode_ids.append(ep_num)
                all_timesteps.append(step)

                obs_raw, reward, done, info = env.step(policy_output["action"].cpu().numpy().flatten())
                step += 1

            ep_num += 1
            elapsed = time.time() - t0
            print(f"  Episode {ep_num}/{EPISODES_EVAL * len(BLOB_COUNTS_EVAL)} "
                  f"(blobs={n_blobs}, steps={step}): {elapsed:.0f}s", end="\r")

        env.close()

    print()
    elapsed = time.time() - t0

    data = {
        "h_t": np.stack(all_h_t),
        "counts": np.array(all_counts, dtype=np.int32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "timesteps": np.array(all_timesteps, dtype=np.int32),
    }

    save_path = pathlib.Path(output_dir) / f"{model_type}_eval.npz"
    np.savez_compressed(save_path, **data)
    print(f"Extracted {len(data['counts'])} timesteps in {elapsed:.0f}s")
    print(f"  Saved to: {save_path}")
    return data


# ---------------------------------------------------------------------------
# Measurement Battery (copied from full_battery.py, zero DreamerV3 dependency)
# ---------------------------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def compute_centroids(h_t, counts):
    max_count = int(counts.max())
    centroids, valid_counts = [], []
    for c in range(max_count + 1):
        mask = counts == c
        if mask.sum() > 0:
            centroids.append(h_t[mask].mean(axis=0))
            valid_counts.append(c)
    return np.stack(centroids), np.array(valid_counts)


def run_battery(h_t, counts, centroids, valid_counts, output_dir, label):
    """Run full measurement battery. Returns metrics dict."""
    n = len(valid_counts)
    results = {"label": label, "n_counts": n, "n_timesteps": len(counts)}

    # --- GHE + arc-length R² ---
    best_ghe, best_geo = None, None
    for k in [4, 5, 6, 7, 8]:
        try:
            A = kneighbors_graph(centroids, n_neighbors=min(k, n-1), mode='distance')
            A = 0.5 * (A + A.T)
            geo = shortest_path(A, directed=False)
            if np.any(np.isinf(geo)):
                continue
            consec = [geo[i, i+1] for i in range(n-1)]
            ghe = np.std(consec) / np.mean(consec)
            if best_ghe is None or ghe < best_ghe:
                best_ghe = ghe
                best_geo = consec
        except Exception:
            continue

    results["ghe"] = float(best_ghe) if best_ghe else None

    if best_geo:
        arc_length = np.cumsum([0] + best_geo)
        coeffs = np.polyfit(np.arange(n), arc_length, 1)
        fit_line = np.polyval(coeffs, np.arange(n))
        ss_res = np.sum((arc_length - fit_line)**2)
        ss_tot = np.sum((arc_length - np.mean(arc_length))**2)
        results["arc_length_r2"] = float(1 - ss_res / ss_tot)
    else:
        results["arc_length_r2"] = None

    # --- Topology ---
    try:
        from ripser import ripser
        rips = ripser(centroids, maxdim=1)
        h0, h1 = rips['dgms'][0], rips['dgms'][1]
        results["beta_0"] = int(len(h0[h0[:, 1] == np.inf]))
        results["beta_1"] = int(len([1 for b, d in h1 if d == np.inf])) if len(h1) > 0 else 0
    except Exception:
        results["beta_0"] = None
        results["beta_1"] = None

    # --- RSA ---
    rdm_agent = squareform(pdist(centroids, metric='euclidean'))
    rdm_ideal = np.abs(valid_counts[:, None] - valid_counts[None, :]).astype(float)
    triu = np.triu_indices(n, k=1)
    rsa_rho, _ = spearmanr(rdm_agent[triu], rdm_ideal[triu])
    results["rsa_spearman"] = float(rsa_rho)

    # --- PCA ---
    pca = PCA(n_components=min(10, n))
    pca.fit(centroids)
    results["pca_variance_explained"] = [float(v) for v in pca.explained_variance_ratio_]
    results["pca_pc1"] = float(pca.explained_variance_ratio_[0])

    # --- Linear probe R² (from full h_t, not centroids) ---
    ridge = Ridge(alpha=1.0)
    cv_scores = cross_val_score(ridge, h_t, counts, cv=5, scoring='r2')
    results["linear_probe_r2"] = float(np.mean(cv_scores))

    # --- Nearest neighbor accuracy ---
    nn_model = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn_model.fit(h_t)
    distances, indices = nn_model.kneighbors(h_t)
    nn_counts = counts[indices[:, 1]]
    adjacent = np.abs(nn_counts.astype(int) - counts.astype(int)) <= 1
    results["nn_accuracy"] = float(np.mean(adjacent))

    # --- Step size CV ---
    if best_geo:
        results["step_size_cv"] = float(np.std(best_geo) / np.mean(best_geo))
        results["step_sizes"] = [float(s) for s in best_geo]
    else:
        results["step_size_cv"] = None

    # --- Per-projection R² ---
    projection_r2 = {}

    # PCA R²
    pca2 = PCA(n_components=2)
    pca_emb = pca2.fit_transform(centroids)
    lr = LinearRegression()
    lr.fit(pca_emb, valid_counts)
    projection_r2["pca"] = float(lr.score(pca_emb, valid_counts))

    # PaCMAP
    try:
        import pacmap
        pac = pacmap.PaCMAP(n_components=2, n_neighbors=min(10, n-1))
        pac_emb = pac.fit_transform(centroids)
        lr.fit(pac_emb, valid_counts)
        projection_r2["pacmap"] = float(lr.score(pac_emb, valid_counts))
    except ImportError:
        pac_emb = None
        projection_r2["pacmap"] = None

    # TriMap
    try:
        import trimap
        tri = trimap.TRIMAP(n_dims=2)
        tri_emb = tri.fit_transform(centroids)
        lr.fit(tri_emb, valid_counts)
        projection_r2["trimap"] = float(lr.score(tri_emb, valid_counts))
    except ImportError:
        tri_emb = None
        projection_r2["trimap"] = None

    results["projection_r2"] = projection_r2

    # =========================================================================
    # FIGURES
    # =========================================================================
    fig_dir = pathlib.Path(output_dir) / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cmap = 'viridis'

    # --- PCA figure ---
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(pca_emb[:, 0], pca_emb[:, 1], c=valid_counts,
                    cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
    for i in range(n - 1):
        ax.plot([pca_emb[i, 0], pca_emb[i+1, 0]],
                [pca_emb[i, 1], pca_emb[i+1, 1]],
                'k-', alpha=0.2, linewidth=0.5, zorder=1)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=7)
    pv = pca2.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({pv[0]:.0%})', fontsize=8)
    ax.set_ylabel(f'PC2 ({pv[1]:.0%})', fontsize=8)
    ax.set_title(f'PCA — {label}', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / 'pca.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- PaCMAP figure ---
    if pac_emb is not None and projection_r2.get("pacmap") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        sc = ax.scatter(pac_emb[:, 0], pac_emb[:, 1], c=valid_counts,
                        cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
        for i in range(n - 1):
            ax.plot([pac_emb[i, 0], pac_emb[i+1, 0]],
                    [pac_emb[i, 1], pac_emb[i+1, 1]],
                    'k-', alpha=0.2, linewidth=0.5, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=7)
        ax.set_xlabel('PaCMAP 1', fontsize=8)
        ax.set_ylabel('PaCMAP 2', fontsize=8)
        ax.set_title(f'PaCMAP — {label}', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'pacmap.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- TriMap figure ---
    if tri_emb is not None and projection_r2.get("trimap") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        sc = ax.scatter(tri_emb[:, 0], tri_emb[:, 1], c=valid_counts,
                        cmap=cmap, s=30, edgecolors='k', linewidths=0.3, zorder=2)
        for i in range(n - 1):
            ax.plot([tri_emb[i, 0], tri_emb[i+1, 0]],
                    [tri_emb[i, 1], tri_emb[i+1, 1]],
                    'k-', alpha=0.2, linewidth=0.5, zorder=1)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=7)
        ax.set_xlabel('TriMap 1', fontsize=8)
        ax.set_ylabel('TriMap 2', fontsize=8)
        ax.set_title(f'TriMap — {label}', fontsize=9, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'trimap.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- RDM figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(rdm_agent, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title(f'Agent RDM ($\\rho$={rsa_rho:.3f})', fontsize=8, fontweight='bold')
    ax1.set_xlabel('Count', fontsize=7)
    ax1.set_ylabel('Count', fontsize=7)
    im2 = ax2.imshow(rdm_ideal, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title('Ideal ordinal RDM', fontsize=8, fontweight='bold')
    ax2.set_xlabel('Count', fontsize=7)
    ax2.set_ylabel('Count', fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / 'rdm.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Arc-length figure ---
    if best_geo:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        arc_length = np.cumsum([0] + best_geo)
        ax.scatter(valid_counts, arc_length, c=valid_counts, cmap=cmap, s=20,
                   edgecolors='k', linewidths=0.3, zorder=2)
        coeffs = np.polyfit(valid_counts, arc_length, 1)
        fit_line = np.polyval(coeffs, valid_counts)
        r2 = results["arc_length_r2"]
        ax.plot(valid_counts, fit_line, 'k--', linewidth=1, alpha=0.7,
                label=f'$R^2$ = {r2:.3f}')
        ax.set_xlabel('Count', fontsize=8)
        ax.set_ylabel('Geodesic arc-length', fontsize=8)
        ax.set_title(f'Arc-length linearity — {label}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(fig_dir / 'arc_length.png', dpi=200, bbox_inches='tight')
        plt.close()

    # --- Step size figure ---
    if best_geo:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.bar(range(n-1), best_geo, color='#2ca02c', edgecolor='k', linewidth=0.3)
        ax.axhline(y=np.mean(best_geo), color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Count transition', fontsize=8)
        ax.set_ylabel('Geodesic step size', fontsize=8)
        cv = results["step_size_cv"]
        ax.set_title(f'Step sizes (CV={cv:.2f}) — {label}', fontsize=8, fontweight='bold')
        plt.tight_layout()
        fig.savefig(fig_dir / 'step_sizes.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Figures saved to {fig_dir}")
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def generate_comparison(models_metrics, output_dir):
    """Generate side-by-side comparison figure and summary JSON.

    models_metrics: dict of {name: metrics_dict}, e.g. {"LSTM": {...}, "MLP": {...}, "MLP-nocount": {...}}
    """

    # Load DreamerV3 baseline from existing battery (seed 0)
    dreamer_baseline = None
    baseline_path = pathlib.Path("/workspace/bridge/artifacts/battery/randproj_s0/randproj_s0_metrics.json")
    if not baseline_path.exists():
        for p in pathlib.Path("/workspace/bridge/artifacts/battery").glob("*_metrics.json"):
            if "randproj" not in str(p) and "lstm" not in str(p) and "mlp" not in str(p):
                baseline_path = p
                break

    if baseline_path.exists():
        with open(baseline_path) as f:
            dreamer_baseline = json.load(f)
        print(f"  Loaded DreamerV3 baseline from {baseline_path}")

    # Model names and colors
    model_names = list(models_metrics.keys())
    colors = {
        "LSTM": "#2ca02c",
        "MLP": "#d62728",
        "MLP-nocount": "#ff7f0e",
    }

    # --- Comparison bar chart ---
    metric_keys = [
        ("GHE", "ghe"),
        ("Arc R\u00b2", "arc_length_r2"),
        ("RSA", "rsa_spearman"),
        ("PCA PC1", "pca_pc1"),
        ("Linear R\u00b2", "linear_probe_r2"),
        ("NN Acc", "nn_accuracy"),
    ]

    labels = []
    model_vals = {name: [] for name in model_names}
    dreamer_vals = []

    for display_name, key in metric_keys:
        all_present = all(models_metrics[n].get(key) is not None for n in model_names)
        if all_present:
            labels.append(display_name)
            for name in model_names:
                model_vals[name].append(models_metrics[name][key])
            dreamer_vals.append(dreamer_baseline.get(key, 0) if dreamer_baseline else 0)

    n_models = len(model_names) + (1 if dreamer_baseline else 0)
    width = 0.8 / n_models
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 4))
    bar_idx = 0
    if dreamer_baseline:
        ax.bar(x + (bar_idx - n_models/2 + 0.5) * width, dreamer_vals, width,
               label='DreamerV3 (ref)', color='#1f77b4', edgecolor='k', linewidth=0.3)
        bar_idx += 1
    for name in model_names:
        ax.bar(x + (bar_idx - n_models/2 + 0.5) * width, model_vals[name], width,
               label=name, color=colors.get(name, '#999999'), edgecolor='k', linewidth=0.3)
        bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('Architecture Independence: Recurrence vs Feedforward',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(pathlib.Path(output_dir) / 'comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # --- Summary JSON ---
    summary = {name.lower().replace("-", "_"): m for name, m in models_metrics.items()}
    if dreamer_baseline:
        summary["dreamer_baseline"] = {
            k: dreamer_baseline.get(k) for k in
            ["ghe", "arc_length_r2", "beta_0", "beta_1", "rsa_spearman",
             "pca_pc1", "linear_probe_r2", "nn_accuracy", "step_size_cv",
             "projection_r2"]
        }

    with open(pathlib.Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print comparison table ---
    print()
    print("=" * 72)
    print("ARCHITECTURE INDEPENDENCE RESULTS")
    print("=" * 72)
    col_w = 14
    header = f"{'Metric':<20}"
    if dreamer_baseline:
        header += f"{'DreamerV3':>{col_w}}"
    for name in model_names:
        header += f"{name:>{col_w}}"
    print(header)
    print("-" * len(header))

    all_keys = ["ghe", "arc_length_r2", "beta_0", "beta_1", "rsa_spearman",
                "pca_pc1", "linear_probe_r2", "nn_accuracy", "step_size_cv"]
    display = {
        "ghe": "GHE", "arc_length_r2": "Arc R\u00b2", "beta_0": "beta_0",
        "beta_1": "beta_1", "rsa_spearman": "RSA", "pca_pc1": "PCA PC1",
        "linear_probe_r2": "Linear probe R\u00b2", "nn_accuracy": "NN accuracy",
        "step_size_cv": "Step size CV",
    }

    for key in all_keys:
        disp_name = display.get(key, key)
        row = f"{disp_name:<20}"
        if dreamer_baseline:
            row += f"{_fmt(dreamer_baseline.get(key)):>{col_w}}"
        for name in model_names:
            row += f"{_fmt(models_metrics[name].get(key)):>{col_w}}"
        print(row)

    for proj in ["pca", "pacmap", "trimap"]:
        disp_name = f"Proj R\u00b2 {proj}"
        row = f"{disp_name:<20}"
        if dreamer_baseline:
            dv = dreamer_baseline.get("projection_r2", {}).get(proj)
            row += f"{_fmt(dv):>{col_w}}"
        for name in model_names:
            mv = models_metrics[name].get("projection_r2", {}).get(proj)
            row += f"{_fmt(mv):>{col_w}}"
        print(row)

    print()
    print(f"Comparison saved to: {output_dir}/comparison.png")
    print(f"Full summary saved to: {output_dir}/summary.json")
    return summary


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LSTM/MLP Architecture Independence Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  all      Run everything (collect → train → eval → compare)
  collect  Collect training data from DreamerV3 policy
  train    Train LSTM and MLP (loads training_data.npz)
  eval     Extract representations (loads trained models)
  compare  Generate comparison figures (loads eval .npz files)
        """)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "collect", "train", "eval", "compare"])
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--lstm_epochs", type=int, default=100)
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp_mask_count", action="store_true",
                        help="Also train/eval a second MLP with count scalars (idx 80-81) zeroed out")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lstm_dir = output_dir / "lstm"
    mlp_dir = output_dir / "mlp"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    mlp_dir.mkdir(parents=True, exist_ok=True)
    if args.mlp_mask_count:
        mlp_nocount_dir = output_dir / "mlp_nocount"
        mlp_nocount_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LSTM/MLP ARCHITECTURE INDEPENDENCE TEST")
    print("=" * 60)
    print(f"  Phase:       {args.phase}")
    print(f"  Checkpoint:  {args.checkpoint_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  LSTM epochs: {args.lstm_epochs}")
    print(f"  MLP epochs:  {args.mlp_epochs}")
    print(f"  Hidden dim:  {HIDDEN_DIM}")
    print(f"  Seed:        {args.seed}")
    print(f"  MLP mask count: {args.mlp_mask_count}")
    print()

    # ---- COLLECT ----
    data_path = output_dir / "training_data.npz"
    if args.phase in ("all", "collect"):
        if data_path.exists() and args.phase != "collect":
            print(f"Training data already exists at {data_path}, skipping collection.")
        else:
            print("=" * 40)
            print("PHASE: DATA COLLECTION")
            print("=" * 40)
            collect_training_data(args.checkpoint_dir, output_dir, seed=args.seed)
        print()

    # ---- TRAIN ----
    lstm_ckpt = output_dir / "lstm_predictor.pt"
    mlp_ckpt = output_dir / "mlp_predictor.pt"

    if args.phase in ("all", "train"):
        print("=" * 40)
        print("PHASE: TRAINING")
        print("=" * 40)

        # Load training data
        assert data_path.exists(), f"Training data not found at {data_path}. Run --phase collect first."
        print(f"Loading training data from {data_path}...")
        data = np.load(data_path)
        obs = data["obs"]
        next_obs = data["next_obs"]
        episode_ids = data["episode_ids"]
        print(f"  {len(obs)} transitions, {len(np.unique(episode_ids))} episodes")
        print()

        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Train LSTM
        print("-" * 30)
        print("Training LSTM predictor...")
        print("-" * 30)
        lstm_model = LSTMPredictor()
        seq_dataset = SequenceDataset(obs, next_obs, episode_ids, seq_len=SEQ_LEN)
        lstm_losses = train_lstm(lstm_model, seq_dataset, args.lstm_epochs, output_dir, lr=args.lr)
        print()

        # Train MLP
        print("-" * 30)
        print("Training MLP predictor...")
        print("-" * 30)
        mlp_model = MLPPredictor()
        trans_dataset = TransitionDataset(obs, next_obs)
        mlp_losses = train_mlp(mlp_model, trans_dataset, args.mlp_epochs, output_dir, lr=args.lr)
        print()

        # Train MLP-nocount (count scalars zeroed out)
        loss_curves = {"LSTM": lstm_losses, "MLP": mlp_losses}
        if args.mlp_mask_count:
            print("-" * 30)
            print("Training MLP-nocount predictor (indices 80-81 masked)...")
            print("-" * 30)
            torch.manual_seed(args.seed)  # same init as MLP for fair comparison
            mlp_nocount_model = MLPPredictor()
            trans_dataset_nc = TransitionDataset(obs.copy(), next_obs.copy(),
                                                 mask_indices=COUNT_SCALAR_INDICES)
            mlp_nc_losses = train_mlp(mlp_nocount_model, trans_dataset_nc, args.mlp_epochs,
                                       output_dir, lr=args.lr,
                                       ckpt_name="mlp_nocount_predictor.pt")
            loss_curves["MLP-nocount"] = mlp_nc_losses
            print()

        # Training curves
        plot_training_curves(loss_curves, output_dir)
        print()

    # ---- EVAL ----
    if args.phase in ("all", "eval"):
        print("=" * 40)
        print("PHASE: EVALUATION")
        print("=" * 40)

        # Load models
        assert lstm_ckpt.exists(), f"LSTM checkpoint not found at {lstm_ckpt}. Run --phase train first."
        assert mlp_ckpt.exists(), f"MLP checkpoint not found at {mlp_ckpt}. Run --phase train first."

        lstm_model = LSTMPredictor()
        lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location="cpu"))
        lstm_model.eval()

        mlp_model = MLPPredictor()
        mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location="cpu"))
        mlp_model.eval()

        # Extract LSTM representations
        print("-" * 30)
        print("Extracting LSTM representations...")
        print("-" * 30)
        lstm_data = extract_representations(lstm_model, "lstm", args.checkpoint_dir,
                                            lstm_dir, seed=args.seed)
        print()

        # Run LSTM battery
        print("Running LSTM measurement battery...")
        lstm_centroids, lstm_valid = compute_centroids(lstm_data["h_t"], lstm_data["counts"])
        print(f"  LSTM centroids: {len(lstm_valid)} counts ({lstm_valid[0]}-{lstm_valid[-1]})")
        lstm_metrics = run_battery(lstm_data["h_t"], lstm_data["counts"],
                                   lstm_centroids, lstm_valid, lstm_dir, "LSTM")
        with open(lstm_dir / "lstm_metrics.json", "w") as f:
            json.dump(lstm_metrics, f, indent=2)
        print()

        # Extract MLP representations
        print("-" * 30)
        print("Extracting MLP representations...")
        print("-" * 30)
        mlp_data = extract_representations(mlp_model, "mlp", args.checkpoint_dir,
                                           mlp_dir, seed=args.seed)
        print()

        # Run MLP battery
        print("Running MLP measurement battery...")
        mlp_centroids, mlp_valid = compute_centroids(mlp_data["h_t"], mlp_data["counts"])
        print(f"  MLP centroids: {len(mlp_valid)} counts ({mlp_valid[0]}-{mlp_valid[-1]})")
        mlp_metrics = run_battery(mlp_data["h_t"], mlp_data["counts"],
                                  mlp_centroids, mlp_valid, mlp_dir, "MLP")
        with open(mlp_dir / "mlp_metrics.json", "w") as f:
            json.dump(mlp_metrics, f, indent=2)
        print()

        # MLP-nocount: same model architecture, count scalars masked
        if args.mlp_mask_count:
            mlp_nc_ckpt = output_dir / "mlp_nocount_predictor.pt"
            assert mlp_nc_ckpt.exists(), (
                f"MLP-nocount checkpoint not found at {mlp_nc_ckpt}. "
                "Run --phase train --mlp_mask_count first.")

            mlp_nc_model = MLPPredictor()
            mlp_nc_model.load_state_dict(torch.load(mlp_nc_ckpt, map_location="cpu"))
            mlp_nc_model.eval()

            print("-" * 30)
            print("Extracting MLP-nocount representations (indices 80-81 masked)...")
            print("-" * 30)
            mlp_nc_data = extract_representations(
                mlp_nc_model, "mlp_nocount", args.checkpoint_dir,
                mlp_nocount_dir, seed=args.seed, mask_indices=COUNT_SCALAR_INDICES)
            print()

            print("Running MLP-nocount measurement battery...")
            mlp_nc_centroids, mlp_nc_valid = compute_centroids(
                mlp_nc_data["h_t"], mlp_nc_data["counts"])
            print(f"  MLP-nocount centroids: {len(mlp_nc_valid)} counts "
                  f"({mlp_nc_valid[0]}-{mlp_nc_valid[-1]})")
            mlp_nc_metrics = run_battery(
                mlp_nc_data["h_t"], mlp_nc_data["counts"],
                mlp_nc_centroids, mlp_nc_valid, mlp_nocount_dir, "MLP-nocount")
            with open(mlp_nocount_dir / "mlp_nocount_metrics.json", "w") as f:
                json.dump(mlp_nc_metrics, f, indent=2)
            print()

    # ---- COMPARE ----
    if args.phase in ("all", "compare"):
        print("=" * 40)
        print("PHASE: COMPARISON")
        print("=" * 40)

        # Load metrics
        lstm_metrics_path = lstm_dir / "lstm_metrics.json"
        mlp_metrics_path = mlp_dir / "mlp_metrics.json"
        assert lstm_metrics_path.exists(), f"LSTM metrics not found at {lstm_metrics_path}. Run --phase eval first."
        assert mlp_metrics_path.exists(), f"MLP metrics not found at {mlp_metrics_path}. Run --phase eval first."

        with open(lstm_metrics_path) as f:
            lstm_metrics = json.load(f)
        with open(mlp_metrics_path) as f:
            mlp_metrics = json.load(f)

        models = {"LSTM": lstm_metrics, "MLP": mlp_metrics}

        if args.mlp_mask_count:
            mlp_nc_metrics_path = mlp_nocount_dir / "mlp_nocount_metrics.json"
            if mlp_nc_metrics_path.exists():
                with open(mlp_nc_metrics_path) as f:
                    models["MLP-nocount"] = json.load(f)
            else:
                print(f"  Warning: MLP-nocount metrics not found at {mlp_nc_metrics_path}, skipping.")

        generate_comparison(models, output_dir)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
