#!/usr/bin/env python3
"""
Symbolic RSSM — Token-Input World Model for Binary Counting
=============================================================
Self-contained RSSM matching the physical binary specialist architecture
(GRU 512, categorical 32×32) but with:
  - Encoder: learned embeddings (vocab 2, 4 positions) → concat → linear → 512
  - Decoder: MLP → 4 × 2-class logits (cross-entropy loss)

All other components identical to the physical specialist:
  - GRU hidden (deter): 512
  - Stochastic latent: 32 variables × 32 classes = 1024 flat
  - Prior/posterior heads with LayerNorm + SiLU
  - KL balancing: dyn_scale=0.5, rep_scale=0.1, free_bits=1.0
  - Reward/continuation heads (trivial for passive env but kept for compatibility)

Architecture differences from physical specialist (documented for comparison):
  1. Encoder: Embedding(2, 128) × 4 positions → concat(512) → Linear(512, 512)
     vs physical: Linear(72, 512) → SiLU → LN → Linear(512, 512) → SiLU → LN → Linear(512, 512)
  2. Decoder: Linear(1536, 512) → SiLU → Linear(512, 4×2) with cross-entropy
     vs physical: Linear(1536, 512) → SiLU → LN → ... → Linear(512, 72) with symlog MSE
  3. No displacement loss (symbolic env has no spatial structure)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Architecture constants — matching physical binary specialist
# ---------------------------------------------------------------------------

DETER_DIM = 512          # GRU hidden size
STOCH_DIM = 32           # Number of stochastic variables
STOCH_CLASSES = 32        # Classes per variable
STOCH_FLAT = STOCH_DIM * STOCH_CLASSES  # 1024
HIDDEN_DIM = 512          # Internal layer size
EMBED_DIM = 512           # Encoder output

# Encoder specifics
NUM_BITS = 4
VOCAB_SIZE = 2            # {0, 1}
TOKEN_EMBED_DIM = 128     # Per-token embedding dimension

# Feature size = deter + stoch_flat
FEAT_DIM = DETER_DIM + STOCH_FLAT  # 1536

# Loss weights (matching physical specialist defaults)
DYN_SCALE = 0.5
REP_SCALE = 0.1
FREE_BITS = 1.0
UNIMIX = 0.01


# ---------------------------------------------------------------------------
# Encoder: Token Embeddings → Fixed-size vector
# ---------------------------------------------------------------------------

class TokenEncoder(nn.Module):
    """Embed 4 binary tokens and project to RSSM input space.

    Each bit position gets its own embedding table (2 entries × 128 dims).
    The 4 embeddings are concatenated (512) and projected through a linear
    layer to produce the 512-dim embedding vector.

    This is deliberately simple — no attention, no positional encoding beyond
    separate embedding tables per position. The RSSM's recurrence provides
    all temporal context.
    """

    def __init__(self):
        super().__init__()
        # Separate embedding per bit position (preserves positional information)
        self.embeddings = nn.ModuleList([
            nn.Embedding(VOCAB_SIZE, TOKEN_EMBED_DIM)
            for _ in range(NUM_BITS)
        ])
        # Projection: concat(4 × 128) = 512 → 512
        self.proj = nn.Linear(NUM_BITS * TOKEN_EMBED_DIM, EMBED_DIM)
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (..., 4) int64 → (..., 512)"""
        parts = []
        for i in range(NUM_BITS):
            parts.append(self.embeddings[i](tokens[..., i]))  # (..., 128)
        x = torch.cat(parts, dim=-1)  # (..., 512)
        return F.silu(self.ln(self.proj(x)))  # (..., 512)


# ---------------------------------------------------------------------------
# Decoder: Feature vector → Token logits
# ---------------------------------------------------------------------------

class TokenDecoder(nn.Module):
    """Predict 4 binary tokens from RSSM feature vector.

    Input: concat(deter, stoch_flat) = 1536
    Output: (4, 2) logits — one binary classification per bit position.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, NUM_BITS * VOCAB_SIZE),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (..., 1536) → (..., 4, 2) logits"""
        logits = self.net(feat)  # (..., 8)
        shape = logits.shape[:-1] + (NUM_BITS, VOCAB_SIZE)
        return logits.reshape(shape)


# ---------------------------------------------------------------------------
# RSSM Core — matching DreamerV3 NM512 architecture
# ---------------------------------------------------------------------------

class NormedGRUCell(nn.Module):
    """GRU cell with LayerNorm, matching DreamerV3's implementation."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.ln = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]:
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        HS = self.hidden_size
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h
        return self.ln(h_new)


class SymbolicRSSM(nn.Module):
    """Full RSSM for symbolic binary counting.

    Architecture matches physical binary specialist (NM512 DreamerV3):
      deter=512, stoch=32×32, hidden=512, embed=512.

    Differences:
      - Encoder: TokenEncoder (embedding-based)
      - Decoder: TokenDecoder (cross-entropy)
    """

    def __init__(self):
        super().__init__()

        # Encoder / Decoder
        self.encoder = TokenEncoder()
        self.decoder = TokenDecoder()

        # GRU input projection: stoch_flat(1024) + action(1) → hidden(512)
        # (action dim=1 for compatibility, always zero)
        self.img_in = nn.Linear(STOCH_FLAT + 1, HIDDEN_DIM)
        self.img_in_ln = nn.LayerNorm(HIDDEN_DIM)

        # GRU cell
        self.gru = NormedGRUCell(HIDDEN_DIM, DETER_DIM)

        # Prior head: deter → stoch logits
        self.img_out = nn.Linear(DETER_DIM, HIDDEN_DIM)
        self.img_out_ln = nn.LayerNorm(HIDDEN_DIM)
        self.prior_head = nn.Linear(HIDDEN_DIM, STOCH_FLAT)

        # Posterior head: deter + embed → stoch logits
        self.obs_out = nn.Linear(DETER_DIM + EMBED_DIM, HIDDEN_DIM)
        self.obs_out_ln = nn.LayerNorm(HIDDEN_DIM)
        self.post_head = nn.Linear(HIDDEN_DIM, STOCH_FLAT)

        # Reward head (trivial — always 0, but kept for architecture match)
        self.reward_head = nn.Sequential(
            nn.Linear(FEAT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

        # Continuation head
        self.cont_head = nn.Sequential(
            nn.Linear(FEAT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

        # Learned initial state
        self.init_deter = nn.Parameter(torch.zeros(DETER_DIM))
        self.init_stoch = nn.Parameter(torch.zeros(STOCH_FLAT))

    def initial_state(self, batch_size: int, device: torch.device):
        """Returns (deter, stoch_flat) initial states."""
        h = self.init_deter.unsqueeze(0).expand(batch_size, -1)
        z = self.init_stoch.unsqueeze(0).expand(batch_size, -1)
        return h, z

    def _categorical_straight_through(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, STOCH_DIM, STOCH_CLASSES) → (B, STOCH_FLAT)"""
        # Add uniform mixture for exploration (matching DreamerV3 unimix)
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / STOCH_CLASSES
        probs = (1 - UNIMIX) * probs + UNIMIX * uniform

        # Straight-through
        hard = torch.zeros_like(probs)
        hard.scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
        z = hard - probs.detach() + probs
        return z.reshape(logits.shape[0], -1)

    def observe_step(self, tokens, action, h_prev, z_prev):
        """Single observation step.

        Args:
            tokens: (B, 4) int64
            action: (B, 1) float32
            h_prev: (B, 512) deter
            z_prev: (B, 1024) stoch_flat

        Returns dict with: deter, stoch, prior_logits, post_logits, feat, embed
        """
        B = tokens.shape[0]

        # Encode observation
        embed = self.encoder(tokens)  # (B, 512)

        # GRU input from previous stochastic + action
        gru_in = torch.cat([z_prev, action], dim=-1)  # (B, 1025)
        gru_in = F.silu(self.img_in_ln(self.img_in(gru_in)))  # (B, 512)

        # GRU step
        h = self.gru(gru_in, h_prev)  # (B, 512)

        # Prior
        prior_x = F.silu(self.img_out_ln(self.img_out(h)))
        prior_logits = self.prior_head(prior_x).reshape(B, STOCH_DIM, STOCH_CLASSES)

        # Posterior
        post_in = torch.cat([h, embed], dim=-1)  # (B, 1024)
        post_x = F.silu(self.obs_out_ln(self.obs_out(post_in)))
        post_logits = self.post_head(post_x).reshape(B, STOCH_DIM, STOCH_CLASSES)

        # Sample stochastic
        z_flat = self._categorical_straight_through(post_logits)  # (B, 1024)

        # Feature vector
        feat = torch.cat([h, z_flat], dim=-1)  # (B, 1536)

        return {
            "deter": h,
            "stoch": z_flat,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "feat": feat,
            "embed": embed,
        }

    def imagine_step(self, action, h_prev, z_prev):
        """Imagination step (no observation — prior only).

        Used for latent rollout divergence analysis.
        """
        B = h_prev.shape[0]

        gru_in = torch.cat([z_prev, action], dim=-1)
        gru_in = F.silu(self.img_in_ln(self.img_in(gru_in)))
        h = self.gru(gru_in, h_prev)

        prior_x = F.silu(self.img_out_ln(self.img_out(h)))
        prior_logits = self.prior_head(prior_x).reshape(B, STOCH_DIM, STOCH_CLASSES)

        z_flat = self._categorical_straight_through(prior_logits)
        feat = torch.cat([h, z_flat], dim=-1)

        return {
            "deter": h,
            "stoch": z_flat,
            "prior_logits": prior_logits,
            "feat": feat,
        }

    def forward(self, tokens_seq, actions_seq, is_first_seq):
        """Process full sequence.

        Args:
            tokens_seq:  (B, T, 4) int64
            actions_seq: (B, T, 1) float32
            is_first_seq: (B, T) bool/float — episode boundaries

        Returns dict of (B, T, ...) tensors.
        """
        B, T, _ = tokens_seq.shape
        device = tokens_seq.device

        h, z = self.initial_state(B, device)

        all_deter, all_stoch = [], []
        all_prior, all_post = [], []
        all_feat, all_embed = [], []

        for t in range(T):
            # Reset state at episode boundaries
            mask = is_first_seq[:, t].float().unsqueeze(-1)  # (B, 1)
            h_init, z_init = self.initial_state(B, device)
            h = h * (1 - mask) + h_init * mask
            z = z * (1 - mask) + z_init * mask

            out = self.observe_step(
                tokens_seq[:, t],
                actions_seq[:, t],
                h, z,
            )
            h = out["deter"]
            z = out["stoch"]

            all_deter.append(out["deter"])
            all_stoch.append(out["stoch"])
            all_prior.append(out["prior_logits"])
            all_post.append(out["post_logits"])
            all_feat.append(out["feat"])
            all_embed.append(out["embed"])

        return {
            "deter": torch.stack(all_deter, dim=1),
            "stoch": torch.stack(all_stoch, dim=1),
            "prior_logits": torch.stack(all_prior, dim=1),
            "post_logits": torch.stack(all_post, dim=1),
            "feat": torch.stack(all_feat, dim=1),
            "embed": torch.stack(all_embed, dim=1),
        }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_kl_loss(post_logits, prior_logits):
    """KL divergence with free bits, matching DreamerV3.

    Returns (dyn_loss, rep_loss) separately for balancing.
    """
    post_probs = F.softmax(post_logits, dim=-1)
    prior_probs = F.softmax(prior_logits, dim=-1)

    # Add unimix
    uniform = torch.ones_like(post_probs) / STOCH_CLASSES
    post_probs_mix = (1 - UNIMIX) * post_probs + UNIMIX * uniform
    prior_probs_mix = (1 - UNIMIX) * prior_probs + UNIMIX * uniform

    # KL per dim, summed over classes
    def kl(p, q):
        return (p * (p.log() - q.log())).sum(dim=-1).sum(dim=-1)  # (B, T)

    # Dynamic loss: KL(sg(post) || prior) — trains prior
    dyn_kl = kl(post_probs_mix.detach(), prior_probs_mix)
    dyn_loss = torch.clamp(dyn_kl, min=FREE_BITS).mean()

    # Representation loss: KL(post || sg(prior)) — trains encoder/posterior
    rep_kl = kl(post_probs_mix, prior_probs_mix.detach())
    rep_loss = torch.clamp(rep_kl, min=FREE_BITS).mean()

    return dyn_loss, rep_loss


def compute_decoder_loss(model, feat, tokens_target):
    """Cross-entropy reconstruction loss for token prediction.

    feat: (B, T, 1536)
    tokens_target: (B, T, 4) int64
    """
    logits = model.decoder(feat)  # (B, T, 4, 2)
    B, T, N, V = logits.shape
    # Reshape for cross-entropy: (B*T*4, 2) vs (B*T*4,)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        tokens_target.reshape(-1),
        reduction="mean",
    )
    # Per-bit accuracy for logging
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B, T, 4)
        acc = (preds == tokens_target).float().mean()
    return loss, acc


def compute_total_loss(model, outputs, tokens_seq):
    """Total world model loss matching DreamerV3 structure."""
    dyn_loss, rep_loss = compute_kl_loss(outputs["post_logits"], outputs["prior_logits"])
    decoder_loss, token_acc = compute_decoder_loss(model, outputs["feat"], tokens_seq)

    # Reward and continuation are trivial (constant 0 and constant 1)
    # but we include them for architecture completeness
    feat = outputs["feat"]
    reward_pred = model.reward_head(feat).squeeze(-1)
    reward_loss = reward_pred.pow(2).mean()  # target is always 0

    cont_pred = model.cont_head(feat).squeeze(-1)
    # continuation is 1 except at episode end — approximate as all-1 target
    cont_loss = F.binary_cross_entropy_with_logits(
        cont_pred,
        torch.ones_like(cont_pred),
        reduction="mean",
    )

    total = decoder_loss + DYN_SCALE * dyn_loss + REP_SCALE * rep_loss + reward_loss + cont_loss

    return {
        "total": total,
        "decoder": decoder_loss,
        "dyn_kl": dyn_loss,
        "rep_kl": rep_loss,
        "reward": reward_loss,
        "cont": cont_loss,
        "token_acc": token_acc,
    }


# ---------------------------------------------------------------------------
# Param counting
# ---------------------------------------------------------------------------

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "gru": sum(p.numel() for p in model.gru.parameters()),
        "prior_head": sum(p.numel() for p in list(model.img_out.parameters()) +
                         list(model.img_out_ln.parameters()) +
                         list(model.prior_head.parameters())),
        "post_head": sum(p.numel() for p in list(model.obs_out.parameters()) +
                        list(model.obs_out_ln.parameters()) +
                        list(model.post_head.parameters())),
        "reward_head": sum(p.numel() for p in model.reward_head.parameters()),
        "cont_head": sum(p.numel() for p in model.cont_head.parameters()),
    }
    return total, parts


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SymbolicRSSM — Smoke Test")
    print("=" * 60)

    device = torch.device("cpu")
    model = SymbolicRSSM().to(device)

    total, parts = count_params(model)
    print(f"\nTotal trainable parameters: {total:,}")
    for name, count in parts.items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    B, T = 4, 16
    tokens = torch.randint(0, 2, (B, T, NUM_BITS), device=device)
    actions = torch.zeros(B, T, 1, device=device)
    is_first = torch.zeros(B, T, device=device)
    is_first[:, 0] = 1.0

    print(f"\n[1] Forward pass (B={B}, T={T})...")
    outputs = model(tokens, actions, is_first)
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # Test loss
    print("\n[2] Loss computation...")
    losses = compute_total_loss(model, outputs, tokens)
    for k, v in losses.items():
        print(f"  L_{k}: {v.item():.4f}")

    # Test backward
    print("\n[3] Backward pass...")
    losses["total"].backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {grad_count}")

    # Test imagination
    print("\n[4] Imagination step...")
    h, z = model.initial_state(B, device)
    action = torch.zeros(B, 1, device=device)
    obs = model.observe_step(tokens[:, 0], action, h, z)
    imag = model.imagine_step(action, obs["deter"], obs["stoch"])
    print(f"  deter: {imag['deter'].shape}")
    print(f"  stoch: {imag['stoch'].shape}")

    # Verify feature dimensions match physical specialist
    assert outputs["deter"].shape[-1] == 512, "deter must be 512"
    assert outputs["stoch"].shape[-1] == 1024, "stoch flat must be 1024"
    assert outputs["feat"].shape[-1] == 1536, "feat must be 1536"

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print(f"Feature dim: {outputs['feat'].shape[-1]} (matches physical specialist)")
    print("=" * 60)
