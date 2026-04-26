"""Small causal Transformer mapping per-frame audio features → VAE latents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class SequencerConfig:
    feature_dim: int = 146       # matches laser_ai.audio.features.FEATURE_DIM
    latent_dim: int = 64
    hidden: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 4096          # supports up to ~2.3 min at 30 fps


class _CausalTransformerBlock(nn.Module):
    def __init__(self, cfg: SequencerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden, num_heads=cfg.n_heads,
            dropout=cfg.dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden * cfg.ff_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden * cfg.ff_mult, cfg.hidden),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class AudioToLatentSequencer(nn.Module):
    def __init__(self, cfg: SequencerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SequencerConfig()
        self.in_proj = nn.Linear(self.cfg.feature_dim, self.cfg.hidden)
        # Learned positional embedding; simpler than sinusoidal and adequate at this scale
        self.pos_emb = nn.Embedding(self.cfg.max_len, self.cfg.hidden)
        self.blocks = nn.ModuleList([
            _CausalTransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(self.cfg.hidden)
        self.out_proj = nn.Linear(self.cfg.hidden, self.cfg.latent_dim)

    def forward(self, features: torch.Tensor, *, pos_offset: int = 0) -> torch.Tensor:
        B, T, _ = features.shape
        if T + pos_offset > self.cfg.max_len:
            raise ValueError(
                f"sequence length {T} + pos_offset {pos_offset} > max_len {self.cfg.max_len}"
            )
        pos = torch.arange(T, device=features.device) + pos_offset
        x = self.in_proj(features) + self.pos_emb(pos).unsqueeze(0)
        # Causal mask (T, T) — True = disallow attending
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=features.device),
                          diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out_proj(self.ln_out(x))
