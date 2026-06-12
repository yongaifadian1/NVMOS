from __future__ import annotations

import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, key_padding_mask: torch.Tensor | None):
        out, weights = self.attn(
            query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        query = self.norm1(query + self.drop(out))
        query = self.norm2(query + self.drop(self.ffn(query)))
        return query, weights


class TextQueryNVMOS(nn.Module):
    def __init__(
        self,
        audio_dim: int = 1024,
        text_dim: int = 1024,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim))
        self.blocks = nn.ModuleList([CrossBlock(hidden_dim, heads, ffn_dim, dropout) for _ in range(layers)])
        self.reg = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frames: torch.Tensor, tag_ctx: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        memory = self.audio_proj(frames)
        query = self.text_proj(tag_ctx).unsqueeze(1)
        attn = None
        for block in self.blocks:
            query, attn = block(query, memory, key_padding_mask)
        return self.reg(query.squeeze(1)).squeeze(-1), attn
