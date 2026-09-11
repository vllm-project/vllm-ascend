# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision tower (ViT + aligner), replicated (no TP/DP sharding).

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=True,
            enable_fp32_compute=True,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        qk = self.apply_rotary_emb(torch.stack((q, k)), cos.squeeze(1), sin.squeeze(1))
        q, k = qk.unbind()
        o = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        return self.wo(o.squeeze(0).reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)))


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x + self.attn(self.norm1(x), cos, sin)
        return residual + self.mlp(self.norm2(residual))


class DeepseekV4ViT(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention per image, 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList([DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)])
        self.norm = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)

    def forward(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
