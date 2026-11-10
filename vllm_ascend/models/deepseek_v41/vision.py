# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend DeepSeek vision tower (ViT + aligner), replicated (no TP/DP sharding).

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time.
"""

import logging
from collections import OrderedDict
from functools import lru_cache
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

_ROPE_CACHE_LOG_INTERVAL = 100
_ROPE_CACHE_MAX_ENTRIES = 8
_ROPE_CACHE_MAX_BYTES = 64 * 1024 * 1024


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


class DeepseekV41PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV41VisionAttention(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        qk = self.apply_rotary_emb(torch.stack((q, k)), cos.squeeze(1), sin.squeeze(1))
        q, k = qk.unbind()
        o = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.wo(o.squeeze(0).reshape(n, -1))


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)))


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.attn = DeepseekV41VisionAttention(config)
        self.norm2 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return residual + self.mlp(self.norm2(residual))


class DeepseekV41ViT(nn.Module):
    """Ascend DeepSeek ViT: full bidirectional attention per image, 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self._rope_cache_logger = init_logger(__name__)
        self._rope_cache_hits = 0
        self._rope_cache_misses = 0
        self._rope_cache_evictions = 0
        self._rope_cache_oversized = 0
        self._rope_cache = OrderedDict()
        self._rope_cache_bytes = 0
        self.patch_embed = DeepseekV41PatchEmbed(config)
        self.blocks = nn.ModuleList([DeepseekV41VisionBlock(config) for _ in range(config.vision_n_layers)])
        self.norm = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)

    def _apply(self, fn, recurse=True):
        # Cached FP32 tables are derived state, not weights or checkpoint buffers.
        self._rope_cache.clear()
        self._rope_cache_bytes = 0
        return super()._apply(fn, recurse=recurse)

    def _log_rope_cache_stats(self, key):
        # Lifetime counters: include warmup and survive model device/dtype changes.
        calls = self._rope_cache_hits + self._rope_cache_misses
        if calls != 1 and calls % _ROPE_CACHE_LOG_INTERVAL:
            return
        self._rope_cache_logger.info(
            "ViT RoPE cache instance=%s device=%s calls=%d hits=%d misses=%d "
            "hit_rate=%.2f%% evictions=%d oversized=%d entries=%d bytes=%d "
            "last_grid=%dx%d",
            hex(id(self)),
            key[-1],
            calls,
            self._rope_cache_hits,
            self._rope_cache_misses,
            100.0 * self._rope_cache_hits / calls,
            self._rope_cache_evictions,
            self._rope_cache_oversized,
            len(self._rope_cache),
            self._rope_cache_bytes,
            key[0],
            key[1],
        )

    def _get_rope(self, n_h: int, n_w: int, device: torch.device):
        key = (n_h, n_w, self.rope_dim, self.rope_theta, device)
        if key in self._rope_cache:
            self._rope_cache_hits += 1
            self._rope_cache.move_to_end(key)
            self._log_rope_cache_stats(key)
            return self._rope_cache[key]

        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        self._rope_cache_misses += 1
        size = sum(t.numel() * t.element_size() for t in (cos, sin))
        # Oversized tables remain usable without displacing the working set.
        if size <= _ROPE_CACHE_MAX_BYTES:
            while self._rope_cache and (
                len(self._rope_cache) >= _ROPE_CACHE_MAX_ENTRIES
                or self._rope_cache_bytes + size > _ROPE_CACHE_MAX_BYTES
            ):
                _, old = self._rope_cache.popitem(last=False)
                self._rope_cache_evictions += 1
                self._rope_cache_bytes -= sum(t.numel() * t.element_size() for t in old)
        result = (cos.to(device=device), sin.to(device=device))
        if size <= _ROPE_CACHE_MAX_BYTES:
            self._rope_cache[key] = result
            self._rope_cache_bytes += size
        else:
            self._rope_cache_oversized += 1
        self._log_rope_cache_stats(key)
        return result

    def forward(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        # Diagnostic only: synchronization changes scheduling; use eager mode.
        timing = self._rope_cache_logger.isEnabledFor(logging.DEBUG)
        if timing:
            torch.npu.synchronize(patches.device)
            started = perf_counter()
        x = self.patch_embed(patches)
        cos, sin = self._get_rope(n_vit_h, n_vit_w, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        result = self.norm(x)
        if timing:
            torch.npu.synchronize(patches.device)
            elapsed_ms = (perf_counter() - started) * 1000
            self._rope_cache_logger.debug(
                "MM_VIT device=%s grid=%dx%d patches=%d elapsed_ms=%.3f",
                patches.device,
                n_vit_h,
                n_vit_w,
                patches.shape[0],
                elapsed_ms,
            )
        return result

    def forward_packed(
        self,
        patches: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Tensor-only capture path with precomputed RoPE and image boundaries."""
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, max_seqlen)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
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

    def forward_packed(
        self,
        x: torch.Tensor,
        merge_idx: torch.Tensor,
        merge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match unfold's channel-major merge, including odd-grid padding."""
        rows = merge_idx.shape[0]
        gathered = (x[merge_idx] * merge_mask).view(rows, self.downsample_ratio**2, -1)
        gathered = gathered.permute(0, 2, 1).reshape(rows, -1)
        return self.w2(F.gelu(self.w1(gathered)))
