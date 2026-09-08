# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enable the GLM-5 kpool sparse indexer on Ascend.

``GLM53_KPOOL_INDEXER=1`` (default off) swaps the NVIDIA backend's
``Indexer`` init/forward -- CUDA FWHT + fp8 scoring, which triton-ascend
cannot compile -- for a BF16 implementation that drives
``vllm_ascend.ops.glm5_kpool_indexer_ascend``, and routes the fp8
``kpool_compress_and_write_cache`` to the BF16 triton kernel.

The patched ``Indexer.__init__`` keeps the checkpoint parameter names
(the CUDA implementation's weights load unchanged) but binds the two
Ascend cache layers from ``vllm_ascend.ops.glm5_kpool_caches``: the
pool-granular indexer K cache and the token-granular compressor-state
cache.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_KPOOL_INDEXER_ENABLED = os.environ.get("GLM53_KPOOL_INDEXER", "0") == "1"


def _install_kpool_indexer() -> bool:
    """Patch the GLM Indexer (init + forward) onto the Ascend path."""
    if not _KPOOL_INDEXER_ENABLED:
        return False
    try:
        import vllm.models.glm5next.nvidia.attention as glm_attn

        from vllm_ascend.ops.glm5_kpool_caches import (
            AscendGlm5NextCompressorStateCache,
            AscendGlm5NextIndexerKPoolCache,
        )
        from vllm_ascend.ops.glm5_kpool_indexer_ascend import (
            AscendGlm5KpoolIndexerOp,
        )
    except Exception as exc:
        logger.warning("kpool indexer patch import failed: %r", exc)
        return False

    def _ascend_indexer_init(
        self,
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix,
    ):
        # Same weights and parameter names as the CUDA Indexer (checkpoint
        # loads unchanged), but the caches are the Ascend BF16 layouts and
        # there is no CUDA indexer op.
        import torch.nn as nn
        from vllm.model_executor.layers.layernorm import LayerNorm
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            ReplicatedLinear,
        )
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,
        )

        nn.Module.__init__(self)
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_kpool = config.index_kpool
        self.is_rope_neox_style = not getattr(config, "indexer_rope_interleave", False)
        self.q_lora_rank = q_lora_rank
        self.topk_indices_buffer = topk_indices_buffer
        self.softmax_scale = self.head_dim**-0.5

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
        )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.state_cache = AscendGlm5NextCompressorStateCache(
            state_dim=2 * self.head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.index_kpool,
            cache_config=cache_config,
            prefix=f"{prefix}.compressor.state_cache",
        )
        self.k_cache = AscendGlm5NextIndexerKPoolCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            cache_role="indexer",
            cache_config=cache_config,
            prefix=f"{prefix}.k_cache",
            compress_ratio=self.index_kpool,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.prefix = prefix
        self.quant_block_size = self.head_dim
        self.scale_fmt = None
        self.indexer_op = AscendGlm5KpoolIndexerOp(
            self.k_cache,
            self.state_cache,
            topk_tokens=self.topk_tokens,
            head_dim=self.head_dim,
            topk_indices_buffer=self.topk_indices_buffer,
            attn_layer_name=prefix.removesuffix(".indexer") + ".attn",
        )
        self._wp_fp32 = None  # forward-compat with the CUDA signature (unused here)
        self.indexer_rotary_emb = (
            get_rope(
                self.rope_dim,
                max_position=self.max_model_len,
                rope_parameters=config.rope_parameters,
                is_neox_style=self.is_rope_neox_style,
            )
            if self.rope_dim > 0
            else None
        )

    def _ascend_indexer_forward(self, hidden_states, qr, positions, rotary_emb):
        # BF16 q/k, weights scaled by softmax_scale * n_head**-0.5; no
        # FWHT/fp8. Drives the Ascend orchestrator.
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        k = self.k_norm(k)
        weights = kw[:, self.head_dim :] * (self.softmax_scale * self.n_head**-0.5)

        if self.rope_dim > 0:
            if rotary_emb is None:
                raise ValueError("GLM-5 Indexer requires rope when qk_rope_head_dim > 0.")
            q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            k_pe = k_pe.reshape(-1, 1, self.rope_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        gate_score = F.linear(hidden_states, self.index_kpool_compress_gate)
        return self.indexer_op(
            hidden_states,
            q,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )

    glm_attn.Indexer.__init__ = _ascend_indexer_init
    glm_attn.Indexer.forward = _ascend_indexer_forward
    logger.info("kpool indexer: Indexer init/forward patched to the Ascend path")
    return True


def _install_bf16_kpool_compress() -> bool:
    """Route the fp8 kpool compress to the BF16 triton kernel."""
    if not _KPOOL_INDEXER_ENABLED:
        return False
    try:
        import vllm.models.glm5next.nvidia.ops.kpool_compress as glm_ops

        from vllm_ascend.ops.triton.glm5_next_kpool_compress import (
            glm5_next_kpool_compress_and_write_cache_triton,
        )
    except Exception as exc:
        logger.warning("bf16 kpool compress patch import failed: %r", exc)
        return False

    def _bf16_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size,
        head_dim,
        write_mask=None,
        round_scale=True,
        return_compressed=False,
        write_cache=True,
    ):
        del round_scale  # The BF16 path has no fp8 scale rounding.
        # The triton kernel's cache view is [blocks, block, 1, head_dim];
        # callers pass the cache as allocated, so only the shape contract
        # differs and is enforced by the kernel's validation.
        out = glm5_next_kpool_compress_and_write_cache_triton(
            kv_cache,
            slot_k,
            slot_score,
            ape,
            loc,
            write_mask=write_mask,
            return_compressed=return_compressed,
            write_cache=write_cache,
        )
        return out

    glm_ops.kpool_compress_and_write_cache = _bf16_compress_and_write_cache
    logger.info("kpool indexer: fp8 kpool compress routed to the BF16 triton kernel")
    return True


def apply_kpool_indexer_patches() -> None:
    """Entry point for the patch chain (no-op unless GLM53_KPOOL_INDEXER=1)."""
    if not _KPOOL_INDEXER_ENABLED:
        return
    ok_fwd = _install_kpool_indexer()
    ok_cmp = _install_bf16_kpool_compress()
    logger.info(
        "kpool indexer patches: forward=%s compress=%s", ok_fwd, ok_cmp
    )


# The patch chain imports this module for its side effects.
apply_kpool_indexer_patches()
