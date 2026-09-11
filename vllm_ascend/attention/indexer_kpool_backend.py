# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend for the GLM-Next pooled sparse indexer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolStateMetadata,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import (
    SparseAttnIndexerKpool,
)


class Glm5NextKPoolIndexerBackend(AscendSFAIndexerBackend):
    """Triton KPool implementation of the unified seven-argument indexer API."""

    def __init__(self, vllm_indexer: nn.Module, qk_rope_head_dim: int) -> None:
        nn.Module.__init__(self)
        if qk_rope_head_dim != 0:
            raise ValueError(
                f"GLM-Next KPool indexing supports NoPE queries only, got qk_rope_head_dim={qk_rope_head_dim}."
            )
        parallel_config = vllm_indexer.vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("GLM-Next KPool indexing does not support PCP or DCP.")

        if get_current_hardware_profile().attention_backend_family is AttentionBackendFamily.COMPATIBILITY:
            raise NotImplementedError("KPool sparse attention requires Ascend A2, A3 or A5.")

        self.n_head: int = vllm_indexer.n_head
        self.head_dim: int = vllm_indexer.head_dim
        self.topk_tokens: int = vllm_indexer.topk_tokens
        self.q_lora_rank: int = vllm_indexer.q_lora_rank
        self.index_kpool: int = vllm_indexer.index_kpool
        self.wq_b = vllm_indexer.wq_b
        self.wk_weights_proj = vllm_indexer.wk_weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        self.index_kpool_compress_ape = vllm_indexer.index_kpool_compress_ape
        self.index_kpool_compress_gate = vllm_indexer.index_kpool_compress_gate
        self.k_cache: Any = vllm_indexer.k_cache
        self.state_cache: Any = vllm_indexer.state_cache
        self.topk_indices_buffer: torch.Tensor | None = vllm_indexer.topk_indices_buffer
        self.indexer_op = SparseAttnIndexerKpool(self.topk_tokens, self.head_dim)
        self.enable_sparse_li_c8 = False
        for name in ("_wk_weight_f32", "_gate_weight_f32", "_norm_weight_f32", "_norm_bias_f32"):
            self.register_buffer(name, None, persistent=False)

    @property
    def topk_output_width(self) -> int:
        return self.topk_tokens + self.index_kpool - 1

    def get_topk_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        visible = (positions + 1).clamp_min(0)
        history = (visible // self.index_kpool * self.index_kpool).clamp(max=self.topk_tokens)
        return history + visible % self.index_kpool

    @property
    def num_cache_tensors(self) -> int:
        return 1

    def process_weights_after_loading(self) -> None:
        self._wk_weight_f32 = self.wk_weights_proj.weight.detach().float()
        self._gate_weight_f32 = self.index_kpool_compress_gate.detach().float()
        self._norm_weight_f32 = self.k_norm.weight.detach().float() if self.k_norm.weight is not None else None
        self._norm_bias_f32 = self.k_norm.bias.detach().float() if self.k_norm.bias is not None else None

    @staticmethod
    def _bound_cache(layer: Any) -> torch.Tensor:
        context = get_forward_context()
        cache = layer.kv_cache
        if isinstance(cache, (list, tuple)):
            virtual_engine = getattr(context, "virtual_engine", 0) or 0
            if virtual_engine >= len(cache):
                raise IndexError(f"Cache virtual engine {virtual_engine} is out of range.")
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            if len(cache) != 1:
                raise TypeError("GLM KPool cache must contain one tensor.")
            cache = cache[0]
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"GLM KPool cache {layer.prefix!r} is not bound to a tensor.")
        return cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        k_hidden_states: torch.Tensor,
        indexer_metadata: Any,
        compute_topk: bool = True,
    ) -> torch.Tensor | None:
        del cos, sin
        if not isinstance(indexer_metadata, AscendIndexerKPoolMetadata):
            raise TypeError("GLM KPool backend requires AscendIndexerKPoolMetadata.")
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            raise TypeError("GLM KPool backend requires per-layer metadata.")
        state_metadata = context.attn_metadata[self.state_cache.prefix]
        if not isinstance(state_metadata, AscendIndexerKPoolStateMetadata):
            raise TypeError("GLM KPool backend requires compressor-state metadata.")

        num_tokens = hidden_states.shape[0]
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            num_tokens = min(num_tokens, indexer_metadata.num_actual_tokens)
        hidden = hidden_states[:num_tokens]
        k_hidden = k_hidden_states[:num_tokens]
        if self._wk_weight_f32 is None:
            self.process_weights_after_loading()
        assert self._wk_weight_f32 is not None
        hidden_f32 = hidden.float()
        k_hidden_f32 = hidden_f32 if k_hidden_states is hidden_states else k_hidden.float()
        projected = F.linear(k_hidden_f32, self._wk_weight_f32)
        k = F.layer_norm(
            projected[:, : self.head_dim],
            (self.head_dim,),
            self._norm_weight_f32,
            self._norm_bias_f32,
            getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6)),
        )
        gate_score = F.linear(k_hidden_f32, self._gate_weight_f32)
        q_values = None
        weights = None
        if compute_topk:
            if isinstance(q_c, tuple):
                raise TypeError("GLM KPool backend requires an unquantized q_c tensor.")
            q_values = self.wq_b(q_c[:num_tokens])[0].view(num_tokens, self.n_head, self.head_dim)
            weights = (
                projected[:, self.head_dim :]
                if k_hidden_states is hidden_states
                else F.linear(hidden_f32, self._wk_weight_f32[self.head_dim :])
            ).to(q_values.dtype)
            weights = weights * (self.softmax_scale * self.n_head**-0.5)

        indexer_cache = self._bound_cache(self.k_cache)
        state_cache = self._bound_cache(self.state_cache)
        positions = indexer_metadata.positions[:num_tokens]
        result = self.indexer_op(
            k,
            q_values,
            weights,
            positions,
            indexer_cache,
            state_cache,
            indexer_metadata,
            state_metadata,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            max_pool_seq_len=(
                indexer_metadata.block_table.shape[1] * indexer_cache.shape[1]
                if context.cudagraph_runtime_mode == CUDAGraphMode.FULL or indexer_metadata.seq_lens_cpu is None
                else int(indexer_metadata.seq_lens_cpu.max())
                if indexer_metadata.seq_lens_cpu.numel()
                else 0
            ),
            compute_topk=compute_topk,
        )
        if result is None or self.topk_indices_buffer is None:
            return result

        if num_tokens > self.topk_indices_buffer.shape[0]:
            raise RuntimeError(
                f"GLM KPool output exceeds the top-k buffer rows: {num_tokens} > {self.topk_indices_buffer.shape[0]}."
            )
        output = self.topk_indices_buffer[:num_tokens]
        output.fill_(-1)
        if result.shape[-1] > output.shape[-1]:
            raise RuntimeError(
                f"GLM KPool output exceeds the top-k buffer width: {result.shape[-1]} > {output.shape[-1]}."
            )
        output[:, : result.shape[-1]].copy_(result[:, 0])
        return result
