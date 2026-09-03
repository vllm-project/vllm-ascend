# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer layer for the GLM-5.3-Flash kpool indexer.

GLM-5.3-Flash enables its sparse indexer only when the checkpoint sets
``index_topk``; with ``index_topk`` unset the model runs as dense NoPE MLA plus
KDA, which is the configuration vLLM Ascend currently supports.

This module keeps the layer so the sparse configuration still builds its index-K
and tail caches (the KV cache specs are wired up and PD transfer works), but the
scoring path is not implemented on Ascend. The upstream implementation is a set
of fused CUDA kernels -- block-FP8 MQA logits through DeepGEMM, paged MQA logits,
and radix top-k over a device workspace -- with no NPU equivalent yet. Rather
than carry that unreachable code, ``forward_oot`` reports the gap so a sparse
checkpoint fails at load with an actionable message instead of dispatching into
a CUDA-only kernel.
"""

import torch
from vllm.logger import logger
from vllm.model_executor.custom_op import CustomOp

_UNSUPPORTED_MESSAGE = (
    "GLM-5.3-Flash sparse (kpool) attention indexing is not implemented on "
    "Ascend NPU. The dense NoPE MLA path is supported: serve a checkpoint whose "
    "config leaves `index_topk` unset."
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse attention indexer op for the GLM-5.3-Flash kpool indexer.

    The op is kept as a ``CustomOp`` so the Ascend scoring path can be added
    later as a plain ``forward_oot`` implementation, matching how the other
    Ascend custom ops are wired up.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        logger.warning_once(_UNSUPPORTED_MESSAGE)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)
