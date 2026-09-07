# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer layer for the GLM-5.3-Flash kpool indexer.

Upstream calls this op from the model's ``Indexer.forward``, where it scores and
selects on CUDA through a set of fused kernels: block-FP8 MQA logits via
DeepGEMM, paged MQA logits, and radix top-k over a device workspace.

Ascend does it a layer lower instead. ``AscendKpoolMLAImpl`` writes the pooled
keys, scores them with ``npu_lightning_indexer`` and attends over the winners
all inside the attention layer, which is what lets the selection see the keys
this very step just wrote. So on Ascend this op is never dispatched; the class
stays because the layer still owns the index-K and tail caches, whose KV cache
specs and PD transfer are wired through it.
"""

import torch
from vllm.model_executor.custom_op import CustomOp

_UNSUPPORTED_MESSAGE = (
    "This layer is not the Ascend kpool indexer. The scoring and selection live in "
    "AscendKpoolMLAImpl, which runs them inside the attention layer so the pooled "
    "keys it just wrote are the ones it scores. Reaching this op means the model was "
    "routed to a backend that does not do that."
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
