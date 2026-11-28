# SPDX-License-Identifier: Apache-2.0
"""QSA's original-token and compressed-group coordinates under DCP."""

import torch
from vllm.logger import logger


def configure_qsa_dcp(parallel, num_kv_heads: int, compress_ratio: int) -> None:
    if parallel.decode_context_parallel_size == 1:
        return
    if compress_ratio != 4:
        raise ValueError("QSA DCP GBSA requires compression ratio 4")
    if parallel.prefill_context_parallel_size != 1:
        raise NotImplementedError("Qwen4Exp QSA DCP currently requires PCP=1")
    # A DCP group must consist only of TP ranks holding the same KV head.
    if parallel.tensor_parallel_size % (num_kv_heads * parallel.decode_context_parallel_size):
        raise ValueError("QSA DCP requires TP to be divisible by num_key_value_heads * DCP")
    interleave = parallel.cp_kv_cache_interleave_size
    if interleave == 1 and parallel._allow_auto_resolve_cp_interleave_size:
        parallel.cp_kv_cache_interleave_size = compress_ratio
        logger.info("QSA DCP selects cp_kv_cache_interleave_size=%d to preserve compression groups.", compress_ratio)
    elif interleave < compress_ratio or interleave % compress_ratio:
        raise ValueError(
            f"QSA DCP requires cp_kv_cache_interleave_size to be a positive multiple of {compress_ratio}; "
            "a compression group must have one KV owner"
        )


def qsa_dcp_indexer_inputs(
    metadata, compress_ratio: int, rows: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One query per row: interleaved local keys are not a causal query suffix."""
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    if rows is None:
        rows = metadata.num_actual_tokens
    requests = metadata.token_to_req[:rows].long()
    local_lengths = get_dcp_local_seq_lens(
        metadata.visible_blocks[:rows],
        metadata.dcp_size,
        metadata.dcp_rank,
        metadata.dcp_interleave // compress_ratio,
    )
    cu_query = torch.arange(rows + 1, dtype=torch.int32, device=local_lengths.device)
    return cu_query, local_lengths, metadata.block_table[requests].contiguous()
