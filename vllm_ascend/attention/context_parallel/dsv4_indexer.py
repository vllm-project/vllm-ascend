# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch


def local_to_global_indices(
    indices: torch.Tensor,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Invert C4 compressed-page striping, preserving the invalid sentinel."""
    local_page = torch.div(indices, block_size, rounding_mode="floor")
    global_indices = (local_page * dcp_size + dcp_rank) * block_size + indices.remainder(block_size)
    return torch.where(indices >= 0, global_indices, -1)


def merge_dcp_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    dcp_size: int,
    topk: int,
) -> torch.Tensor:
    """Merge rank-major candidates, preserving their input order at equal scores.

    Each rank contributes K candidates, sufficient to recover the global TopK
    scores. Empty ranks contribute (-1, -inf).
    Local QLI may choose a different valid subset at an equal-score cutoff;
    this merge does not recover equal-scoring keys already discarded locally.
    """
    num_tokens = indices.shape[0] // dcp_size
    candidates = indices.reshape(dcp_size, num_tokens, -1).transpose(0, 1).reshape(num_tokens, -1)
    candidate_scores = scores.reshape(dcp_size, num_tokens, -1).transpose(0, 1).reshape(num_tokens, -1)
    valid = candidates >= 0
    candidate_scores = candidate_scores.masked_fill(~valid, -torch.inf)
    # All ranks see the same candidate order; no integer ID sort is needed.
    score_order = torch.argsort(candidate_scores, dim=-1, descending=True, stable=True)[..., :topk]
    return candidates.gather(-1, score_order).unsqueeze(1)


def _all_gather_dcp_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # dsa_cp imports the Indexer, so load its shared stream only at execution.
    from vllm.distributed import get_dcp_group

    from vllm_ascend.attention.context_parallel.dsa_cp import _dsv4_dcp_comm_stream
    from vllm_ascend.distributed.utils import all_gather_async

    main_stream = torch.npu.current_stream()
    comm_stream = _dsv4_dcp_comm_stream()
    ready = main_stream.record_event()
    gathered = []
    works = []
    # The existing attention Q gather was queued on this same stream before
    # Indexer work. Retaining one stream fixes collective ordering across ranks.
    with torch.npu.stream(comm_stream):
        comm_stream.wait_event(ready)
        for tensor in tensors:
            tensor.record_stream(comm_stream)
            output, work = all_gather_async(tensor, get_dcp_group())
            gathered.append(output)
            works.append(work)
        complete = comm_stream.record_event()
    for output, work in zip(gathered, works):
        output.record_stream(main_stream)
        if work is not None:
            work.wait()
    main_stream.wait_event(complete)
    return tuple(gathered)


def gather_dcp_topk(
    local_indices: torch.Tensor,
    local_scores: torch.Tensor,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    global_indices = local_to_global_indices(local_indices, block_size, dcp_size, dcp_rank)
    indices, scores = _all_gather_dcp_tensors(global_indices.contiguous(), local_scores.contiguous())
    return merge_dcp_topk(indices, scores, dcp_size, local_indices.shape[-1])


def gather_dcp_indexer_cache(
    key_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    metadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore compact global K and scale pages for causal prefill QLI."""
    block_ids = metadata.dcp_prefill_kv_gather_block_ids
    local_key = torch.index_select(key_cache, 0, block_ids).contiguous()
    local_scale = torch.index_select(scale_cache, 0, block_ids).contiguous()
    return _all_gather_dcp_tensors(local_key, local_scale)
