"""Prefill adapter for the CANN 9.3 BF16 LightningIndexerV2."""
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from .li_postprocess import current_tail, merge_group_rows

BATCH_ROWS = 4096
SCORE_ROWS = 64
WIDE_SCORE_ROWS = 512
RATIO = 4
BUDGET = 511
HEADS = 16
DIM = 128
LAYOUT = "BSND"


def eligible(query, weights, pool, visible_len, chunk, ratio, budget):
    return (
        ratio == RATIO
        and budget == BUDGET
        and chunk == 16
        and visible_len - chunk >= (BUDGET + 1) * RATIO
        and tuple(query.shape) == (chunk, HEADS, DIM)
        and weights.dtype == torch.float32
        and weights.shape == (chunk, HEADS)
        and pool.ndim == 2
        and pool.shape[1] == DIM
        and 0 < (visible_len - 1) // RATIO <= pool.shape[0]
    )


@lru_cache(maxsize=2048)
def metadata_for_shape(device, stream_handle, query_rows, used_k, remainder):
    import cann_ops_transformer  # registers the paired ACLNN V2 operators

    scalars = torch.tensor([used_k, remainder], dtype=torch.int32, device=device)
    lens, residual = scalars[:1], scalars[1:]
    offsets = offsets_for_stream(device, stream_handle)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        HEADS, 1, DIM, BUDGET, seqused_k=lens, cmp_residual_k=residual,
        batch_size=1, max_seqlen_q=query_rows, max_seqlen_k=used_k,
        layout_q=LAYOUT, layout_k=LAYOUT, mask_mode=3, cmp_ratio=RATIO,
    )
    return lens, residual, metadata, offsets


@lru_cache(maxsize=32)
def offsets_for_stream(device, stream_handle):
    return torch.arange(RATIO, dtype=torch.int32, device=device)


def prepare_tail(positions):
    stream = torch.npu.current_stream(positions.device)
    offsets = offsets_for_stream(positions.device, stream.npu_stream)
    return current_tail(positions, offsets, RATIO)


def _score_chunk(query_bf16, weights, pool_bf16, visible_len, stream):
    query_rows = query_bf16.shape[0]
    used_k, remainder = divmod(visible_len - 1, RATIO)
    lens, residual, metadata, offsets = metadata_for_shape(
        query_bf16.device, stream.npu_stream, query_rows, used_k, remainder
    )
    group_ids, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query_bf16.unsqueeze(0),
        pool_bf16[:used_k].view(1, used_k, 1, DIM),
        weights.contiguous().unsqueeze(0),
        BUDGET,
        seqused_k=lens,
        cmp_residual_k=residual,
        metadata=metadata,
        max_seqlen_q=query_rows,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=3,
        cmp_ratio=RATIO,
        return_value=0,
    )
    return group_ids.view(query_rows, BUDGET), used_k, offsets


def select_many(query, weights, pool_bf16, first_visible, *, tail, tp_shard=False):
    """Score eligible Prefill row batches against their shared key pool."""
    stream = torch.npu.current_stream(query.device)
    query_bf16 = query.to(torch.bfloat16).contiguous()
    weights = weights.contiguous()
    query_rows = query_bf16.shape[0]
    minimum_keys = (first_visible - 1) // RATIO
    if (tp_shard and query_rows >= 2048
            and query_rows % 64 == 0 and minimum_keys >= 8192):
        context = get_forward_context()
        metadata = context.attn_metadata
        batch = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
        tp = get_tp_group()
        if (tp.world_size == 4 and tp.device_group is not None and batch is not None
                and batch.prefill_reqs and all(batch.prefill_reqs)):
            shard_rows = query_rows // tp.world_size
            lo = shard_rows * tp.rank_in_group
            hi = lo + shard_rows
            local_ids, _, offsets = _score_chunk(query_bf16[lo:hi], weights[lo:hi], pool_bf16,
                                                first_visible + hi - 16, stream)
            group_ids = torch.empty((query_rows, BUDGET), dtype=torch.int32, device=query.device)
            dist.all_gather_into_tensor(group_ids, local_ids.contiguous(), group=tp.device_group)
            last_visible = first_visible + query_rows - 16
            used_k = (last_visible - 1) // RATIO
            return merge_group_rows(group_ids, used_k, offsets, tail, RATIO)
    chunks = []
    lo = 0
    while lo < query.shape[0]:
        remaining = query.shape[0] - lo
        if remaining >= 1024 and (first_visible + lo - 1) // RATIO >= 8192:
            rows = remaining // WIDE_SCORE_ROWS * WIDE_SCORE_ROWS
        elif remaining >= WIDE_SCORE_ROWS:
            rows = WIDE_SCORE_ROWS
        else:
            rows = SCORE_ROWS if remaining >= SCORE_ROWS else 16
        hi = lo + rows
        # The right-down causal mask aligns the block's last query to its visible length.
        visible = first_visible + hi - 16
        groups, used_k, offsets = _score_chunk(
            query_bf16[lo:hi], weights[lo:hi], pool_bf16, visible, stream
        )
        chunks.append(groups)
        lo = hi
    # Each legal ID is smaller than its chunk's used_k, hence also smaller than the last used_k.
    return merge_group_rows(torch.cat(chunks, dim=0), used_k, offsets, tail, RATIO)


def select(query, weights, pool_bf16, visible_len, *, tail):
    group_ids, used_k, offsets = _score_chunk(
        query.to(torch.bfloat16).contiguous(), weights, pool_bf16, visible_len,
        torch.npu.current_stream(query.device),
    )
    return merge_group_rows(group_ids, used_k, offsets, tail, RATIO)
