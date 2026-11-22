"""Prefill adapter for the CANN 9.3 BF16 LightningIndexerV2."""

import json
import logging
import os
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.models.hy_v4_flash.reference import sparse_candidates

from .li_postprocess import current_tail, expand_blocks, merge_group_rows

logger = logging.getLogger(__name__)
USE_ALPHA = os.getenv("HY4FLASH_LI_POST_ADD_ALPHA") == "1"
_post_audits_left = int(os.getenv("HY4FLASH_LI_POST_AUDIT_CHUNKS", "0"))
MODE = os.getenv("HY4FLASH_PREFILL_LI_V2", "off")
if MODE not in {"off", "audit", "on"}:
    raise ValueError(f"Unsupported HY4 Prefill LightningIndexer mode: {MODE}")
_score_rows = int(os.getenv("HY4FLASH_LI_SCORE_ROWS", "16"))
if _score_rows not in (16, 64):
    raise ValueError("HY4FLASH_LI_SCORE_ROWS must be 16 or 64")
_wide_score_rows = int(os.getenv("HY4FLASH_LI_WIDE_SCORE_ROWS", "0"))
if _wide_score_rows not in (0, 128, 256, 512):
    raise ValueError("HY4FLASH_LI_WIDE_SCORE_ROWS must be 0, 128, 256 or 512")
# Preserve the existing rollout boundary for narrower configurations and audit mode.
BATCH_ROWS = 4096 if MODE == "on" and _wide_score_rows == 512 else 512
_score_audits_left = int(os.getenv("HY4FLASH_LI_SCORE_AUDIT", "0"))
_score_audit_remainders = False
_batch_audits_left = int(os.getenv("HY4FLASH_LI_BATCH_AUDIT", "0"))
RATIO = 4
BUDGET = 511
HEADS = 16
DIM = 128
LAYOUT = "BSND"
_selections_audited = set()
_seen_dispatch = False


def eligible(query, weights, pool, visible_len, chunk, ratio, budget):
    return (
        MODE != "off"
        and ratio == RATIO
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
    # Import registers the paired ACLNN V2 operators.
    import cann_ops_transformer  # noqa: F401

    scalars = torch.tensor([used_k, remainder], dtype=torch.int32, device=device)
    lens, residual = scalars[:1], scalars[1:]
    offsets = offsets_for_stream(device, stream_handle)
    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        HEADS,
        1,
        DIM,
        BUDGET,
        seqused_k=lens,
        cmp_residual_k=residual,
        batch_size=1,
        max_seqlen_q=query_rows,
        max_seqlen_k=used_k,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=3,
        cmp_ratio=RATIO,
    )
    return lens, residual, metadata, offsets


@lru_cache(maxsize=32)
def offsets_for_stream(device, stream_handle):
    return torch.arange(RATIO, dtype=torch.int32, device=device)


def prepare_tail(positions):
    stream = torch.npu.current_stream(positions.device)
    offsets = offsets_for_stream(positions.device, stream.npu_stream)
    return current_tail(positions, offsets, RATIO, USE_ALPHA)


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


def _score_for_audit(query, weights, pool, visible, stream, step):
    return [
        _score_chunk(
            query[i : i + step],
            weights[i : i + step],
            pool,
            visible - query.shape[0] + min(query.shape[0], step + i),
            stream,
        )[0]
        for i in range(0, query.shape[0], step)
    ]


def select_many(query, weights, pool, pool_bf16, positions, first_visible, *, tail, tp_shard=False):
    """Group independent row operations while keeping each 16-query LI invocation intact."""
    global _batch_audits_left, _score_audits_left, _score_audit_remainders
    stream = torch.npu.current_stream(query.device)
    query_bf16 = query.to(torch.bfloat16).contiguous()
    weights = weights.contiguous()
    query_rows = query_bf16.shape[0]
    minimum_keys = (first_visible - 1) // RATIO
    if (
        tp_shard
        and MODE == "on"
        and BATCH_ROWS == 4096
        and query_rows >= 2048
        and query_rows % 64 == 0
        and minimum_keys >= 8192
    ):
        context = get_forward_context()
        metadata = context.attn_metadata
        batch = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
        tp = get_tp_group()
        if (
            tp.world_size == 4
            and tp.device_group is not None
            and batch is not None
            and batch.prefill_reqs
            and all(batch.prefill_reqs)
            and not _batch_audits_left
        ):
            shard_rows = query_rows // tp.world_size
            lo = shard_rows * tp.rank_in_group
            hi = lo + shard_rows
            local_ids, _, offsets = _score_chunk(
                query_bf16[lo:hi], weights[lo:hi], pool_bf16, first_visible + hi - 16, stream
            )
            group_ids = torch.empty((query_rows, BUDGET), dtype=torch.int32, device=query.device)
            dist.all_gather_into_tensor(group_ids, local_ids.contiguous(), group=tp.device_group)
            last_visible = first_visible + query_rows - 16
            used_k = (last_visible - 1) // RATIO
            selected = merge_group_rows([group_ids], used_k, offsets, tail, RATIO, USE_ALPHA)
            if _score_audits_left:
                expected = _score_for_audit(query_bf16, weights, pool_bf16, last_visible, stream, 512)
                previous = merge_group_rows(expected, used_k, offsets, tail, RATIO, USE_ALPHA)
                differs = (selected != previous).any().to(torch.int32)
                dist.all_reduce(differs, op=dist.ReduceOp.MAX, group=tp.device_group)
                if differs.item():
                    raise RuntimeError("HY4 TP-sharded LightningIndexer changed selected KV rows")
                _score_audits_left -= 1
                logger.info("HY4_LI_TP_AUDIT pid=%d rows=%d k=%d", os.getpid(), query_rows, used_k)
            return selected
    chunks = []
    lo = 0
    while lo < query.shape[0]:
        remaining = query.shape[0] - lo
        if BATCH_ROWS > 512 and remaining >= 1024 and (first_visible + lo - 1) // RATIO >= 8192:
            rows = remaining // 512 * 512
        elif _wide_score_rows and remaining >= _wide_score_rows:
            rows = _wide_score_rows
        else:
            rows = _score_rows if remaining >= _score_rows else 16
        hi = lo + rows
        # The right-down causal mask aligns the block's last query to its visible length.
        visible = first_visible + hi - 16
        groups, used_k, offsets = _score_chunk(query_bf16[lo:hi], weights[lo:hi], pool_bf16, visible, stream)
        if rows >= 64 and _score_audits_left and (_score_audits_left > 1 or used_k >= 8192):
            q, w = query_bf16[lo:hi], weights[lo:hi]
            row_tail = tail[lo:hi]
            audit_step = 64 if rows > 64 else 16
            expected = _score_for_audit(q, w, pool_bf16, visible, stream, audit_step)
            actual_rows = merge_group_rows([groups], used_k, offsets, row_tail, RATIO, USE_ALPHA)
            previous_rows = merge_group_rows(expected, used_k, offsets, row_tail, RATIO, USE_ALPHA)
            if not torch.equal(actual_rows, previous_rows):
                different = (actual_rows != previous_rows).any(dim=-1).sum().item()
                raise RuntimeError(f"LightningIndexer differs on {different}/{rows} rows at used_k={used_k}")
            if not _score_audit_remainders:
                for shift in range(RATIO):
                    limit = visible - shift
                    wide, k, off = _score_chunk(q, w, pool_bf16, limit, stream)
                    narrow = _score_for_audit(q, w, pool_bf16, limit, stream, audit_step)
                    adjusted_positions = positions[lo:hi] - shift
                    adjusted_tail = current_tail(adjusted_positions, off, RATIO, USE_ALPHA)
                    cutoff = adjusted_positions[:, None] // RATIO
                    if ((wide >= cutoff) | (wide < -1)).any().item():
                        raise RuntimeError(f"Q64 returned an invalid causal index at visible={limit}")
                    actual = merge_group_rows([wide], k, off, adjusted_tail, RATIO, USE_ALPHA)
                    expected = merge_group_rows(narrow, k, off, adjusted_tail, RATIO, USE_ALPHA)
                    if not torch.equal(actual, expected):
                        raise RuntimeError(f"Q64 LightningIndexer differs at visible={limit}")
                _score_audit_remainders = True
                logger.info("HY4_LI_Q64_REMAINDERS pid=%d values=%d", os.getpid(), RATIO * actual.numel())
            _score_audits_left -= 1
            logger.info(
                "HY4_LI_Q64_AUDIT pid=%d k=%d values=%d remaining=%d",
                os.getpid(),
                used_k,
                actual_rows.numel(),
                _score_audits_left,
            )
        chunks.append(groups)
        lo = hi
    # Each legal ID is smaller than its chunk's used_k, hence also smaller than the last used_k.
    selected = merge_group_rows(chunks, used_k, offsets, tail, RATIO, USE_ALPHA)
    if _batch_audits_left:
        previous = [
            select(
                query[i : i + 16],
                weights[i : i + 16],
                pool,
                pool_bf16,
                positions[i : i + 16],
                first_visible + i,
                tail=tail[i : i + 16],
            )
            for i in range(0, query.shape[0], 16)
        ]
        if not torch.equal(selected, torch.cat(previous)):
            raise RuntimeError("Batched LI index mismatch")
        _batch_audits_left -= 1
        logger.info(
            "HY4_LI_BATCH_AUDIT pid=%d values=%d remaining=%d", os.getpid(), selected.numel(), _batch_audits_left
        )
    return selected


def select(query, weights, pool, pool_bf16, positions, visible_len, *, tail):
    global _seen_dispatch, _post_audits_left
    group_ids, used_k, offsets = _score_chunk(
        query.to(torch.bfloat16).contiguous(),
        weights,
        pool_bf16,
        visible_len,
        torch.npu.current_stream(query.device),
    )
    ordered = group_ids.masked_fill(group_ids < 0, used_k).sort(dim=-1).values
    history = expand_blocks(ordered, offsets, RATIO, USE_ALPHA)
    history = history.masked_fill(ordered[..., None] == used_k, -1).flatten(1)
    selected = torch.cat((history, tail), dim=-1)
    if _post_audits_left:
        old_history = ordered[..., None] * RATIO + offsets
        old_history = old_history.masked_fill(ordered[..., None] == used_k, -1).flatten(1)
        old_tail = current_tail(positions, offsets, RATIO, False)
        old_selected = torch.cat((old_history, old_tail), dim=-1)
        if not torch.equal(selected, old_selected):
            raise RuntimeError("LI postprocessing index mismatch")
        _post_audits_left -= 1
        logger.info("HY4_LI_POST_AUDIT pid=%d values=%d remaining=%d", os.getpid(), selected.numel(), _post_audits_left)
    if not _seen_dispatch:
        _seen_dispatch = True
        logger.warning("HY4_PREFILL_LI_V2 active q=16 heads=%d first_used_k=%d mode=%s", HEADS, used_k, MODE)

    if MODE == "audit":
        remainder = (visible_len - 1) % RATIO
        bucket = (used_k.bit_length(), remainder)
        if bucket not in _selections_audited and len(_selections_audited) < 32:
            _selections_audited.add(bucket)
            reference = sparse_candidates(query, weights, pool, positions, RATIO, BUDGET)
            same = selected == reference
            positions32 = positions.to(torch.int32)
            invalid = (group_ids < 0) | (group_ids >= (positions32 // RATIO)[:, None])
            stats = torch.stack((same.sum(), same.all(-1).sum(), invalid.sum())).cpu().tolist()
            logger.warning(
                "HY4_PREFILL_LI_V2_AUDIT %s",
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        q=16,
                        used_k=used_k,
                        remainder=remainder,
                        matched_values=stats[0],
                        total_values=selected.numel(),
                        matched_rows=stats[1],
                        invalid_groups=stats[2],
                    ),
                    separators=(",", ":"),
                ),
            )
            if stats[2]:
                raise RuntimeError("LightningIndexerV2 returned an invalid causal group")
    return selected
