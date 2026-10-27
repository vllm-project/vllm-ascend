# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP row sharding of DeepSeek V4.1 prefill indexer query rows.

Port of the shared DSA idea from vllm#54951 (GLM-5.3-Flash indexer row
sharding, merged upstream) and its DeepSeek-V4 Ascend sibling
vllm-ascend#18799 onto the V4.1 ratio-1/2 indexer:

- prefill query rows are independent, so each TP rank scores only its
  contiguous, work-balanced slice and still scores the FULL visible key
  history of every row it owns (exact equivalence, no approximation);
- the group then exchanges final int32 top-k indices instead of
  recomputing the whole batch on every rank. The NPU communicator has no
  ``all_gatherv``, so the exchange is a padded, fixed-shape HCCL
  all-gather trimmed back to the real per-rank row counts;
- rows are rebased to the request tail (drop the sliced prefix, keep the
  window's right end) so the request-level QLI metadata
  (``cu_seqlens_q``/``seqused_k``/``cmp_residual_k``) reproduces each
  row's true causal bound without a per-row-offset kernel interface:
  for a window ``[a, b)`` of a request with ``seq_len``/``qlen``,
  ``local_qlen = b - a`` and ``local_seq = seq_len - qlen + b`` give the
  rebased row ``i`` the mask bound ``local_seq - local_qlen + i =
  seq_len - qlen + a + i``, exactly the original row ``a + i``.

The candidate-source layer stays replicated: its published candidate
blocks must be complete on every rank for the consumer indexers, so
``select_projected`` only engages the shard when it is not the candidate
source. Decode rows (when decode requests lead the batch) are replicated
by every rank and stay outside the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend import envs

# Mirrors vllm-ascend#18799's per-rank floor: smaller batches spend more
# on the exchange than the replicated scoring they save.
MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK = 1024

_DSV41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")


@dataclass(frozen=True)
class DSV41PrefillRowShardPlan:
    """Rank-local QLI geometry, before the metadata op runs."""

    num_decode_tokens: int
    start: int  # inclusive, whole-batch token layout
    stop: int  # exclusive, whole-batch token layout
    sizes: tuple[int, ...]  # prefill rows per TP rank
    request_indices: tuple[int, ...]  # block-table rows: decode + active prefill
    local_query_lens: tuple[int, ...]  # per local request
    local_seq_lens: tuple[int, ...]  # rebased request lengths


@dataclass(frozen=True)
class AscendDSAV41PrefillRowShard:
    """One TP rank's contiguous slice of the scheduled prefill rows.

    Rows are independent, so each rank scores only its slice and the group
    exchanges the final int32 top-k indices, not the logits.
    """

    num_decode_tokens: int
    start: int
    stop: int
    sizes: tuple[int, ...]
    compress_ratio: int
    block_table: torch.Tensor
    qli_cu_seqlens_q: torch.Tensor
    qli_seqused_k: torch.Tensor
    qli_cmp_residual_k: torch.Tensor | None
    qli_metadata: torch.Tensor | None = None
    max_query_len: int = 0


def dsv41_prefill_indexer_tp_shard_supported(vllm_config) -> bool:
    """Static gate, evaluated identically on every TP rank.

    Any condition that changes the indexer's local KV view (PCP/DCP/DSA-CP)
    or fixes the row count ahead of execution (FULL graphs) disables the
    shard; so does the A5 packed-cache indexer path, which runs a different
    operator bundle. Model type gates the V4-only sibling (#18799) out.
    """
    if not envs.VLLM_ASCEND_DSV41_PREFILL_INDEXER_TP_SHARD:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    model_type = getattr(getattr(model_config, "hf_text_config", None), "model_type", "")
    if model_type not in _DSV41_MODEL_TYPES:
        return False
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size < 2:
        return False
    if parallel.prefill_context_parallel_size > 1 or parallel.decode_context_parallel_size > 1:
        return False
    compilation = vllm_config.compilation_config
    cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
    if cudagraph_mode == CUDAGraphMode.FULL:
        return False
    # Late import to avoid a cycle with the attention package.
    from vllm_ascend.ascend_config import get_ascend_config

    if get_ascend_config().enable_dsa_cp:
        return False
    return True


def balanced_prefill_row_shard(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    compress_ratio: int,
    tp_size: int,
    min_rows: int,
) -> tuple[int, ...] | None:
    """Split prefill rows into contiguous, indexer-work-balanced TP slices.

    A row's cost is the number of compressed keys it must score (its causal
    prefix), so a plain row-count split would tail-load the last rank on
    long causal prefills. Equal cuts on the prefix-sum of per-row cost keep
    rank imbalance within a few percent.
    """
    num_rows = int(query_lens_cpu.sum().item())
    if tp_size < 2 or num_rows < max(min_rows, tp_size):
        return None
    query_lens = query_lens_cpu.to(torch.int64)
    first_key = torch.repeat_interleave(seq_lens_cpu.to(torch.int64) - query_lens + 1, query_lens)
    row_in_request = torch.arange(num_rows) - torch.repeat_interleave(
        torch.cumsum(query_lens, 0) - query_lens, query_lens
    )
    cost = torch.cumsum((first_key + row_in_request) // compress_ratio, 0)
    total = int(cost[-1].item()) if cost.numel() else 0
    if total <= 0:
        return None
    targets = torch.arange(1, tp_size) * total // tp_size
    bounds = [0, *torch.searchsorted(cost, targets).tolist(), num_rows]
    # Keep >= 1 row per rank (the floor above leaves ample room); this also
    # handles repeated costs from pool compression.
    for i in range(1, tp_size):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    for i in range(tp_size - 1, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)
    return tuple(bounds[i + 1] - bounds[i] for i in range(tp_size))


def plan_dsv41_prefill_row_shard(
    *,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    is_prefilling: torch.Tensor,
    num_actual_tokens: int,
    compress_ratio: int,
    tp_size: int,
    tp_rank: int,
    min_rows: int = MIN_DSV41_PREFILL_INDEXER_TP_SHARD_ROWS_PER_RANK,
) -> DSV41PrefillRowShardPlan | None:
    """Plan this rank's slice, or decline (caller falls back to replication).

    Requires decode requests to lead the batch (vLLM's usual layout) so the
    replicated decode rows form a prefix outside the exchange. Mixed rows
    in any other order, missing CPU lengths or an inconsistent layout all
    decline silently.
    """
    if (
        is_prefilling is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return None
    flags = is_prefilling.to(torch.bool)
    num_flags = min(flags.numel(), query_start_loc_cpu.numel() - 1)
    if num_flags <= 0 or seq_lens_cpu.numel() < num_flags:
        return None
    flags = flags[:num_flags]
    query_lens_cpu = query_start_loc_cpu[1 : num_flags + 1] - query_start_loc_cpu[:num_flags]
    seq_lens_cpu = seq_lens_cpu[:num_flags]
    num_prefills = int(flags.sum().item())
    num_decodes = num_flags - num_prefills
    if num_prefills == 0:
        return None
    # Decode rows must be the batch's leading prefix.
    if not bool(torch.all(~flags[:num_decodes])) or not bool(torch.all(flags[num_decodes:])):
        return None
    num_decode_tokens = int(query_lens_cpu[:num_decodes].sum().item())
    prefill_query_lens = query_lens_cpu[num_decodes:]
    prefill_seq_lens = seq_lens_cpu[num_decodes:]
    num_prefill_tokens = int(prefill_query_lens.sum().item())
    if num_decode_tokens + num_prefill_tokens != num_actual_tokens:
        return None

    sizes = balanced_prefill_row_shard(
        prefill_seq_lens, prefill_query_lens, compress_ratio, tp_size, min_rows
    )
    if sizes is None:
        return None

    prefill_shard_start = int(sum(sizes[:tp_rank]))
    prefill_shard_stop = prefill_shard_start + sizes[tp_rank]
    request_starts = torch.cumsum(prefill_query_lens, 0) - prefill_query_lens
    local_starts = (prefill_shard_start - request_starts).clamp_min(0)
    local_stops = (prefill_shard_stop - request_starts).clamp_min(0)
    local_starts = torch.minimum(local_starts, prefill_query_lens)
    local_stops = torch.minimum(local_stops, prefill_query_lens)
    active = local_stops > local_starts
    local_prefill_query_lens = (local_stops - local_starts)[active]
    # Rebase: keep the window's right end, drop the sliced prefix, so every
    # row keeps its true causal bound (see the module docstring).
    local_prefill_seq_lens = (
        prefill_seq_lens[active] - prefill_query_lens[active] + local_stops[active]
    )
    local_rows = int(local_prefill_query_lens.sum().item())
    if local_rows != sizes[tp_rank]:
        return None

    local_query_lens = torch.cat([query_lens_cpu[:num_decodes], local_prefill_query_lens])
    local_seq_lens = torch.cat([seq_lens_cpu[:num_decodes], local_prefill_seq_lens])
    active_indices = torch.nonzero(active, as_tuple=False).flatten() + num_decodes
    request_indices = tuple(range(num_decodes)) + tuple(int(i) for i in active_indices.tolist())
    return DSV41PrefillRowShardPlan(
        num_decode_tokens=num_decode_tokens,
        start=num_decode_tokens + prefill_shard_start,
        stop=num_decode_tokens + prefill_shard_stop,
        sizes=sizes,
        request_indices=request_indices,
        local_query_lens=tuple(int(v) for v in local_query_lens.tolist()),
        local_seq_lens=tuple(int(v) for v in local_seq_lens.tolist()),
    )


def build_prefill_row_shard_tensors(
    plan: DSV41PrefillRowShardPlan,
    block_table: torch.Tensor,
    compress_ratio: int,
    qli_metadata: torch.Tensor | None = None,
) -> AscendDSAV41PrefillRowShard:
    """Materialize the plan's rank-local QLI metadata tensors on device."""
    device = block_table.device
    local_query_lens = torch.tensor(plan.local_query_lens, dtype=torch.int32, device=device)
    local_seq_lens = torch.tensor(plan.local_seq_lens, dtype=torch.int32, device=device)
    qli_cu_seqlens_q = torch.zeros(local_query_lens.numel() + 1, dtype=torch.int32, device=device)
    torch.cumsum(local_query_lens, 0, out=qli_cu_seqlens_q[1:])
    if compress_ratio == 1:
        # ratio-1 planes must not pass a residual tensor (host contract).
        qli_seqused_k = local_seq_lens
        qli_cmp_residual_k = None
    else:
        qli_seqused_k = torch.div(local_seq_lens, compress_ratio, rounding_mode="floor")
        qli_cmp_residual_k = torch.remainder(local_seq_lens, compress_ratio)
    request_rows = torch.tensor(plan.request_indices, dtype=torch.int64, device=device)
    local_block_table = block_table.index_select(0, request_rows)
    return AscendDSAV41PrefillRowShard(
        num_decode_tokens=plan.num_decode_tokens,
        start=plan.start,
        stop=plan.stop,
        sizes=plan.sizes,
        compress_ratio=compress_ratio,
        block_table=local_block_table,
        qli_cu_seqlens_q=qli_cu_seqlens_q,
        qli_seqused_k=qli_seqused_k,
        qli_cmp_residual_k=qli_cmp_residual_k,
        qli_metadata=qli_metadata,
        max_query_len=max(plan.local_query_lens) if plan.local_query_lens else 0,
    )


def pack_prefill_row_shard_rows(tensor: torch.Tensor | None, shard: AscendDSAV41PrefillRowShard):
    """Decode prefix rows + this rank's prefill rows, in batch order.

    Pure prefill batches take a zero-copy slice; mixed batches pay one cat
    over the leading decode rows.
    """
    if tensor is None:
        return None
    if shard.num_decode_tokens == 0:
        return tensor[shard.start : shard.stop]
    return torch.cat(
        [tensor[: shard.num_decode_tokens], tensor[shard.start : shard.stop]], dim=0
    )


def gather_prefill_row_shard_rows(
    local_rows: torch.Tensor,
    sizes: tuple[int, ...],
    leading: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reassemble global row order from per-rank prefill slices.

    The NPU communicator has no ``all_gatherv``, so every rank's slice is
    padded to the largest slice, one fixed-shape HCCL all-gather runs, and
    the padding is trimmed afterwards. Leading (replicated decode) rows
    stay outside the exchange.
    """
    from vllm.distributed import get_tp_group

    max_rows = max(sizes)
    if local_rows.shape[0] > max_rows:
        raise ValueError("Local prefill rows exceed the shard capacity")
    if local_rows.shape[0] < max_rows:
        padded = torch.full(
            (max_rows, *local_rows.shape[1:]), -1, dtype=local_rows.dtype, device=local_rows.device
        )
        padded[: local_rows.shape[0]].copy_(local_rows)
    else:
        padded = local_rows.contiguous()
    gathered = get_tp_group().all_gather(padded, dim=0)
    chunks = [gathered[r * max_rows : r * max_rows + size] for r, size in enumerate(sizes)]
    if leading is not None:
        chunks.insert(0, leading)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)
