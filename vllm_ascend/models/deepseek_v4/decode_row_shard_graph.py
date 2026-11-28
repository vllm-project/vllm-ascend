# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Fixed-address buffers for MRV2 V4 C4 FULL_DECODE_ONLY row sharding.

QLI metadata is refreshed outside the captured graph, matching the
MRV2 V4 metadata protocol. Query selection, QLI, masking, HCCL and inverse
permutation are graph nodes. Nothing branches on the live request count in
those nodes; the bucket plan includes padded requests.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_ascend.models.deepseek_v4.decode_row_shard import DecodeIndexerRowPlan


@dataclass(frozen=True)
class GraphDecodeIndexerState:
    plan: DecodeIndexerRowPlan
    request_ids: torch.Tensor
    query_rows: torch.Tensor
    restore_rows: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    qli_cu_seqlens_q: torch.Tensor
    qli_seqused_k: torch.Tensor
    qli_cmp_residual_k: torch.Tensor
    qli_metadata: torch.Tensor
    request_valid: torch.Tensor
    invalid_rows: torch.Tensor
    query_buffer: torch.Tensor
    weights_buffer: torch.Tensor
    query_scale_buffer: torch.Tensor
    padded_topk: torch.Tensor
    gathered_topk: torch.Tensor
    restored_topk: torch.Tensor
    max_seqlen_k: int


def allocate_graph_decode_indexer_state(
    plan: DecodeIndexerRowPlan,
    block_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    topk: int,
    max_model_len: int,
    metadata_size: int,
) -> GraphDecodeIndexerState:
    """Allocate before capture; one state is retained per builder and bucket."""
    device = block_table.device
    local_reqs = len(plan.request_ids)
    local_rows = len(plan.query_rows)
    if block_table.shape[0] != plan.num_requests:
        raise ValueError("Graph block table must span the padded request bucket")
    return GraphDecodeIndexerState(
        plan=plan,
        request_ids=torch.tensor(plan.request_ids, dtype=torch.int64, device=device),
        query_rows=torch.tensor(plan.query_rows, dtype=torch.int64, device=device),
        restore_rows=torch.tensor(plan.restore_rows, dtype=torch.int64, device=device),
        block_table=torch.zeros((local_reqs, block_table.shape[1]), dtype=block_table.dtype, device=device),
        seq_lens=torch.zeros(local_reqs, dtype=torch.int32, device=device),
        qli_cu_seqlens_q=torch.arange(local_reqs + 1, dtype=torch.int32, device=device) * plan.rows_per_request,
        qli_seqused_k=torch.zeros(local_reqs, dtype=torch.int32, device=device),
        qli_cmp_residual_k=torch.zeros(local_reqs, dtype=torch.int32, device=device),
        qli_metadata=torch.zeros(metadata_size, dtype=torch.int32, device=device),
        request_valid=torch.zeros(local_reqs, dtype=torch.bool, device=device),
        invalid_rows=torch.ones(local_rows, dtype=torch.bool, device=device),
        # This integration targets the 910B/A2 int8 query + fp16 scale ABI.
        query_buffer=torch.empty((local_rows, num_heads, head_dim), dtype=torch.int8, device=device),
        weights_buffer=torch.empty((local_rows, num_heads), dtype=torch.float16, device=device),
        query_scale_buffer=torch.empty((local_rows, num_heads), dtype=torch.float16, device=device),
        padded_topk=torch.empty((plan.max_local_rows, 1, topk), dtype=torch.int32, device=device),
        gathered_topk=torch.empty((plan.tp_size * plan.max_local_rows, 1, topk), dtype=torch.int32, device=device),
        restored_topk=torch.empty(
            (plan.num_requests * plan.rows_per_request, 1, topk), dtype=torch.int32, device=device
        ),
        max_seqlen_k=max_model_len // 4,
    )


def refresh_graph_decode_indexer_inputs(
    state: GraphDecodeIndexerState,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_actual_reqs: int,
) -> None:
    """Run during graph-external preparation, never with a captured live count."""
    plan = state.plan
    if not 0 <= num_actual_reqs <= plan.num_requests:
        raise ValueError("Live request count exceeds the captured bucket")
    if block_table.shape != (plan.num_requests, state.block_table.shape[1]):
        raise RuntimeError("Graph block-table geometry changed; rebuild the model and recapture")
    if seq_lens.shape[0] < plan.num_requests:
        raise ValueError("Sequence lengths do not span the graph request bucket")
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    torch.index_select(seq_lens, 0, state.request_ids, out=state.seq_lens)
    torch.index_select(block_table, 0, state.request_ids, out=state.block_table)
    torch.lt(state.request_ids, num_actual_reqs, out=state.request_valid)
    state.seq_lens.masked_fill_(~state.request_valid, 0)
    state.block_table.masked_fill_(~state.request_valid.unsqueeze(1), 0)
    # A zero-length capture/padding row must never publish uninitialized indices.
    valid = state.request_valid & (state.seq_lens > 0)
    state.invalid_rows.copy_((~valid).repeat_interleave(plan.rows_per_request, output_size=len(plan.query_rows)))
    torch.div(state.seq_lens, 4, rounding_mode="floor", out=state.qli_seqused_k)
    torch.remainder(state.seq_lens, 4, out=state.qli_cmp_residual_k)


def select_graph_decode_indexer_inputs(
    state: GraphDecodeIndexerState,
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Captured row selection into fixed buffers, after normal dtype adaptation."""
    total_rows = state.plan.num_requests * state.plan.rows_per_request
    for value, output in (
        (query, state.query_buffer),
        (weights, state.weights_buffer),
        (query_scale, state.query_scale_buffer),
    ):
        if value.shape[0] != total_rows or value.shape[1:] != output.shape[1:] or value.dtype != output.dtype:
            raise RuntimeError("QLI graph input shape/dtype changed; recapture with a compatible int8 layout")
        torch.index_select(value, 0, state.query_rows, out=output)
    return state.query_buffer, state.weights_buffer, state.query_scale_buffer


def gather_graph_decode_indexer_topk(
    state: GraphDecodeIndexerState,
    local_topk: torch.Tensor,
    group,
) -> torch.Tensor:
    """Captured fixed-payload HCCL and inverse permutation; no live-size slicing."""
    plan = state.plan
    if group.world_size != plan.tp_size or group.rank_in_group != plan.tp_rank:
        raise RuntimeError("Captured decode indexer plan no longer matches the TP group")
    if local_topk.shape != (len(plan.query_rows), *state.padded_topk.shape[1:]):
        raise RuntimeError("QLI graph output shape changed")
    if local_topk.dtype != state.padded_topk.dtype:
        raise RuntimeError("QLI graph output must be int32")
    state.padded_topk.fill_(-1)
    local_output = state.padded_topk[: len(plan.query_rows)]
    local_output.copy_(local_topk)
    local_output.masked_fill_(state.invalid_rows[:, None, None], -1)
    # Same HCCL primitive as the NPUCommunicator base, with an explicit
    # persistent output instead of allocating a new output at each Python call.
    dist.all_gather_into_tensor(state.gathered_topk, state.padded_topk, group=group.device_group)
    torch.index_select(state.gathered_topk, 0, state.restore_rows, out=state.restored_topk)
    return state.restored_topk
