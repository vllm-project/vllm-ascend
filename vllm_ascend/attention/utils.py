from dataclasses import dataclass
from typing import Any, List

import torch
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


@dataclass
class AscendCommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens_cpu: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    seq_lens: torch.Tensor
    """same to seq_lens_cpu, for compatibility with some new attn metadata
    (such as GDN)."""

    num_computed_tokens_cpu: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""

    max_query_len: int
    """Max token number of request in batch"""

    decode_token_per_req: int
    """decode token number per request"""

    block_table_tensor: torch.Tensor

    slot_mapping: torch.Tensor

    actual_seq_lengths_q: list[int]

    positions: torch.Tensor = None

    attn_mask: torch.Tensor = None

    spec_attn_mask: torch.Tensor = None

    attn_state: Any = None

    enable_dbo_across_dp: bool = False

    is_only_prefill: bool = False

    graph_pad_size: int = -1

    # NOTE: This is a temporary solution for rotary embedding in MLA
    cos: torch.Tensor = None
    sin: torch.Tensor = None


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: AscendCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[first_prefill:] > decode_threshold)
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)

def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.
    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return query_start_loc[request_slice.start: request_slice.stop + 1] -\
        query_start_loc[request_slice.start]

def _make_metadata_with_slice(
        ubatch_slice: UBatchSlice,
        attn_metadata: AscendCommonAttentionMetadata) -> AscendCommonAttentionMetadata:
    """
    This function creates a new AscendCommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), (
        f"Ubatch slice {ubatch_slice} is empty")

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], \
        "Token slice start outside of first request"
    assert start_locs[last_req] <= last_tok < start_locs[last_req+1], \
        "Token slice end outside of last request"

    # If the "middle" request has tokens in both ubatches, we have to split it.
    # If ubatch_slice is the first ubatch then we will be splitting the last
    # request. If it's the second microbatch, then we will be splitting the
    # first request
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(attn_metadata.query_start_loc,
                                             request_slice)

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, "
        f"got {len(query_start_loc)}")

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    max_seq_len = int(seq_lens_cpu.max())
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[
        request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] -
                            query_start_loc_cpu[:-1])).item())

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[request_slice]
    #TODO 这里需要重新写，decode_token_per_req是个int
    decode_token_per_req = attn_metadata.decode_token_per_req

    slot_mapping = attn_metadata.slot_mapping[token_slice]
    positions = attn_metadata.positions[token_slice].clone()  # 令牌级（位置编码）
    # TODO attn_metadata.attn_mask是nonetype，不能这么搞
    attn_mask = attn_metadata.attn_mask
    # TODO attn_metadata.spec_attn_mask是nonetype
    spec_attn_mask = attn_metadata.spec_attn_mask

    enable_dbo_across_dp = attn_metadata.enable_dbo_across_dp
    is_only_prefill = attn_metadata.is_only_prefill
    graph_pad_size = attn_metadata.graph_pad_size
    cos = attn_metadata.cos
    sin = attn_metadata.sin
    attn_state = attn_metadata.attn_state

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        actual_seq_lengths_q=actual_seq_lengths_q,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        positions=positions,
        attn_mask=attn_mask,
        spec_attn_mask=spec_attn_mask,
        attn_state=attn_state,
        enable_dbo_across_dp=enable_dbo_across_dp,
        is_only_prefill=is_only_prefill,
        max_query_len=max_query_len,
        graph_pad_size=graph_pad_size,
        decode_token_per_req=decode_token_per_req,
        cos=cos,
        sin=sin,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: AscendCommonAttentionMetadata,
) -> list[AscendCommonAttentionMetadata]:
    """
    Creates a new AscendCommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.
    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(
            _make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
