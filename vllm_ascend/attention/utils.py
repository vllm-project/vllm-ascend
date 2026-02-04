from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.utils import (AscendDeviceType, get_ascend_config,
                               get_ascend_device_type)


def using_paged_attention(runtime_shape: int, vllm_config: VllmConfig) -> bool:
    if vllm_config.speculative_config is not None:
        return False
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    from vllm.config.compilation import CUDAGraphMode
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


@lru_cache(maxsize=1)
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 \
                or prefill_config.decode_context_parallel_size > 1


@dataclass
class AscendLIMetadata:
    li_reorder_indices: torch.Tensor = None
    li_cum_query_lens: torch.Tensor = None
    li_seq_lens: torch.Tensor = None
    li_skip_request_mask: torch.Tensor = None


@dataclass
# class AscendCommonLongSequenceMetadata:
class AscendPrefillContextParallelMetadata:
    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    q_full_idx: torch.Tensor = None

    # original query_lens before pcp split
    query_lens_pcp_full_cpu: torch.Tensor = None

    # original max_query_len before pcp split
    max_query_len_pcp_full: int = 0


@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """
    seq_lens_cpu: torch.Tensor = None
    num_computed_tokens_cpu: torch.Tensor = None

    decode_token_per_req: int = 1
    """decode token number per request"""

    actual_seq_lengths_q: list[int] = field(default_factory=list)

    positions: torch.Tensor = None

    attn_state: Any = None

    graph_pad_size: int = -1

    # num_input_tokens refers to total number of tokens including
    # padding tokens. It is used to handle some padding operations.
    num_input_tokens: int = 0

    prefill_context_parallel_metadata: Optional[
        AscendPrefillContextParallelMetadata] = None

    li_skip_metadata: AscendLIMetadata | None = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int,
                 num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[:num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[:num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.
            num_computed_tokens_cpu[:num_actual_reqs],
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.
            prefill_context_parallel_metadata,
            max_seq_len=self.max_seq_len)


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: Optional[List[bool]],
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(
        mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat([
        torch.arange(offsets[i], offsets[i] + seq_len[i])
        for i in range(len(mask_for_non_zero_chunk))
        if mask_for_non_zero_chunk[i]
    ])
    return filtered_indices


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.
    While pcp > 1, query_lens is split across pcp ranks, so we pass in the
    original query_lens and max_query_len to distinguish prefills and decodes.

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
    long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = long_seq_metadata.query_lens_pcp_full_cpu \
        if long_seq_metadata else None
    max_query_len_pcp_full = long_seq_metadata.max_query_len_pcp_full \
        if long_seq_metadata else 0
    max_query_len = common_attn_metadata.max_query_len \
        if max_query_len_pcp_full == 0 else max_query_len_pcp_full
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]) \
        if query_lens_pcp_full is None else query_lens_pcp_full
    is_prefill = query_lens > decode_threshold
    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
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


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def trans_rope_weight(weight, rope_dim):
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat(
        (rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1],
             block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat,
        (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def get_sfa_skip_indices(num_comptuted_tokens, query_lens):

    # calculate num of tokens need skip for each request
    num_comptuted_tokens = num_comptuted_tokens[:len(query_lens)]
    num_skip_tokens = np.maximum(2048 - num_comptuted_tokens, 0)
    skip_query_lens = np.minimum(num_skip_tokens, query_lens)

    # calcualte query lens for non-skip part
    no_skip_query_lens = np.maximum(query_lens - skip_query_lens, 0)

    # mask the block table need copy
    li_skipped_mask = query_lens - no_skip_query_lens > 0

    # (1) combine no-skip and skip part query lens
    # (2) put skip part at the end
    # (3) for skip-part, only select request with len > 0
    # (4) cumsum of query len
    li_cum_query_lens = np.cumsum(
        np.concatenate(
            [no_skip_query_lens, skip_query_lens[skip_query_lens != 0]]))

    # (1) for no-skip part of a request:
    #  seq_len  = skip_query_len + no_skip_query_len + num_computed_tokens
    #           = query_len + num_computed_tokens
    # (2) for skip part of a request:
    #  seq_len = skip_query_len + num_computed_tokens
    # (3) for skip-part, only select request with len > 0
    # (4) combine skip and no-skip parts as seq_lens

    li_seq_lens = np.concatenate([
        query_lens + num_comptuted_tokens,
        skip_query_lens[skip_query_lens != 0] +
        num_comptuted_tokens[skip_query_lens != 0]
    ])

    # calculate li skip reorder indices
    q_cal_part_indices = np.array([], dtype=np.int32)
    q_skip_indices = np.array([], dtype=np.int32)

    seq_offset = np.cumsum(np.insert(query_lens[:-1], 0, 0))

    for req_idx in range(len(query_lens)):
        full_indices = np.arange(query_lens[req_idx])

        keep_indices = full_indices[
            full_indices >= num_skip_tokens[req_idx]] + seq_offset[req_idx]
        skip_indices = full_indices[
            full_indices < num_skip_tokens[req_idx]] + seq_offset[req_idx]

        q_cal_part_indices = np.concatenate([q_cal_part_indices, keep_indices])
        q_skip_indices = np.concatenate([q_skip_indices, skip_indices])

    indices = np.concatenate([q_cal_part_indices,
                              q_skip_indices]).astype(np.int32)

    return indices, li_cum_query_lens, li_seq_lens, li_skipped_mask


def maybe_pad_and_reorder_inputs(input_ids, positions, reorder_indices):
    input_ids_reorder = torch.index_select(input_ids, 0, reorder_indices)
    positions_reorder = torch.index_select(positions, 0, reorder_indices)

    input_actual_length = input_ids_reorder.size(0)
    input_ids_reorder_pad = torch.zeros_like(input_ids)
    input_ids_reorder_pad[:input_actual_length] = input_ids_reorder

    positions_reorder_pad = torch.zeros_like(input_ids)
    positions_reorder_pad[:input_actual_length] = positions_reorder

    input_ids = input_ids_reorder_pad
    positions = positions_reorder_pad

    return input_ids_reorder_pad, positions_reorder_pad


def hidden_states_reorder(hidden_states, reorder_indices):
    return torch.index_select(
        hidden_states, 0,
        reorder_indices.to(torch.float32).argsort().to(torch.int32))
