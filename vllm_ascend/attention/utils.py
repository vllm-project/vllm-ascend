from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend import envs
from vllm_ascend.utils import AscendDeviceType, get_ascend_config, get_ascend_device_type


def ascend_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    chunked_prefill_workspace_size = min(
        # Make sure there is enough for 8 full length request or at least
        # 4 pages of cache per request
        max(8 * model_config.max_model_len, 4 * scheduler_config.max_num_seqs * cache_config.block_size),
        # For long-context models try not to over-allocate limiting
        # kv-cache space, limiting it to 128k tokens,
        # which would result in the workspace being:
        #   2*(576)*(128*1024) = 288mb
        # (assuming 576 MLA head dim, and fp16)
        # which would result in up-projected context being
        #   2*(192*128)*(128*1024) = 6gb
        # (assuming 192 QK head dim, 128 heads, and fp16)
        128 * 1024,
    )

    chunked_prefill_workspace_size = max(
        chunked_prefill_workspace_size,
        scheduler_config.max_num_seqs * cache_config.block_size,
    )

    return chunked_prefill_workspace_size


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
    return prefill_config.prefill_context_parallel_size > 1 or prefill_config.decode_context_parallel_size > 1


@dataclass
class AscendPrefillContextParallelMetadata:
    """
    Metadata for Prefill Context Parallelism (PCP) in CommonAttentionMetadata.

    Contains index tensors and sequence lengths for PCP operations.
    """

    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: list[list[list[int]]] | None = None

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

    # CPU tensor of sequence lengths for host-side operations.
    # E.g., tensor([128, 256, 64]) for 3 requests with different seq lengths.
    seq_lens_cpu: torch.Tensor = None

    # CPU tensor of already computed tokens count per request.
    # E.g., tensor([100, 200, 50]) means req0 has 100 tokens already computed.
    num_computed_tokens_cpu: torch.Tensor = None

    # Number of decode tokens per request, used for speculative decoding.
    # E.g., 1 for normal decoding, >1 for speculative decoding.
    decode_token_per_req: int = 1

    # Actual query sequence lengths for each token in the batch (CPU list).
    # E.g., [1, 1, 1, 128] for 3 decode tokens and 1 prefill with 128 tokens.
    actual_seq_lengths_q: list[int] = field(default_factory=list)

    # NPU tensor of position indices for rotary embeddings computation.
    # E.g., tensor([0, 1, 2, ...]) indicating token positions in sequence.
    positions: torch.Tensor = None

    # Current attention state (e.g., ChunkedPrefill, DecodeOnly).
    attn_state: Any = None

    # Padding size for graph capture, -1 means not in graph mode.
    graph_pad_size: int = -1

    # Total number of tokens including padding, used for padding operations.
    num_input_tokens: int = 0

    # Metadata for Prefill Context Parallelism (PCP) operations.
    prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata | None = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int, num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=self.seq_lens_cpu[:num_actual_reqs],
            num_computed_tokens_cpu=self.num_computed_tokens_cpu[:num_actual_reqs],
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
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            max_seq_len=self.max_seq_len,
        )


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: list[bool] | None,
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat(
        [
            torch.arange(offsets[i], offsets[i] + seq_len[i])
            for i in range(len(mask_for_non_zero_chunk))
            if mask_for_non_zero_chunk[i]
        ]
    )
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
    query_lens_pcp_full = long_seq_metadata.query_lens_pcp_full_cpu if long_seq_metadata else None
    max_query_len_pcp_full = long_seq_metadata.max_query_len_pcp_full if long_seq_metadata else 0
    max_query_len = common_attn_metadata.max_query_len if max_query_len_pcp_full == 0 else max_query_len_pcp_full
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, num_tokens, 0

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]) if query_lens_pcp_full is None else query_lens_pcp_full
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
    kv_cache_layer: list[torch.Tensor],
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
    reordered_rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
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
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def enabling_malpo(vllm_config: VllmConfig) -> bool:
    is_decode_instance = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer
    return bool(envs.VLLM_ASCEND_ENABLE_MLAPO and is_decode_instance)


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
    return query_start_loc[request_slice.start : request_slice.stop + 1] - query_start_loc[request_slice.start]


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: AscendCommonAttentionMetadata, max_num_tokens: int = 0
) -> AscendCommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], "Token slice start outside of first request"
    assert start_locs[last_req] <= last_tok < start_locs[last_req + 1], "Token slice end outside of last request"

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(attn_metadata.query_start_loc, request_slice)

    assert len(query_start_loc) >= 2, f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"

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
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item())

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    # adapt to Ascend common metadata
    num_input_tokens = token_slice.stop - token_slice.start
    positions = attn_metadata.positions[token_slice]
    attn_state = attn_metadata.attn_state
    # if attn_metadata.attn_state != AscendAttentionState.ChunkedPrefill:
    # attn_mask = attn_metadata.attn_mask

    if len(attn_metadata.actual_seq_lengths_q) > 0:
        actual_seq_lengths_q = list(
            range(attn_metadata.decode_token_per_req, max_num_tokens + 1, attn_metadata.decode_token_per_req)
        )
    else:
        actual_seq_lengths_q = []

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        num_input_tokens=num_input_tokens,
        actual_seq_lengths_q=actual_seq_lengths_q,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        positions=positions,
        attn_state=attn_state,
        max_query_len=max_query_len,
        decode_token_per_req=attn_metadata.decode_token_per_req,
        max_seq_len=max_seq_len,
        # attn_mask=attn_mask,
        # spec_attn_mask=attn_metadata.spec_attn_mask,
        graph_pad_size=attn_metadata.graph_pad_size,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: AscendCommonAttentionMetadata,
    max_num_tokens: int = 0,
) -> list[AscendCommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata, max_num_tokens))

    return results
