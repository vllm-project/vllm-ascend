from dataclasses import dataclass

from vllm.config import SpeculativeConfig
from vllm_ascend.attention.attention_v1 import AscendAttentionState

import torch


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

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""

    actual_seq_lengths_q: list[int] = None

    block_table_tensor: torch.Tensor
    slot_mapping_cpu: torch.Tensor

    positions: torch.Tensor = None

    attn_mask: torch.Tensor = None
    spec_attn_mask: torch.Tensor = None
    attn_state: AscendAttentionState = None

    decode_token_per_req: int

    max_num_blocks_per_req: int

    enable_dbo_across_dp: bool = False

    is_only_prefill: bool = False

    graph_pad_size: int = -1

@dataclass
class TorchairCommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""

    actual_seq_lengths_q: list[int] = None

    attn_mask: torch.Tensor = None
    spec_attn_mask: torch.Tensor = None

    graph_pad_size: int = -1


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


def get_decode_token_per_req(speculative_config: SpeculativeConfig):
    decode_token_per_req = 1
    if not speculative_config:
        return decode_token_per_req
    spec_token_num = speculative_config.num_speculative_tokens
    assert spec_token_num > 0
    return decode_token_per_req + spec_token_num
