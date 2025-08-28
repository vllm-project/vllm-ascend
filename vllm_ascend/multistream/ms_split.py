from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import torch

from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)

from .base import MSAttentionMetadataSplitConfig


def compute_split_seq_index(
    query_lens: Optional[list[int]],
    attn_state: AscendAttentionState,
    num_tokens: int,
    imbalance_ratio: float = 0.1,
) -> list[int]:
    if attn_state != AscendAttentionState.DecodeOnly:
        assert query_lens is not None
        total_tokens = sum(query_lens)
        # the first index in last split
        tokens, split_index = 0, 0
        for value in query_lens:
            tokens += value
            split_index += 1
            if tokens >= total_tokens // 2:
                # check the current split index
                if abs(tokens -
                       total_tokens // 2) < total_tokens * imbalance_ratio:
                    return [tokens, split_index]
                # check the previous split index
                elif abs(tokens - total_tokens // 2 -
                         value) < total_tokens * imbalance_ratio:
                    return [tokens - value, split_index - 1]
                # fail to split if it is imbalanced
                # TODO: split tokens in seq
                else:
                    return [0, 0]
    else:
        tokens = num_tokens // 2
        return [tokens, tokens]
    return [0, 0]


def split_attn_tensor_type(
    input_tensor: torch.Tensor,
    index: int,
) -> List[torch.Tensor]:
    return [input_tensor[:index], input_tensor[index:]]


def split_attn_int_type(
    var: int,
    index: int,
) -> List[torch.Tensor]:
    return [min(var, index), max(var - index, 0)]


def model_input_split_v1_mla_attn(
    attn_metadata,
    _metadata_cls,
    ms_split_config: MSAttentionMetadataSplitConfig,
) -> List[Any]:
    assert 0 < ms_split_config.num_micro_batches < 3
    if attn_metadata is None:
        return [attn_metadata]
    [token_index,
     seq_index] = compute_split_seq_index(attn_metadata.query_lens,
                                          attn_metadata.attn_state,
                                          attn_metadata.num_decode_tokens)
    if token_index == 0 or seq_index == 0 or seq_index == len(
            attn_metadata.query_lens):
        return [attn_metadata]

    query_start_loc_cpu: Any = np.zeros(shape=(len(attn_metadata.query_lens) +
                                               1, ),
                                        dtype=int)
    np.cumsum(attn_metadata.query_lens, out=query_start_loc_cpu[1:])
    if attn_metadata.num_prefills > 0:
        prefill_query_start_loc: Any = np.zeros(
            shape=(len(attn_metadata.prefill.query_lens) + 1, ), dtype=int)
        np.cumsum(attn_metadata.prefill.query_lens,
                  out=prefill_query_start_loc[1:])

    # split attn metadata
    [slot_mapping_pre,
     slot_mapping_post] = split_attn_tensor_type(attn_metadata.slot_mapping,
                                                 token_index)
    [num_decodes_pre,
     num_decodes_post] = split_attn_int_type(attn_metadata.num_decodes,
                                             seq_index)
    [num_decode_tokens_pre, num_decode_tokens_post
     ] = split_attn_int_type(attn_metadata.num_decode_tokens, token_index)
    [num_prefills_pre, num_prefills_post
     ] = split_attn_int_type(attn_metadata.num_prefills,
                             max(0, seq_index - attn_metadata.num_decodes))
    seq_lens = attn_metadata.prefill.seq_lens if attn_metadata.num_prefills > 0 else attn_metadata.decode.seq_lens
    [seq_lens_pre, seq_lens_post
     ] = split_attn_tensor_type(seq_lens,
                                max(0, seq_index - attn_metadata.num_decodes))

    query_start_loc_pre = query_start_loc_post = None
    if attn_metadata.query_start_loc is not None:
        query_start_loc_pre = attn_metadata.query_start_loc[:seq_index + 1]
        query_start_loc_post = deepcopy(
            attn_metadata.query_start_loc[seq_index:]
        ) - attn_metadata.query_start_loc[seq_index]
    [block_table_pre,
     block_table_post] = split_attn_tensor_type(attn_metadata.block_tables,
                                                seq_index)

    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
        # the attn_mla kernel in torch npu only accept 128*128 attn mask
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = attn_metadata.attn_state
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # should be none in decode only state
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = AscendAttentionState.DecodeOnly
    else:
        # chunked prefill
        if num_prefills_pre > 0:
            attn_state_pre = attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_mask_pre = None
            attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_mask_post = None
        else:
            attn_state_pre = AscendAttentionState.DecodeOnly
            attn_mask_pre = None
            attn_state_post = AscendAttentionState.ChunkedPrefill
            attn_mask_post = None
    from vllm_ascend.attention.mla_v1 import (AscendMLADecodeMetadata,
                                              AscendMLAPrefillMetadata)
    if num_prefills_pre > 0:
        # split metadata.prefill
        [input_positions_pre, input_positions_post] = split_attn_tensor_type(
            attn_metadata.prefill.input_positions,
            token_index - attn_metadata.num_decode_tokens)
        [block_tables_pre, block_tables_post
         ] = split_attn_tensor_type(attn_metadata.prefill.block_table,
                                    seq_index - attn_metadata.num_decodes)
        [prefill_query_lens_pre, prefill_query_lens_post
         ] = split_attn_tensor_type(attn_metadata.prefill.query_lens,
                                    seq_index - attn_metadata.num_decodes)
        prefill_query_start_loc_pre = attn_metadata.prefill.query_start_loc[:
                                                                            seq_index
                                                                            +
                                                                            1 -
                                                                            attn_metadata
                                                                            .
                                                                            num_decodes]
        prefill_query_start_loc_post = deepcopy(
            attn_metadata.prefill.query_start_loc[seq_index -
                                                  attn_metadata.num_decodes:]
        ) - attn_metadata.prefill.query_start_loc[seq_index -
                                                  attn_metadata.num_decodes]
        context_len_pre = seq_lens_pre
        context_len_post = seq_lens_post
        prefill_max_query_len_pre = max(prefill_query_lens_pre)
        prefill_max_query_len_post = max(prefill_query_lens_post)
        [cos_pre, cos_post] = split_attn_tensor_type(
            attn_metadata.prefill.cos,
            token_index - attn_metadata.num_decode_tokens)
        [sin_pre, sin_post] = split_attn_tensor_type(
            attn_metadata.prefill.sin,
            token_index - attn_metadata.num_decode_tokens)
        prefill_pre = AscendMLAPrefillMetadata(
            attn_mask=attn_metadata.prefill.attn_mask,
            query_lens=prefill_query_lens_pre,
            seq_lens=seq_lens_pre,
            query_start_loc=prefill_query_start_loc_pre,
            input_positions=input_positions_pre,
            context_lens=context_len_pre,
            block_table=block_tables_pre,
            max_query_len=prefill_max_query_len_pre,
            max_seq_lens=context_len_pre.max().item(),
            cos=cos_pre,
            sin=sin_pre)
        prefill_post = AscendMLAPrefillMetadata(
            attn_mask=attn_metadata.prefill.attn_mask,
            query_lens=prefill_query_lens_post,
            seq_lens=seq_lens_post,
            query_start_loc=prefill_query_start_loc_post,
            input_positions=input_positions_post,
            context_lens=context_len_post,
            block_table=block_tables_post,
            max_query_len=prefill_max_query_len_post,
            max_seq_lens=context_len_post.max().item(),
            cos=cos_post,
            sin=sin_post)
        decode_pre = attn_metadata.decode
        decode_post = None
    else:
        # prefill is None, split metadata.decode
        [input_positions_pre, input_positions_post
         ] = split_attn_tensor_type(attn_metadata.decode.input_positions,
                                    token_index)
        [block_tables_pre, block_tables_post
         ] = split_attn_tensor_type(attn_metadata.decode.block_table,
                                    seq_index)
        [decode_seq_lens_pre,
         decode_seq_lens_post] = split_attn_tensor_type(seq_lens, seq_index)
        decode_pre = AscendMLADecodeMetadata(
            input_positions=input_positions_pre,
            block_table=block_tables_pre,
            seq_lens=decode_seq_lens_pre,
            max_seq_lens=max(decode_seq_lens_pre),
            seq_lens_list=decode_seq_lens_pre.tolist(),
        )
        decode_post = AscendMLADecodeMetadata(
            input_positions=input_positions_post,
            block_table=block_tables_post,
            seq_lens=decode_seq_lens_post,
            max_seq_lens=max(decode_seq_lens_post),
            seq_lens_list=decode_seq_lens_post.tolist(),
        )
        prefill_pre = None
        prefill_post = attn_metadata.prefill
    # construct metadata
    from vllm_ascend.attention.mla_v1 import AscendMLAPrefillMetadata
    attention_metadata_pre = _metadata_cls(
        num_actual_tokens=token_index,
        num_input_tokens=token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_pre,
        seq_lens=seq_lens_pre,
        query_start_loc=query_start_loc_pre,
        block_tables=block_table_pre,
        num_decodes=num_decodes_pre,
        num_prefills=num_prefills_pre,
        num_decode_tokens=num_decode_tokens_pre,
        attn_state=attn_state_pre,
        attn_mask=attn_mask_pre,
        prefill=prefill_pre,
        decode=decode_pre,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    attention_metadata_post = _metadata_cls(
        num_actual_tokens=attn_metadata.num_actual_tokens - token_index,
        num_input_tokens=attn_metadata.num_input_tokens - token_index,
        head_dim=attn_metadata.head_dim,
        slot_mapping=slot_mapping_post,
        seq_lens=seq_lens_post,
        query_start_loc=query_start_loc_post,
        block_tables=block_table_post,
        num_decodes=num_decodes_post,
        num_prefills=num_prefills_post,
        num_decode_tokens=num_decode_tokens_post,
        attn_mask=attn_mask_post,
        attn_state=attn_state_post,
        prefill=prefill_post,
        decode=decode_post,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )
    return [attention_metadata_pre, attention_metadata_post]


def model_input_split_v1_attn(
    attn_metadata: AscendMetadata,
    _metadata_cls,
    ms_split_config: MSAttentionMetadataSplitConfig,
) -> List[Any]:
    assert 0 < ms_split_config.num_micro_batches < 3
    if attn_metadata is None:
        return [attn_metadata]
    [token_index,
     seq_index] = compute_split_seq_index(attn_metadata.query_lens,
                                          attn_metadata.attn_state,
                                          attn_metadata.num_actual_tokens)
    if token_index == 0 or seq_index == 0 or seq_index == len(
            attn_metadata.query_lens):
        return [attn_metadata]

    # split attn metadata

    [block_table_pre,
     block_table_post] = split_attn_tensor_type(attn_metadata.block_tables,
                                                seq_index)

    query_start_loc_pre = query_start_loc_post = None
    if attn_metadata.query_start_loc is not None:
        query_start_loc_pre = attn_metadata.query_start_loc[:seq_index + 1]
        query_start_loc_post = deepcopy(
            attn_metadata.query_start_loc[seq_index:]
        ) - attn_metadata.query_start_loc[seq_index]

    [query_lens_pre,
     query_lens_post] = split_attn_tensor_type(attn_metadata.query_lens,
                                               seq_index)
    [seq_lens_pre,
     seq_lens_post] = split_attn_tensor_type(attn_metadata.seq_lens, seq_index)

    max_query_len_pre = max_query_len_post = None
    if attn_metadata.max_query_len is not None:
        max_query_len_pre, max_query_len_post = max(query_lens_pre), max(
            query_lens_post)

    [slot_mapping_pre,
     slot_mapping_post] = split_attn_tensor_type(attn_metadata.slot_mapping,
                                                 token_index)

    is_only_prefill_pre = is_only_prefill_post = attn_metadata.is_only_prefill
    has_prefill_pre, _ = torch.any(query_lens_pre > 1).item(), torch.any(
        query_lens_post > 1).item()

    if not attn_metadata.is_only_prefill:
        is_only_prefill_post = torch.all(query_lens_post > 1).item()

    if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
        # the attn_mla kernel in torch npu only accept 128*128 attn mask
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = attn_metadata.attn_state
    elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        # should be none in decode only state
        attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
        attn_state_pre = attn_state_post = AscendAttentionState.DecodeOnly  # type: ignore
    else:
        # chunked prefill
        assert attn_metadata.attn_mask is not None
        if has_prefill_pre:
            attn_state_pre = attn_state_post = AscendAttentionState.ChunkedPrefill  # type: ignore
            attn_mask_pre = attn_metadata.attn_mask[:token_index, :max(
                seq_lens_pre)].contiguous()
            attn_state_post = AscendAttentionState.ChunkedPrefill  # type: ignore
            attn_mask_post = attn_metadata.attn_mask[
                token_index:, :max(seq_lens_post)].contiguous()
        else:
            attn_state_pre = AscendAttentionState.DecodeOnly  # type: ignore
            attn_mask_pre = None
            attn_state_post = AscendAttentionState.ChunkedPrefill  # type: ignore
            attn_mask_post = attn_metadata.attn_mask[
                token_index:, :max(seq_lens_post)].contiguous()

    # construct metadata
    attention_metadata_pre = _metadata_cls(
        num_actual_tokens=token_index,
        block_tables=block_table_pre,
        query_start_loc=query_start_loc_pre,
        query_lens=query_lens_pre,
        seq_lens=seq_lens_pre,
        seq_lens_list=seq_lens_pre.tolist(),
        max_query_len=max_query_len_pre,
        slot_mapping=slot_mapping_pre,
        is_only_prefill=is_only_prefill_pre,
        attn_state=attn_state_pre,
        attn_mask=attn_mask_pre,
        num_input_tokens=token_index,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )

    attention_metadata_post = _metadata_cls(
        num_actual_tokens=attn_metadata.num_actual_tokens - token_index,
        block_tables=block_table_post,
        query_start_loc=query_start_loc_post,
        query_lens=query_lens_post,
        seq_lens=seq_lens_post,
        seq_lens_list=seq_lens_post.tolist(),
        max_query_len=max_query_len_post,
        slot_mapping=slot_mapping_post,
        is_only_prefill=is_only_prefill_post,
        attn_state=attn_state_post,
        attn_mask=attn_mask_post,
        num_input_tokens=attn_metadata.num_input_tokens - token_index,
        enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp,
    )

    return [attention_metadata_pre, attention_metadata_post]
