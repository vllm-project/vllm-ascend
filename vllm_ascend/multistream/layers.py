from typing import List, Optional, Tuple, Union

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.multistream.ms_split import find_best_split_point

from .base import MSEventKey
from .context import (get_multistream_layer_context,
                      reset_multistream_layer_context,
                      set_multistream_layer_context)
from .metadata import MultiStreamMetadata


class MultiStreamPreTransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    def forward(
        self,
        intput_tensors: List[torch.Tensor],
    ):
        attn_metadata = get_forward_context().attn_metadata
        if self.multistream_metadata is None or attn_metadata is None:
            set_multistream_layer_context(-1, None, None)
            return attn_metadata, intput_tensors
        # TODO add attn_metadata management
        do_ms, attn_metadata, intput_tensors, _ = self.multistream_metadata.split_micro_batch(
            attn_metadata, intput_tensors)
        if do_ms:
            set_multistream_layer_context(
                self.multistream_metadata.start_layer,
                self.multistream_metadata, attn_metadata)
        else:
            set_multistream_layer_context(-1, None, None)
        return attn_metadata, intput_tensors


class MultiStreamPostTransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    def forward(self,
                input_tensors: Union[List[Tuple[torch.Tensor]],
                                     List[torch.Tensor],
                                     List[List[torch.Tensor]]],
                wait_layer_index: Optional[int] = None):
        if self.multistream_metadata is None or self.multistream_metadata.ms_config is None:
            return input_tensors
        layer_index, ms_metadata, ms_attn_metadata = get_multistream_layer_context(
        )
        if layer_index >= 0:
            true_wait_layer = self.multistream_metadata.end_layer - 1 if wait_layer_index is None else wait_layer_index
            self.multistream_metadata.try_wait_event(
                true_wait_layer,
                self.multistream_metadata.ms_config.num_micro_batches - 1,
                MSEventKey.FFN_AR_FINISH)
            reset_multistream_layer_context()
        return self.multistream_metadata.merge_micro_batches(input_tensors)


class MultiStreamPreQwen3TransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    @staticmethod
    def _split_micro_batch_(split_token_index, input_tensors):
        positions, hidden_states, residual = input_tensors

        positions_pre = positions[:split_token_index].contiguous()
        positions_post = positions[split_token_index:].contiguous()

        hidden_states_pre = hidden_states[:split_token_index].contiguous()
        hidden_states_post = hidden_states[split_token_index:].contiguous()

        residual_pre = residual[:split_token_index].contiguous(
        ) if residual is not None else None
        residual_post = residual[split_token_index:].contiguous(
        ) if residual is not None else None

        return ([positions_pre,
                 positions_post], [hidden_states_pre, hidden_states_post],
                [residual_pre, residual_post])

    @staticmethod
    def _split_attn_metadata_(split_bs_point, split_token_index,
                              attn_metadata: AscendMetadata):
        num_actual_tokens_pre = split_token_index
        num_actual_tokens_post = attn_metadata.num_actual_tokens - num_actual_tokens_pre

        slot_mapping_pre = attn_metadata.slot_mapping[:split_token_index]
        slot_mapping_post = attn_metadata.slot_mapping[split_token_index:]

        seq_lens_pre = attn_metadata.seq_lens[:split_bs_point]
        seq_lens_post = attn_metadata.seq_lens[split_bs_point:]

        query_lens_pre = attn_metadata.query_lens[:split_bs_point]
        query_lens_post = attn_metadata.query_lens[split_bs_point:]

        attn_state_pre: AscendAttentionState
        attn_state_post: AscendAttentionState
        attn_mask_pre: Optional[torch.Tensor] = None
        attn_mask_post: Optional[torch.Tensor] = None

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
            attn_state_pre = attn_state_post = attn_metadata.attn_state
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            attn_mask_pre = attn_mask_post = attn_metadata.attn_mask
            attn_state_pre = attn_state_post = AscendAttentionState.DecodeOnly
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            attn_state_pre = AscendAttentionState.ChunkedPrefill
            attn_state_post = AscendAttentionState.ChunkedPrefill
            if attn_metadata.attn_mask is not None:
                attn_mask_pre = attn_metadata.attn_mask[:split_token_index, :
                                                        max(seq_lens_pre
                                                            )].contiguous()
                attn_mask_post = attn_metadata.attn_mask[
                    split_token_index:, :max(seq_lens_post)].contiguous()
        else:
            attn_state_pre = AscendAttentionState.DecodeOnly
            attn_mask_pre = None
            attn_state_post = AscendAttentionState.ChunkedPrefill
            if attn_metadata.attn_mask is not None:
                attn_mask_post = attn_metadata.attn_mask[
                    split_bs_point:, :max(seq_lens_post)].contiguous()
            else:
                attn_mask_pre = None

        block_tables_pre = attn_metadata.block_tables[:split_bs_point]
        block_tables_post = attn_metadata.block_tables[split_bs_point:]

        attn_metadata_pre = AscendMetadata(
            num_actual_tokens=num_actual_tokens_pre,
            slot_mapping=slot_mapping_pre,
            attn_state=attn_state_pre,
            attn_mask=attn_mask_pre,
            seq_lens=seq_lens_pre,
            block_tables=block_tables_pre,
            query_lens=query_lens_pre,
            seq_lens_list=attn_metadata.seq_lens_list,
            query_start_loc=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            is_only_prefill=attn_metadata.is_only_prefill,
            num_input_tokens=attn_metadata.num_input_tokens,
            enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp)

        attn_metadata_post = AscendMetadata(
            num_actual_tokens=num_actual_tokens_post,
            slot_mapping=slot_mapping_post,
            attn_state=attn_state_post,
            attn_mask=attn_mask_post,
            seq_lens=seq_lens_post,
            block_tables=block_tables_post,
            query_lens=query_lens_post,
            seq_lens_list=attn_metadata.seq_lens_list,
            query_start_loc=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            is_only_prefill=attn_metadata.is_only_prefill,
            num_input_tokens=attn_metadata.num_input_tokens,
            enable_dbo_across_dp=attn_metadata.enable_dbo_across_dp)

        return [attn_metadata_pre, attn_metadata_post]

    def forward(self, input_tensors: List[torch.Tensor]
                ):  # input_tensors = [positions, hidden_states, residual]
        attn_metadata = get_forward_context().attn_metadata
        if self.multistream_metadata is None or self.multistream_metadata.ms_config is None:
            set_multistream_layer_context(-1, None, None)
            return input_tensors

        split_bs_point, split_token_index = find_best_split_point(
            attn_metadata.query_lens,
            self.multistream_metadata.ms_config.min_total_tokens_to_split,
            self.multistream_metadata.ms_config.imbalance_ratio)

        input_tensors = self._split_micro_batch_(split_token_index,
                                                 input_tensors)
        attn_metadata = self._split_attn_metadata_(split_bs_point,
                                                   split_token_index,
                                                   attn_metadata)

        set_multistream_layer_context(self.multistream_metadata.start_layer,
                                      self.multistream_metadata, attn_metadata)

        return attn_metadata, input_tensors


class MultiStreamPostQwen3TransformerLayer(torch.nn.Module):

    def __init__(self, multistream_metadata: MultiStreamMetadata):
        super().__init__()
        self.multistream_metadata = multistream_metadata

    def forward(self,
                input_tensor: Union[List[Tuple[torch.Tensor]],
                                    List[torch.Tensor],
                                    List[List[torch.Tensor]]],
                wait_layer_index: Optional[int] = None):
        if self.multistream_metadata is None:
            return input_tensor
        layer_index, ms_metadata, ms_attn_metadata = get_multistream_layer_context(
        )
        if layer_index >= 0:
            true_wait_layer = self.multistream_metadata.end_layer - 1 if wait_layer_index is None else wait_layer_index
            if self.multistream_metadata is not None and self.multistream_metadata.ms_config is not None:
                num_micro_batches = self.multistream_metadata.ms_config.num_micro_batches
            else:
                num_micro_batches = 2
            self.multistream_metadata.try_wait_event(true_wait_layer,
                                                     num_micro_batches - 1,
                                                     MSEventKey.FFN_AR_FINISH)
        return self.multistream_metadata.merge_micro_batches(input_tensor)