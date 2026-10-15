# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.spec_decode.qwen4_exp import Qwen4ExpMTPProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)


class AscendQwen4ExpMTPProposer(
    Qwen4ExpMTPProposer,
    AscendSpecDecodeBaseProposer,
):
    """Ascend proposer for Qwen4Exp's multi-stream MTP draft."""

    hidden_size: int

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            True,
            runner=runner,
        )
        qwen_hidden_size = self._get_hidden_size()
        if self.hidden_size != qwen_hidden_size:
            self.hidden_size = qwen_hidden_size
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=device,
            )
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def _get_hidden_size(self) -> int:
        text_config = self.draft_model_config.hf_text_config
        return int(text_config.hidden_size * text_config.hc_count)

    def build_draft_attn_metadata(
        self,
        common_attn_metadata,
        num_input_tokens,
        num_actual_tokens,
    ):
        """Build step-0 metadata with each Qwen cache group's block table.

        Ascend's merged MTP path bypasses upstream ``_propose`` and therefore
        does not call Qwen4ExpMTPProposer.build_per_group_and_layer_attn_metadata
        unless it is wired in here.  Without this override the raw and
        compressed QSA builders receive the main-attention block table.
        """
        del num_input_tokens, num_actual_tokens
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata,
            draft_index=0,
        )
        primary_group = self._get_primary_draft_attn_group()
        primary_metadata = per_layer_attn_metadata[primary_group.layer_names[0]]
        return [per_layer_attn_metadata], primary_metadata

    def _build_cache_only_group_next_step_attn_metadata(
        self,
        common_attn_metadata,
        draft_index,
        num_input_tokens,
        primary_group,
        primary_metadata,
        cache_only_groups,
    ):
        """Rebind later MTP draft steps to every QSA cache group's table."""
        del num_input_tokens
        per_layer_attn_metadata = {
            layer_name: primary_metadata for layer_name in primary_group.layer_names
        }
        for attn_group in cache_only_groups:
            gid = attn_group.kv_cache_group_id
            group_common = common_attn_metadata
            if gid != self.kv_cache_gid:
                block_table = self._per_group_block_tables.get(gid)
                assert block_table is not None, (
                    f"Missing Qwen draft block table for KV cache group {gid}"
                )
                group_common = self.shallow_copy_metadata(common_attn_metadata)
                group_common.block_table_tensor = block_table[: group_common.num_reqs]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                group_common,
                draft_index,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def _get_draft_slot_block_size(self) -> int:
        """Return the kernel block size of Qwen's main draft cache group.

        Qwen3.8 Next is a hybrid GDN/QSA model.  ``has_gdn`` describes the
        target as a whole, while the MTP draft attention writes the QSA main
        cache, whose per-group kernel block size is stored in ``block_size``.
        """
        return self.block_size

    def maybe_pad_and_reduce(self, hidden_states, positions):
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states,
        positions,
        hidden_states=None,
    ):
        return last_hidden_states, positions, hidden_states
