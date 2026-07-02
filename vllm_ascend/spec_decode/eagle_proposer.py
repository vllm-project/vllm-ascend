# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.triton.spec_decode.utils import _multi_layer_eagle_shift_and_cache
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

if TYPE_CHECKING:
    from vllm.v1.spec_decode.metadata import MultiLayerEagleMetadata
else:
    try:
        from vllm.v1.spec_decode.metadata import MultiLayerEagleMetadata
    except ImportError:

        @dataclass
        class MultiLayerEagleMetadata:
            """Fallback when vllm does not yet provide MultiLayerEagleMetadata."""

            cached_len: torch.Tensor
            cached_token_ids: torch.Tensor
            cached_positions: torch.Tensor
            cached_hidden_states: torch.Tensor
            cached_slot_mappings: torch.Tensor

            @staticmethod
            def make_dummy(
                layer_num: int,
                hidden_size: int,
                device: torch.device,
            ) -> MultiLayerEagleMetadata:
                return MultiLayerEagleMetadata(
                    cached_len=torch.zeros(1, dtype=torch.int32, device=device),
                    cached_token_ids=torch.zeros(1, layer_num, dtype=torch.int32, device=device),
                    cached_positions=torch.zeros(1, layer_num, dtype=torch.int32, device=device),
                    cached_hidden_states=torch.zeros(1, layer_num, hidden_size, dtype=torch.float16, device=device),
                    cached_slot_mappings=torch.zeros(1, layer_num, dtype=torch.int64, device=device),
                )


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device, pass_hidden_states_to_model=True, runner=runner
        )


class MultiLayerEagleProposer(AscendSpecDecodeBaseProposer):
    """Proposer for multi-layer Eagle/MTP models.

    This proposer uses multiple draft model layers to generate draft tokens
    in a single forward pass. It introduces an ``adjust_input`` method that
    leverages Triton-based shift-and-gather kernels to efficiently prepare
    inputs for each layer.
    """

    _runnable: ACLGraphWrapper

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        super().__init__(vllm_config, device, pass_hidden_states_to_model, runner)

        self.layer_num: int = getattr(
            self.speculative_config.draft_model_config.hf_text_config,
            "n_predict",
            0,
        )
        self.num_speculative_tokens: int = self.speculative_config.num_speculative_tokens

    def adjust_input(
        self,
        batch_size: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        multi_layer_eagle_metadata: MultiLayerEagleMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        assert multi_layer_eagle_metadata is not None
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        MAX_SHIFT = self.layer_num
        assert MAX_SHIFT > 0

        prev_token_ids = target_token_ids.clone()
        prev_positions = target_positions.clone()
        prev_hidden_states = target_hidden_states.clone()
        slot_mapping = common_attn_metadata.slot_mapping

        start_token_indices = common_attn_metadata.query_start_loc[:-1]
        end_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        pos_for_shift = target_positions[0] if target_positions.dim() == 2 else target_positions
        start_token_pos = pos_for_shift[start_token_indices]

        shift = torch.minimum(
            end_token_indices - token_indices_to_sample,
            start_token_pos,
        )
        shift = torch.clamp(shift, min=0)

        # Metadata updates
        token_indices_to_sample.add_(shift)
        common_attn_metadata.seq_lens.sub_(shift)

        # NOTE: avoid device sync by copying from GPU
        common_attn_metadata.seq_lens_cpu.copy_(common_attn_metadata.seq_lens, non_blocking=True)

        cached_lens = multi_layer_eagle_metadata.cached_len
        shift = torch.minimum(shift, cached_lens)

        _multi_layer_eagle_shift_and_cache(
            batch_size=batch_size,
            max_shift=MAX_SHIFT,
            src_token_ids=target_token_ids,
            dst_token_ids=prev_token_ids,
            src_positions=target_positions,
            dst_positions=prev_positions,
            src_hidden_states=target_hidden_states,
            dst_hidden_states=prev_hidden_states,
            src_slot_mapping=slot_mapping,
            dst_slot_mapping=slot_mapping,
            start_token_indices=start_token_indices,
            end_token_indices=end_token_indices,
            token_indices_to_sample=token_indices_to_sample,
            shift=shift,
            cached_lens=cached_lens,
            cached_prev_token_ids=multi_layer_eagle_metadata.cached_token_ids,
            cached_prev_positions=multi_layer_eagle_metadata.cached_positions,
            cached_prev_hidden_states=multi_layer_eagle_metadata.cached_hidden_states,
            cached_slot_mappings=multi_layer_eagle_metadata.cached_slot_mappings,
            common_attn_metadata=common_attn_metadata,
        )

        return prev_token_ids, prev_positions, prev_hidden_states, common_attn_metadata

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ):
        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)

        multi_steps_attn_metadata = []

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
            self.query_start_loc.cpu[: num_reqs + 1].copy_(self.runner.query_start_loc.cpu[: num_reqs + 1])
            self.query_start_loc.copy_to_gpu()

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                seq_lens_cpu=self.runner.seq_lens.cpu,
                seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_input_tokens,
                num_input_tokens=num_input_tokens,
                max_query_len=self.num_speculative_tokens + 1,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                positions=self.runner.positions.gpu,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
                max_seq_len=0,
            )
            if self.pcp_size * self.dcp_size > 1:
                common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata
                common_attn_metadata.block_table_tensor = self.runner.input_batch.block_table[0].get_device_tensor()[
                    : num_reqs * self.decode_threshold
                ]

            from vllm_ascend.attention.attention_v1 import AscendAttentionState

            builder = self.runner.attn_groups[0][0].get_metadata_builder()
            for draft_step in range(self.num_speculative_tokens):
                common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
                common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
                attn_metadata_eagle = builder.build_for_graph_capture(
                    common_attn_metadata,
                    AscendAttentionState.SpecDecoding
                    if self.method in ("mtp", "mtp3")
                    else AscendAttentionState.ChunkedPrefill,
                )
                per_layer_attn_metadata = {}
                for layer_name in self.attn_layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata_eagle
                multi_steps_attn_metadata.append(per_layer_attn_metadata)

        model_positions = self._get_positions(num_input_tokens)
        batch_size = max(num_input_tokens // (self.num_speculative_tokens + 1), 1)
        if is_profile:
            batch_size = min(batch_size, self.runner.max_num_reqs)

        adjust_input_kwargs = {
            "batch_size": batch_size,
            "target_token_ids": self.input_ids[:num_input_tokens],
            "target_positions": self._get_positions(num_input_tokens),
            "target_hidden_states": self.hidden_states[:num_input_tokens],
            "token_indices_to_sample": torch.tensor([num_input_tokens - 1], dtype=torch.int32, device=self.device),
            "common_attn_metadata": AscendCommonAttentionMetadata(
                query_start_loc=torch.tensor([0, num_input_tokens], dtype=torch.int32, device=self.device),
                query_start_loc_cpu=torch.tensor([0, num_input_tokens], dtype=torch.int32, device="cpu"),
                seq_lens=torch.tensor([num_input_tokens], dtype=torch.int32, device=self.device),
                seq_lens_cpu=torch.tensor([num_input_tokens], dtype=torch.int32, device="cpu"),
                num_reqs=1,
                num_actual_tokens=num_input_tokens,
                max_query_len=num_input_tokens,
                max_seq_len=self.max_model_len,
                block_table_tensor=torch.tensor([], dtype=torch.int32, device=self.device),
                slot_mapping=self.arange[:num_input_tokens],
                logits_indices_padded=None,
                num_logits_indices=None,
                causal=True,
                encoder_seq_lens=None,
            ),
            "multi_layer_eagle_metadata": MultiLayerEagleMetadata.make_dummy(
                layer_num=self.layer_num,
                hidden_size=self.hidden_size,
                device=self.device,
            ),
        }
        self.adjust_input(**adjust_input_kwargs)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            self._runnable(
                num_input_tokens=num_input_tokens,
                batch_size=batch_size,
                token_indices_to_sample=self.token_indices_to_sample[: batch_size * self.extra_slots_per_request],
                target_positions=model_positions,
                inputs_embeds=None,
                common_attn_metadata=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                is_dummy=True,
                num_tokens=num_input_tokens,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)


class AscendMultiLayerEagleProposer(MultiLayerEagleProposer):
    """Ascend-specific multi-layer Eagle proposer.

    Thin wrapper that configures the multi-layer proposer for use on
    Ascend NPUs with ``pass_hidden_states_to_model=True``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
