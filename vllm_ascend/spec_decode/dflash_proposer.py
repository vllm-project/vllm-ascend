# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from typing_extensions import override
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer


class AscendDFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        num_query_per_req = 1 + self.num_speculative_tokens
        self.max_query_tokens = self.max_batch_size * num_query_per_req
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_positions + 1,
            device=device,
            dtype=torch.int32,
        )

        self.parallel_drafting_hidden_state_tensor = None
        self.method = "dflash"
        self._dflash_num_context = 0
        self._dflash_hidden_states: torch.Tensor | None = None

    @override
    def _raise_if_multimodal(self):
        # DFlash is intended to support Qwen3.5 multimodal variants as well.
        return

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: AscendCommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, AscendCommonAttentionMetadata, tuple[Any, Any] | None]:
        del target_token_ids, token_indices_to_sample
        del req_scheduled_tokens, long_seq_metadata, num_prefill_reqs, num_decode_reqs

        batch_size = cad.batch_size()
        num_context = target_hidden_states.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        block_size = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, block_size)
        grid = (batch_size, num_blocks)
        has_num_rejected = num_rejected_tokens_gpu is not None

        copy_and_expand_dflash_inputs_kernel[grid](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=num_rejected_tokens_gpu if has_num_rejected else 0,
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.kernel_block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=block_size,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        )
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.seq_lens_cpu = cad.seq_lens.cpu()
        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        cad.causal = False
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @override
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
    ):
        del with_prefill, in_graph_capturing, dummy_compute_logits

        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req
        (
            num_query_total,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_total, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_tokens]
        context_states = self.hidden_states[:num_tokens]
        self.model.precompute_and_store_context_kv(context_states, context_positions)

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            assert len(self.draft_attn_groups) > 0
            builder = self.draft_attn_groups[0].get_metadata_builder()
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.arange[: num_reqs + 1] * num_query_per_req,
                query_start_loc_cpu=torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu[:num_reqs],
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_query_total,
                num_input_tokens=num_query_total,
                max_query_len=num_query_per_req,
                slot_mapping=self._slot_mapping_buffer[:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                max_seq_len=0,
            )
            attn_metadata = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.ChunkedPrefill,
            )
            attn_metadata.attn_mask = None
            per_layer_attn_metadata = {layer_name: attn_metadata for layer_name in self.attn_layer_names}
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_query_total,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_query_total,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            self.model(
                input_ids=self.input_ids[:num_query_total],
                positions=self._get_positions(num_query_total),
                inputs_embeds=None,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_query_total, multi_steps_attn_metadata)

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        del num_tokens, inputs_embeds
        assert self._dflash_hidden_states is not None

        num_context = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        positions = self._get_positions(num_input_tokens)
        return (
            {
                "input_ids": self.input_ids[:num_input_tokens],
                "positions": positions,
                "inputs_embeds": None,
            },
            positions,
        )

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        dflash_config = getattr(draft_model_config.hf_config, "dflash_config", None)
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    @override
    def model_returns_tuple(self) -> bool:
        return False
