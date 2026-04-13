# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from sympy import false
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm.forward_context import BatchDescriptor, get_forward_context

logger = init_logger(__name__)


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

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
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
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

        self.method = "dflash"

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

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
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            )
        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        cad.seq_lens_cpu = cad.seq_lens.cpu()

        return num_query_total, token_indices_to_sample, cad, None

@override
@torch.inference_mode()
def dummy_run(
    self,
    num_tokens: int,
    use_cudagraphs: bool = True,
    is_graph_capturing: bool = False,
    in_graph_capturing: bool = False,
    slot_mappings: dict[str, torch.Tensor] | None = None,
    with_prefill: bool = False,
    num_reqs: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor=None,
    dummy_compute_logits=lambda hidden_states: None,
    is_profile=False,
) -> None:
    num_query_per_req = 1 + self.num_speculative_tokens
    num_query_total = num_reqs * num_query_per_req

   
    num_query_tokens = min(num_query_total, self.max_query_tokens)
    cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
        self._determine_batch_execution_and_padding(
            num_query_tokens, use_cudagraphs=use_cudagraphs
        )
    )

    if (
        self._draft_attn_layer_names
        and slot_mappings is not None
        and next(iter(self._draft_attn_layer_names)) in slot_mappings
    ):
        slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
    else:
        slot_mapping_dict = slot_mappings or {}

    context_positions = self._context_positions_buffer[:num_tokens]
    context_states = self.hidden_states[:num_tokens]

    self.model.precompute_and_store_context_kv(context_states, context_positions)

    multi_steps_attn_metadata = []
    if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
        builder = self.draft_attn_groups[0].get_metadata_builder()
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.arange[: num_reqs + 1] * num_query_per_req,
            query_start_loc_cpu=torch.from_numpy(
                self.token_arange_np[: num_reqs + 1]
            ).clone() * num_query_per_req,
            seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
            seq_lens=self.runner.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=0,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            block_table_tensor=self.runner.input_batch.block_table[
                0
            ].get_device_tensor()[:num_reqs],
        )

        attn_metadata_dflash = builder.build_for_graph_capture(
            common_attn_metadata,
            AscendAttentionState.ChunkedPrefill,
        )
        attn_metadata_dflash.attn_mask = None
        attn_metadata_dflash.attn_state = AscendAttentionState.ChunkedPrefill

        per_layer_attn_metadata = dict()
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata_dflash
        multi_steps_attn_metadata.append(per_layer_attn_metadata)

    with set_ascend_forward_context(
        multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
        self.vllm_config,
        num_tokens=num_input_tokens,         
        num_tokens_across_dp=num_tokens_across_dp,
        num_actual_tokens=num_query_total,   
        in_profile_run=is_profile,
        batch_descriptor=batch_descriptor,
        aclgraph_runtime_mode=aclgraph_runtime_mode,
        is_draft_model=True,
        draft_attn_metadatas=multi_steps_attn_metadata,
    ):
        
        self.model(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self._get_positions(num_input_tokens),
            inputs_embeds=None,
        )

        forward_context = get_forward_context()
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not _EXTRA_CTX.capturing
        ):
            self._update_full_graph_params(
                forward_context, num_input_tokens, multi_steps_attn_metadata
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> dict[str, object]:
        per_layer_attention_metadata = super().build_per_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer_attention_metadata.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_layer_attention_metadata

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
