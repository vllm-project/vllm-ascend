from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel_single_grid
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class AscendDflashProposer(AscendEagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
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
            dtype=torch.int32,
            device=device,
        )

        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.arange_dflash = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)

        self.parallel_drafting_hidden_state_tensor = None

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.num_reqs
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        has_num_rejected = num_rejected_tokens_gpu is not None

        torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
            # Inputs
            next_token_ids,
            target_positions,
            cad.query_start_loc,
            num_rejected_tokens_gpu,
            cad.block_table_tensor,
            # Scalars
            self.parallel_drafting_token_id,
            num_query_per_req,
            self.num_speculative_tokens,
            self.block_size,
            # In-place outputs
            self.input_ids,
            self._context_positions_buffer,
            self.positions,
            self._context_slot_mapping_buffer,
            self._slot_mapping_buffer,
            token_indices_to_sample
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_query_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        num_input_tokens = num_query_tokens

        num_context = num_tokens
        context_positions = self._context_positions_buffer[:num_context]
        context_states = self.hidden_states[:num_context]

        input_ids = self.input_ids[:num_input_tokens]
        positions = self.positions[:num_input_tokens]

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
        ):
            self.model.precompute_and_store_context_kv(context_states, context_positions)
            self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=None,
            )

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        num_context = self._dflash_num_context

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )

        return dict(
            input_ids=self.input_ids[:num_input_tokens], positions=self.positions[:num_input_tokens], inputs_embeds=None
        )
