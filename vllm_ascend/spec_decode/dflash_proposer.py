from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
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

        copy_and_expand_dflash_inputs_kernel_single_grid[1,](
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
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            batch_size=batch_size,
            HAS_NUM_REJECTED=has_num_rejected,
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
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req

        # init block table tensor clone is only available after profile run
        # and is only used for graph mode
        if self.use_cuda_graph and not is_profile and self.block_table_tensor_clone is None:
            self.block_table_tensor_clone = torch.zeros(
                (
                    self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs,
                    self.runner.input_batch.block_table[0].get_device_tensor().shape[1],
                ),
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.runner.pin_memory,
            )

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        # Build attn_metadata for FULL graph capture so that FIA operators
        # are properly included in the captured graph (DFlash needs non-causal attention).
        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.draft_attn_groups) > 0:
            # DFlash uses num_query_per_req tokens per request
            num_query_per_req = 1 + self.num_speculative_tokens
            batch_size = max(num_input_tokens // num_query_per_req, 1)

            # Build correct DFlash query_start_loc: [0, nq, 2*nq, ...]
            dflash_qsl = self.arange_dflash[: batch_size + 1] * num_query_per_req
            dflash_qsl_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * num_query_per_req
            # Correct actual_seq_lengths_q for DFlash: cumulative query lengths
            dflash_actual_seq_q = self.arange_dflash[1 : batch_size + 1] * num_query_per_req

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=dflash_qsl,
                query_start_loc_cpu=dflash_qsl_cpu,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:batch_size],
                num_reqs=batch_size,
                num_actual_tokens=num_input_tokens,
                num_input_tokens=num_input_tokens,
                max_query_len=num_query_per_req,
                num_computed_tokens_cpu=None,
                actual_seq_lengths_q=dflash_actual_seq_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:batch_size],
                slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                positions=self.runner.positions,
                attn_state=AscendAttentionState.ChunkedPrefill,
                decode_token_per_req=num_query_per_req,
                max_seq_len=0,
                causal=False,
            )

            builder = self.draft_attn_groups[0].get_metadata_builder()
            attn_metadata_dflash = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.ChunkedPrefill,
            )
            per_layer_attn_metadata = dict()
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_dflash
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata if multi_steps_attn_metadata else None,
        ):
            self.model.precompute_and_store_context_kv(context_states, context_positions)

            self.model(
                input_ids=self.input_ids[:num_query_total],
                positions=self._get_positions(num_query_total),
                inputs_embeds=None,
            )
            if multi_steps_attn_metadata:
                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                    self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)

        # Clear ALL draft_graph_params stored during dummy_run warmup.
        # Issues with keeping dummy_run's params:
        # 1. attn_params: duplicate entries when real call uses same key
        # 2. workspaces: computed with stale seq_lens (near 0), too small for
        #    real inference (e.g. seq_lens=118) → NPU buffer overflow (error 507057)
        # 3. events/handles: stale references
        # The first real ACLGraphWrapper capture will re-populate everything correctly.
        if multi_steps_attn_metadata:
            from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, get_draft_graph_params

            draft_gp = get_draft_graph_params()
            if draft_gp is not None:
                for key in list(draft_gp.attn_params.keys()):
                    draft_gp.attn_params[key] = []
                    draft_gp.events[key] = []
                    draft_gp.handles[key] = []
                for key in list(draft_gp.workspaces.keys()):
                    draft_gp.workspaces[key] = None
            # Invalidate ACLGraphWrapper's captured graph entries so the first
            # real _propose() call re-captures with correct runtime params.
            # During dummy_run, self.model (ACLGraphWrapper) captured a graph
            # with stale dummy values. Without clearing, the first real call
            # would replay that stale graph instead of re-capturing.
            if isinstance(self.model, ACLGraphWrapper):
                self.model.concrete_aclgraph_entries.clear()

    def prepare_context_kv(self) -> None:
        """Precompute and store context KV OUTSIDE the graph boundary.

        This must run eagerly (before graph replay) because num_context varies
        between calls, making it incompatible with fixed-shape graph replay.
        """
        num_context = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
    ) -> dict[str, Any]:
        return dict(
            input_ids=self.input_ids[:num_input_tokens], positions=self.positions[:num_input_tokens], inputs_embeds=None
        )
