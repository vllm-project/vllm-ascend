import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla, update_cos_sin
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.utils import lmhead_tp_enable


class MtpProposer(EagleProposer):
    # TODO: Find out why ModelRunner does not this explicit typing?
    model: nn.Module | ACLGraphWrapper

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ) -> None:
        # Currently, both GLM and DS encounter issues when enabling the fullgraph mode and running on EagleProposer.
        # Therefore, we temporarily bypass this problem by adding a conditional check for fullgraph.
        # TODO: this conditional check should be removed after bug fixing.
        if (
            self.pcp_size * self.dcp_size == 1
            and not self.speculative_config.disable_padded_drafter_batch
            and not self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            super().dummy_run(
                num_tokens,
                with_prefill,
                in_graph_capturing,
                num_reqs,
                num_tokens_across_dp,
                aclgraph_runtime_mode,
                batch_descriptor,
                dummy_compute_logits,
                is_profile,
            )
            return
        (
            num_tokens,
            num_tokens_across_dp,
            with_prefill,
        ) = self.runner._sync_metadata_across_dp(num_tokens, with_prefill)
        if not self.use_cuda_graph:
            # there is synchronization between mtp steps when enabling aclgraph,
            # disable aclgraph when use async scheduling to avoid the
            # synchronization overhead.
            # NOTE: we need to set aclgraph_runtime_mode to None in both dummy_run
            # and _propose.
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            if len(self.runner.attn_groups) > 0:
                num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.runner.query_start_loc.gpu[: num_reqs + 1],
                    query_start_loc_cpu=self.runner.query_start_loc.cpu[: num_reqs + 1],
                    seq_lens_cpu=self.runner.seq_lens.cpu,
                    seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    max_query_len=self.num_speculative_tokens + 1,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                    block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor(),
                    slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                    positions=self.runner.positions.gpu,
                    attn_state=self.runner.attn_state,
                    decode_token_per_req=self.runner.decode_token_per_req,
                    max_seq_len=0,
                )
                if self.pcp_size * self.dcp_size > 1:
                    # update long_seq related params and flatten block_table
                    common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata
                    common_attn_metadata.block_table_tensor = self.runner.input_batch.block_table[
                        0
                    ].get_device_tensor()[: num_reqs * self.decode_threshold]

                builder = self.runner.attn_groups[0][0].get_metadata_builder()
                # `AscendAttentionState.SpecDecoding` is only designed for MLA.
                # `AscendAttentionState.ChunkedPrefill` is used in self-attention.
                attn_state = (
                    AscendAttentionState.SpecDecoding
                    if self.vllm_config.model_config.use_mla
                    else AscendAttentionState.ChunkedPrefill
                )
                attn_metadata_mtp = builder.build_for_graph_capture(common_attn_metadata, attn_state)
                attn_metadata = {}
                for layer_name in self.attn_layer_names:
                    attn_metadata[layer_name] = attn_metadata_mtp
            else:
                attn_metadata = None
        else:
            attn_metadata = None

        input_ids = self.input_ids[:num_tokens]
        positions = self._get_positions(num_tokens)
        previous_hidden_states = self.hidden_states[:num_tokens]
        for i in range(self.num_speculative_tokens):
            if i > 0 and not in_graph_capturing and aclgraph_runtime_mode == CUDAGraphMode.FULL:
                aclgraph_runtime_mode = CUDAGraphMode.NONE
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_actual_tokens=0,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                is_draft_model=True,
                in_profile_run=is_profile,
            ):
                # Reset MOE layer index for each MTP step iteration
                forward_context = get_forward_context()
                if forward_context is not None:
                    forward_context.moe_layer_index = 0
                previous_hidden_states, positions = self.maybe_pad_and_reduce(previous_hidden_states, positions)
                self.model(input_ids=input_ids, positions=positions, hidden_states=previous_hidden_states)
                forward_context = get_forward_context()
                if (
                    forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                    and not forward_context.capturing
                    and not self.use_sparse
                ):
                    self._update_full_graph_params(forward_context, num_tokens)

                previous_hidden_states, positions, _ = self.maybe_all_gather_and_unpad(
                    previous_hidden_states, positions
                )
                dummy_compute_logits(previous_hidden_states)
            if with_prefill:
                break

    def _propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
    ) -> torch.Tensor:
        # Currently, both GLM and DS encounter issues when enabling the fullgraph mode and running on EagleProposer.
        # Therefore, we temporarily bypass this problem by adding a conditional check for fullgraph.
        # TODO: this conditional check should be removed after bug fixing.
        if (
            self.pcp_size * self.dcp_size == 1
            and not self.speculative_config.disable_padded_drafter_batch
            and not self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            draft_token_ids = super()._propose(
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                last_token_indices,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs,
                req_scheduled_tokens,
                long_seq_metadata,
                num_prefill_reqs,
                num_decode_reqs,
                scheduler_output,
                num_scheduled_tokens,
            )
            return draft_token_ids

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        # update pcp related params
        if self.pcp_size * self.dcp_size > 1:
            assert long_seq_metadata is not None
            common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
            ori_last_token_indices = last_token_indices.clone()
            query_lens_d = self.runner.query_lens[:num_decode_reqs]
        if self.pcp_size > 1:
            # 1. preprocess decode/prefill input_ids & target_hidden_states
            # decode input_ids: keep unchanged
            # decode target_hidden_states: remove padding
            # prefill input_ids: add padding and pcp split
            # prefill target_hidden_states: pcp split
            num_tokens_d = query_lens_d.sum().item()
            num_tokens_d_padded = num_tokens_d * self.pcp_size
            input_ids_d = self.input_ids[:num_tokens_d]
            input_ids_p = self.input_ids[num_tokens_d:num_tokens]
            target_hidden_states_d_padded = target_hidden_states[:num_tokens_d_padded]
            if num_tokens_d:
                # remove padding (from pcp all-gather) in decode part
                mask_start_loc = torch.cat(
                    [torch.tensor([0], dtype=torch.int32), torch.cumsum(query_lens_d * self.pcp_size, dim=0)[:-1]]
                )
                mask_len = query_lens_d
                mask = []
                for req_id in range(num_decode_reqs):
                    mask += list(range(mask_start_loc[req_id], mask_start_loc[req_id] + mask_len[req_id]))
                target_hidden_states_d = target_hidden_states_d_padded[mask]
            else:
                target_hidden_states_d = target_hidden_states_d_padded
            target_hidden_states_p = target_hidden_states[num_tokens_d_padded:]
            req_scheduled_tokens_p = {}
            for i, req_id in enumerate(self.runner.input_batch.req_ids):
                if i >= num_decode_reqs:
                    req_scheduled_tokens_p[req_id] = req_scheduled_tokens[req_id]
            (num_tokens_p, input_ids_p, target_hidden_states_p, max_query_len_p, seq_lens_p, cu_num_tokens_p) = (
                self._split_pcp_input(req_scheduled_tokens_p, input_ids_p, target_hidden_states_p)
            )
            num_tokens = num_tokens_d + num_tokens_p
            target_positions = target_positions[:num_tokens]
            self.input_ids[:num_tokens].copy_(torch.cat([input_ids_d, input_ids_p], dim=0))
            target_hidden_states = torch.cat([target_hidden_states_d, target_hidden_states_p], dim=0)
            # 2. update sample_indices according to main model
            if num_decode_reqs:
                last_token_indices[:num_decode_reqs] = self.runner.logits_indices[last_token_indices[:num_decode_reqs]]
            if num_prefill_reqs:
                last_token_indices[-num_prefill_reqs:] = self.runner.logits_indices[-num_prefill_reqs:]
                # 3. update attn_metadata params that may be influenced by pcp
                common_attn_metadata.num_actual_tokens = num_tokens
                common_attn_metadata.max_query_len = max(self.decode_threshold, max_query_len_p)
                common_attn_metadata.seq_lens[-num_prefill_reqs:] = seq_lens_p
                common_attn_metadata.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
                query_start_loc_p = cu_num_tokens_p[1:] + common_attn_metadata.query_start_loc[num_decode_reqs].item()
                common_attn_metadata.query_start_loc[-num_prefill_reqs:] = query_start_loc_p
                common_attn_metadata.query_start_loc_cpu[-num_prefill_reqs:] = query_start_loc_p

        assert self.runner is not None

        # Note(qcs): We may need to refactor these check logics.
        if self.use_cuda_graph and num_scheduled_tokens <= self.runner.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.runner.cudagraph_dispatcher._bs_to_padded_graph_size[num_scheduled_tokens]
        else:
            # Eager mode, no padding needed
            num_input_tokens = num_tokens

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states
        # eager/acl piecewise mode need to update num_tokens_across_dp
        (num_input_tokens, num_tokens_across_dp, with_prefill) = self.runner._sync_metadata_across_dp(
            num_input_tokens, self.runner.with_prefill
        )

        # Enable shared_expert_dp and MTP FULL graph may cause accuracy issues.
        if scheduler_output and not self.enable_shared_expert_dp:
            max_query_len = common_attn_metadata.max_query_len
            uniform_decode = (max_query_len in list(range(1, self.num_speculative_tokens + 2))) and (
                scheduler_output.total_num_scheduled_tokens
                == self.runner.input_batch.num_reqs * (self.num_speculative_tokens + 1)
            )
        else:
            uniform_decode = False
        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )
        if not self.use_cuda_graph:
            # there is synchronization between mtp steps when enabling aclgraph,
            # disable aclgraph when use async scheduling to avoid the
            # synchronization overhead.
            # NOTE: we need to set aclgraph_runtime_mode to None in both dummy_run
            # and _propose.
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        if (
            self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and aclgraph_runtime_mode == CUDAGraphMode.FULL
        ):
            graph_pad_size = num_input_tokens
        else:
            graph_pad_size = -1

        # If use fullgraph and disable_padded_drafter_batch=True, We need to
        # update the graph_pad_size in common_attn_metadata, to tell the
        # builder padding some elements.
        common_attn_metadata.graph_pad_size = graph_pad_size
        common_attn_metadata.num_input_tokens = num_input_tokens
        builder = self.runner.attn_groups[0][0].get_metadata_builder()
        attn_metadata_mtp = builder.build(0, common_attn_metadata, self.runner.get_model())
        attn_metadata = {}
        for layer_name in self.attn_layer_names:
            attn_metadata[layer_name] = attn_metadata_mtp

        update_cos_sin(self._get_positions(num_input_tokens))
        for step in range(self.num_speculative_tokens):
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                num_actual_tokens=num_tokens,
                is_draft_model=True,
            ):
                # Reset MOE layer index for each MTP step to match all_moe_layers registration
                forward_context = get_forward_context()
                if forward_context is not None:
                    forward_context.moe_layer_index = 0

                with record_function_or_nullcontext("mtp_forward"):
                    model_kwargs = {}
                    model_kwargs["attn_metadata"] = attn_metadata
                    input_ids = self.input_ids[:num_input_tokens]
                    positions = self._get_positions(num_input_tokens)
                    hidden_states = self.hidden_states[:num_input_tokens]

                    hidden_states, positions = self.maybe_pad_and_reduce(hidden_states, positions)

                    hidden_states = self.model(input_ids=input_ids, positions=positions, hidden_states=hidden_states)
                    forward_context = get_forward_context()
                    if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not self.use_sparse:
                        self._update_full_graph_params(forward_context, num_input_tokens)

                    hidden_states, positions, _ = self.maybe_all_gather_and_unpad(hidden_states, positions)

            num_indices = last_token_indices.shape[0]
            if lmhead_tp_enable():
                max_num_reqs_across_dp = (
                    self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
                )
                last_token_indices = nn.functional.pad(last_token_indices, (0, max_num_reqs_across_dp - num_indices))

            if self.pcp_size > 1 and step == 0:
                # remove graph padding before all_gather
                hidden_states = hidden_states[:num_tokens]
                hidden_states = get_pcp_group().all_gather(hidden_states, 0)
                hidden_states = torch.index_select(
                    hidden_states, 0, self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
                )

            sample_hidden_states = hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states)
            if lmhead_tp_enable() and num_indices < logits.shape[0]:
                logits = logits[:num_indices]
                last_token_indices = last_token_indices[:num_indices]
            draft_token_ids = logits.argmax(dim=-1)

            if self.num_speculative_tokens == 1:
                # [batch_size, 1]
                return draft_token_ids.view(-1, 1)

            if step == 0:
                draft_token_ids_list = [draft_token_ids]
            else:
                draft_token_ids_list.append(draft_token_ids)

            # prepare next mtp inputs
            # mtp>1: prefill skip or decode skip last loop
            if with_prefill:
                for _ in range(self.num_speculative_tokens - 1):
                    draft_token_ids_list.append(draft_token_ids)
            if step == self.num_speculative_tokens - 1 or with_prefill:
                break

            attn_metadata_i = attn_metadata[self.attn_layer_names[0]]

            if step == 0:
                positions = target_positions[last_token_indices]
                hidden_states = hidden_states[last_token_indices]
                slot_mapping = attn_metadata_i.slot_mapping[last_token_indices]
                attn_metadata_i.slot_mapping.fill_(-1)
                attn_metadata_i.query_start_loc = self.arange[: batch_size + 1]
                last_token_indices = self.arange[:batch_size]
                if getattr(attn_metadata_i, "num_decode_tokens", 0):
                    attn_metadata_i.num_decode_tokens = batch_size
                if self.pcp_size * self.dcp_size > 1:
                    positions = target_positions[ori_last_token_indices]
                    # For pcp/dcp, tokens are split across different cp ranks,
                    # so we can not simply update slot_mapping by += 1.
                    # Instead, we pre-allocate mtp slot_mapping in model_runner
                    # (_generate_pcp_mtp_input), and use updated slot_indices
                    # to get corresponding slot_mapping in each step.
                    num_reject_tokens = (
                        torch.tensor(self.runner.pcp_manager.cu_num_tokens_pcp_full, dtype=torch.int32).to(self.device)
                        - ori_last_token_indices
                        - 1
                    )
                    num_accept_tokens = query_lens_d.to(self.device) - num_reject_tokens
                    # `AscendAttentionState.SpecDecoding` is only designed for MLA.
                    # `AscendAttentionState.ChunkedPrefill` is used in self-attention.
                    mtp_slot_mapping = self.runner.pcp_manager.mtp_slot_pad

                    # slot_mapping index base offset:
                    # scheduled tokens + pre-allocated mtp tokens + accepted tokens
                    slot_idx_base = (
                        torch.cat(
                            [
                                torch.tensor([0], dtype=torch.int32, device=self.device),
                                (torch.cumsum(query_lens_d, dim=0)[:-1] * self.pcp_size).to(self.device),
                            ]
                        )
                        + torch.arange(num_decode_reqs, device=self.device)
                        * (self.num_speculative_tokens - 1)
                        * self.pcp_size
                        + (num_accept_tokens - 1) * self.pcp_size
                    )
                    slot_indices_list = []
                    for req_id in range(num_decode_reqs):
                        slot_indices_list.append(
                            torch.arange(
                                slot_idx_base[req_id], slot_idx_base[req_id] + self.pcp_size, device=self.device
                            )
                        )
                    slot_indices = torch.cat(slot_indices_list, dim=0)

                    # fold block_table (restore it to original size before flattened)
                    block_indices = torch.cat(
                        [torch.tensor([0], dtype=torch.int32), torch.cumsum(query_lens_d, dim=0)[:-1]]
                    )
                    attn_metadata_i.decode.block_table[:batch_size] = attn_metadata_i.decode.block_table[block_indices]
                    attn_metadata_i.decode.block_table = attn_metadata_i.decode.block_table[:batch_size]

            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            decode_metadata = getattr(attn_metadata_i, "decode", None)
            prefill_metadata = getattr(attn_metadata_i, "prefill", None)
            # When disable_padded_drafter_batch=False, it should not to be updating these params, maybe.
            if decode_metadata is not None and (
                self.speculative_config.disable_padded_drafter_batch or aclgraph_runtime_mode != CUDAGraphMode.FULL
            ):
                decode_metadata.actual_seq_lengths_q = self.arange_cpu[1 : batch_size + 1].tolist()
                if aclgraph_runtime_mode == CUDAGraphMode.FULL:
                    decode_metadata.actual_seq_lengths_q = builder.pad_actual_seq_len_q_mtp_disable_pad(
                        graph_pad_size - batch_size, batch_size, decode_metadata.actual_seq_lengths_q
                    )
                decode_metadata.cos, decode_metadata.sin = get_cos_and_sin_mla(positions[:batch_size])
            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions[:batch_size] >= self.runner.model_config.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions[:batch_size])
            # Increment the sequence lengths.
            # This is an out-of-place operation to avoid modifying the original tensor
            # when enable async_scheduling.
            attn_metadata_i.seq_lens = attn_metadata_i.seq_lens + 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            exceeds_mask = attn_metadata_i.seq_lens[:batch_size] > self.runner.model_config.max_model_len
            attn_metadata_i.seq_lens[:batch_size].masked_fill_(exceeds_mask, 1)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping += 1
            if self.pcp_size > 1:
                exceeds_max_model_len = exceeds_max_model_len.repeat_interleave(
                    slot_mapping.size(0) // exceeds_max_model_len.size(0)
                )
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[: hidden_states.shape[0]] = hidden_states
            if self.pcp_size * self.dcp_size > 1:
                # update local seq_len
                num_computed_tokens_of_pcp_dcp = self.runner.pcp_manager._get_cp_local_seq_lens(
                    attn_metadata_i.seq_lens[:batch_size],
                    self.pcp_size,
                    self.dcp_size,
                    self.runner.parallel_config.cp_kv_cache_interleave_size,
                )
                cp_seq_len = num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank]
                attn_metadata_i.decode.cp_seq_len = cp_seq_len
                # update slot_mapping
                slot_indices += self.pcp_size
                slot_mapping = mtp_slot_mapping[slot_indices]
                attn_metadata_i.slot_mapping[: batch_size * self.pcp_size] = slot_mapping
            else:
                attn_metadata_i.slot_mapping[:batch_size] = slot_mapping
            if self.speculative_config.disable_padded_drafter_batch:
                if self.uses_mrope:
                    self.mrope_positions[:, batch_size:num_input_tokens] = 0
                else:
                    self.positions[batch_size:num_input_tokens] = 0
                self.input_ids[batch_size:num_input_tokens] = 0
                self.hidden_states[batch_size:num_input_tokens].fill_(0)

            if prefill_metadata is not None:
                prefill_metadata.seq_lens = attn_metadata_i.seq_lens
                prefill_metadata.seq_lens_list = prefill_metadata.seq_lens.tolist()
                prefill_metadata.context_lens = attn_metadata_i.seq_lens
                prefill_metadata.input_positions = self._get_positions(num_input_tokens)
                prefill_metadata.max_seq_lens += 1
                prefill_metadata.max_seq_lens = min(
                    prefill_metadata.max_seq_lens, self.runner.model_config.max_model_len
                )
            if decode_metadata is not None:
                decode_metadata.seq_lens = attn_metadata_i.seq_lens
                decode_metadata.seq_lens_list = decode_metadata.seq_lens.tolist()
                decode_seq_lens_list = decode_metadata.seq_lens_list
                if aclgraph_runtime_mode == CUDAGraphMode.FULL and self.speculative_config.disable_padded_drafter_batch:
                    decode_metadata.seq_lens_list = decode_seq_lens_list + [0] * (
                        graph_pad_size - len(decode_seq_lens_list)
                    )
                decode_metadata.input_positions = self._get_positions(num_input_tokens)
                decode_metadata.max_seq_lens += 1
                decode_metadata.max_seq_lens = min(decode_metadata.max_seq_lens, self.runner.model_config.max_model_len)

        # mtp>1: [batch_size, k]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
