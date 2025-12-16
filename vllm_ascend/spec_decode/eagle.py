import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_forward_context import MoECommType, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, update_mla_attn_params
from vllm_ascend.utils import lmhead_tp_enable, shared_expert_dp_enabled


class NPUEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)

        self.decode_threshold = 1 + self.num_speculative_tokens
        self.enable_shared_expert_dp = shared_expert_dp_enabled()

        self.pcp_size = self.runner.pcp_size
        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank

        self.use_cuda_graph = self.runner._use_aclgraph()

        self.cudagraph_batch_sizes = (list(
            sorted(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))
                                      if self.use_cuda_graph else [])

        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size +
            self.pcp_size * self.dcp_size * self.runner.max_num_reqs)

        self.arange_cpu = torch.arange(self.arange.shape[0],
                                       device="cpu",
                                       dtype=torch.int32)

        self.use_sparse = hasattr(vllm_config.model_config.hf_config,
                                  "index_topk")
        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        # TODO: different model may have different logic
        # take care, or it will break others

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ):
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs=True,
        in_graph_capturing=False,
        with_prefill=False,
        num_reqs=0,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
    ) -> None:

        (
            num_tokens,
            num_tokens_across_dp,
            with_prefill,
        ) = self.runner._sync_metadata_across_dp(num_tokens, with_prefill)
        if self.use_async_scheduling:
            # there is synchronization between mtp steps when enabling aclgraph,
            # disable aclgraph when use async scheduling to avoid the
            # synchronization overhead.
            # NOTE: we need to set aclgraph_runtime_mode to None in both dummy_run
            # and _propose.
            cudagraph_runtime_mode = CUDAGraphMode.NONE
        moe_comm_type = self.runner._select_moe_comm_method(num_tokens)
        # TODO: remove this after moe_comm_type selection logic is finalized
        moe_comm_type = (MoECommType.ALLTOALL if moe_comm_type
                         == MoECommType.FUSED_ALLTOALL else moe_comm_type)

        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if len(self.runner.attn_groups) > 0:
                num_computed_tokens_cpu = (
                    self.runner.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs])
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.runner.query_start_loc.gpu[:num_reqs +
                                                                    1],
                    query_start_loc_cpu=self.runner.query_start_loc.
                    cpu[:num_reqs + 1],
                    seq_lens_cpu=self.runner.seq_lens.cpu,
                    seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    max_query_len=self.num_speculative_tokens + 1,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                    block_table_tensor=self.runner.input_batch.block_table[0].
                    get_device_tensor(),
                    slot_mapping=self.runner.input_batch.block_table[0].
                    slot_mapping.gpu,
                    positions=self.runner.positions.gpu,
                    attn_mask=self.runner.attn_mask,
                    spec_attn_mask=self.runner.spec_attn_mask,
                    attn_state=self.runner.attn_state,
                    decode_token_per_req=self.runner.decode_token_per_req,
                )

                builder = self.runner.attn_groups[0][0].get_metadata_builder()
                attn_metadata_mtp = builder.build_for_graph_capture(
                    common_attn_metadata, AscendAttentionState.SpecDecoding,
                    self.runner.get_model())
                attn_metadata = {}
                for layer_name in self.attn_layer_names:
                    attn_metadata[layer_name] = attn_metadata_mtp
            else:
                attn_metadata = None
        else:
            attn_metadata = None

        input_ids = self.input_ids[:num_tokens]
        positions = self.positions[:num_tokens]
        previous_hidden_states = self.hidden_states[:num_tokens]
        for i in range(self.num_speculative_tokens):
            if i > 0 and not in_graph_capturing and cudagraph_runtime_mode == CUDAGraphMode.FULL:
                cudagraph_runtime_mode = CUDAGraphMode.NONE
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    with_prefill=with_prefill,
                    num_tokens_across_dp=num_tokens_across_dp,
                    moe_comm_type=moe_comm_type,
                    in_profile_run=self.runner.in_profile_run,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    is_mtp_model=True):
                if self.enable_shared_expert_dp:
                    positions = positions.unsqueeze(-1)
                    positions = torch.ops.vllm.maybe_pad_and_reduce(positions)
                    positions = positions.squeeze(-1)
                    previous_hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
                        previous_hidden_states)
                self.model(input_ids=input_ids,
                           positions=positions,
                           hidden_states=previous_hidden_states)
                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and \
                    not forward_context.capturing:
                    if self.vllm_config.model_config.use_mla and not self.use_sparse:
                        update_mla_attn_params(
                            self.update_stream, forward_context, num_tokens,
                            self.vllm_config.speculative_config)
                if self.enable_shared_expert_dp:
                    positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        positions, True)
                    previous_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        previous_hidden_states, True)
                dummy_compute_logits(previous_hidden_states)
            if with_prefill:
                break

    def propose(
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
        scheduler_output=None,
        num_scheduled_tokens=0,
    ) -> torch.Tensor:
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
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        # update pcp related params
        if self.pcp_size > 1:
            assert long_seq_metadata is not None
            common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
            # 1. preprocess decode/prefill input_ids & target_hidden_states
            # decode input_ids: keep unchanged
            # decode target_hidden_states: remove padding
            # prefill input_ids: add padding and pcp split
            # prefill target_hidden_states: pcp split
            num_tokens_d = num_decode_reqs * self.decode_threshold
            num_tokens_d_padded = num_tokens_d * self.pcp_size
            input_ids_d = self.input_ids[:num_tokens_d]
            input_ids_p = self.input_ids[num_tokens_d:num_tokens]
            target_hidden_states_d_padded = \
                target_hidden_states[:num_tokens_d_padded]
            if num_tokens_d:
                # remove padding (from pcp all-gather) in decode part
                target_hidden_states_d = target_hidden_states_d_padded.reshape(
                    [
                        num_decode_reqs, self.decode_threshold * self.pcp_size,
                        -1
                    ])[:, :self.decode_threshold, :].reshape(
                        [num_tokens_d, -1])
            else:
                target_hidden_states_d = target_hidden_states_d_padded
            target_hidden_states_p = target_hidden_states[num_tokens_d_padded:]
            req_scheduled_tokens_p = {}
            for i, req_id in enumerate(self.runner.input_batch.req_ids):
                if i >= num_decode_reqs:
                    req_scheduled_tokens_p[req_id] = \
                        req_scheduled_tokens[req_id]
            (num_tokens_p, input_ids_p, target_hidden_states_p,
             max_query_len_p, seq_lens_p, cu_num_tokens_p) = \
                self._split_pcp_input(
                    req_scheduled_tokens_p, input_ids_p, target_hidden_states_p)
            num_tokens = num_tokens_d + num_tokens_p
            target_positions = target_positions[:num_tokens]
            self.input_ids[:num_tokens].copy_(
                torch.cat([input_ids_d, input_ids_p], dim=0))
            target_hidden_states = torch.cat(
                [target_hidden_states_d, target_hidden_states_p], dim=0)
            # 2. update attn_metadata params that may be influenced by pcp
            common_attn_metadata.num_actual_tokens = num_tokens
            common_attn_metadata.max_query_len = max(self.decode_threshold,
                                                     max_query_len_p)
            common_attn_metadata.seq_lens[num_decode_reqs:] = seq_lens_p
            common_attn_metadata.seq_lens_cpu[num_decode_reqs:] = seq_lens_p
            query_start_loc_p = cu_num_tokens_p[1:] + \
                common_attn_metadata.query_start_loc[num_decode_reqs].item()
            common_attn_metadata.query_start_loc[num_decode_reqs + 1:] = \
                query_start_loc_p
            common_attn_metadata.query_start_loc_cpu[num_decode_reqs + 1:] = \
                query_start_loc_p
            # 3. update sample_indices according to main model
            if num_decode_reqs:
                last_token_indices[:num_decode_reqs] = \
                    self.runner.logits_indices[last_token_indices[:num_decode_reqs]]
            if num_prefill_reqs:
                last_token_indices[-num_prefill_reqs:] = \
                    self.runner.logits_indices[-num_prefill_reqs:]

        num_input_tokens = num_tokens
        if self.use_cuda_graph:
            if num_scheduled_tokens <= self.cudagraph_batch_sizes[-1] and \
                self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
            elif num_tokens <= self.cudagraph_batch_sizes[-1]:
                num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
            else:
                pass  # eager mode, no padding

        num_input_tokens, num_tokens_across_dp, with_prefill = self.runner._sync_metadata_across_dp(
            num_input_tokens, self.runner.with_prefill)

        uniform_decode = False
        if scheduler_output and not self.enable_shared_expert_dp:
            max_query_len = common_attn_metadata.max_query_len
            uniform_decode_query_len = 1 + self.num_speculative_tokens
            uniform_decode = (
                max_query_len in list(range(1, uniform_decode_query_len + 1)) and
                num_scheduled_tokens == self.runner.input_batch.num_reqs * uniform_decode_query_len)

        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0

        cudagraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_input_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
        )
        original_cudagraph_runtime_mode = cudagraph_runtime_mode

        if self.use_async_scheduling:
            # there is synchronization between mtp steps when enabling aclgraph,
            # disable aclgraph when use async scheduling to avoid the
            # synchronization overhead.
            # NOTE: we need to set aclgraph_runtime_mode to None in both dummy_run
            # and _propose.
            cudagraph_runtime_mode = CUDAGraphMode.NONE

        # If use fullgraph and disable_padded_drafter_batch=True, We need to
        # update the graph_pad_size in common_attn_metadata, to tell the
        # builder padding some elements.
        graph_pad_size = -1
        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and \
            cudagraph_runtime_mode == CUDAGraphMode.FULL:
            graph_pad_size = num_input_tokens
        common_attn_metadata.graph_pad_size = graph_pad_size

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0,
            model=self.runner.get_model(),
        )

        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

        in_profile_run = self.runner.in_profile_run
        moe_comm_type = self.runner._select_moe_comm_method(num_input_tokens)
        # TODO: remove this after moe_comm_type selection logic is finalized
        moe_comm_type = (MoECommType.ALLTOALL if moe_comm_type
                         == MoECommType.FUSED_ALLTOALL else moe_comm_type)
        is_mtp_model = True

        for token_index in range(self.num_speculative_tokens):
            with set_ascend_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                in_profile_run=in_profile_run,
                moe_comm_type=moe_comm_type,
                num_actual_tokens=num_tokens,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                is_mtp_model=is_mtp_model,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None
                positions = self._get_positions(num_input_tokens)
                hidden_states = self.hidden_states[:num_input_tokens]

                if self.enable_shared_expert_dp:
                    # positions [N] -> [N, 1] for padding
                    positions = positions.unsqueeze(-1)
                    positions = torch.ops.vllm.maybe_pad_and_reduce(positions)
                    positions = positions.squeeze(-1)
                    hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)

                if original_cudagraph_runtime_mode == CUDAGraphMode.FULL and \
                    self.use_async_scheduling and \
                    per_layer_attn_metadata[self.attn_layer_names[0]].decode is not None:
                    for layer_name in self.attn_layer_names:
                        actual_size = len(
                            per_layer_attn_metadata[layer_name].decode.actual_seq_lengths_q)

                        per_layer_attn_metadata[layer_name].decode.seq_lens_list = \
                            per_layer_attn_metadata[layer_name].decode.seq_lens_list[:actual_size]
                        per_layer_attn_metadata[layer_name].decode.block_table = \
                            per_layer_attn_metadata[layer_name].decode.block_table[:actual_size]

                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = last_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    if self.vllm_config.model_config.use_mla and not self.use_sparse:
                        update_mla_attn_params(
                            self.update_stream, forward_context,
                            num_input_tokens, self.vllm_config.speculative_config)

                if self.enable_shared_expert_dp:
                    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        hidden_states.contiguous(), True)
                    positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        positions.contiguous(), True)

            num_indices = last_token_indices.shape[0]
            if lmhead_tp_enable():
                max_num_reqs_across_dp = \
                    self.vllm_config.scheduler_config.max_num_seqs * \
                    self.runner.uniform_decode_query_len
                last_token_indices = nn.functional.pad(
                    last_token_indices,
                    (0, max_num_reqs_across_dp - num_indices))

            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(hidden_states, 0)
                hidden_states = torch.index_select(
                    hidden_states, 0,
                    self.runner.pcp_allgather_restore_idx[:hidden_states.shape[0]])

            sample_hidden_states = last_hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states)

            if lmhead_tp_enable() and num_indices < logits.shape[0]:
                logits = logits[:num_indices]
                last_token_indices = last_token_indices[:num_indices]

            draft_token_ids = logits.argmax(dim=-1)

            # Early exit if there is only one draft token to be generated.
            if self.num_speculative_tokens == 1:
                return draft_token_ids.view(-1, 1)

            if token_index == 0:
                draft_token_ids_list = [draft_token_ids]
            else:
                draft_token_ids_list.append(draft_token_ids)

            # prepare next mtp inputs
            # mtp>1: prefill skip or decode skip last loop
            if with_prefill:
                for _ in range(self.num_speculative_tokens - 1):
                    draft_token_ids_list.append(draft_token_ids)
            if token_index == self.num_speculative_tokens - 1 or with_prefill:
                break

            attn_metadata = per_layer_attn_metadata[self.attn_layer_names[0]]

            if token_index == 0:
                if self.uses_mrope:
                    positions = target_positions[:, last_token_indices]
                else:
                    positions = target_positions[last_token_indices]
                if self.method in (
                    "deepseek_mtp",
                    "ernie_mtp",
                    "longcat_flash_mtp",
                    "pangu_ultra_moe_mtp",
                ):
                    hidden_states = self.hidden_states[last_token_indices]
                else:
                    hidden_states = hidden_states[last_token_indices]

                slot_mapping = attn_metadata.slot_mapping[last_token_indices]
                last_token_indices = self.arange[:batch_size]

            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions[:batch_size] >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions[:batch_size])

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:hidden_states.shape[0]] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_input_tokens]
            else:
                input_ids = self.input_ids[:num_input_tokens]
                inputs_embeds = None

            if self.speculative_config.disable_padded_drafter_batch:
                self.input_ids[batch_size:num_input_tokens] = 0
                self.positions[batch_size:num_input_tokens] = 0
                self.hidden_states[batch_size:num_input_tokens].fill_(0)

            attn_metadata_builder.update_for_drafting(
                attn_metadata,
                token_index,
                slot_mapping,
                self.arange,
                self.arange_cpu,
                self.token_arange_np,
                batch_size,
                cudagraph_runtime_mode,
                positions,
                exceeds_max_model_len,
                clamped_positions,
                num_input_tokens,
            )

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    # NOTE: overload it
    # TODO: unify CAM ACAM
    #       fix weird indexing
    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_actual_reqs = len(num_draft_tokens)
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_actual_reqs
                                                                       + 1]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_actual_reqs]
        new_seq_lens_cpu = seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        common_attn_metadata.slot_mapping[:token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices])
        common_attn_metadata.slot_mapping[token_indices.shape[0]:].fill_(-1)

        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions[token_indices],
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
        )

        return spec_common_attn_metadata, token_indices

    # NOTE: overload it (v0.11.2 style -> v0.12.0 style is needed)
    # TODO: unify CAM ACAM
    #       modify weird query_start_loc
    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_draft_tokens_gpu = torch.cat(
            [
                spec_decode_metadata.cu_num_draft_tokens[0:1],
                spec_decode_metadata.cu_num_draft_tokens[1:]
                - spec_decode_metadata.cu_num_draft_tokens[:-1],
            ]
        )

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
        )

        query_start_loc = common_attn_metadata.query_start_loc[
            1:1 + num_rejected_tokens_gpu.shape[0]]
        token_indices_to_sample = query_start_loc - 1 - num_rejected_tokens_gpu

        return spec_common_attn_metadata, token_indices, token_indices_to_sample

    # TODO: delete it when model runner is ready for v0.12.0
    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # TODO(Ben): Combine this into a custom fused kernel

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[
            :num_discarded_requests
        ]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(
            0, discard_sampled_tokens_req_indices, -1
        )

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count

    # TODO: refactor it
    def _split_pcp_input(self, req_scheduled_tokens, input_ids,
                         target_hidden_states):
        """
        Split prefill input_ids and target_hidden_states in pcp group.
        1. input_ids padding: [t0, t1, t2, t3, t4, t5] -> [t0, t1, t2, t3, t4, t5, pad, pad]
        2. split input_ids: pcp0 [t0, t1, pad, pad], pcp1 [t2, t3, t4, t5]
        3. split target_hidden_states (already include pcp padding):
        [h0, h1, h2, h3, h4, h5, pad, pad] -> pcp0 [h0, h1, pad, pad], pcp1 [h2, h3, h4, h5]
        4. also update max_query_len, seq_lens, cu_num_tokens according to pcp split.
        """
        if len(req_scheduled_tokens) == 0:
            # no prefill inputs to split, return empty result
            return (
                0,
                torch.zeros([0], device='npu'),
                torch.zeros([0, target_hidden_states.size(1)], device='npu'),
                0,
                torch.zeros([0]),
                torch.tensor([0], dtype=torch.int32),
            )

        def _pcp_pad_and_split(num_tokens):
            num_pcp_padded_scheduled_tokens = cdiv(
                num_tokens, 2 * self.pcp_size) * 2 * self.pcp_size
            pcp_pad = num_pcp_padded_scheduled_tokens - num_tokens
            chunk_size = num_pcp_padded_scheduled_tokens // (2 * self.pcp_size)

            # split position_ids (and use split position_ids to split input_ids afterwards)
            req_position_cp: list[int] = []
            req_position_cp.extend(
                self.full_indices[self.pcp_rank *
                                  chunk_size:(self.pcp_rank + 1) * chunk_size])
            req_position_cp.extend(
                self.full_indices[num_pcp_padded_scheduled_tokens -
                                  (self.pcp_rank + 1) *
                                  chunk_size:num_pcp_padded_scheduled_tokens -
                                  self.pcp_rank * chunk_size])

            return req_position_cp, num_pcp_padded_scheduled_tokens, pcp_pad

        num_pcp_scheduled_tokens = []
        ori_start_index = 0
        pad_start_index = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            req_position_pcp, num_pcp_padded_scheduled_tokens, num_pcp_pad = \
                _pcp_pad_and_split(ori_num_tokens)
            actual_num_tokens = len(req_position_pcp)
            num_pcp_scheduled_tokens.append(actual_num_tokens)
            pad_input_ids = F.pad(
                input_ids[ori_start_index:ori_start_index + ori_num_tokens],
                (0, num_pcp_pad))
            ori_start_index += ori_num_tokens
            pcp_chunk_indices = [
                pad_start_index + pos for pos in req_position_pcp
            ]
            pcp_split_input_ids = pad_input_ids[req_position_pcp]
            pcp_split_hidden_states = target_hidden_states[pcp_chunk_indices]
            pcp_split_input_ids_list.append(pcp_split_input_ids)
            pcp_split_hidden_states_list.append(pcp_split_hidden_states)
            pad_start_index += num_pcp_padded_scheduled_tokens
        num_tokens = sum(num_pcp_scheduled_tokens)
        input_ids = torch.cat(pcp_split_input_ids_list)
        target_hidden_states = torch.cat(pcp_split_hidden_states_list, dim=0)
        max_query_len = max(num_pcp_scheduled_tokens)
        seq_lens = torch.tensor(num_pcp_scheduled_tokens, dtype=torch.int32)
        cu_num_tokens = torch.tensor(
            np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens
