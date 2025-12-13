# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from vllm.attention.layer import Attention
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType

PADDING_SLOT_ID = -1

_DEFAULT_FIRST_LAYER = 'model.layers.0.self_attn.attn'

_FIRST_LAYERS = {"Qwen3NextForCausalLM": 'model.layers.3.self_attn.attn'}


class EagleProposer(Proposer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None):
        self.name = SpecDcodeType.EAGLE if vllm_config.speculative_config.method == "eagle" else SpecDcodeType.EAGLE3
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner

        self.block_size = vllm_config.cache_config.block_size
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size(
        )

        self.use_cuda_graph = (self.vllm_config.compilation_config.mode
                               == CompilationMode.VLLM_COMPILE and
                               not self.vllm_config.model_config.enforce_eager)

        self.cudagraph_batch_sizes = list(
            sorted(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Currently we do not use pcp. This is used to adapt the pcp branch.
        self.pcp_size = 0
        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        self.decode_threshold = 1 + \
            self.vllm_config.speculative_config.num_speculative_tokens

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device)
        self.positions = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device)
        self.hidden_states = torch.zeros(
            (self.vllm_config.scheduler_config.max_num_batched_tokens,
             self.hidden_size),
            dtype=self.vllm_config.model_config.dtype,
            device=device)
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.token_arange_np = np.arange(self.max_num_tokens)
        max_num_slots_for_arange = max(self.max_num_tokens, max_batch_size + 1)
        self.arange = torch.arange(max_num_slots_for_arange,
                                   device=device,
                                   dtype=torch.int32)
        self.arange_cpu = torch.arange(max_num_slots_for_arange,
                                       device="cpu",
                                       dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    def load_model(self, model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=self.vllm_config.
                               speculative_config.draft_model_config)
        draft_attn_layer_names = (get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase).keys() -
                                  target_attn_layer_names)
        self.attn_layer_name = next(iter(draft_attn_layer_names))

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            logger.info(
                "The EAGLE head shares the same vocab embedding" \
                " with the target model."
            )
            self.model.model.embed_tokens = model.model.embed_tokens
        else:
            logger.info(
                "Since PP > 1, the EAGLE head loaded its own vocab embedding" \
                " weights instead of sharing them with the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.name == SpecDcodeType.EAGLE and hasattr(model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            if supports_multimodal(model):
                self.model.lm_head = model.get_language_model().lm_head
            else:
                self.model.lm_head = model.lm_head

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp: Optional[torch.Tensor] = None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None):
        moe_comm_type = self.runner._select_moe_comm_method(num_tokens)
        with set_ascend_forward_context(None,
                                        self.vllm_config,
                                        moe_comm_type=moe_comm_type,
                                        num_tokens=num_tokens):
            self.model(
                input_ids=self.input_ids[:num_tokens],
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )
            dummy_compute_logits(self.hidden_states)

    def generate_token_ids(self,
                           sampled_token_ids: torch.Tensor | list[list[int]],
                           sampling_metadata: SamplingMetadata = None,
                           scheduler_output: SchedulerOutput = None,
                           spec_decode_metadata: SpecDecodeMetadata = None,
                           positions: torch.Tensor = None,
                           num_scheduled_tokens: int = 0,
                           hidden_states: torch.Tensor = None,
                           attn_metadata=None,
                           aux_hidden_states: torch.Tensor = None):
        common_attn_metadata = self.runner.spec_decode_common_attn_metadata

        if self.vllm_config.speculative_config.disable_padded_drafter_batch:
            # When padded-batch is disabled, the sampled_token_ids should be
            # the cpu-side list[list[int]] of valid sampled tokens for each
            # request, with invalid requests having empty lists.
            assert isinstance(sampled_token_ids, list), \
                "sampled_token_ids should be a python list when" \
                "padded-batch is disabled."
            next_token_ids = self.prepare_next_token_ids_cpu(
                sampled_token_ids, self.runner.requests,
                self.runner.input_batch, scheduler_output.num_scheduled_tokens)
        else:
            # When using padded-batch, the sampled_token_ids should be
            # the gpu tensor of sampled tokens for each request, of shape
            # (num_reqs, num_spec_tokens + 1) with rejected tokens having
            # value -1.
            assert isinstance(sampled_token_ids, torch.Tensor), \
                "sampled_token_ids should be a torch.Tensor when" \
                "padded-batch is enabled."
            next_token_ids, valid_sampled_tokens_count = \
                self.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.runner.requests,
                    self.runner.input_batch,
                    self.runner.discard_request_indices.gpu,
                    self.runner.num_discarded_requests
                )
            self._copy_valid_sampled_token_count(next_token_ids,
                                                 valid_sampled_tokens_count)

        req_scheduled_tokens = scheduler_output.num_scheduled_tokens
        if self.pcp_size > 1:
            long_seq_metadata = self.runner.long_seq_metadata
            input_ids_pcp_full = self.runner.input_ids_pcp_full
            query_start_loc_pcp_full = self.runner.query_start_loc_pcp_full
            query_start_loc_pcp_full_cpu = self.runner.query_start_loc_pcp_full_cpu
            num_reqs = self.runner.input_batch.num_reqs
            ori_query_lens = query_start_loc_pcp_full_cpu[1:num_reqs+1] - \
                query_start_loc_pcp_full_cpu[:num_reqs]
            num_prefill_reqs = (ori_query_lens
                                > self.decode_threshold).sum().item()
            num_decode_reqs = num_reqs - num_prefill_reqs
        else:
            long_seq_metadata = None
            num_prefill_reqs = 0
            num_decode_reqs = 0
        if spec_decode_metadata is None:
            # update pcp related params
            if self.pcp_size > 1:
                token_indices_to_sample = \
                    query_start_loc_pcp_full_cpu[1:num_reqs + 1] - 1
                target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                target_hidden_states = hidden_states
            else:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.runner.input_ids.gpu[:
                                                             num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                if self.name == SpecDcodeType.EAGLE3:
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
        else:
            if self.pcp_size > 1:
                common_attn_metadata.query_start_loc_cpu = \
                    query_start_loc_pcp_full_cpu[:num_reqs + 1]
                common_attn_metadata.query_start_loc = \
                    query_start_loc_pcp_full[:num_reqs + 1]
            if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                token_indices_to_sample = None
                common_attn_metadata, token_indices =\
                    self.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens)
            else:
                common_attn_metadata, token_indices, \
                    token_indices_to_sample =\
                        self.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count)
            if self.pcp_size > 1:
                target_token_ids = input_ids_pcp_full[token_indices]
                target_positions = positions
                target_hidden_states = hidden_states
            else:
                target_token_ids = self.runner.input_ids.gpu[token_indices]
                target_positions = positions[token_indices]
                if self.name == SpecDcodeType.EAGLE3:
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]

        draft_token_ids = self._propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            last_token_indices=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
            scheduler_output=scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        return draft_token_ids

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
        last_token_indices: Optional[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: Optional[tuple[list[torch.Tensor],
                                        torch.Tensor]] = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
    ) -> torch.Tensor:

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.name == SpecDcodeType.EAGLE3:
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        if self.use_cuda_graph and \
            num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens

        moe_comm_type = self.runner._select_moe_comm_method(num_input_tokens)

        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        builder = self.runner.attn_groups[0][0].get_metadata_builder()
        attn_metadata = builder.build(0, common_attn_metadata,
                                      self.runner.get_model())

        with set_ascend_forward_context(attn_metadata,
                                        self.vllm_config,
                                        moe_comm_type=moe_comm_type,
                                        num_tokens=num_input_tokens):
            last_hidden_states, hidden_states = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self.positions[:num_input_tokens],
                hidden_states=self.hidden_states[:num_input_tokens],
            )
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.vllm_config.speculative_config.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_tensor = torch.zeros(
            (self.vllm_config.speculative_config.num_speculative_tokens,
             *draft_token_ids.shape),
            dtype=draft_token_ids.dtype,
            device=self.device)
        draft_token_ids_tensor[0] = draft_token_ids

        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]
        last_token_indices = self.arange[:batch_size]

        if self.use_cuda_graph and \
            batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size

        moe_comm_type = self.runner._select_moe_comm_method(input_batch_size)

        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange_cpu[:batch_size + 1]
        attn_metadata.query_start_loc_list = attn_metadata.query_start_loc[
            1:].tolist()
        attn_metadata.num_decodes, attn_metadata.num_prefills, attn_metadata.num_decode_tokens, attn_metadata.num_prefill_tokens = 0, batch_size, 0, batch_size
        attn_metadata.num_actual_tokens_pcp_padded = attn_metadata.num_decode_tokens + attn_metadata.num_prefill_tokens

        attn_metadata.actual_seq_lengths_q = [1 + i for i in range(batch_size)]
        attn_metadata.seq_lens_list = attn_metadata.seq_lens.tolist()
        attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        for now_speculative in range(
                self.vllm_config.speculative_config.num_speculative_tokens -
                1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_tensor[now_speculative]
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= self.vllm_config.model_config.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            # TODO: Increment the sequence lengths.

            attn_metadata.seq_lens += 1
            attn_metadata.seq_lens_list = [
                _ + 1 for _ in attn_metadata.seq_lens_list
            ]
            # TODO: Consider max model length.
            # attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
            #                                 self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # TODO: sequence length to 1 to minimize their overheads in attention.

            # Compute the slot mapping.
            block_numbers = (clamped_positions // self.block_size)
            block_ids = attn_metadata.block_tables.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            slot_mapping_tmp = (
                block_ids * self.vllm_config.cache_config.block_size +
                clamped_positions % self.block_size)

            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping_tmp.masked_fill_(exceeds_max_model_len,
                                          PADDING_SLOT_ID)
            # NOTE: ASCEND slot_mapping must on cpu
            attn_metadata.slot_mapping[:slot_mapping_tmp.shape[0]].copy_(
                slot_mapping_tmp.to(torch.int32))
            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask()

            attn_metadata.attn_mask = attn_mask
            # Run the model.
            with set_ascend_forward_context(attn_metadata,
                                            self.vllm_config,
                                            moe_comm_type=moe_comm_type,
                                            num_tokens=input_batch_size):

                last_hidden_states, hidden_states = self.model(
                    input_ids=self.input_ids[:input_batch_size],
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                )
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])

            # TODO(wenlong): get more than one token for tree attention
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_tensor[now_speculative + 1] = draft_token_ids

        # [batch_size, num_speculative_tokens]
        draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
        return draft_token_ids

    def _get_attn_metadata(self, attn_metadata):
        if attn_metadata is not None and isinstance(attn_metadata, dict):
            architecture = self.vllm_config.model_config.architecture
            layer_name = _FIRST_LAYERS.get(architecture, _DEFAULT_FIRST_LAYER)
            attn_metadata = attn_metadata[layer_name]

        return attn_metadata

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[
                    req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.input_ids.device)
        return next_token_ids

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
        self.backup_next_token_ids.np[:num_reqs] = np.array([
            requests[gpu_input_batch.req_ids[i]].get_token_id(
                common_attn_metadata.seq_lens_cpu[i].item())
            for i in range(num_reqs)
        ])
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[:
                                                                     num_discarded_requests]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(
            0, discard_sampled_tokens_req_indices, -1)

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1,
            last_valid_indices_safe.unsqueeze(1)).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count

    def _copy_valid_sampled_token_count(
            self, next_token_ids: torch.Tensor,
            valid_sampled_tokens_count: torch.Tensor) -> None:
        if self.runner.valid_sampled_token_count_event is not None:
            default_stream = torch.npu.current_stream()
            # initialize a new stream to overlap the copy operation with
            # prepare_input of draft model.
            with torch.npu.stream(
                    self.runner.valid_sampled_token_count_copy_stream):
                self.runner.valid_sampled_token_count_copy_stream.wait_stream(
                    default_stream)  # type: ignore
                self.runner.valid_sampled_token_count_cpu[:
                                                          valid_sampled_tokens_count
                                                          .shape[0]].copy_(
                                                              valid_sampled_tokens_count,
                                                              non_blocking=True
                                                          )
                self.runner.valid_sampled_token_count_event.record()

            self.runner.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(
                1)

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
        num_rejected_tokens = torch.tensor(num_rejected_tokens,
                                           dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_actual_reqs
                                                                       + 1]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_actual_reqs]
        new_seq_lens_cpu = seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[
            1:] - query_start_loc_cpu[:-1]
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
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                                  new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = (self.token_arange_np[:total_num_tokens] -
                         new_query_start_locs_expanded)

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True)

        common_attn_metadata.slot_mapping[:token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices])
        common_attn_metadata.slot_mapping[token_indices.shape[0]:].fill_(-1)

        # NOTE: Currently positions and seq_lens are not used in mla_v1 forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device,
                                                       non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions[token_indices],
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
        )
        return spec_common_attn_metadata, token_indices

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
        num_draft_tokens_gpu = torch.cat([
            spec_decode_metadata.cu_num_draft_tokens[0:1],
            spec_decode_metadata.cu_num_draft_tokens[1:] -
            spec_decode_metadata.cu_num_draft_tokens[:-1],
        ])

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[
            1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        # NOTE: Currently positions and seq_lens are not used in mla_v1 forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            attn_mask=self.runner.attn_mask,
            spec_attn_mask=self.runner.spec_attn_mask,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens)

        token_indices_to_sample = (common_attn_metadata.query_start_loc[1:] -
                                   1 - num_rejected_tokens_gpu)

        return spec_common_attn_metadata, token_indices, token_indices_to_sample
