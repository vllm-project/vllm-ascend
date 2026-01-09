# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID
from vllm.v1.spec_decode.eagle import EagleProposer as VllmEagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               update_attn_dcp_pcp_params,
                                               update_attn_params,
                                               update_mla_attn_dcp_pcp_params,
                                               update_mla_attn_params)
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.ops.triton.spec_decode.utils import \
    prepare_inputs_padded_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.utils import shared_expert_dp_enabled

# Currently we will fix block size to a small one since `num_reqs` can't be too large
_PREPARE_INPUTS_BLOCK_SIZE = 4


class EagleProposer(VllmEagleProposer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 device: torch.device,
                 runner=None):
        super().__init__(vllm_config, device, runner)

        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
        # there is synchronization between mtp steps when enabling aclgraph,
        # disable aclgraph when use async scheduling to avoid the
        # synchronization overhead.
        # NOTE: we need to set aclgraph_runtime_mode to None in both dummy_run
        # and _propose.
        self.use_cuda_graph = (
            self.vllm_config.compilation_config.mode
            == CompilationMode.VLLM_COMPILE
            and not self.vllm_config.model_config.enforce_eager
            and not self.use_async_scheduling
            and not self.vllm_config.speculative_config.enforce_eager)

        self.cudagraph_batch_sizes = list(
            sorted(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        self.pcp_size = self.runner.pcp_size
        self.decode_threshold = 1 + self.num_speculative_tokens
        self.query_start_loc = self.runner._make_buffer(
            self.runner.max_num_reqs + 1, dtype=torch.int32)
        self.arange_cpu = torch.arange(self.arange.shape[0],
                                       device="cpu",
                                       dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.enable_shared_expert_dp = shared_expert_dp_enabled()

        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank
        self.dcp_rank = self.runner.dcp_rank

        self.use_aclgraph = self.runner._use_aclgraph()

        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size +
            self.pcp_size * self.dcp_size * self.runner.max_num_reqs)

        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config,
                                  "index_topk")

    def load_model(self, model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        DeepseekV32IndexerCache).keys())

        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=self.vllm_config.
                               speculative_config.draft_model_config)

        indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, DeepseekV32IndexerCache).keys()
        draft_attn_layer = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase).keys()

        draft_attn_layer_names = draft_attn_layer - target_attn_layer_names
        draft_indexer_layer_names = indexer_layers - target_indexer_layer_names
        draft_attn_layer_names = draft_attn_layer_names - draft_indexer_layer_names
        assert len(draft_attn_layer_names) == 1
        self.attn_layer_name = list(draft_attn_layer_names)
        self.attn_layer_names = self.attn_layer_name

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            if self.method == "mtp":
                if self.vllm_config.model_config.is_deepseek_mla and \
                    torch.equal(self.model.model.embed_tokens.weight,
                                model.model.embed_tokens.weight):
                    # If pp>1, the weights of mtp and the main model's embedding are not on the same device.
                    # check if mtp model use main model's embedding and LMhead
                    logger.info(
                        "The MTP head shares the same vocab embedding" \
                        " with the target model."
                    )
                    self.model.model.embed_tokens = model.model.embed_tokens
                else:
                    logger.info(
                        " The MTP head loaded its own vocab embedding" \
                        " weights instead of sharing them with the target model."
                    )
            else:
                logger.info(
                    "The EAGLE head shares the same vocab embedding" \
                    " with the target model."
                )
                self.model.model.embed_tokens = model.model.embed_tokens
        else:
            logger.info(
                "Since PP > 1 or other reasons the model head loaded its own vocab embedding" \
                " weights instead of sharing them with the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.method == "eagle" and hasattr(model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            if supports_multimodal(model):
                self.model.lm_head = model.get_language_model().lm_head
            else:
                self.model.lm_head = model.lm_head

        if self.method == "mtp" and \
            self.vllm_config.model_config.is_deepseek_mla:
            for _, layer_module in self.model.model.layers.items():
                if torch.equal(layer_module.shared_head.head.weight,
                               model.lm_head.weight):
                    layer_module.shared_head.head = model.lm_head

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ) and self.use_cuda_graph:
            self.update_stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    @torch.inference_mode()
    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  in_graph_capturing: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp: Optional[torch.Tensor] = None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None,
                  is_profile=False):
        # update global cos, sin
        update_cos_sin(self.positions[:num_tokens])

        attn_metadata = None
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(
                self.runner.attn_groups) > 0:
            num_computed_tokens_cpu = (
                self.runner.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs])
            self.query_start_loc.cpu[:num_reqs + 1] = torch.tensor(
                [0] + self.runner.actual_seq_lengths_q[:num_reqs],
                device="cpu",
                dtype=torch.int32)
            self.query_start_loc.copy_to_gpu()
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.runner.seq_lens.cpu,
                seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                max_query_len=self.num_speculative_tokens + 1,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].
                get_device_tensor()[:num_reqs],
                slot_mapping=self.runner.input_batch.block_table[0].
                slot_mapping.gpu,
                positions=self.runner.positions.gpu,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
                max_seq_len=0,
            )

            builder = self.runner.attn_groups[0][0].get_metadata_builder()
            attn_metadata_eagle = builder.build_for_graph_capture(
                common_attn_metadata, AscendAttentionState.ChunkedPrefill)
            attn_metadata = {}
            for layer_name in self.attn_layer_name:
                attn_metadata[layer_name] = attn_metadata_eagle

        model_input_ids = self.input_ids[:num_tokens]
        model_positions = self.positions[:num_tokens]
        model_previous_hidden_states = self.hidden_states[:num_tokens]
        for i in range(self.num_speculative_tokens):
            if i > 0 and in_graph_capturing and aclgraph_runtime_mode == CUDAGraphMode.FULL:
                aclgraph_runtime_mode = CUDAGraphMode.NONE
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_actual_tokens=0,
                    in_profile_run=is_profile,
                    batch_descriptor=batch_descriptor,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    is_draft_model=True):

                if self.enable_shared_expert_dp:
                    model_previous_hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
                        model_previous_hidden_states)

                self.model(
                    input_ids=model_input_ids,
                    positions=model_positions,
                    hidden_states=model_previous_hidden_states,
                )
                forward_context = get_forward_context()
                if (forward_context.cudagraph_runtime_mode
                        == CUDAGraphMode.FULL
                        and not forward_context.capturing):
                    update_attn_params(
                        self.update_stream,
                        forward_context,
                        num_tokens,
                        self.vllm_config,
                    )

                if self.enable_shared_expert_dp:
                    model_previous_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        model_previous_hidden_states, True)

                dummy_compute_logits(self.hidden_states)

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

        if self.method == "eagle3":
            assert isinstance(self.get_model(), Eagle3LlamaForCausalLM)
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

        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = \
                self.runner.cudagraph_dispatcher.dispatch(num_tokens=num_input_tokens, uniform_decode=True, has_lora=has_lora)
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None

        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        builder = self.runner.attn_groups[0][0].get_metadata_builder()
        attn_metadata = builder.build(0, common_attn_metadata,
                                      self.runner.get_model())
        # update global cos, sin
        update_cos_sin(self.positions[:num_input_tokens])
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_name:
            per_layer_attn_metadata[layer_name] = attn_metadata
        with set_ascend_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_actual_tokens=num_tokens,
                batch_descriptor=batch_descriptor,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                is_draft_model=True):

            # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all speculative tokens' proposings.
            # `model_input_ids`, `model_positions` and `model_hidden_states` are used to represent the inputs of speculative model.
            model_input_ids = self.input_ids[:num_input_tokens]
            model_positions = self.positions[:num_input_tokens]
            model_hidden_states = self.hidden_states[:num_input_tokens]

            if self.enable_shared_expert_dp:
                # split hidden states along sequence dimension
                # positions should not be split?
                model_hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
                    model_hidden_states)
                # in acl-graph, `model_hidden_states` should be copy back to `self.hidden_states`?

            last_hidden_states, hidden_states = self.model(
                input_ids=model_input_ids,
                positions=model_positions,
                hidden_states=model_hidden_states,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                # TODO: support mla in future.
                update_attn_params(
                    self.update_stream,
                    forward_context,
                    num_input_tokens,
                    self.vllm_config,
                )

            if self.enable_shared_expert_dp:
                # merge hidden states along sequence dimension
                last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    last_hidden_states.contiguous(), True)
                hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    hidden_states.contiguous(), True)

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_token_ids.shape),
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

        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange_cpu[:batch_size + 1]
        attn_metadata.num_decodes, attn_metadata.num_prefills, attn_metadata.num_decode_tokens, attn_metadata.num_prefill_tokens = 0, batch_size, 0, batch_size
        attn_metadata.num_actual_tokens_pcp_padded = attn_metadata.num_decode_tokens + attn_metadata.num_prefill_tokens

        attn_metadata.actual_seq_lengths_q = attn_metadata.query_start_loc[
            1:].tolist()
        attn_metadata.seq_lens_list = attn_metadata.seq_lens.tolist()
        attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = \
                self.runner.cudagraph_dispatcher.dispatch(num_tokens=input_batch_size, uniform_decode=True, has_lora=has_lora)
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        for now_speculative in range(self.num_speculative_tokens - 1):
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

            attn_metadata.seq_lens = attn_metadata.seq_lens + 1
            attn_metadata.seq_lens_list = [
                _ + 1 for _ in attn_metadata.seq_lens_list
            ]
            # TODO: Consider max model length.
            # attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
            #                                 self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # TODO: sequence length to 1 to minimize their overheads in attention.

            if self.attn_metadata_builder is None:
                attn_metadata_builder = self._get_attention_metadata_builder()
            else:
                attn_metadata_builder = self.attn_metadata_builder
            block_size = attn_metadata_builder.kv_cache_spec.block_size

            # Compute the slot mapping.
            block_numbers = (clamped_positions // block_size)
            block_ids = attn_metadata.block_tables.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            slot_mapping_tmp = (block_ids * block_size +
                                clamped_positions % block_size)

            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping_tmp.masked_fill_(exceeds_max_model_len,
                                          PADDING_SLOT_ID)
            # NOTE: ASCEND slot_mapping must on cpu
            attn_metadata.slot_mapping[:slot_mapping_tmp.shape[0]].copy_(
                slot_mapping_tmp.to(torch.int32))
            attn_metadata.slot_mapping[slot_mapping_tmp.shape[0]:].fill_(
                PADDING_SLOT_ID)
            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            attn_mask = self.attn_mask_builder.get_splitfuse_attn_mask()

            attn_metadata.attn_mask = attn_mask

            # update global cos, sin
            update_cos_sin(self.positions[:input_batch_size])

            # Run the model.
            with set_ascend_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    num_actual_tokens=batch_size,
                    batch_descriptor=batch_descriptor,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    is_draft_model=True):

                # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all speculative tokens' proposings.
                # `model_input_ids`, `model_positions` and `model_hidden_states` are used to represent the inputs of speculative model.
                model_input_ids = self.input_ids[:input_batch_size]
                model_positions = self.positions[:input_batch_size]
                model_hidden_states = self.hidden_states[:input_batch_size]

                if self.enable_shared_expert_dp:
                    # split hidden states along sequence dimension
                    # positions should not be split？
                    model_hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
                        model_hidden_states)
                    # in acl-graph, `model_hidden_states` should be copy back to `self.hidden_states`?

                last_hidden_states, hidden_states = self.model(
                    input_ids=model_input_ids,
                    positions=model_positions,
                    hidden_states=model_hidden_states,
                )
                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    update_attn_params(
                        self.update_stream,
                        forward_context,
                        input_batch_size,
                        self.vllm_config,
                    )

                if self.enable_shared_expert_dp:
                    # merge hidden states along sequence dimension
                    last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        last_hidden_states.contiguous(), True)
                    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        hidden_states.contiguous(), True)

            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])

            # TODO(wenlong): get more than one token for tree attention
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_tensor[now_speculative + 1] = draft_token_ids

        # [batch_size, num_speculative_tokens]
        draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
        return draft_token_ids

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

        # NOTE: Currently positions and seq_lens are not used in attn forward
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
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            max_seq_len=0)
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
        if HAS_TRITON:
            num_reqs = common_attn_metadata.num_reqs
            device = valid_sampled_tokens_count.device

            token_indices_to_sample = torch.empty((num_reqs, ),
                                                  dtype=torch.int32,
                                                  device=device)

            num_blocks_needed = triton.cdiv(num_reqs,
                                            _PREPARE_INPUTS_BLOCK_SIZE)
            num_vector_core = get_vectorcore_num()
            grid_size = min(num_blocks_needed, num_vector_core)
            grid = (grid_size, )

            prepare_inputs_padded_kernel[grid](
                spec_decode_metadata.cu_num_draft_tokens,
                valid_sampled_tokens_count,
                common_attn_metadata.query_start_loc,
                token_indices_to_sample,
                num_reqs,
                BLOCK_SIZE=_PREPARE_INPUTS_BLOCK_SIZE,
            )
        else:
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

            token_indices_to_sample = (
                common_attn_metadata.query_start_loc[1:] - 1 -
                num_rejected_tokens_gpu)

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[
            1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        # NOTE: Currently positions and seq_lens are not used in attn forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens
            if self.pcp_size > 1 else total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=0)

        return spec_common_attn_metadata, token_indices, token_indices_to_sample

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

    # update full-graph params for one spec token
    def _update_full_graph_params(self, forward_context, num_tokens):
        if self.vllm_config.model_config.use_mla:
            if self.pcp_size * self.dcp_size > 1:
                update_mla_attn_dcp_pcp_params(self.update_stream,
                                               forward_context, num_tokens)
            else:
                update_mla_attn_params(self.update_stream, forward_context,
                                       num_tokens,
                                       self.vllm_config.speculative_config)
        else:
            if self.pcp_size * self.dcp_size > 1:
                update_attn_dcp_pcp_params(self.update_stream, forward_context,
                                           num_tokens)
            else:
                update_attn_params(self.update_stream, forward_context,
                                   num_tokens, self.vllm_config)
