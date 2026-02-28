# SPDX-License-Identifier: Apache-2.0
import copy
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
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
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, update_full_graph_params
from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.utils import enable_sp, lmhead_tp_enable, shared_expert_dp_enabled

# Currently we will fix block size to a small one since `num_reqs` can't be too large
_PREPARE_INPUTS_BLOCK_SIZE = 4


# TODO: Remove it when the bug of fx-graph is solved
# patch vllm_config to be in CompilationMode.NONE temporarily
@contextmanager
def _maybe_eager_context(vllm_config):
    raw_compilation_config_mode = vllm_config.compilation_config.mode
    vllm_config.compilation_config.mode = CompilationMode.NONE
    try:
        yield
    finally:
        vllm_config.compilation_config.mode = raw_compilation_config_mode


# split hidden states along dimension of sequence
def split_inputs_tp_to_sp(hidden_states, out):
    # tp and sp share the same group
    group = get_tp_group()

    world_size = group.world_size
    rank = group.rank

    num_tokens = hidden_states.shape[0]
    # the size per rank after padded
    padded_num_tokens_per_rank = (num_tokens + world_size - 1) // world_size
    # compute the start and end of slice
    start = padded_num_tokens_per_rank * rank
    end = padded_num_tokens_per_rank * (rank + 1)

    # copy only hidden_states in current rank
    hidden_states_curr_rank = hidden_states[start:end]
    out[: hidden_states_curr_rank.shape[0]] = hidden_states_curr_rank
    return out[:padded_num_tokens_per_rank]


class EagleProposer(VllmEagleProposer):
    _runnable: ACLGraphWrapper | Callable

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner)

        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling

        self.decode_threshold = 1 + self.num_speculative_tokens
        self.query_start_loc = self.runner._make_buffer(self.runner.max_num_reqs + 1, dtype=torch.int32)
        self.arange_cpu = torch.arange(self.arange.shape[0], device="cpu", dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.enable_shared_expert_dp = shared_expert_dp_enabled()

        self.pcp_size = self.runner.pcp_size
        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank
        self.dcp_rank = self.runner.dcp_rank

        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size
            + self.pcp_size * self.dcp_size * self.runner.max_num_reqs
        )

        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        # NOTE:
        # `draft_tensor_parallel_size` does not take effect for Eagle:
        # the draft model uses the same TP size as the target model in practice.
        # so we applied this patch to set tp=1 of draft model separately.
        # Due to verification of `_verify_and_get_draft_tp` in vllm,
        # the value of `draft_tensor_parallel_size` here will either be 1 separately
        # or the same as target model.
        # TODO(zhaomingyu13): If we want to adapt to the case where draft model tp
        # is not 1 and differs from target model, this part should be rewritten.
        if vllm_config.parallel_config.tensor_parallel_size != self.speculative_config.draft_tensor_parallel_size:
            tp_group = init_model_parallel_group(
                [[get_world_group().rank]],
                get_world_group().rank,
                torch.distributed.get_backend(get_world_group().device_group),
                use_message_queue_broadcaster=True,
                group_name="tp",
            )
            self.tp_group_context = patch_tensor_parallel_group(tp_group)
        else:
            self.tp_group_context = nullcontext()

        self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
        if self.method == "mtp":
            self.use_cuda_graph = self.use_cuda_graph and not self.use_async_scheduling

        # TODO: Remove it when the bug of fx-graph is solved
        self.maybe_eager_context: AbstractContextManager[Any] = nullcontext()
        if not self.use_cuda_graph and enable_sp(vllm_config):
            self.maybe_eager_context = _maybe_eager_context(vllm_config)

        self.last_token_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.int32, device=device
        )
        slot_mapping_lens = self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs
        self.slot_mapping_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]

        self._runnable = self._run_merged_draft

    def load_model(self, model: nn.Module) -> None:
        target_attn_layer_names = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys())
        target_indexer_layer_names = set(get_layers_from_vllm_config(self.vllm_config, DeepseekV32IndexerCache).keys())

        with self.maybe_eager_context:
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.speculative_config.draft_model_config
            )

        indexer_layers = get_layers_from_vllm_config(self.vllm_config, DeepseekV32IndexerCache).keys()
        draft_attn_layer = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()

        draft_attn_layer_names = draft_attn_layer - target_attn_layer_names
        draft_indexer_layer_names = indexer_layers - target_indexer_layer_names
        draft_attn_layer_names = draft_attn_layer_names - draft_indexer_layer_names
        assert len(draft_attn_layer_names) == 1
        self.attn_layer_names = list(sorted(draft_attn_layer_names))
        self.piece_all_attn_layer_name = []
        for _ in range(self.num_speculative_tokens):
            self.piece_all_attn_layer_name.append([name for name in self.attn_layer_names])
        self.attn_layer_names = list(sorted(draft_attn_layer_names))

        self.piece_all_attn_layer_name = []
        for _ in range(self.num_speculative_tokens):
            self.piece_all_attn_layer_name.append([name for name in self.attn_layer_names])

        if supports_multimodal(model):
            # handle multimodality
            if self.get_model_name(model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = model.config.image_token_id
            elif self.get_model_name(model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = model.config.vision_config.image_token_id
            else:
                self.model.config.image_token_index = model.config.image_token_index
            target_language_model = model.get_language_model()
        else:
            target_language_model = model

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            if hasattr(target_language_model.model, "embed_tokens"):
                target_embed_tokens = target_language_model.model.embed_tokens
            elif hasattr(target_language_model.model, "embedding"):
                target_embed_tokens = target_language_model.model.embedding
            else:
                raise AttributeError("Target model does not have 'embed_tokens' or 'embedding' attribute")
            # If pp>1, the weights of mtp and the main model's embedding are not on the same device.
            # check if mtp model use main model's embedding and LMhead
            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra NPU memory
                    # usage in CI testing environments with limited NPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info("Detected MTP model. Sharing target model embedding weights with the draft model.")

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "Since PP > 1 or other reasons the model head loaded its own vocab embedding"
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

        if self.method == "mtp" and self.vllm_config.model_config.is_deepseek_mla:
            for _, layer_module in self.model.model.layers.items():
                if torch.equal(layer_module.shared_head.head.weight, model.lm_head.weight):
                    layer_module.shared_head.head = model.lm_head

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
            self.update_stream = torch.npu.Stream()
            if self.method == "mtp":
                self.model = ACLGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
            else:
                self._runnable = ACLGraphWrapper(
                    self._run_merged_draft, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
                )

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def shallow_copy_metadata(self, attn_metadata):
        # Currently, new objects will be assigned to the lists in attn_metadata
        # when update. So we can use the shallow copy.
        return copy.copy(attn_metadata)

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
        (
            num_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)

        multi_steps_attn_metadata = []
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
            self.query_start_loc.cpu[: num_reqs + 1] = torch.tensor(
                [0] + self.runner.actual_seq_lengths_q[:num_reqs], device="cpu", dtype=torch.int32
            )
            self.query_start_loc.copy_to_gpu()
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                seq_lens_cpu=self.runner.seq_lens.cpu,
                seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                max_query_len=self.num_speculative_tokens + 1,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                # This is used to hold a position.
                slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                positions=self.runner.positions.gpu,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
                max_seq_len=0,
            )
            if self.pcp_size * self.dcp_size > 1:
                # update long_seq related params and flatten block_table
                common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata
                common_attn_metadata.block_table_tensor = self.runner.input_batch.block_table[0].get_device_tensor()[
                    : num_reqs * self.decode_threshold
                ]

            builder = self.runner.attn_groups[0][0].get_metadata_builder()
            # update the tensor's address for each step.
            for draft_step in range(self.num_speculative_tokens):
                common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
                # Set the real slot_mapping.
                common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
                attn_metadata_eagle = builder.build_for_graph_capture(
                    common_attn_metadata, AscendAttentionState.ChunkedPrefill
                )
                per_layer_attn_metadata = dict()
                for layer_name in self.attn_layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata_eagle
                multi_steps_attn_metadata.append(per_layer_attn_metadata)

        model_positions = self._get_positions(num_tokens)

        batch_size = num_tokens // (self.num_speculative_tokens + 1)  # if not is_profile else self.runner.max_num_reqs
        if is_profile:
            batch_size = min(batch_size, self.runner.max_num_reqs)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            # Reset MOE layer index before first model call
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            self._runnable(
                num_input_tokens=num_tokens,
                batch_size=batch_size,
                last_token_indices=self.last_token_indices[:batch_size],
                # The target_position's address is same as the model_positions's
                target_positions=model_positions,
                inputs_embeds=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                is_dummy=True,
                num_tokens=num_tokens,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not forward_context.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

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
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.get_model(), Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None
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
        if self.use_cuda_graph and num_tokens <= self.runner.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.runner.cudagraph_dispatcher._bs_to_padded_graph_size[num_tokens]
            if not (
                self.speculative_config.disable_padded_drafter_batch
                and self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
            ):
                # TODO: Due to the inconsistency between the proposer `dispatcher` and model runner, this padding
                # should have been done in model runner but not. For example, at prefill stage, target model
                # is run in eager mode currently, which means `_pad_query_start_loc_for_fia` is not called,
                # while draft model is run in graph model, which means we should pad the `query_start_loc`.
                # Need to be fixed in the future.
                num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
                    num_input_tokens, common_attn_metadata.num_reqs, common_attn_metadata.num_reqs
                )
                common_attn_metadata.num_reqs = num_reqs_padded
                common_attn_metadata.query_start_loc = self.runner.query_start_loc.gpu[: num_reqs_padded + 1]
                common_attn_metadata.query_start_loc_cpu = self.runner.query_start_loc.cpu[: num_reqs_padded + 1]
        else:
            num_input_tokens = num_tokens

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_input_tokens, is_draft_model=True)

        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens, uniform_decode=True, has_lora=has_lora
            )
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
            inputs_embeds = self.model.embed_input_ids(
                self.input_ids[:num_tokens], multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
            )
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            inputs_embeds = None

        # Update slot_mapping for different speculative.
        # NOTE: Currently, we only remake the slot_mapping, because it's the
        # only tensor which will be used in current FIA.
        # Strictly speaking, `query_start_loc`, `seq_lens` should also have
        # their memory allocated separately for each step just like `slot_mapping`.
        slot_mapping_lens = common_attn_metadata.slot_mapping.shape[0]
        self.slot_mapping_group[0][:slot_mapping_lens].copy_(common_attn_metadata.slot_mapping[:slot_mapping_lens])
        self.slot_mapping_group[0][slot_mapping_lens:].fill_(-1)
        common_attn_metadata.slot_mapping = self.slot_mapping_group[0][:slot_mapping_lens]
        common_attn_metadata.num_input_tokens = num_input_tokens
        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        builder = self.runner.attn_groups[0][0].get_metadata_builder()
        attn_metadata = builder.build(0, common_attn_metadata, self.runner.get_model())

        if self.uses_mrope:
            used_update_positions = target_positions[:, last_token_indices]
        else:
            used_update_positions = target_positions[last_token_indices]
        per_layer_attn_metadata = dict()
        # The first step of speculative.
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        multi_steps_attn_metadata = [per_layer_attn_metadata]

        attn_metadata_i = per_layer_attn_metadata[self.attn_layer_names[0]]
        if self.pcp_size * self.dcp_size > 1:
            if self.num_speculative_tokens > 1 and not attn_metadata_i.num_prefills:
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
                ori_seq_len = attn_metadata_i.seq_lens[:batch_size].clone()
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
                        torch.arange(slot_idx_base[req_id], slot_idx_base[req_id] + self.pcp_size, device=self.device)
                    )
                slot_indices = torch.cat(slot_indices_list, dim=0)

                # fold block_table (restore it to original size before flattened)
                block_indices = torch.cat(
                    [torch.tensor([0], dtype=torch.int32), torch.cumsum(query_lens_d, dim=0)[:-1]]
                )
                common_attn_metadata.block_table_tensor[:batch_size] = common_attn_metadata.block_table_tensor[
                    block_indices
                ]
                common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor[:batch_size]

                # Copy the old attn_metadata and update
                for draft_step in range(1, self.num_speculative_tokens):
                    common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
                        draft_step,
                        attn_metadata,
                        common_attn_metadata,
                        batch_size,
                        num_input_tokens,
                        used_update_positions,
                        aclgraph_runtime_mode,
                        ori_seq_len,
                        slot_indices,
                        mtp_slot_mapping,
                    )
                    per_layer_attn_metadata = dict()
                    for layer_name in self.attn_layer_names:
                        per_layer_attn_metadata[layer_name] = attn_metadata
                    multi_steps_attn_metadata.append(per_layer_attn_metadata)
        else:
            # Copy the old attn_metadata and update
            for draft_step in range(1, self.num_speculative_tokens):
                common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
                    draft_step,
                    attn_metadata,
                    common_attn_metadata,
                    batch_size,
                    num_input_tokens,
                    used_update_positions,
                    aclgraph_runtime_mode,
                )
                per_layer_attn_metadata = dict()
                for layer_name in self.attn_layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata
                multi_steps_attn_metadata.append(per_layer_attn_metadata)

        last_token_indices_len = last_token_indices.shape[0]
        self.last_token_indices[:last_token_indices_len].copy_(last_token_indices)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0],
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            # Reset MOE layer index for forward pass
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            draft_token_ids = self._runnable(
                num_input_tokens=num_input_tokens,
                batch_size=batch_size,
                last_token_indices=self.last_token_indices[:last_token_indices_len],
                target_positions=target_positions,
                inputs_embeds=inputs_embeds,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
                is_prefill=attn_metadata_i.num_prefills,
            )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)
        return draft_token_ids

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        last_token_indices,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_dummy=False,
        is_prefill=None,
    ) -> torch.Tensor:
        # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all
        # speculative tokens' proposings. `model_input_ids`, `model_positions` and
        # `model_hidden_states` represent the speculative model inputs.
        model_input_ids = self.input_ids[:num_input_tokens]
        model_positions = self._get_positions(num_input_tokens)
        model_hidden_states = self.hidden_states[:num_input_tokens]

        model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)

        ret_hidden_states = self.model(
            input_ids=model_input_ids,
            positions=model_positions,
            hidden_states=model_hidden_states,
            inputs_embeds=inputs_embeds,
        )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
            last_hidden_states, model_positions, hidden_states
        )

        if self.pcp_size > 1:
            # remove graph padding before all_gather
            hidden_states = hidden_states[:num_tokens]
            hidden_states = get_pcp_group().all_gather(hidden_states, 0)
            hidden_states = torch.index_select(
                hidden_states, 0, self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
            )
            last_hidden_states = hidden_states  # TODO: check it

        num_indices = last_token_indices.shape[0]
        if lmhead_tp_enable() and not is_dummy:
            max_num_reqs_across_dp = (
                self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
            )
            last_token_indices = nn.functional.pad(last_token_indices, (0, max_num_reqs_across_dp - num_indices))

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        if lmhead_tp_enable() and num_indices < logits.shape[0] and not is_dummy:
            logits = logits[:num_indices]
            last_token_indices = last_token_indices[:num_indices]

        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        if self.pcp_size * self.dcp_size > 1 and is_prefill:
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list = []
            for _ in range(self.num_speculative_tokens):
                draft_token_ids_list.append(draft_token_ids)
            return torch.stack(draft_token_ids_list, dim=1)

        # Generate the remaining draft tokens.
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_token_ids.shape), dtype=draft_token_ids.dtype, device=self.device
        )
        draft_token_ids_tensor[0] = draft_token_ids
        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]
        last_token_indices = self.arange[:batch_size]

        input_batch_size = num_input_tokens if (self.method == "mtp" or self.use_cuda_graph) else batch_size

        forward_context = get_forward_context()
        forward_context.num_tokens = input_batch_size
        forward_context.num_accept_tokens = batch_size

        for draft_step in range(self.num_speculative_tokens - 1):
            # Reset MOE layer index for each draft step iteration
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_tensor[draft_step]
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            if self.uses_mrope:
                exceeds_max_model_len = positions[0] >= self.vllm_config.model_config.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0), torch.zeros_like(positions), positions
                )
            else:
                exceeds_max_model_len = positions >= self.vllm_config.model_config.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.

            # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all
            # speculative tokens' proposings. `model_input_ids`, `model_positions` and
            # `model_hidden_states` represent the speculative model inputs.
            model_input_ids = self.input_ids[:input_batch_size]
            model_positions = self._get_positions(input_batch_size)
            model_hidden_states = self.hidden_states[:input_batch_size]

            model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)

            forward_context.attn_metadata = (
                multi_steps_attn_metadata[draft_step + 1] if multi_steps_attn_metadata else None
            )
            ret_hidden_states = self.model(
                input_ids=model_input_ids,
                positions=model_positions,
                hidden_states=model_hidden_states,
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states
            )

            num_indices = last_token_indices.shape[0]
            if lmhead_tp_enable() and not is_dummy:
                max_num_reqs_across_dp = (
                    self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
                )
                last_token_indices = nn.functional.pad(
                    last_token_indices,
                    (0, max_num_reqs_across_dp - num_indices),
                )

            sample_hidden_states = last_hidden_states[last_token_indices]
            logits = self.model.compute_logits(sample_hidden_states)

            if lmhead_tp_enable() and num_indices < logits.shape[0] and not is_dummy:
                logits = logits[:num_indices]
                last_token_indices = last_token_indices[:num_indices]

            # TODO(wenlong): get more than one token for tree attention
            hidden_states = hidden_states[:batch_size]
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_tensor[draft_step + 1] = draft_token_ids

        # [batch_size, num_speculative_tokens]
        draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
        return draft_token_ids

    def attn_update_stack_num_spec_norm(
        self,
        # `draft_step` must start from `1`, no `0`
        draft_step,
        old_attn_metadata,
        old_common_metadata,
        batch_size,
        input_batch_size,
        used_update_positions,
        aclgraph_runtime_mode,
        ori_seq_len=None,
        slot_indices=None,
        mtp_slot_mapping=None,
    ):
        assert draft_step > 0
        common_attn_metadata = self.shallow_copy_metadata(old_common_metadata)

        if draft_step == 1:
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and (pad_size := input_batch_size - batch_size) > 0:
                common_attn_metadata.num_reqs = input_batch_size
                common_attn_metadata.block_table_tensor = self._pad_tensor(
                    common_attn_metadata.block_table_tensor, pad_size
                )
                common_attn_metadata.seq_lens = self._pad_tensor(common_attn_metadata.seq_lens, pad_size)
                common_attn_metadata.seq_lens_cpu = self._pad_tensor(common_attn_metadata.seq_lens_cpu, pad_size)
                common_attn_metadata.num_computed_tokens_cpu = self._pad_tensor(
                    common_attn_metadata.num_computed_tokens_cpu, pad_size
                )
                common_attn_metadata.query_start_loc = self.arange[: input_batch_size + 1]
                common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                    self.token_arange_np[: input_batch_size + 1]
                ).clone()
            else:
                common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
                common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                    self.token_arange_np[: batch_size + 1]
                ).clone()

            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.decode_token_per_req = 1
            common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
            common_attn_metadata.graph_pad_size = -1
            common_attn_metadata.num_input_tokens = input_batch_size

        # The loop part

        used_update_positions += 1

        # NOTE(woosuk): We should handle the case where the draft model
        # generates tokens beyond the max model length. Since it is complex
        # to remove such requests from the batch, we keep them in the batch
        # but adjust the position ids and slot mappings to avoid the
        # out-of-range access during the model execution. The draft tokens
        # generated with this adjustment should be ignored.
        if self.uses_mrope:
            exceeds_max_model_len = used_update_positions[0] >= self.vllm_config.model_config.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(
                exceeds_max_model_len.unsqueeze(0), torch.zeros_like(used_update_positions), used_update_positions
            )
        else:
            exceeds_max_model_len = used_update_positions >= self.vllm_config.model_config.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, used_update_positions)

        # For data integrity when async scheduling, we shouldn't use in place
        # operations in case they are modified in next step's `prepare_input`
        # of main model.
        # Increment the sequence lengths.
        common_attn_metadata.seq_lens[:batch_size] += 1
        # For the requests that exceed the max model length, we set the
        # sequence length to 1 to minimize their overheads in attention.
        common_attn_metadata.seq_lens[:batch_size].masked_fill_(exceeds_max_model_len, 1)

        common_attn_metadata.seq_lens_cpu[:batch_size] = common_attn_metadata.seq_lens_cpu[:batch_size] + 1
        exceeds_mask = common_attn_metadata.seq_lens_cpu[:batch_size] >= self.max_model_len
        common_attn_metadata.seq_lens_cpu[:batch_size].masked_fill_(exceeds_mask, 1)
        common_attn_metadata.num_computed_tokens_cpu[:batch_size] += 1
        if self.uses_mrope:
            common_attn_metadata.positions[:batch_size].copy_(clamped_positions[0])
        else:
            common_attn_metadata.positions[:batch_size].copy_(clamped_positions)

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        if self.pcp_size * self.dcp_size > 1:
            num_computed_tokens_of_pcp_dcp = self.runner.pcp_manager._get_cp_local_seq_lens(
                ori_seq_len + draft_step,
                self.pcp_size,
                self.dcp_size,
                self.runner.parallel_config.cp_kv_cache_interleave_size,
            )
            cp_seq_len = num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank]
            # update slot_mapping
            slot_indices += self.pcp_size
            slot_mapping = mtp_slot_mapping[slot_indices]
            common_attn_metadata.slot_mapping[: batch_size * self.pcp_size] = slot_mapping
        else:
            block_size = attn_metadata_builder.kv_cache_spec.block_size

            # Compute the slot mapping.
            if self.uses_mrope:
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = old_common_metadata.block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                slot_mapping = block_ids * block_size + clamped_positions[0] % block_size
            else:
                slot_mapping = block_ids * block_size + clamped_positions % block_size

            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
            self.slot_mapping_group[draft_step][: slot_mapping.shape[0]].copy_(slot_mapping.to(torch.int32))
            self.slot_mapping_group[draft_step][slot_mapping.shape[0] :].fill_(PADDING_SLOT_ID)
            # Set the address of the attn_metadata.slot_mapping to the self.slot_mapping_group[idx]
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step][: slot_mapping.shape[0]]

        # Rebuild attention metadata
        attn_metadata = attn_metadata_builder.build_for_drafting(  # type: ignore
            common_attn_metadata=common_attn_metadata,
            draft_index=draft_step,
        )

        if self.pcp_size * self.dcp_size > 1:
            if self.vllm_config.model_config.use_mla:
                attn_metadata.decode.cp_seq_len = cp_seq_len
            else:
                attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp

        return common_attn_metadata, attn_metadata

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
                requests[gpu_input_batch.req_ids[i]].get_token_id(common_attn_metadata.seq_lens_cpu[i].item())
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[:num_discarded_requests]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(0, discard_sampled_tokens_req_indices, -1)

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)).squeeze(1)

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
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_actual_reqs + 1]
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
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1], new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        common_attn_metadata.slot_mapping[: token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices]
        )
        common_attn_metadata.slot_mapping[token_indices.shape[0] :].fill_(-1)

        # NOTE: Currently positions and seq_lens are not used in attn forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
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
            max_seq_len=0,
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
        if HAS_TRITON:
            num_reqs = common_attn_metadata.num_reqs
            device = valid_sampled_tokens_count.device

            token_indices_to_sample = torch.empty((num_reqs,), dtype=torch.int32, device=device)

            num_blocks_needed = triton.cdiv(num_reqs, _PREPARE_INPUTS_BLOCK_SIZE)
            num_vector_core = get_vectorcore_num()
            grid_size = min(num_blocks_needed, num_vector_core)
            grid = (grid_size,)

            prepare_inputs_padded_kernel[grid](
                spec_decode_metadata.cu_num_draft_tokens,
                valid_sampled_tokens_count,
                common_attn_metadata.query_start_loc,
                token_indices_to_sample,
                num_reqs,
                BLOCK_SIZE=_PREPARE_INPUTS_BLOCK_SIZE,
            )
        else:
            num_draft_tokens_gpu = torch.cat(
                [
                    spec_decode_metadata.cu_num_draft_tokens[0:1],
                    spec_decode_metadata.cu_num_draft_tokens[1:] - spec_decode_metadata.cu_num_draft_tokens[:-1],
                ]
            )

            num_rejected_tokens_gpu = torch.where(
                num_draft_tokens_gpu > 0,
                num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
                torch.zeros_like(num_draft_tokens_gpu),
            )

            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1 - num_rejected_tokens_gpu

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

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
            num_actual_tokens=common_attn_metadata.num_actual_tokens if self.pcp_size > 1 else total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=0,
        )

        return spec_common_attn_metadata, token_indices, token_indices_to_sample

    def _split_pcp_input(self, req_scheduled_tokens, input_ids, target_hidden_states):
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
                torch.zeros([0], device="npu"),
                torch.zeros([0, target_hidden_states.size(1)], device="npu"),
                0,
                torch.zeros([0]),
                torch.tensor([0], dtype=torch.int32),
            )

        def _pcp_pad_and_split(num_tokens):
            num_pcp_padded_scheduled_tokens = cdiv(num_tokens, 2 * self.pcp_size) * 2 * self.pcp_size
            pcp_pad = num_pcp_padded_scheduled_tokens - num_tokens
            chunk_size = num_pcp_padded_scheduled_tokens // (2 * self.pcp_size)

            # split position_ids (and use split position_ids to split input_ids afterwards)
            req_position_cp: list[int] = []
            req_position_cp.extend(self.full_indices[self.pcp_rank * chunk_size : (self.pcp_rank + 1) * chunk_size])
            req_position_cp.extend(
                self.full_indices[
                    num_pcp_padded_scheduled_tokens - (self.pcp_rank + 1) * chunk_size : num_pcp_padded_scheduled_tokens
                    - self.pcp_rank * chunk_size
                ]
            )

            return req_position_cp, num_pcp_padded_scheduled_tokens, pcp_pad

        num_pcp_scheduled_tokens = []
        ori_start_index = 0
        pad_start_index = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            req_position_pcp, num_pcp_padded_scheduled_tokens, num_pcp_pad = _pcp_pad_and_split(ori_num_tokens)
            actual_num_tokens = len(req_position_pcp)
            num_pcp_scheduled_tokens.append(actual_num_tokens)
            pad_input_ids = F.pad(input_ids[ori_start_index : ori_start_index + ori_num_tokens], (0, num_pcp_pad))
            ori_start_index += ori_num_tokens
            pcp_chunk_indices = [pad_start_index + pos for pos in req_position_pcp]
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
        cu_num_tokens = torch.tensor(np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens

    # update full-graph params for one spec token
    def _update_full_graph_params(self, forward_context, num_tokens, draft_attn_metadatas=None):
        update_full_graph_params(
            self.runner.attn_backend,
            self.update_stream,
            forward_context,
            num_tokens,
            self.vllm_config,
            self.vllm_config.speculative_config,
            draft_attn_metadatas=draft_attn_metadatas,
        )

    # padding tensor into desired size
    def _pad_tensor(self, tensor, pad_size):
        pad = [0] * (2 * tensor.dim() - 1) + [pad_size]
        padded_tensor = F.pad(tensor, pad, mode="constant", value=0)
        return padded_tensor

    def maybe_pad_and_reduce(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "mtp":
            if self.enable_shared_expert_dp:
                hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
                positions = positions.unsqueeze(-1)
                positions = torch.ops.vllm.maybe_pad_and_reduce(positions)
                positions = positions.squeeze(-1)
        else:
            forward_context = get_forward_context()
            if forward_context.flash_comm_v1_enabled:
                hidden_states = split_inputs_tp_to_sp(hidden_states, hidden_states)
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.method == "mtp":
            if self.enable_shared_expert_dp:
                last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    last_hidden_states.contiguous(), True
                )
                positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(positions.contiguous(), True)
                if hidden_states is not None:
                    hidden_states = last_hidden_states
        else:
            forward_context = get_forward_context()
            if forward_context.flash_comm_v1_enabled:
                last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    last_hidden_states.contiguous(), True
                )
                if hidden_states is not None:
                    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states.contiguous(), True)
        return last_hidden_states, positions, hidden_states
