#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import time
import types
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
import torch._dynamo.cache_size
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.utils import bind_kv_cache

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import CommonAttentionMetadata
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.torchair.utils import (check_torchair_cache_exist,
                                        write_kv_cache_bytes_to_file)
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               ProfileExecuteDuration,
                               converting_weight_acl_format, is_310p)
from vllm_ascend.worker.eagle_proposer_v1 import EagleProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import torch_npu
import vllm.envs as envs_vllm

import vllm_ascend.envs as envs_ascend

if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)


class NPUTorchairModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        ascend_config = get_ascend_config()
        self.new_kv_cache_bytes = -1
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
        self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes
        if ascend_config.torchair_graph_config.graph_batch_sizes_init:
            self._init_torchair_graph_batch_sizes()
        if len(self.torchair_graph_batch_sizes) == 0:
            # TODO(zzzzwwjj): check torchair_graph_batch_sizes init code
            self.torchair_graph_batch_sizes = [self.max_num_reqs]

        torch._dynamo.cache_size.config.cache_size_limit += len(
            self.torchair_graph_batch_sizes)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._logging.set_logs(
            recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

    def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        if (self.use_aclgraph and
                total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            num_valid_tokens[i] = num_tokens - \
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                             non_blocking=True)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        self.positions[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        seq_lens = self.seq_lens_cpu[:num_reqs]

        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)

        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        ascend_config = get_ascend_config()
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.use_eagle:
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.SpecDecoding
        # splitfuse
        elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        self.attn_mask = self._make_attention_mask(
            seq_lens=seq_lens,
            query_lens=num_scheduled_tokens,
            position=positions,
            attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        extra_builder_kwargs = {}

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        if self.dp_size > 1:
            max_num_tokens, with_prefill = self._get_forward_metadata_across_dp(
                total_num_scheduled_tokens, with_prefill)
            extra_builder_kwargs['max_num_tokens_across_dp'] = max_num_tokens
            extra_builder_kwargs['with_prefill_across_dp'] = with_prefill

        # Add graph_pad_size here
        if not with_prefill:
            if self.dp_size > 1:
                padded_batch_size = self._select_torchair_padded_batch_size(
                    max_num_tokens)
            else:
                padded_batch_size = self._select_torchair_padded_batch_size(
                    total_num_scheduled_tokens)
            graph_pad_size = padded_batch_size - total_num_scheduled_tokens

            extra_builder_kwargs['graph_pad_size'] = graph_pad_size

        if self.vllm_config.model_config.use_mla:
            query_start_loc = self.query_start_loc[:num_reqs + 1]
            seq_lens = self.seq_lens[:num_reqs]
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc, seq_lens=seq_lens)
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_attn_metadata=common_attn_metadata,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
        else:
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_prefix_len=None,
                **extra_builder_kwargs,
            )
        attn_metadata.num_input_tokens = num_input_tokens

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:total_num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]

        if not with_prefill:
            input_ids = self.input_ids[:padded_batch_size]
            positions = self.positions[:padded_batch_size]

        # Run forward pass
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            with ProfileExecuteDuration().capture_async("forward"):
                model_kwargs = {}
                model_kwargs["kv_caches"] = self.kv_caches
                model_kwargs["attn_metadata"] = attn_metadata
                if not with_prefill:
                    converting_weight_acl_format(self.model,
                                                 ACL_FORMAT_FRACTAL_NZ)
                    compiled_model = self._get_torchair_lazy_compiled_model(
                        padded_batch_size)
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            sample_indices = spec_decode_metadata.logits_indices

        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states

        return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                total_num_scheduled_tokens, sample_indices, aux_hidden_states,
                num_scheduled_tokens)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        is_compile: bool = False,
        with_prefill: bool = True,
    ) -> torch.Tensor:
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        num_reqs = self.max_num_reqs if num_tokens >= self.max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=num_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k: v[:num_tokens]
                    for k, v in self.intermediate_tensors.items()
                })

            with set_forward_context(None,
                                     self.vllm_config,
                                     num_tokens=num_tokens):
                if not with_prefill:
                    attn_metadata = self.attn_metadata_builder.build_dummy(
                        num_reqs=num_tokens, num_actual_tokens=1)
                    # Only mark static while compiling
                    if is_compile:
                        torch._dynamo.mark_static(input_ids)
                        torch._dynamo.mark_static(positions)
                        torch._dynamo.mark_static(
                            attn_metadata.decode.block_table)
                        torch._dynamo.mark_static(
                            attn_metadata.decode.input_positions)
                        torch._dynamo.mark_static(attn_metadata.slot_mapping)
                        for kv in self.kv_caches:
                            assert isinstance(
                                kv, tuple), "kv_cache must be a tuple"
                            torch._dynamo.mark_static(kv[0])
                            torch._dynamo.mark_static(kv[1])

                    converting_weight_acl_format(self.model,
                                                 ACL_FORMAT_FRACTAL_NZ)

                    compiled_model = self._get_torchair_lazy_compiled_model(
                        num_tokens)
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=None,
                        kv_caches=self.kv_caches,
                        attn_metadata=attn_metadata,
                    )
                else:
                    converting_weight_acl_format(self.model,
                                                 ACL_FORMAT_FRACTAL_ND)

                    hidden_states = model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds)
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, _ = hidden_states
                    else:
                        hidden_states = hidden_states
                    if self.use_spec_decode and isinstance(
                            self.drafter, EagleProposer):
                        self.drafter.dummy_run(num_tokens)
            return hidden_states

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        import torch_npu
        acl_format = ACL_FORMAT_FRACTAL_NZ if is_310p(
        ) else ACL_FORMAT_FRACTAL_ND
        kv_caches: Dict[str, torch.Tensor] = {}

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=True,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
        )

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in "
                "NPU.")
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes

                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    if self.vllm_config.additional_config.get(
                            "kv_cache_dtype", None) == 'int8':
                        kv_cache_shape = self.attn_backend.get_bsh_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    if len(kv_cache_shape) == 3:
                        # for non MLA attention backend that use torchair, we consider to pass kv_cache layout
                        # of BSH ([num_blocks, block_size, kv_head_dim * head_size]) to attention.

                        kv_caches[layer_name] = (torch.zeros(
                            kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device),
                                                 torch.zeros(
                                                     kv_cache_shape,
                                                     dtype=self.kv_cache_dtype,
                                                     device=self.device))
                        # atb reshape_and_cache does not support torchair.
                        kv_caches[layer_name] = (
                            torch_npu.npu_format_cast(kv_caches[layer_name][0],
                                                      ACL_FORMAT_FRACTAL_ND),
                            torch_npu.npu_format_cast(kv_caches[layer_name][1],
                                                      ACL_FORMAT_FRACTAL_ND),
                        )
                    else:
                        # for MLA attention backend that use torchair.
                        layer_kv_cache_nope = torch.zeros(
                            kv_cache_shape[:-1] +
                            (self.model_config.hf_text_config.kv_lora_rank, ),
                            dtype=self.dtype,
                            pin_memory=True,
                            device=self.device)
                        layer_kv_cache_pe = torch.zeros(
                            kv_cache_shape[:-1] +
                            (self.model_config.hf_text_config.qk_rope_head_dim,
                             ),
                            dtype=self.dtype,
                            pin_memory=True,
                            device=self.device)
                        kv_caches[layer_name] = (layer_kv_cache_nope,
                                                 layer_kv_cache_pe)
                        kv_caches[layer_name] = (
                            torch_npu.npu_format_cast(kv_caches[layer_name][0],
                                                      acl_format),
                            torch_npu.npu_format_cast(kv_caches[layer_name][1],
                                                      acl_format),
                        )
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.

        torchair_graph_batch_sizes = self.torchair_graph_batch_sizes
        graph_num = len(torchair_graph_batch_sizes)

        if self.use_cached_npu_graph and not check_torchair_cache_exist():
            # If caching is enabled but does not exist, we will compile the model twice. The first
            # time is used to generate the cache, and the second time is used to load the cache to
            # skip the overhead caused by Dynamo guard mechanism.
            logger.info(
                "Use cached npu graph but cache doesn't exist! Now we compile graph to genetate torchair cache, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)
            NPUPlatform.synchronize()
            torch._dynamo.reset()
            self.torchair_compiled_models.clear()
        if self.use_cached_npu_graph:
            logger.info(
                "Loading torchair graph cache, this usually takes %.1f~%.1f mins.",
                0.3 * graph_num, 0.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)
        else:
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            self._compile_torchair_graph(torchair_graph_batch_sizes)

        if self.new_kv_cache_bytes > 0:
            write_kv_cache_bytes_to_file(torch.distributed.get_rank(),
                                         self.new_kv_cache_bytes)
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _init_torchair_graph_batch_sizes(self):
        start_graph_batch_size = 4
        tp_size = get_tensor_model_parallel_world_size()

        # NOTE: When use all2all | mc2, We need to slice the `num_tokens` dimension into `tp_size` blocks
        start_graph_batch_size = max(start_graph_batch_size, tp_size)

        while (start_graph_batch_size <= self.max_num_reqs):
            self.torchair_graph_batch_sizes.append(start_graph_batch_size)
            start_graph_batch_size *= 2

    def _select_torchair_padded_batch_size(self, batch_size: int):
        selected_batch_size = self.max_num_reqs
        for padded_batch_size in self.torchair_graph_batch_sizes:
            if batch_size <= padded_batch_size < selected_batch_size:
                selected_batch_size = padded_batch_size
        return selected_batch_size

    def _compile_torchair_graph(self, torchair_graph_batch_sizes) -> None:
        # Trigger torchair graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        for idx, num_tokens in enumerate(reversed(torchair_graph_batch_sizes)):
            for _ in range(self.vllm_config.compilation_config.
                           cudagraph_num_of_warmups):
                self._dummy_run(num_tokens,
                                is_compile=True,
                                with_prefill=False)
            self._dummy_run(num_tokens, is_compile=True, with_prefill=False)
            logger.info("Batchsize %d is compiled successfully: %d/%d.",
                        num_tokens, idx + 1, len(torchair_graph_batch_sizes))

    def _get_torchair_lazy_compiled_model(self, batch_size: int):
        if batch_size < 0 or batch_size > self.max_num_reqs:
            raise ValueError(
                f"Bad graph batch size:{batch_size}! max_num_reqs:{self.max_num_reqs}"
            )

        compiled_model = self.torchair_compiled_models.get(
            batch_size
        ) if self.use_cached_npu_graph else self.torchair_compiled_model

        if compiled_model:
            return compiled_model

        import torchair  # type: ignore
        from torchair import patch_for_hcom  # type: ignore

        patch_for_hcom()

        if is_310p():
            # on 300I Duo platform, we need to patch broadcast. however, this patch will be
            # overwritten by patch_for_hcom in torchair. so we need to re-patch it here.
            from vllm_ascend.patch.platform.patch_common.patch_distributed import \
                communication_adaptation_310p
            communication_adaptation_310p()

        config = torchair.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        # enabling tiling_schedule_optimize on 300I Duo has some bugs, so we have to
        # disable it on 300I Duo platform now.
        config.experimental_config.tiling_schedule_optimize = not is_310p()
        config.experimental_config.enable_view_optimize = \
        get_ascend_config().torchair_graph_config.enable_view_optimize
        torch.npu.set_compile_mode(jit_compile=False)
        if not self.use_cached_npu_graph:
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self.torchair_compiled_model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=npu_backend)
            return self.torchair_compiled_model
        else:
            # Generate a new forward proxy code object to prevent the invalidation of
            # compilation cache caused by dynamo retracing
            forward_proxy_name = f"{self.model.__class__.__name__}_forward_with_batch_size_{batch_size}"
            forward_fn = self.model.forward
            code = forward_fn.__code__
            # Mark code object with a new proxy name
            modified_code = code.replace(co_name=forward_proxy_name, )

            modified_func = types.FunctionType(modified_code,
                                               forward_fn.__globals__,
                                               name=forward_proxy_name,
                                               argdefs=forward_fn.__defaults__)

            self.model.__dict__[forward_proxy_name] = modified_func.__get__(
                self.model, nn.Module)
            self.torchair_compiled_models[
                batch_size] = torchair.inference.cache_compile(
                    self.model.__dict__[forward_proxy_name],
                    dynamic=True,
                    fullgraph=envs_vllm.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)
            return self.torchair_compiled_models[batch_size]
