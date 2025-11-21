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

import copy
import gc
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from omni.adaptors.vllm.ems.ems_env import EmsEnv
from omni.adaptors.vllm.forward_context import set_forward_context
from omni.adaptors.vllm.platform import NPUPlatform
from omni.adaptors.vllm.spec_decode.post_drafter import PostDrafter
from omni.adaptors.vllm.utils import get_attr_by_names
from omni.adaptors.vllm.worker.cache_engine import CacheEngine
from omni.models.common.config.model_config import (model_extra_config,
                                                    update_model_extra_config)
from omni.models.common.layers.attention.backend.attention import (
    AscendAttentionState, AttentionMaskBuilder)
from omni.models.common.layers.attention.backend.attention_dummy_builder import \
    DummyAttentionMetadataBuilder
from omni.models.common.layers.sampler import AscendSamplerV1, SimpleSampler
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (
    get_dp_group, get_pp_group, get_tensor_model_parallel_world_size)
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.sequence import VLLM_INVALID_TOKEN_ID, IntermediateTensors
from vllm.utils import (DeviceMemoryProfiler, LayerBlockType, LazyLoader, cdiv,
                        is_pin_memory_available)
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

MTP_METHOD_NAME_LIST = ["deepseek_mtp", "pangu_ultra_moe_mtp"]

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

if model_extra_config.operator_opt_config.use_omni_placement:
    from omni.accelerators.placement.omni_placement.omni_planner import \
        OmniPlanner
    _GLOBAL_STEP = 0

MAX_GEAR_NUM = 6
NPU_GENERATOR_OFFSET_STEP = 12  # ascend npu, move 12 every one generation, which is 4 on cuda.


def _get_pad_size(num_seqs):
    tp_size = get_tensor_model_parallel_world_size()
    if model_extra_config.parall_config.attn_sp_size > 1:
        tp_size = tp_size * 2
    return (tp_size - num_seqs % tp_size) % tp_size


class GraphCompileConfiguration:
    """
    When the graph mode is turned on
    you can set the gear or clarify the static shape by inheriting this class to speed up the model running
    """

    def set_dynamic_gears(self, *args, **kwargs):
        pass

    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(args[0])
        torch._dynamo.mark_static(args[1])


def mark_static_for_graph_default(
        input_ids,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        hidden_states: Optional[torch.Tensor] = None):
    torch._dynamo.mark_static(input_ids)
    if positions is not None:
        torch._dynamo.mark_static(positions)

    if kv_caches is not None:
        for kv_cache in kv_caches:
            if kv_cache is not None:
                torch._dynamo.mark_static(kv_cache[0])  # k_cache
                torch._dynamo.mark_static(kv_cache[1])  # v_cache

    if hidden_states is not None:
        torch._dynamo.mark_static(hidden_states)


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.cache_engine = None
        self.head_size = self.model_config.get_head_size()
        self.block_size = vllm_config.cache_config.block_size

        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            vllm_config.parallel_config, LayerBlockType.attention)
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        if self.use_spec_decode:
            self.rejection_sampler = SimpleSampler(self.sampler)
            self.drafter = PostDrafter(vllm_config, device, self)
        else:
            self.sampler = AscendSamplerV1()
        self._init_graph_options()

        self.slot_mapping_cpu = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device="cpu",
            pin_memory=is_pin_memory_available())
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=is_pin_memory_available())
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int64,
                                    device=self.device)
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int64,
                                        device="cpu",
                                        pin_memory=is_pin_memory_available())
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        num_tokens_per_reqs_decode = 1 if not self.use_spec_decode else (
            1 + self.speculative_config.num_speculative_tokens)
        self.chunk_next_tokens = torch.zeros(self.max_num_reqs *
                                             num_tokens_per_reqs_decode,
                                             dtype=torch.int64,
                                             device=self.device)
        # TODO: support arbitrary spec tokens
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs * num_tokens_per_reqs_decode,
             (self.model_config.max_model_len + self.block_size - 1) //
             self.block_size),
            dtype=np.int32)
        self.attn_mask = None
        self.attn_state = None
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)

        mask_len = os.getenv("PAGED_ATTENTION_MASK_LEN", "2048")
        self.attn_mask_len = min(self.model_config.max_model_len,
                                 int(mask_len))
        self.attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
            self.attn_mask_len, self.dtype)

        self.model_mark_static = False
        self.dummy_model_mark_static = False
        self.drafter_mark_static = False
        self.dummy_drafter_mark_static = False

        self.total_step = 1
        self.curr_step = 0
        self.arange_npu = torch.arange(max(self.max_num_reqs + 1,
                                           self.max_model_len,
                                           self.max_num_tokens),
                                       dtype=torch.int64,
                                       device=self.device)
        #init ems adapters
        if EmsEnv.enable_vllm_ems:
            from omni.adaptors.vllm.ems.ems_adapter import EmsAdapter
            self.ems_adapter = EmsAdapter(vllm_config=vllm_config)

        self.omni_cache = None

    def _init_graph_options(self):
        from vllm.utils import supports_dynamo

        self.enable_torchair_graph_mode = (
            self.vllm_config.npu_compilation_config.level
            > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.use_cached_npu_graph = self.vllm_config.npu_compilation_config.use_ge_graph_cached
        self.decode_gear_list = self.vllm_config.npu_compilation_config.decode_gear_list
        self.max_batch_size = self.max_num_reqs if not self.use_spec_decode else self.max_num_reqs * (
            1 + self.speculative_config.num_speculative_tokens)
        if self.decode_gear_list is None:
            self.decode_gear_list = []
            self.decode_gear_list.append(self.max_num_reqs if not self.use_spec_decode else self.max_num_reqs * \
                                            (1 + self.speculative_config.num_speculative_tokens))
        update_model_extra_config(
            decode_gear_list=self.decode_gear_list,
            enable_torchair_graph_mode=self.enable_torchair_graph_mode)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, Any], int, torch.Tensor, torch.Tensor, bool]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if total_num_scheduled_tokens <= 0:
            raise RuntimeError(
                "total_num_scheduled_tokens must be greater than 0")
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            raise RuntimeError("num_reqs must be greater than 0")
        num_input_tokens = total_num_scheduled_tokens
        logger.warning(
            f"current num reqs = {num_reqs}, num_input_tokens = {num_input_tokens}"
        )

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens = np.array([
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in self.input_batch.req_ids
        ],
                                        dtype=np.int32)
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_scheduled_spec_decode_reqs = len(
            scheduler_output.scheduled_spec_decode_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)
        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]

        self.seq_lens_np[:
                         num_reqs] = self.input_batch.num_computed_tokens_cpu[:
                                                                              num_reqs] + num_scheduled_tokens

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            # NOTE(runze): since each request has at most M blocks, the offset is at most M-1
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                np.minimum(positions_np // block_size,
                           block_table.max_num_blocks_per_req - 1))
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # check and set attention state
        can_decode = self.vllm_config.kv_transfer_config is None or self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        if can_decode and (np.all(num_scheduled_tokens == 1)
                           or num_scheduled_spec_decode_reqs == num_reqs):
            attn_state = AscendAttentionState.DecodeOnly
        elif np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # splitfuse
        else:
            attn_state = AscendAttentionState.ChunkedPrefill

        self.attn_state = attn_state

        # calculate max_batch_size and padding size
        graph_pad_size = 0
        if self.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly and len(
                self.decode_gear_list) > 1:
            self.max_batch_size = self._get_max_token_num(
                self.vllm_config.parallel_config.data_parallel_size > 1,
                num_reqs)
        if attn_state == AscendAttentionState.DecodeOnly:
            if num_reqs > self.max_batch_size:
                raise RuntimeError("num_reqs is bigger than max_batch_size")
            if self.use_spec_decode:
                graph_pad_size = self.max_batch_size - num_reqs * (
                    1 + self.speculative_config.num_speculative_tokens)
            else:
                graph_pad_size = self.max_batch_size - num_reqs
        else:
            # The reduce_scatter in the TP communication domain after embedding, P goes through this
            graph_pad_size = _get_pad_size(num_input_tokens)

        # padding positions
        if graph_pad_size >= 0:
            padding_positions = torch.zeros(graph_pad_size,
                                            dtype=positions.dtype,
                                            device=positions.device)
            positions = torch.cat([positions, padding_positions])

        extra_builder_kwargs = {'graph_pad_size': graph_pad_size}

        # build attention metadata
        attn_metadata = {}
        self.full_attn_metadata = None
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id],
                    kv_cache_group_spec.kv_cache_spec,
                    self.attn_metadata_builders[kv_cache_group_id],
                )
            attn_metadata_i = self.attn_metadata_builders[
                kv_cache_group_id].build(
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    common_prefix_len=None,
                    **extra_builder_kwargs,
                )
            if kv_cache_group_id == 0:
                self.full_attn_metadata = attn_metadata_i

            if not isinstance(self.attn_metadata_builders[kv_cache_group_id],
                              DummyAttentionMetadataBuilder):
                raise ValueError(
                    f"{self.attn_metadata_builders[kv_cache_group_id]} does not implement DummyAttentionMetadataBuilder"
                )
            if self.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly:
                self.attn_metadata_builders[
                    kv_cache_group_id].mark_static_for_attn_metadata(
                        attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

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

        # spec decode tokens
        has_spec_tokens = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0

        if has_spec_tokens:
            # 当前仅在DecodeOnly时才可能到此逻辑
            # TODO 复用GPU ModelRunner中的_calc_spec_decode_metadata及SpecDecodeMetadata
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            sample_indices = torch.arange(total_num_scheduled_tokens,
                                          dtype=torch.int32,
                                          device=self.device)
        else:
            if model_extra_config.parall_config.attn_sp_size > 1:
                sp_size = model_extra_config.parall_config.attn_sp_size * 2
                cu_num_tokens = np.empty_like(num_scheduled_tokens)
                cu_num_tokens[0] = num_scheduled_tokens[0]
                for i in range(1, num_scheduled_tokens.size):
                    prev_aligned = ((cu_num_tokens[i - 1] + sp_size - 1) //
                                    sp_size) * sp_size
                    cu_num_tokens[i] = prev_aligned + num_scheduled_tokens[i]
            sample_indices = cu_num_tokens - 1
            sample_indices = torch.from_numpy(sample_indices).to(
                self.device, non_blocking=True)
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)
        return attn_metadata, graph_pad_size, sample_indices, positions

    def _simple_prepare_inputs(self,
                               attn_metadata,
                               positions,
                               cached_token,
                               cached_spec,
                               accepted_num=0) -> torch.Tensor:
        token_each_reqs = 1
        if cached_spec is not None:
            token_each_reqs = 1 + len(cached_spec[0])
        num_reqs = self.input_batch.num_reqs
        total_num_scheduled_tokens = token_each_reqs * num_reqs

        if isinstance(accepted_num, torch.Tensor):
            positions[:total_num_scheduled_tokens] += torch.repeat_interleave(
                accepted_num, token_each_reqs) + 1
        else:
            positions[:total_num_scheduled_tokens] += 1

        req_indices = torch.repeat_interleave(self.arange_npu[:num_reqs],
                                              token_each_reqs,
                                              dim=0)
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions[:total_num_scheduled_tokens] // block_size)
            block_table_cpu = block_table.get_device_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices]
            block_offsets = positions[:total_num_scheduled_tokens] % block_size
            block_table.slot_mapping[:
                                     total_num_scheduled_tokens] = block_numbers * block_size + block_offsets

        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            first_layer_in_group = kv_cache_group_spec.layer_names[0]
            attn_metadata_i = attn_metadata[first_layer_in_group]
            if kv_cache_group_spec.kv_cache_spec.use_mla:
                attn_metadata_i.slot_mapping[:
                                             total_num_scheduled_tokens] = block_table.slot_mapping[:
                                                                                                    total_num_scheduled_tokens]
                input_positions = positions[:total_num_scheduled_tokens]
                attn_metadata_i.decode.input_positions[:
                                                       total_num_scheduled_tokens] = input_positions
                attn_metadata_i.decode.seq_lens[:
                                                total_num_scheduled_tokens] = (
                                                    input_positions + 1).to(
                                                        self.seq_lens.dtype)
                cos, sin = self.model.model.layers[
                    0].self_attn.rotary_emb.get_cos_sin(
                        attn_metadata_i.decode.input_positions)
                attn_metadata_i.decode.cos = cos
                attn_metadata_i.decode.sin = sin
            else:
                attn_metadata_i.slot_mapping[:
                                             total_num_scheduled_tokens] = block_table.slot_mapping[:
                                                                                                    total_num_scheduled_tokens]
                attn_metadata_i.slot_indices = torch.stack([
                    attn_metadata_i.slot_mapping // block_size,
                    attn_metadata_i.slot_mapping % block_size
                ],
                                                           dim=1)
                input_positions = positions[:total_num_scheduled_tokens]
                attn_metadata_i.seq_lens[:total_num_scheduled_tokens] = (
                    input_positions + 1).to(self.seq_lens.dtype)
                attn_metadata_i.seq_lens_list = attn_metadata_i.seq_lens.tolist(
                )
            if kv_cache_group_id == 0:
                self.full_attn_metadata = attn_metadata_i

            if self.enable_torchair_graph_mode and self.attn_state == AscendAttentionState.DecodeOnly:
                self.attn_metadata_builders[
                    kv_cache_group_id].mark_static_for_attn_metadata(
                        attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        index = torch.argmin(torch.cat(
            [cached_token,
             torch.full((num_reqs, 1), -1, device=self.device)],
            dim=1),
                             dim=1) - 1
        last_tokens = cached_token[torch.arange(num_reqs), index]
        if token_each_reqs == 1:
            self.input_ids[:num_reqs] = last_tokens.to(
                dtype=self.input_ids.dtype)
        else:
            input_ids_2d = self.input_ids.reshape(-1, token_each_reqs)
            input_ids_2d[:num_reqs, 0] = last_tokens
            input_ids_2d[:num_reqs, 1:] = cached_spec

        return attn_metadata, positions

    def _execute_model(
        self,
        scheduler_output,
        attn_metadata,
        graph_pad_size,
        sample_indices,
        positions,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start_before_f = time.time()
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        input_ids = self.input_ids[:num_input_tokens]
        model_kwargs = {}
        raw_hidden_states = None
        attn_state = next(iter(attn_metadata.values())).attn_state

        # padding input_ids
        if graph_pad_size >= 0:
            if attn_state == AscendAttentionState.DecodeOnly:
                padding = torch.zeros(graph_pad_size,
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
            else:
                vocab_size = self.model_config.get_vocab_size()
                padding = torch.randint(1,
                                        vocab_size, (graph_pad_size, ),
                                        dtype=input_ids.dtype,
                                        device=input_ids.device)
            input_ids = torch.cat([input_ids, padding])
        model_kwargs[
            "selected_indices"] = sample_indices if attn_state != AscendAttentionState.DecodeOnly else None

        start_fc = time.time()
        start_fc_exit = 0
        # Run forward pass
        with set_forward_context(attn_metadata, self.vllm_config):
            start_setup_connector = time.time()
            self.maybe_setup_kv_connector(scheduler_output)
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            # model_kwargs["use_cpu_cache"] =
            start_f = time.time()

            if model_extra_config.operator_opt_config.use_omni_placement:
                is_prompt = attn_state != AscendAttentionState.DecodeOnly
                global _GLOBAL_STEP
                self.planner.place_experts()
                _GLOBAL_STEP = _GLOBAL_STEP + 1 if not is_prompt else 0

            if self.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly:
                start_debug = time.time()
                logger.debug("Start running compiled model.")
                if not self.model_mark_static:
                    if isinstance(self.model, GraphCompileConfiguration):
                        self.model.mark_static_for_graph(
                            input_ids, positions, attn_metadata,
                            self.kv_caches)
                    else:
                        mark_static_for_graph_default(input_ids, positions,
                                                      self.kv_caches)
                    self.model_mark_static = True
                start_os_env = time.time()
                start_time = time.time()
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=None,
                    **model_kwargs,
                )
                end_model = time.time()
                cost_model = end_model - start_time
                cost_os_env = start_time - start_os_env
                cost_debug = start_debug - start_os_env
                logger.info(
                    f" ***** model forward: {cost_model:.6f}, os env: {cost_os_env:.6f}, debug: {cost_debug:.6f}"
                )
            else:
                logger.info("Start running eager model.")
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=None,
                    **model_kwargs,
                )
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfers(
                scheduler_output)
            start_fc_exit = time.time()
        if isinstance(forward_results, tuple):
            raw_hidden_states, hidden_states = forward_results
        else:
            hidden_states = forward_results
        start_ret = time.time()
        cost_before_fc = start_fc - start_before_f
        cost_fc = start_ret - start_fc
        cost_setup_connector = start_f - start_setup_connector
        cost_fc_exit = start_ret - start_fc_exit
        logger.debug(
            f" ***** before fc {cost_before_fc:.6f}, fc {cost_fc:.6f}={cost_setup_connector:.6f}+{cost_fc_exit:.6f}"
        )
        return hidden_states, raw_hidden_states, input_ids, finished_sending, finished_recving

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))
            loading_kv_failure = self.get_loading_kv_failure_req_ids()
        if not finished_sending and not finished_recving and not loading_kv_failure:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        output.loading_kv_failure = loading_kv_failure
        return output

    def load_kv_cache(self, info_load_reqs) -> List[int]:
        result = self.ems_adapter.load(info_load_reqs)
        return result

    @staticmethod
    def get_loading_kv_failure_req_ids() -> Optional[set[str]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_load_kv_failure_reqs()
        return None

    def _prepare_kv_cache(self, scheduler_output):
        if scheduler_output.blocks_to_swap_in is not None and len(
                scheduler_output.blocks_to_swap_in) > 0:
            blocks_to_swap_in = torch.tensor(
                scheduler_output.blocks_to_swap_in,
                device="cpu",
                dtype=torch.int64).view(-1, 2)
            self.cache_engine.swap_in(blocks_to_swap_in)

        if scheduler_output.blocks_to_swap_out is not None and len(
                scheduler_output.blocks_to_swap_out) > 0:
            blocks_to_swap_out = torch.tensor(
                scheduler_output.blocks_to_swap_out,
                device="cpu",
                dtype=torch.int64).view(-1, 2)
            self.cache_engine.swap_out(blocks_to_swap_out)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start = time.time()

        if EmsEnv.enable_vllm_ems:
            self.ems_adapter.sync_save_event()

        # Update KVConnector with the KVConnector metadata forward().
        self._update_states(scheduler_output)

        self._prepare_kv_cache(scheduler_output)

        self.total_step = scheduler_output.num_step
        # cached values
        attn_metadata = None
        positions = None
        graph_pad_size = None
        sampled_tokens = None
        sample_indices = None

        # cached return values
        cached_sampled_token_ids = []
        cached_spec_token = []
        cached_logprobs = []
        cached_prompt_logprobs_dict = []
        finished_sending = set()
        accepted_num = 0
        finished_recving = set()
        loading_kv_failure = set()
        for self.curr_step in range(self.total_step):
            start_1 = time.time()
            if not scheduler_output.total_num_scheduled_tokens:
                if get_dp_group().world_size > 1:
                    self._dummy_run(1)
                else:
                    time.sleep(0.001)  # release GIL
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)

            if self.curr_step == 0:
                attn_metadata, graph_pad_size, sample_indices, positions = self._prepare_inputs(
                    scheduler_output)
            else:
                attn_metadata, positions = self._simple_prepare_inputs(
                    attn_metadata, positions, sampled_tokens,
                    cached_spec_token[-1], accepted_num)
            hidden_states, raw_hidden_states, input_ids, temp_finished_sending, temp_finished_recving = self._execute_model(
                scheduler_output, attn_metadata, graph_pad_size,
                sample_indices, positions, intermediate_tensors)

            if temp_finished_sending is not None:
                finished_sending.update(temp_finished_sending)
            if temp_finished_recving is not None:
                finished_recving.update(temp_finished_recving)
            tmp_loading_kv_failure = self.get_loading_kv_failure_req_ids()
            if tmp_loading_kv_failure is not None:
                loading_kv_failure.update(tmp_loading_kv_failure)
            start_2 = time.time()
            if hidden_states.shape[0] == sample_indices.shape[0]:
                # assume indices=[x1,x2,...,xn], if xn >= n, we cannot slice,
                # if xn < n, then indices=[0,1,...,n-1], no need to slice.
                logits = self.model.compute_logits(hidden_states, None)
            else:
                logits = self.model.compute_logits(
                    hidden_states[sample_indices], None)
            start_3 = time.time()
            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)
            start_4 = time.time()

            # TODO move into scheduler or prepare_inputs
            # find the requests that are doing chunk prefill
            discard_sampled_tokens_req_indices = []
            chunk_next_tokens = [] if self.use_spec_decode else None
            chunk_next_indices = [] if self.use_spec_decode else None

            num_decodes = self.attn_metadata_builders[0]._num_decodes
            num_prefills = self.attn_metadata_builders[0]._num_prefills
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() -
                                             NPU_GENERATOR_OFFSET_STEP)
                    # Record the index of the request that should not be sampled,
                    # so that we could clear the sampled tokens before returning.
                    discard_sampled_tokens_req_indices.append(i)
                    if self.use_spec_decode:
                        chunk_next_tokens.append(
                            req_state.get_token_id(seq_len))
                        chunk_next_indices.append(
                            sample_indices[-num_prefills + i])
            if self.use_spec_decode and len(chunk_next_tokens) > 0:
                chunk_next_tokens = torch.tensor(chunk_next_tokens)  # CPU
                chunk_next_tokens_buffer = self.chunk_next_tokens[:
                                                                  chunk_next_tokens
                                                                  .numel()]
                chunk_next_tokens_buffer.copy_(chunk_next_tokens,
                                               non_blocking=True)
                chunk_next_tokens = chunk_next_tokens_buffer
                chunk_next_indices = torch.stack(chunk_next_indices)
            else:
                chunk_next_tokens = None
                chunk_next_indices = None
            start_5 = time.time()

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if not self.use_spec_decode:
                sampler_output = self.sampler(
                    logits=logits, sampling_metadata=sampling_metadata)
            else:
                sampler_output, last_accepted_index, accepted_num = self.drafter.verify_and_prepare_inputs(
                    input_ids=input_ids,
                    logits=logits,
                    logits_indices=sample_indices,
                    sampling_metadata=sampling_metadata,
                    num_decodes=num_decodes,
                    num_prefills=num_prefills,
                    chunk_next_tokens=chunk_next_tokens,
                    chunk_next_indices=chunk_next_indices,
                )
            start_6 = time.time()

            if not self.use_spec_decode:
                # Speculative decoding is not enabled.
                spec_tokens_tensor = None
            elif self.speculative_config.method in MTP_METHOD_NAME_LIST:
                spec_tokens_tensor = self.drafter.propose(
                    num_tokens=input_ids.numel(),
                    positions=positions,
                    kv_caches=self.kv_caches,
                    attn_metadata=attn_metadata,
                    previous_hidden_states=raw_hidden_states,
                    last_accepted_index=last_accepted_index,
                    sample_indices=sample_indices,
                )
            else:
                raise ValueError(
                    f"Speculative method {self.speculative_config.method} is not supported in this version."
                )

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists(
            ) if logprobs_tensors is not None else None

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                # [[bonus,b_forward], [forward], [bonus,b_forward], [bonus,b_forward],..]
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            sampled_tokens = sampled_token_ids
            spec_token_ids: list[list[
                int]] = None if spec_tokens_tensor is None else spec_tokens_tensor.tolist(
                )

            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
                if spec_token_ids is not None:
                    spec_token_ids[i].clear()
            # Clear KVConnector state after all KVs are generated.
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            cached_sampled_token_ids.append(valid_sampled_token_ids)
            cached_spec_token.append(spec_tokens_tensor)
            cached_logprobs.append(logprobs_lists)
            cached_prompt_logprobs_dict.append({})

            cost_upd_states = start_1 - start
            cost_proc_reqs = start_2 - start_1
            cost_logits = start_3 - start_2
            cost_bitmask = start_4 - start_3
            cost_disc = start_5 - start_4
            cost_sampler = start_6 - start_5
            cost_output = time.time() - start_6
            cost = cost_upd_states + cost_proc_reqs + cost_logits + cost_bitmask + cost_sampler + cost_disc + cost_output
            logger.info(
                f" ***** execute model cost:{cost:.6f}={cost_upd_states:.6f}+{cost_proc_reqs:.6f}+{cost_logits:.6f}+{cost_bitmask:.6f}+{cost_sampler:.6f}+{cost_disc:.6f}+{cost_output:.6f}"
            )
        return_spec_token = [None] * (self.total_step - 1)
        return_spec_token.append(spec_token_ids)

        if EmsEnv.enable_vllm_ems:
            self.ems_adapter.async_save(scheduler_output)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=cached_sampled_token_ids,
            spec_token_ids=return_spec_token,
            logprobs=cached_logprobs,
            prompt_logprobs_dict=cached_prompt_logprobs_dict,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            loading_kv_failure=loading_kv_failure,
        )

    @torch.inference_mode()
    def _dummy_run(self,
                   num_tokens: int,
                   is_capture_model: bool = False) -> torch.Tensor:
        if self.is_multimodal_model:
            input_ids, inputs_embeds = None, self.inputs_embeds[:num_tokens]
        else:
            input_ids, inputs_embeds = self.input_ids[:num_tokens], None

        # Prepare intermediate_tensors
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=num_tokens,
                    dtype=self.dtype,
                    device=self.device)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        positions = self.mrope_positions[:, :
                                         num_tokens] if self.uses_mrope else self.positions[:
                                                                                            num_tokens]
        raw_hidden_states = None

        # No kv_caches: profile run
        if not self.kv_caches:
            with set_forward_context(None, self.vllm_config):
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
                if isinstance(forward_results, tuple):
                    raw_hidden_states, hidden_states = forward_results
                else:
                    hidden_states = forward_results
                if self.use_spec_decode:
                    self.drafter.propose(
                        num_tokens=num_tokens,
                        positions=positions,
                        kv_caches=None,
                        attn_metadata=None,
                        previous_hidden_states=raw_hidden_states,
                        last_accepted_index=None,
                        sample_indices=None,
                    )
            return hidden_states

        # With kv_caches: dummy run for graph capture/placement
        if self.enable_torchair_graph_mode and len(self.decode_gear_list) > 1:
            self.max_batch_size = self._get_max_token_num(
                self.vllm_config.parallel_config.data_parallel_size > 1,
                num_tokens)
        fake_input = torch.zeros(self.max_batch_size,
                                 dtype=input_ids.dtype,
                                 device=input_ids.device)
        fake_positions = torch.zeros(self.max_batch_size,
                                     dtype=input_ids.dtype,
                                     device=input_ids.device)
        input_ids, positions = fake_input, fake_positions
        self.attn_mask = None
        self.attn_state = AscendAttentionState.DecodeOnly

        # Build dummy attn_metadata
        attn_metadata = {}
        is_pd_seperate_d = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if not isinstance(builder, DummyAttentionMetadataBuilder):
                raise ValueError(
                    f"{builder} does not implement DummyAttentionMetadataBuilder"
                )
            attn_metadata_i = builder.build_dummy(num_tokens,
                                                  self.max_batch_size)
            if self.enable_torchair_graph_mode and is_pd_seperate_d:
                builder.mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata,
            "selected_indices": None
        }
        with set_forward_context(attn_metadata, self.vllm_config):
            is_not_pd_seperate_and_capture_model = self.vllm_config.kv_transfer_config is None and is_capture_model
            use_compile = self.enable_torchair_graph_mode and (
                is_pd_seperate_d or is_not_pd_seperate_and_capture_model)
            if use_compile:
                logger.debug("Start running dummy compiled model.")
                if not self.dummy_model_mark_static:
                    if isinstance(self.model, GraphCompileConfiguration):
                        self.model.mark_static_for_graph(
                            input_ids, positions, attn_metadata,
                            self.kv_caches)
                    else:
                        mark_static_for_graph_default(input_ids, positions,
                                                      self.kv_caches)
                    self.dummy_model_mark_static = True
            else:
                logger.debug("Start running dummy eager model.")
            forward_results = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None if use_compile else inputs_embeds,
                **model_kwargs)
            if isinstance(forward_results, tuple):
                raw_hidden_states, hidden_states = forward_results
            else:
                hidden_states = forward_results
            if self.use_spec_decode:
                self.drafter.prepare_dummy_input(input_ids)
                self.drafter.propose(
                    num_tokens=input_ids.numel(),
                    positions=positions,
                    kv_caches=self.kv_caches,
                    attn_metadata=attn_metadata,
                    previous_hidden_states=raw_hidden_states,
                    last_accepted_index=None,
                    sample_indices=None,
                )
        return hidden_states

    def profile_run(self) -> None:
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
            hidden_states = self._dummy_run(self.max_batch_size *
                                            get_dp_group().world_size)
        else:
            hidden_states = self._dummy_run(self.max_num_tokens)

        NPUPlatform.synchronize()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() if not int(os.getenv(
                "NO_NPU_MOCK", "0")) else nullcontext() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
            self.drafter_list = []
            if hasattr(self, "drafter"):
                self.drafter.load_model(self.model)

        if not int(os.getenv("NO_NPU_MOCK", "0")):
            logger.info("Loading model weights took %.4f GB",
                        m.consumed_memory / float(2**30))

        if model_extra_config.operator_opt_config.use_omni_placement:
            first_k_dense_replace_names = [
                'num_dense_layers', 'first_k_dense_replace'
            ]
            first_k_dense_replace = get_attr_by_names(
                self.model.config, first_k_dense_replace_names, 3)
            param_dict = dict(self.model.named_parameters())
            self.planner = OmniPlanner(
                config_file=model_extra_config.operator_opt_config.
                omni_placement_config_path)
            self.planner.init_dram_weights(
                param_dict, first_k_dense_replace=first_k_dense_replace)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        cpu_caches: Dict[str, torch.Tensor] = {}
        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=is_pin_memory_available(),
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.cache_config.block_size)
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy(
        )
        self.initialize_attn_backend(kv_cache_config)
        preemption_mode = self.vllm_config.scheduler_config.preemption_mode

        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                if tensor_config.size % kv_cache_spec.page_size_bytes != 0:
                    raise RuntimeError(
                        "tensor_config.size must be divisible by kv_cache_spec.page_size_bytes"
                    )
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    kv_caches[layer_name] = self.attn_backends[
                        i].init_kv_cache_each_layer(
                            kv_cache_shape, self.dtype, self.device,
                            self.model_config, self.enable_torchair_graph_mode)
                    if preemption_mode and preemption_mode == "swap":
                        cpu_num_blocks = int(
                            self.vllm_config.cache_config.swap_space_bytes //
                            kv_cache_spec.page_size_bytes //
                            len(kv_cache_config.tensors))
                        cpu_kv_cache_shape = self.attn_backends[
                            i].get_kv_cache_shape(cpu_num_blocks,
                                                  kv_cache_spec.block_size,
                                                  kv_cache_spec.num_kv_heads,
                                                  kv_cache_spec.head_size)
                        cpu_caches[layer_name] = self.attn_backends[
                            i].init_kv_cache_each_layer(
                                cpu_kv_cache_shape, self.dtype, "cpu",
                                self.model_config,
                                self.enable_torchair_graph_mode)
                else:
                    raise ValueError("Unknown KV cache spec type.")

        if preemption_mode and preemption_mode == "swap":
            self.cache_engine = CacheEngine(self.attn_backends,
                                            self.kv_cache_config,
                                            gpu_cache=kv_caches,
                                            cpu_cache=cpu_caches)

        if not int(os.getenv("NO_NPU_MOCK", "0")):
            bind_kv_cache(
                kv_caches,
                self.vllm_config.compilation_config.static_forward_context,
                self.kv_caches)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if EmsEnv.enable_vllm_ems:
            self.ems_adapter.bind_kvcaches(self.kv_caches)

    def initialize_omni_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        from omni.accelerators.cache.omni_cache import create_omni_cache
        if self.vllm_config.kv_transfer_config is None:
            raise NotImplementedError(
                "Currently only support PD disaggregation, but KV transfer config is None."
            )
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise RuntimeError(
                f"Only support single KV cache group, but got {len(kv_cache_config.kv_cache_groups)}."
            )

        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=is_pin_memory_available(),
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.cache_config.block_size)
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy(
        )
        self.initialize_attn_backend(kv_cache_config)

        omni_cache = create_omni_cache(
            kv_cache_config=self.kv_cache_config,
            vllm_config=self.vllm_config,
            runner=self,
        )

        get_kv_transfer_group().register_kv_caches(
            omni_cache.MEMMAP_PATH,
            omni_cache.dtype,
            block_len_dtype=omni_cache.block_len_dtype,
            start_offset=omni_cache.dp_offset,
            omni_cache=omni_cache)

        self.omni_cache = omni_cache

    def capture_model(self) -> None:
        if self.enable_torchair_graph_mode:
            decode_gear_list = self.decode_gear_list
            graph_num = len(decode_gear_list)
            use_spec_decode = False if not self.vllm_config.speculative_config else (
                self.vllm_config.speculative_config.method
                in MTP_METHOD_NAME_LIST)
            base_time = 4
            min_time = base_time * graph_num
            max_time = 2 * base_time * graph_num
            mtp_time_rate = 1.5
            if use_spec_decode:
                min_time *= mtp_time_rate
                max_time *= mtp_time_rate

            logger.info(f"The current directory is {os.getcwd()}")
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                min_time, max_time)
            # Trigger torchair graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for idx, num_tokens in enumerate(reversed(decode_gear_list)):
                self._dummy_run(num_tokens, True)
                logger.info("Batchsize %d is compiled successfully: %d/%d.",
                            num_tokens, idx + 1, graph_num)
        else:
            logger.warning(
                "Skipping NPU graph capture. Please add "
                "-O %s to use NPU graphs.", CompilationLevel.PIECEWISE)

        if model_extra_config.operator_opt_config.use_omni_placement:
            self.planner.start_dynamic_optimize_expert_load_balance()

    def _get_closest_gear(self, max_num_token):
        for gear in self.decode_gear_list:
            if gear >= max_num_token:
                return gear
        raise ValueError(
            f"decode input batch size {max_num_token} exceeds maximum gear {max(self.decode_gear_list)}."
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        if int(os.getenv("NO_NPU_MOCK", "0")):
            kv_cache_spec: dict[str, KVCacheSpec] = {}
            block_size = self.vllm_config.cache_config.block_size
            use_mla = self.vllm_config.model_config.use_mla
            kv_cache_spec["mock.0"] = FullAttentionSpec(block_size=block_size,
                                                        num_kv_heads=1,
                                                        head_size=16,
                                                        dtype=torch.bfloat16,
                                                        use_mla=use_mla)
            return kv_cache_spec
        return super().get_kv_cache_spec()

    def _get_max_token_num(self, is_enable_dp, num_tokens):
        if is_enable_dp:
            local_batch_tensor = torch.tensor([num_tokens],
                                              dtype=torch.int64,
                                              device='cpu')
            dist.all_reduce(local_batch_tensor,
                            group=get_dp_group().cpu_group,
                            op=dist.ReduceOp.MAX)
            global_batch_size = local_batch_tensor.item()
            return self._get_closest_gear(global_batch_size)
        return self._get_closest_gear(num_tokens)
