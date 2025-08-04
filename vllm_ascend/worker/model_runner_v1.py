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
import math
import os
import time
import types
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
import torch.nn as nn
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (get_dp_group, get_pp_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LazyLoader, cdiv)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (bind_kv_cache, gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.attention.attention_v1_torchair import AscendTorchairMetadata
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.multistream.ms_split import compute_split_seq_index
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.torchair.utils import (check_torchair_cache_exist,
                                        write_kv_cache_bytes_to_file)
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               ProfileExecuteDuration, is_310p,
                               maybe_converting_weight_acl_format,
                               vllm_version_is)
from vllm_ascend.worker.eagle_proposer_v1 import EagleProposer
from vllm_ascend.worker.mtp_proposer_v1 import MtpProposer
from vllm_ascend.worker.npu_input_batch import CachedRequestState, InputBatch

if not vllm_version_is("0.10.0"):
    from vllm.tasks import GenerationTask, SupportedTask
    from vllm.v1.worker.kv_connector_model_runner_mixin import \
        KVConnectorOutput

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu
import vllm.envs as envs_vllm

import vllm_ascend.envs as envs_ascend

if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


class NPUModelRunner(LoRAModelRunnerMixin):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.device = device
        self.dtype = self.model_config.dtype
        if envs.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:
            # TODO: drop the env config to use ascend sampler by default
            from vllm_ascend.sample.sampler import AscendSampler

            self.sampler = AscendSampler()
        else:
            from vllm.v1.sample.sampler import Sampler

            self.sampler = Sampler()

        # Lazy initialization, these will be set after __init__
        self.kv_caches: List[torch.Tensor] = []
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}
        self.attn_mask = None
        self.attn_state = None
        self.requests: Dict[str, CachedRequestState] = {}
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        ascend_config = get_ascend_config()
        if ascend_config.ascend_scheduler_config.enabled:
            self.chunked_prefill_enabled = self.scheduler_config.chunked_prefill_enabled
        else:
            self.chunked_prefill_enabled = True

        if self.cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.is_pooling_model = self.model_config.pooler_config is not None
        if self.is_multimodal_model:
            self.inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.model_config.get_hidden_size()),
                dtype=self.dtype,
                device=self.device)

        self.graph_block_tables = np.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req), dtype=np.int32)

        # Set up Attention
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))
        self.attn_mask_builder = AttentionMaskBuilder(
            min(self.model_config.max_model_len,
                int(os.getenv("PAGED_ATTENTION_MASK_LEN", 10000))), self.dtype)

        # Set up speculative decoding.
        self.use_aux_hidden_state_outputs = False
        self.use_spec_decode = False
        self.spec_attn_mask = None
        self.use_eagle = False
        self.drafter: Optional[Union[NgramProposer, EagleProposer,
                                     MtpProposer]] = None
        if self.speculative_config:
            self.use_spec_decode = True
            self.spec_attn_mask = torch.triu(torch.ones(2048,
                                                        2048,
                                                        dtype=torch.bool),
                                             diagonal=1).to(self.device)
            if get_pp_group().is_last_rank:
                if self.speculative_config.method == "ngram":
                    self.drafter = NgramProposer(self.vllm_config)
                elif self.speculative_config.method in ["eagle", "eagle3"]:
                    self.use_eagle = True
                    self.drafter = EagleProposer(self.vllm_config, self.device,
                                                 self)  # type: ignore
                    if self.speculative_config.method == "eagle3":
                        self.use_aux_hidden_state_outputs = True
                elif self.speculative_config.method == 'deepseek_mtp':
                    self.drafter = MtpProposer(self.vllm_config, self)
                else:
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                self.rejection_sampler = AscendRejectionSampler()

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)

        self.uses_mrope = self.model_config.uses_mrope
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True)
            self.mrope_positions_np = self.mrope_positions_cpu.numpy()

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np: npt.NDArray[np.int32] = np.arange(max(
            self.max_num_reqs + 1, self.model_config.max_model_len,
            self.max_num_tokens),
                                                          dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=True)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=True)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        self.use_aclgraph = (self.vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not self.model_config.enforce_eager)
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        self.new_kv_cache_bytes = -1
        self.torchair_compiled_model = None  # type: ignore
        self.torchair_compiled_models = {}  # type: ignore
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.use_cached_npu_graph = ascend_config.torchair_graph_config.use_cached_graph
        self.torchair_graph_batch_sizes = ascend_config.torchair_graph_config.graph_batch_sizes
        if ascend_config.torchair_graph_config.graph_batch_sizes_init:
            self.init_torchair_graph_batch_sizes()
        if len(self.torchair_graph_batch_sizes) == 0:
            # TODO(zzzzwwjj): check torchair_graph_batch_sizes init code
            self.torchair_graph_batch_sizes = [self.max_num_reqs]

        torch._dynamo.cache_size.config.cache_size_limit += len(
            self.torchair_graph_batch_sizes)
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._logging.set_logs(
            recompiles=envs_ascend.VLLM_ASCEND_TRACE_RECOMPILES)

        # NOTE: we need to use `in_profile_run` to determine whether `enable_force_load_balance` is True
        self.in_profile_run = False

        # kv role
        self.is_kv_producer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params
            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if pooling_params:
                assert (task := pooling_params.task) is not None, (
                    "You did not set `task` in the API")
                model = cast(VllmModelForPooling, self.model)
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=new_req_data.pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        is_last_rank = get_pp_group().is_last_rank
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            req_state.num_computed_tokens = num_computed_tokens
            if not is_last_rank:
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec decode tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(  # type: ignore[call-overload]
                        req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)

            self.input_batch.block_table.append_row(new_block_ids, req_index)

            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                # Add spec_token_ids to token_ids_cpu.
                spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ())
                if spec_token_ids:
                    start_index = end_token_index
                    end_token_index += len(spec_token_ids)
                    self.input_batch.token_ids_cpu[
                        req_index,
                        start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
                self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices.sort(reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _get_forward_metadata_across_dp(
        self,
        maybe_padded_num_tokens: int,
        num_tokens: int,
        with_prefill: bool,
        enable_dbo: bool = False,
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        if self.dp_size == 1:
            return maybe_padded_num_tokens, None, with_prefill, enable_dbo

        num_tokens_across_dp = [0] * self.dp_size * 2
        num_tokens_across_dp[self.dp_rank] = maybe_padded_num_tokens
        num_tokens_across_dp[self.dp_size + self.dp_rank] = num_tokens
        forward_metadata = torch.tensor(num_tokens_across_dp +
                                        [with_prefill, not enable_dbo],
                                        device="cpu",
                                        dtype=torch.int32)
        dist.all_reduce(forward_metadata, group=get_dp_group().cpu_group)
        with_prefill = bool(forward_metadata[-2])

        # NOTE: when with_prefill is false before all_reduce and true after all_reduce, we need to revert pad.
        if with_prefill:
            num_tokens_across_dp = forward_metadata[self.dp_size:self.dp_size *
                                                    2]
            maybe_padded_num_tokens = num_tokens
        else:
            num_tokens_across_dp = forward_metadata[:self.dp_size]

        # NOTE: when in torchair_graph_mode, we need to pad local_num_tokens to
        # `max_tokens_across_dp`, in other situation it is not necessary.
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = torch.max(num_tokens_across_dp).item()
            num_tokens_across_dp = torch.tensor([maybe_padded_num_tokens] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, not bool(
            forward_metadata[-1])

    def _check_dbo_is_valid(self, query_lens: torch.Tensor,
                            attn_state: AscendAttentionState,
                            num_tokens: int) -> bool:
        # do the checks for dp + dbo
        if attn_state in [
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        ]:
            return False
        # considering the case that one dp rank may enable dbo while others may not
        if not self.vllm_config.model_config.use_mla or not envs_ascend.VLLM_ASCEND_ENABLE_DBO:
            return False
        # TODO: remove it if token-level microbatch is enabled
        [token_index,
         seq_index] = compute_split_seq_index(query_lens, attn_state,
                                              num_tokens)
        if token_index == 0 or seq_index == 0 or seq_index == len(
                query_lens) or num_tokens < 256:
            return False
        return True

    def get_eagle_atten_dict(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Union[AscendMetadata, AscendMLAMetadata,
                         AscendTorchairMetadata]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        # NOTE(Chen): there is exactly one KV cache group that contains all
        # attetnion layers in the model for now, so the current logic for
        # getting attn_metadata is not related to kv_cache_group information.
        # Will extend this part to support multiple KV cache groups later.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table = self.input_batch.block_table[kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        attn_metadata: dict[str, Union[AscendMetadata, AscendMLAMetadata,
                                       AscendTorchairMetadata]] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            attn_metadata_i = self.attn_metadata_builder.build(
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
            )
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_generation_tasks(self) -> "list[GenerationTask]":
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _make_attention_mask(self, seq_lens, query_lens, position,
                             attn_state) -> torch.Tensor:
        # Chunk Prefill situation.
        if attn_state == AscendAttentionState.ChunkedPrefill:
            return self.attn_mask_builder.get_splitfuse_attn_mask(
                seq_lens, query_lens, position, self.dtype, self.device)
        # Prefill without cache situation.
        elif attn_state == AscendAttentionState.PrefillNoCache:
            max_seq_len = max(seq_lens, default=0)
            return self.attn_mask_builder.get_attn_mask(
                max_seq_len, self.dtype, self.device)
        # Prefill with cache hit.
        elif attn_state == AscendAttentionState.PrefillCacheHit:
            return self.attn_mask_builder.get_attn_mask(
                128, self.dtype, self.device)
        # Decode-only situation.
        else:
            return None

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions_np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                           device=self.device)

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=len(grouped_mm_inputs),
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[Union[AscendMetadata, AscendMLAMetadata,
                     AscendTorchairMetadata], torch.Tensor, SpecDecodeMetadata,
               torch.Tensor, int, torch.Tensor, torch.Tensor, np.ndarray,
               Optional[set[str]], Optional[set[str]]]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
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
        self.input_batch.block_table.commit_block_table(num_reqs)

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
        logits_indices = cu_num_tokens - 1
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
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
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        maybe_padded_num_tokens = total_num_scheduled_tokens
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                total_num_scheduled_tokens)
        (padded_num_tokens_across_dp, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._get_forward_metadata_across_dp(
             maybe_padded_num_tokens, total_num_scheduled_tokens, with_prefill,
             enable_dbo)
        extra_builder_kwargs['enable_dbo_across_dp'] = enable_dbo

        if self.torchair_graph_enabled and not with_prefill:
            graph_pad_size = padded_num_tokens_across_dp - total_num_scheduled_tokens

            extra_builder_kwargs['graph_pad_size'] = graph_pad_size

        if self.vllm_config.model_config.use_mla:
            extra_builder_kwargs[
                "query_start_loc"] = self.query_start_loc[:num_reqs + 1]
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                **extra_builder_kwargs,
            )
            attn_metadata.num_input_tokens = num_input_tokens
        else:
            attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                **extra_builder_kwargs,
            )

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

        if self.torchair_graph_enabled and not with_prefill:
            input_ids = self.input_ids[:padded_num_tokens_across_dp]
            positions = self.positions[:padded_num_tokens_across_dp]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        # Run forward pass
        with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=padded_num_tokens_across_dp,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=with_prefill,
                num_actual_tokens=total_num_scheduled_tokens):
            with ProfileExecuteDuration().capture_async("forward"):
                self.maybe_setup_kv_connector(scheduler_output)
                model_kwargs = {}
                if self.torchair_graph_enabled:
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                if self.torchair_graph_enabled and not with_prefill:
                    maybe_converting_weight_acl_format(self.model,
                                                       ACL_FORMAT_FRACTAL_NZ)

                    compiled_model = self._get_torchair_lazy_compiled_model(
                        padded_num_tokens_across_dp)
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
                else:
                    assert self.model is not None
                    maybe_converting_weight_acl_format(self.model,
                                                       ACL_FORMAT_FRACTAL_ND)

                    hidden_states = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )

        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfer(
            scheduler_output)
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
            logits_indices = spec_decode_metadata.logits_indices

        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states

        return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                total_num_scheduled_tokens, logits_indices, aux_hidden_states,
                num_scheduled_tokens, finished_sending, finished_recving)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ) -> torch.Tensor:
        grammar_bitmask = scheduler_output.grammar_bitmask

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.input_batch.req_id_to_index.items(),
                     key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.zeros_like(grammar_bitmask,
                                       shape=(logits.shape[0],
                                              grammar_bitmask.shape[1]))
        cumulative_index = 0
        seq = sorted(scheduler_output.structured_output_request_ids.items(),
                     key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # NOTE:
        # 1. XGrammar bitmask applying only supports CPU and GPU.
        # 2. The logits and bitmask should be on the same device.
        # 3. XGrammar logits on CPU only supports float32 dtype.
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask,
            indices=out_indices,
        )
        return logits.to(self.device).to(logits_dtype)

    def _get_spec_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: Union[AscendMetadata, AscendMLAMetadata,
                             AscendTorchairMetadata],
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        elif self.speculative_config.method == "ngram":
            spec_token_ids = self._generate_ngram_token_ids(
                valid_sampled_token_ids)
        elif self.speculative_config.method == "eagle":
            raise NotImplementedError("Eagle Is Not Supported Yet.")
        elif self.speculative_config.method == "eagle3":
            spec_token_ids = self._generate_eagle3_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, aux_hidden_states)
        elif self.speculative_config.method == 'deepseek_mtp':
            spec_token_ids = self._generate_mtp_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, attn_metadata)
        return spec_token_ids

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        finished_sending: Optional[set[str]] = None,
        finished_recving: Optional[set[str]] = None,
        kv_connector_output: Optional["KVConnectorOutput"] = None,
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        extracted_hidden_states = list(
            torch.split(hidden_states[:num_scheduled_tokens],
                        num_scheduled_tokens_np.tolist()))

        pooling_metadata = self.input_batch.pooling_metadata

        raw_pooler_output = self.model.pooler(
            hidden_states=extracted_hidden_states,
            pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        seq_lens = self.seq_lens[:self.input_batch.num_reqs]
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.cpu())
            else:
                pooler_output.append(None)
        extra_args = ({
            "finished_sending": finished_sending,
            "finished_recving": finished_recving
        } if vllm_version_is("0.10.0") else {
            "kv_connector_output": kv_connector_output
        })

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            **extra_args,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        with ProfileExecuteDuration().capture_async(
                "prepare input and forward"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)
            (attn_metadata, hidden_states, spec_decode_metadata, positions,
             num_scheduled_tokens, logits_indices, aux_hidden_states,
             num_scheduled_tokens_np, finished_sending,
             finished_recving) = (self._process_reqs(scheduler_output,
                                                     intermediate_tensors))
        kv_connector_output = None
        if not vllm_version_is("0.10.0"):
            kv_connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving)
            finished_sending = None
            finished_recving = None
        with ProfileExecuteDuration().capture_async("post process"):
            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return the hidden states.
                if not broadcast_pp_output:
                    if kv_connector_output is not None:
                        hidden_states.kv_connector_output = kv_connector_output
                    else:
                        #TODO: Remove this after we drop vllm v0.10.0
                        hidden_states.finished_sending = finished_sending
                        hidden_states.finished_recving = finished_recving
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(hidden_states, num_scheduled_tokens,
                                      num_scheduled_tokens_np,
                                      finished_sending, finished_recving,
                                      kv_connector_output)
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                logits = self.apply_grammar_bitmask(scheduler_output, logits)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self.input_batch.sampling_metadata
            if spec_decode_metadata is None:
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                # When indexing with a tensor (bonus_logits_indices), PyTorch
                # creates a new tensor with separate storage from the original
                # logits tensor. This means any in-place operations on bonus_logits
                # won't affect the original logits tensor.
                assert logits is not None
                bonus_logits = logits[
                    spec_decode_metadata.bonus_logits_indices]
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
                bonus_token_ids = sampler_output.sampled_token_ids

                # Just like `bonus_logits`, `target_logits` is a new tensor with
                # separate storage from the original `logits` tensor. Therefore,
                # it is safe to update `target_logits` in place.
                target_logits = logits[
                    spec_decode_metadata.target_logits_indices]
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    None,  # draft_probs
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
                sampler_output.sampled_token_ids = output_token_ids

            discard_sampled_tokens_req_indices: list[int] = []
            # TODO(woosuk): The following loop can be slow since it iterates over
            # the requests one by one. Optimize.
            discard_sampled_tokens_req_indices = []
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - 4)
                    discard_sampled_tokens_req_indices.append(i)

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
                if logprobs_tensors is not None else None

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:num_scheduled_tokens],
                scheduler_output,
            )

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = sampled_token_ids.tolist()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )

            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
            # Cache the sampled tokens in the model runner, so that the schedulerAdd commentMore actions
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
                if not sampled_ids:
                    continue

                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(sampled_ids)
                assert end_idx <= self.model_config.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.model_config.max_model_len}")

                self.input_batch.token_ids_cpu[req_idx,
                                               start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx
                self.input_batch.num_tokens[req_idx] = end_idx
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests[req_id]
                req_state.output_token_ids.extend(sampled_ids)

            spec_token_ids = self._get_spec_token_ids(
                valid_sampled_token_ids,
                sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({
            "finished_sending": finished_sending,
            "finished_recving": finished_recving
        } if vllm_version_is("0.10.0") else {
            "kv_connector_output": kv_connector_output
        })

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            **extra_args,
        )

        durations = ProfileExecuteDuration().pop_captured_sync()
        if durations:
            dr_str = [
                f"[{tag}]:{duration:.2f}ms"
                for tag, duration in durations.items()
            ]
            captured_name = "Decode" if self.attn_state == AscendAttentionState.DecodeOnly else "Prefill"
            logger.info("Profile execute duration [%s]:%s", captured_name,
                        " ".join(dr_str))

        return model_runner_output

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        with set_ascend_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfer(scheduler_output))
            # For the case of no forward caused by receiving remote kv,
            # one round of dummy inference is necessary
            # to prevent hang over the collective calls.
        if not finished_sending and not finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        return output

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        skip_attn: bool = True,
        with_prefill: bool = False,
        is_torchair_compile: bool = False,
    ) -> torch.Tensor:
        maybe_padded_num_tokens = num_tokens
        if self.torchair_graph_enabled and not with_prefill:
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                num_tokens)

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill,
         _) = self._get_forward_metadata_across_dp(maybe_padded_num_tokens,
                                                   num_tokens, with_prefill,
                                                   False)

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer:
            with_prefill = True

        # NOTE: If torchair graph mode and not with_prefill,
        # we can't skip_attn, it will cause graph recompile.
        if self.torchair_graph_enabled and not with_prefill:
            attn_metadata = self.attn_metadata_builder.build_torchair_graph_dummy(
                num_reqs=num_tokens, num_actual_tokens=1)
        elif skip_attn:
            attn_metadata = None
        else:
            # TODO(zzzzwwjj): when aclgraph and full graph mode, we need build attn_metadata
            attn_metadata = None

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

            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=self.in_profile_run,
                    num_actual_tokens=0,
            ):
                model_kwargs = {}
                if self.torchair_graph_enabled and not with_prefill:
                    # Only mark static while compiling
                    if is_torchair_compile:
                        torch._dynamo.mark_static(input_ids)
                        torch._dynamo.mark_static(positions)
                        torch._dynamo.mark_static(
                            attn_metadata.decode.block_table)
                        torch._dynamo.mark_static(
                            attn_metadata.decode.input_positions)
                        torch._dynamo.mark_static(
                            get_forward_context().mc2_mask)
                        if hasattr(attn_metadata.decode, "sin"):
                            torch._dynamo.mark_static(attn_metadata.decode.sin)
                            torch._dynamo.mark_static(attn_metadata.decode.cos)
                        torch._dynamo.mark_static(attn_metadata.slot_mapping)
                        for kv in self.kv_caches:
                            assert isinstance(
                                kv, tuple), "kv_cache must be a tuple"
                            torch._dynamo.mark_static(kv[0])
                            torch._dynamo.mark_static(kv[1])

                    maybe_converting_weight_acl_format(self.model,
                                                       ACL_FORMAT_FRACTAL_NZ)

                    compiled_model = self._get_torchair_lazy_compiled_model(
                        num_tokens)
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                    hidden_states = compiled_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=None,
                        **model_kwargs,
                    )
                else:
                    maybe_converting_weight_acl_format(self.model,
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

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    def profile_run(self) -> None:
        # Trigger compilation for general shape.
        with self.set_in_profile_run():
            hidden_states = self._dummy_run(self.max_num_tokens,
                                            with_prefill=True)
        output = None
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                # For profile, have maximum num_reqs and that collectively have
                # maximum num_tokens.
                min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
                num_scheduled_tokens_list = [min_tokens_per_req
                                             ] * self.max_num_reqs
                num_scheduled_tokens_list[
                    -1] += self.max_num_tokens % self.max_num_reqs
                num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                                dtype=np.int32)
                logit_indices = np.cumsum(num_scheduled_tokens) - 1
                # TODO: need to rum a dummy sampler for generate task
                hidden_states = hidden_states[logit_indices]
                output = self.model.compute_logits(hidden_states, None)

        NPUPlatform.synchronize()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        hidden_states_list = list(
            torch.split(hidden_states, num_scheduled_tokens_list))

        req_num_tokens = num_tokens // num_reqs

        model = cast(VllmModelForPooling, self.model)
        dummy_task = self.get_supported_pooling_tasks()[0]
        dummy_pooling_params = PoolingParams(task=dummy_task)

        to_update = model.pooler.get_pooling_updates(dummy_task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=torch.tensor([h.shape[0] for h in hidden_states_list],
                                     device=self.device),
            prompt_token_ids=torch.zeros((num_reqs, req_num_tokens),
                                         dtype=torch.int32,
                                         device=self.device),
            pooling_params=[dummy_pooling_params] * num_reqs)

        try:
            pooler_output = model.pooler(hidden_states=hidden_states_list,
                                         pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "NPU out of memory occurred when warming up pooler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e

        return pooler_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)

            if is_310p():
                from vllm.model_executor.layers.linear import (
                    MergedColumnParallelLinear, QKVParallelLinear,
                    RowParallelLinear)
                for module in self.model.modules():
                    if isinstance(module,
                                  (MergedColumnParallelLinear,
                                   QKVParallelLinear, RowParallelLinear)):
                        module.weight.data = torch_npu.npu_format_cast(
                            module.weight.data, ACL_FORMAT_FRACTAL_NZ)
            if self.drafter:
                logger.info("Loading drafter model...")
                if isinstance(self.drafter, EagleProposer):
                    if self.use_aux_hidden_state_outputs:
                        self.drafter.load_model(self.model)
                        self.model.set_aux_hidden_state_layers(
                            self.model.get_eagle3_aux_hidden_state_layers())
                else:
                    self.drafter.load_model()
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

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
        ) and not self.torchair_graph_enabled else ACL_FORMAT_FRACTAL_ND
        kv_caches: Dict[str, torch.Tensor] = {}

        def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
            data_ptr = tensor.data_ptr()
            aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
            offset = (aligned_addr - data_ptr) // tensor.element_size()
            return tensor[int(offset):]

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
                alignment = 2 * 1024 * 1024
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
                    dtype = kv_cache_spec.dtype
                    if self.model_config.is_deepseek_mla:

                        num_blocks, block_size, num_kv_heads, head_size = kv_cache_shape
                        rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
                        nope_dim = head_size - rope_dim
                        nope_cache_shape = (num_blocks, block_size,
                                            num_kv_heads, nope_dim)
                        rope_cache_shape = (num_blocks, block_size,
                                            num_kv_heads, rope_dim)
                        if self.vllm_config.kv_transfer_config is None:
                            # For no disaggregate pd scenario, allocate kv cache in normal way
                            rope_cache = torch.zeros(rope_cache_shape,
                                                     dtype=dtype,
                                                     device=self.device)
                            nope_cache = torch.zeros(nope_cache_shape,
                                                     dtype=dtype,
                                                     device=self.device)
                            rope_cache = torch_npu.npu_format_cast(
                                rope_cache, acl_format)
                            nope_cache = torch_npu.npu_format_cast(
                                nope_cache, acl_format)
                        else:

                            # In order to transfer kv cache through the reigster_memory api from llmdatadist, the memory
                            # address should be aligned by 2M. In most case, torch_npu can allocate 2M aligned memory, but
                            # we found there are also some exceptions during test, so we manual align those memory here, this part
                            # of code may consume 2M * 2 * elem_size memory every layer.
                            nope_allocate_shape = num_blocks * block_size * num_kv_heads * nope_dim
                            nope_allocate_shape_alignment = nope_allocate_shape + alignment
                            rope_allocate_shape = num_blocks * block_size * num_kv_heads * rope_dim
                            rope_allocate_shape_alignment = rope_allocate_shape + alignment

                            nope_cache = torch.zeros(
                                nope_allocate_shape_alignment,
                                dtype=dtype,
                                device=self.device)
                            rope_cache = torch.zeros(
                                rope_allocate_shape_alignment,
                                dtype=dtype,
                                device=self.device)
                            nope_cache = align_memory(
                                nope_cache,
                                alignment)[:nope_allocate_shape].view(
                                    nope_cache_shape)
                            rope_cache = align_memory(
                                rope_cache,
                                alignment)[:rope_allocate_shape].view(
                                    rope_cache_shape)
                        kv_caches[layer_name] = (nope_cache, rope_cache)
                    else:
                        num_caches = kv_cache_shape[0]
                        kv_cache_list = []
                        for i in range(num_caches):
                            cache_shape = kv_cache_shape[1:]
                            if self.vllm_config.kv_transfer_config is None:
                                kv_cache = torch.zeros(cache_shape,
                                                       dtype=dtype,
                                                       device=self.device)
                                kv_cache = torch_npu.npu_format_cast(
                                    kv_cache, acl_format)
                            else:
                                cache_size = math.prod(cache_shape)
                                cache_size_aligned = cache_size + alignment
                                kv_cache = torch.zeros(cache_size_aligned,
                                                       dtype=dtype,
                                                       device=self.device)
                                kv_cache = align_memory(
                                    kv_cache,
                                    alignment)[:cache_size].view(cache_shape)
                            kv_cache_list.append(kv_cache)
                        kv_caches[layer_name] = tuple(kv_cache_list)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=self.block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _compile_torchair_graph(self, torchair_graph_batch_sizes) -> None:
        # Trigger torchair graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        for idx, num_tokens in enumerate(reversed(torchair_graph_batch_sizes)):
            for _ in range(self.vllm_config.compilation_config.
                           cudagraph_num_of_warmups):
                self._dummy_run(num_tokens, is_torchair_compile=True)
            self._dummy_run(num_tokens, is_torchair_compile=True)
            logger.info("Batchsize %d is compiled successfully: %d/%d.",
                        num_tokens, idx + 1, len(torchair_graph_batch_sizes))

    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        # TODO(NeverRaR): Calling graph_capture(device=self.device) in
        # torchair graph capture can cause some issues, so now we just
        # temporarily split the codepath for the two different graph patterns.
        if self.torchair_graph_enabled:
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
        elif self.use_aclgraph:
            # Trigger ACL graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            # TODO(zzzzwwjj): Check dummy_run with ACL Graph and full graph mode
            with graph_capture(device=self.device):
                for num_tokens in reversed(self.aclgraph_batch_sizes):
                    for _ in range(self.vllm_config.compilation_config.
                                   cudagraph_num_of_warmups):
                        self._dummy_run(num_tokens)
                    self._dummy_run(num_tokens)
        else:
            logger.info("Skipping NPU graph capture for eager mode.")
            return
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _generate_ngram_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = self.input_batch.req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            assert isinstance(self.drafter, NgramProposer)
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :end_idx])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def _generate_eagle3_token_ids(self,
                                   valid_sampled_token_ids: list[list[int]],
                                   sampling_metadata: SamplingMetadata,
                                   scheduler_output: "SchedulerOutput",
                                   spec_decode_metadata: SpecDecodeMetadata,
                                   positions: torch.Tensor,
                                   num_scheduled_tokens: int,
                                   hidden_states: torch.Tensor,
                                   aux_hidden_states: torch.Tensor = None):
        assert isinstance(self.drafter, EagleProposer)
        attn_metadata = self.get_eagle_atten_dict(scheduler_output)
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])

                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.device)
        eagle_attn_metadata = attn_metadata[self.drafter.attn_layer_name]
        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            if self.use_aux_hidden_state_outputs:
                target_hidden_states = torch.cat(
                    [h[:num_scheduled_tokens] for h in aux_hidden_states],
                    dim=-1)
            else:
                target_hidden_states = hidden_states[:num_scheduled_tokens]
            target_slot_mapping = eagle_attn_metadata.slot_mapping
            cu_num_tokens = eagle_attn_metadata.query_start_loc
        else:
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            num_tokens = num_scheduled_tokens - sum(num_rejected_tokens)
            cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                eagle_attn_metadata.query_start_loc, num_rejected_tokens,
                num_tokens)
            target_token_ids = self.input_ids[token_indices]
            target_positions = positions[token_indices]
            if self.use_aux_hidden_state_outputs:
                target_hidden_states = torch.cat(
                    [h[token_indices] for h in aux_hidden_states], dim=-1)
            else:
                target_hidden_states = hidden_states[token_indices]
            target_slot_mapping = eagle_attn_metadata.slot_mapping[
                token_indices]

        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=eagle_attn_metadata.block_tables,
            sampling_metadata=sampling_metadata,
        )
        spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def _generate_mtp_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: Union[AscendMetadata, AscendMLAMetadata,
                             AscendTorchairMetadata],
    ):
        assert isinstance(self.drafter, MtpProposer)
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(valid_sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(next_token_ids,
                                      dtype=torch.int32,
                                      device=self.device)

        if spec_decode_metadata is None:
            # input_ids can be None for multimodal models.
            target_token_ids = self.input_ids[:num_scheduled_tokens]
            target_positions = positions[:num_scheduled_tokens]
            target_hidden_states = hidden_states[:num_scheduled_tokens]
            target_slot_mapping = attn_metadata.slot_mapping
            cu_num_tokens = attn_metadata.query_start_loc
        else:
            # TODO(woosuk): Refactor this.
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            num_rejected_tokens = [
                n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                for i, n in enumerate(num_draft_tokens)
            ]
            num_rejected_tokens = torch.tensor(
                num_rejected_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                attn_metadata.query_start_loc,
                num_rejected_tokens,
            )
            target_token_ids = self.input_ids[token_indices]
            target_positions = positions[token_indices]
            target_hidden_states = hidden_states[token_indices]
            target_slot_mapping = attn_metadata.slot_mapping[token_indices]

        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            target_slot_mapping=target_slot_mapping,
            next_token_ids=next_token_ids,
            cu_num_tokens=cu_num_tokens,
            block_table=attn_metadata.block_tables,
            sampling_metadata=sampling_metadata,
        )
        spec_token_ids = draft_token_ids.tolist()
        return spec_token_ids

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer NPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking NPU->CPU transfers.
        if prompt_logprobs_dict:
            torch.npu.synchronize()

        return prompt_logprobs_dict

    def init_torchair_graph_batch_sizes(self):
        start_graph_batch_size = 4
        tp_size = get_tensor_model_parallel_world_size()

        # NOTE: When use all2all | mc2, We need to slice the `num_tokens` dimension into `tp_size` blocks
        start_graph_batch_size = max(start_graph_batch_size, tp_size)

        while (start_graph_batch_size <= self.max_num_reqs):
            self.torchair_graph_batch_sizes.append(start_graph_batch_size)
            start_graph_batch_size *= 2

    def select_torchair_padded_batch_size(self, batch_size: int):
        selected_batch_size = self.max_num_reqs
        for padded_batch_size in self.torchair_graph_batch_sizes:
            if batch_size <= padded_batch_size < selected_batch_size:
                selected_batch_size = padded_batch_size
        return selected_batch_size

    def get_supported_pooling_tasks(self):
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())
