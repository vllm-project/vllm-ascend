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
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo.cache_size
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm  # type: ignore
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CompilationLevel, CUDAGraphMode, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (
    get_dp_group, get_pp_group, get_tp_group, is_global_first_rank)
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LazyLoader, cdiv, is_pin_memory_available)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, LogprobsTensors, ModelRunnerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (bind_kv_cache, gather_mm_placeholders,
                                  sanity_check_mm_encoder_outputs,
                                  scatter_mm_placeholders)

from vllm_ascend.utils import context_parallel_enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         AscendCommonLongSequenceMetadata)
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.multistream.ms_split import compute_split_seq_index
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.sample.logits_processor import build_logitsprocs
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.interface import SpecDcodeType
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.torchair.torchair_attention import AscendTorchairMetadata
from vllm_ascend.torchair.torchair_mla import AscendMLATorchairMetadata
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ,
                               AscendSocVersion, ProfileExecuteDuration,
                               get_ascend_soc_version, is_310p,
                               lmhead_tp_enable)
from vllm_ascend.worker.npu_input_batch import CachedRequestState, InputBatch
from vllm_ascend.distributed.parallel_state import is_sp_enabled

if context_parallel_enable:
    from vllm.distributed import get_cp_group
    from vllm.distributed.parallel_state import (
    get_context_model_parallel_rank, get_context_model_parallel_world_size)

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

import vllm_ascend.envs as envs_ascend

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ
else:
    ACL_FORMAT = ACL_FORMAT_FRACTAL_ND


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


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncNPUModelRunnerOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.npu.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self._async_copy_ready_event = torch.npu.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.npu.current_stream()
        with torch.npu.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._sampled_token_ids_cpu = self._sampled_token_ids.to(
                'cpu', non_blocking=True)
            self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        self._async_copy_ready_event.synchronize()

        # Release the device tensor once the copy has completed
        del self._sampled_token_ids

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


class NPUModelRunner(LoRAModelRunnerMixin):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.load_config = vllm_config.load_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.pin_memory = is_pin_memory_available()
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        decode_max_num_seqs = getattr(self.scheduler_config,
                                      'decode_max_num_seqs', 0)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                decode_max_num_seqs)
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.cp_size = get_context_model_parallel_world_size(
        ) if context_parallel_enable else 1
        self.cp_rank = get_context_model_parallel_rank(
        ) if self.cp_size > 1 else 0
        self.enable_sp = is_sp_enabled()
        self.sp_size = get_tensor_model_parallel_world_size(
        ) if self.enable_sp else 1
        self.sp_rank = get_tensor_model_parallel_rank(
        ) if self.enable_sp else 0
        self.device = device
        if envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP:
            self.prefetch_stream = torch.npu.Stream(device=device)
        else:
            self.prefetch_stream = None
        self.dtype = self.model_config.dtype
        if envs_ascend.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:
            # TODO: drop the env config to use ascend sampler by default
            from vllm_ascend.sample.sampler import AscendSampler

            self.sampler = AscendSampler()
        else:
            from vllm.v1.sample.sampler import Sampler

            self.sampler = Sampler()

        # Lazy initialization, these will be set after __init__
        self.kv_caches: List[torch.Tensor] = []
        self.encoder_cache: Dict[str, torch.Tensor] = {}
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
            vllm_config, device)
        if self.cp_size > 1:
            self.attn_mask_builder = None
        else:
            self.attn_mask_builder = AttentionMaskBuilder(
                self.model_config.max_model_len, self.dtype)

        # Set up speculative decoding.
        self.spec_attn_mask = None
        self.drafter: Optional[Union[NgramProposer, EagleProposer,
                                     MtpProposer]] = None
        self.actual_seq_lengths_q = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            self.actual_seq_lengths_q = [
                len for len in
                range(self.decode_token_per_req, self.max_num_tokens +
                      1, self.decode_token_per_req)
            ]
            self.spec_attn_mask = torch.triu(torch.ones(2048,
                                                        2048,
                                                        dtype=torch.bool),
                                             diagonal=1).to(self.device)
            if get_pp_group().is_last_rank:
                self.drafter = get_spec_decode_method(
                    self.speculative_config.method, self.vllm_config,
                    self.device, self)
                self.rejection_sampler = AscendRejectionSampler()

        # Persistent batch.
        self.input_ids = torch.zeros(self.max_num_tokens * self.sp_size,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens * self.sp_size,
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
        self.arange_np: npt.NDArray[np.int32] = np.arange(
            max(self.max_num_reqs + 1, self.model_config.max_model_len,
                self.max_num_tokens) * self.cp_size * self.sp_size +
            self.cp_size * self.sp_size * self.max_num_reqs,
            dtype=np.int32)
        self.position_cp = np.zeros(self.max_num_tokens * self.sp_size,
                                    dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens * self.sp_size,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_cpu = torch.zeros(self.max_num_tokens * self.sp_size,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(
            self.max_num_tokens * self.cp_size * self.sp_size +
            self.cp_size * self.sp_size * self.max_num_reqs,
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

        self.use_aclgraph = self._use_aclgraph()
        self.aclgraph_batch_sizes = list(
            reversed(self.compilation_config.cudagraph_capture_sizes))

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens
        # aclgraph dispatcher for runtime aclgraph dispatching.
        self.aclgraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]],
                                              torch.Tensor]] = None

        # NOTE: we need to use `in_profile_run` to determine whether `enable_force_load_balance` is True
        self.in_profile_run = False

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        self.mc2_tokens_capacity = 512 * self.parallel_config.tensor_parallel_size
        self.reserved_mc2_mask = torch.zeros(
            self.mc2_tokens_capacity,
            dtype=torch.bool,
            device=self.device,
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.async_output_copy_stream = torch.npu.Stream() if \
            self.use_async_scheduling else None

    def _use_aclgraph(self) -> bool:
        return self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.compilation_config.level == CompilationLevel.PIECEWISE and not self.model_config.enforce_eager

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)
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
            self.input_batch.remove_request(req_id)

        req_ids_to_add: list[str] = []
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
                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_kwargs=new_req_data.mm_kwargs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                mm_hashes=new_req_data.mm_hashes,
                num_computed_tokens_of_cp_sp=new_req_data.
                num_computed_tokens_of_cp_sp,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_item in self.requests[req_id].mm_kwargs:
                    mm_input = mm_item.get_data()
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.append(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.append(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.append(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.append(
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
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]

            req_state.kv_rank = req_data.kv_rank[i]
            req_state.num_computed_tokens_of_cp_sp = req_data.num_computed_tokens_of_cp_sp[
                i]

            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
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
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
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

            # cp param
            self.input_batch.kv_rank[req_index] = req_state.kv_rank
            self.input_batch.num_computed_tokens_of_cp_sp[
                req_index] = req_state.num_computed_tokens_of_cp_sp

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()

        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _sync_metadata_across_dp(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill, enable_dbo

        # Sync num_tokens, with_prefill, enable_dbo across dp ranks
        num_tokens_tensor = torch.tensor([
            num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
        ],
                                         dtype=torch.int32,
                                         device="npu")

        flags_tensor = torch.tensor(
            [int(with_prefill), int(not enable_dbo)],
            dtype=torch.int32,
            device="npu")

        packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])

        dist.all_reduce(packed_tensor, group=get_dp_group().device_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:-2]
        synced_flags = packed_tensor[-2:]

        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])
        global_enable_dbo = not bool(synced_flags[1])

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                                self.dp_size,
                                                device="npu",
                                                dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill, global_enable_dbo

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

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
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

    def _make_attention_mask(self, seq_lens, position,
                             attn_state) -> torch.Tensor:
        # Check if attn_mask_builder is None before using it
        if self.attn_mask_builder is None:
            raise ValueError("attn_mask_builder is not initialized (got None)")
        # Chunk Prefill situation.
        if attn_state == AscendAttentionState.ChunkedPrefill and not self.vllm_config.model_config.use_mla:
            return self.attn_mask_builder.get_splitfuse_attn_mask(
                seq_lens, position, self.dtype, self.device)
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
        mm_kwargs = list[MultiModalKwargsItem]()
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_hash = req_state.mm_hashes[mm_input_id]
                mm_kwargs.append(req_state.mm_kwargs[mm_input_id])
                mm_hashes_pos.append(
                    (mm_hash, req_state.mm_positions[mm_input_id]))
        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=True,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
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
            mm_hashes = req_state.mm_hashes
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
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                mm_hash = mm_hashes[i]
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                    f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def _num_scheduled_tokens_prefill_cp(self, num_tokens,
                                         num_comnputed_tokens,
                                         cp_kv_recover_idx):
        num_scheduled_tokens = num_tokens - num_comnputed_tokens
        num_cp_padded_scheduled_tokens = cdiv(
            num_scheduled_tokens, 2 * self.cp_size) * (2 * self.cp_size
                                                       )  # pad to 2*cp_size
        cp_pad = num_cp_padded_scheduled_tokens - num_scheduled_tokens  # 给sample用
        full_indices = list(
            range(self.max_num_tokens * self.cp_size * self.sp_size +
                  self.cp_size * self.sp_size * self.max_num_reqs))
        chunk_size = num_cp_padded_scheduled_tokens // (2 * self.cp_size)

        # split position_ids (and use split position_ids to split input_ids afterwards)
        req_position_cp = []
        req_position_cp.extend(
            full_indices[self.cp_rank * chunk_size:(self.cp_rank + 1) *
                         chunk_size])
        req_position_cp.extend(
            full_indices[num_cp_padded_scheduled_tokens - (self.cp_rank + 1) *
                         chunk_size:num_cp_padded_scheduled_tokens -
                         self.cp_rank * chunk_size])

        # used to recover kv order in cp prefill (after all-gather kv and before storing kv_cache)
        num_added_recover_tokens = len(cp_kv_recover_idx[0]) * self.cp_size
        for rank in range(self.cp_size):
            cp_kv_recover_idx[rank].extend(
                full_indices[rank * chunk_size +
                             num_added_recover_tokens:(rank + 1) * chunk_size +
                             num_added_recover_tokens])
            cp_kv_recover_idx[rank].extend(full_indices[
                num_cp_padded_scheduled_tokens - (rank + 1) * chunk_size +
                num_added_recover_tokens:num_cp_padded_scheduled_tokens -
                rank * chunk_size + num_added_recover_tokens])

        return req_position_cp, num_cp_padded_scheduled_tokens, cp_pad

    def _slot_mapping_prefill_cp(
        self,
        num_scheduled_tokens_for_slot,
    ):
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        num_computed_and_new_tokens_batch = np.array(
            self.input_batch.num_computed_tokens_of_cp_sp[:self.input_batch.
                                                          num_reqs])
        start_index = 0
        for i in range(self.input_batch.num_reqs):
            block_table_req = block_table_cpu[i]
            block_table_indices = np.repeat(block_table_req, self.block_size)
            num_save_tokens_rank = num_computed_and_new_tokens_batch[i][
                self.cp_rank][self.sp_rank]

            positions_for_slot = self.arange_np[:num_save_tokens_rank]
            block_offsets = positions_for_slot % self.block_size
            slot_mapping = (block_table_indices * self.block_size
                            )[:num_save_tokens_rank] + block_offsets

            num_cp_padded_scheduled_tokens = num_scheduled_tokens_for_slot[i]
            sum_tokens_before_cp: Union[np.ndarray, int] = np.sum(
                num_computed_and_new_tokens_batch[i][:self.cp_rank])
            sum_tokens_before_sp_in_cp: Union[np.ndarray, int] = np.sum(
                num_computed_and_new_tokens_batch[i][
                    self.cp_rank][:self.sp_rank])
            kv_save_start = sum_tokens_before_cp + sum_tokens_before_sp_in_cp

            self.slot_mapping_np[start_index + kv_save_start:start_index +
                                 kv_save_start +
                                 num_save_tokens_rank] = slot_mapping

            start_index += num_cp_padded_scheduled_tokens

    def _slot_mapping_decode_cp(
        self,
        num_scheduled_tokens,
    ):
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        num_computed_and_new_tokens_batch = self.input_batch.num_computed_tokens_of_cp_sp
        start_index = 0
        for i in range(self.input_batch.num_reqs):
            num_scheduled_tokens_req = num_scheduled_tokens[i]
            if self.input_batch.kv_rank[i] == (self.cp_rank, self.sp_rank):
                block_table_req = block_table_cpu[i]
                block_table_indices = np.repeat(block_table_req,
                                                self.block_size)
                if i < 0 or i >= len(num_computed_and_new_tokens_batch):
                    raise IndexError(
                        f"i={i} is out of bounds for num_computed_and_new_tokens_batch"
                    )

                current_batch_item = num_computed_and_new_tokens_batch[i]
                if current_batch_item is None:
                    raise ValueError(
                        f"num_computed_and_new_tokens_batch[{i}] is None, cannot index"
                    )
                if self.cp_rank < 0 or self.cp_rank >= len(current_batch_item):
                    raise IndexError(
                        f"cp_rank={self.cp_rank} is out of bounds for batch item {i} (length {len(current_batch_item)})"
                    )
                current_cp_group = current_batch_item[self.cp_rank]
                if self.sp_rank < 0 or self.sp_rank >= len(current_cp_group):
                    raise IndexError(
                        f"sp_rank={self.sp_rank} is out of bounds for cp_rank {self.cp_rank} (length {len(current_cp_group)})"
                    )
                num_save_tokens_rank: int = current_cp_group[self.sp_rank]
                # num_save_tokens_rank = num_computed_and_new_tokens_batch[i][self.cp_rank][self.sp_rank]

                positions_for_slot = self.arange_np[:num_save_tokens_rank]
                block_offsets = positions_for_slot % self.block_size
                slot_mapping = (block_table_indices * self.block_size
                                )[:num_save_tokens_rank] + block_offsets

                self.slot_mapping_np[start_index:start_index +
                                     num_scheduled_tokens_req] = slot_mapping[
                                         -num_scheduled_tokens_req:]
            start_index += num_scheduled_tokens_req

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

    def _prepare_input_ids(self, total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        NPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens],
                non_blocking=True)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the NPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                flattened_indices.append(flattened_index)
                indices_match &= (prev_index == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(flattened_indices)
        if num_commmon_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the NPU first.
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens],
                non_blocking=True)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens,
                                                        0],
                non_blocking=True)
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        self.input_ids.scatter_(dim=0,
                                index=input_ids_index_tensor,
                                src=self.input_batch.prev_sampled_token_ids[
                                    prev_common_req_indices_tensor, 0])

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[Union[AscendMetadata, AscendMLAMetadata, AscendTorchairMetadata,
                     AscendMLATorchairMetadata], torch.Tensor, np.ndarray, int,
               torch.Tensor, int, torch.Tensor, SpecDecodeMetadata,
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        is_prefill = list(
            scheduler_output.num_scheduled_tokens.values())[0] > 1

        self.attn_metadata_builder.reorder_batch(self.input_batch,
                                                 scheduler_output)
        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        num_valid_tokens = np.empty(num_reqs, dtype=np.int32)
        num_cp_pads = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        start_index = 0
        num_scheduled_tokens_for_slot = np.empty(num_reqs, dtype=np.int32)
        self.cp_kv_recover_idx: List[List[int]] = [[]
                                                   for _ in range(self.cp_size)
                                                   ]
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if self.cp_size > 1 and num_tokens > 1:
                # when cp > 1 & prefill, need to pad & split sequence here
                req_position_cp, num_cp_padded_scheduled_tokens, num_cp_pads[
                    i] = self._num_scheduled_tokens_prefill_cp(
                        num_tokens,
                        self.input_batch.num_computed_tokens_cpu[i],
                        self.cp_kv_recover_idx)
                num_tokens = len(req_position_cp)
                self.position_cp[start_index:start_index +
                                 num_tokens] = req_position_cp
                start_index += num_tokens
                num_scheduled_tokens_for_slot[
                    i] = num_cp_padded_scheduled_tokens
            else:
                num_scheduled_tokens_for_slot[i] = num_tokens
            num_scheduled_tokens[i] = num_tokens
            num_valid_tokens[i] = num_tokens - \
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        # update total_num_scheduled_tokens
        total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        enable_dbo = self._check_dbo_is_valid(self.query_lens.tolist(),
                                              attn_state,
                                              total_num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp, with_prefill,
         enable_dbo) = self._sync_metadata_across_dp(num_input_tokens,
                                                     with_prefill, enable_dbo)

        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        if self.cp_size > 1 and is_prefill:
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                   self.position_cp[:total_num_scheduled_tokens],
                   out=positions_np)
        else:
            np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
                   arange,
                   out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Prepare some information for building Attention-Metadata
        # Compute and commit slot mapping
        if not self.cp_size * self.sp_size > 1:
            self.input_batch.block_table.compute_slot_mapping(
                req_indices, positions_np)
            self.input_batch.block_table.commit_slot_mapping(
                total_num_scheduled_tokens)
            self.slot_mapping_cpu[:total_num_scheduled_tokens].copy_(
                self.input_batch.block_table[0].
                slot_mapping_cpu[:total_num_scheduled_tokens])

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.query_start_loc[num_reqs + 1:].fill_(-1)
        self.seq_lens[num_reqs:].fill_(0)

        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        self.positions_cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions[:num_input_tokens].copy_(
            self.positions_cpu[:num_input_tokens], non_blocking=True)

        # Make Attention metadata
        positions_cpu = self.positions_cpu[:num_input_tokens]
        positions = self.positions[:num_input_tokens]
        seq_lens_cpu = self.seq_lens_cpu[:num_reqs]
        seq_lens = self.seq_lens_cpu[:num_reqs]

        if self.cp_size * self.sp_size > 1:
            if is_prefill:
                self.slot_mapping_np.fill(-1)
                self._slot_mapping_prefill_cp(num_scheduled_tokens_for_slot)
            else:
                self.slot_mapping_np.fill(-1)
                self._slot_mapping_decode_cp(num_scheduled_tokens)
        else:
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions_np // self.block_size)

            block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % self.block_size
            np.add(block_numbers * self.block_size,
                   block_offsets,
                   out=self.slot_mapping_np[:total_num_scheduled_tokens])

        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        if self.cp_size > 1:
            self.attn_mask = None
        else:
            self.attn_mask = self._make_attention_mask(seq_lens=seq_lens_cpu,
                                                       position=positions_cpu,
                                                       attn_state=attn_state)
        self.attn_state = attn_state  # type: ignore

        num_actual_tokens_cp_full = total_num_scheduled_tokens * (
            self.cp_size if is_prefill > 0 else 1)
        long_seq_metadata = None
        if self.cp_size > 1 and is_prefill > 0:
            cp_kv_recover_idx = torch.zeros(num_actual_tokens_cp_full,
                                            dtype=torch.int32,
                                            device=self.device)
            cp_kv_recover_idx.copy_(torch.tensor(
                np.array(self.cp_kv_recover_idx).flatten().tolist()),
                                    non_blocking=True)
            self.cp_kv_recover_idx = cp_kv_recover_idx.to(
                torch.float32).argsort().to(torch.int32)

            q_head_idx, q_tail_idx = [], []
            kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
            kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
            chunk_seqlens = []
            kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
            q_req_offset = 0
            kv_req_offset = 0
            q_head_chunk_id = self.cp_rank
            q_tail_chunk_id = self.cp_size * 2 - 1 - self.cp_rank
            for seq_len in seq_lens:
                chunk_len = seq_len // 2
                chunk_seqlens.append(chunk_len)
                q_head_idx.extend(
                    list(range(q_req_offset, q_req_offset + chunk_len)))
                kv_with_q_head_nomask_idx.extend(
                    list(
                        range(kv_req_offset,
                              kv_req_offset + chunk_len * q_head_chunk_id)))
                kv_with_q_head_mask_idx.extend(
                    list(
                        range(
                            kv_req_offset + chunk_len * q_head_chunk_id,
                            kv_req_offset + chunk_len *
                            (q_head_chunk_id + 1))))
                kv_with_q_head_nomask_seqlens.append(chunk_len *
                                                     q_head_chunk_id)

                q_tail_idx.extend(
                    list(
                        range(q_req_offset + chunk_len,
                              q_req_offset + chunk_len * 2)))
                kv_with_q_tail_nomask_idx.extend(
                    list(
                        range(kv_req_offset,
                              kv_req_offset + chunk_len * q_tail_chunk_id)))
                kv_with_q_tail_mask_idx.extend(
                    list(
                        range(
                            kv_req_offset + chunk_len * q_tail_chunk_id,
                            kv_req_offset + chunk_len *
                            (q_tail_chunk_id + 1))))
                kv_with_q_tail_nomask_seqlens.append(chunk_len *
                                                     q_tail_chunk_id)

                q_req_offset += seq_len
                kv_req_offset += seq_len * self.cp_size

            # Convert lists to tensors and move to device
            def _list_to_tensor(lst, device, dtype=torch.int32):
                tensor_npu = torch.zeros(len(lst), dtype=dtype, device=device)
                tensor_npu.copy_(torch.tensor(lst, dtype=dtype),
                                 non_blocking=True)
                return tensor_npu

            q_head_idx_tensor = _list_to_tensor(q_head_idx, self.device)
            q_tail_idx_tensor = _list_to_tensor(q_tail_idx, self.device)
            self.q_head_idx_tensor = q_head_idx_tensor
            self.q_tail_idx_tensor = q_tail_idx_tensor

            q_full_idx = torch.cat([q_head_idx_tensor, q_tail_idx_tensor])
            q_full_idx = q_full_idx.to(torch.float32).argsort().to(torch.int32)
            self.q_full_idx = q_full_idx

            self.kv_idx_names = {
                'kv_with_q_head_nomask_idx_tensor': kv_with_q_head_nomask_idx,
                'kv_with_q_head_mask_idx_tensor': kv_with_q_head_mask_idx,
                'kv_with_q_tail_nomask_idx_tensor': kv_with_q_tail_nomask_idx,
                'kv_with_q_tail_mask_idx_tensor': kv_with_q_tail_mask_idx
            }
            for key, value in self.kv_idx_names.items():
                tensor_npu = _list_to_tensor(value, self.device)
                self.kv_idx_names[key] = tensor_npu

            attn_mask_seqlens = torch.tensor([chunk_seqlens, chunk_seqlens],
                                             dtype=torch.int32)
            head_attn_nomask_seqlens = torch.tensor(
                [chunk_seqlens, kv_with_q_head_nomask_seqlens],
                dtype=torch.int32)
            tail_attn_nomask_seqlens = torch.tensor(
                [chunk_seqlens, kv_with_q_tail_nomask_seqlens],
                dtype=torch.int32)
            if self.vllm_config.model_config.use_mla:
                cp_prefill_mask = torch.triu(
                    torch.ones(512, 512, device=self.device, dtype=self.dtype),
                    1)
            else:
                max_seq_len = max(seq_lens, default=0)
                cp_prefill_mask = torch.triu(
                    torch.full((seq_lens.shape[0], max_seq_len, max_seq_len),
                               True,
                               device=self.device,
                               dtype=torch.bool), 1)

            self.extra_long_seq_kwargs = {
                'attn_mask_seqlens': attn_mask_seqlens,
                'head_attn_nomask_seqlens': head_attn_nomask_seqlens,
                'tail_attn_nomask_seqlens': tail_attn_nomask_seqlens,
                'cp_prefill_mask': cp_prefill_mask
            }
            long_seq_metadata = AscendCommonLongSequenceMetadata(
                cp_kv_recover_idx=self.cp_kv_recover_idx,
                num_actual_tokens_cp_full=num_actual_tokens_cp_full,
                num_computed_tokens_of_cp_sp=self.input_batch.
                num_computed_tokens_of_cp_sp[:self.input_batch.num_reqs],
                q_head_idx_tensor=self.q_head_idx_tensor,
                q_tail_idx_tensor=self.q_tail_idx_tensor,
                q_full_idx=self.q_full_idx,
                kv_with_q_head_nomask_idx_tensor=self.
                kv_idx_names['kv_with_q_head_nomask_idx_tensor'],
                kv_with_q_head_mask_idx_tensor=self.
                kv_idx_names['kv_with_q_head_mask_idx_tensor'],
                kv_with_q_tail_nomask_idx_tensor=self.
                kv_idx_names['kv_with_q_tail_nomask_idx_tensor'],
                kv_with_q_tail_mask_idx_tensor=self.
                kv_idx_names['kv_with_q_tail_mask_idx_tensor'],
                attn_mask_seqlens=self.
                extra_long_seq_kwargs['attn_mask_seqlens'],
                head_attn_nomask_seqlens=self.
                extra_long_seq_kwargs['head_attn_nomask_seqlens'],
                tail_attn_nomask_seqlens=self.
                extra_long_seq_kwargs['tail_attn_nomask_seqlens'],
                cp_prefill_mask=self.extra_long_seq_kwargs['cp_prefill_mask'])
        else:
            long_seq_metadata = AscendCommonLongSequenceMetadata(
                num_actual_tokens_cp_full=num_actual_tokens_cp_full,
                num_computed_tokens_of_cp_sp=self.input_batch.
                num_computed_tokens_of_cp_sp[:self.input_batch.num_reqs])

        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        self.with_prefill = with_prefill
        self.num_tokens_across_dp = num_tokens_across_dp
        self._update_graph_pad_size(with_prefill, maybe_padded_num_tokens)

        # Make AscendCommonAttentionMetadata
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[:num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[:num_reqs + 1],
            seq_lens_cpu=self.seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            block_table_tensor=self.input_batch.block_table[0].
            get_device_tensor(),
            slot_mapping_cpu=self.slot_mapping_cpu,
            positions=self.positions,
            attn_mask=self.attn_mask,
            spec_attn_mask=self.spec_attn_mask,
            attn_state=self.attn_state,
            enable_dbo_across_dp=enable_dbo,
            is_only_prefill=bool(np.all(num_valid_tokens != 1)),
            max_query_len=max_num_scheduled_tokens,
            graph_pad_size=self.graph_pad_size,
            decode_token_per_req=self.decode_token_per_req,
            common_long_seq_metadata=long_seq_metadata)
        attn_metadata = self.attn_metadata_builder.build(
            common_attn_metadata, self.model)
        if self.vllm_config.model_config.use_mla:
            attn_metadata.num_input_tokens = num_input_tokens
        # token id pad
        if self.cp_size > 1:
            for i in range(num_reqs):
                if num_scheduled_tokens[i] > 1:
                    num_padded_tokens = num_scheduled_tokens_for_slot[
                        i] + self.input_batch.num_computed_tokens_cpu[i]
                    self.input_batch.token_ids_cpu[
                        i][num_padded_tokens -
                           num_cp_pads[i]:num_padded_tokens] = 0

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        # Copy the tensors to the NPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

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
        positions = self.positions[:num_input_tokens]
        input_ids, positions = self._update_input_ids_and_positions(
            input_ids, positions, num_input_tokens, with_prefill,
            maybe_padded_num_tokens)

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

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            if self.cp_size > 1 and is_prefill:
                # logits_indices = cu_num_tokens - num_cp_pads[:num_reqs] - 1 # if without all-gather and only sample on cp0
                logits_indices = cu_num_tokens * self.cp_size - num_cp_pads[:
                                                                            num_reqs] - 1
            else:
                logits_indices = cu_num_tokens - 1
            logits_indices = torch.from_numpy(logits_indices).to(
                self.device, non_blocking=True)
        else:
            # cp sp not supported now
            assert self.cp_size == 1 and self.sp_size == 1
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

        if lmhead_tp_enable():
            max_num_reqs_across_dp = maybe_padded_num_tokens if not with_prefill else self.max_num_reqs
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors)

    def _generate_process_reqs_hidden_states(self, attn_metadata, with_prefill,
                                             maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds):
        assert self.model is not None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if get_forward_context().flashcomm_v1_enabled:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            pad_size = get_forward_context().pad_size
            if pad_size > 0:
                hidden_states = hidden_states[:-pad_size, :]
        if self.enable_sp and with_prefill:
            hidden_states = get_tp_group().all_gather(hidden_states, 0)
            hidden_states = hidden_states[:attn_metadata.num_input_tokens]
        if self.cp_size > 1 and with_prefill:
            hidden_states = get_cp_group().all_gather(hidden_states, 0)
            hidden_states = torch.index_select(
                hidden_states, 0, attn_metadata.prefill.cp_kv_recover_idx)
        return hidden_states

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        ascend_config = get_ascend_config()
        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'deepseek_mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.drafter and (self.drafter.name == SpecDcodeType.EAGLE
                                 or self.drafter.name == SpecDcodeType.EAGLE3):
                attn_state = AscendAttentionState.ChunkedPrefill
            else:
                attn_state = AscendAttentionState.SpecDecoding
        # splitfuse
        elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _update_graph_pad_size(self, with_prefill, graph_pad_size):
        self.graph_pad_size = -1

    def _update_input_ids_and_positions(self, input_ids, positions,
                                        num_input_tokens, with_prefill,
                                        maybe_padded_num_tokens):
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        return input_ids, positions

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

    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: Union[AscendMetadata, AscendMLAMetadata,
                             AscendTorchairMetadata,
                             AscendMLATorchairMetadata],
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            draft_token_ids = self.drafter.generate_token_ids(
                valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, positions, num_scheduled_tokens,
                hidden_states, attn_metadata, aux_hidden_states)
        return draft_token_ids

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

        hidden_states = hidden_states[:num_scheduled_tokens]
        pooling_metadata = self.input_batch.pooling_metadata
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_np.tolist(),
                                              device=hidden_states.device)
        seq_lens_cpu = self.seq_lens_cpu[:self.input_batch.num_reqs]

        # Pooling models D2H & synchronize occurs in pooler.py:build_output
        raw_pooler_output = self.model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data)
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

    def _select_moe_comm_method(self, num_tokens: int) -> str:
        """1. If expert parallel is not enabled, we use all-gather since MC2 and all-to-all
        are designed for expert parallelism.
        2. If expert parallel is enabled, we need to consider the soc version and the
        number of tokens. This is based on the observation that all-gather is more
        efficient than all-to-all when running on A2.

            a. For A2, we choose from MC2 and all-gather.

            b. For A3, we choose from MC2 and all-to-all.

            In both cases, we use MC2 when the number of tokens is smaller than
            a its capacity threshold.

        Args:
            num_tokens (int): The number of tokens in the current batch.

        Raises:
            ValueError: If the soc version is unsupported.

        Returns:
            str: The selected MoE communication method, either "allgather", "mc2", or "alltoall".
        """
        soc_version = get_ascend_soc_version()

        if not self.parallel_config.enable_expert_parallel:
            moe_comm_method = "allgather"
        elif soc_version in {AscendSocVersion.A2}:
            if num_tokens <= self.mc2_tokens_capacity and self.parallel_config.world_size_across_dp >= 16:
                moe_comm_method = "mc2"
            else:
                moe_comm_method = "allgather"
        elif soc_version in {AscendSocVersion.A3}:
            moe_comm_method = "mc2" if num_tokens <= self.mc2_tokens_capacity else "alltoall"
        else:
            raise ValueError(f"Unsupported soc_version: {soc_version}")

        if is_global_first_rank():
            logger.debug(f"num_tokens: {num_tokens}, "
                         f"moe_comm_method: {moe_comm_method}")

        return moe_comm_method

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)
            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors) = (self._prepare_inputs(
                 scheduler_output, intermediate_tensors))

        moe_comm_method = self._select_moe_comm_method(num_input_tokens)

        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=False)
        aclgraph_runtime_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(batch_descriptor)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=self.with_prefill,
                    reserved_mc2_mask=self.reserved_mc2_mask,
                    moe_comm_method=moe_comm_method,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model):
                self.maybe_setup_kv_connector(scheduler_output)

                hidden_states = self._generate_process_reqs_hidden_states(
                    attn_metadata, self.with_prefill, maybe_padded_num_tokens,
                    input_ids, positions, intermediate_tensors, inputs_embeds)

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
                hidden_states, aux_hidden_states = hidden_states

        kv_connector_output = None
        if finished_sending is not None or finished_recving is not None:
            kv_connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving)
        else:
            kv_connector_output = None
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
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    return self._pool(
                        hidden_states,
                        scheduler_output.total_num_scheduled_tokens,
                        num_scheduled_tokens_np, finished_sending,
                        finished_recving, kv_connector_output)
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
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:self.input_batch.num_reqs]
                sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
            else:
                if lmhead_tp_enable() and logits is not None:
                    logits = logits[:len(spec_decode_metadata.logits_indices)]
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

            # Copy some objects so they don't get modified after returning.
            # This is important when using async scheduling.
            req_ids_output_copy = self.input_batch.req_ids.copy()
            req_id_to_index_output_copy = \
                self.input_batch.req_id_to_index.copy()

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.
            logprobs_tensors = sampler_output.logprobs_tensors
            logprobs_lists = logprobs_tensors.tolists() \
                if logprobs_tensors is not None else None

            # Compute prompt logprobs if needed.
            prompt_logprobs_dict = self._get_prompt_logprobs_dict(
                hidden_states[:scheduler_output.total_num_scheduled_tokens],
                scheduler_output,
            )

            num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
            sampled_token_ids = sampler_output.sampled_token_ids
            if not self.use_async_scheduling:
                # Get the valid generated tokens.
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
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[i].clear()
            else:
                valid_sampled_token_ids = []
                invalid_req_indices = list(discard_sampled_tokens_req_indices)
                invalid_req_indices_set = set(invalid_req_indices)
                assert sampled_token_ids.shape[-1] == 1

                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = \
                    sampled_token_ids
                self.input_batch.prev_sampled_token_ids_invalid_indices = \
                    invalid_req_indices_set
                self.input_batch.prev_req_id_to_index = {
                    req_id: i
                    for i, req_id in enumerate(self.input_batch.req_ids)
                    if i not in invalid_req_indices_set
                }
            # Cache the sampled tokens in the model runner, so that the scheduler
            # doesn't need to send them back.
            # NOTE(woosuk): As an exception, when using PP, the scheduler sends
            # the sampled tokens back, because there's no direct communication
            # between the first-stage worker and the last-stage worker.
            for req_idx in range(num_sampled_tokens):
                if self.use_async_scheduling:
                    sampled_ids = [-1] * 1 if \
                        req_idx not in invalid_req_indices_set else None
                else:
                    sampled_ids = valid_sampled_token_ids[req_idx]
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

            if self.speculative_config:
                self._draft_token_ids = self.propose_draft_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    scheduler_output.total_num_scheduled_tokens,
                    hidden_states,
                    attn_metadata,
                    aux_hidden_states,
                )

            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

        extra_args = ({"kv_connector_output": kv_connector_output})

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
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

        if not self.use_async_scheduling:
            return model_runner_output

        return AsyncNPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

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
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving)
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

    def _build_attention_metadata(self, with_prefill, num_reqs, skip_attn):
        if skip_attn:
            attn_metadata = None
        else:
            # TODO(zzzzwwjj): when aclgraph and full graph mode, we need build attn_metadata
            attn_metadata = None
        return attn_metadata

    def _generate_dummy_run_hidden_states(self, with_prefill,
                                          is_torchair_compile, input_ids,
                                          positions, attn_metadata, num_tokens,
                                          intermediate_tensors, inputs_embeds):
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds)
        if self.drafter and self.drafter.name == SpecDcodeType.EAGLE3:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        is_torchair_compile: bool = False,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        force_attention: bool = False,
        uniform_decode: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE
        }
        if force_attention:
            raise RuntimeError(
                "Capturing attention in aclgraph is unexpected, because full graph is not supported now"
            )

        # Padding for DP
        (num_tokens, num_tokens_across_dp, with_prefill,
         _) = self._sync_metadata_across_dp(num_tokens, with_prefill, False)

        moe_comm_method = self._select_moe_comm_method(num_tokens)

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.seperate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                        num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
            assert num_reqs <= max_num_reqs, \
                "Do not capture num_reqs > max_num_reqs for uniform batch"
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            if with_prefill:
                num_reqs = num_tokens
            else:
                num_reqs = (num_tokens + self.decode_token_per_req -
                            1) // self.decode_token_per_req
            num_reqs = min(num_reqs, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer:
            with_prefill = True

        attn_metadata = self._build_attention_metadata(with_prefill,
                                                       num_reqs,
                                                       skip_attn=True)

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
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
            if aclgraph_runtime_mode == CUDAGraphMode.NONE:
                batch_descriptor = None
            else:
                # filter out the valid batch descriptor
                _cg_mode, batch_descriptor = \
                    self.aclgraph_dispatcher.dispatch(
                        BatchDescriptor(num_tokens=num_tokens,
                                        uniform_decode=uniform_decode))
                # sanity check
                assert aclgraph_runtime_mode == _cg_mode, (
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {aclgraph_runtime_mode}.")

            need_dummy_logits = (not self.in_profile_run
                                 and lmhead_tp_enable())

            if need_dummy_logits:
                max_num_reqs_across_dp = num_tokens if not with_prefill else max_num_reqs
                dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                            dtype=torch.int32)

                def dummy_compute_logits(hidden_states):
                    return self.model.compute_logits(
                        hidden_states[dummy_indices], None)

            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=with_prefill,
                    in_profile_run=self.in_profile_run,
                    reserved_mc2_mask=self.reserved_mc2_mask,
                    moe_comm_method=moe_comm_method,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model):
                hidden_states = self._generate_dummy_run_hidden_states(
                    with_prefill, is_torchair_compile, input_ids, positions,
                    attn_metadata, num_tokens, intermediate_tensors,
                    inputs_embeds)
                if need_dummy_logits:
                    dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens,
                    with_prefill=with_prefill,
                    skip_attn=True,
                    num_reqs=num_reqs,
                    num_tokens_across_dp=num_tokens_across_dp)
                if need_dummy_logits:
                    dummy_compute_logits(hidden_states)
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
            hidden_states = self._dummy_run(
                self.max_num_tokens // self.cp_size if self.cp_size > 1 else self.max_num_tokens,
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

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_token_ids = torch.zeros((num_reqs, req_num_tokens),
                                      dtype=torch.int32,
                                      device=self.device)

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_prompt_lens = torch.tensor(
            num_scheduled_tokens_list,
            device="cpu",
        )
        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )

        dummy_metadata.build_pooling_cursor(num_scheduled_tokens_list,
                                            device=hidden_states.device)

        try:
            return model.pooler(hidden_states=hidden_states,
                                pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        # Find the task that has the largest output for subsequent steps
        output_size = dict[PoolingTask, float]()
        for task in self.get_supported_pooling_tasks():
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = output.get_data_nbytes()
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

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
                        module.weight.data = self._convert_torch_format(
                            module.weight.data)
            if self.drafter:
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
                if self.drafter.name == SpecDcodeType.EAGLE3:
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

    def _convert_torch_format(self, tensor):
        tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT)
        return tensor

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
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
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
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
                            rope_cache = self._convert_torch_format(rope_cache)
                            nope_cache = self._convert_torch_format(nope_cache)
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
                                kv_cache = self._convert_torch_format(kv_cache)
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

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
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

        forward_ctx = self.compilation_config.static_forward_context
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

    def initialize_aclgraph_capture(self) -> None:
        # TODO: Add check of AttentionCGSupport and cudagraph_mode.decode_mode when full graph is supported
        # Trigger aclgraph dispatching keys initialization here (after
        # initializing attn backends).
        self.aclgraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

    def _capture_aclgraphs(self, compilation_cases: list[int],
                           aclgraph_runtime_mode: CUDAGraphMode,
                           uniform_decode: bool):
        assert aclgraph_runtime_mode != CUDAGraphMode.NONE and \
            aclgraph_runtime_mode in [CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing ACL graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    aclgraph_runtime_mode.name))
        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = (aclgraph_runtime_mode == CUDAGraphMode.FULL)
                self._dummy_run(num_tokens,
                                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode)
            self._dummy_run(num_tokens,
                            aclgraph_runtime_mode=aclgraph_runtime_mode,
                            uniform_decode=uniform_decode)

    def _capture_model(self):
        if not self.use_aclgraph:
            logger.warning(
                "Skipping ACL graph capture. To turn on ACL graph capture, "
                "ensure `aclraph_mode` was not manually set to `NONE`")
            return
        else:
            self.initialize_aclgraph_capture()

        set_cudagraph_capturing_enabled(True)
        # Trigger ACL graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            aclgraph_mode = self.compilation_config.cudagraph_mode
            if aclgraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                aclgraph_runtime_mode = aclgraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.aclgraph_batch_sizes))
                self._capture_aclgraphs(
                    compilation_cases,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    uniform_decode=False)

        # Disable aclgraph capturing globally, so any unexpected aclgraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may doing lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

    def capture_model(self) -> None:

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        self._capture_model()

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

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

    def get_supported_pooling_tasks(self):
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def _build_drafter_prepare_inputs_torchair_param(self):
        return False
