#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
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

import math
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.attention.backends.abstract import AttentionBackend, AttentionType
from vllm.attention.layer import Attention, MLAAttention
from vllm.attention.selector import get_attn_backend
from vllm.config import (CompilationMode, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_dcp_group, get_dp_group,
                                             get_pcp_group, get_pp_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (AttentionSpec, CrossAttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             SamplerOutput,
                             make_empty_encoder_model_runner_output)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.worker.gpu_model_runner import (AsyncGPUModelRunnerOutput,
                                             GPUModelRunner)
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               set_draft_graph_params,
                                               set_graph_params,
                                               update_attn_dcp_pcp_params,
                                               update_attn_params,
                                               update_mla_attn_dcp_pcp_params,
                                               update_mla_attn_params)
# yapf: enable
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import \
    D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_utils import EPLBParamUtils
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (AscendDeviceType, ProfileExecuteDuration,
                               enable_sp, get_ascend_device_type, is_moe_model,
                               lmhead_tp_enable, maybe_trans_nz,
                               set_weight_prefetch_method, vllm_version_is)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType, get_mc2_tokens_capacity, select_moe_comm_method,
    set_ascend_forward_context, set_mc2_mask, set_mc2_tokens_capacity)
if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import torch_npu

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

if get_ascend_device_type() == AscendDeviceType._310P:
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


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: dict[str, Any]
    positions: torch.Tensor


class NPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group(
            ).rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = (self.max_num_tokens +
                                     self.max_num_reqs * 2 * self.pcp_size)
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.pin_memory,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens,
                                               dtype=torch.int32)
            self.positions = self._make_buffer(max_buffer_num_tokens,
                                               dtype=torch.int64)
        self.sampler = AscendSampler()
        self.attn_state = None

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger
                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                raise RuntimeError(
                    "Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config,
                                  "index_topk")
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm)

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs,
                        self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs,
                                self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        self.dynamic_eplb = self.ascend_config.dynamic_eplb or self.ascend_config.expert_map_record_path
        if self.dynamic_eplb:
            EPLBParamUtils.check_dynamic_eplb(self.ascend_config.dynamic_eplb)
            EPLBParamUtils.check_expert_map_record_path(
                self.ascend_config.expert_map_record_path)
            self.is_eplb_warmuped = False
            self.policy_type = self.ascend_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({
                "expert_map": None,
                "moe_load": None,
                "expert_maps": None
            })
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict,
                                            policy_type=self.policy_type,
                                            enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            ascend_config = get_ascend_config()
            self.eplb_updator = EplbUpdator(ascend_config, self.eplb_loader,
                                            self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len,
                              self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens
                if self.vllm_config.speculative_config else 0),
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        self.long_seq_metadata = None

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: Optional[Union[NgramProposer, EagleProposer, MtpProposer,
                                     SuffixDecodingProposer]] = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, EagleProposer)
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state)
                self.rejection_sampler = RejectionSampler(self.sampler)
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1,
                      self.decode_token_per_req))
        self.discard_request_indices = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method,
                                      self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.compilation_config.mode == CompilationMode.VLLM_COMPILE and not self.model_config.enforce_eager

    def _skip_all_reduce_across_dp_group(self) -> bool:
        """
        Decide whether to skip the all-reduce across the data-parallel (DP) group.

        Skipping is only applicable for MoE models and only on ranks that act as
        KV consumers. We skip the DP all-reduce when either:
        - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
        - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.
        """
        # Only applicable to MoE models and KV consumer ranks.
        if not is_moe_model(self.vllm_config) or not self.is_kv_consumer:
            return False

        def needs_mc2(num_tokens: int) -> bool:
            return select_moe_comm_method(num_tokens, self.vllm_config) in {
                MoECommType.MC2, MoECommType.FUSED_MC2
            }

        # Determine whether decode must use MC2. Use max cudagraph capture size
        # if available, otherwise use the maximal uniform decode token count.
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_must_use_mc2 = needs_mc2(potential_max_tokens)

        # For prefill, use the scheduler's max_num_batched_tokens for a single
        # batch.
        prefill_must_use_mc2 = needs_mc2(
            self.vllm_config.scheduler_config.max_num_batched_tokens)

        # Skip all-reduce if decode requires MC2 and either prefill also
        # requires MC2 or recompute-based scheduler is enabled.
        return decode_must_use_mc2 and (
            prefill_must_use_mc2
            or self.ascend_config.recompute_scheduler_enable)

    def _sync_metadata_across_dp(
            self, num_tokens: int,
            with_prefill: bool) -> tuple[int, Optional[torch.Tensor], bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill

        if self._skip_all_reduce_across_dp_group():
            num_tokens_after_padding = torch.tensor([num_tokens] *
                                                    self.dp_size,
                                                    device="cpu",
                                                    dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor([
            num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)
        ],
                                         dtype=torch.int32,
                                         device="cpu")

        flags_tensor = torch.tensor([int(with_prefill)],
                                    dtype=torch.int32,
                                    device="cpu")

        packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])
        # use cpu_group to avoid cpu synchronization issue.
        # it can be overlapped with main moell execution on npu.
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[:-1]
        synced_flags = packed_tensor[-1:]
        max_tokens_across_dp = torch.max(num_tokens_across_dp).item()
        global_with_prefill = bool(synced_flags[0])

        # Create a tensor for num_tokens_after_padding
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] *
                                                self.dp_size,
                                                device="cpu",
                                                dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, np.ndarray, int, torch.Tensor,
               int, torch.Tensor, SpecDecodeMetadata, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], int, dict[str,
                                                                         Any]]:
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

        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(tokens, dtype=np.int32)
        else:
            num_valid_tokens = np.array([
                num_tokens -
                len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for num_tokens, i in zip(tokens, req_ids)
            ],
                                        dtype=np.int32)
        # Get the attention state.
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens,
                                            num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)
        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.pcp_size * self.dcp_size > 1:
            self.pcp_manager.generate_pcp_mtp_input(
                num_reqs, total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens, with_prefill,
                self.input_batch, self.arange_np, req_indices, positions_np,
                cu_num_tokens)

        if self.pcp_size > 1:
            if not self.vllm_config.model_config.use_mla:
                self.pcp_manager.generate_kv_idx(scheduler_output,
                                                 self.input_batch)
            num_scheduled_tokens[:
                                 num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                                     num_scheduled_tokens[:num_reqs],
                                     self.arange_np,
                                     self.input_batch.num_reqs,
                                     self.reorder_batch_threshold,
                                 )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens)
            req_indices = np.repeat(self.arange_np[:num_reqs],
                                    num_scheduled_tokens)
            cu_num_tokens, _ = self._get_cumsum_and_arange(
                num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        max_num_scheduled_tokens = max(tokens)
        if (self.use_aclgraph and total_num_scheduled_tokens
                <= self.cudagraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        elif self.use_aclgraph and enable_sp(self.vllm_config):
            # When using aclgraph, if total_num_scheduled_tokens exceeds the maximum graph size,
            # the model will fall back to running its FX graph in eager mode.
            # In this case, when sequence parallelism is enabled, we need to pad tokens to align
            # with tp_size because pad_size cannot be captured by the FX graph
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_input_tokens = math.ceil(
                total_num_scheduled_tokens / tp_size) * tp_size
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get info across DP ranks.
        # NOTE: maybe_padded_num_tokens is only used when using TorchAir with DP,
        # Otherwise, it's just max_tokens_across_dp_cpu
        (maybe_padded_num_tokens, num_tokens_across_dp,
         with_prefill) = self._sync_metadata_across_dp(num_input_tokens,
                                                       with_prefill)
        self.with_prefill = with_prefill
        # TODO: Now that num_input_tokens is basically identical with maybe_padded_num_tokens
        # We should consider removing maybe_padded_num_tokens later
        num_input_tokens = maybe_padded_num_tokens

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or
                                                   self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)
        self.positions.cpu[total_num_scheduled_tokens:num_input_tokens].zero_()
        self.positions.copy_to_gpu()
        attn_metadata: dict[str, Any] = {}

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            original_seq_lens_np = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                np.array(tokens_original, dtype=np.int32))
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # _prepare_inputs may reorder the batch, so we must gather
        # multi-modal outputs after that to ensure the correct order
        if vllm_version_is('0.13.0'):
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        else:
            model_kwargs = self._init_model_kwargs()
        if self.is_multimodal_model and not self.model_config.is_encoder_decoder:
            self.multimodal_cpu_fields = ["grid_thw"]
            self._prepare_multimodal_fields()
            with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
            ):
                # Run the multimodal encoder if any.
                self._execute_mm_encoder(scheduler_output)

                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(
                    scheduler_output)

            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:total_num_scheduled_tokens].copy_(
                inputs_embeds)
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the acl graph all the time. The v0
            # engine avoids this by "double compiling" the acl graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the acl graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:total_num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the ACL graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        # Run the encoder, just like we do with other multimodal inputs.
        if self.model_config.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            input_ids = self.input_ids.gpu[:total_num_scheduled_tokens]
            positions = self.positions.gpu[:total_num_scheduled_tokens]
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        # type: ignore
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # If both flashcomm1 and pp are used simultaneously,
            # the shape of the received data and the shape of the space to be copied to will not match,
            # requiring a recalculation of the incoming data's shape.
            tp_size = get_tensor_model_parallel_world_size()
            num_input_tokens_with_flashcomm1 = num_input_tokens
            if enable_sp():
                num_input_tokens_with_flashcomm1 = (num_input_tokens +
                                                    tp_size - 1) // tp_size
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[
                    k][:num_input_tokens_with_flashcomm1].copy_(
                        v[:num_input_tokens_with_flashcomm1],
                        non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k:
                v[:num_input_tokens_with_flashcomm1]
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
            if self.pcp_size * self.dcp_size > 1:
                logits_indices = self.pcp_manager.get_logits_indices(
                    cu_num_tokens, num_reqs)
                logits_indices = logits_indices.pin_memory().to(
                    self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1:num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs]
                if self.pcp_size > 1 else None)
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Used in the below loop.
        self.spec_decode_common_attn_metadata = None
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens, encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                scheduler_output.num_scheduled_tokens or {},
                kv_cache_group_spec.kv_cache_spec,
                self.input_batch.num_reqs,
            )
            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                maybe_pcp_full_tokens = (
                    num_input_tokens if self.pcp_size == 1 else
                    total_num_scheduled_tokens * self.pcp_size -
                    sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs]))
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping.gpu[:
                                                          maybe_pcp_full_tokens]
                if self.pcp_size == 1:
                    slot_mapping[
                        total_num_scheduled_tokens:num_input_tokens].fill_(-1)
            if self.pcp_size * self.dcp_size > 1:
                self.long_seq_metadata = self.pcp_manager.generate_pcp_metadata(
                    total_num_scheduled_tokens, self.query_lens,
                    self.input_batch)
                blk_table.slot_mapping.gpu[maybe_pcp_full_tokens:].fill_(-1)
                if self.pcp_size > 1:
                    slot_mapping_pcp = self.pcp_manager.get_padded_slot_mapping(
                        total_num_scheduled_tokens,
                        slot_mapping,
                    )
                    blk_table.slot_mapping.gpu[:self.pcp_manager.
                                               num_actual_tokens_pcp_padded] = slot_mapping_pcp
                    slot_mapping = blk_table.slot_mapping.gpu[:self.
                                                              pcp_manager.
                                                              num_actual_tokens_pcp_padded]

            # NOTE: This is a temporary hack, now in GPUModelRunner, this prepare_inputs
            # has been split to multiple parts, and there are 3 parts that is related to this
            # `num_reqs`, we'll take `query_start_loc` as an example:
            # 1. self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            # 2. get `num_reqs_padded`, this depends on dispatcher and which is why we have the
            #    following simplified `dispatch` logic here, we try to minimize the impact
            # 3. query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            uniform_decode = (max_num_scheduled_tokens == self.uniform_decode_query_len) \
                and (total_num_scheduled_tokens == max_num_scheduled_tokens * num_reqs)

            # TODO: We should make this official ASAP. Also note that if we pad here,
            # the builders won’t need to add any extra padding.
            if self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                uniform_decode:
                max_decode_tokens = min(
                    self.scheduler_config.max_num_seqs *
                    self.uniform_decode_query_len,
                    self.cudagraph_batch_sizes[-1])
                if self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens:
                    num_reqs_padded = num_input_tokens // self.uniform_decode_query_len
                    pad_size = num_reqs_padded - num_reqs
                    if pad_size > 0:
                        last_query_loc = self.query_start_loc.np[num_reqs]

                        self.query_start_loc.np[
                            num_reqs + 1:num_reqs_padded + 1] = self.arange_np[
                                1:pad_size +
                                1] * self.uniform_decode_query_len + last_query_loc
                        self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)

                    # So we are trying to simulate the behavior of GPUModelRunner's
                    # prepare_inputs for uniform decode mode by padding query_start_loc
                    num_reqs = num_reqs_padded

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs + 1],
                seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                seq_lens=self.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                num_input_tokens=num_input_tokens,
                actual_seq_lengths_q=self.actual_seq_lengths_q,
                # TODO: change this to the right block table for linear attn
                block_table_tensor=blk_table_tensor[:num_reqs],
                slot_mapping=slot_mapping,
                num_computed_tokens_cpu=self.input_batch.
                num_computed_tokens_cpu_tensor[:num_reqs],
                positions=self.positions.gpu,
                attn_state=self.attn_state,
                max_query_len=max_num_scheduled_tokens,
                decode_token_per_req=self.decode_token_per_req,
                prefill_context_parallel_metadata=self.long_seq_metadata,
                max_seq_len=0,
                encoder_seq_lens=encoder_seq_lens,
                encoder_seq_lens_cpu=encoder_seq_lens_cpu)

            if self.speculative_config and self.pcp_size * self.dcp_size > 1:
                # For pcp + spec decode, we flatten block_table
                # to avoid irregular attn_mask shape, e.g.,
                # num_decode_req=2, num_prefill_req=3, num_speculative_tokens=1,
                # ori block_table: # [d0, d1, p0, p1, p2]
                # (num_reqs_d + num_reqs_p, max_num_blocks),
                # flattened block_table: [d0, d0, d1, d1, p0, p1, p2]
                # (num_reqs_d * decode_threshold + num_reqs_p, max_num_blocks),
                ori_query_lens_cpu = self.pcp_manager.query_lens_pcp_full.cpu[:
                                                                              num_reqs]
                ori_query_lens = self.pcp_manager.query_lens_pcp_full.gpu[:
                                                                          num_reqs]
                num_prefill_reqs = (ori_query_lens
                                    > self.decode_threshold).sum().item()
                num_decode_reqs = num_reqs - num_prefill_reqs
                num_decode_reqs_flatten = \
                    ori_query_lens_cpu[:num_decode_reqs].sum().item()
                blk_table_tensor[
                    num_decode_reqs_flatten:num_decode_reqs_flatten +
                    num_prefill_reqs].copy_(
                        blk_table_tensor[num_decode_reqs:num_decode_reqs +
                                         num_prefill_reqs].clone())
                blk_table_tensor[:num_decode_reqs_flatten].copy_(
                    blk_table_tensor[:num_decode_reqs].repeat_interleave(
                        ori_query_lens[:num_decode_reqs], dim=0))
                common_attn_metadata.block_table_tensor = \
                    blk_table_tensor[:num_decode_reqs_flatten + num_prefill_reqs]
                assert self.long_seq_metadata is not None
                self.long_seq_metadata.query_lens_pcp_full_cpu = ori_query_lens_cpu

                if 'pad_size' in locals() and pad_size > 0:
                    ori_query_lens_cpu[-pad_size:] = \
                        torch.full([pad_size], ori_query_lens_cpu[-pad_size - 1].item())
                self.long_seq_metadata.max_query_len_pcp_full = \
                    ori_query_lens_cpu.max().item()



            if self.speculative_config and \
                self.spec_decode_common_attn_metadata is None:
                self.spec_decode_common_attn_metadata = common_attn_metadata
                if num_reqs != base_num_reqs or total_num_scheduled_tokens != num_input_tokens:
                    self.spec_decode_common_attn_metadata = \
                        self.spec_decode_common_attn_metadata.unpadded(
                            total_num_scheduled_tokens, base_num_reqs)

            for attn_group in self.attn_groups[kv_cache_group_id]:
                common_prefix_len = 0
                extra_attn_metadata_args = {}
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    if use_spec_decode:
                        patch_torch_npu_argsort()
                        extra_attn_metadata_args = dict(
                            num_accepted_tokens=self.num_accepted_tokens.
                            gpu[:num_reqs],
                            num_decode_draft_tokens_cpu=self.
                            num_decode_draft_tokens.cpu[:num_reqs],
                        )
                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        # update global cos, sin
        update_cos_sin(positions)

        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(
                logits_indices,
                (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (attn_metadata, positions, num_scheduled_tokens,
                num_input_tokens, num_tokens_across_dp,
                maybe_padded_num_tokens, logits_indices, spec_decode_metadata,
                input_ids, inputs_embeds, intermediate_tensors,
                max_num_scheduled_tokens, model_kwargs)

    # all-gather one hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states(hidden_states):
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        pad_size = get_forward_context().pad_size
        if pad_size > 0:
            hidden_states = hidden_states[:-pad_size, :]

        return hidden_states

    # all-gather a list of hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_list(hidden_states_list):
        return [
            NPUModelRunner._all_gather_hidden_states(hidden_states)
            for hidden_states in hidden_states_list
        ]

    # all-gather hidden-states in last layer with aux-hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        if isinstance(hidden_states, tuple):
            return (NPUModelRunner._all_gather_hidden_states(hidden_states[0]),
                    NPUModelRunner._all_gather_hidden_states_list(
                        hidden_states[1]))
        return NPUModelRunner._all_gather_hidden_states(hidden_states)

    def _generate_process_reqs_hidden_states(self, maybe_padded_num_tokens,
                                             input_ids, positions,
                                             intermediate_tensors,
                                             inputs_embeds, model_kwargs):
        assert self.model is not None
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds,
                                   **model_kwargs)

        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL \
            and not self.use_sparse:
            # TODO: maybe_padded_num_tokens will be removed, use num_input_tokens instead
            if self.vllm_config.model_config.use_mla:
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   maybe_padded_num_tokens)
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_params(self.update_stream, forward_context,
                                           maybe_padded_num_tokens,
                                           self.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               maybe_padded_num_tokens)
                else:
                    update_attn_params(self.update_stream, forward_context,
                                       maybe_padded_num_tokens,
                                       self.vllm_config)

        if get_forward_context().sp_enabled and not isinstance(
                hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(
                hidden_states)
        return hidden_states if self.pcp_size == 1 else self.pcp_manager.get_restore_hidden_states(
            hidden_states)

    def _build_attn_state(self, num_reqs, num_scheduled_tokens,
                          num_valid_tokens):
        if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == 'mtp':
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == 'mtp':
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit
        return attn_state

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        num_pcp_pads: np.ndarray | None,
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

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(
                cu_num_scheduled_tokens - num_sampled_tokens,
                num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(
                logits_indices_pcp).pin_memory().to(self.device,
                                                    non_blocking=True)

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
        cu_num_draft_tokens = (
            torch.from_numpy(cu_num_draft_tokens).pin_memory().to(
                self.device, non_blocking=True))
        cu_num_sampled_tokens = (
            torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(
                self.device, non_blocking=True))
        logits_indices = (torch.from_numpy(logits_indices).pin_memory().to(
            self.device, non_blocking=True))
        target_logits_indices = (
            torch.from_numpy(target_logits_indices).pin_memory().to(
                self.device, non_blocking=True))
        bonus_logits_indices = torch.from_numpy(
            bonus_logits_indices).pin_memory().to(self.device,
                                                  non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        if self.pcp_size > 1:
            logits_indices = logits_indices_pcp
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    # TODO: Once the PCP features are complete, it will fully inherit the classes from the VLLM community.
    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: dict[str, Any],
        aux_hidden_states: torch.Tensor = None,
    ) -> Optional[list[list[int]]]:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            if self.speculative_config.method in ("suffix", "ngram"):
                draft_token_ids = self.drafter.generate_token_ids(
                    valid_sampled_token_ids, sampling_metadata,
                    scheduler_output, spec_decode_metadata, positions,
                    num_scheduled_tokens, hidden_states, aux_hidden_states)

            elif self.speculative_config.use_eagle():
                common_attn_metadata = self.spec_decode_common_attn_metadata
                sampled_token_ids = valid_sampled_token_ids

                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    # When padded-batch is disabled, the sampled_token_ids should be
                    # the cpu-side list[list[int]] of valid sampled tokens for each
                    # request, with invalid requests having empty lists.
                    assert isinstance(sampled_token_ids, list), \
                        "sampled_token_ids should be a python list when" \
                        "padded-batch is disabled."
                    assert self.drafter is not None
                    next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                        sampled_token_ids, self.requests, self.input_batch,
                        scheduler_output.num_scheduled_tokens)
                else:
                    # When using padded-batch, the sampled_token_ids should be
                    # the gpu tensor of sampled tokens for each request, of shape
                    # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                    # value -1.
                    assert isinstance(sampled_token_ids, torch.Tensor), \
                        "sampled_token_ids should be a torch.Tensor when" \
                        "padded-batch is enabled."
                    assert self.drafter is not None
                    next_token_ids, valid_sampled_tokens_count = \
                        self.drafter.prepare_next_token_ids_padded(
                            common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_indices.gpu,
                            self.num_discarded_requests
                        )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count)

                req_scheduled_tokens = scheduler_output.num_scheduled_tokens
                if self.pcp_size * self.dcp_size > 1:
                    long_seq_metadata = self.long_seq_metadata  # type: ignore
                    input_ids_pcp_full = self.pcp_manager.input_ids_pcp_full.gpu
                    query_start_loc_pcp_full = self.pcp_manager.query_start_loc_pcp_full.gpu
                    query_start_loc_pcp_full_cpu = self.pcp_manager.query_start_loc_pcp_full.cpu
                    num_reqs = self.input_batch.num_reqs
                    ori_query_lens = query_start_loc_pcp_full_cpu[1:num_reqs+1] - \
                        query_start_loc_pcp_full_cpu[:num_reqs]
                    num_prefill_reqs = (ori_query_lens
                                        > self.decode_threshold).sum().item()
                    num_decode_reqs = num_reqs - num_prefill_reqs
                else:
                    long_seq_metadata = None  # type: ignore
                    num_prefill_reqs = 0
                    num_decode_reqs = 0
                if spec_decode_metadata is None:
                    # update pcp related params
                    if self.pcp_size > 1:
                        token_indices_to_sample = \
                            query_start_loc_pcp_full[1:num_reqs + 1] - 1
                        target_token_ids = input_ids_pcp_full[:
                                                              num_scheduled_tokens]
                        target_positions = positions[:num_scheduled_tokens]
                        target_hidden_states = hidden_states
                    else:
                        token_indices_to_sample = None
                        # input_ids can be None for multimodal models.
                        target_token_ids = self.input_ids.gpu[:
                                                              num_scheduled_tokens]
                        target_positions = positions[:num_scheduled_tokens]
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat([
                                h[:num_scheduled_tokens]
                                for h in aux_hidden_states
                            ],
                                                             dim=-1)
                        else:
                            target_hidden_states = hidden_states[:
                                                                 num_scheduled_tokens]
                else:
                    if self.pcp_size > 1:
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc_cpu[:num_reqs + 1] = \
                            query_start_loc_pcp_full_cpu[:num_reqs + 1]
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc[:num_reqs + 1] = \
                            query_start_loc_pcp_full[:num_reqs + 1]
                    if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                        # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                        token_indices_to_sample = None
                        assert self.drafter is not None
                        common_attn_metadata, token_indices =\
                            self.drafter.prepare_inputs(
                                common_attn_metadata,
                                sampled_token_ids,
                                spec_decode_metadata.num_draft_tokens)
                    else:
                        assert self.drafter is not None
                        common_attn_metadata, token_indices, \
                            token_indices_to_sample =\
                                self.drafter.prepare_inputs_padded(
                                    common_attn_metadata,
                                    spec_decode_metadata,
                                    valid_sampled_tokens_count)
                    if self.pcp_size > 1:
                        target_token_ids = input_ids_pcp_full[token_indices]
                        target_positions = positions
                        target_hidden_states = hidden_states
                    else:
                        target_token_ids = self.input_ids.gpu[token_indices]
                        target_positions = positions[token_indices]
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat(
                                [h[token_indices] for h in aux_hidden_states],
                                dim=-1)
                        else:
                            target_hidden_states = hidden_states[token_indices]
                assert self.drafter is not None
                draft_token_ids = self.drafter._propose(
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

            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")

        return draft_token_ids

    @staticmethod
    def get_finished_kv_transfer(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors] | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")

        with ProfileExecuteDuration().capture_async("prepare input"):
            self._update_states(scheduler_output)
            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                ):
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(
                        scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    logger.debug(
                        "skip this step for we receive the data from remote disaggregate prefill node"
                    )
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            if self.dynamic_eplb:
                self.eplb_updator.forward_before()

            (attn_metadata, positions, num_scheduled_tokens_np,
             num_input_tokens, num_tokens_across_dp, maybe_padded_num_tokens,
             logits_indices, spec_decode_metadata, input_ids, inputs_embeds,
             intermediate_tensors, max_query_len,
             model_kwargs) = (self._prepare_inputs(scheduler_output,
                                                   intermediate_tensors))

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(
                getattr(dbg_cfg, "level",
                        "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            scheduler_output.total_num_scheduled_tokens
            == self.input_batch.num_reqs * max_query_len)
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0
        aclgraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators)

        # Run forward pass
        with ProfileExecuteDuration().capture_async("forward"):
            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    num_actual_tokens=scheduler_output.
                    total_num_scheduled_tokens,
                    model_instance=self.model):
                self.maybe_setup_kv_connector(scheduler_output)

                hidden_states = self._generate_process_reqs_hidden_states(
                    maybe_padded_num_tokens, input_ids, positions,
                    intermediate_tensors, inputs_embeds, model_kwargs)

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfer(
                scheduler_output)

            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

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
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                assert isinstance(hidden_states, IntermediateTensors)
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group())
                logits = None
            else:
                if self.input_batch.pooling_params:
                    if vllm_version_is('0.13.0'):
                        pool_output = self._pool(
                            hidden_states,
                            scheduler_output.total_num_scheduled_tokens,
                            num_scheduled_tokens_np)
                    else:
                        pool_output = self._pool(
                            hidden_states,
                            scheduler_output.total_num_scheduled_tokens,
                            num_scheduled_tokens_np, kv_connector_output)
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return pool_output
                # Sometimes, after the model is compiled through the AOT backend,
                # the model output may become a list containing only one Tensor object.
                if isinstance(hidden_states, list) and \
                        len(hidden_states) == 1 and \
                        isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
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
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with ProfileExecuteDuration().capture_async("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert self.spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
            )

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        with ProfileExecuteDuration().capture_async("Draft"):
            if self.speculative_config:
                use_padded_batch_for_eagle = self.speculative_config and \
                    self.speculative_config.use_eagle() and \
                    not self.speculative_config.disable_padded_drafter_batch
                if use_padded_batch_for_eagle:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch_for_eagle:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

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
        if self.dynamic_eplb:
            self.eplb_updator.forward_end()
        if not self.use_async_scheduling:
            if self.debugger is not None:
                assert self.debugger is not None
                self.debugger.stop()
                self.debugger.step()
            return model_runner_output

        if self.debugger is not None:
            assert self.debugger is not None
            self.debugger.stop()
            self.debugger.step()
        return AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[:self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[:len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        if self.need_accepted_tokens:  # TODO remove this if
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids)
        return sampler_output

    # TODO: remove this func after eagle_proposer is refactored and
    #  _bookkeeping_sync is moved after propose_draft_token_ids
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
            LogprobsLists | None,
            list[list[int]],
            dict[str, LogprobsTensors | None],
            list[str],
            dict[str, int],
            list[int],
    ]:
        # TODO: implement PR 28597 from vllm
        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    return_cu_num_tokens=logprobs_tensors is not None,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            if self.num_spec_tokens <= 0:
                assert sampled_token_ids.shape[-1] == 1
                # Cache the sampled tokens on the NPU and avoid CPU sync.
                # These will be copied into input_ids in the next step
                # when preparing inputs.
                self.input_batch.prev_sampled_token_ids = sampled_token_ids

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
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [
                    -1
                ] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (logprobs_tensors.tolists(cu_num_tokens)
                          if not self.use_async_scheduling
                          and logprobs_tensors is not None else None)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    def _build_dummy_attn_metadata(
        self,
        with_prefill: bool,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        num_scheduled_tokens: np.ndarray,
        aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
    ) -> Optional[dict[str, Any]]:
        attn_metadata: Optional[dict[str, Any]] = None

        if force_attention or aclgraph_runtime_mode == CUDAGraphMode.FULL:
            assert with_prefill is False, \
                "Full decode graph only supports uniform batch now."

            attn_metadata = {}

            seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cu_num_tokens, arange = self._get_cumsum_and_arange(
                num_scheduled_tokens)

            self.query_start_loc.cpu[1:num_reqs +
                                     1] = torch.Tensor(cu_num_tokens)
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

            num_computed_tokens_cpu = (
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                block_table_tensor = self.input_batch.block_table[
                    kv_cache_group_id].get_device_tensor()
                slot_mapping = self.input_batch.block_table[
                    kv_cache_group_id].slot_mapping
                long_seq_metadata = None if self.pcp_size * self.dcp_size == 1 else self.pcp_manager.generate_pcp_metadata(
                    num_tokens, self.query_lens, self.input_batch)
                if long_seq_metadata is not None:
                    pcp_world_size = get_pcp_group().world_size
                    dcp_world_size = get_dcp_group().world_size
                    num_computed_tokens_of_pcp_dcp = [[
                        [0] * dcp_world_size for _ in range(pcp_world_size)
                    ] for _ in range(num_tokens)]
                    long_seq_metadata.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp

                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens_cpu=self.seq_lens.cpu,
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    actual_seq_lengths_q=self.actual_seq_lengths_q,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    positions=self.positions.gpu,
                    attn_state=self.attn_state,
                    max_query_len=max_query_len,
                    decode_token_per_req=self.decode_token_per_req,
                    prefill_context_parallel_metadata=long_seq_metadata,
                    max_seq_len=0)
                if self.pcp_size * self.dcp_size > 1:
                    common_attn_metadata.block_table_tensor = \
                        block_table_tensor[:num_reqs * self.decode_threshold]
                attn_state = AscendAttentionState.DecodeOnly
                if self.speculative_config and \
                        self.speculative_config.method == "mtp":
                    # `AscendAttentionState.SpecDecoding` is only designed for mla
                    if self.vllm_config.model_config.use_mla:
                        attn_state = AscendAttentionState.SpecDecoding
                    else:
                        attn_state = AscendAttentionState.ChunkedPrefill

                common_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    seq_lens=self.seq_lens.cpu[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    block_table_tensor=block_table_tensor[:num_reqs],
                    slot_mapping=slot_mapping.gpu,
                    _num_computed_tokens_cpu=num_computed_tokens_cpu,
                    max_query_len=max_query_len,
                    max_seq_len=seq_lens)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    builder = attn_group.get_metadata_builder()
                    if isinstance(builder, GDNAttentionMetadataBuilder):
                        attn_metadata_gdn_attention = builder.build_for_cudagraph_capture(
                            common_metadata)
                    else:
                        attn_metadata_full_attention = builder.build_for_graph_capture(
                            common_attn_metadata, attn_state)
                    for layer_name in kv_cache_group_spec.layer_names:
                        if "linear_attn" in layer_name:
                            attn_metadata[
                                layer_name] = attn_metadata_gdn_attention
                        else:
                            attn_metadata[
                                layer_name] = attn_metadata_full_attention

        return attn_metadata

    def _generate_dummy_run_hidden_states(self, input_ids, positions,
                                          num_tokens, intermediate_tensors,
                                          inputs_embeds):
        hidden_states = self.model(input_ids=input_ids,
                                   positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds)
        forward_context = get_forward_context()
        assert forward_context is not None
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            not forward_context.capturing and not self.use_sparse:
            if self.vllm_config.model_config.use_mla:
                # FIXME: Try using `auto_dispatch_capture=True`
                if self.pcp_size * self.dcp_size > 1:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_dcp_pcp_params(self.update_stream,
                                                   forward_context,
                                                   positions.shape[0])
                else:
                    # FIXME: Try using `auto_dispatch_capture=True`
                    update_mla_attn_params(self.update_stream, forward_context,
                                           num_tokens, self.speculative_config)
            else:
                if self.pcp_size * self.dcp_size > 1:
                    update_attn_dcp_pcp_params(self.update_stream,
                                               forward_context,
                                               positions.shape[0])
                else:
                    update_attn_params(self.update_stream, forward_context,
                                       num_tokens, self.vllm_config)

        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = hidden_states
        else:
            hidden_states = hidden_states
        return hidden_states

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> torch.Tensor:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # In multi-DP scenarios, there may be situations where all DP groups are executing dummy runs.
        # If sequence parallelism is enabled, it is essential to ensure that num_tokens is divisible by tp_size.
        if self.use_aclgraph and enable_sp(self.vllm_config):
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            num_tokens = math.ceil(num_tokens / tp_size) * tp_size

        # Force dummy run on prefill stage when this node is deemed as kv producer.
        if self.is_kv_producer and not self.is_kv_consumer:
            with_prefill = True

        # Padding for DP
        (num_tokens, num_tokens_across_dp,
         with_prefill) = self._sync_metadata_across_dp(num_tokens,
                                                       with_prefill)

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
        max_num_reqs = self.max_num_reqs
        if uniform_decode:
            num_reqs = cdiv(num_tokens, max_query_len)
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
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        has_lora = True if self.lora_config and self.compilation_config.cudagraph_specialize_lora else False
        _ag_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora)

        num_tokens_padded = batch_descriptor.num_tokens
        num_reqs_padded = (batch_descriptor.num_reqs if
                           batch_descriptor.num_reqs is not None else num_reqs)
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)

        # filter out the valid batch descriptor
        if cudagraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            if cudagraph_runtime_mode != CUDAGraphMode.NONE and cudagraph_runtime_mode != _ag_mode:
                raise ValueError(
                    f"Aclgraph runtime mode mismatch at dummy_run. "
                    f"Expected {_ag_mode}, but got {cudagraph_runtime_mode}.")
        else:
            cudagraph_runtime_mode = _ag_mode

        # TODO(Mengqing): Set create_mixed_batch to False since it's only used in FI warmup
        # and not supported in ASCEND now. We could remove it in the future.
        attn_metadata = self._build_dummy_attn_metadata(
            False,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            force_attention=force_attention,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens,
                                            num_sampled_tokens):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by tp_size;
                # otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading to incorrect memory estimation and potentially causing OOM.
                actual_tokens = num_tokens
                if enable_sp():
                    tp_size = get_tensor_model_parallel_world_size()
                    actual_tokens = num_tokens // tp_size
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=actual_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k:
                    v[:num_tokens_padded]
                    for k, v in self.intermediate_tensors.items()
                })

            need_dummy_logits = (not is_profile and lmhead_tp_enable())
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp,
                                        dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(
                        self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(
                        hidden_states[dummy_indices])

            with set_ascend_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    in_profile_run=is_profile,
                    num_actual_tokens=0,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    model_instance=self.model):
                hidden_states = self._generate_dummy_run_hidden_states(
                    input_ids, positions, num_tokens_padded,
                    intermediate_tensors, inputs_embeds)
                dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile)
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()
                self.eplb_updator.forward_end()
            return hidden_states, hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = None

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        min_tokens_per_req = self.max_num_tokens // self.max_num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * self.max_num_reqs
        num_scheduled_tokens_list[
            -1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        # Sometimes, after the model is compiled through the AOT backend,
        # the model output may become a list containing only one Tensor object.
        if isinstance(hidden_states, list) and \
            len(hidden_states) == 1 and \
            isinstance(hidden_states[0], torch.Tensor):
            hidden_states = hidden_states[0]
            hidden_states = hidden_states[logit_indices]
            output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and \
            select_moe_comm_method(mc2_tokens_capacity, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity,
                            with_prefill=True,
                            is_profile=True)
        origin_max_num_tokens = self.max_num_tokens
        # in the pcp scenario, the split sequence needs to be used for profile run
        # TODO: after the vllm pcp function is launched, this logic needs to be brought up to the community
        if self.pcp_size > 1:
            self.max_num_tokens = math.ceil(self.max_num_tokens /
                                            (self.pcp_size * 2)) * 2
        super().profile_run()
        self.max_num_tokens = origin_max_num_tokens

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.dynamic_eplb:
                model_register(self.model, self.model_config)
            if self.drafter:
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
                if self.use_aux_hidden_state_outputs:
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = (len(self.attn_groups) > 1)
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any([
            isinstance(attn_group[0].kv_cache_spec, MambaSpec)
            for attn_group in self.attn_groups
        ])

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def _align_memory(self, tensor: torch.Tensor,
                      alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache
        num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches, num_attn_module)
        return kv_caches

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cahce and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
         """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, Union[torch.Tensor,
                                              Optional[torch.Tensor]]] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # for mamba linear attention
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                             dtype=torch.int8,
                                             device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned,
                                             dtype=torch.int8,
                                             device=self.device)
                        tensor = self._align_memory(
                            tensor, alignment)[:kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors.keys(
                ):
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    if self.model_config.use_mla:
                        head_size = self.model_config.hf_text_config.qk_rope_head_dim + \
                            self.model_config.hf_text_config.kv_lora_rank

                    dsa_k_cache_factor = None
                    dsa_k_cache_size = None
                    if not self.model_config.use_mla:
                        # for non-mla model, use FullAttentionSpec
                        k_tensor_split_factor = 2
                        v_tensor_split_factor = 2
                    elif self.use_sparse:
                        # for deepseek v3.2, DSA use FullAttentionSpec
                        # FullAttentionSpec allocate 2 * mla page size bytes,
                        # and we use half of that for k cache in DSA
                        dsa_k_cache_factor = 2
                        k_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = 2 * head_size / self.model_config.hf_text_config.qk_rope_head_dim
                        dsa_k_cache_size = int(kv_cache_tensor.size //
                                               dsa_k_cache_factor)
                    else:
                        # for other deepseek models, use MLAAttentionSpec
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim

                    k_tensor_size = int(kv_cache_tensor.size //
                                        k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size //
                                        v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size,
                                               dtype=torch.int8,
                                               device=self.device)
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size,
                                dtype=torch.int8,
                                device=self.device)
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment,
                                               dtype=torch.int8,
                                               device=self.device)
                        k_tensor = self._align_memory(
                            k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(
                            v_tensor, alignment)[:v_tensor_size]
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None and dsa_k_cache_size is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size + alignment,
                                dtype=torch.int8,
                                device=self.device)
                            dsa_k_cache_tensor = self._align_memory(
                                dsa_k_cache_tensor,
                                alignment)[:dsa_k_cache_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if ("attn" in layer_name_inner
                                and "linear_attn" not in layer_name_inner):
                            kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor) if \
                                not self.use_sparse else (k_tensor, v_tensor, dsa_k_cache_tensor)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, AttentionSpec):
                    raw_dsa_k_tensor = None
                    if self.use_sparse:
                        raw_k_tensor, raw_v_tensor, raw_dsa_k_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        assert raw_dsa_k_tensor is not None
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel(
                        ) + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert raw_v_tensor is not None
                    assert sum_page_size_bytes % kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // kv_cache_spec.page_size_bytes

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.
                    assert num_blocks >= kv_cache_config.num_blocks

                    if hasattr(attn_backend, "get_supported_block_size"
                               ) and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_block_size()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk, block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if not self.model_config.use_mla:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank
                        ]
                        v_shape = [
                            mla_num_blocks, mla_block_size, num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim
                        ]
                    k_cache = raw_k_tensor.view(dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(dtype).view(v_shape)
                    if get_ascend_device_type() == AscendDeviceType._310P:
                        k_cache = maybe_trans_nz(k_cache)
                        v_cache = maybe_trans_nz(v_cache)
                    if self.use_sparse and raw_dsa_k_tensor is not None:
                        dsa_k_cache_shape = (num_blocks,
                                             kv_cache_spec.block_size, 1, 128)
                        dsa_k_cache_size = (
                            num_blocks
                        ) * kv_cache_spec.block_size * 128 * dtype.itemsize
                        dsa_k_cache = raw_dsa_k_tensor[:dsa_k_cache_size].view(
                            dtype).view(dsa_k_cache_shape)
                        kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel(
                    ) % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel(
                    ) // kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    # `num_blocks` is the number of blocks the model runner can use.
                    # `kv_cache_config.num_blocks` is the number of blocks that
                    # KVCacheManager may allocate.
                    # Since different GPUs may have different number of layers and
                    # different memory capacities, `num_blocks` can be different on
                    # different GPUs, and `kv_cache_config.num_blocks` is set to
                    # the min of all `num_blocks`. Verify it here.

                    state_tensors = []
                    target_idx = 0
                    start_idx = 0
                    for shape, dtype in zip(kv_cache_spec.shapes,
                                            kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += torch.prod(
                            torch.tensor(target_shape)).item()
                        tensor = raw_tensor.view(
                            dtype)[start_idx:target_idx].view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(
                kv_cache_config.kv_cache_groups):

            if isinstance(kv_cache_group.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # the backend.
                try:
                    attn_groups = self.attn_groups[kv_cache_group_id]
                except IndexError:
                    attn_groups = None
                if attn_groups and self.use_hybrid_blocks:
                    # Use the backend's supported block size list
                    backend = attn_groups[0].backend
                    supported_sizes = backend.get_supported_block_size()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = (supported_sizes
                                              if supported_sizes else
                                              [self.cache_config.block_size])
                else:
                    # Fallback to cache config block_size if no backend found
                    kernel_block_size_list = [self.cache_config.block_size]
                kernel_block_sizes.append(kernel_block_size_list)
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                kernel_block_sizes.append([0])
        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len,
                                  self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
                kernel_block_sizes=kernel_block_sizes,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]],
                   set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
                kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {
                    attn_backends[k]: v
                    for k, v in attn_backend_layers.items()
                },
                set(group_key.attn_backend
                    for group_key in attn_backends.values()),
            )

        def create_attn_groups(attn_backends_map: dict[AttentionBackend,
                                                       list[str]],
                               kv_cache_group_id: int) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = []
                attn_metadata_builders.append(attn_backend.get_builder_cls()(
                    kv_cache_spec,
                    layer_names,
                    self.vllm_config,
                    self.device,
                ))
                attn_group = AttentionGroup(attn_backend, layer_names,
                                            kv_cache_spec, kv_cache_group_id,
                                            attn_metadata_builders)
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        self._check_and_update_cudagraph_mode(attention_backend_list,
                                              kv_cache_config.kv_cache_groups)

        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec)
            self.attn_groups.append(create_attn_groups(attn_backends[0], i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()
            if hasattr(attn_metadata_builder_i,
                       "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = (
                    attn_metadata_builder_i.reorder_batch_threshold)
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != \
                            self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}")
                    else:
                        self.reorder_batch_threshold = reorder_batch_threshold_i  # noqa

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                  AttentionLayerBase)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                # TODO: Support other attention modules, e.g., cross-attention
                # TODO(lucas): move the attention specs into the model layers like
                # the attention backends
                if attn_module.attn_type == AttentionType.DECODER:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                elif attn_module.attn_type in (AttentionType.ENCODER,
                                               AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    kv_cache_spec[layer_name] = CrossAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")

            elif isinstance(attn_module, MLAAttention):
                if use_mla and not self.use_sparse:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.cache_config.cache_dtype)
                else:
                    # TODO(cmq): This is a hack way to fix deepseek kvcache when
                    # using DSA. Fix the spec in vLLM is a finnal way.
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            if (self.vllm_config.speculative_config is not None
                    and self.vllm_config.model_config.hf_text_config.model_type
                    not in ["qwen3_next"]):
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type,
                    num_speculative_blocks=(
                        self.speculative_config.num_speculative_tokens
                        if self.speculative_config else 0),
                )

        return kv_cache_spec

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        super()._check_and_update_cudagraph_mode(attention_backends,
                                                 kv_cache_groups)

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(self.cudagraph_batch_sizes)
            if self.speculative_config:
                set_draft_graph_params(self.cudagraph_batch_sizes)

    def capture_model(self) -> None:
        parent_module_name = self.__class__.__base__.__module__
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(
                parent_module_name):
            super().capture_model()

    def _prepare_multimodal_fields(self):
        """
        Ensures specific multimodal tensors are on CPU.
        This is necessary for fields like 'grid_thw' which are converted to numpy 
        inside the model's forward pass.
        """
        if not self.multimodal_cpu_fields:
            return

        req_ids = self.input_batch.req_ids
        for req_id in req_ids:
            req = self.requests.get(req_id)
            if req is None:
                continue

            mm_data = getattr(req, 'multimodal_data', None)
            if not mm_data:
                continue

            for field in self.multimodal_cpu_fields:
                if field in mm_data:
                    tensor = mm_data[field]
                    if isinstance(
                            tensor,
                            torch.Tensor) and tensor.device.type != 'cpu':
                        mm_data[field] = tensor.cpu()


@contextmanager
def _torch_cuda_wrapper():

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        yield
    except Exception as e:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # if anything goes wrong, just patch it with a placeholder
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info


# TODO: This method will be removed subsequently and implemented in platform.
@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    try:
        target_module = sys.modules[target_module_name]
        setattr(target_module, "graph_capture", graph_capture)
        yield
    finally:
        setattr(target_module, "graph_capture", graph_capture)
