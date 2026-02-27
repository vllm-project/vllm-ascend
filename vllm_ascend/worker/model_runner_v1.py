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
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_dcp_group, get_dp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, using_paged_attention

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    set_draft_graph_params,
    set_graph_params,
    update_full_graph_params,
)

# yapf: enable
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.medusa_proposer import MedusaProposer
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.utils import (
    enable_flash_comm_v1,
    enable_sp,
    is_drafter_moe_model,
    is_moe_model,
    lmhead_tp_enable,
    set_weight_prefetch_method,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


from vllm.model_executor.layers.attention import Attention, MLAAttention

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144


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
    graph_capture_context = GraphCaptureContext(torch.npu.Stream(device=device))
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


def get_tp_context(drafter):
    return getattr(drafter, "tp_group_context", nullcontext())


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: "PerLayerAttnMetadata"
    positions: torch.Tensor
    ec_connector_output: "ECConnectorOutput | None"
    cudagraph_stats: CUDAGraphStat | None


class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = (
            vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        )
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 2,  # type: ignore[has-type]
            dtype=torch.int32,
        )

        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = self.max_num_tokens + self.max_num_reqs * 2 * self.pcp_size
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens, dtype=torch.int32)
            self.positions = self._make_buffer(max_buffer_num_tokens, dtype=torch.int64)
        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None

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
                raise RuntimeError("Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            self.block_size,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None and self.model_config.is_mm_prefix_lm,
        )

        self._set_up_drafter()

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        eplb_config = self.ascend_config.eplb_config
        self.dynamic_eplb = eplb_config.dynamic_eplb
        if self.dynamic_eplb:
            self.is_eplb_warmuped = False
            self.policy_type = eplb_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict, policy_type=self.policy_type, enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
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
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors,
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
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
        self.cpu_slot_mapping = None

    @property
    def use_cp(self) -> bool:
        return self.pcp_size * self.dcp_size > 1

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: NgramProposer | EagleProposer | MtpProposer | SuffixDecodingProposer | MedusaProposer | None = (
            None
        )
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
                    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
                self.rejection_sampler = RejectionSampler(self.sampler)
            self.actual_seq_lengths_q = list(
                range(self.decode_token_per_req, self.max_num_tokens + 1, self.decode_token_per_req)
            )
        self.discard_request_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )

    def _skip_all_reduce_across_dp_group(self, is_draft_model=False) -> bool:
        """
        Decide whether to skip the all-reduce across the data-parallel (DP) group.

        Skipping is applicable for all dense models and for moe models only on ranks
        that act as KV consumers. We skip the DP all-reduce when either:
        - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
        - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.
        """
        # For dense models, since we don't actually need dp communication, we simply skip it.
        # This usually happens when main model is moe while eagle draft model is dense.
        is_context_moe_model = (
            is_drafter_moe_model(self.vllm_config) if is_draft_model else is_moe_model(self.vllm_config)
        )
        if not is_context_moe_model:
            return True

        # Only applicable to MoE models on KV consumer ranks.
        if not self.is_kv_consumer:
            return False

        def needs_mc2(num_tokens: int) -> bool:
            return select_moe_comm_method(num_tokens, self.vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

        # Determine whether decode must use MC2. Use max cudagraph capture size
        # if available, otherwise use the maximal uniform decode token count.
        if self.compilation_config.cudagraph_capture_sizes:
            potential_max_tokens = self.compilation_config.max_cudagraph_capture_size
        else:
            potential_max_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_must_use_mc2 = needs_mc2(potential_max_tokens)

        # For prefill, use the scheduler's max_num_batched_tokens for a single
        # batch.
        prefill_must_use_mc2 = needs_mc2(self.vllm_config.scheduler_config.max_num_batched_tokens)

        # Skip all-reduce if decode requires MC2 and either prefill also
        # requires MC2 or recompute-based scheduler is enabled.
        return decode_must_use_mc2 and (prefill_must_use_mc2 or self.ascend_config.recompute_scheduler_enable)

    def _sync_metadata_across_dp(
        self, num_tokens: int, with_prefill: bool = False, is_draft_model: bool = False
    ) -> tuple[int, torch.Tensor | None, bool]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, with_prefill

        if self._skip_all_reduce_across_dp_group(is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, with_prefill

        # Sync num_tokens, with_prefill across dp ranks
        num_tokens_tensor = torch.tensor(
            [num_tokens if i == self.dp_rank else 0 for i in range(self.dp_size)], dtype=torch.int32, device="cpu"
        )

        flags_tensor = torch.tensor([int(with_prefill)], dtype=torch.int32, device="cpu")

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
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32)

        return max_tokens_across_dp, num_tokens_after_padding, global_with_prefill

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def _pad_query_start_loc_for_fia(self, num_tokens_padded: int, num_reqs_padded: int, num_reqs: int) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """

        if num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()

        return num_reqs_padded

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # Get the attention state.
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = num_scheduled_tokens
        else:
            num_valid_tokens = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
        self.attn_state = attn_state  # type: ignore

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
            )

        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.use_cp:
            self.pcp_manager.generate_pcp_mtp_input(
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens,
            )

        if self.pcp_size > 1:
            num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs],
                self.arange_np,
            )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens)
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            positions_np = self.positions.np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
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
                    self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                        req_embeds[start_pos:actual_end]
                    )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        self.seq_lens.np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens, cu_num_tokens)
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

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            num_draft_tokens = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            if self.use_cp:
                logits_indices = self.pcp_manager.get_logits_indices(cu_num_tokens)
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (
                    len(draft_token_ids)
                    if (
                        self.input_batch.num_computed_tokens_cpu[req_idx] >= self.input_batch.num_prompt_tokens[req_idx]
                    )
                    else -1
                )

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs] if self.pcp_size > 1 else None,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return logits_indices, spec_decode_metadata

    def _build_attn_state(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
        if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # SpecDecoding now supports seq_len=1 and seq_len=2
                # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
                attn_state = AscendAttentionState.SpecDecoding
        # Speculative decoding.
        elif np.all(num_valid_tokens == 1):
            if self.speculative_config and self.speculative_config.method == "mtp":
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
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # while pcp > 1, decode results may contain padding (from pcp all-gather),
        # update logits_indices after getting draft_token_ids from ori logits_indices
        if self.pcp_size > 1:
            cu_num_scheduled_tokens = cu_num_scheduled_tokens * self.pcp_size - num_pcp_pads
            logits_indices_pcp = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
            logits_indices_pcp += arange
            logits_indices_pcp = torch.from_numpy(logits_indices_pcp).pin_memory().to(self.device, non_blocking=True)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens, num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).pin_memory().to(self.device, non_blocking=True)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).pin_memory().to(self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).pin_memory().to(self.device, non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).pin_memory().to(self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).pin_memory().to(self.device, non_blocking=True)

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
        spec_decode_common_attn_metadata: AscendCommonAttentionMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: list[dict[str, Any]] | dict[str, Any],
        aux_hidden_states: torch.Tensor = None,
        sample_hidden_states: torch.Tensor = None,
    ) -> list[list[int]] | None:
        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        else:
            if self.speculative_config.method in ("suffix", "ngram"):
                draft_token_ids = self.drafter.generate_token_ids(
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    positions,
                    num_scheduled_tokens,
                    hidden_states,
                    aux_hidden_states,
                )
            elif isinstance(self.drafter, MedusaProposer):
                draft_token_ids = self.drafter.generate_token_ids(
                    valid_sampled_token_ids, sampling_metadata, spec_decode_metadata, sample_hidden_states
                )
            elif self.speculative_config.use_eagle():
                common_attn_metadata = spec_decode_common_attn_metadata
                sampled_token_ids = valid_sampled_token_ids

                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    # When padded-batch is disabled, the sampled_token_ids should be
                    # the cpu-side list[list[int]] of valid sampled tokens for each
                    # request, with invalid requests having empty lists.
                    assert isinstance(sampled_token_ids, list), (
                        "sampled_token_ids should be a python list whenpadded-batch is disabled."
                    )
                    assert self.drafter is not None
                    next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                        sampled_token_ids, self.requests, self.input_batch, scheduler_output.num_scheduled_tokens
                    )
                else:
                    # When using padded-batch, the sampled_token_ids should be
                    # the gpu tensor of sampled tokens for each request, of shape
                    # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                    # value -1.
                    assert isinstance(sampled_token_ids, torch.Tensor), (
                        "sampled_token_ids should be a torch.Tensor whenpadded-batch is enabled."
                    )
                    assert self.drafter is not None
                    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

                req_scheduled_tokens = scheduler_output.num_scheduled_tokens
                if self.use_cp:
                    long_seq_metadata = self.long_seq_metadata  # type: ignore
                    input_ids_pcp_full = self.pcp_manager.input_ids_pcp_full.gpu
                    query_start_loc_pcp_full = self.pcp_manager.query_start_loc_pcp_full.gpu
                    query_start_loc_pcp_full_cpu = self.pcp_manager.query_start_loc_pcp_full.cpu
                    num_reqs = self.input_batch.num_reqs
                    num_prefill_reqs = self.pcp_manager.num_prefill_reqs
                    num_decode_reqs = self.pcp_manager.num_decode_reqs
                else:
                    long_seq_metadata = None  # type: ignore
                    num_prefill_reqs = 0
                    num_decode_reqs = 0
                if spec_decode_metadata is None:
                    # update pcp related params
                    if self.pcp_size > 1:
                        token_indices_to_sample = query_start_loc_pcp_full[1 : num_reqs + 1] - 1
                        target_token_ids = input_ids_pcp_full[:num_scheduled_tokens]
                        target_positions = self._get_positions(num_scheduled_tokens)
                        target_hidden_states = hidden_states
                    else:
                        token_indices_to_sample = None
                        # input_ids can be None for multimodal models.
                        target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                        target_positions = self._get_positions(num_scheduled_tokens)
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat(
                                [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                            )
                        else:
                            target_hidden_states = hidden_states[:num_scheduled_tokens]
                else:
                    if self.pcp_size > 1:
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc_cpu[: num_reqs + 1] = query_start_loc_pcp_full_cpu[
                            : num_reqs + 1
                        ]
                        assert common_attn_metadata is not None
                        common_attn_metadata.query_start_loc[: num_reqs + 1] = query_start_loc_pcp_full[: num_reqs + 1]
                    if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                        # NOTE: Currently, MTP-fullgraph is incompatibility with pcp
                        token_indices_to_sample = None
                        assert self.drafter is not None
                        common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                            common_attn_metadata, sampled_token_ids, spec_decode_metadata.num_draft_tokens
                        )
                    else:
                        assert self.drafter is not None
                        common_attn_metadata, token_indices, token_indices_to_sample = (
                            self.drafter.prepare_inputs_padded(
                                common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count
                            )
                        )
                    if self.pcp_size > 1:
                        target_token_ids = input_ids_pcp_full[token_indices]
                        target_positions = positions
                        target_hidden_states = hidden_states
                    else:
                        target_token_ids = self.input_ids.gpu[token_indices]
                        target_positions = self._get_positions(token_indices)
                        if self.use_aux_hidden_state_outputs:
                            target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)
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
                raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")

        return draft_token_ids

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if (
            self.use_async_scheduling and self.num_spec_tokens and self._draft_token_ids is None  # type: ignore[has-type]
        ):
            scheduler_output = deepcopy(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        return make_empty_encoder_model_runner_output(scheduler_output)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                if self.dynamic_eplb:
                    self.eplb_updator.forward_before()

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size == 1  # TODO(lxs): fix this
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(num_tokens_padded, num_reqs_padded, num_reqs)

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

            if self.dynamic_eplb:
                self.eplb_updator.take_update_info_from_eplb_process()

            # update global cos, sin
            update_cos_sin(positions)

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
            )
            self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
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
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        if self.need_accepted_tokens:
            self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                attn_metadata,
                aux_hidden_states,
                sample_hidden_states,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

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

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch_for_eagle = (
                    self.speculative_config
                    and self.speculative_config.use_eagle()
                    and not self.speculative_config.disable_padded_drafter_batch
                )
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

        if self.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.save_captured_experts(indices=self.cpu_slot_mapping)
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            pooler_output=[],
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )

        if self.dynamic_eplb:
            self.eplb_updator.forward_end()

        if self.debugger is not None:
            self.debugger.stop()
            self.debugger.step()
        if not self.use_async_scheduling:
            return model_runner_output
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
                logits = logits[: self.input_batch.num_reqs]
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[: len(spec_decode_metadata.logits_indices)]
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
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
        discard_sampled_tokens_req_indices = self.discard_request_indices.np[: self.num_discarded_requests]
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
                    logprobs_tensors=logprobs_tensors,
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
                req_id: i for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
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
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (
            logprobs_tensors.tolists(cu_num_tokens)
            if not self.use_async_scheduling and logprobs_tensors is not None
            else None
        )

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
        return [NPUModelRunner._all_gather_hidden_states(hidden_states) for hidden_states in hidden_states_list]

    # all-gather hidden-states in last layer with aux-hidden-states in sp scene
    @staticmethod
    def _all_gather_hidden_states_and_aux(hidden_states):
        if isinstance(hidden_states, tuple):
            return (
                NPUModelRunner._all_gather_hidden_states(hidden_states[0]),
                NPUModelRunner._all_gather_hidden_states_list(hidden_states[1]),
            )
        return NPUModelRunner._all_gather_hidden_states(hidden_states)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        assert self.model is not None
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        forward_context = get_forward_context()
        assert forward_context is not None
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not self.use_sparse
        ):
            assert positions is not None
            update_full_graph_params(
                self.attn_backend,
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                self.speculative_config,
                positions.shape[0],
            )
        if get_forward_context().flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if enable_sp(self.vllm_config):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _sync_batch_across_dp(
        self,
        num_tokens_padded: int | None = None,
        cudagraph_mode: int = 0,
    ) -> tuple[bool, torch.Tensor | None, int]:
        """
        Coordinates amongst all DP ranks to determine if and how the full batch
        should be split into microbatches.

        Args:
            num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
                TP, etc)
            cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL)

        Returns: tuple[
            ubatch_slices: if this is set then all DP ranks have agreed to
            microbatch
            num_tokens_after_padding: A tensor containing the total number of
            tokens per-microbatch for each DP rank including padding. Will be
            padded up to the max value across all DP ranks when allow_dp_padding
            is True.
            synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
        ]

        """

        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.

        if self.dp_size == 1:
            return False, None, cudagraph_mode

        if self._skip_all_reduce_across_dp_group():
            num_tokens_after_padding = torch.tensor([num_tokens_padded] * self.dp_size, device="cpu", dtype=torch.int32)
            return False, num_tokens_after_padding, cudagraph_mode

        tensor = torch.zeros(2, self.dp_size, device="cpu", dtype=torch.int32)
        tensor[0][self.dp_rank] = num_tokens_padded
        tensor[1][self.dp_rank] = cudagraph_mode
        dist.all_reduce(tensor, group=get_dp_group().cpu_group)

        num_tokens_across_dp = tensor[0, :]
        max_num_tokens = int(num_tokens_across_dp.max().item())
        num_tokens_after_padding = torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
        # Synchronize cudagraph_mode across ranks (take min)
        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
        return False, num_tokens_after_padding, synced_cudagraph_mode

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = False,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None]:
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        uniform_decode = (
            (
                (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        has_lora = len(self.input_batch.lora_id_to_lora_request) > 0 if force_has_lora is None else force_has_lora

        # ruff: noqa: E731
        dispatch_cudagraph = (
            lambda num_tokens, disable_full: self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                disable_full=disable_full,
            )
            if not force_eager
            else (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))
        )
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(num_tokens_padded, use_cascade_attn or has_encoder_output)
        num_tokens_padded = batch_descriptor.num_tokens
        if enable_sp(self.vllm_config):
            assert batch_descriptor.num_tokens % self.vllm_config.parallel_config.tensor_parallel_size == 0, (
                "Sequence parallelism requires num_tokens to be a multiple of tensor parallel size"
            )
        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            _, num_tokens_across_dp, synced_cudagraph_mode = self._sync_batch_across_dp(
                num_tokens_padded=num_tokens_padded,
                cudagraph_mode=cudagraph_mode.value,
            )

            # Extract DP padding if there is any
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    disable_full=synced_cudagraph_mode <= CUDAGraphMode.PIECEWISE.value,
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        num_scheduled_tokens_np: np.ndarray | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]
        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()
        if use_spec_decode and self.need_accepted_tokens:
            self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_pcp_metadata(block_table_tensor):
            if not self.use_cp:
                return None, block_table_tensor
            return self.pcp_manager.generate_pcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
            )

        def _get_block_table_and_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            maybe_pcp_full_tokens = (
                num_tokens_padded
                if self.pcp_size == 1
                else num_tokens * self.pcp_size - sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs])
            )
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:maybe_pcp_full_tokens]
                maybe_num_reqs_padded = num_reqs_padded * self.decode_token_per_req if self.use_cp else num_reqs_padded
                blk_table_tensor = blk_table.get_device_tensor()[:maybe_num_reqs_padded]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                if self.pcp_size == 1:
                    slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                    blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if self.pcp_size > 1:
                slot_mapping = self.pcp_manager.get_padded_slot_mapping(
                    num_tokens,
                    num_tokens_padded,
                    slot_mapping,
                )
            if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
                self.cpu_slot_mapping = slot_mapping.cpu().numpy()
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        self.long_seq_metadata, block_table_gid_0 = _get_pcp_metadata(block_table_gid_0)

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            # TODO
            seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            # TODO
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions.gpu,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            prefill_context_parallel_metadata=self.long_seq_metadata,
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid] if cascade_attn_prefix_lens else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                patch_torch_npu_argsort()
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            cm = copy(cm_base)  # shallow copy
            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(kv_cache_gid)
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)
        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
        return attn_metadata, spec_decode_common_attn_metadata

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
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
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            raise NotImplementedError("create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it")
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        if not is_profile and self.dynamic_eplb:
            self.eplb_updator.forward_before()

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        self.query_lens = torch.from_numpy(num_scheduled_tokens)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        _cudagraph_mode, batch_desc, _, num_tokens_across_dp, _ = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=activate_lora,
        )
        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
            )
            if self.speculative_config:
                self.pcp_manager.query_lens_pcp_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.pcp_manager.query_lens_pcp_full.copy_to_gpu()
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        if num_tokens_across_dp is not None and num_tokens_padded != num_tokens:
            # pad is needed if the pad of `num_tokens` is triggered inside CudagraphDispatcher
            num_tokens_across_dp[:] = num_tokens_padded
            num_scheduled_tokens = num_scheduled_tokens.repeat(num_reqs_padded)
        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                raise NotImplementedError(
                    "create_mixed_batch is used for warmup deepgemm, vllm-ascend does not need it"
                )
            self.attn_state = AscendAttentionState.DecodeOnly
            if self.speculative_config and self.speculative_config.method == "mtp":
                # `AscendAttentionState.SpecDecoding` is only designed for mla
                if self.vllm_config.model_config.use_mla:
                    self.attn_state = AscendAttentionState.SpecDecoding
                else:
                    self.attn_state = AscendAttentionState.ChunkedPrefill
            # The reason why we use a fixed seq_len rather than max_query_len is that
            # _npu_paged_attention_get_workspace only returns max workspace with specific
            # seq_lens. We use this seq_len only when capturing graph, and still use max_query_len
            # in inference. This will be removed once npu_fused_infer_attention_score
            # outperforms _npu_paged_attention on all cases.
            seq_lens = (
                SEQ_LEN_WITH_MAX_PA_WORKSPACE
                if is_graph_capturing and using_paged_attention(num_tokens, self.vllm_config)
                else max_query_len
            )  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs_padded] = seq_lens
            self.seq_lens.np[num_reqs_padded:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            num_reqs_padded = self._pad_query_start_loc_for_fia(num_tokens_padded, num_reqs_padded, num_reqs)

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
                num_scheduled_tokens_np=num_scheduled_tokens,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.is_multimodal_model and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:
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
                # When PP and flashcomm1 are enabled, during dummy_run the estimated space should divide num_tokens by
                # tp_size; otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading
                # to incorrect memory estimation and potentially causing OOM.
                intermediate_tokens = num_tokens_padded
                if enable_flash_comm_v1():
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_tokens = (num_tokens_padded + tp_size - 1) // tp_size
                if self.intermediate_tensors is None:
                    max_actual_tokens = self.max_num_tokens
                    if enable_sp():
                        max_actual_tokens = (self.max_num_tokens + tp_size - 1) // tp_size
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=max_actual_tokens, dtype=self.dtype, device=self.device
                    )
                intermediate_tensors = IntermediateTensors(
                    {k: v[:intermediate_tokens] for k, v in self.intermediate_tensors.items()}
                )

            need_dummy_logits = not is_profile and lmhead_tp_enable()
            max_num_reqs_across_dp = max_num_reqs * self.uniform_decode_query_len
            dummy_indices = torch.zeros(max_num_reqs_across_dp, dtype=torch.int32)

            def dummy_compute_logits(hidden_states):
                if not need_dummy_logits:
                    return None
                return self.model.compute_logits(hidden_states[dummy_indices])

            def dummy_drafter_compute_logits(hidden_states):
                if not need_dummy_logits or self.drafter is None:
                    return
                if hasattr(self.drafter, "model") and hasattr(self.drafter.model, "compute_logits"):
                    return self.drafter.model.compute_logits(hidden_states[dummy_indices])

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                in_profile_run=is_profile,
                num_actual_tokens=num_tokens_padded,
                aclgraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                model_instance=self.model,
            ):
                outputs = self._model_forward(
                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            dummy_compute_logits(hidden_states)

            if self.drafter:
                self.drafter.dummy_run(
                    num_tokens=num_tokens_padded,
                    with_prefill=with_prefill,
                    num_reqs=num_reqs_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    aclgraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    dummy_compute_logits=dummy_drafter_compute_logits,
                    in_graph_capturing=not force_attention,
                    is_profile=is_profile,
                )
            if is_profile and self.dynamic_eplb:
                self.model.clear_all_moe_loads()
            if self.dynamic_eplb:
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
        num_scheduled_tokens_list[-1] += self.max_num_tokens % self.max_num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        # TODO: need to rum a dummy sampler for generate task
        hidden_states = hidden_states[logit_indices]
        output = self.model.compute_logits(hidden_states)
        return output

    def profile_run(self) -> None:
        self.eplb_warmup()
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and select_moe_comm_method(
            mc2_tokens_capacity, self.vllm_config
        ) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity, with_prefill=True, is_profile=True)
        origin_max_num_tokens = self.max_num_tokens
        # in the pcp scenario, the split sequence needs to be used for profile run
        # TODO: after the vllm pcp function is launched, this logic needs to be brought up to the community
        if self.pcp_size > 1:
            self.max_num_tokens = math.ceil(self.max_num_tokens / (self.pcp_size * 2)) * 2
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
                model_register(self.model)
            if self.drafter:
                logger.info("Loading drafter model...")
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)
                if self.use_aux_hidden_state_outputs:
                    self.model.set_aux_hidden_state_layers(self.model.get_eagle3_aux_hidden_state_layers())

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        # wrap the model with full graph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)

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
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # NOTE: Currently, we determine whether we need `num_accepted_tokens` through `MambaSpec`.
        self.need_accepted_tokens = any(
            [isinstance(attn_group[0].kv_cache_spec, MambaSpec) for attn_group in self.attn_groups]
        )

        self.may_reinitialize_input_batch(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def _align_memory(self, tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset) :]

    def initialize_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
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
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        from vllm.v1.worker.utils import bind_kv_cache

        num_attn_module = 2 if self.model_config.hf_text_config.model_type == "longcat_flash" else 1
        bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)
        return kv_caches

    def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        NOTE: To support prefill disaggregation, we need to split kvcache tensor into
        k_cache and v cache, and the addr of both are aligned by 2M

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
            dict[str, tuple(torch.Tensor, torch.Tensor)] A map between layer names
            to their corresponding memory buffer for K cache and V cache.
        """
        # init kv cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor | torch.Tensor | None] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # TODO: REFACTOR ME to sharing hybrid cache
            for idx in range(len(kv_cache_tensor.shared_by)):
                layer_name = kv_cache_tensor.shared_by[idx]
                if "linear_attn" in layer_name and layer_name not in kv_cache_raw_tensors:
                    # for mamba linear attention
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                        tensor = self._align_memory(tensor, alignment)[: kv_cache_tensor.size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "linear_attn" in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = tensor
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors:
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    if self.model_config.use_mla:
                        head_size = (
                            self.model_config.hf_text_config.qk_rope_head_dim
                            + self.model_config.hf_text_config.kv_lora_rank
                        )

                    dsa_k_cache_factor = None
                    dsa_k_cache_size = None
                    if not self.model_config.use_mla:
                        # for non-mla model, use FullAttentionSpec
                        k_tensor_split_factor = 2
                        v_tensor_split_factor = 2
                    elif self.use_sparse:
                        # for deepseek v3.2, we split the kv cache according to the corresponding ratio
                        sparse_sum_head_size = sum(self._get_sparse_kv_cache_ratio())
                        k_tensor_split_factor, v_tensor_split_factor, dsa_k_cache_factor = [  # type: ignore
                            sparse_sum_head_size / ratio for ratio in self._get_sparse_kv_cache_ratio()
                        ]
                        dsa_k_cache_size = int(kv_cache_tensor.size // dsa_k_cache_factor)
                    else:
                        # for other deepseek models, use MLAAttentionSpec
                        k_tensor_split_factor = head_size / self.model_config.hf_text_config.kv_lora_rank
                        v_tensor_split_factor = head_size / self.model_config.hf_text_config.qk_rope_head_dim

                    k_tensor_size = int(kv_cache_tensor.size // k_tensor_split_factor)
                    v_tensor_size = int(kv_cache_tensor.size // v_tensor_split_factor)

                    # for other attentions, e.g., self_attn, sliding window attn
                    if self.vllm_config.kv_transfer_config is None:
                        k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=self.device)
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None:
                            dsa_k_cache_tensor = torch.zeros(dsa_k_cache_size, dtype=torch.int8, device=self.device)
                    else:
                        k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=self.device)
                        k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
                        v_tensor = self._align_memory(v_tensor, alignment)[:v_tensor_size]
                        #### k cache: for deepseek sparse attention
                        if dsa_k_cache_factor is not None and dsa_k_cache_size is not None:
                            dsa_k_cache_tensor = torch.zeros(
                                dsa_k_cache_size + alignment, dtype=torch.int8, device=self.device
                            )
                            dsa_k_cache_tensor = self._align_memory(dsa_k_cache_tensor, alignment)[:dsa_k_cache_size]

                    for layer_name_inner in kv_cache_tensor.shared_by:
                        # shared the kvcache between the self_attn specs in the same group
                        if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                            kv_cache_raw_tensors[layer_name_inner] = (
                                (k_tensor, v_tensor)
                                if not self.use_sparse
                                else (k_tensor, v_tensor, dsa_k_cache_tensor)
                            )

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

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
        kv_caches: dict[str, torch.Tensor] = {}
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
                            layer_name
                        ]
                        assert raw_dsa_k_tensor is not None
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel() + raw_dsa_k_tensor.numel()
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
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

                    if hasattr(attn_backend, "get_supported_kernel_block_sizes") and self.use_hybrid_blocks:
                        block_size = attn_backend.get_supported_kernel_block_sizes()[0]

                        block_size_chunk = kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size
                        )
                    dtype = kv_cache_spec.dtype
                    if not self.model_config.use_mla:
                        k_shape = kv_cache_shape[1:]
                        v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_shape = [
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            self.model_config.hf_text_config.kv_lora_rank,
                        ]
                        v_shape = [
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            self.model_config.hf_text_config.qk_rope_head_dim,
                        ]
                    k_cache = raw_k_tensor.view(dtype).view(k_shape)
                    v_cache = raw_v_tensor.view(dtype).view(v_shape)

                    if self.use_sparse and raw_dsa_k_tensor is not None:
                        index_head_dim = self._get_sparse_kv_cache_ratio()[-1]
                        dsa_k_cache_shape = (
                            num_blocks,
                            kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads,
                            index_head_dim,
                        )
                        dsa_k_cache = raw_dsa_k_tensor.view(dtype).view(dsa_k_cache_shape)
                        kv_caches[layer_name] = (k_cache, v_cache, dsa_k_cache)
                    else:
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
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
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += torch.prod(torch.tensor(target_shape)).item()
                        tensor = raw_tensor.view(dtype)[start_idx:target_idx].view(target_shape)
                        start_idx = target_idx
                        state_tensors.append(tensor)
                    kv_caches[layer_name] = state_tensors
                else:
                    raise ValueError("Unknown KV cache spec type.")

        return kv_caches

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
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
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        kernel_block_sizes = []
        for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
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
                    supported_sizes = backend.get_supported_kernel_block_sizes()
                    # If no specific sizes supported, use cache config
                    # block_size
                    kernel_block_size_list = supported_sizes if supported_sizes else [self.cache_config.block_size]
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
        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [[self.cache_config.block_size]]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
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
                    if self.vllm_config.speculative_config
                    else 0
                ),
                kernel_block_sizes=kernel_block_sizes,
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names)
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
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend, layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionBackend, list[str]], kv_cache_group_id: int
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_metadata_builders = []
                attn_metadata_builders.append(
                    attn_backend.get_builder_cls()(
                        kv_cache_spec,
                        layer_names,
                        self.vllm_config,
                        self.device,
                    )
                )
                attn_group = AttentionGroup(
                    attn_backend, layer_names, kv_cache_spec, kv_cache_group_id, attn_metadata_builders
                )
                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        self._check_and_update_cudagraph_mode(attention_backend_list, kv_cache_config.kv_cache_groups)

        for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            attn_backends = get_attn_backends_for_group(  # type: ignore
                kv_cache_group_spec
            )
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
            if hasattr(attn_metadata_builder_i, "reorder_batch_threshold"):  # noqa
                # check that if any backends reorder batches; that the reordering
                # is compatible (e.g., decode threshold is the same)
                reorder_batch_threshold_i = attn_metadata_builder_i.reorder_batch_threshold
                if reorder_batch_threshold_i is not None:  # noqa
                    if self.reorder_batch_threshold is not None:
                        if reorder_batch_threshold_i != self.reorder_batch_threshold:
                            raise ValueError(
                                f"Attention backend reorders decodes with "
                                f"threshold {reorder_batch_threshold_i} but other "
                                f"backend uses threshold "
                                f"{self.reorder_batch_threshold}"
                            )
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

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
        # ordering expected by graph parameter update logic in attention backends.
        mamba_layers: dict[str, MambaBase] = {}
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention):
                if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec

            elif isinstance(attn_module, MLAAttention):
                if self.use_sparse:
                    # TODO(cmq): This is a hack way to fix deepseek kvcache when
                    # using DSA. Fix the spec in vLLM is the final way.
                    sparse_sum_head_size = sum(self._get_sparse_kv_cache_ratio())
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=self.block_size,
                        num_kv_heads=1,
                        head_size=sparse_sum_head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    )
                elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec

            elif isinstance(attn_module, MambaBase):
                mamba_layers[layer_name] = attn_module

        if len(mamba_layers) > 0:
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError("Prefix caching is not supported for Mamba yet.")
            for layer_name, mamba_module in mamba_layers.items():
                if spec := mamba_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _get_sparse_kv_cache_ratio(self) -> list[int]:
        # TODO:If C8 is supported, we need to consider the number of bytes occupied by different dtypes
        #  when calculating the ratio，for example:
        #  [kv_lora_rank * torch.int8.itemsize, qk_rope_head_dim * torch.bfloat16.itemsize, ...]
        return [
            self.model_config.hf_text_config.kv_lora_rank,
            self.model_config.hf_text_config.qk_rope_head_dim,
            self.model_config.hf_text_config.index_head_dim,
        ]

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(self.cudagraph_batch_sizes)
            if self.speculative_config:
                set_draft_graph_params(self.cudagraph_batch_sizes)

    def capture_model(self) -> None:
        gpu_model_runner_cls = next((cls for cls in self.__class__.__mro__ if cls.__name__ == "GPUModelRunner"), None)
        if gpu_model_runner_cls is None:
            raise TypeError("Could not find GPUModelRunner in the MRO. The class hierarchy may have changed.")
        parent_module_name = gpu_model_runner_cls.__module__
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
            GPUModelRunner.capture_model(self)

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

            mm_data = getattr(req, "multimodal_data", None)
            if not mm_data:
                continue

            for field in self.multimodal_cpu_fields:
                if field in mm_data:
                    tensor = mm_data[field]
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                        mm_data[field] = tensor.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())


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
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        yield
    finally:
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
