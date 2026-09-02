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

import logging
import math
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation import breakable_cudagraph
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.model_executor.offloader.base import get_offloader, set_offloader
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import PIN_MEMORY, get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    RoutedExpertsLists,
    RoutedExpertsTensors,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import copy_num_valid_draft_tokens
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup, select_common_block_size

# yapf: enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    get_sfa_qsfa_packed_head_dim,
    using_paged_attention,
)

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    set_draft_graph_params,
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    allocate_kv_cache_tensors_for_sparse_kv_offload,
    allocate_kv_offload_topk_profile_buffers,
    init_sparse_kv_offload_manager,
    reshape_kv_cache_tensors_for_sparse_kv_offload,
    update_sparse_kv_offload_metadata,
)
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.model_executor.offloader import create_offloader
from vllm_ascend.ops.fused_moe.force_eplb import build_force_eplb_topk
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.ops.triton.spec_decode.ngram import triton_ngram_spec_decode
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer
from vllm_ascend.spec_decode.utils import (
    correct_optimistic_seq_lens_cpu,
    update_num_computed_tokens_for_batch_change,
)
from vllm_ascend.utils import (
    calc_split_factor,
    check_gdn_layer,
    embedding_tp_enable,
    enable_dsa_cp,
    enable_sfa,
    enable_sfa_dcp_replicated_indexer,
    enable_sp,
    get_c_env,
    get_kv_cache_tensor_layers,
    global_stream,
    is_hidden_state_cache_spec,
    is_score_encoder_cache_manager,
    kv_cache_spec_uses_sparse_sfa_c8,
    lmhead_tp_enable,
    oproj_tp_enable,
    set_potential_max_tokens,
    should_skip_allreduce_across_dp_group,
    vllm_version_is,
    weak_ref_tensor,
    weak_ref_tensors,
)
from vllm_ascend.worker.dcp_utils import DCPAsyncSpecDecodeRebuildResult, DCPManager
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.utils import AscendKVBlockZeroer

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)

from vllm.model_executor.models.interfaces import supports_multimodal_pruning

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


from vllm.model_executor.layers.attention import Attention, MLAAttention

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    AscendSlidingWindowMLASpec,
)

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
    batch_desc: BatchDescriptor


class NPUModelRunner(GPUModelRunner):
    # vLLM #51718 describes compatible KV cache groups as views over one
    # standardized backing allocation. The default runner preserves that
    # contract for its layer/block-compact Attention+Mamba path.
    supports_standardized_shared_kv_backing = True

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Must be set before super().__init__() because parent init may call
        # _allocate_kv_cache_tensors which accesses self.use_compress.
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None) if model_config else None
        self.use_compress = (
            hf_config is not None and hasattr(hf_config, "compress_ratios")
        )

        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        self.pin_memory = PIN_MEMORY

        set_offloader(create_offloader(self.offload_config))

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 2,  # type: ignore[has-type]
            dtype=torch.int32,
        )
        self.group_len = self._make_buffer(
            vllm_config.scheduler_config.max_num_batched_tokens , dtype=torch.int32
        )        
        self.group_key_idx = self._make_buffer(
           vllm_config.scheduler_config.max_num_batched_tokens , dtype=torch.int32
        )        
        self.group_key_cache_idx = self._make_buffer(
            vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.int32
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        self._has_gdn = check_gdn_layer(vllm_config)
        self._has_sinks = False
        if self._has_gdn:
            self.gdn_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1,  # type: ignore[has-type]
                dtype=torch.int32,
            )

        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        self.use_score_encoder_cache = is_score_encoder_cache_manager(self.vllm_config)

        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            self._debugger_started = False
            if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
                from msprobe.pytorch import PrecisionDebugger

                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                try:
                    from msprobe.pytorch import AclGraphDumper
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to import AclGraphDumper from msprobe. "
                        "Please install/rebuild msprobe with aclgraph_dump enabled."
                    ) from exc

                self.debugger = AclGraphDumper(dump_cfg)
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = enable_sfa(vllm_config)
        # dsa c8
        self.enable_sparse_sfa_c8 = self.ascend_config.enable_sparse_sfa_c8
        self.enable_sparse_li_c8 = self.ascend_config.enable_sparse_li_c8
        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.c8_k_cache_dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16

        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None
            and self.model_config.is_mm_prefix_lm,
        )

        # reinit valid_sampled_token_count_cpu with torch.int64 dtype
        if self.use_async_scheduling and self.num_spec_tokens:
            self.valid_sampled_token_count_cpu = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
        max_buffer_num_tokens = self.max_num_tokens
        if self.dcp_size > 1:
            self.dcp_manager = DCPManager(
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
                self.use_sparse,
            )
        self.sfa_dcp_replicated_indexer_size = 1
        if enable_sfa_dcp_replicated_indexer():
            self.sfa_dcp_replicated_indexer_size = self.dcp_size

        # Create a CPU numpy buffer for positions computation when
        # self.positions is a plain tensor (non-CpuGpuBuffer case).
        self._positions_cpu_buf = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._positions_np_buf = self._positions_cpu_buf.numpy()
        # For deepseek-v4 use only
        self._dsa_positions_cpu_buf = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._dsa_positions_np_buf = self._dsa_positions_cpu_buf.numpy()

        self.use_eagle = (
            vllm_config.speculative_config.use_eagle()
            if vllm_config.speculative_config
            else None
        )
        # When True, run update_full_graph_params before self.model (ENPU / graph capture order).
        # Internal / non-public toggle: read C getenv ``ENPU_ENABLE`` from enpu code (not in envs.py).
        _enpu = get_c_env("ENPU_ENABLE")
        self.enable_enpu = _enpu is not None and _enpu.lower() == "true"

        self._set_up_drafter()

        # Backends that consume CPU seq_lens (AscendAttentionBackend,
        # AscendMLABackend, and DSV4 compressed attention metadata) need
        # ``optimistic_seq_lens_cpu`` to match the corrected GPU seq_lens
        # in async spec decode mode; others (SFA, GDN, etc.) do not.
        self._needs_seq_lens_cpu_sync = self.use_compress or issubclass(
            self.attn_backend, (AscendAttentionBackend, AscendMLABackend)
        )

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        # Compute potential_max_tokens once here; it is reused by the skip-allreduce
        # decision and the o_proj static-exchange buffer sizing (see get_potential_max_tokens).
        set_potential_max_tokens(vllm_config)
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        eplb_config = self.ascend_config.eplb_config
        self.dynamic_eplb = eplb_config.dynamic_eplb
        self.eplb_enable = self.dynamic_eplb or (eplb_config.expert_map_path is not None)
        if self.dynamic_eplb:
            self.is_eplb_warmuped = False
            self.policy_type = eplb_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
            self.eplb_process = EplbProcess(
                shared_dict=self.shared_dict,
                policy_type=self.policy_type,
                enable_d2d=True,
                tp_size=self.parallel_config.tensor_parallel_size,
            )
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
            # In pd colocation scenarios, we find that prefill/decode requests result in different
            # expert workloads. To reduce expert imbalance more effectively, we can coolect eplb
            # heat exclusively on a single stage rather than both prefill/decode.
            self.eplb_heat_collection_stage = eplb_config.eplb_heat_collection_stage
            # Currently, we set the maximum of tokens in decode stage as the threshold to distinguish
            # prefill with decode.
            self.eplb_pd_thresholds = self.max_num_reqs * self.uniform_decode_query_len
            self.eplb_heat_collection_status = True

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
            logitsprocs_need_output_token_ids=bool(
                self.vllm_config.model_config.logits_processors
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            reasoning_config = getattr(self.vllm_config, "reasoning_config", None),
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
        self.query_lens: torch.Tensor | None = None
        self.sampling_done_event: torch.npu.Event | None = None
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None

        self.update_stream: torch.npu.Stream | None = None

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        else:
            self.cudagraph_batch_sizes = []
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_bufs: Any | None = None
        self._mamba_copy_bufs: Any | None = None
        # Score encoder cache state.
        self.tmp_encoder_cache: dict[str, torch.Tensor] = {}
        self.cpu_encoder_cache: dict[str, torch.Tensor] = {}
        self.cached: dict[str, set[str]] = {}
        self._pending_encoder_cache_copies: deque[
            tuple[torch.Tensor, torch.npu.Event]
        ] = deque()

        self.sparse_kv_offload_config = self.ascend_config.sparse_kv_offload_config
        self.sparse_kv_offload_enabled = self.sparse_kv_offload_config.enabled
        self.sparse_kv_offload_manager = None
        self.tp_rank = get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0

        # Per-request metadata consumed by the Sparse KV offload resident LRU.
        self._offload_req_ids_tensor = None
        self._offload_token_to_req = None
        if self.sparse_kv_offload_enabled:
            self._offload_req_ids_tensor = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
            self._offload_token_to_req = self._make_buffer(self.max_num_tokens, dtype=torch.int32)

    @property
    def use_dcp(self) -> bool:
        return self.dcp_size > 1

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def reset_encoder_cache(self) -> None:
        """Clear device and score-based encoder cache state."""
        if (
            self.tmp_encoder_cache
            or self.cpu_encoder_cache
            or self._pending_encoder_cache_copies
        ):
            self._sync_device()

        super().reset_encoder_cache()
        self.tmp_encoder_cache.clear()
        self.cpu_encoder_cache.clear()
        self.cached.clear()
        self._pending_encoder_cache_copies.clear()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: (
            AscendNgramProposer
            | AscendNgramProposerNPU
            | AscendEagleProposer
            | AscendStep3p5MTPProposer
            | AscendDraftModelProposer
            | AscendDflashProposer
            | AscendDSparkProposer
            | AscendSuffixDecodingProposer
            | AscendMedusaProposer
            | AscendExtractHiddenStatesProposer
            | None
        ) = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, AscendEagleProposer)
                    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
                elif self.speculative_config.method == "extract_hidden_states":
                    assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
                    self.use_aux_hidden_state_outputs = True
                elif self.speculative_config.use_dspark():
                    assert isinstance(self.drafter, AscendDSparkProposer)
                    self.use_aux_hidden_state_outputs = True
                self.rejection_sampler = AscendRejectionSampler(
                    self.sampler, self.speculative_config, self.device
                )
        self.discard_request_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _eagle3_uses_aux_hidden_state(self) -> bool:
        if self.speculative_config is None or self.speculative_config.method != "eagle3":
            return False

        draft_model_config = self.speculative_config.draft_model_config
        if draft_model_config is None:
            return True

        eagle_config = getattr(draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is None:
            return True
        return eagle_config.get("use_aux_hidden_state", True)

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        layer_ids = super()._get_eagle3_aux_layers_from_config()
        if layer_ids:
            return layer_ids
        if self.speculative_config.use_dspark():
            hf_config = self.speculative_config.draft_model_config.hf_config
            # deepseek v4 dspark
            dspark_layer_ids = getattr(hf_config, "dspark_target_layer_ids", None)
            if dspark_layer_ids:
                return tuple(i + 1 for i in dspark_layer_ids)
            # gqa backend dspark
            dspark_layer_ids = getattr(hf_config, "target_layer_ids", None)
            if dspark_layer_ids:
                return tuple(i + 1 for i in dspark_layer_ids)
        return None

    def _draft_uses_qwen3_gqa_dspark(self) -> bool:
        """Return whether the draft expects Kimi's materialized residual."""
        if self.speculative_config is None or not self.speculative_config.use_dspark():
            return False
        draft_model_config = self.speculative_config.draft_model_config
        if draft_model_config is None:
            return False
        hf_config = draft_model_config.hf_config
        architectures = getattr(hf_config, "architectures", ()) or ()
        return getattr(hf_config, "model_type", None) == "qwen3" and "Qwen3DSparkModel" in architectures

    def _use_aclgraph(self) -> bool:
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                or breakable_cudagraph.is_breakable_cudagraph_enabled()
            )
            and not self.model_config.enforce_eager
        )

    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, cudagraph_mode

        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, cudagraph_mode

        packed_tensor = torch.zeros(2, self.dp_size, device="cpu", dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[0, :]
        max_tokens_across_dp = int(num_tokens_across_dp.max().item())
        synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor))

        # Create a tensor for num_tokens_after_padding
        if allow_dp_padding or is_draft_model:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
            )
        else:
            num_tokens_after_padding = num_tokens_across_dp.cpu()

        return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, (ACLGraphWrapper, BreakableACLGraphWrapper)):
            return self.model.unwrap()
        return self.model

    def _is_pd_prefill_worker(self) -> bool:
        return self.is_kv_producer and not self.is_kv_consumer

    def _apply_pp_sampled_tokens_from_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        pp = get_pp_group()
        if (
            not self.use_async_scheduling
            or pp.is_last_rank
            or self._is_pd_prefill_worker()
        ):
            return

        self.input_batch.prev_sampled_token_ids = None
        self.input_batch.prev_req_id_to_index = {}

        req_data = scheduler_output.scheduled_cached_reqs
        new_token_ids = req_data.new_token_ids
        if not new_token_ids:
            return

        num_prev_reqs = self.input_batch.num_reqs
        if num_prev_reqs == 0:
            return

        discard_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_prev_reqs]
        )[0]
        discarded = set(discard_req_indices)
        prev_req_indices = {
            req_id: req_index
            for req_index, req_id in enumerate(
                self.input_batch.req_ids[:num_prev_reqs]
            )
            if req_index not in discarded
        }
        prev_req_id_to_index: dict[str, int] = {}
        prev_sampled_token_ids = [PLACEHOLDER_TOKEN_ID] * num_prev_reqs

        for req_index, req_id in enumerate(req_data.req_ids):
            if req_index >= len(new_token_ids):
                break
            token_ids = new_token_ids[req_index]
            if not token_ids or req_data.num_output_tokens[req_index] <= 0:
                continue
            prev_req_index = prev_req_indices.get(req_id)
            if prev_req_index is None:
                continue
            prev_req_id_to_index[req_id] = prev_req_index
            prev_sampled_token_ids[prev_req_index] = token_ids[-1]
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(PLACEHOLDER_TOKEN_ID)
            pos = self.input_batch.num_tokens_no_spec[prev_req_index]
            self.input_batch.is_token_ids[prev_req_index, pos] = True
            self.input_batch.num_tokens_no_spec[prev_req_index] = pos + 1

        if not prev_req_id_to_index:
            return

        self.input_batch.prev_req_id_to_index = prev_req_id_to_index
        self.input_batch.prev_sampled_token_ids = torch.tensor(
            prev_sampled_token_ids,
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(1)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        # Temporary rewind guard for KV-load-failure recompute.
        # This can be removed after the upstream fix is merged.
        req_data = scheduler_output.scheduled_cached_reqs

        if self.use_async_scheduling:
            for i, req_id in enumerate(req_data.req_ids):
                req_state = self.requests.get(req_id)
                if req_state is None:
                    continue

                num_computed_tokens = req_data.num_computed_tokens[i]
                if num_computed_tokens < req_state.num_computed_tokens:
                    req_state.prev_num_draft_len = 0

        self._apply_pp_sampled_tokens_from_scheduler_output(scheduler_output)
        sampling_metadata = super()._update_states(scheduler_output)
        self._track_tmp_encoder_cache_refs(scheduler_output)
        return sampling_metadata

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        if not self.use_async_scheduling:
            return super()._update_states_after_model_execute(output_token_ids, scheduler_output)
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # InputBatch can be condensed/reordered before the next input preparation.
        # Keep the asynchronous D2H result in the previous iteration's row order,
        # independently of InputBatch, until the existing event is synchronized.
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)
        if self.cache_config.mamba_cache_mode == "align":
            mamba_utils.postprocess_mamba_align_gpu(
                bufs=self._get_mamba_bufs(),
                num_reqs=num_reqs,
                num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,
                num_accepted_tokens_cpu_tensor=self.num_accepted_tokens.cpu,
                input_batch=self.input_batch,
                kv_cache_config=self.kv_cache_config,
                forward_context=self.compilation_config.static_forward_context,
                mamba_state_copy_funcs=self.model.get_mamba_state_copy_func(),
            )
        else:
            self.num_accepted_tokens.copy_to_cpu(num_reqs)
            if self.cache_config.mamba_cache_mode == "all":
                mamba_utils.postprocess_mamba_all(
                    scheduler_output,
                    self.kv_cache_config,
                    self.input_batch,
                    self.requests,
                    self.mamba_state_idx,
                    self.num_spec_tokens,
                    num_reqs,
                )
        assert self.num_accepted_tokens_event is not None
        self.num_accepted_tokens_event.record()

    def _sync_num_accepted_tokens(self, num_reqs: int, has_prev_mapping: bool) -> None:
        """Publish accepted counts in current request order after the D2H event."""
        accepted = self.num_accepted_tokens.np
        if not self.use_async_scheduling:
            # Synchronous scheduling already updates counts in current batch order.
            accepted[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            return

        if has_prev_mapping:
            prev_idx = self.prev_positions.np[:num_reqs]
            new_mask = prev_idx < 0
            # Advanced indexing reads the snapshot before overwriting its rows.
            accepted[:num_reqs] = accepted[np.where(new_mask, 0, prev_idx)]
            accepted[:num_reqs][new_mask] = 1
        else:
            accepted[:num_reqs].fill(1)
        self.input_batch.num_accepted_tokens_cpu[:num_reqs] = accepted[:num_reqs]

    def _pad_query_start_loc_for_fia(
        self,
        query_start_loc: torch.Tensor,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        # avoid corner case of cudagraph config mode FULL to enter the first padding logic
        # e.g. 1 request with 1 token when num_spec > 1 (num_spec = 3 and cudagraph_batch_size = 4 for example)
        # will cause tokens are padded but requests are not
        if (
            num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.FULL
        ):
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc.np[num_reqs]
            query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Do not insert if the last value already equals the num_tokens
            if query_start_loc.np[num_reqs_padded] < num_tokens_padded:
                # Insert a dummy request instead of change the last value directly
                query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
                num_reqs_padded = num_reqs_padded + 1

        query_start_loc.copy_to_gpu()

        return num_reqs_padded

    def _track_tmp_encoder_cache_refs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        if not self.use_score_encoder_cache:
            return

        for new_req_data in scheduler_output.scheduled_new_reqs:
            for mm_feature in new_req_data.mm_features:
                cur_hash = mm_feature.identifier
                self.cached.setdefault(cur_hash, set()).add(new_req_data.req_id)

    def _on_request_state_removed(self, req_id: str, req_state: Any | None) -> None:
        if req_state is None:
            return
        if not self.use_score_encoder_cache:
            self.cached.clear()
            return
        free_mm_hashes = set()
        if req_state.mm_features is None:
            return
        for mm_feature in req_state.mm_features:
            free_mm_hashes.add(mm_feature.identifier)

        for mm_hash in free_mm_hashes:
            if mm_hash not in self.cached:
                continue
            self.cached[mm_hash].discard(req_id)
            if not self.cached.get(mm_hash):
                del self.cached[mm_hash]
                if mm_hash in self.tmp_encoder_cache:
                    del self.tmp_encoder_cache[mm_hash]

    def _process_encoder_cache_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        self._clear_finished_encoder_cache_copies()
        if not self.use_score_encoder_cache:
            super()._process_encoder_cache_scheduler_output(scheduler_output)
            return

        ec_manager_metadata = scheduler_output.ec_manager_metadata
        if ec_manager_metadata is None:
            super()._process_encoder_cache_scheduler_output(scheduler_output)
            return

        # Free the cached encoder outputs.
        promoting_mm_hashes = ec_manager_metadata.promoting_mm_hashes
        cpu_get_encoder_mm_hashes = ec_manager_metadata.cpu_get_encoder_mm_hashes

        for mm_hash in ec_manager_metadata.npu_freed:
            self.encoder_cache.pop(mm_hash, None)

        for mm_hash in ec_manager_metadata.cpu_freed:
            self.cpu_encoder_cache.pop(mm_hash, None)

        for mm_hash in promoting_mm_hashes:
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue

            self.encoder_cache[mm_hash] = self._copy_cpu_encoder_cache_to_device(
                cpu_value
            )

        for mm_hash in cpu_get_encoder_mm_hashes:
            if mm_hash in self.encoder_cache or mm_hash in self.tmp_encoder_cache:
                continue
            cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
            if cpu_value is None:
                continue
            self.tmp_encoder_cache[mm_hash] = self._copy_cpu_encoder_cache_to_device(
                cpu_value
            )

    def _get_encoder_output_from_cache(self, mm_hash: str) -> torch.Tensor | None:
        encoder_output = self.encoder_cache.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        encoder_output = self.tmp_encoder_cache.get(mm_hash, None)
        if encoder_output is not None:
            return encoder_output

        if not self.use_score_encoder_cache:
            return None

        cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
        if cpu_value is None:
            return None

        # The score encoder cache scheduler asks the worker to stage CPU cache
        # hits into a device cache before gather.  Keep a defensive synchronous
        # fallback here so a stale worker-side temporary entry cannot turn a
        # scheduler CPU-cache hit into an engine-fatal cache miss.
        encoder_output = self._copy_cpu_encoder_cache_to_device(
            cpu_value, non_blocking=False
        )
        self.tmp_encoder_cache[mm_hash] = encoder_output
        return encoder_output

    def _clear_finished_encoder_cache_copies(self) -> None:
        while (
            self._pending_encoder_cache_copies
            and self._pending_encoder_cache_copies[0][1].query()
        ):
            self._pending_encoder_cache_copies.popleft()

    def _copy_cpu_encoder_cache_to_device(
        self,
        cpu_value: torch.Tensor,
        *,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        if not non_blocking:
            return cpu_value.to(self.device, non_blocking=False)

        self._clear_finished_encoder_cache_copies()
        source = cpu_value if cpu_value.is_pinned() else cpu_value.pin_memory()
        npu_value = torch.empty_like(source, device=self.device)
        npu_value.copy_(source, non_blocking=True)

        copy_done = torch.npu.Event()
        copy_done.record(torch.npu.current_stream())
        self._pending_encoder_cache_copies.append((source, copy_done))
        npu_value.record_stream(torch.npu.current_stream())
        return npu_value

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
        ec_manager_metadata: Any | None,
        free_encoder_mm_hashes: list[str],
    ) -> None:
        if not self.use_score_encoder_cache:
            super()._cache_encoder_output(
                mm_hash,
                output,
                ec_manager_metadata,
                free_encoder_mm_hashes,
            )
            return

        promoting_mm_hashes = (
            ec_manager_metadata.promoting_mm_hashes
            if ec_manager_metadata is not None
            else []
        )
        npu_freed = (
            ec_manager_metadata.npu_freed
            if ec_manager_metadata is not None
            else []
        )

        staging = torch.empty_like(output, device="cpu", pin_memory=True)
        staging.copy_(output.detach(), non_blocking=True)
        self.cpu_encoder_cache[mm_hash] = staging

        if (
            mm_hash in promoting_mm_hashes
            and mm_hash not in npu_freed
        ):
            self.encoder_cache[mm_hash] = output
        else:
            self.tmp_encoder_cache[mm_hash] = output

        self.maybe_save_ec_to_connector({mm_hash: output}, mm_hash)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
        int,
    ]:
        """
        :return: tuple[
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
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

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # Get positions.
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[: cu_num_tokens[-1]],
            out=positions_np,
        )

        if self.use_dcp:
            self.dcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
            )

        # Build previous positions before DCP prepares speculative inputs.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)
        prev_positions_gpu = None
        if (
            self.use_async_scheduling
            and self.input_batch.prev_sampled_token_ids is not None
            and prev_req_id_to_index
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

        if self.speculative_config and self.use_dcp:
            self.dcp_manager.generate_dcp_mtp_input(
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
                prev_positions=prev_positions_gpu,
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

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()


        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # Fill unused with -1. Needed for reshape_and_cache in attention_cp
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu.copy_(
                self.mrope_positions.cpu,
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        self.discard_request_mask.np[:num_reqs] = discard_requests_mask
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # Sync num_accepted_tokens from CPU (set by
        # _update_states_after_model_execute for hybrid models).
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            self._sync_num_accepted_tokens(num_reqs, has_prev_mapping=bool(prev_req_id_to_index))
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        # Update num_computed_tokens on GPU. In async spec decode,
        # CPU values are optimistic (all drafts accepted). The kernel
        # corrects on GPU using the previous step's
        # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
        valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu
        if self.use_async_spec_decode:
            computed_token_tensor_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
        if (
            self.use_async_spec_decode
            and valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            if prev_positions_gpu is None:
                self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                computed_token_tensor_cpu,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]

        dcp_manager = getattr(self, "dcp_manager", None)
        if dcp_manager is not None:
            cp_async_rebuild = dcp_manager.rebuild_async_spec_decode_inputs(
                use_async_spec_decode=self.use_async_spec_decode,
                valid_sampled_token_count_gpu=valid_sampled_token_count_gpu,
                prev_req_id_to_index=prev_req_id_to_index,
                prev_positions_gpu=prev_positions_gpu,
                with_prefill=with_prefill,
                enable_prompt_embeds=self.enable_prompt_embeds,
                has_req_prompt_embeds=bool(self.input_batch.req_prompt_embeds),
                supports_mm_inputs=self.supports_mm_inputs,
                num_reqs=num_reqs,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                req_indices=req_indices,
                req_indices_gpu=req_indices_gpu,
                query_pos_gpu=self.query_pos.gpu,
                query_pos_np=self.query_pos.np,
                positions=self.positions,
                positions_np=positions_np,
                num_computed_tokens=self.num_computed_tokens,
                num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
                prev_positions_np=self.prev_positions.np,
                prev_num_draft_tokens_np=self.prev_num_draft_tokens.np,
                valid_sampled_token_count_event=self.valid_sampled_token_count_event,
                valid_sampled_token_count_cpu=self.valid_sampled_token_count_cpu,
                input_batch=self.input_batch,
                input_ids=self.input_ids,
                scheduler_output=scheduler_output,
                arange_np=self.arange_np,
                cu_num_tokens=cu_num_tokens,
                draft_token_ids=self._draft_token_ids,  # type: ignore[has-type]
                num_spec_tokens=self.num_spec_tokens,
                prepare_input_ids=self._prepare_input_ids,
            )
        else:
            cp_async_rebuild = DCPAsyncSpecDecodeRebuildResult(
                rebuilt=False,
                positions_ready_on_device=False,
            )

        if cp_async_rebuild.positions_ready_on_device:
            pass
        elif cp_async_rebuild.rebuilt:
            # The async rebuild computed corrected positions on CPU.
            # Copy positions_np to GPU so input_ids and positions stay aligned.

            self.positions[:total_num_scheduled_tokens].copy_(
                torch.from_numpy(
                    positions_np[:total_num_scheduled_tokens]
                ).to(self.device),
                non_blocking=True,
            )
        else:
            self.positions[:total_num_scheduled_tokens] = (
                self.num_computed_tokens[req_indices_gpu].to(torch.int64)
                + self.query_pos.gpu[:total_num_scheduled_tokens]
            )

        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # In async spec decode mode, optimistic_seq_lens_cpu assumes all
        # tokens from the previous speculative step were accepted. Correct it
        # on CPU using the valid-sampled-token counts that are already copied
        # asynchronously for scheduler bookkeeping. This avoids an extra
        # NPU->CPU seq_lens copy and the synchronize() in attention metadata.
        # Mirrors update_num_computed_tokens_for_batch_change on the GPU side.
        async_spec_decode_active = (
            self.use_async_spec_decode
            and valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        )
        if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
            self._correct_optimistic_seq_lens_cpu(num_reqs)

        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - computed_token_tensor_cpu[req_indices_gpu]
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift

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
            logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
                   (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                    self.input_batch.num_prompt_tokens[req_idx]):
                    num_decode_draft_tokens[req_idx] = draft_len
                else:
                    num_decode_draft_tokens[req_idx] = -1

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        self.logits_indices = logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        )

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
            if self.speculative_config:
                attn_state = AscendAttentionState.SpecDecoding
            else:
                attn_state = AscendAttentionState.ChunkedPrefill
        # splitfuse
        elif self.scheduler_config.enable_chunked_prefill:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        if attn_state == AscendAttentionState.SpecDecoding and self.speculative_config.method != "mtp":
            self.attn_state = AscendAttentionState.ChunkedPrefill  # type: ignore
        else:
            self.attn_state = attn_state  # type: ignore

        return attn_state

    def _sanitize_placeholder_input_ids_for_forward(
        self,
        scheduler_output: "SchedulerOutput",
        num_forward_tokens: int,
    ) -> None:
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not scheduled_spec_tokens:
            return
        if not any(
            PLACEHOLDER_TOKEN_ID in token_ids
            for token_ids in scheduled_spec_tokens.values()
        ):
            return

        input_ids = self.input_ids.gpu[:num_forward_tokens]
        input_ids.masked_fill_(input_ids == PLACEHOLDER_TOKEN_ID, 0)

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
        # Step 1.
        # cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # _arange_scratch[:11]: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]

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
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _correct_optimistic_seq_lens_cpu(self, num_reqs: int) -> None:
        """Correct ``optimistic_seq_lens_cpu`` for async spec-decode drift.

        The valid-sampled-token counts that drive the correction are copied
        device->host on a side stream at the end of the *previous* step (see
        :meth:`_copy_valid_sampled_token_count`). The host buffer must not be
        read until that copy has completed, otherwise the correction consumes
        stale counts and corrupts the CPU seq_lens. Callers that still build
        metadata from optimistic CPU seq_lens need this correction before
        attention metadata construction.

        Synchronizing on the event before the host read mirrors vLLM's own
        :meth:`_get_valid_sampled_token_count`. Because the copy was launched a
        full step earlier, the event is already signalled in steady state and
        the synchronize is effectively a no-op -- it does not reintroduce the
        seq_lens device->host copy + synchronize that this optimization removed.
        """
        assert self.valid_sampled_token_count_event is not None
        assert self.valid_sampled_token_count_cpu is not None
        self.valid_sampled_token_count_event.synchronize()
        correct_optimistic_seq_lens_cpu(
            self.optimistic_seq_lens_cpu.numpy(),
            self.prev_positions.np,
            self.prev_num_draft_tokens.np,
            self.valid_sampled_token_count_cpu.numpy(),
            num_reqs,
        )

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        default_stream = torch.npu.current_stream()
        with torch.npu.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count # type: ignore[no-redef]
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

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
        aux_hidden_states: torch.Tensor = None,
        sample_hidden_states: torch.Tensor = None,
        target_model_batch_desc: BatchDescriptor = None,
    ) -> list[list[int]] | None:
        self._log_propose_draft_token_ids_entry(spec_decode_metadata, num_scheduled_tokens)

        # Reset cached draft probs from the previous step so that stale data
        # is never used when the drafter does not produce fresh probabilities.
        self._draft_probs = None
        self._draft_prob_req_ids = None

        if not self.drafter:
            # Speculative decoding is not enabled.
            draft_token_ids = None
        elif isinstance(self.drafter, AscendNgramProposer):
            draft_token_ids = self.drafter.propose(
                scheduler_output.num_spec_tokens_to_schedule,
                valid_sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
            )
        elif isinstance(self.drafter, AscendSuffixDecodingProposer):
            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=scheduler_output.num_spec_tokens_to_schedule,
                sampled_token_ids=valid_sampled_token_ids,
            )
        elif isinstance(self.drafter, AscendNgramProposerNPU):
            batch_size = min(self.input_batch.num_reqs,
                             self.token_ids_gpu_tensor.shape[0])
            vocab_size = self.model_config.get_vocab_size()
            (next_token_ids, draft_token_ids,
             num_valid_draft_tokens, valid_sampled_tokens_count
             ) = triton_ngram_spec_decode(
                self.token_ids_gpu_tensor[:batch_size],       # [B, max_seq_len], in-place
                self.num_tokens_no_spec_gpu[:batch_size],      # [B]
                valid_sampled_token_ids,                       # list[list[int]] or [B, max_new_tokens]
                self.discard_request_mask.gpu[:batch_size],    # [B]
                vocab_size=vocab_size,
                min_n=self.drafter.min_n,
                max_n=self.drafter.max_n,
                k=self.drafter.k,
            )

            # Communicate verified token count to scheduler for async scheduling.
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

            # save num_valid_draft_tokens for scheduler trim
            self._num_valid_draft_tokens = num_valid_draft_tokens

            # async D2H copy num_valid_draft_tokens
            copy_num_valid_draft_tokens(
                self._num_valid_draft_tokens_cpu,
                self._num_valid_draft_tokens_copy_stream,
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens,
                batch_size,
            )
        elif isinstance(self.drafter, AscendMedusaProposer):
            draft_token_ids = self.drafter.propose(
                # Dynamic SD: forward the scheduled K (equals the configured
                # maximum when the feature is disabled).
                scheduler_output.num_spec_tokens_to_schedule,
                valid_sampled_token_ids,
                sampling_metadata,
                spec_decode_metadata,
                sample_hidden_states,
            )
        elif self.speculative_config.uses_extract_hidden_states():
            # Handle extract_hidden_states method
            assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
            assert isinstance(valid_sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            common_attn_metadata = spec_decode_common_attn_metadata
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids = self.drafter.propose(
                # Dynamic SD: honor the per-step K chosen by the scheduler
                # (equals the configured maximum when the feature is disabled).
                scheduler_output.num_spec_tokens_to_schedule,
                sampled_token_ids=valid_sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    valid_sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
            )
            self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
        elif self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model():
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
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
                )
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

            req_scheduled_tokens = scheduler_output.num_scheduled_tokens
            if self.use_dcp:
                long_seq_metadata = self.long_seq_metadata  # type: ignore
                num_prefill_reqs = self.dcp_manager.num_prefill_reqs
                num_decode_reqs = self.dcp_manager.num_decode_reqs
            else:
                long_seq_metadata = None  # type: ignore
                num_prefill_reqs = 0
                num_decode_reqs = 0

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). Safe to
            # rebind here: hidden_states was already consumed for sampling
            # above and is not used again in this branch.
            mtp_hidden_states = getattr(
                self.get_model(), "get_mtp_target_hidden_states", lambda: None
            )()
            if self.speculative_config.method == "mtp" and mtp_hidden_states is not None:
                hidden_states = mtp_hidden_states

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat([h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if self.vllm_config.speculative_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    assert self.drafter is not None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata, sampled_token_ids, spec_decode_metadata.num_draft_tokens
                    )
                else:
                    assert self.drafter is not None
                    common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count
                        )
                    )
                target_token_ids = self.input_ids.gpu[token_indices]
                target_positions = self._get_positions(token_indices)
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
            assert self.drafter is not None
            # Dynamic SD: pass the scheduled per-step K explicitly, unified with
            # the other proposers (ngram/suffix/medusa/extract) and matching
            # vLLM's ``propose(num_speculative_tokens=...)``. ``_propose`` sets
            # ``self.num_speculative_tokens`` from it, so the model runner no
            # longer mutates the drafter's state here.
            draft_token_ids = self.drafter._propose(
                num_speculative_tokens=scheduler_output.num_spec_tokens_to_schedule,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                target_model_batch_desc=target_model_batch_desc,
                sampling_metadata=sampling_metadata,
                req_scheduled_tokens=req_scheduled_tokens,
                long_seq_metadata=long_seq_metadata,
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
                scheduler_output=scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            if hasattr(self.drafter, "take_last_draft_probs"):
                draft_probs = self.drafter.take_last_draft_probs()
                if draft_probs is not None:
                    self._draft_probs = draft_probs
                    self._draft_prob_req_ids = self.input_batch.req_ids.copy()
        else:
            raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")

        return draft_token_ids

    def _log_propose_draft_token_ids_entry(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        num_scheduled_tokens: int,
    ) -> None:
        """DFX entry probe for propose_draft_token_ids.

        Records which speculative-decoding sub-path is about to run
        (ngram / medusa / eagle / extract_hidden_states / ...). When the
        `Unknown speculative decoding method` ValueError fires further down,
        or when end-to-end speedup is unexpectedly absent, this tells the
        operator which drafter object was actually constructed and whether k
        (num_speculative_tokens) is non-trivial. Type lookups and getattr are
        host-only; gated by isEnabledFor(DEBUG).
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        drafter_type = type(self.drafter).__name__ if self.drafter else None
        spec_meta_state = (
            "None" if spec_decode_metadata is None
            else f"max_spec_len={spec_decode_metadata.max_spec_len}"
        )
        logger.debug(
            "[spec/dfx] propose_draft_token_ids entry: "
            "drafter=%s, method=%s, k=%d, num_reqs=%d, "
            "num_scheduled_tokens=%d, spec_decode_metadata=%s, "
            "use_dcp=%s, dcp_size=%d",
            drafter_type,
            self.speculative_config.method if self.speculative_config else None,
            self.num_spec_tokens,
            self.input_batch.num_reqs,
            num_scheduled_tokens,
            spec_meta_state,
            getattr(self, "use_dcp", False),
            getattr(self, "dcp_size", 1),
        )

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Record the width of the drafts just produced so the next step unpacks
        # the previous-step draft tensor with the right stride. Under dynamic SD
        # this width can change step to step (mirrors vLLM's GPUModelRunner
        # ``prev_num_spec_tokens`` bookkeeping). Kept before the early returns so
        # it is updated even when the CPU copy itself is skipped.
        if torch.is_tensor(self._draft_token_ids):  # type: ignore[has-type]
            self.prev_num_spec_tokens = self._draft_token_ids.shape[1]  # type: ignore[has-type]
        if not self.num_spec_tokens:
            return
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
            or get_pp_group().world_size > 1
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids  # type: ignore[has-type]
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.npu.current_stream()
        num_reqs = draft_token_ids.shape[0]
        # Slice the pre-allocated CPU buffer (sized for the configured maximum
        # K) to the *actual* draft width. Under dynamic SD the produced width
        # can be < max, so an unsliced copy would shape-mismatch.
        num_spec_tokens = draft_token_ids.shape[1]
        with torch.npu.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens] = 0
            self.draft_token_ids_event.record()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        out = super().take_draft_token_ids()
        if out is None:
            return None
        dynamic_spec = getattr(self.drafter, "dynamic_spec", None)
        if dynamic_spec is None:
            return out
        per_req_k = dynamic_spec.num_verify_tokens
        if per_req_k is None:
            return out
        per_req_k = [
            max(0, min(int(k), self.num_spec_tokens))
            for k in per_req_k
        ]
        cut_tokens = DraftTokenIds(
            req_ids=out.req_ids,
            draft_token_ids=[
                tokens[:k]
                for tokens, k in zip(out.draft_token_ids, per_req_k)
            ],
        )
        return cut_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if vllm_version_is("0.27.1"):
            if self.vllm_config.model_config.enable_return_routed_experts and self.routed_experts_initialized:
                self.routed_experts_capturer.clear_buffer()

        profiling_chunk_config = self.ascend_config.scheduler_config.profiling_chunk_config
        if profiling_chunk_config.enabled and profiling_chunk_config.need_timing:
            # Check if the scheduler signaled that calibration is complete.
            # This flag is set cross-process via scheduler_output because
            # modifying the config singleton in the scheduler process does
            # not affect this worker process.
            if getattr(scheduler_output, "disable_profiling_timing", False):
                profiling_chunk_config.need_timing = False
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        # NOTE: The async-scheduling deepcopy was removed in
        # vllm-project/vllm#29821. Draft token ids are now propagated via
        # _copy_draft_token_ids_to_cpu()/take_draft_token_ids(), and the
        # scheduler output is updated by EngineCore in its own process
        # (Scheduler.update_draft_token_ids_in_output). Nothing below mutates
        # scheduler_output in the worker, so no copy is needed. 
        pp_group = get_pp_group()
        if pp_group.world_size > 1 and not pp_group.is_last_rank:
            new_token_ids = scheduler_output.scheduled_cached_reqs.new_token_ids
            if new_token_ids and all(not token_ids for token_ids in new_token_ids):
                scheduler_output = deepcopy(scheduler_output)
                scheduler_output.scheduled_cached_reqs.new_token_ids = []

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            # Preemption stores must run before _update_states() zeroes newly
            # allocated blocks that may reuse the same physical KV cache IDs.
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Fix up prev_req_id_to_index for requests that were discarded
                # in the previous sample_tokens step. If a request has
                # prev_num_draft_len > 0 but is missing from
                # prev_req_id_to_index, the parent _update_states would
                # hit a KeyError. Reset prev_num_draft_len to 0 for such
                # requests so they fall through safely.
                if (
                    self.use_async_scheduling
                    and self.num_spec_tokens
                    and self.input_batch.prev_req_id_to_index is not None
                ):
                    for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                        if (
                            req_id not in self.input_batch.prev_req_id_to_index
                            and (req_state := self.requests.get(req_id)) is not None
                            and req_state.prev_num_draft_len
                        ):
                            req_state.prev_num_draft_len = 0

                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(
                    scheduler_output
                )

                if has_ec_transfer() and not get_ec_transfer().is_consumer:
                    self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        self._finalize_dump_data()
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

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                if (scheduler_output.total_num_scheduled_tokens <= 0
                        or not tokens or sum(tokens) == 0):
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                self._start_dump_data(scheduled_tokens = scheduler_output.num_scheduled_tokens)
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
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
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                if logger.isEnabledFor(logging.DEBUG):
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

                if self.dynamic_eplb:
                    self.update_eplb_heat_collection_status(num_tokens_padded)

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # postprocess_mamba runs later in _update_states_after_model_execute.
                if self.cache_config.mamba_cache_mode == "align":
                    # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                    # to decide copy operations, so we must apply deferred
                    # corrections before it runs.
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    mamba_bufs = self._get_mamba_bufs()
                    preprocess_bufs = mamba_bufs.preprocess
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        preprocess_bufs,
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = (
                        self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    )
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                    if mamba_bufs.postprocess_align is not None:
                        mamba_utils.stage_postprocess_inputs_to_gpu(
                            mamba_bufs.postprocess_align,
                            scheduler_output,
                            self.input_batch.req_ids,
                            num_reqs,
                            self.requests,
                            self.mamba_state_idx,
                        )
                if self.use_compress:
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    num_reqs = self.input_batch.num_reqs
                    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_np)
                    dsa_positions_np = self._dsa_positions_np_buf[:total_num_scheduled_tokens]
                    np.add(
                        self.input_batch.num_computed_tokens_cpu[req_indices],
                        self.query_pos.np[:total_num_scheduled_tokens],
                        out=dsa_positions_np,
                    )

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        batch_desc.num_reqs,
                    )

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

                self._sanitize_placeholder_input_ids_for_forward(
                    scheduler_output,
                    num_tokens_padded,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
            )

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        # vLLM v0.27.1 still supports runtime KV scale calculation. Upstream main
        # removed this state in vllm-project/vllm#49389.
        if vllm_version_is("0.27.1") and self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        defer_kv_connector_finalize = self.speculative_config is not None and (
            get_pp_group().is_last_rank or self.broadcast_pp_output
        )
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
                has_sinks=self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": defer_kv_connector_finalize}
                ),
            ) as kv_connector_output,
        ):
            # Mamba state copy must run AFTER the KV transfer load finishes,
            # otherwise the copy would race with in-flight layerwise loads and
            # read half-loaded state. With a layerwise-capable connector we
            # defer the copy: prepare_mamba_state_copy stages per-layer copy
            # metadata here, and each layer's copy is executed right after its
            # own KV load completes (wait_for_layer_load), overlapping the copy
            # with the remaining layers' loads. Non-layerwise connectors keep
            # the batched copy after all loads finish (do_mamba_copy_block).
            mamba_copy_connector = None
            if self.cache_config.mamba_cache_mode == "align":
                if has_kv_transfer_group():
                    connector = get_kv_transfer_group()
                    prepare_mamba_state_copy = getattr(
                        connector,
                        "prepare_mamba_state_copy",
                        None,
                    )
                    if callable(prepare_mamba_state_copy) and prepare_mamba_state_copy(preprocess_bufs):
                        mamba_copy_connector = connector
                if mamba_copy_connector is None:
                    mamba_utils.do_mamba_copy_block(preprocess_bufs)
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
            # Verify every scheduled layer executed its deferred copy.
            if self.cache_config.mamba_cache_mode == "align" and mamba_copy_connector is not None:
                mamba_copy_connector.finish_mamba_state_copy()
        with record_function_or_nullcontext("post process"):
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
                    self._finalize_dump_data()
                    if self.dynamic_eplb:
                        self.eplb_updator.forward_end(self.eplb_heat_collection_status)
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    self._finalize_dump_data()
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
                batch_desc,
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        profiling_chunk_config = self.ascend_config.scheduler_config.profiling_chunk_config
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
        pp = get_pp_group()
        use_pp_spec_decode = self.speculative_config is not None and pp.world_size > 1

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
            batch_desc,
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
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu = None

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
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        output_spec_token_ids = None
        use_padded_batch = False
        early_pp_padded_drafter = False
        if self.speculative_config:
            use_padded_batch = (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
                or self.speculative_config.use_ngram_gpu()
            ) and not self.speculative_config.disable_padded_drafter_batch
            early_pp_padded_drafter = (
                use_pp_spec_decode
                and not self.use_async_scheduling
                and use_padded_batch
            )
            if early_pp_padded_drafter:
                self._draft_token_ids = None
                self._draft_token_req_ids = None
                with record_function_or_nullcontext("draft_token"):
                    propose_draft_token_ids(sampler_output.sampled_token_ids)

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
                if not early_pp_padded_drafter:
                    self._draft_token_ids = None
                    self._draft_token_req_ids = None
                if use_padded_batch and not early_pp_padded_drafter:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

            draft_token_ids = self._draft_token_ids if use_pp_spec_decode else None
            if draft_token_ids is not None:
                if isinstance(draft_token_ids, torch.Tensor):
                    num_reqs = draft_token_ids.shape[0]
                    draft_ids_list = draft_token_ids[:num_reqs].cpu().tolist()
                    draft_req_ids = self._draft_token_req_ids
                else:
                    draft_ids_list = draft_token_ids
                    draft_req_ids = self.input_batch.req_ids
                if draft_ids_list and draft_req_ids:
                    draft_by_req_id = dict(zip(draft_req_ids, draft_ids_list))
                    output_spec_token_ids = [
                        draft_by_req_id.get(req_id, [])
                        for req_id in req_ids_output_copy
                    ]

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=output_spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            pooler_output=[],
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
            routed_experts=None,
        )
        if (
            profiling_chunk_config.enabled
            and profiling_chunk_config.need_timing
            and hasattr(self, "_execution_start_time")
        ):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            self.eplb_updator.forward_end(self.eplb_heat_collection_status)

        self._finalize_dump_data()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        if not self.use_async_scheduling:
            if self.routed_experts_initialized:
                # Sync path: D2H was issued in ``_bookkeeping_sync`` and
                # synchronized by ``_to_list``'s event.synchronize(), so
                # the pinned buffers are ready to be wrapped as numpy.
                total = scheduler_output.total_num_scheduled_tokens
                model_runner_output.routed_experts = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )
            return model_runner_output

        # Async path: produce a device-side snapshot that the async
        # copy stream can D2H later. Both tensors must be private
        # clones because:
        #   - ``routing_data`` source is the shared capturer buffer,
        #     which is ``clear_buffer()``-ed at the start of the
        #     next step on the default stream.
        #   - ``slot_mapping`` source is our own
        #     ``routed_experts_slot_mapping_device``, which the
        #     next ``_prepare_inputs`` overwrites on the default
        #     stream while the D2H is still pending on the copy
        #     stream.
        # Without clones, the copy stream would read torn data.
        routed_experts_snapshot = None
        if self.routed_experts_initialized:
            buf = self.routed_experts_capturer.get_device_buffer()
            total = scheduler_output.total_num_scheduled_tokens
            routed_experts_snapshot = RoutedExpertsTensors(
                routing_data=buf[:total].clone(),
                slot_mapping=self.routed_experts_slot_mapping_device[
                    :total
                ].clone(),
            )
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
            routed_experts=routed_experts_snapshot,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    # overwrite _sample for lmhead_tp_enable and need_accepted_tokens
    def _sample(self, logits, spec_decode_metadata):
        # Sample the next token and get logprobs if needed.
        self.input_batch.update_async_output_token_ids()
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            if lmhead_tp_enable() and logits is not None:
                logits = logits[: self.input_batch.num_reqs]
            if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
                max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
                self.sampler.prepare_sampling(max_topk)
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        if lmhead_tp_enable() and logits is not None:
            logits = logits[: len(spec_decode_metadata.logits_indices)]
        if self.input_batch.sampling_metadata.top_k is not None and get_ascend_config().enable_reduce_sample:
            max_topk = self.input_batch.top_k_cpu[self.input_batch.top_k_cpu < logits.shape[1]].max()
            self.rejection_sampler.prepare_sampling(max_topk)
        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
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
        logprobs_lists = None
        if not self.use_async_scheduling:
            # Sync scheduling: issue routed experts D2H into the pinned
            # CPU buffer BEFORE ``_to_list`` below. ``_to_list`` does
            # ``event.synchronize()`` on the async copy stream which
            # waits for every D2H queued on the default stream since
            # the last sync, so this enqueue is naturally covered
            # without requiring its own synchronize.
            if self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
                self.routed_experts_slot_mapping_cpu[:total].copy_(
                    self.routed_experts_slot_mapping_device[:total],
                    non_blocking=True,
                )

            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                # parse_output returns (list[list[int]], LogprobsLists | None)
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
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

        # logprobs_lists is already set above:
        # - max_gen_len == 1: logprobs_tensors.tolists() (no cu_num_tokens)
        # - max_gen_len > 1: from RejectionSampler.parse_output() (filtered
        #   with cu_num_generated_tokens already set)

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

    def _update_full_graph_params_if_needed(
        self,
        forward_context: ForwardContext,
        num_tokens_padded: int,
    ) -> None:
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not forward_context.capturing
            and not self.use_sparse and not self.use_compress
        ):
            if self.enable_enpu:
                torch.npu.current_stream().synchronize()

            update_full_graph_params(
                self.attn_backend,
                self.update_stream,
                forward_context,
                num_tokens_padded,
                self.vllm_config,
                self.speculative_config,
            )

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
        forward_context = get_forward_context()
        assert forward_context is not None

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }
        run_model = partial(self.model, **model_inputs)

        if self.enable_enpu:
            # The soft segmentation scenario requires event.record first, then event.wait
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded)
            hidden_states = run_model()
        else:
            hidden_states = run_model()
            self._update_full_graph_params_if_needed(forward_context, num_tokens_padded)

        return hidden_states

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        # Native MoE SP and DSA-CP shard tokens at different boundaries, but
        # both require equal token counts on every TP rank.
        if enable_sp(self.vllm_config) or enable_dsa_cp():
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    # Keep PP intermediate tensors local to the sequence-parallel shard.
    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None
        tp = self.vllm_config.parallel_config.tensor_parallel_size

        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                copy_len = (num_tokens + tp - 1) // tp if enable_sp() else num_tokens
                if k not in self.intermediate_tensors.tensors:
                    base_tensor = self.intermediate_tensors["hidden_states"]
                    self.intermediate_tensors[k] = v.new_empty(
                        (base_tensor.shape[0], *v.shape[1:])
                    )
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: (num_tokens + tp - 1) // tp]
                if enable_sp()
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def sync_and_gather_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        # vLLM renamed sync_and_slice to sync_and_gather.
        return self.sync_and_slice_intermediate_tensors(
            num_tokens, intermediate_tensors, sync_self
        )

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
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None]:
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
        # A stateful P/D handoff can use a uniform decode graph even at
        # prompt_len - 1 computed tokens. Keep first-token prefills out.
        has_initial_state = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
        uniform_decode = (
            (
                has_initial_state
                and (max_num_scheduled_tokens == self.uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # ruff: noqa: E731
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            if force_eager:
                return (CUDAGraphMode.NONE, BatchDescriptor(num_tokens_padded))

            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                valid_modes=valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
                num_active_loras=num_active_loras,
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
            _, num_tokens_across_dp, synced_cudagraph_mode = self._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
                allow_dp_padding=((cudagraph_mode != CUDAGraphMode.NONE)
                                  or enable_sp(self.vllm_config)
                                  or oproj_tp_enable()
                                  or embedding_tp_enable()),
            )

            # Extract DP padding if there is any
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
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
        if self.sparse_kv_offload_enabled:
            update_sparse_kv_offload_metadata(
                num_tokens,
                num_reqs,
                num_tokens_padded,
                num_reqs_padded,
                self.input_batch.req_ids[:num_reqs],
                self.query_start_loc,
                self._offload_req_ids_tensor,
                self._offload_token_to_req,
            )
        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()


        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_dcp_metadata(block_table_tensor):
            if not self.use_dcp:
                return None, block_table_tensor

            fixed_decode_seq_lens_cpu = None
            if self.use_async_spec_decode:
                fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].numpy()

            assert num_reqs_padded is not None
            return self.dcp_manager.generate_dcp_metadata(
                num_tokens,
                self.query_lens,
                self.input_batch,
                num_scheduled_tokens_np,
                block_table_tensor,
                num_reqs_padded,
                num_reqs,
                fixed_decode_seq_lens_cpu,
            )

        def _get_block_table_and_slot_mapping(
            kv_cache_gid: int,
        ):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
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
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
                blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
                if self.routed_experts_initialized:
                    # snapshot slot_mapping into a private device
                    # buffer so the next ``_prepare_inputs`` does not
                    # overwrite it while D2H is still pending.
                    n = slot_mapping.shape[0]
                    self.routed_experts_slot_mapping_device[:n].copy_(
                        slot_mapping
                    )
            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        self.long_seq_metadata, block_table_gid_0 = _get_dcp_metadata(block_table_gid_0)
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        is_prefilling[num_reqs:] = False
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        if self.use_async_spec_decode:
            # GPU tensors are authoritative in async mode.
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        cm_base = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            # Always pass optimistic_seq_lens_cpu via _seq_lens_cpu so NPU
            # attention backends can get CPU seq_lens without GPU->CPU sync.
            # This is separate from seq_lens_cpu (None in async) which eagle
            # proposer checks to distinguish async/non-async behavior.
            _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
            # TODO
            seq_lens_cpu=seq_lens_cpu,
            # TODO
            # num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            is_prefilling=is_prefilling,
            num_input_tokens=num_tokens_padded,
            actual_seq_lengths_q=self.actual_seq_lengths_q,
            positions=self.positions,
            positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
            attn_state=self.attn_state,
            decode_token_per_req=self.decode_token_per_req,
            context_parallel_metadata=self.long_seq_metadata,
            group_len = self.group_len.gpu[:num_reqs_padded],
            group_key_idx = self.group_key_idx.gpu[:num_reqs_padded],
            group_key_cache_idx = self.group_key_cache_idx.gpu[:num_reqs_padded],
            req_ids_tensor=(
                self._offload_req_ids_tensor.gpu[:num_reqs_padded]
                if self._offload_req_ids_tensor is not None
                else None
            ),
            token_to_req=(
                self._offload_token_to_req.gpu[:num_tokens_padded]
                if self._offload_token_to_req is not None
                else None
            ),
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(logits_indices)

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            common_ratio_to_sas_metadata: dict,
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
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[:num_reqs_padded],
                )

            if isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder)):
                if for_cudagraph_capture:
                    common_ratio_to_sas_metadata = {}
                extra_attn_metadata_args = dict(
                    num_actual_reqs=num_reqs,
                    common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                )

            if (for_cudagraph_capture
                    and not isinstance(builder, (
                        AscendDSAMetadataBuilder,
                        AscendDSACPMetadataBuilder,
                        AscendSFADCPMetadataBuilder,
                    ))):
                attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                # NOTE(zxr): Due to the Triton operator does not deal with -1 padding in FullGraph mode,
                # the padding needs to be changed from -1 to 0 to avoid writing invalid mamba block.
                if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() \
                    and isinstance(builder, GDNAttentionMetadataBuilder) and attn_metadata_i.num_prefills == 0:
                    if attn_metadata_i.num_decodes == 0 and attn_metadata_i.num_spec_decodes > 0:
                        attn_metadata_i.spec_state_indices_tensor[attn_metadata_i.num_spec_decodes:].fill_(0)
            if isinstance(builder, AscendDSAMetadataBuilder):
                common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata  # type: ignore[assignment]

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
        common_ratio_to_sas_metadata: dict[Any, Any] = {}
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

            # Now, query_start_loc is padded.
            # But gdn needs an unpadded one.
            # gdn_query_start_loc is an unpadded version of query_start_loc.
            # TODO delete it if fia's check is removed.
            if self._has_gdn:
                attn_group = self.attn_groups[kv_cache_gid][0]
                builder = attn_group.get_metadata_builder(0)
                if isinstance(builder, GDNAttentionMetadataBuilder):
                    cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[: num_reqs_padded + 1]
                    cm.query_start_loc = self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]

            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(
                    kv_cache_gid
                )
            if self.speculative_config and isinstance(self.drafter, (AscendStep3p5MTPProposer, AscendDSparkProposer)):
                # step3p5 MTP draft layers span multiple KV cache groups; capture
                # each group's block table / slot mapping so the proposer can
                # build per-step attention metadata for the active MTP layer.
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping)
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, AscendEagleProposer | AscendDraftModelProposer | AscendDflashProposer 
                    | AscendDSparkProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                elif isinstance(self.drafter, AscendExtractHiddenStatesProposer):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                _build_attn_group_metadata(
                    kv_cache_gid,
                    attn_gid,
                    cm,
                    common_ratio_to_sas_metadata,
                )
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

    def _should_build_dummy_attn_metadata(
        self,
        force_attention: bool = False,
        is_profile: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> bool:
        """
        Determine whether attention metadata should be built during dummy_run.
        SubClass can override this to add custom conditions.
        """
        # If force_attention is True, we always capture attention, Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        return force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL

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
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
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
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        elif profile_cpp:
            num_reqs = 1
            num_scheduled_tokens_list = [num_tokens] * num_reqs
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
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE) or profile_cpp,
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            force_num_active_loras=num_active_loras,
        )
        if self.use_dcp:
            self.dcp_manager.init_batch_info(
                num_scheduled_tokens,
                num_reqs,
                self.input_batch.num_computed_tokens_cpu,
                self.input_batch.num_prompt_tokens,
            )
            if self.speculative_config:
                self.dcp_manager.query_lens_full.cpu[:num_reqs] = torch.from_numpy(num_scheduled_tokens)
                self.dcp_manager.query_lens_full.copy_to_gpu()
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

        if self.dynamic_eplb:
            self.update_eplb_heat_collection_status(num_tokens_padded)

        # vllm-ascend does not support ubatch now
        ubatch_slices, ubatch_slices_padded = None, None
        attn_metadata: PerLayerAttnMetadata | None = None
        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # gdn_query_start_loc, etc.) with execute_model. It must participate in
        # the same event protocol so that back-to-back dummy/real steps don't
        # overwrite pinned memory while a prior non_blocking H2D DMA is still
        # reading. Mirrors upstream gpu_model_runner._dummy_run.
        with self.synchronize_input_prep():
            # Build attention metadata for dummy_run
            if self._should_build_dummy_attn_metadata(force_attention, is_profile, cudagraph_runtime_mode):
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
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens
                else:
                    seq_lens = (
                        SEQ_LEN_WITH_MAX_PA_WORKSPACE
                        if is_graph_capturing and using_paged_attention(num_tokens, self.vllm_config)
                        else max_query_len
                    )  # type: ignore[assignment]

                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens, self.query_pos.np)
                self.query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()
                if self._has_gdn:
                    self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                    self.gdn_query_start_loc.copy_to_gpu()

                if not profile_cpp:
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_runtime_mode,
                        batch_desc.num_reqs,
                    )

                # Dummy graph runs do not go through _prepare_inputs(), but GDN/Mamba
                # metadata reads block_table[:num_reqs_padded] below. Sync padded
                # rows as well so device-side metadata does not see stale block ids.
                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                # Invalidate real-request slots before attention backends derive
                # or copy their backend-specific metadata for dummy execution.
                if not is_graph_capturing:
                    for kv_cache_gid in range(len(self.kv_cache_config.kv_cache_groups)):
                        blk_table = self.input_batch.block_table[kv_cache_gid]
                        blk_table.slot_mapping.gpu.fill_(-1)

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                # check how to build dummy
                if self.use_compress:
                    self.positions.fill_(127)
                    self._dsa_positions_cpu_buf.fill_(127)
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                    for_cudagraph_capture=is_graph_capturing,
                    num_scheduled_tokens_np=num_scheduled_tokens,
                )
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            # TODO: The next line is a temporary workaround
            # to fix the accuracy issue of test_llama32_lora.py,
            # which is introduced by vllm-project/vllm#32005
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:
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
                positions = self.positions[:num_tokens_padded]

            # update global cos, sin
            update_cos_sin(positions)

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                # When PP and sequence parallelism are enabled, during dummy_run the estimated space should divide
                # num_tokens by
                # tp_size; otherwise, on non-first PP ranks it would effectively perform an extra all-gather, leading
                # to incorrect memory estimation and potentially causing OOM.
                intermediate_tokens = num_tokens_padded
                if enable_sp():
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
            # Keep indices on the same device as hidden_states to avoid an
            # implicit synchronous CPU-to-NPU copy during dummy-run indexing.
            dummy_indices = torch.zeros(
                max_num_reqs_across_dp,
                dtype=torch.int32,
                device=self.device,
            )

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
                has_sinks = self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ):
                if not is_graph_capturing and self.ascend_config.enable_force_eplb \
                    and self.vllm_config.model_config.is_moe:
                    build_force_eplb_topk(self.device, self.max_num_tokens)
                outputs = self._model_forward(
                    num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            dummy_compute_logits(hidden_states)

            if self.drafter and not profile_cpp:
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
                self.eplb_updator.adaptor.clear_all_moe_loads()
            if not is_profile and self.dynamic_eplb:
                self.eplb_updator.forward_end(self.eplb_heat_collection_status)
            self._finalize_dump_data(dump=False)
            if self.use_compress and force_attention:
                self.positions.fill_(0)
                self._dsa_positions_cpu_buf.fill_(0)
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
        if self.sparse_kv_offload_enabled:
            allocate_kv_offload_topk_profile_buffers(
                getattr(self, "kv_cache_spec", None) or self.get_kv_cache_spec(),
                self.vllm_config,
                self.sparse_kv_offload_config,
            )

        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if self.max_num_tokens > mc2_tokens_capacity and select_moe_comm_method(
            mc2_tokens_capacity, self.vllm_config
        ) in {MoECommType.MC2, MoECommType.FUSED_MC2}:
            self._dummy_run(mc2_tokens_capacity, with_prefill=True, is_profile=True)
        super().profile_run()

    def eplb_warmup(self):
        if self.dynamic_eplb and not self.is_eplb_warmuped:
            self.is_eplb_warmuped = True
            self.eplb_adaptor = VllmEplbAdaptor(model=self.model)
            self.eplb_loader.set_adator(self.eplb_adaptor)
            self.eplb_updator.set_adaptor(self.eplb_adaptor)
            self.eplb_updator.warm_up_eplb()

    def update_eplb_heat_collection_status(self, num_tokens_padded: int):
        if self.eplb_heat_collection_stage == "prefill":
            # collect eplb heat for prefill requests.
            self.eplb_heat_collection_status = num_tokens_padded > self.eplb_pd_thresholds
        elif self.eplb_heat_collection_stage == "decode":
            # collect eplb heat for decode requests.
            self.eplb_heat_collection_status = num_tokens_padded <= self.eplb_pd_thresholds
        else:
            # collect eplb heat for all requests.
            self.eplb_heat_collection_status =  True

    def load_model(self) -> None:
        load_model_start_time = time.perf_counter()
        logger.info("Starting to load model %s...", self.model_config.model)

        if self.ascend_config.mix_placement:
            # TODO: Enabling the mix placement in deepseek_v2.py
            # remove this part after the mix placement merged into vllm
            def mock_true():
                return True
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled = mock_true
            rocm_aiter_ops.is_fused_moe_enabled = mock_true

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            if self.eplb_enable:
                def mock_pass(param1, param2):
                    return
                from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
                DefaultModelLoader._init_ep_weight_filter = mock_pass
            self.model: nn.Module = get_model(vllm_config=self.vllm_config)
            for name, _ in self.model.named_parameters():
                # sinks is a kind of parameter in attention
                # only set in weight name
                # TODO: remove it when fia merge in fiav2
                if "sink" in name:
                    self._has_sinks = True
                    break
            if self.drafter:
                logger.info("Loading drafter model...")
                with get_tp_context(self.drafter):
                    self.drafter.load_model(self.model)

            pp_group = get_pp_group()
            should_configure_aux_hidden_states = (
                self.use_aux_hidden_state_outputs
                if pp_group.world_size == 1
                else self._eagle3_uses_aux_hidden_state()
            )
            if should_configure_aux_hidden_states:
                from vllm.model_executor.models.interfaces import supports_eagle3

                if not supports_eagle3(self.model):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )

                aux_layers = self._get_eagle3_aux_layers_from_config()
                if not aux_layers:
                    aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
                self.model.set_aux_hidden_state_layers(aux_layers)
                if self.speculative_config.use_dspark():
                    set_capture_mode = getattr(
                        self.model,
                        "set_dspark_aux_capture_materialized",
                        None,
                    )
                    if set_capture_mode is not None:
                        materialized = self._draft_uses_qwen3_gqa_dspark()
                        set_capture_mode(materialized)
                        logger.info(
                            "Kimi K3 DSpark auxiliary capture uses %s stream.",
                            "materialized GQA" if materialized else "raw MLA",
                        )

                if pp_group.world_size > 1:
                    inner_model = self.model
                    if hasattr(inner_model, "get_language_model"):
                        inner_model = inner_model.get_language_model()
                    elif hasattr(inner_model, "language_model"):
                        language_model = inner_model.language_model
                        inner_model = (
                            language_model()
                            if callable(language_model)
                            else language_model
                        )
                    if hasattr(inner_model, "model"):
                        inner_model = inner_model.model
                    from vllm_ascend.patch.worker.patch_eagle3_pp_aux import (
                        patch_eagle3_pp_aux_propagation,
                    )

                    if patch_eagle3_pp_aux_propagation(inner_model):
                        self.model.make_empty_intermediate_tensors = (
                            inner_model.make_empty_intermediate_tensors
                        )

            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        get_offloader().post_init()

        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        ) # type: bool

        # wrap the model with full graph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None

        if cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

            if self.drafter is not None:
                self.drafter.update_stream = self.update_stream

        with _torch_cuda_wrapper():
            if (
                breakable_cudagraph.is_breakable_cudagraph_enabled()
                and cudagraph_mode != CUDAGraphMode.NONE
            ):
                self.model = BreakableACLGraphWrapper(
                    self.model,
                    self.vllm_config,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )
                drafter = getattr(self, "drafter", None)
                if drafter is not None and hasattr(drafter, "model"):
                    drafter.model = BreakableACLGraphWrapper(
                        drafter.model,
                        self.vllm_config,
                        use_eagle=self.use_eagle,
                        enable_enpu=self.enable_enpu,
                    )
            elif cudagraph_mode.has_full_cudagraphs():
                self.model = ACLGraphWrapper(
                    self.model,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )

        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self._start_dump_data()

        load_model_total_time = time.perf_counter() - load_model_start_time
        logger.info(
            "Model runner load_model total time: %.2f seconds",
            load_model_total_time,
        )

    def _start_dump_data(self, **kwargs) -> None:
        if self.debugger is None or self._debugger_started:
            return
        self.debugger.start(self.model, **kwargs)
        self._debugger_started = True

    def _finalize_dump_data(self, **kwargs) -> None:
        if self.debugger is None or not self._debugger_started:
            return
        if hasattr(self.debugger, "stop"):
            self.debugger.stop()
            self._debugger_started = False

        self.debugger.step(**kwargs)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        apply_layerwise_kv_cache_plan(kv_cache_config, self.vllm_config)
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        # NOTE(cmq): initialize_attn_backend must before using self.attn_groups
        self.initialize_attn_backend(kv_cache_config)
        self.use_hybrid_blocks = len(self.attn_groups) > 1
        # K3's packed layout keeps Mamba specs inside UniformType groups, so the
        # old first-spec scan over attn_groups cannot recognize them.
        self.need_accepted_tokens = kv_cache_config.has_mamba_layers

        self.may_reinitialize_input_batch(kv_cache_config)
        if self.sparse_kv_offload_enabled:
            self.sparse_kv_offload_manager = init_sparse_kv_offload_manager(
                self.vllm_config,
                kv_cache_config,
                self.sparse_kv_offload_config,
            )
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
        # TODO: refactor the logic of attention
        if (
            self.speculative_config
            and self.drafter is not None
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
            )
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer | AscendDSparkProposer | AscendDraftModelProposer,
            )
            kernel_block_sizes = self.kernel_block_sizes
            if isinstance(self.drafter, AscendDSparkProposer):
                sizes = kernel_block_sizes if isinstance(kernel_block_sizes, list) else [kernel_block_sizes]
                draft_kernel_block_sizes = [
                    int(size[0] if isinstance(size, (list, tuple)) else size) for size in sizes
                ]
            else:
                draft_kernel_block_sizes = (
                    kernel_block_sizes[0] if isinstance(kernel_block_sizes, list) else kernel_block_sizes
                )
            self.drafter.initialize_attn_backend(kv_cache_config, draft_kernel_block_sizes)

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if self.sparse_kv_offload_enabled:
            assert self.sparse_kv_offload_manager is not None
            self.sparse_kv_offload_manager.register_kv_caches(kv_caches)
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

        if self.model_config.hf_text_config.model_type == "deepseek_v4":
            from vllm_ascend.utils import extract_dsv4_layer_index

            assert len(self.kv_caches) == 0
            for layer_name in sorted(
                    kv_caches,
                    key=lambda name: (extract_dsv4_layer_index(
                        self.model_config.hf_text_config, name), name)):
                self.kv_caches.append(kv_caches[layer_name])
            for layer_name, kv_cache in kv_caches.items():
                self.compilation_config.static_forward_context[
                    layer_name].kv_cache = [kv_cache]
        else:
            from vllm.v1.worker.utils import bind_kv_cache

            model_type = self.model_config.hf_text_config.model_type
            num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
            bind_kv_cache(
                kv_caches,
                self.compilation_config.static_forward_context,
                self.kv_caches,
                num_attn_module,
            )

        return kv_caches

    def _get_layer_kv_cache_specs(self, kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
        compilation_config = getattr(self, "compilation_config", None)
        if compilation_config is None:
            compilation_config = getattr(self.vllm_config, "compilation_config", None)
        static_forward_context = getattr(compilation_config, "static_forward_context", None)

        layer_kv_cache_spec: dict[str, KVCacheSpec] = {}
        for group_kv_cache_spec in kv_cache_config.kv_cache_groups:
            group_spec = group_kv_cache_spec.kv_cache_spec
            for layer_name in group_kv_cache_spec.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_kv_cache_spec[layer_name] = group_spec
                if static_forward_context is not None:
                    attn_layer = static_forward_context.get(layer_name)
                    if isinstance(attn_layer, AttentionLayerBase):
                        spec = attn_layer.get_kv_cache_spec(self.vllm_config)
                        if isinstance(spec, AscendSFAIndexerCacheSpec):
                            layer_kv_cache_spec[layer_name] = spec
        return layer_kv_cache_spec

    def _get_attention_kv_cache_dims(self, layer_name: str, kv_cache_spec: AttentionSpec) -> tuple[int, int]:
        if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
                [layer_name],
            )
            attn_layer = attn_layers[layer_name]
            if isinstance(attn_layer, MLAAttention):
                # DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim
                return attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim
            # CacheOnlyAttentionLayer uses AscendMLAAttentionSpec but isn't MLAAttention
            if isinstance(attn_layer, CacheOnlyAttentionLayer):
                return kv_cache_spec.head_size, kv_cache_spec.head_size
            raise TypeError(
                f"Expected MLAAttention layer for {layer_name}, got {type(attn_layer).__name__}."
            )

        head_size_v = kv_cache_spec.head_size_v if hasattr(kv_cache_spec, "head_size_v") else kv_cache_spec.head_size
        return kv_cache_spec.head_size, head_size_v

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def _allocate_int8_cache_tensor(
        self,
        numel: int,
        alignment: int,
    ) -> torch.Tensor:
        """Allocate an int8 raw cache tensor.

        When KV transfer is enabled, the returned tensor's data_ptr is aligned
        to `alignment`. This keeps the original Mooncake/ADXL alignment behavior.
        """
        if numel <= 0:
            raise ValueError(f"Invalid cache tensor size: {numel}")

        if self.vllm_config.kv_transfer_config is None:
            return torch.zeros(numel, dtype=torch.int8, device=self.device)

        raw_tensor = torch.zeros(
            numel + alignment,
            dtype=torch.int8,
            device=self.device,
        )
        return self._align_memory(raw_tensor, alignment)[:numel]

    def _allocate_sparse_c8_indexer_tensors(
        self,
        dsa_k_tensor_size: int,
        dsa_k_scale_tensor_size: int,
        alignment: int,
        scale_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate dsa_k and dsa_k_scale from one aligned int8 raw allocation.

        Both returned tensors are logical views into the same underlying storage:

            sparse_c8_raw
              ├── dsa_k_tensor        int8 raw bytes
              └── dsa_k_scale_tensor  scale dtype raw bytes stored as int8 view

        `dsa_k_scale_tensor` is still returned as int8 raw storage. Later reshape
        code should continue to use:

            raw_dsa_k_scale_tensor.view(scale_dtype).view(scale_shape)

        This reduces HCCL/Mooncake registration count because register_buffer
        can merge these two views into one registered memory range.
        """
        if dsa_k_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_tensor_size: {dsa_k_tensor_size}"
            )
        if dsa_k_scale_tensor_size <= 0:
            raise ValueError(
                f"Invalid dsa_k_scale_tensor_size: {dsa_k_scale_tensor_size}"
            )

        scale_dtype_size = torch.empty((), dtype=scale_dtype).element_size()

        # Ensure the scale view starts at an address aligned for scale_dtype.
        scale_offset = self._align_up(dsa_k_tensor_size, scale_dtype_size)
        total_raw_size = scale_offset + dsa_k_scale_tensor_size

        sparse_c8_raw_tensor = self._allocate_int8_cache_tensor(
            total_raw_size,
            alignment,
        )

        dsa_k_tensor = sparse_c8_raw_tensor[:dsa_k_tensor_size]
        dsa_k_scale_tensor = sparse_c8_raw_tensor[
            scale_offset : scale_offset + dsa_k_scale_tensor_size
        ]

        assert dsa_k_tensor.is_contiguous()
        assert dsa_k_scale_tensor.is_contiguous()
        assert dsa_k_scale_tensor.data_ptr() % scale_dtype_size == 0
        assert dsa_k_scale_tensor.numel() % scale_dtype_size == 0

        return dsa_k_tensor, dsa_k_scale_tensor

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
        kv_cache_raw_tensors: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}
        # prefill disaggregation need the addr of cache tensor be aligned with 2M
        alignment = 2 * 1024 * 1024
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        # v0.27.1 keeps the legacy ``shared_by`` contract: one allocation per
        # descriptor, shared by every listed layer. Main uses #51718's
        # standardized descriptors, whose ``size`` is the size of one common
        # backing allocation rather than the size of an individual layer.
        use_legacy_shared_by_layout = vllm_version_is("0.27.1")
        is_dsv4_main = not use_legacy_shared_by_layout and any(
            getattr(spec, "model_version", None) == "deepseek_v4"
            for spec in layer_kv_cache_spec.values()
        )
        # vLLM #51718 no longer lists layers from different cache groups in one
        # descriptor. Detect hybrid models across all groups instead of within
        # each descriptor so the attention view keeps the combined K/V format.
        self.hybrid_with_attn_and_mamba = any(
            isinstance(spec, MambaSpec) for spec in layer_kv_cache_spec.values()
        ) and any(isinstance(spec, AttentionSpec) for spec in layer_kv_cache_spec.values())

        kv_transfer_config = self.vllm_config.kv_transfer_config
        kv_connector = (
            getattr(kv_transfer_config, "kv_connector", None)
            if kv_transfer_config is not None
            else None
        )
        # ExampleHiddenStatesConnector only consumes the dedicated cache-only
        # layer and does not register or migrate the per-layer Attention/Mamba
        # buffers. It therefore does not require the independent 2 MiB-aligned
        # allocations needed by Mooncake/ADXL connectors.
        supports_shared_backing_with_kv_transfer = (
            kv_transfer_config is None
            or kv_connector == "ExampleHiddenStatesConnector"
        )

        # The restored DSV4 planner on main emits multiple descriptors into a
        # single shared-tuple backing. Its memory budget is computed for that
        # one backing, so allocating ``descriptor.size`` for every descriptor
        # over-commits the device. Validate the complete geometry first, then
        # allocate exactly once and expose the contiguous per-layer regions
        # consumed by the existing page-strided DSV4 reshape path.
        if is_dsv4_main and kv_cache_config.kv_cache_tensors:
            tensor_sizes = {
                descriptor.size
                for descriptor in kv_cache_config.kv_cache_tensors
            }
            if len(tensor_sizes) != 1:
                raise ValueError(
                    "DeepSeek-V4 KV cache descriptors must share one backing "
                    "allocation."
                )
            backing_size = tensor_sizes.pop()
            dsv4_regions: list[tuple[str, int, int]] = []
            for descriptor in kv_cache_config.kv_cache_tensors:
                for layer_idx, layer_name in enumerate(
                    get_kv_cache_tensor_layers(descriptor)
                ):
                    spec = layer_kv_cache_spec[layer_name]
                    if descriptor.block_stride != spec.page_size_bytes:
                        raise ValueError(
                            "DeepSeek-V4 requires contiguous per-layer pages, "
                            f"but {layer_name} has block_stride="
                            f"{descriptor.block_stride} and page_size="
                            f"{spec.page_size_bytes}."
                        )
                    layer_size = (
                        kv_cache_config.num_blocks * spec.page_size_bytes
                    )
                    start = (
                        descriptor.offset
                        + layer_idx * descriptor.layer_stride
                    )
                    if start < 0 or start + layer_size > backing_size:
                        raise ValueError(
                            f"DeepSeek-V4 KV cache view for {layer_name} "
                            "exceeds the shared backing allocation."
                        )
                    dsv4_regions.append((layer_name, start, layer_size))

            if not dsv4_regions:
                raise ValueError(
                    "DeepSeek-V4 KV cache config has no materializable layers."
                )
            backing = self._allocate_int8_cache_tensor(
                backing_size,
                alignment,
            )
            for layer_name, start, layer_size in dsv4_regions:
                kv_cache_raw_tensors[layer_name] = backing[
                    start : start + layer_size
                ]

        # The standardized main descriptors all refer to the same backing
        # allocation. Ascend attention and Mamba backends still require
        # contiguous per-layer tensors, which the default layer/block-compact
        # layout provides. Materialize slices of one backing tensor so cache
        # groups retain #51718's overlay semantics without reducing capacity.
        if (
            not use_legacy_shared_by_layout
            and not is_dsv4_main
            and self.hybrid_with_attn_and_mamba
            and supports_shared_backing_with_kv_transfer
            and not self.use_sparse
            and not self.use_compress
            and kv_cache_config.kv_cache_tensors
        ):
            layout = self.vllm_config.cache_config.get_resolved_kv_cache_layout()
            tensor_sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
            regions: list[tuple[str, int, int]] = []
            if len(tensor_sizes) == 1 and layout.is_layer_compact and layout.is_block_compact:
                backing_size = next(iter(tensor_sizes))
                for descriptor in kv_cache_config.kv_cache_tensors:
                    for layer_idx, layer_name in enumerate(get_kv_cache_tensor_layers(descriptor)):
                        spec = layer_kv_cache_spec[layer_name]
                        layer_size = kv_cache_config.num_blocks * spec.page_size_bytes
                        start = descriptor.offset + layer_idx * descriptor.layer_stride
                        if descriptor.block_stride != spec.page_size_bytes or start + layer_size > backing_size:
                            regions = []
                            break
                        regions.append((layer_name, start, layer_size))
                    if not regions:
                        break
                if regions:
                    backing = self._allocate_int8_cache_tensor(
                        backing_size,
                        alignment,
                    )
                    for layer_name, start, layer_size in regions:
                        kv_cache_raw_tensors[layer_name] = backing[start : start + layer_size]

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            shared_layers = get_kv_cache_tensor_layers(kv_cache_tensor)
            use_mamba = False
            for layer_name in shared_layers:
                if isinstance(layer_kv_cache_spec[layer_name], MambaSpec):
                    use_mamba = True
            for idx in range(len(shared_layers)):
                layer_name = shared_layers[idx]
                # Single tensor path for: mamba, hybrid attn-mamba, or cache_only_layers
                if (
                    "linear_attn" in layer_name
                    or self.hybrid_with_attn_and_mamba
                    or "cache_only_layers" in layer_name
                    or is_hidden_state_cache_spec(layer_kv_cache_spec.get(layer_name))
                ) and layer_name not in kv_cache_raw_tensors:
                    # Check if shared_by contains both MambaSpec and HiddenStateCacheSpec.
                    # If so, they must use separate physical memory to avoid corruption:
                    # writing float32 ssm_state data into the shared buffer overwrites
                    # bfloat16 hidden-states data (same bytes, different interpretation).
                    has_mamba = any(
                        isinstance(layer_kv_cache_spec.get(ln), MambaSpec)
                        for ln in shared_layers
                    )
                    has_hidden = any(
                        is_hidden_state_cache_spec(layer_kv_cache_spec.get(ln))
                        for ln in shared_layers
                    )
                    # vLLM #51718 packs all group layers into one tensor on main;
                    # each layer owns an equal share (the per-layer size v0.27.1
                    # stored directly on the tensor).
                    per_layer_size = (
                        kv_cache_tensor.size
                        if use_legacy_shared_by_layout
                        else kv_cache_tensor.size // len(shared_layers)
                    )
                    if use_legacy_shared_by_layout:
                        if self.vllm_config.kv_transfer_config is None:
                            tensor = torch.zeros(per_layer_size, dtype=torch.int8, device=self.device)
                        else:
                            cache_size_aligned = per_layer_size + alignment
                            tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                            tensor = self._align_memory(tensor, alignment)[: per_layer_size]
                        if has_mamba and has_hidden:
                            # Allocate separate tensor for HiddenStateCacheSpec layers
                            # so ssm_state writes don't corrupt hidden-states data
                            if self.vllm_config.kv_transfer_config is None:
                                tensor_hs = torch.zeros(per_layer_size, dtype=torch.int8, device=self.device)
                            else:
                                cache_size_aligned = per_layer_size + alignment
                                tensor_hs = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                                tensor_hs = self._align_memory(tensor_hs, alignment)[: per_layer_size]
                            for layer_name_inner in shared_layers:
                                if is_hidden_state_cache_spec(layer_kv_cache_spec.get(layer_name_inner)):
                                    kv_cache_raw_tensors[layer_name_inner] = tensor_hs
                                else:
                                    kv_cache_raw_tensors[layer_name_inner] = tensor
                        else:
                            for layer_name_inner in shared_layers:
                                kv_cache_raw_tensors[layer_name_inner] = tensor
                    else:
                        # vLLM #51718 describes the shared allocation through
                        # per-layer page sizes and strides. Ascend translates it
                        # to private contiguous layer buffers, including hybrid
                        # attention/Mamba groups; the worker's memory budget is
                        # scaled for this representation.
                        for layer_name_inner in shared_layers:
                            layer_size = (
                                kv_cache_config.num_blocks
                                * layer_kv_cache_spec[layer_name_inner].page_size_bytes
                            )
                            if self.vllm_config.kv_transfer_config is None:
                                tensor = torch.zeros(layer_size, dtype=torch.int8, device=self.device)
                            else:
                                cache_size_aligned = layer_size + alignment
                                tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                                tensor = self._align_memory(tensor, alignment)[: layer_size]
                            kv_cache_raw_tensors[layer_name_inner] = tensor

                elif "attn" in layer_name and self.use_compress and layer_name not in kv_cache_raw_tensors:
                    if self.vllm_config.kv_transfer_config is None:
                        tensor = torch.zeros(kv_cache_tensor.size,
                                                dtype=torch.int8,
                                                device=self.device)
                    else:
                        cache_size_aligned = kv_cache_tensor.size + alignment
                        tensor = torch.zeros(cache_size_aligned, dtype=torch.int8, device=self.device)
                        tensor = self._align_memory(tensor, alignment)[: kv_cache_tensor.size]
                    for layer_name_inner in shared_layers:
                        # shared the kvcache between the self_attn specs in the same group
                        kv_cache_raw_tensors[layer_name_inner] = tensor
                elif (
                    isinstance(layer_kv_cache_spec[layer_name], AscendSFAIndexerCacheSpec)
                    and layer_name not in kv_cache_raw_tensors
                ):
                    current_kv_cache_spec = layer_kv_cache_spec[layer_name]
                    num_blocks = kv_cache_tensor.size // current_kv_cache_spec.page_size_bytes
                    if not use_legacy_shared_by_layout:
                        # vLLM #51718 packs all group layers into one tensor;
                        # kv_cache_config.num_blocks is the per-layer block count.
                        num_blocks = kv_cache_config.num_blocks
                    k_tensor_size = (
                        num_blocks
                        * current_kv_cache_spec.sfa_dcp_replicated_indexer_size
                        * current_kv_cache_spec.block_size
                        * current_kv_cache_spec.num_kv_heads
                        * current_kv_cache_spec.head_size
                        * get_dtype_size(current_kv_cache_spec.dtype)
                    )
                    if current_kv_cache_spec.scale_dim:
                        scale_tensor_size = (
                            num_blocks
                            * current_kv_cache_spec.sfa_dcp_replicated_indexer_size
                            * current_kv_cache_spec.block_size
                            * current_kv_cache_spec.num_kv_heads
                            * current_kv_cache_spec.scale_dim
                            * get_dtype_size(current_kv_cache_spec.scale_dtype)
                        )
                    else:
                        scale_tensor_size = None

                    if use_legacy_shared_by_layout:
                        # v0.27.1 and Ascend DSV4 descriptors alias physical blocks.
                        if scale_tensor_size is not None:
                            kv_cache_raw_tensors[shared_layers[0]] = self._allocate_sparse_c8_indexer_tensors(
                                dsa_k_tensor_size=k_tensor_size,
                                dsa_k_scale_tensor_size=scale_tensor_size,
                                alignment=alignment,
                                scale_dtype=current_kv_cache_spec.scale_dtype,
                            )
                        else:
                            kv_cache_raw_tensors[shared_layers[0]] = (
                                self._allocate_int8_cache_tensor(k_tensor_size, alignment),
                            )
                        for layer_name_inner in shared_layers[1:]:
                            kv_cache_raw_tensors[layer_name_inner] = kv_cache_raw_tensors[shared_layers[0]]
                    else:
                        # main: every layer owns its own region.
                        for layer_name_inner in shared_layers:
                            if scale_tensor_size is not None:
                                kv_cache_raw_tensors[layer_name_inner] = self._allocate_sparse_c8_indexer_tensors(
                                    dsa_k_tensor_size=k_tensor_size,
                                    dsa_k_scale_tensor_size=scale_tensor_size,
                                    alignment=alignment,
                                    scale_dtype=current_kv_cache_spec.scale_dtype,
                                )
                            else:
                                kv_cache_raw_tensors[layer_name_inner] = (
                                    self._allocate_int8_cache_tensor(k_tensor_size, alignment),
                                )
                elif "attn" in layer_name and layer_name not in kv_cache_raw_tensors and not use_mamba:
                    # NOTE: We need to init k cache tensor (nope cache tensor in mla) and
                    # v cache tensor (rope cache tensor in mla) separately to support prefill disaggregation,
                    # as it only support the 0-dim of kv_cache is `num_blocks`.
                    # For deepseek mla, we need to spilt cache tensor accrodding to the nope head dim
                    # and rope head dim.
                    current_kv_cache_spec = layer_kv_cache_spec[layer_name]
                    assert isinstance(current_kv_cache_spec, AttentionSpec)
                    current_sparse_sfa_c8 = self.use_sparse and kv_cache_spec_uses_sparse_sfa_c8(
                        current_kv_cache_spec
                    )

                    # vLLM #51718 packs every layer of a group into a single
                    # KVCacheTensor on main; the per-layer size is the block
                    # count times this layer's own page size. Using the layer's
                    # spec page (rather than dividing the tensor) stays correct
                    # when the tensor's group is not the largest group.
                    kv_cache_tensor_size = (
                        kv_cache_tensor.size
                        if use_legacy_shared_by_layout
                        else kv_cache_config.num_blocks * current_kv_cache_spec.page_size_bytes
                    )
                    if current_sparse_sfa_c8:
                        k_tensor_size = kv_cache_tensor_size
                        v_tensor_size = None
                    else:
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        assert k_dim > 0 and v_dim > 0
                        kv_head_dim_list = [
                            k_dim,
                            v_dim,
                        ]
                        if not self.use_sparse and enable_fa_quant(self.vllm_config):
                            k_tensor_split_factor, v_tensor_split_factor = (
                                self.vllm_config.quant_config.get_kv_quant_split_factor(layer_name, kv_head_dim_list)
                            )
                        else:
                            k_tensor_split_factor, v_tensor_split_factor = calc_split_factor(kv_head_dim_list)
                        k_tensor_size = int(kv_cache_tensor_size // k_tensor_split_factor)
                        v_tensor_size = int(kv_cache_tensor_size // v_tensor_split_factor)
                    if self.sparse_kv_offload_enabled:
                        assert self.use_sparse, "Sparse KV offload only support sparse attention."
                        assert not current_sparse_sfa_c8, "Sparse KV offload do not support sparse SFA C8."
                        assert v_tensor_size is not None
                        if use_legacy_shared_by_layout:
                            assert len(shared_layers) == 1, "Sparse KV offload do not support HMA."
                            kv_cache_raw_tensors[layer_name] = (
                                allocate_kv_cache_tensors_for_sparse_kv_offload(
                                    k_tensor_size,
                                    v_tensor_size,
                                    alignment,
                                    self.tp_rank,
                                    self.sparse_kv_offload_config.keep_device_kv_cache,
                                    self._allocate_int8_cache_tensor,
                                )
                            )
                        else:
                            for layer_name_inner in shared_layers:
                                if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                                    kv_cache_raw_tensors[layer_name_inner] = (
                                        allocate_kv_cache_tensors_for_sparse_kv_offload(
                                            k_tensor_size,
                                            v_tensor_size,
                                            alignment,
                                            self.tp_rank,
                                            self.sparse_kv_offload_config.keep_device_kv_cache,
                                            self._allocate_int8_cache_tensor,
                                        )
                                    )
                        continue
                    if use_legacy_shared_by_layout:
                        # v0.27.1 and Ascend DSV4 descriptors alias physical
                        # blocks; allocate once and reuse for every shared layer.
                        v_tensor = None
                        k_tensor = self._allocate_int8_cache_tensor(
                            k_tensor_size,
                            alignment,
                        )
                        if v_tensor_size is not None:
                            v_tensor = self._allocate_int8_cache_tensor(
                                v_tensor_size,
                                alignment,
                            )
                        for layer_name_inner in shared_layers:
                            # shared the attn kvcache for all shared layers
                            if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                                if current_sparse_sfa_c8:
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor,)
                                else:
                                    assert v_tensor is not None
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
                    else:
                        # main: every layer owns its own region; give each layer a
                        # private (k, v) so block indices don't collide across layers.
                        for layer_name_inner in shared_layers:
                            if "attn" in layer_name_inner and "linear_attn" not in layer_name_inner:
                                k_tensor = self._allocate_int8_cache_tensor(
                                    k_tensor_size,
                                    alignment,
                                )
                                v_tensor = None
                                if v_tensor_size is not None:
                                    v_tensor = self._allocate_int8_cache_tensor(
                                        v_tensor_size,
                                        alignment,
                                    )
                                if current_sparse_sfa_c8:
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor,)
                                else:
                                    assert v_tensor is not None
                                    kv_cache_raw_tensors[layer_name_inner] = (k_tensor, v_tensor)
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"

        return kv_cache_raw_tensors

    def _adjust_kv_layout(
        self,
        raw_tensor: torch.Tensor,
        kv_cache_shape_list: list[int],
        kv_cache_dtype_list: list[int],
        page_size_bytes: int,
        overlap_full_kv_cache: bool = False,
    ):
        reshaped_kv_tensors = []
        base_storage_offset_bytes = raw_tensor.storage_offset()
        storage_offset_bytes = base_storage_offset_bytes
        for idx, (shape, dtype) in enumerate(zip(kv_cache_shape_list, kv_cache_dtype_list)):
            if overlap_full_kv_cache and idx == 2:
                storage_offset_bytes = base_storage_offset_bytes
            dtype_size = get_dtype_size(dtype)
            num_element_per_page = (
                page_size_bytes // dtype_size
            )

            stride = torch.empty(shape).stride()
            target_stride = (num_element_per_page, *stride[1:])
            assert storage_offset_bytes % dtype_size == 0
            tensor = torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=target_stride,
                storage_offset=storage_offset_bytes // dtype_size,
            )
            reshaped_kv_tensors.append(tensor)
            storage_offset_bytes += stride[0] * dtype_size
        return reshaped_kv_tensors


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
        layer_kv_cache_spec = self._get_layer_kv_cache_specs(kv_cache_config)
        for group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            current_kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue

                current_kv_cache_spec = layer_kv_cache_spec[layer_name]

                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if self.use_compress and isinstance(current_kv_cache_spec,
                                                    (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)):
                    kv_tensor = kv_cache_raw_tensors[layer_name]
                    sum_page_size_bytes = kv_tensor.numel()
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
                    assert num_blocks == kv_cache_config.num_blocks, \
                        f"num_blocks: {num_blocks} should be equal to " \
                        f"kv_cache_config.num_blocks: {kv_cache_config.num_blocks}"
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, current_kv_cache_spec.storage_block_size,
                        current_kv_cache_spec.num_kv_heads,
                        current_kv_cache_spec.head_size)
                    kv_cache_shape_list = [kv_cache_shape]
                    kv_cache_dtype_list = [current_kv_cache_spec.dtype]
                    overlap_full_kv_cache = False

                    if hasattr(current_kv_cache_spec, "scale_dim") and current_kv_cache_spec.scale_dim != 0:
                        indexer_k_shape = kv_cache_shape
                        indexer_scale_shape = self.attn_backend.get_kv_cache_shape(
                                                num_blocks, current_kv_cache_spec.storage_block_size,
                                                current_kv_cache_spec.num_kv_heads,
                                                current_kv_cache_spec.scale_dim
                                                )
                        if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
                            indexer_full_shape = self.attn_backend.get_kv_cache_shape(
                                num_blocks, current_kv_cache_spec.storage_block_size,
                                current_kv_cache_spec.num_kv_heads,
                                current_kv_cache_spec.head_size
                                + current_kv_cache_spec.scale_dim
                                * get_dtype_size(current_kv_cache_spec.scale_dtype))
                            kv_cache_shape_list = [
                                indexer_k_shape, indexer_scale_shape, indexer_full_shape
                            ]
                            kv_cache_dtype_list = [
                                current_kv_cache_spec.dtype,
                                current_kv_cache_spec.scale_dtype,
                                current_kv_cache_spec.dtype,
                            ]
                            overlap_full_kv_cache = True
                        else:
                            kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]
                            kv_cache_dtype_list = [
                                current_kv_cache_spec.dtype, current_kv_cache_spec.scale_dtype
                            ]
                            overlap_full_kv_cache = False

                    kv_cache = self._adjust_kv_layout(kv_tensor,
                                           kv_cache_shape_list,
                                           kv_cache_dtype_list,
                                           current_kv_cache_spec.page_size_bytes,
                                           overlap_full_kv_cache=overlap_full_kv_cache,
                                           )

                    kv_caches[layer_name] = kv_cache
                elif isinstance(current_kv_cache_spec, AscendSFAIndexerCacheSpec):
                    raw_cache = kv_cache_raw_tensors[layer_name]
                    assert isinstance(raw_cache, tuple)
                    if current_kv_cache_spec.scale_dim:
                        raw_k_tensor, raw_scale_tensor = raw_cache
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_scale_tensor.numel()
                    else:
                        (raw_k_tensor,) = raw_cache
                        raw_scale_tensor = None
                        sum_page_size_bytes = raw_k_tensor.numel()

                    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
                    assert num_blocks >= kv_cache_config.num_blocks

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks * current_kv_cache_spec.sfa_dcp_replicated_indexer_size,
                        current_kv_cache_spec.block_size,
                        current_kv_cache_spec.num_kv_heads,
                        current_kv_cache_spec.head_size,
                    )
                    indexer_k_cache = raw_k_tensor.view(current_kv_cache_spec.dtype).view(kv_cache_shape)
                    if raw_scale_tensor is None:
                        kv_caches[layer_name] = (indexer_k_cache,)
                    else:
                        indexer_scale_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * current_kv_cache_spec.sfa_dcp_replicated_indexer_size,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.scale_dim,
                        )
                        indexer_scale_cache = (
                            raw_scale_tensor
                            .view(current_kv_cache_spec.scale_dtype)
                            .view(indexer_scale_cache_shape)
                        )
                        kv_caches[layer_name] = (indexer_k_cache, indexer_scale_cache)
                elif isinstance(current_kv_cache_spec, AttentionSpec):
                    # cache_only_layers (extract_hidden_states) are allocated
                    # as a single tensor by the branch at the top of
                    # _allocate_kv_cache_tensors; route them to the dedicated
                    # elif branch below before the sparse branch tries to
                    # unpack them as a K/V tuple.
                    current_sparse_sfa_c8 = self.use_sparse and kv_cache_spec_uses_sparse_sfa_c8(
                        current_kv_cache_spec
                    )
                    if self.sparse_kv_offload_enabled:
                        assert self.use_sparse, "Sparse KV offload only support sparse attention."
                        assert not current_sparse_sfa_c8, "Sparse KV offload do not support sparse SFA C8."
                        reshaped_tensors = reshape_kv_cache_tensors_for_sparse_kv_offload(
                            kv_cache_raw_tensors[layer_name],
                            current_kv_cache_spec,
                            attn_backend,
                            self.tp_rank,
                            self.vllm_config,
                            self.sparse_kv_offload_config,
                        )
                        kv_caches[layer_name] = reshaped_tensors
                        continue
                    raw_kv_is_combined = False
                    if self.use_sparse and "cache_only_layers" not in layer_name:
                        raw_cache = kv_cache_raw_tensors[layer_name]
                        assert isinstance(raw_cache, tuple)
                        if current_sparse_sfa_c8:
                            (raw_k_tensor,) = raw_cache
                            raw_v_tensor = None
                            sum_page_size_bytes = raw_k_tensor.numel()
                        else:
                            raw_k_tensor, raw_v_tensor = raw_cache
                            sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    elif (
                        self.hybrid_with_attn_and_mamba
                        and "cache_only_layers" not in layer_name
                        and not is_hidden_state_cache_spec(current_kv_cache_spec)
                        and isinstance(kv_cache_raw_tensors[layer_name], torch.Tensor)
                        and (self.use_hybrid_blocks or not vllm_version_is("0.27.1"))
                    ):
                        # Currently, we ensure that the same kvcache format is used even if there
                        # is no shared layer, such as the full attention mtp layer of qwen3.5, etc.
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[layer_name], kv_cache_raw_tensors[layer_name]
                        sum_page_size_bytes = raw_k_tensor.numel()
                        raw_kv_is_combined = True
                    elif (
                        "cache_only_layers" in layer_name
                        or is_hidden_state_cache_spec(current_kv_cache_spec)
                    ):
                        # Single tensor for extract_hidden_states (no K/V split).
                        # The anchor's upstream CacheOnlyAttentionBackend lacks
                        # get_kv_cache_shape (added upstream later); the cache-only
                        # vLLM #51718 standardized cache views as
                        # (blocks, kv_heads, block_size, head).
                        raw_tensor = kv_cache_raw_tensors[layer_name]
                        assert raw_tensor is not None
                        assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                        num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
                        assert num_blocks >= kv_cache_config.num_blocks
                        if vllm_version_is("0.27.1"):
                            kv_cache_shape = (
                                num_blocks,
                                current_kv_cache_spec.block_size,
                                current_kv_cache_spec.num_kv_heads,
                                current_kv_cache_spec.head_size,
                            )
                        else:
                            kv_cache_shape = (
                                num_blocks,
                                current_kv_cache_spec.num_kv_heads,
                                current_kv_cache_spec.block_size,
                                current_kv_cache_spec.head_size,
                            )
                        raw_tensor = raw_tensor.view(current_kv_cache_spec.dtype)
                        page_size_padded = getattr(
                            current_kv_cache_spec, "page_size_padded", None
                        )
                        if page_size_padded is not None:
                            # The cache-only page is aligned to the hybrid common
                            # page, so each block has trailing padding. Stride the
                            # block dim (dim 0) by the full padded page to skip it
                            # (cf. upstream GPUModelRunner page_size_padded view).
                            dtype_size = get_dtype_size(current_kv_cache_spec.dtype)
                            page_stride = page_size_padded // dtype_size
                            strides = [1] * len(kv_cache_shape)
                            for dim_idx in range(len(kv_cache_shape) - 2, -1, -1):
                                strides[dim_idx] = strides[dim_idx + 1] * kv_cache_shape[dim_idx + 1]
                            strides[0] = page_stride
                            k_cache = torch.as_strided(
                                raw_tensor, size=kv_cache_shape, stride=tuple(strides)
                            )
                        else:
                            k_cache = raw_tensor.view(kv_cache_shape)
                        kv_caches[layer_name] = k_cache
                        continue  # Skip the rest of the AttentionSpec handling
                    else:
                        raw_k_tensor, raw_v_tensor = kv_cache_raw_tensors[  # type: ignore
                            layer_name
                        ]
                        sum_page_size_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
                    assert raw_k_tensor is not None
                    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes

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

                        block_size_chunk = current_kv_cache_spec.block_size // block_size
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks * block_size_chunk,
                            block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                    else:
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            num_blocks,
                            current_kv_cache_spec.block_size,
                            current_kv_cache_spec.num_kv_heads,
                            current_kv_cache_spec.head_size,
                        )
                    should_trim_page_padding = (
                        self.hybrid_with_attn_and_mamba and self.use_hybrid_blocks
                    ) or (
                        not vllm_version_is("0.27.1")
                        and (
                            raw_kv_is_combined
                            or getattr(current_kv_cache_spec, "page_size_padded", None) is not None
                        )
                    )
                    if should_trim_page_padding:
                        # vLLM #51718 groups KVCacheTensor.layers by spec, so an
                        # attention tensor no longer has to list a Mamba layer
                        # even though its page is padded to the hybrid common
                        # size. Use the main-lane page contract, rather than the
                        # legacy shared-layer heuristic, to strip that padding.
                        if not isinstance(current_kv_cache_spec, AscendMLAAttentionSpec):
                            attn_tensor_size = int(np.prod(kv_cache_shape[1:])) * get_dtype_size(
                                current_kv_cache_spec.dtype
                            )
                            v_tensor_size = (
                                int(np.prod(kv_cache_shape[1:-1]))
                                * current_kv_cache_spec.head_size_v
                                * get_dtype_size(current_kv_cache_spec.dtype)
                            )
                            if raw_kv_is_combined:
                                padding_size = raw_k_tensor.numel() - attn_tensor_size - v_tensor_size
                                assert padding_size >= 0
                                raw_kv_tensor = raw_k_tensor[padding_size:]
                                raw_k_tensor = raw_kv_tensor[:attn_tensor_size]
                                raw_v_tensor = raw_kv_tensor[attn_tensor_size:]
                            else:
                                assert raw_v_tensor is not None
                                assert raw_k_tensor.numel() >= attn_tensor_size
                                assert raw_v_tensor.numel() >= v_tensor_size
                                raw_k_tensor = raw_k_tensor[:attn_tensor_size]
                                raw_v_tensor = raw_v_tensor[:v_tensor_size]
                        else:
                            k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                            nope_size = int(np.prod(kv_cache_shape[:-1])) * k_dim * get_dtype_size(
                                current_kv_cache_spec.dtype
                            )
                            rope_size = int(np.prod(kv_cache_shape[:-1])) * v_dim * get_dtype_size(
                                current_kv_cache_spec.dtype
                            )
                            if raw_kv_is_combined:
                                padding_size = raw_k_tensor.numel() - nope_size - rope_size
                                assert padding_size >= 0
                                raw_kv_tensor = raw_k_tensor[padding_size:]
                                raw_k_tensor = raw_kv_tensor[:nope_size]
                                raw_v_tensor = raw_kv_tensor[nope_size:]
                            else:
                                assert raw_v_tensor is not None
                                assert raw_k_tensor.numel() >= nope_size
                                assert raw_v_tensor.numel() >= rope_size
                                raw_k_tensor = raw_k_tensor[:nope_size]
                                raw_v_tensor = raw_v_tensor[:rope_size]
                    if not isinstance(current_kv_cache_spec, AscendMLAAttentionSpec):
                        k_shape = kv_cache_shape[1:]
                        if hasattr(current_kv_cache_spec, "head_size_v"):
                            v_shape = (*kv_cache_shape[1:-1], current_kv_cache_spec.head_size_v)
                        else:
                            v_shape = k_shape
                    else:
                        # k_cache: nope_cache    v_cache: rope_cache
                        mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
                        k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
                        k_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            k_dim,
                        )
                        if current_sparse_sfa_c8:
                            k_shape = (
                                mla_num_blocks,
                                mla_block_size,
                                num_kv_heads,
                                current_kv_cache_spec.head_size,
                            )
                            v_dim = 0
                        v_shape = (
                            mla_num_blocks,
                            mla_block_size,
                            num_kv_heads,
                            v_dim,
                        )
                    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype
                    if enable_fa_quant(self.vllm_config):
                        k_cache_dtype, v_cache_dtype = self.vllm_config.quant_config.get_kv_quant_dtype(
                            layer_name, current_kv_cache_spec.dtype, self.model_config
                        )

                    if current_sparse_sfa_c8:
                        k_cache_dtype = self.c8_k_cache_dtype

                    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)
                    if current_sparse_sfa_c8:
                        v_cache = None
                    else:
                        assert raw_v_tensor is not None
                        v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape)

                    if current_sparse_sfa_c8:
                        kv_caches[layer_name] = (k_cache,)
                    else:
                        assert v_cache is not None
                        kv_caches[layer_name] = (k_cache, v_cache)
                elif isinstance(current_kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor is not None
                    assert raw_tensor.numel() % current_kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // current_kv_cache_spec.page_size_bytes
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
                    # NOTE(zxr): in order to keep all tensor contiguous, we align ssm and kv block
                    # with same page size, so have to add extra padding block for kv, the overall
                    # layout of hybrid kv_cache on Ascend is:
                    # tensor1: [(kv_padding), conv           , ...]
                    # tensor2: [k           , ssm            , ...]
                    # tensor3: [v           , (mamba_padding), ...]
                    for shape, dtype in zip(current_kv_cache_spec.shapes, current_kv_cache_spec.dtypes):
                        # normally, there is conv state and ssm state in this loop. And there is only
                        # a conv state in some special models.
                        target_shape = (num_blocks, *shape)

                        target_idx += math.prod(target_shape) * get_dtype_size(dtype)
                        tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
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
        non_encoder_groups = [
            group
            for group in kv_cache_config.kv_cache_groups
            if not isinstance(group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]
        block_sizes = [group.kv_cache_spec.block_size for group in non_encoder_groups]

        # Generate kernel_block_sizes that matches each block_size
        # For attention backends that support virtual block splitting,
        # use the supported block sizes from the backend
        # For other backends (like Mamba), use [0] (no splitting)
        self.kernel_block_sizes = []
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
                attn_groups = self.attn_groups[kv_cache_group_id]
                backends = [attn_group.backend for attn_group in attn_groups]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = select_common_block_size(
                    kv_manager_block_size, backends
                )
                self.kernel_block_sizes.append([selected_kernel_size])
            else:
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                # NOTE: set kernel_block_sizes to 0 to disable slotmapping computation
                # of mamba block. In this case, BlockTable.block_size will never equal
                # to kernel_block_sizes[0]
                self.kernel_block_sizes.append([0])

        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in non_encoder_groups:
            max_num_blocks_per_req = kv_cache_group.kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config,
                max_model_len,
            )
            max_num_blocks.append(max_num_blocks_per_req)

        if (block_sizes != [self.cache_config.block_size]
                or self.kernel_block_sizes != [[self.cache_config.block_size]]
                or len(kv_cache_config.kv_cache_groups) > 1):
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = NPUInputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
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
                kernel_block_sizes=self.kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                kv_cache_groups=kv_cache_config.kv_cache_groups,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                reasoning_config = getattr(self.vllm_config, "reasoning_config", None),
            )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec
            use_mla_rope: bool | None

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)

            def backend_supports_kernel_block_size(
                attn_backend: type[AttentionBackend],
                block_size: int,
            ) -> bool:
                for supported_size in attn_backend.get_supported_kernel_block_sizes():
                    if isinstance(supported_size, int):
                        if block_size == supported_size:
                            return True
                    elif isinstance(supported_size, MultipleOf):
                        if block_size % supported_size.base == 0:
                            return True
                    else:
                        raise ValueError(f"Unknown supported size: {supported_size}")
                return False

            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                layer = layers[layer_name]
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                # Prefer the backend declared by the layer itself. Some
                # indexer-cache layers require their own metadata builder.
                attn_backend = layer.get_attn_backend()
                if (
                    isinstance(layer_kv_cache_spec, AscendSFAIndexerCacheSpec)
                    and not backend_supports_kernel_block_size(
                        attn_backend,
                        layer_kv_cache_spec.block_size,
                    )
                ):
                    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

                    attn_backend = AscendSFAIndexerBackend
                full_cls_name = attn_backend.full_cls_name()
                use_mla_rope = (
                    layer.impl.use_mla_rope
                    if issubclass(attn_backend, AscendMLABackend)
                    else None
                )
                key = (full_cls_name, layer_kv_cache_spec, use_mla_rope)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend,
                    layer_kv_cache_spec,
                    use_mla_rope,
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for group_key, layer_names in attn_backends_map.items():
                attn_backend = group_key.attn_backend
                kv_cache_spec = group_key.kv_cache_spec
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

        self._check_and_update_cudagraph_mode(
            attention_backend_list,
            kv_cache_config.kv_cache_groups,
        )

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            return {}

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

        # NOTE: Must process Attention/MLAAttention before MambaBase to maintain
        # ordering expected by graph parameter update logic in attention backends.
        mamba_layers: dict[str, MambaBase] = {}
        attn_layer_names = set()
        for layer_name, attn_module in attn_layers.items():
            if (isinstance(attn_module, Attention)
                    and (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            elif self.use_compress:
                # Skip modules that don't need KV cache (eg encoder-only attention)
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
            elif isinstance(attn_module, Attention):
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, MLAAttention):
                if self.use_sparse:
                    impl = attn_module.impl
                    cache_sparse_sfa_c8 = bool(
                        getattr(impl, "enable_sparse_sfa_c8", False)
                    )
                    if cache_sparse_sfa_c8:
                        head_size = get_sfa_qsfa_packed_head_dim(
                            self.model_config.hf_text_config.kv_lora_rank,
                            self.model_config.hf_text_config.qk_rope_head_dim,
                        )
                        dtype = self.c8_k_cache_dtype
                    else:
                        head_size = (
                            self.model_config.hf_text_config.kv_lora_rank
                            + self.model_config.hf_text_config.qk_rope_head_dim
                        )
                        dtype = self.kv_cache_dtype
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=self.block_size,
                        num_kv_heads=1,
                        head_size=head_size,
                        dtype=dtype,
                        cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                        cache_sparse_sfa_c8=cache_sparse_sfa_c8,
                        store_on_host=self.sparse_kv_offload_enabled,
                    )
                elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    if getattr(attn_module.impl, "fa_quant_layer", False):
                        head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                        dtype, cache_dtype_str = attn_module.impl.dtype, None
                    else:
                        head_size, dtype, cache_dtype_str = spec.head_size, spec.dtype, spec.cache_dtype_str
                    kv_cache_spec[layer_name] = AscendMLAAttentionSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=head_size,
                        dtype=dtype,
                        cache_dtype_str=cache_dtype_str,
                        non_causal_multi_token_decode=spec.non_causal_multi_token_decode,
                    )
                    attn_layer_names.add(layer_name)

            elif isinstance(attn_module, DeepseekV32IndexerCache):
                # TODO: This mirrors upstream's separated KV/indexer specs for
                # SFA, but keeps Ascend-specific shape/block-size accounting.
                # Remove this special case once the generic vLLM spec/backend
                # path can describe the Ascend SFA indexer layout directly.
                cache_sparse_li_c8 = self.ascend_config.is_sparse_li_c8_layer(layer_name)
                kv_cache_spec[layer_name] = AscendSFAIndexerCacheSpec(
                    block_size=self.block_size,
                    num_kv_heads=1,
                    head_size=self.model_config.hf_text_config.index_head_dim,
                    dtype=self.c8_k_cache_dtype if cache_sparse_li_c8 else self.kv_cache_dtype,
                    cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    scale_dim=1 if cache_sparse_li_c8 else 0,
                    scale_dtype=self.c8_k_scale_cache_dtype if cache_sparse_li_c8 else torch.int8,
                    cache_sparse_li_c8=cache_sparse_li_c8,
                    sfa_dcp_replicated_indexer_size=self.sfa_dcp_replicated_indexer_size,
                )

            elif isinstance(attn_module, MambaBase):
                mamba_layers[layer_name] = attn_module

            elif isinstance(attn_module, CacheOnlyAttentionLayer):
                # Only CacheOnlyAttentionLayer (extract_hidden_states draft model)
                # is handled here.
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                    # Rebuild to a fresh, picklable spec (the returned one
                    # references a stale MLAAttentionSpec class shadowed by
                    # patch_kv_cache_interface.py). Keep the HiddenStateCacheSpec
                    # type so get_kv_cache_groups isolates this cache-only layer
                    # into its own group; downgrading to MLAAttentionSpec would
                    # break page-size unification on hybrid models (e.g. Qwen3.5).
                    kv_cache_spec[layer_name] = HiddenStateCacheSpec(
                        block_size=spec.block_size,
                        num_kv_heads=spec.num_kv_heads,
                        head_size=spec.head_size,
                        dtype=spec.dtype,
                        cache_dtype_str=spec.cache_dtype_str,
                    )
                    attn_layer_names.add(layer_name)

            elif spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec
                if isinstance(spec, AttentionSpec):
                    attn_layer_names.add(layer_name)

        if len(mamba_layers) > 0:
            mamba_page_size_padded = 0
            for layer_name, mamba_module in mamba_layers.items():
                if spec := mamba_module.get_kv_cache_spec(self.vllm_config):
                    kv_cache_spec[layer_name] = spec
                    mamba_page_size_padded = spec.page_size_bytes
            # align attn_page_size to mamba_page_size_padded
            for layer_name in attn_layer_names:
                if kv_cache_spec[layer_name].page_size_bytes < mamba_page_size_padded:  # type: ignore[attr-defined]
                    object.__setattr__(kv_cache_spec[layer_name], "page_size_padded", mamba_page_size_padded)

        if self.sparse_kv_offload_enabled:
            self.kv_cache_spec = kv_cache_spec # reserve for Sparse KV offload usage
        return kv_cache_spec

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()
                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__

        with update_pass_config(self):
            tensor_parallel_size = self.parallel_config.tensor_parallel_size
            resolver_tensor_parallel_size = tensor_parallel_size
            if (
                self.compilation_config.pass_config.enable_sp
                and self.uniform_decode_query_len > 1
                and tensor_parallel_size > 1
            ):
                graph_alignment = math.lcm(
                    self.uniform_decode_query_len,
                    tensor_parallel_size,
                )
                capture_sizes = self.compilation_config.cudagraph_capture_sizes
                # vLLM 0.27 has no path for explicit capture sizes that are
                # already aligned to both speculative steps and SP. Skip its
                # redundant TP adjustment only for that exact case.
                if (
                    graph_alignment
                    > max(self.uniform_decode_query_len, tensor_parallel_size)
                    and capture_sizes
                    and all(size % graph_alignment == 0 for size in capture_sizes)
                ):
                    resolver_tensor_parallel_size = 1
            cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
                min_cg_support=min_cg_support,
                min_cg_attn_backend=min_cg_attn_backend,
                uniform_decode_query_len=self.uniform_decode_query_len,
                use_v2_model_runner=False,
                tensor_parallel_size=resolver_tensor_parallel_size,
                kv_cache_config=self.kv_cache_config,
                max_num_reqs=self.max_num_reqs,
            )
            self.cudagraph_dispatcher.initialize_cudagraph_keys(
                cudagraph_mode, self.uniform_decode_query_len
            )

        if (
            self.speculative_config
            and self.drafter is not None
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_extract_hidden_states()
            )
        ):
            assert isinstance(
                self.drafter,
                AscendEagleProposer | AscendDflashProposer | AscendExtractHiddenStatesProposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        capture_sizes = sorted({
            desc.num_tokens
            for _, descs in capture_descs
            for desc in descs
        })

        # NOTE: Since aclgraph_batch_sizes cannot be determined until here,
        # we set the graph params right before initializing the keys.
        if self.use_aclgraph:
            set_graph_params(capture_sizes)
            if self.speculative_config:
                set_draft_graph_params(capture_sizes)

    def capture_model(self) -> int:
        """Capture NPU graphs and return actual graph pool memory bytes consumed."""
        parent_module_name = _get_gpu_model_runner_module_name(self)
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
            cuda_graph_size = GPUModelRunner.capture_model(self)

        mgr = self.encoder_cudagraph_manager
        if mgr is not None and self.update_stream is not None:
            mgr.update_stream = self.update_stream

        return cuda_graph_size

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

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = AscendKVBlockZeroer(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[1, :].min().item())


def _get_gpu_model_runner_module_name(model_runner) -> str:
    """Return the module name of GPUModelRunner found in the MRO."""
    gpu_model_runner_cls = next(
        (cls for cls in model_runner.__class__.__mro__ if cls.__name__ == "GPUModelRunner"),
        None,
    )
    if gpu_model_runner_cls is None:
        raise TypeError(
            "Could not find GPUModelRunner in the MRO. "
            "The class hierarchy may have changed."
        )
    return gpu_model_runner_cls.__module__


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda *a, **kw: None
            self.synchronize = lambda *a, **kw: None
            self.wait = lambda *a, **kw: None
            self.query = lambda *a, **kw: True

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
        torch.cuda.is_current_stream_capturing = torch.npu.is_current_stream_capturing
        torch.cuda.CUDAGraph = torch.npu.NPUGraph
        breakable_cudagraph.weak_ref_tensor = weak_ref_tensor
        breakable_cudagraph.weak_ref_tensors = weak_ref_tensors
        yield
    except Exception as e:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.default_stream = _StreamPlaceholder
        torch.cuda.current_stream = _StreamPlaceholder
        torch.cuda.stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        torch.cuda.is_current_stream_capturing = lambda: False
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # Async model-runner outputs are created after runner initialization.
        # Keep the CUDA compatibility entry point backed by a real NPU event so
        # their non-blocking device-to-host copies retain synchronization.
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        torch.cuda.is_current_stream_capturing = torch.npu.is_current_stream_capturing
        torch.cuda.CUDAGraph = torch.npu.NPUGraph
        breakable_cudagraph.weak_ref_tensor = weak_ref_tensor
        breakable_cudagraph.weak_ref_tensors = weak_ref_tensors


# TODO: This method will be removed subsequently and implemented in platform.
@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    import vllm.v1.worker.encoder_cudagraph as _vllm_encoder_cudagraph

    from vllm_ascend.worker.encoder_acl_graph import EncoderAclGraphManager

    _encoder_mgr_orig = _vllm_encoder_cudagraph.EncoderCudaGraphManager
    _vllm_encoder_cudagraph.EncoderCudaGraphManager = EncoderAclGraphManager
    target_module = None
    try:
        target_module = sys.modules[target_module_name]
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        yield
    except Exception as e:
        raise RuntimeError(f"NPUModelRunner failed, error is {e}")
    finally:
        _vllm_encoder_cudagraph.EncoderCudaGraphManager = _encoder_mgr_orig
        if target_module is not None:
            setattr(target_module, "graph_capture", graph_capture)  # noqa: B010


@contextmanager
def update_pass_config(model_runner):
    try:
        original_pass_config_sp = model_runner.compilation_config.pass_config.enable_sp
        model_runner.compilation_config.pass_config.enable_sp = enable_sp(model_runner.vllm_config)
        yield
    finally:
        model_runner.compilation_config.pass_config.enable_sp = original_pass_config_sp
