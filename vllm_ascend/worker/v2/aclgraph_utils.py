# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.utils import (
    align_cudagraph_capture_sizes,
    align_num_tokens_to_tp,
    get_tp_graph_token_alignment,
    use_updatable_graph,
    vllm_version_is,
)
from vllm_ascend.worker.v2.utils import communicator_switch


def collect_sorted_captured_token_sizes(capture_descs: dict) -> list[int]:
    """Collect the actual per-graph token counts that will be captured.

    With speculative decoding under FULL_DECODE_ONLY, each raw
    ``cudagraph_capture_size`` is rounded up to a multiple of
    ``decode_query_len`` (see ``CudaGraphManager._init_candidates``), so the
    real graph sizes differ from ``compilation_config.cudagraph_capture_sizes``.
    The attention backend keys its per-size graph params (events/handles/...)
    by these rounded token counts, so they must be derived from the actual
    capture descriptors, not the raw config sizes.
    """
    return sorted({desc.num_tokens for descs in capture_descs.values() for desc in descs})


def align_decode_capture_request_sizes(
    vllm_config: VllmConfig,
    sizes: list[int],
    query_lens: Mapping[int, int],
) -> list[int]:
    """Normalize a request-keyed graph grid to TP-aligned token shapes."""
    tp_size = get_tp_graph_token_alignment(vllm_config)
    if tp_size <= 1 or not sizes:
        return sizes
    if not query_lens:
        return []
    request_alignment = math.lcm(
        *(tp_size // math.gcd(tp_size, query_len) for query_len in query_lens)
    )
    compilation_config = vllm_config.compilation_config
    max_tokens = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        compilation_config.max_decode_cudagraph_capture_tokens
        or vllm_config.scheduler_config.max_num_batched_tokens,
    )
    safe_limit = min(
        (min(max_reqs, max_tokens // query_len) // request_alignment)
        * request_alignment
        for query_len, max_reqs in query_lens.items()
    )
    return align_cudagraph_capture_sizes(
        sizes, request_alignment, safe_limit
    )


def normalize_tp_aligned_token_descriptors(
    vllm_config: VllmConfig, graph_manager: Any
) -> None:
    """Normalize final token-keyed graph descriptors and their dispatch table.

    Upstream builds some private token grids after platform configuration
    (notably adaptive-verification varlen decode). Normalize those final
    descriptors here, without changing the shared MRV2 implementation.
    Request-keyed uniform-decode descriptors are left to the request-grid
    normalization and final validator because their token/request ratio is
    part of the graph ABI.
    """
    tp_size = get_tp_graph_token_alignment(vllm_config)
    if tp_size <= 1:
        return

    replacements: dict[
        BatchExecutionDescriptor, BatchExecutionDescriptor
    ] = {}
    canonical: dict[
        BatchExecutionDescriptor, BatchExecutionDescriptor
    ] = {}
    for descs in graph_manager._capture_descs.values():
        for desc in descs:
            aligned_tokens = align_num_tokens_to_tp(
                desc.num_tokens, tp_size
            )
            normalized = desc
            if (
                aligned_tokens != desc.num_tokens
                and desc.uniform_token_count is None
            ):
                num_reqs = desc.num_reqs
                if desc.max_query_len is not None and num_reqs is not None:
                    num_reqs = max(
                        num_reqs,
                        math.ceil(aligned_tokens / desc.max_query_len),
                    )
                    if num_reqs > graph_manager.max_num_reqs:
                        raise ValueError(
                            "Cannot TP-align graph token gear "
                            f"{desc.num_tokens} to {aligned_tokens}: "
                            f"{num_reqs} requests exceed max_num_reqs="
                            f"{graph_manager.max_num_reqs}"
                        )
                normalized = replace(
                    desc,
                    num_tokens=aligned_tokens,
                    num_reqs=num_reqs,
                )
            normalized = canonical.setdefault(normalized, normalized)
            replacements[desc] = normalized

    for mode, descs in graph_manager._capture_descs.items():
        normalized_descs = []
        seen = set()
        for desc in descs:
            normalized = replacements[desc]
            if normalized not in seen:
                normalized_descs.append(normalized)
                seen.add(normalized)
        graph_manager._capture_descs[mode] = normalized_descs

    for key, candidates in graph_manager._candidates.items():
        normalized_candidates = []
        seen = set()
        for desc in candidates:
            normalized = replacements.get(desc, desc)
            if normalized not in seen:
                normalized_candidates.append(normalized)
                seen.add(normalized)
        graph_manager._candidates[key] = normalized_candidates


def validate_tp_aligned_capture_descriptors(
    vllm_config: VllmConfig, capture_descs: Mapping
) -> None:
    """Fail before capture if any final graph descriptor violates TP layout."""
    tp_size = get_tp_graph_token_alignment(vllm_config)
    if tp_size <= 1:
        return
    invalid_sizes = sorted(
        {
            desc.num_tokens
            for descs in capture_descs.values()
            for desc in descs
            if align_num_tokens_to_tp(desc.num_tokens, tp_size)
            != desc.num_tokens
        }
    )
    if invalid_sizes:
        raise ValueError(
            "All ACL graph token gears must be divisible by tensor parallel "
            f"size {tp_size}; invalid final descriptor sizes: {invalid_sizes}"
        )


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            try:
                impl_cls = backend.get_impl_cls()
            except NotImplementedError:
                # Metadata-only backends such as GDN have no attention impl.
                continue
            if impl_cls is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int | Mapping[int, int],
        model_runner: Any,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: UBatchRunner | None = None,
    ):
        # vLLM main (#51700) passes the microbatch runner into the graph
        # manager.
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )
        normalize_tp_aligned_token_descriptors(vllm_config, self)
        validate_tp_aligned_capture_descriptors(
            vllm_config, self._capture_descs
        )
        self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
        self.model_runner = model_runner
        self.update_stream = self.model_runner.update_stream
        expert_offload.prepare_graph_capture(self, super().needs_capture)
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        if super().needs_capture():
            set_graph_params(self.capture_sizes)

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        assert self.update_stream is not None
        with set_current_vllm_config(self.vllm_config):
            attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
        attn_metadata = self.model_runner.model_state.attn_metadata

        metadata = getattr(self.model_runner.model_state, "device_metadata", None)
        try:
            prepare_engram = getattr(self.model_runner.model_state, "prepare_engram", None)
            if prepare_engram is not None:
                # Match FULL replay's DP padding before lookup consumes it.
                with (
                    set_current_vllm_config(self.vllm_config),
                    set_forward_context(
                        attn_metadata,
                        self.vllm_config,
                        num_tokens=num_tokens,
                        cudagraph_runtime_mode=desc.cg_mode,
                        num_tokens_across_dp=torch.full([self.model_runner.dp_size], num_tokens),
                        batch_descriptor=None,
                        slot_mapping=None,
                    ),
                ):
                    prepare_engram()
            if use_updatable_graph(attn_backend):
                return self._updatable_graph_replay(desc, attn_metadata)
            else:
                # This will be removed once the refactoring is fully complete.
                return self._graph_relay(attn_backend, desc, num_tokens, attn_metadata)
        finally:
            if metadata is not None:
                metadata.finish_replay()

    def _graph_relay(self, attn_backend, desc, num_tokens, attn_metadata):
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        # Attention graph updates must see the target ModelConfig.
        # set_forward_context() does not update the current VllmConfig.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
            )
        logger.info_once("ACL graph replay is active for the V2 target model (logged once).")
        return ret

    def _updatable_graph_replay(self, desc, attn_metadata):
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        resolved_tasks = graph.resolve_tasks(ContextSource(attn_metadata))
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)
        graph.update(self.update_stream, resolved_tasks)
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        # vLLM #53869 supplies PCP slot mappings during graph capture.
        pcp_manager: Any = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        metadata = getattr(model_state, "device_metadata", None)
        model = ModelWithContext(model, device_metadata=metadata)
        legacy_kwargs = {"has_lora": has_lora} if vllm_version_is("0.33.0") else {}
        try:
            with communicator_switch():
                return super().capture(
                    model,
                    model_state,
                    input_buffers,
                    intermediate_tensors,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    pcp_manager=pcp_manager,
                    **legacy_kwargs,
                    use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                    lora_capture_hook=lora_capture_hook,
                    progress_bar_desc=progress_bar_desc,
                )
        finally:
            # Input preparation can fail before ModelWithContext.forward runs.
            if metadata is not None:
                metadata.finish()


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False, device_metadata=None):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill
        self.device_metadata = device_metadata

    def forward(self, *args, **kwargs):
        forward_context = get_forward_context()
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        _EXTRA_CTX.capturing = (
            torch.npu.is_current_stream_capturing()
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        if self.is_draft_model:
            _EXTRA_CTX.is_draft_model = True
        if self.is_draft_model_prefill:
            _EXTRA_CTX.is_draft_model_prefill = True

        try:
            if self.device_metadata is not None:
                self.device_metadata.begin_forward()
            return self.original_model(*args, **kwargs)
        finally:
            # Warmup and actual capture receive separate prepare_attn calls.
            # Retire each submission, including unused producers, after its
            # consumers; do not wait for metadata at graph entry.
            if self.device_metadata is not None:
                self.device_metadata.finish()

    def get_original_model(self):
        return self.original_model

    def __getattr__(self, name: str):
        # vLLM #57632 moved the draft context-K/V precompute into the
        # capture path, which reaches model methods through this wrapper.
        try:
            return super().__getattr__(name)
        except AttributeError:
            original_model = self._modules.get("original_model")
            if original_model is None:
                raise
            return getattr(original_model, name)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
