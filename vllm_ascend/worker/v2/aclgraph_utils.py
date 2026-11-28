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
from collections.abc import Callable, Hashable, Mapping
from contextlib import contextmanager
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
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.distributed.expert_transfer import expert_offload_utils as expert_offload
from vllm_ascend.worker.v2.utils import communicator_switch


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


class AscendGraphManagerMixin:
    vllm_config: VllmConfig
    breakable_cg_runner: BreakableACLGraphWrapper | None

    def init_breakable_cg_runner(self, model: nn.Module, key_fn: Callable[..., Hashable] | None = None) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config, key_fn=key_fn)


class ModelAclGraphManager(AscendGraphManagerMixin, ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_lens: Mapping[int, int],
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
            decode_query_lens,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )
        self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
        self.model_runner = model_runner
        self.update_stream = self.model_runner.update_stream
        expert_offload.prepare_graph_capture(self, super().needs_capture)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        assert self.update_stream is not None
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
            graph = self.graphs[desc]
            assert isinstance(graph, UpdatableGraph)
            resolved_tasks = graph.resolve_tasks(ContextSource(attn_metadata))
            self.update_stream.wait_stream(torch.npu.current_stream())
            ret = super().run_fullgraph(desc)
            graph.update(self.update_stream, resolved_tasks)
            return ret
        finally:
            if metadata is not None:
                metadata.finish_replay()

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        # vLLM #53869 supplies PCP slot mappings during graph capture.
        pcp_manager: Any = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        metadata = getattr(model_state, "device_metadata", None)
        model = ModelWithContext(model, device_metadata=metadata)
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
