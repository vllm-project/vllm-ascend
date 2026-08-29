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
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import (
    ContextSource,
    UpdatableGraph,
)
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.utils import communicator_switch


def _prepare_pcp_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    _block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
    *,
    pcp_manager: Any,
) -> cudagraph_utils.AttentionState:
    """Build graph inputs with the same PCP-local layout used on replay."""
    if vllm_version_is("0.27.1"):
        input_batch = cudagraph_utils.InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    else:
        input_batch = cudagraph_utils.InputBatch.make_dummy(
            num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
        )
    input_batch = pcp_manager.partition_batch(input_batch)
    input_block_tables, slot_mappings = pcp_manager.prepare_attn(input_batch)
    slot_mappings_by_layer = cudagraph_utils.build_slot_mappings_by_layer(slot_mappings, kv_cache_config)

    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return cudagraph_utils.AttentionState(attn_metadata, slot_mappings_by_layer)


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


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            if backend.get_impl_cls() is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    if vllm_version_is("0.27.1"):

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            model_runner: Any,
            lora_capture_cases: list[int] | None = None,
        ):
            super().__init__(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                lora_capture_cases=lora_capture_cases,
            )
            self.model_runner = model_runner
            self.update_stream = self.model_runner.update_stream

    else:

        def __init__(  # type: ignore[misc]
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            model_runner: Any,
            lora_capture_cases: list[int] | None = None,
            varlen_decode: bool = False,
        ):
            super().__init__(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,
            )
            self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
            self.model_runner = model_runner
            self.update_stream = self.model_runner.update_stream

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(
            self, desc: BatchExecutionDescriptor
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a full graph and update its registered tasks."""
        assert self.update_stream is not None
        
        graph = self.graphs[desc]
        assert isinstance(graph, UpdatableGraph)
        # Resolve every provider before replay so a provider failure cannot
        # leave the graph blocked on an update event that will never be recorded.
        resolved_tasks = graph.resolve_tasks(
            ContextSource(self.model_runner.model_state.attn_metadata)
        )
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
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        model = ModelWithContext(model)
        pcp_manager = getattr(self.model_runner, "pcp_manager", None)
        if pcp_manager is not None:
            cudagraph_utils.prepare_inputs_to_capture = partial(
                _prepare_pcp_inputs_to_capture,
                pcp_manager=pcp_manager,
            )
        with communicator_switch():
            return super().capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc=progress_bar_desc,
            )


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill

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

        return self.original_model(*args, **kwargs)

    def get_original_model(self):
        return self.original_model

    def compute_logits(self, hidden_states: torch.Tensor):
        # draft model has `compute_logits`, which is not in ModelWithContext
        return self.original_model.compute_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor):
        return self.original_model.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor):
        return self.original_model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor):
        return self.original_model.markov_bias(markov_embed)

    def map_draft_to_target(self, draft_ids: torch.Tensor):
        return self.original_model.map_draft_to_target(draft_ids)

    def embed_input_ids(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.original_model.embed_input_ids(input_ids, *args, **kwargs)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
