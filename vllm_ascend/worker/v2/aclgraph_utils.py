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
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params
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


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

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
        # set model runner attribute, so we can access attributes model runner
        # when call `run_fullgraph` method in CudaGraphManager,
        # then we don't need to # copy `execute_model` method in `NPUModelRunner` class.
        self.model_runner = model_runner
        # The attention backend keys its per-size graph params by the actual
        # captured token counts (rounded up to decode_query_len when using
        # speculative decoding), so derive them from the capture descriptors
        # instead of the raw config sizes.
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        # vllm-ascend need to update graph params of attention backend.
        # so we need to set graph params before capture full graph.
        if super().needs_capture():
            set_graph_params(self.capture_sizes)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        logger.info_once("run_fullgraph with num_tokens=%s", num_tokens)
        ret = super().run_fullgraph(desc)

        positions = self.model_runner.input_buffers.positions[:num_tokens]
        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        with set_forward_context(
            self.model_runner.model_state.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=None,
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                self.model_runner.attn_groups[0][0].backend,
                self.model_runner.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
                positions.shape[0],
            )
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
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        if torch.npu.is_current_stream_capturing():
            _EXTRA_CTX.capturing = True
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


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
