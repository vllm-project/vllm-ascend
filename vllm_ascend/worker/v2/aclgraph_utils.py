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
from typing import Any
from collections.abc import Callable

import torch
import torch.nn as nn
from tqdm import tqdm
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager, CudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup
from vllm.logger import logger
from vllm.model_executor.offloader.base import get_offloader

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params


# class AclGraphManager(CudaGraphManager):
#     """ACL Cuda Graph Manager for Ascend NPUs."""

#     @torch.inference_mode()
#     def capture(
#         self,
#         create_forward_fn: Callable[[BatchExecutionDescriptor], Callable[[CUDAGraphMode], None]],
#         progress_bar_desc: str = "Capturing CUDA graphs",
#     ) -> None:
#         """Override capture method to set capturing flag to True before capture."""
#         with graph_capture(device=self.device):
#             # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
#             # activations so FULL activations should fit in already allocated
#             # buffers in the graph pool.
#             for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
#                 if mode not in self._capture_descs:
#                     continue

#                 descs = self._capture_descs[mode]
#                 if is_global_first_rank():
#                     descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
#                 for desc in descs:
#                     # Prepare inputs and get forward function
#                     forward_fn = create_forward_fn(desc)

#                     # Warmup
#                     forward_fn(CUDAGraphMode.NONE)

#                     # Capture
#                     logger.debug("CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc)
#                     # Set capturing flag to True before capture, this is only needed for vllm-ascend.
#                     _EXTRA_CTX.capturing = True
#                     if desc.cg_mode == CUDAGraphMode.PIECEWISE:
#                         forward_fn(CUDAGraphMode.PIECEWISE)
#                     else:
#                         assert desc not in self.graphs, f"Graph already captured for {desc}"
#                         graph = torch.cuda.CUDAGraph()
#                         # Sync offloader's copy stream before capture.
#                         # Ensure any pre-capture prefetches from offloader are complete.
#                         get_offloader().sync_prev_onload()
#                         with torch.cuda.graph(graph, self.pool):
#                             forward_fn(CUDAGraphMode.NONE)
#                             # Join offloader's copy stream after forward to avoid
#                             # unjoined stream error. The last layer's start_prefetch
#                             # forks copy_stream, but wait_prefetch only happens in
#                             # the next forward pass.
#                             get_offloader().join_after_forward()
#                         self.graphs[desc] = graph
#         self._graphs_captured = True


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner: Any,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
        )
        # set model runner attribute, so we can access attributes model runner
        # when call `run_fullgraph` method in CudaGraphManager,
        # then we don't need to # copy `execute_model` method in `NPUModelRunner` class.
        self.model_runner = model_runner
        # capture_sizes sorts in ascending order.
        self.capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        # vllm-ascend need to update graph params of attention backend.
        # so we need to set graph params before capture full graph.
        if super().needs_capture():
            set_graph_params(self.capture_sizes)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        logger.info_once(f"run_fullgraph with num_tokens={num_tokens}")
        ret = super().run_fullgraph(desc)
        assert self.model_runner.cudagraph_and_dp_padding is not None

        positions = self.model_runner.input_buffers.positions[:num_tokens]
        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.Tensor([num_tokens] * self.model_runner.dp_size, device=self.device)
        with set_forward_context(
            self.model_runner.input_batch.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=self.model_runner.input_batch.slot_mappings,
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                list(self.model_runner.attn_backends.values())[0],
                self.model_runner.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
                positions.shape[0],
            )
        return ret
