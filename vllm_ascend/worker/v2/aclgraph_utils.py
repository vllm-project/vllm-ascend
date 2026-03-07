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
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import vllm
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture as prepare_inputs_to_capture_gpu
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.utils import torch_cuda_wrapper
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.compilation.acl_graph import update_full_graph_params


class AclGraphManager(CudaGraphManager):
    """ACL Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_mrope: bool,
        use_aux_hidden_state_outputs: bool,
        device: torch.device,
        model_runner: Any,  # NPUModelRunner type, in case circular import, so we pass it as Any
    ):
        # set model runner attribute, so we can access attributes model runner
        # when call `run_fullgraph` method in CudaGraphManager,
        # then we don't need to # copy `execute_model` method in `NPUModelRunner` class.
        self.model_runner = model_runner
        with torch_cuda_wrapper():
            super().__init__(
                vllm_config,
                use_mrope,
                use_aux_hidden_state_outputs,
                device,
            )

    def _capture_full_graph(
        self,
        num_tokens: int,
        num_reqs: int,
        model: nn.Module,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        has_lora: bool = False,
    ) -> None:
        """Override _capture_full_graph because we need to set capturing=True in forward context."""
        # set capturing=True in before model forward.
        model = ModelWithContext(model)
        return super()._capture_full_graph(
            num_tokens,
            num_reqs,
            model,
            input_ids,
            positions,
            inputs_embeds,
            num_tokens_across_dp,
            attn_metadata,
            slot_mappings,
            has_lora,
        )

    def capture_graph(
        self,
        num_tokens: int,
        capture_cg_mode: CUDAGraphMode,
        model: nn.Module,
        input_buffers: InputBuffers,
        mrope_positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        uniform_decode: bool = False,
    ) -> None:
        with torch_cuda_wrapper(), prepare_capture_inputs_wrapper():
            super().capture_graph(
                num_tokens,
                capture_cg_mode,
                model,
                input_buffers,
                mrope_positions,
                inputs_embeds,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora,
                uniform_decode,
            )

    def run_fullgraph(self, num_tokens: int) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        ret = super().run_fullgraph(num_tokens)
        assert self.model_runner.cudagraph_and_dp_padding is not None

        positions = self.model_runner.input_buffers.positions[:num_tokens]
        _num_tokens_after_padding, num_tokens_across_dp, synced_cudagraph_mode = (
            self.model_runner.cudagraph_and_dp_padding
        )
        cudagraph_runtime_mode = CUDAGraphMode(synced_cudagraph_mode)

        with set_forward_context(
            self.model_runner.input_batch.attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,  # Full graph model don't need batch_descriptor
            slot_mapping=self.model_runner.input_batch.slot_mappings,
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                self.model_runner.attn_backend[0],  # FIXME(Ronald1995): support hybrid attn backend
                self.model_runner.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
                positions.shape[0],
            )
        return ret


@contextmanager
def prepare_capture_inputs_wrapper():
    """Context manager to override input preparation for NPU graph capture."""
    # TODO(Ronald1995): make prepare_inputs_to_capture as static method
    # in CudaGraphManager.
    ori = vllm.v1.worker.gpu.cudagraph_utils.prepare_inputs_to_capture
    try:
        vllm.v1.worker.gpu.cudagraph_utils.prepare_inputs_to_capture = prepare_inputs_to_capture
        yield
    finally:
        vllm.v1.worker.gpu.cudagraph_utils.prepare_inputs_to_capture = ori


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    max_model_len: int,
    kv_cache_config: KVCacheConfig,
    uniform_decode_query_len: int = 0,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if uniform_decode_query_len > 0:
        num_tokens_per_req = uniform_decode_query_len
    else:
        num_tokens_per_req = num_tokens // num_reqs

    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32) * num_tokens_per_req
    query_start_loc_np[-1] = num_tokens
    query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
    input_buffers.query_start_loc[: num_reqs + 1] = query_start_loc_cpu
    input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
    query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

    # HACK(woosuk): For faster warmup, we set seq_lens (GPU) to num_tokens
    # rather than max_model_len.
    input_buffers.seq_lens[:num_reqs] = num_tokens
    input_buffers.seq_lens[num_reqs:] = 0
    input_buffers.seq_lens_cpu[:num_reqs] = num_tokens
    input_buffers.seq_lens_cpu[num_reqs:] = 0

    input_buffers.dcp_local_seq_lens[:num_reqs] = num_tokens
    input_buffers.dcp_local_seq_lens[num_reqs:] = 0

    input_block_tables = [x[:num_reqs] for x in block_tables.input_block_tables]
    slot_mappings = block_tables.slot_mappings[:, :num_tokens]
    slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, kv_cache_config)

    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=num_tokens_per_req,
        seq_lens=input_buffers.seq_lens,
        max_seq_len=max_model_len,
        block_tables=input_block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_np=input_buffers.seq_lens_np,
    )
    return attn_metadata, slot_mappings_by_layer


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, *args, **kwargs):
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        forward_context = get_forward_context()
        forward_context.additional_kwargs.setdefault("capturing", True)

        return self.original_model(*args, **kwargs)
