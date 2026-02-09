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
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture as prepare_inputs_to_capture_gpu
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class AclGraphManager(CudaGraphManager):
    """ACL Graph Manager for Ascend NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_mrope: bool,
        device: torch.device,
    ):
        with torch_cuda_wrapper():
            super().__init__(vllm_config, use_mrope, device)

    def capture_graph(
        self,
        num_tokens: int,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        with torch_cuda_wrapper(), prepare_capture_inputs_wrapper():
            super().capture_graph(
                num_tokens,
                model,
                input_buffers,
                block_tables,
                attn_metadata_builders,
                kv_cache_config,
            )


@contextmanager
def prepare_capture_inputs_wrapper():
    """Context manager to override input preparation for NPU graph capture."""
    # TODO(Ronald1995): make prepare_inputs_to_capture as static method
    # in CudaGraphManager.
    global prepare_inputs_to_capture_gpu
    try:
        ori_func = prepare_inputs_to_capture_gpu
        prepare_inputs_to_capture_gpu = prepare_inputs_to_capture
        yield
    finally:
        prepare_inputs_to_capture_gpu = ori_func


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_metadata_builders: list[AttentionMetadataBuilder],
    max_model_len: int,
    kv_cache_config: KVCacheConfig,
) -> dict[str, Any]:
    # TODO(Ronald1995): Implement NPU specific input preparation.
    return {}
