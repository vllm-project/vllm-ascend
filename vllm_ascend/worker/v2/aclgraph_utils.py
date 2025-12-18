# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.cudagraph_utils import \
    prepare_inputs_to_capture as prepare_inputs_to_capture_gpu
from vllm.v1.worker.gpu.input_batch import InputBuffers

from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class AclGraphManager(CudaGraphManager):
    """ACL Graph Manager for Ascend NPUs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)

    def capture_graph(
        self,
        num_tokens: int,
        model: nn.Module,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        with (torch_cuda_wrapper(), prepare_capture_inputs_wrapper()):
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
