# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-offload metadata buffers for autoregressive draft steps."""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    update_sparse_kv_offload_metadata,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_ascend.worker.v2.model_states.default import AscendModelState


class SparseKVOffloadMetadata:
    """Own stable per-step buffers and prepare sparse draft attention inputs."""

    def __init__(
        self,
        *,
        enabled: bool,
        num_steps: int,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ) -> None:
        self.req_ids_buffers: list[CpuGpuBuffer] = []
        self.token_to_req_buffers: list[CpuGpuBuffer] = []
        if enabled:
            pin_memory = is_pin_memory_available()
            for _ in range(num_steps):
                self.req_ids_buffers.append(
                    CpuGpuBuffer(max_num_reqs, dtype=torch.int64, device=device, pin_memory=pin_memory)
                )
                self.token_to_req_buffers.append(
                    CpuGpuBuffer(max_num_tokens, dtype=torch.int32, device=device, pin_memory=pin_memory)
                )

    def build_kwargs(
        self,
        input_batch: "InputBatch",
        model_state: "AscendModelState",
        num_reqs: int,
        batch_desc: "BatchExecutionDescriptor",
        query_start_loc_np: np.ndarray,
        step: int,
    ) -> dict[str, Any] | None:
        if not self.req_ids_buffers:
            return None
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens = int(query_start_loc_np[num_reqs])
        num_tokens_padded = batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else num_tokens
        req_ids_buffer = self.req_ids_buffers[step]
        token_to_req_buffer = self.token_to_req_buffers[step]
        req_ids = input_batch.req_ids[:num_reqs]
        if getattr(input_batch, "is_dummy", False):
            req_ids = [f"offload-dummy-{row}" for row in range(num_reqs)]
        update_sparse_kv_offload_metadata(
            num_tokens,
            num_reqs,
            num_tokens_padded,
            num_reqs_padded,
            req_ids,
            query_start_loc_np,
            req_ids_buffer,
            token_to_req_buffer,
        )
        pool_slots = model_state._offload_pool_slots
        pool_active = model_state._offload_pool_active
        return {
            "req_ids_tensor": req_ids_buffer.gpu[:num_reqs_padded],
            "token_to_req": token_to_req_buffer.gpu[:num_tokens_padded],
            "req_topk_buffer_slots": pool_slots.cpu[:num_reqs_padded] if pool_slots is not None else None,
            "req_topk_buffer_active": pool_active.cpu[:num_reqs_padded] if pool_active is not None else None,
            "copy_sfa_draft_index": step,
            "offload_dummy": getattr(input_batch, "is_dummy", False),
        }
