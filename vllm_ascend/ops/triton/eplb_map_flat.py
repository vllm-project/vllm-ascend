# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from importlib.util import find_spec

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.fused_moe.eplb import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS

_SMALL_ASSIGNMENT_LIMIT = 2048
_MEDIUM_ASSIGNMENT_LIMIT = 8192
_SMALL_TILE = 64
_MEDIUM_TILE = 128
_LARGE_TILE = 1024


def _mapping_block_size(numel: int) -> int:
    if numel <= _SMALL_ASSIGNMENT_LIMIT:
        return _SMALL_TILE
    return _MEDIUM_TILE if numel <= _MEDIUM_ASSIGNMENT_LIMIT else _LARGE_TILE


@triton.jit(do_not_specialize=["numel"])
def _map_to_physical_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    num_logical_experts,
    numel,
    routing_table_rows,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1)
    valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
    safe_logical_id = tl.where(valid_logical_id, logical_id, 0)
    routing_row = (offsets // TOPK) % routing_table_rows
    routing_index = routing_row * num_logical_experts + safe_logical_id
    physical_id = tl.load(
        routing_table_ptr + routing_index,
        mask=mask & valid_logical_id,
        other=-1,
    )
    tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)


# The original EPLB branch uses the worker warmup hook. Newer Ascend
# releases collect the same variants through the model JIT registry.
if find_spec("vllm_ascend.ops.triton.ascend_jit_warmup") is not None:
    from dataclasses import dataclass
    from typing import Any

    from vllm.model_executor.warmup.jit_warmup import kernel_launcher
    from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor

    from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

    class EplbMapKernel(AscendTritonJitKernel["EplbMapKernel.CompileKey"]):
        kernel = _map_to_physical_kernel

        @dataclass(frozen=True)
        class CompileKey:
            experts: int
            topk: int
            block: int
            ids_dtype: torch.dtype

        def dispatch(
            self, *, experts: int, topk: int, block: int, ids_dtype: torch.dtype
        ) -> "EplbMapKernel.CompileKey":
            return self.CompileKey(experts, topk, block, ids_dtype)

        def get_warmup_keys(self, experts: int, topk: int) -> list["EplbMapKernel.CompileKey"]:
            # The EPLB table builder produces a contiguous INT32 table of fixed height.
            # Routing and custom routing functions can supply INT32 or INT64 IDs.
            return [
                self.dispatch(experts=experts, topk=topk, block=block, ids_dtype=dtype)
                for block in (_SMALL_TILE, _MEDIUM_TILE, _LARGE_TILE)
                for dtype in (torch.int32, torch.int64)
            ]

        def warmup_inputs(self, compile_key: "EplbMapKernel.CompileKey") -> dict[str, Any]:
            ids = TritonWarmupTensor(compile_key.ids_dtype)
            return dict(
                topk_ids=ids,
                routing_table=TritonWarmupTensor(torch.int32),
                physical_ids=ids,
                num_logical_experts=compile_key.experts,
                numel=compile_key.block,
                routing_table_rows=EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
                TOPK=compile_key.topk,
                BLOCK_SIZE=compile_key.block,
            )

        @kernel_launcher
        def __call__(
            self,
            topk_ids,
            routing_table,
            physical_ids,
            num_logical_experts,
            numel,
            routing_table_rows,
            TOPK,
            BLOCK_SIZE,
        ) -> LaunchSpec:
            return (triton.cdiv(numel, BLOCK_SIZE),), {}

    _EPLB_MAP_KERNEL = EplbMapKernel()
else:
    from vllm_ascend.ops.triton.eplb_map_worker_warmup import WorkerMapKernel

    _EPLB_MAP_KERNEL = WorkerMapKernel(
        _map_to_physical_kernel,
        (_SMALL_TILE, _MEDIUM_TILE, _LARGE_TILE),
        EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
    )


def map_to_physical_triton(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map logical IDs to physical IDs without collecting expert load."""
    if topk_ids.numel() == 0:
        return topk_ids

    physical_ids = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    topk = topk_ids.shape[1]
    block_size = _mapping_block_size(numel)
    _EPLB_MAP_KERNEL(
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_replica_routing_table.shape[1],
        numel,
        expert_replica_routing_table.shape[0],
        TOPK=topk,
        BLOCK_SIZE=block_size,
    )
    return physical_ids
