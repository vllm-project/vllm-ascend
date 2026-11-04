# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse Core's prefill plan with NPU-safe column selection."""

import torch
from torch.overrides import TorchFunctionMode
from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan
from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan as build_core_prefill_plan


class _PrefillColumnSelection(TorchFunctionMode):
    """Keep a local workaround for the NPU singleton-column indexing defect.

    torch-npu 2.10.0.post7 / CANN 9.2.1 can interleave rows for ``x[:, ids]``
    when x has one column. index_select has the same semantics and returns
    correct rows. Remove this scope when the supported device stack is fixed.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.__getitem__:
            tensor, index = args
            if (
                tensor.device.type == "npu"
                and tensor.ndim == 2
                and isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], slice)
                and index[0] == slice(None)
                and isinstance(index[1], torch.Tensor)
                and index[1].ndim == 1
            ):
                return torch.index_select(tensor, 1, index[1])
        return func(*args, **(kwargs or {}))


class AscendHiSparsePrefillStagingPlan(HiSparsePrefillStagingPlan):
    def ensure_gpu_sources(self, resident_state_rows, state_indices, resident_block_size) -> None:
        # Core retains source-key caching, representative-page selection and
        # current Resident/Host masks. The override affects only this call.
        with _PrefillColumnSelection():
            super().ensure_gpu_sources(resident_state_rows, state_indices, resident_block_size)


def build_hisparse_prefill_staging_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    staging_block_capacity: int,
) -> AscendHiSparsePrefillStagingPlan:
    plan = build_core_prefill_plan(block_table, seq_lens, block_size, staging_block_capacity)
    return AscendHiSparsePrefillStagingPlan(
        block_table=plan.block_table,
        row_ids=plan.row_ids,
        dst_rows=plan.dst_rows,
        miss_mask=plan.miss_mask,
        block_size=plan.block_size,
    )
