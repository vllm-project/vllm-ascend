# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared no-sleep BF16 Direct matrix transaction."""

from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.hccl import run_hccl_weight_transfer_transaction
from tests.e2e.common.rlhf.weight_transfer.npu_ipc import run_npu_ipc_weight_transfer_transaction


def run_direct_reload_transaction(case: WeightUpdateModelCase, backend: str, packed: bool, execution_mode: str):
    options = dict(
        reload_mode="direct",
        execution_mode=execution_mode,
        cold_start=True,
        tensor_parallel_size=case.tensor_parallel_size,
        gpu_memory_utilization=0.65,
    )
    if backend == "hccl":
        run_hccl_weight_transfer_transaction(case, packed, enable_sleep=False, **options)
    elif backend == "npu_ipc":
        run_npu_ipc_weight_transfer_transaction(case, packed, sleep_level=None, **options)
    else:
        raise ValueError(f"Unsupported Direct transport: {backend}")
