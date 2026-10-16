# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend weight-transfer backend registration."""

from vllm.distributed.weight_transfer.factory import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)


def register_engine() -> None:
    """Register Ascend weight transfer engines as vLLM plugins."""
    WeightTransferEngineFactory.register_engine(
        "hccl",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLWeightTransferEngine",
    )
    WeightTransferEngineFactory.register_engine(
        "npu_ipc",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCWeightTransferEngine",
    )
    WeightTransferTrainerFactory.register_engine(
        "npu_ipc",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCTrainerWeightTransferEngine",
    )
    WeightTransferTrainerFactory.register_engine(
        "hccl",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLTrainerWeightTransferEngine",
    )
