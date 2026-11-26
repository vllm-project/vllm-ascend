# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU device operations over the upstream IPC weight-transfer lifecycle."""

import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import torch
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)

_NPU_IPC_DEVICE_INDEX = 6
if not hasattr(IPCTrainerWeightTransferEngine, "_get_device_key"):
    raise RuntimeError("NPU IPC requires a vLLM build with the device extension hooks from vllm-project/vllm#70027.")

# Existing import names share the upstream wire schema.
NPUIPCWeightTransferInitInfo = IPCWeightTransferInitInfo
NPUIPCWeightTransferUpdateInfo = IPCWeightTransferUpdateInfo


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Identify the colocated physical NPU through ASCEND_RT_VISIBLE_DEVICES."""
    if logical_device is None:
        logical_device = torch.accelerator.current_device_index()
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible_devices:
        physical_device = int(visible_devices.split(",")[logical_device].strip())
    else:
        physical_device = logical_device
    return f"{get_ip()}-{physical_device}"


@dataclass
class NPUIPCTrainerInitInfo(IPCTrainerInitInfo):
    """Select the registered NPU backend with the common IPC configuration."""

    backend: ClassVar[str] = "npu_ipc"


class NPUIPCWeightTransferEngine(IPCWeightTransferEngine):
    """Inherit receiving and reload; supply only NPU identity and rebuilding."""

    def _get_device_key(self) -> str:
        return npu_generate_uuid(self.device.index)

    def _rebuild_ipc_tensor(self, args: tuple) -> torch.Tensor:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        if len(args) <= _NPU_IPC_DEVICE_INDEX:
            raise ValueError(f"NPU IPC rebuild arguments do not contain a device index: got {len(args)} values")
        local_args = list(args)
        local_args[_NPU_IPC_DEVICE_INDEX] = self.device.index
        return rebuild_npu_tensor(*local_args)


class NPUIPCTrainerWeightTransferEngine(IPCTrainerWeightTransferEngine):
    """Inherit initialization and sending; supply NPU identity and completion."""

    init_info_cls = NPUIPCTrainerInitInfo

    def _get_device_key(self) -> str:
        return npu_generate_uuid(self.device_index)

    @classmethod
    def _post_send_sync(cls) -> None:
        cls._barrier()
        torch.npu.synchronize()
