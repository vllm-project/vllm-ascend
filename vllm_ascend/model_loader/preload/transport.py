# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import itertools
from dataclasses import dataclass

import torch
import torch_npu
from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry

ND_FORMAT = 2
IPC_DEVICE_INDEX = 6


@dataclass
class NPUTensorEntry(TensorEntry):
    """An upstream tensor entry backed by a torch_npu IPC allocation."""

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, kind: str) -> TensorEntry:
        if tensor.device.type != "npu":
            return TensorEntry.from_tensor(tensor, kind)
        tensor = tensor.detach()
        if torch_npu.get_npu_format(tensor) != ND_FORMAT:
            raise ValueError("Ascend preload only shares ND tensors; NZ IPC is unsupported")
        # torch_npu installs its reducer on this PyTorch module. Resolve the
        # patched function at export time instead of binding the CUDA reducer.
        _, ipc_args = torch.multiprocessing.reductions.reduce_tensor(tensor)
        return cls(kind=kind, ipc_args=ipc_args)

    def rebuild(self, device_index: int) -> torch.Tensor:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        if self.ipc_args is None:
            raise ValueError("NPU IPC entry is missing its storage handle")
        args = list(self.ipc_args)
        args[IPC_DEVICE_INDEX] = device_index
        tensor = rebuild_npu_tensor(*args)
        if tensor.device.type != "npu" or torch_npu.get_npu_format(tensor) != ND_FORMAT:
            raise ValueError("Rebuilt preload tensor is not an ND NPU tensor")
        return tensor


def exported_npu_weight_bytes(model: torch.nn.Module | None) -> int:
    """Count mapped storages once, excluding daemon-only runtime allocations."""
    if model is None:
        return 0
    storages = {}
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.device.type == "npu":
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())
