# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .residency import ExpertResidency

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class ExpertMapBuffer:
    """Reuse host/device maps; serialized executions own their contents."""

    def __init__(self, num_experts: int, device: torch.device):
        self.host_map = torch.zeros(num_experts, dtype=torch.long, device="cpu", pin_memory=device.type == "npu")
        # Keep a zero-copy CPU view so updates create no temporary mapping tensors.
        self._host_array = self.host_map.numpy()
        self.device_map = torch.empty(num_experts, dtype=torch.long, device=device)

    def update(self, residency: ExpertResidency) -> None:
        self._host_array.fill(-1)
        for slot, expert_id in enumerate(residency.slot_to_expert):
            if expert_id is not None:
                self._host_array[expert_id] = slot

    def remap(self, selected_ids: torch.Tensor) -> torch.Tensor:
        self.device_map.copy_(self.host_map, non_blocking=True)
        return self.device_map[selected_ids.long()].to(selected_ids.dtype)


@dataclass(frozen=True)
class ExpertTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class PreparedExpertTensor:
    """Pinned physical bytes plus the native tensor layout they encode."""

    data: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    npu_format: int
    expert_bytes: int
    extra_data: tuple[torch.Tensor, ...] = ()

    @property
    def num_experts(self) -> int:
        if not self.extra_data:
            return self.data.shape[0]
        return self.data.shape[0] * len(self.extra_data) + self.extra_data[-1].shape[0]

    def source_chunk(self, expert_id: int) -> tuple[torch.Tensor, int]:
        chunk_id, row = divmod(expert_id, self.data.shape[0])
        chunk = self.data if chunk_id == 0 else self.extra_data[chunk_id - 1]
        return chunk, row


class ExpertWeightView:
    """Pass one consistent weight bundle to both GEMMs without rebinding Parameters."""

    def __init__(self, layer, tensors: dict[str, torch.Tensor]):
        self._layer = layer
        self.__dict__.update(tensors)
        self.ascend_expert_map = None

    def __getattr__(self, name):
        return getattr(self._layer, name)


def finalize_expert_offload_loading(runner: NPUModelRunner) -> None:
    """Finalize enabled offload after native warmup joins, before dumper setup."""
    allocated_before = torch.npu.memory_allocated(runner.device)
    runner.expert_offload_manager.finalize_loading(runner.model)
    # The shared pool is allocated after the parent's loading profiler.
    runner.model_memory_usage += torch.npu.memory_allocated(runner.device) - allocated_before
