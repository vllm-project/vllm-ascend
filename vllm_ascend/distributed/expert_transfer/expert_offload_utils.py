# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class ExpertMapBuffer:
    """Reuse host/device maps; serialized executions own their contents."""

    def __init__(self, num_experts: int, device: torch.device):
        self.host_map = torch.zeros(num_experts, dtype=torch.long, device="cpu", pin_memory=device.type == "npu")
        # Keep a zero-copy CPU view so updates create no temporary mapping tensors.
        self._host_array = self.host_map.numpy()
        self.device_map = torch.empty(num_experts, dtype=torch.long, device=device)

    def update(self, slot_to_expert: tuple[int, ...]) -> None:
        self._host_array.fill(-1)
        for slot, expert_id in enumerate(slot_to_expert):
            self._host_array[expert_id] = slot

    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        self.device_map.copy_(self.host_map, non_blocking=True)
        return self.device_map[topk_ids.long()].to(topk_ids.dtype)


@dataclass(frozen=True)
class PreparedExpertTensor:
    """Pinned physical bytes plus the native tensor layout they encode."""

    data: torch.Tensor
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


@contextmanager
def expert_offload_weights(layer, ascend_config):
    """Keep native expert creation on CPU, then attach target-only offload."""
    expert_offload_config = getattr(ascend_config, "expert_offload_config", None)
    if expert_offload_config is None or not expert_offload_config.enabled:
        yield
        return

    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        yield
        return

    with torch.device("cpu"):
        yield

    from .expert_offload_manager import LayerOffloadBinding
    from .expert_offload_method import ExpertOffloadMoEMethod

    layer.quant_method = ExpertOffloadMoEMethod(layer.quant_method, layer.moe_config, expert_offload_config)
    layer._expert_offload_binding = LayerOffloadBinding(layer, layer.quant_method.quant_method, expert_offload_config)


def get_manager(model: torch.nn.Module | None) -> ExpertOffloadManager | None:
    return cast("ExpertOffloadManager | None", getattr(model, "_expert_offload_manager", None))


def validate_loading(runner: NPUModelRunner, load_dummy_weights: bool) -> None:
    expert_offload_config = getattr(getattr(runner, "ascend_config", None), "expert_offload_config", None)
    if load_dummy_weights and expert_offload_config is not None and expert_offload_config.enabled:
        raise ValueError("expert offload requires checkpoint shard tracking; dummy checkpoint loading is unsupported")


def initialize(runner: NPUModelRunner) -> None:
    """Initialize enabled offload after native warmup joins, before dumper setup."""
    expert_offload_config = getattr(getattr(runner, "ascend_config", None), "expert_offload_config", None)
    if expert_offload_config is None or not expert_offload_config.enabled:
        return

    from .expert_offload_manager import initialize_expert_offload

    initialize_expert_offload(runner)


def prepare_batch(runner: NPUModelRunner, batch_state) -> None:
    expert_offload_manager = get_manager(getattr(runner, "model", None))
    if expert_offload_manager is not None:
        # Dummy batches have no request state and use the full expert pool.
        expert_offload_manager.has_prefill = batch_state is None or batch_state.has_prefill


def prepare_graph_capture(graph_manager: ModelAclGraphManager, needs_capture: Callable[[], bool]) -> None:
    expert_offload_manager = get_manager(getattr(graph_manager.model_runner, "model", None))
    if expert_offload_manager is not None:
        expert_offload_manager.prepare_graph_capture(graph_manager, needs_capture)


def shutdown(runner: NPUModelRunner) -> None:
    expert_offload_manager = get_manager(getattr(runner, "model", None))
    if expert_offload_manager is not None:
        expert_offload_manager.unsubscribe_graph_streams()
        del runner.model._expert_offload_manager
