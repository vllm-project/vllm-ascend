# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration helpers for Ascend weight transfer engines."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import WeightTransferEngine

EngineLoader = Callable[[], "type[WeightTransferEngine]"]

ASCEND_WEIGHT_TRANSFER_BACKENDS: dict[str, tuple[str, str]] = {
    "hccl": (
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLWeightTransferEngine",
    ),
    "npu_ipc": (
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCWeightTransferEngine",
    ),
}

ASCEND_WEIGHT_TRANSFER_ALIASES: dict[str, str] = {
    "nccl": "hccl",
    "ipc": "npu_ipc",
}


def _make_lazy_loader(module_path: str, class_name: str) -> EngineLoader:
    def loader() -> "type[WeightTransferEngine]":
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    return loader


def register_ascend_weight_transfer_engines(
    include_upstream_aliases: bool = True,
    override_existing: bool = True,
) -> None:
    """Register Ascend weight transfer engines and compatibility aliases."""
    registry = WeightTransferEngineFactory._registry

    loaders = {
        name: _make_lazy_loader(module_path, class_name)
        for name, (module_path, class_name) in ASCEND_WEIGHT_TRANSFER_BACKENDS.items()
    }

    for name, loader in loaders.items():
        if override_existing or name not in registry:
            registry[name] = loader

    if not include_upstream_aliases:
        return

    for alias, backend_name in ASCEND_WEIGHT_TRANSFER_ALIASES.items():
        if override_existing or alias not in registry:
            registry[alias] = loaders[backend_name]
