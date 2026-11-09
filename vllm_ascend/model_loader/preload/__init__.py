# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend adapters for the upstream IPC weight cache."""


def register_preload_loader() -> None:
    # General plugins also run in spawned workers, before model construction.
    from vllm.model_executor import model_loader
    from vllm.model_executor.model_loader.weight_cache import daemon
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

    if not hasattr(daemon, "exported_weight_bytes") or not hasattr(daemon.WeightCacheDaemon, "release_distributed"):
        # Ordinary loading on older supported vLLM releases stays available.
        return

    from .loader import AscendIpcModelLoader

    registry = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    current = registry.get("ipc_cache")
    if current is AscendIpcModelLoader:
        return
    if current is IpcModelLoader:
        # Expected platform adaptation of the upstream loader, not a conflict.
        registry["ipc_cache"] = AscendIpcModelLoader
        return

    # Preserve upstream warnings for unexpected conflicts with other plugins.
    model_loader.register_model_loader("ipc_cache")(AscendIpcModelLoader)
