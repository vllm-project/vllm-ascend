# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend adapters for the upstream IPC weight cache."""


def register_preload_loader() -> None:
    # General plugins also run in spawned workers, before model construction.
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.weight_cache import daemon

    if not hasattr(daemon, "exported_weight_bytes") or not hasattr(daemon.WeightCacheDaemon, "release_distributed"):
        # Ordinary loading on older supported vLLM releases stays available.
        return

    from .loader import AscendIpcModelLoader

    register_model_loader("ipc_cache")(AscendIpcModelLoader)
