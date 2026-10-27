# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device-only adapters around upstream's IPC weight cache."""

import sys


def apply_preload_patch() -> None:
    # Lazy imports avoid the loader/platform import cycle during discovery.
    from vllm.model_executor.model_loader.weight_cache import daemon, ipc_loader, protocol

    if not hasattr(daemon, "exported_weight_bytes") or not hasattr(daemon.WeightCacheDaemon, "release_distributed"):
        return

    from vllm_ascend.model_loader.preload.daemon import AscendWeightCacheDaemon, run_ascend_daemon
    from vllm_ascend.model_loader.preload.transport import NPUTensorEntry

    if hasattr(daemon, "_run_daemon_original"):
        return
    daemon._run_daemon_original = daemon._run_daemon
    protocol._check_ipc_platform_support_original = protocol.check_ipc_platform_support
    daemon._exported_weight_bytes_original = daemon.exported_weight_bytes
    daemon._run_daemon = run_ascend_daemon
    daemon.WeightCacheDaemon = AscendWeightCacheDaemon
    daemon.TensorEntry = NPUTensorEntry
    daemon.exported_weight_bytes = exported_weight_bytes
    protocol.check_ipc_platform_support = check_ipc_platform_support
    ipc_loader.check_ipc_platform_support = check_ipc_platform_support
    cli = sys.modules.get("vllm.entrypoints.cli.preload")
    if cli is not None:
        cli.check_ipc_platform_support = check_ipc_platform_support


def check_ipc_platform_support() -> None:
    from vllm.model_executor.model_loader.weight_cache import protocol
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        return
    protocol._check_ipc_platform_support_original()


def exported_weight_bytes(model) -> int:
    from vllm.model_executor.model_loader.weight_cache import daemon

    from vllm_ascend.model_loader.preload.transport import exported_npu_weight_bytes

    return daemon._exported_weight_bytes_original(model) + exported_npu_weight_bytes(model)
