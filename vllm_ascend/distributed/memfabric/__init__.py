# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local resource coordination for MemFabric APIs."""

from vllm_ascend.distributed.memfabric.runtime import (
    MEMFABRIC_API_BM,
    MEMFABRIC_API_MEMCACHE,
    MEMFABRIC_API_OFFLOAD,
    MEMFABRIC_API_TRANSFER_ENGINE,
    MemFabricRuntimeLease,
    MemFabricRuntimeRegistry,
    global_memfabric_runtime_registry,
)

__all__ = [
    "MEMFABRIC_API_BM",
    "MEMFABRIC_API_MEMCACHE",
    "MEMFABRIC_API_OFFLOAD",
    "MEMFABRIC_API_TRANSFER_ENGINE",
    "MemFabricRuntimeLease",
    "MemFabricRuntimeRegistry",
    "global_memfabric_runtime_registry",
]
