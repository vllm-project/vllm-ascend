# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent primitives for MemFabric GVA-backed tables."""

from vllm_ascend.distributed.gva.base import GvaBackend, GvaObject, GvaPool
from vllm_ascend.distributed.gva.layout import GvaRowPartition, GvaRowRange
from vllm_ascend.distributed.gva.types import (
    GvaCapabilities,
    GvaObjectRef,
    GvaObjectSpec,
    GvaPoolMember,
    GvaPoolRef,
    GvaPoolSpec,
)

__all__ = [
    "GvaBackend",
    "GvaCapabilities",
    "GvaObject",
    "GvaObjectRef",
    "GvaObjectSpec",
    "GvaPoolMember",
    "GvaPool",
    "GvaPoolRef",
    "GvaPoolSpec",
    "GvaRowPartition",
    "GvaRowRange",
]
