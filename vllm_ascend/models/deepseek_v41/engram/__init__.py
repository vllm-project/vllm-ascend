# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Engram for DeepSeek V4.1.

Split the way upstream vLLM splits it for other accelerators: ``common`` holds
the platform-independent n-gram hashing and gating, ``npu`` holds the storage
and lookup the NPU runs.
"""

from .common import PagedNgramHistory, engram_enabled, engram_gate
from .npu import EngramQueryGroup, NodeShardedEngram, engram_cpu_offload

__all__ = [
    "EngramQueryGroup",
    "NodeShardedEngram",
    "PagedNgramHistory",
    "engram_cpu_offload",
    "engram_enabled",
    "engram_gate",
]
