# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Engram for DeepSeek V4.1.

Split the way upstream vLLM splits it for other accelerators: ``common`` holds
the platform-independent n-gram hashing and gating, ``npu`` holds the storage
and lookup the NPU runs.
"""

from .common import engram_enabled, engram_gate
from .hash_state import AscendEngramSlotCache, create_engram_hash_state, engram_dead_mask
from .npu import engram_cpu_offload

__all__ = [
    "AscendEngramSlotCache",
    "create_engram_hash_state",
    "engram_cpu_offload",
    "engram_dead_mask",
    "engram_enabled",
    "engram_gate",
]
