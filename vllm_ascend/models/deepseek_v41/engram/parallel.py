# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend node-local Engram shared-memory policy."""

from vllm.distributed.parallel_state import get_engram_dp_size


def resolve_dp_shared_memory(requested: bool) -> bool:
    """A node with one DP replica uses ordinary TP shards, without sharing."""
    return requested and get_engram_dp_size() > 1
