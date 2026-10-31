# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-owned pool bindings for MemFabric Offload."""

from vllm_ascend.distributed.memfabric.pool import OffloadPoolBinding, plan_offload_pools

__all__ = ["OffloadPoolBinding", "plan_offload_pools"]
