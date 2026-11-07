# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-owned instance bindings for MemFabric Offload."""

from vllm_ascend.distributed.memfabric.instance import OffloadInstanceBinding, plan_offload_instances

__all__ = ["OffloadInstanceBinding", "plan_offload_instances"]
