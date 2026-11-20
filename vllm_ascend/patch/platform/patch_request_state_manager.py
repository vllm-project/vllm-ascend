# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission bridge until vLLM exposes a per-pool reservation hook.

Pool construction lives in the Ascend coordinator factory. This adapter only
wraps allocate_slots; it does not replace framework constructors or free().
"""

from functools import wraps

from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_ascend.core.kv_cache_interface import get_request_state_plan
from vllm_ascend.core.request_state_manager import allocate_request_slots

_original_allocate_slots = KVCacheManager.allocate_slots


@wraps(_original_allocate_slots)
def allocate_slots(self, request, *args, **kwargs):
    if get_request_state_plan(self.kv_cache_config) is None:
        return _original_allocate_slots(self, request, *args, **kwargs)
    return allocate_request_slots(
        self.coordinator.single_type_managers,
        request.request_id,
        lambda: _original_allocate_slots(self, request, *args, **kwargs),
    )


KVCacheManager.allocate_slots = allocate_slots
