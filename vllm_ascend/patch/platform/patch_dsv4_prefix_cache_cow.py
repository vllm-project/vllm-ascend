#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""DSv4 compressed prefix-cache copy-on-write patches.

Keep the upstream vLLM package independent from vLLM-Ascend by installing the
compressed KV manager and scheduler copy-pair plumbing from this plugin.
"""

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.kv_cache_block_copy import attach_kv_cache_block_copy_pairs
from vllm_ascend.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec as ascend_get_manager_for_kv_cache_spec,
)


def _take_block_copy_pairs(self: KVCacheManager) -> list[tuple[int, int, int, int, int]]:
    pairs: list[tuple[int, int, int, int, int]] = []
    for mgr in self.coordinator.single_type_managers:
        take_pairs = getattr(mgr, "take_block_copy_pairs", None)
        if take_pairs is not None:
            pairs.extend(take_pairs())
    return pairs


def _patch_kv_manager_factory() -> None:
    import vllm.v1.core.kv_cache_coordinator as kv_cache_coordinator
    import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager

    single_type_kv_cache_manager.get_manager_for_kv_cache_spec = ascend_get_manager_for_kv_cache_spec
    # kv_cache_coordinator imports the factory into its module namespace, so
    # patch that reference too.
    kv_cache_coordinator.get_manager_for_kv_cache_spec = ascend_get_manager_for_kv_cache_spec


def _patch_kv_cache_manager() -> None:
    if not hasattr(KVCacheManager, "take_block_copy_pairs"):
        KVCacheManager.take_block_copy_pairs = _take_block_copy_pairs


def _patch_scheduler_output() -> None:
    if getattr(Scheduler.schedule, "_vllm_ascend_dsv4_cow_patched", False):
        return

    original_schedule = Scheduler.schedule

    def _wrapped_schedule(self):
        output = original_schedule(self)
        if output is not None:
            take_pairs = getattr(self.kv_cache_manager, "take_block_copy_pairs", None)
            if take_pairs is not None:
                pairs = take_pairs() or None
                if pairs is not None:
                    attach_kv_cache_block_copy_pairs(output, pairs)
        return output

    _wrapped_schedule._vllm_ascend_dsv4_cow_patched = True
    Scheduler.schedule = _wrapped_schedule


_patch_kv_manager_factory()
_patch_kv_cache_manager()
_patch_scheduler_output()
