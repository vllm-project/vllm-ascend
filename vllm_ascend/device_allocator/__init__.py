#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def use_sleep_persistent_allocation(*, enabled: bool) -> Iterator[None]:
    """Preserve allocations across sleep/wake when sleep mode is enabled.

    This context overrides the allocation tag and must be entered while a
    CaMem memory pool is active.
    """
    if not enabled:
        yield
        return

    from vllm_ascend.device_allocator.camem import CaMemAllocator

    allocator = CaMemAllocator.get_instance()
    with allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag):
        yield


__all__ = ["use_sleep_persistent_allocation"]
