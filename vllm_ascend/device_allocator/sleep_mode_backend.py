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
# This file is a part of the vllm-ascend project.
#
"""Ascend implementation of the upstream sleep-mode backend interface."""

from __future__ import annotations

from collections.abc import Callable, Collection

from vllm.device_allocator.sleep_mode_backend import CuMemBackend, SleepModeBackendFactory

from vllm_ascend.device_allocator.camem import CaMemAllocator


class CaMemBackend(CuMemBackend):
    """Default Ascend backend.

    Wraps the platform sleep-mode allocator, `CaMemAllocator` on Ascend.
    Suspend offloads per-allocation between NPU and host.
    """

    def __init__(self, allocator: Callable[[], CaMemAllocator] | None = None) -> None:
        super().__init__()
        self._allocator = allocator or CaMemAllocator.get_instance

    def suspend(self, discardable: Collection[int] = ()) -> None:
        self._state = "SUSPENDED"
        self._allocator().sleep(("weights", "runtime", "cudagraph"), discardable)

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        self._allocator().wake_up(tags)
        self._state = "RUNNING"

    def discard(self, tags: tuple[str, ...]) -> None:
        self._allocator().discard(tags)
        self._state = "SUSPENDED"

    @classmethod
    def preserves_communicators(cls) -> bool:
        # HCCL groups survive unless sleep_mode_extra_cleanup destroys them.
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            rl_config = get_ascend_config().rl_config
        except RuntimeError:
            return False
        return not (getattr(rl_config, "enabled", False) and getattr(rl_config, "sleep_mode_extra_cleanup", False))


# Register at import time, mirroring the upstream built-in backend.
SleepModeBackendFactory.register_backend(
    "camem",
    "vllm_ascend.device_allocator.sleep_mode_backend",
    "CaMemBackend",
)
