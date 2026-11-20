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

from collections.abc import Callable

from vllm.device_allocator.sleep_mode_backend import SleepModeBackend, SleepModeBackendFactory
from vllm.logger import logger

from vllm_ascend.device_allocator.camem import CaMemAllocator, camem_available


class CaMemBackend(SleepModeBackend):
    """Sleep-mode backend backed by the Ascend CaMem allocator.

    Level 1 offloads the weights pool; level 2 discards it. Ascend currently
    creates only `weights` and `kv_cache` pools, so there are no runtime or
    graph pools to offload at level 2.
    """

    def __init__(self, allocator: Callable[[], CaMemAllocator] | None = None) -> None:
        super().__init__()
        self._allocator = allocator or CaMemAllocator.get_instance

    @classmethod
    def is_supported(cls) -> bool:
        return camem_available

    def suspend(self, level: int = 1) -> None:
        if level not in (1, 2):
            raise ValueError(f"Unsupported sleep level: {level}")

        self._state = "SUSPENDED"
        offload_tags = ("weights",) if level == 1 else ()
        self._allocator().sleep(offload_tags)
        logger.info("CaMem backend suspended: level=%d, offload_tags=%s", level, offload_tags)

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RESUMING"
        self._allocator().wake_up(tags)
        self._state = "RUNNING"

    def discard(self, tags: tuple[str, ...]) -> None:
        self._allocator().discard(tags)
        self._state = "SUSPENDED"


# Register at import time, mirroring the upstream built-in backend.
SleepModeBackendFactory.register_backend(
    "camem",
    "vllm_ascend.device_allocator.sleep_mode_backend",
    "CaMemBackend",
)

# "cumem" is the platform sleep allocator. Point the upstream default name at
# the CaMem implementation instead of changing ModelConfig.
SleepModeBackendFactory._registry["cumem"] = SleepModeBackendFactory._registry["camem"]
logger.info_once("Ascend maps sleep-mode backend 'cumem' to CaMemBackend.")
