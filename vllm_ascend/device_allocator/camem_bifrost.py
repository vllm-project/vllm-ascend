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
# CANN-mem-based pytorch pluggable allocator to implement sleep mode.
#

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from acl.rt import memcpy  # type: ignore # noqa: F401
from vllm.logger import logger

from vllm_ascend.platform import NPUPlatform

from vllm_ascend.core.kvcache.elastic_config import ElasticPolicyConfig
from vllm_ascend.core.kvcache.page_allocator import ElasticPageAllocator


GiB = 1024**3
MiB = 1024**2


def find_loaded_library(lib_name: str) -> Optional[str]:
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        return None
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}")
    return path


camem_available = False
try:
    from vllm_ascend.bifrost_ascend_C import (  # type: ignore # noqa: F401
        init_module,
        python_finalize_space,
        python_init_space,
        python_map_kvcache_until,
        python_map_weight,
        python_unmap_kvcache_until,
        python_unmap_weight,
    )

    lib_name = find_loaded_library("bifrost_ascend_C")
    camem_available = True
except ImportError as e:
    logger.warning(
        "Failed to import bifrost_ascend_C:%s. Sleep mode will be disabled. ", e)
    init_module = None
    python_init_space = None
    python_finalize_space = None
    python_map_weight = None
    python_unmap_weight = None
    python_map_kvcache_until = None
    python_unmap_kvcache_until = None
    lib_name = None


HandleType = Tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    kind: str
    mapped_bytes: int
    cpu_backup_tensor: Optional[torch.Tensor] = None


def _to_env_mb(name: str, default_mb: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default_mb
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        logger.warning("Invalid env %s=%s, fallback to %d", name, raw,
                       default_mb)
        return default_mb


def _build_allocator(lib_path: str, malloc_symbol: str,
                     free_symbol: str) -> torch.npu.memory.NPUPluggableAllocator:
    return torch.npu.memory.NPUPluggableAllocator(lib_path, malloc_symbol,
                                                  free_symbol)


def _parse_visible_devices() -> Optional[list[int]]:
    raw = (os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
           or os.environ.get("ASCEND_VISIBLE_DEVICES")
           or os.environ.get("CUDA_VISIBLE_DEVICES"))
    if raw is None or raw.strip() == "":
        return None
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            return None
    return out or None


def _resolve_local_device_id(device_id: Optional[int]) -> int:
    if device_id is not None:
        return int(device_id)
    try:
        return int(torch.npu.current_device())
    except Exception:
        return 0


def _resolve_canonical_device_id(local_device_id: int) -> int:
    visible = _parse_visible_devices()
    if not visible:
        return local_device_id
    if local_device_id < 0 or local_device_id >= len(visible):
        return local_device_id
    return int(visible[local_device_id])


class CaMemAllocator:
    instance = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance

    def __init__(self, device_id: Optional[int] = None) -> None:
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates.")

        if not camem_available or init_module is None or python_init_space is None:
            raise RuntimeError("bifrost_ascend_C is unavailable; cannot initialize CaMemAllocator")

        self.pointer_to_data: Dict[int, AllocationData] = {}
        self.current_tag: str = CaMemAllocator.default_tag
        self._active_kind: str = "weight"
        self._device_id: int = _resolve_local_device_id(device_id)
        self._canonical_device_id: int = _resolve_canonical_device_id(
            self._device_id)
        self._kvcache_init_map_mb: int = _to_env_mb("MDAEMON_KV_INIT_MB", 128)
        self._kvcache_init_map_bytes: int = self._kvcache_init_map_mb * MiB
        self._kvcache_page_mb: int = _to_env_mb("MDAEMON_KV_GRAN_MB", 4)
        self._kvcache_page_bytes: int = self._kvcache_page_mb * MiB
        self._elastic_allocator: Optional[ElasticPageAllocator] = None
        self._kvcache_managers: list[Any] = []

        ok = python_init_space(self._device_id, self._canonical_device_id)
        if not ok:
            raise RuntimeError(
                f"python_init_space failed (local_device_id={self._device_id}, "
                f"canonical_device_id={self._canonical_device_id})")
        print(f"[SD-DEBUG] Initialized CaMemAllocator with local_device_id={self._device_id}, "
              f"canonical_device_id={self._canonical_device_id}, "
              f"kvcache_init_map_mb={self._kvcache_init_map_mb} MiB")
        init_module(self.python_malloc_callback, self.python_free_callback)

        if lib_name is None:
            raise RuntimeError("Failed to locate bifrost_ascend_C shared library")

        self._weight_allocator = _build_allocator(lib_name, "my_malloc_weight",
                                                  "my_free_weight")
        self._kvcache_allocator = _build_allocator(lib_name,
                                                   "my_malloc_kvcache",
                                                   "my_free_kvcache")
        self._weight_pool = torch.npu.memory.MemPool(self._weight_allocator._allocator)
        self._kvcache_pool = torch.npu.memory.MemPool(
            self._kvcache_allocator._allocator)

        self.allocator_and_pools: Dict[str, Any] = {
            "weight": (self._weight_pool, self._weight_allocator),
            "kvcache": (self._kvcache_pool, self._kvcache_allocator),
        }

    def _map_on_malloc(self, kind: str, device: int, d_mem: int,
                       aligned_size: int) -> int:
        if kind == "weight":
            python_map_weight(device, d_mem, aligned_size)
            return aligned_size

        init_map = min(aligned_size, self._kvcache_init_map_bytes)
        python_map_kvcache_until(device, d_mem, init_map)
        return init_map

    def _remap_on_wakeup(self, data: AllocationData) -> int:
        device, aligned_size, d_mem, _ = data.handle
        if data.kind == "weight":
            python_map_weight(device, d_mem, aligned_size)
            return aligned_size

        init_map = min(aligned_size, self._kvcache_init_map_bytes)
        python_map_kvcache_until(device, d_mem, init_map)
        return init_map

    def _unmap_for_entry(self, data: AllocationData) -> None:
        device, _, d_mem, _ = data.handle
        if data.kind == "weight":
            python_unmap_weight(device, d_mem)
        else:
            python_unmap_kvcache_until(device, d_mem, 0)

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        device, aligned_size, d_mem, _ = allocation_handle
        mapped_bytes = self._map_on_malloc(self._active_kind, device, d_mem,
                                           aligned_size)
        self.pointer_to_data[d_mem] = AllocationData(
            handle=allocation_handle,
            tag=self.current_tag,
            kind=self._active_kind,
            mapped_bytes=mapped_bytes,
        )

    def python_free_callback(self, ptr: int) -> HandleType:
        data = self.pointer_to_data.pop(ptr)
        # C++ my_free_* already performs unmap/return and address release.
        # Keep python callback as bookkeeping only to avoid duplicate unmap.
        data.cpu_backup_tensor = None
        return data.handle

    def register_kvcache_manager(self, manager: Any) -> None:
        self._kvcache_managers.append(manager)

    def sleep(self,
              offload_tags: Optional[Union[Tuple[str, ...], str]] = None) -> None:
        if offload_tags is None:
            offload_tags = (CaMemAllocator.default_tag, )
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags, )

        assert isinstance(offload_tags, tuple)

        if self._elastic_allocator is not None:
            self._elastic_allocator.stop()
            self._elastic_allocator.reset(0)

        for manager in self._kvcache_managers:
            manager.reset()

        for ptr, data in self.pointer_to_data.items():
            should_offload = data.kind != "kvcache" and data.tag in offload_tags
            mapped_size = int(data.mapped_bytes)
            if should_offload and mapped_size > 0:
                cpu_backup_tensor = torch.empty(
                    mapped_size,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=NPUPlatform.is_pin_memory_available(),
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_DEVICE_TO_HOST = 2
                dest_max = cpu_ptr + mapped_size * 2
                memcpy(cpu_ptr, dest_max, ptr, mapped_size,
                       ACL_MEMCPY_DEVICE_TO_HOST)
                data.cpu_backup_tensor = cpu_backup_tensor

            self._unmap_for_entry(data)
            data.mapped_bytes = 0

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        for ptr, data in self.pointer_to_data.items():
            if tags is not None and data.tag not in tags:
                continue

            if data.cpu_backup_tensor is not None:
                data.mapped_bytes = self._remap_on_wakeup(data)
                cpu_backup_tensor = data.cpu_backup_tensor
                size_in_bytes = min(
                    data.mapped_bytes,
                    cpu_backup_tensor.numel() * cpu_backup_tensor.element_size(),
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_HOST_TO_DEVICE = 1
                dest_max = ptr + size_in_bytes * 2
                memcpy(ptr, dest_max, cpu_ptr, size_in_bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE)
                data.cpu_backup_tensor = None

        if self._elastic_allocator is not None:
            self._elastic_allocator.reset(self.get_kvcache_mapped_pages())
            self._elastic_allocator.start()

    @contextmanager
    def use_weight_memory_pool(self, tag: Optional[str] = None) -> Iterator[None]:
        if tag is None:
            tag = CaMemAllocator.default_tag

        old_tag = self.current_tag
        old_kind = self._active_kind
        self.current_tag = tag
        self._active_kind = "weight"
        with torch.npu.memory.use_mem_pool(self._weight_pool):
            yield
        self.current_tag = old_tag
        self._active_kind = old_kind

    @contextmanager
    def use_kvcache_memory_pool(self, tag: Optional[str] = None) -> Iterator[None]:
        if tag is None:
            tag = CaMemAllocator.default_tag

        old_tag = self.current_tag
        old_kind = self._active_kind
        self.current_tag = tag
        self._active_kind = "kvcache"
        with torch.npu.memory.use_mem_pool(self._kvcache_pool):
            yield
        self.current_tag = old_tag
        self._active_kind = old_kind

    @contextmanager
    def use_memory_pool(self, tag: Optional[str] = None) -> Iterator[None]:
        # Backward-compatible route: default `use_memory_pool` means weight pool.
        with self.use_weight_memory_pool(tag=tag):
            yield

    def close(self) -> None:
        if self._elastic_allocator is not None:
            self._elastic_allocator.stop()
            self._elastic_allocator = None
        if python_finalize_space is not None:
            python_finalize_space()

    def get_current_usage(self) -> int:
        total = 0
        for data in self.pointer_to_data.values():
            total += data.mapped_bytes
        return total

    def _iter_kvcache_data(self) -> list[AllocationData]:
        return [data for data in self.pointer_to_data.values() if data.kind == "kvcache"]

    def get_kvcache_page_bytes(self) -> int:
        return self._kvcache_page_bytes

    def get_kvcache_capacity_pages(self) -> int:
        entries = self._iter_kvcache_data()
        if not entries:
            return 0
        return min(int(data.handle[1] // self._kvcache_page_bytes) for data in entries)

    def get_kvcache_mapped_pages(self) -> int:
        entries = self._iter_kvcache_data()
        if not entries:
            return 0
        return min(int(data.mapped_bytes // self._kvcache_page_bytes) for data in entries)

    def resize_kvcache_mapped_pages(self, target_pages: int) -> int:
        entries = self._iter_kvcache_data()
        if not entries:
            return 0

        capacity_pages = self.get_kvcache_capacity_pages()
        clamped_pages = max(0, min(target_pages, capacity_pages))
        target_bytes = clamped_pages * self._kvcache_page_bytes
        original_mapped_bytes = {
            id(data): int(data.mapped_bytes) for data in entries
        }
        resized_entries: list[AllocationData] = []

        for data in entries:
            device, aligned_size, d_mem, _ = data.handle
            bounded_target = min(target_bytes, aligned_size)
            if bounded_target > data.mapped_bytes:
                try:
                    python_map_kvcache_until(device, d_mem, bounded_target)
                except RuntimeError as exc:
                    rollback_target = original_mapped_bytes[id(data)]
                    for resized in resized_entries:
                        rollback_device, _, rollback_d_mem, _ = resized.handle
                        python_unmap_kvcache_until(
                            rollback_device,
                            rollback_d_mem,
                            original_mapped_bytes[id(resized)],
                        )
                        resized.mapped_bytes = original_mapped_bytes[id(resized)]
                    logger.warning(
                        "resize_kvcache_mapped_pages rollback after map failure: "
                        "target_pages=%d, target_bytes=%d, rollback_target=%d, error=%s",
                        clamped_pages,
                        target_bytes,
                        rollback_target,
                        exc,
                    )
                    return self.get_kvcache_mapped_pages()
            elif bounded_target < data.mapped_bytes:
                python_unmap_kvcache_until(device, d_mem, bounded_target)
            data.mapped_bytes = bounded_target
            resized_entries.append(data)

        return self.get_kvcache_mapped_pages()

    def build_elastic_page_allocator(
        self,
        policy: Optional[ElasticPolicyConfig] = None,
        autostart: bool = True,
    ) -> ElasticPageAllocator:
        total_pages = self.get_kvcache_capacity_pages()
        mapped_pages = self.get_kvcache_mapped_pages()

        allocator = ElasticPageAllocator(
            total_pages=total_pages,
            mapped_pages=mapped_pages,
            resize_callback=self.resize_kvcache_mapped_pages,
            policy=policy or ElasticPolicyConfig.from_env(),
        )
        if autostart:
            allocator.start()
        self._elastic_allocator = allocator
        return allocator

    def get_elastic_page_allocator(self) -> Optional[ElasticPageAllocator]:
        return self._elastic_allocator
