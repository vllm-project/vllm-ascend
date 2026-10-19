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
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from acl.rt import memcpy  # type: ignore # noqa: F401
from vllm.device_allocator import AllocationData, HandleType
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.logger import logger


def find_loaded_library(lib_name) -> str | None:
    """Find a loaded shared library from this process's memory maps."""
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
    assert filename.rpartition(".so")[0].startswith(lib_name), f"Unexpected filename: {filename} for library {lib_name}"
    return path


camem_available = False
try:
    from vllm_ascend.vllm_ascend_C import (  # type: ignore # noqa: F401
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )

    lib_name = find_loaded_library("vllm_ascend_C")
    camem_available = True
except ImportError as e:
    logger.warning("Failed to import vllm_ascend_C:%s. Sleep mode will be disabled. ", e)
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> torch.npu.memory.NPUPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    return torch.npu.memory.NPUPluggableAllocator(lib_name, "my_malloc", "my_free")


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
) -> Iterator[tuple[torch.npu.memory.MemPool, torch.npu.memory.NPUPluggableAllocator]]:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CaMemAllocator(CuMemAllocator):
    """CANN mechanisms for the shared vLLM memory-pool lifecycle."""

    instance: "CaMemAllocator | None" = None
    # Allocations with this tag stay mapped across sleep/wake cycles.
    sleep_persistent_tag: str = "sleep_persistent"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        """Return the singleton required by the C-extension callbacks."""
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )
        super().__init__()

    def _create_and_map(self, allocation_handle: HandleType) -> None:
        create_and_map(allocation_handle)

    def _unmap_and_release(self, allocation_handle: HandleType) -> None:
        unmap_and_release(allocation_handle)

    def _synchronize(self, device: int | None = None) -> None:
        # Pending NPU work must finish before CANN mappings are released.
        torch.npu.synchronize()

    def _empty_cache(self) -> None:
        torch.npu.empty_cache()

    def _copy_to_cpu(self, ptr: int, size_in_bytes: int) -> torch.Tensor:
        tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        cpu_ptr = tensor.data_ptr()
        memcpy(cpu_ptr, cpu_ptr + size_in_bytes * 2, ptr, size_in_bytes, 2)
        return tensor

    def _copy_to_device(self, ptr: int, tensor: torch.Tensor) -> None:
        size_in_bytes = tensor.numel() * tensor.element_size()
        memcpy(ptr, ptr + size_in_bytes * 2, tensor.data_ptr(), size_in_bytes, 1)

    def _is_persistent(self, data: AllocationData) -> bool:
        return data.tag == self.sleep_persistent_tag

    def _prepare_pool_config(self) -> str | None:
        # PYTORCH_NPU_ALLOC_CONF is validated at construction time.
        return None

    def _use_memory_pool_with_allocator(self):
        return use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback)
