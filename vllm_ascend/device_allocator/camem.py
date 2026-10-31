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
import atexit
import gc
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from acl.rt import memcpy  # type: ignore # noqa: F401
from vllm.device_allocator import AllocationData, HandleType  # noqa: F401
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.logger import logger
from vllm.utils.system_utils import find_loaded_library

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
    new_alloc = torch.npu.memory.NPUPluggableAllocator(lib_name, "my_malloc", "my_free")
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[HandleType], None],
    python_free_func: Callable[[int], HandleType],
):
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CaMemAllocator(CuMemAllocator):
    """
    A singleton class that manages a memory pool for CANN tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.
    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.
    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.
    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        """
        CaMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        assert camem_available, "camem allocator is not available"
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
            # Ensure MemPool/allocator wrappers are released before interpreter
            # finalization tears down PyTorch allocator internals.
            atexit.register(CaMemAllocator._shutdown_singleton)
        return CaMemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        instance = CaMemAllocator.instance
        if instance is None:
            return
        try:
            instance.release_pools()
        except Exception:
            logger.exception("CaMemAllocator singleton shutdown failed")

    def __init__(self):
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )
        super().__init__()
        self.current_tag: str = CaMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}

    def release_pools(self) -> None:
        """Drop Python references to MemPool/pluggable allocators eagerly."""
        if not self.allocator_and_pools:
            return

        pool_entries = list(self.allocator_and_pools.values())
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        # Phase 1: drop MemPool refs while allocators are still strongly held.
        mem_pools.clear()
        gc.collect()

        # Phase 2: now it is safe to release allocator wrappers.
        allocators.clear()

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Internal method to look up the allocation data
        when memory is freed in the memory pool.
        """
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        # Drain pending kernels before the C extension's aclrtUnmapMem.
        # The pluggable allocator path doesn't defer reclaim like the
        # regular caching allocator, so without this, in-flight work
        # (e.g. quant helpers' transient tensors during weight loading)
        # races the unmap and surfaces as ACL error.
        torch.npu.synchronize(data.handle[0])
        logger.debug(
            "Freed %s bytes for %s with address %s from camem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.
        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CaMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        offload_count = sum(1 for data in self.pointer_to_data.values() if data.tag in offload_tags)
        logger.info(
            "CaMem sleep: offloading %s/%s allocations (tags=%s)",
            offload_count,
            len(self.pointer_to_data),
            offload_tags,
        )
        # Workaround for verl v0.8.0: synchronize pending NPU work before
        # unmapping tensor memory. The exact root cause is unclear, but this
        # avoids rtMemcpy errors observed when releasing the mappings too early.
        torch.npu.synchronize()

        has_policy_conflict = False

        for ptr, data in self.pointer_to_data.items():
            if data.is_asleep:
                requests_offload = data.tag in offload_tags
                was_offloaded = data.cpu_backup_tensor is not None
                if requests_offload != was_offloaded:
                    has_policy_conflict = True
                continue
            handle = data.handle
            if data.tag in offload_tags:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_DEVICE_TO_HOST = 2
                dest_max = cpu_ptr + size_in_bytes * 2
                memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
                data.cpu_backup_tensor = cpu_backup_tensor
            try:
                unmap_and_release(handle)
            finally:
                data.is_asleep = True

        if has_policy_conflict:
            logger.warning(
                "CaMemAllocator: sleep cannot change the policy of "
                "already-asleep allocations; the existing policy was kept."
            )

        gc.collect()
        torch.npu.empty_cache()

    def discard(self, tags: tuple[str, ...] | str) -> None:
        """Discard mapped allocations with the given tags without CPU backup."""
        if isinstance(tags, str):
            tags = (tags,)

        discarded_bytes = 0
        has_policy_conflict = False
        for data in self.pointer_to_data.values():
            if data.tag not in tags:
                continue
            if data.is_asleep:
                if data.cpu_backup_tensor is not None:
                    has_policy_conflict = True
                continue
            torch.accelerator.synchronize(data.handle[0])
            unmap_and_release(data.handle)
            data.is_asleep = True
            discarded_bytes += data.handle[1]

        logger.info(
            "CaMemAllocator: discarded %.2f GiB for tags %s.",
            discarded_bytes / 1024**3,
            tags,
        )

        if has_policy_conflict:
            logger.warning(
                "CaMemAllocator: discard cannot change the policy of "
                "already-asleep allocations; the existing policy was kept."
            )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory."""
        gc.collect()
        torch.accelerator.empty_cache()

        restore_count = sum(1 for data in self.pointer_to_data.values() if tags is None or data.tag in tags)
        logger.info(
            "CaMem wake_up: restoring %s/%s allocations (tags=%s)",
            restore_count,
            len(self.pointer_to_data),
            tags or "all",
        )
        for ptr, data in self.pointer_to_data.items():
            if not data.is_asleep:
                continue
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        ACL_MEMCPY_HOST_TO_DEVICE = 1
                        dest_max = ptr + size_in_bytes * 2
                        memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.
        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CaMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback) as data:
                # start to hit another PyTorch bug in PyTorch 2.6,
                # possibly because of gc-related issue w.r.t. the allocator and
                # the memory pool.
                # to avoid the issue, we keep a reference of the data.
                # see https://github.com/pytorch/pytorch/issues/146431 .
                self.allocator_and_pools[tag] = data
                yield
        finally:
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released.
            # right now it is fine, because we only use this allocator
            # during weight loading and kv cache creation, where we only
            # allocate memory.
            # TODO: we need to find a way to release the memory,
            # i.e. calling torch.cuda.empty_cache()
            self.current_tag = old_tag
