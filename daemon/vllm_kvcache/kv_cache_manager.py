from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from vllm.logger import logger

from vllm_ascend.device_allocator.camem_bifrost import CaMemAllocator
from .elastic_config import ElasticPolicyConfig


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        policy: Optional[ElasticPolicyConfig] = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.cell_size = cell_size
        self.num_layers = num_layers

        self.block_mem_size = block_size * cell_size
        logger.info(f"Initializing KVCacheManager with num_blocks={num_blocks}, block_mem_size=" +
                    f"{self.block_mem_size/1024/1024:.4f} MB, num_layers={num_layers}")

        allocator = CaMemAllocator.get_instance()
        self.page_size = allocator.get_kvcache_page_bytes()
        self.blocks_per_page = max(1, self.page_size // self.block_mem_size)
        if self.page_size % self.block_mem_size != 0:
            logger.warning(
                "KV page size (%d) is not a multiple of block bytes (%d); "
                "tail bytes in each page are unused.",
                self.page_size,
                self.block_mem_size,
            )

        self.mem_size = self.num_blocks * self.block_mem_size

        existing = allocator.get_elastic_page_allocator()
        if existing is None:
            self.page_allocator = allocator.build_elastic_page_allocator(
                policy=policy or ElasticPolicyConfig.from_env(),
                autostart=True,
            )
        else:
            self.page_allocator = existing
            self.page_allocator.start()

        self._lock = threading.RLock()

        self._active_pages: set[int] = set()
        self._page_free_blocks: dict[int, set[int]] = {}
        self._free_block_queue: deque[int] = deque()
        self._allocated_blocks: set[int] = set()
        allocator.register_kvcache_manager(self)

    def reset(self) -> None:
        with self._lock:
            self._active_pages.clear()
            self._page_free_blocks.clear()
            self._free_block_queue.clear()
            self._allocated_blocks.clear()

    def alloc(self, need_size: int) -> Optional[list[int]]:
        if need_size <= 0:
            return []
        with self._lock:
            if not self._ensure_free_blocks_locked(need_size):
                logger.warning(
                    "Failed to allocate %d blocks: not enough free blocks available",
                    need_size,
                )
                return None

            block_ids: list[int] = []
            while len(block_ids) < need_size and self._free_block_queue:
                block_id = self._free_block_queue.popleft()
                page_id = self._block_to_page_id(block_id)
                page_free = self._page_free_blocks.get(page_id)
                if page_free is None or block_id not in page_free:
                    continue
                page_free.remove(block_id)
                self._allocated_blocks.add(block_id)
                block_ids.append(block_id)

            if len(block_ids) != need_size:
                logger.error(
                    "KVCacheManager internal inconsistency during alloc: need=%d got=%d",
                    need_size,
                    len(block_ids),
                )
                self._rollback_alloc_locked(block_ids)
                return None

            return block_ids

    def free(self, indices: list[int]) -> None:
        if not indices:
            return
        with self._lock:
            pages_to_try_release: set[int] = set()
            for block_id in indices:
                if block_id < 0 or block_id >= self.num_blocks:
                    logger.warning("Ignoring invalid block_id=%d in free", block_id)
                    continue
                if block_id not in self._allocated_blocks:
                    continue

                self._allocated_blocks.remove(block_id)
                page_id = self._block_to_page_id(block_id)
                page_free = self._page_free_blocks.get(page_id)
                if page_free is None:
                    continue
                page_free.add(block_id)
                self._free_block_queue.append(block_id)
                pages_to_try_release.add(page_id)

            for page_id in pages_to_try_release:
                self._try_release_page_locked(page_id)

    def resize(self, new_mem_size: int) -> bool:
        if new_mem_size <= 0:
            raise ValueError("new_mem_size must be positive")
        with self._lock:
            target_blocks = max(1, min(self.num_blocks,
                                       new_mem_size // self.block_mem_size))
            target_pages = self._blocks_to_pages(target_blocks)
            return self.page_allocator.resize(target_pages)

    def get_kv_size(self) -> int:
        return self.page_allocator.snapshot()["mapped_pages"] * self.page_size

    def available_size(self) -> int:
        with self._lock:
            local_free = sum(len(v) for v in self._page_free_blocks.values())
            snapshot = self.page_allocator.snapshot()
            latent_free_pages = snapshot["free_pages"] + snapshot["unmapped_pages"]
            latent_free = latent_free_pages * self.blocks_per_page
            alloced = len(self._allocated_blocks)
            approx = local_free + latent_free
            capacity_left = max(0, self.num_blocks - alloced)
            return max(0, min(capacity_left, approx))

    def get_mapped_memory_size(self, unit: str = "bytes") -> float:
        memory_bytes = self.get_kv_size() * self.num_layers * 2
        if unit == "bytes":
            return float(memory_bytes)
        if unit == "kb":
            return float(memory_bytes / 1024)
        if unit == "mb":
            return float(memory_bytes / (1024**2))
        if unit == "gb":
            return float(memory_bytes / (1024**3))
        raise ValueError(f"Unknown unit: {unit}")

    def trim(self) -> None:
        return

    def clear(self) -> None:
        raise NotImplementedError("clear() is not supported")

    def _ensure_free_blocks_locked(self, need_size: int) -> bool:
        if self._count_effective_free_blocks_locked() >= need_size:
            return True

        still_need = need_size - self._count_effective_free_blocks_locked()
        pages_needed = self._blocks_to_pages(still_need)
        page_ids = self.page_allocator.alloc(pages_needed)
        if page_ids is None:
            return False

        for page_id in page_ids:
            self._activate_page_locked(page_id)

        return self._count_effective_free_blocks_locked() >= need_size

    def _activate_page_locked(self, page_id: int) -> None:
        if page_id in self._active_pages:
            return

        block_ids = self._all_block_ids_of_page(page_id)
        if not block_ids:
            self.page_allocator.free([page_id])
            return

        self._active_pages.add(page_id)
        free_set = set(block_ids)
        self._page_free_blocks[page_id] = free_set
        for block_id in block_ids:
            if block_id in free_set:
                self._free_block_queue.append(block_id)

    def _try_release_page_locked(self, page_id: int) -> None:
        if page_id not in self._active_pages:
            return
        page_free = self._page_free_blocks.get(page_id)
        if page_free is None:
            return

        all_blocks = self._all_block_ids_of_page(page_id)
        fully_free = all(block_id in page_free for block_id in all_blocks)
        if not fully_free:
            return

        self.page_allocator.free([page_id])
        self._active_pages.remove(page_id)
        del self._page_free_blocks[page_id]

    def _count_effective_free_blocks_locked(self) -> int:
        return sum(len(v) for v in self._page_free_blocks.values())

    def _rollback_alloc_locked(self, block_ids: list[int]) -> None:
        if not block_ids:
            return

        for block_id in block_ids:
            page_id = self._block_to_page_id(block_id)
            page_free = self._page_free_blocks.get(page_id)
            if page_free is None:
                continue

            self._allocated_blocks.discard(block_id)
            if block_id not in page_free:
                page_free.add(block_id)
            self._free_block_queue.appendleft(block_id)

    def _blocks_to_pages(self, blocks: int) -> int:
        return (blocks + self.blocks_per_page - 1) // self.blocks_per_page

    def _block_to_page_id(self, block_id: int) -> int:
        return block_id // self.blocks_per_page

    def _all_block_ids_of_page(self, page_id: int) -> list[int]:
        start = page_id * self.blocks_per_page
        end = min(start + self.blocks_per_page, self.num_blocks)
        if start >= end:
            return []
        return list(range(start, end))
