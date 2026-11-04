from __future__ import annotations

from vllm_ascend.core.kvcache.page_allocator import ElasticPageAllocator


class KVCacheResizeDispatcher:

    def __init__(self, allocator: ElasticPageAllocator, page_bytes: int) -> None:
        if page_bytes <= 0:
            raise ValueError("page_bytes must be > 0")
        self.allocator = allocator
        self.page_bytes = page_bytes

    def resize_pages(self, target_pages: int) -> bool:
        return self.allocator.resize(target_pages)

    def resize_bytes(self, target_bytes: int) -> bool:
        target_pages = max(0, target_bytes // self.page_bytes)
        return self.allocator.resize(target_pages)

    def snapshot(self) -> dict[str, int]:
        state = self.allocator.snapshot()
        state["page_bytes"] = self.page_bytes
        state["mapped_bytes"] = state["mapped_pages"] * self.page_bytes
        return state
