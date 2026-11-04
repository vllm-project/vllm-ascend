from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

from .elastic_config import ElasticPolicyConfig


class ElasticPageAllocator:

    def __init__(
        self,
        total_pages: int,
        mapped_pages: int,
        resize_callback: Callable[[int], int],
        policy: Optional[ElasticPolicyConfig] = None,
    ) -> None:
        if total_pages < 0:
            raise ValueError("total_pages must be >= 0")
        if mapped_pages < 0:
            raise ValueError("mapped_pages must be >= 0")
        if mapped_pages > total_pages:
            raise ValueError("mapped_pages must be <= total_pages")

        self.total_pages = total_pages
        self._mapped_pages = mapped_pages
        self._resize_callback = resize_callback
        self.policy = policy or ElasticPolicyConfig.from_env()

        self._free_page_list = deque(range(mapped_pages))
        self._in_use_pages: set[int] = set()
        self._unmapped_page_ids = deque(range(mapped_pages, total_pages))

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    def alloc(self, num_pages: int = 1) -> Optional[list[int]]:
        if num_pages <= 0:
            return []
        with self._lock:
            if len(self._free_page_list) < num_pages and self.policy.auto_grow:
                need = num_pages - len(self._free_page_list)
                grow_pages = max(need, self.policy.grow_step_pages)
                self._grow_locked(grow_pages)

            if len(self._free_page_list) < num_pages:
                return None

            page_ids = [self._free_page_list.popleft() for _ in range(num_pages)]
            self._in_use_pages.update(page_ids)
            return page_ids

    def free(self, page_ids: list[int]) -> None:
        if not page_ids:
            return
        with self._lock:
            for page_id in page_ids:
                if page_id in self._in_use_pages:
                    self._in_use_pages.remove(page_id)
                    self._free_page_list.append(page_id)

    def resize(self, target_pages: int) -> bool:
        with self._lock:
            target_pages = max(0, min(target_pages, self.total_pages))
            if target_pages == self._mapped_pages:
                return True
            if target_pages > self._mapped_pages:
                return self._grow_locked(target_pages - self._mapped_pages)
            return self._shrink_locked(self._mapped_pages - target_pages)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_pages": self.total_pages,
                "mapped_pages": self._mapped_pages,
                "free_pages": len(self._free_page_list),
                "in_use_pages": len(self._in_use_pages),
                "unmapped_pages": len(self._unmapped_page_ids),
            }

    def reset(self, mapped_pages: int) -> None:
        with self._lock:
            mapped_pages = max(0, min(mapped_pages, self.total_pages))
            self._mapped_pages = mapped_pages
            self._free_page_list = deque(range(mapped_pages))
            self._in_use_pages.clear()
            self._unmapped_page_ids = deque(range(mapped_pages, self.total_pages))

    def _control_loop(self) -> None:
        interval_s = self.policy.control_interval_ms / 1000.0
        while not self._stop_event.is_set():
            with self._lock:
                free_pages = len(self._free_page_list)
                if self.policy.auto_grow and free_pages < self.policy.min_free_pages:
                    self._grow_locked(self.policy.grow_step_pages)

                free_pages = len(self._free_page_list)
                if self.policy.auto_shrink and free_pages > self.policy.max_free_pages:
                    self._shrink_locked(self.policy.shrink_step_pages)

            self._stop_event.wait(interval_s)

    def _grow_locked(self, pages: int) -> bool:
        if pages <= 0:
            return True
        grow = min(pages, len(self._unmapped_page_ids))
        if grow <= 0:
            return False

        target = self._mapped_pages + grow
        applied = self._resize_callback(target)
        if applied < target:
            return False

        for _ in range(grow):
            self._free_page_list.append(self._unmapped_page_ids.popleft())
        self._mapped_pages += grow
        return True

    def _shrink_locked(self, pages: int) -> bool:
        if pages <= 0:
            return True
        shrink_ids = self._collect_shrinkable_tail_pages_locked(pages)
        shrink = len(shrink_ids)
        if shrink <= 0:
            return False

        target = self._mapped_pages - shrink
        if target < len(self._in_use_pages):
            return False

        applied = self._resize_callback(target)
        if applied > target:
            return False

        shrink_id_set = set(shrink_ids)
        self._free_page_list = deque(
            page_id for page_id in self._free_page_list
            if page_id not in shrink_id_set)

        for page_id in reversed(shrink_ids):
            self._unmapped_page_ids.appendleft(page_id)
        self._mapped_pages -= shrink
        return True

    def _collect_shrinkable_tail_pages_locked(self, pages: int) -> list[int]:
        if pages <= 0 or self._mapped_pages <= 0:
            return []

        free_set = set(self._free_page_list)
        max_candidate = min(pages, self._mapped_pages)
        shrink_ids: list[int] = []

        for page_id in range(self._mapped_pages - 1,
                             self._mapped_pages - max_candidate - 1, -1):
            if page_id in self._in_use_pages:
                break
            if page_id not in free_set:
                break
            shrink_ids.append(page_id)

        shrink_ids.reverse()
        return shrink_ids
