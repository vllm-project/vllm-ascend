"""Memcache backend owned by one multiprocess Worker service."""

from typing import Any

from .....backend.memcache_backend import MemcacheBackend


class MPMemcacheBackend(MemcacheBackend):
    """Add explicit buffer and store cleanup to the original backend."""

    def __init__(self, parallel_config: Any, device_index: int, lazy_init: bool = False):
        self._mp_registered_buffers: list[tuple[int, int]] = []
        super().__init__(parallel_config, local_rank=device_index, lazy_init=lazy_init)

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        if self._mp_registered_buffers or self._pending_buffers is not None:
            raise RuntimeError("Memcache buffers are already registered for this Worker")
        self._pending_buffers = (list(ptrs), list(sizes))
        self._register_buffers_if_needed()

    def _register_buffers_if_needed(self) -> None:
        if self._pending_buffers is None or not self._store_initialized:
            return

        assert self.store is not None
        ptrs, sizes = self._pending_buffers
        registered: list[tuple[int, int]] = []
        try:
            for ptr, size in zip(ptrs, sizes):
                result = self.store.register_buffer(ptr, size)
                if result not in (None, 0):
                    raise RuntimeError(f"Memcache buffer registration failed with code {result}")
                registered.append((ptr, size))
        except BaseException:
            for ptr, size in reversed(registered):
                self.store.unregister_buffer(ptr, size)
            self._pending_buffers = None
            raise

        self._mp_registered_buffers = registered
        self._pending_buffers = None

    def unregister_buffer(self) -> None:
        self._pending_buffers = None
        if not self._mp_registered_buffers:
            return

        assert self.store is not None
        failed: list[tuple[int, int, object]] = []
        released: set[tuple[int, int]] = set()
        for ptr, size in reversed(self._mp_registered_buffers):
            result = self.store.unregister_buffer(ptr, size)
            if result not in (None, 0):
                failed.append((ptr, size, result))
            else:
                released.add((ptr, size))
        self._mp_registered_buffers = [buffer for buffer in self._mp_registered_buffers if buffer not in released]
        if failed:
            raise RuntimeError(f"Memcache buffer unregistration failed: {failed!r}")

    def close(self) -> None:
        try:
            self.unregister_buffer()
        finally:
            close = getattr(self.store, "close", None)
            if callable(close):
                close()
