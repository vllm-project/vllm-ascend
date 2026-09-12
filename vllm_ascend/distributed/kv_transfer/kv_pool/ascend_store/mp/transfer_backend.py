"""Device selection and buffer ownership for the private transfer process."""

from typing import Any

import torch


def create_transfer_backend(
    name: str, device_index: int, global_rank: int, lazy_init: bool = False
) -> "TransferBackend":
    # Import only the selected SDK. Model-worker identity arrives as plain
    # values, so this process never joins the worker's distributed groups.
    if name == "mooncake":
        from .mooncake_backend import MPMooncakeBackend

        backend = MPMooncakeBackend(device_index, global_rank, lazy_init=lazy_init)
    elif name == "memcache":
        from ..backend.memcache_backend import MemcacheBackend

        backend = MemcacheBackend(None, device_id=device_index, lazy_init=lazy_init)  # type: ignore[assignment, arg-type]
    elif name == "yuanrong":
        from ..backend.yuanrong_backend import YuanrongBackend

        backend = YuanrongBackend(None, lazy_init=lazy_init)  # type: ignore[assignment, arg-type]
    else:
        raise ValueError(f"Unknown transfer backend: {name}")
    return TransferBackend(name, backend, device_index)


class TransferBackend:
    """Keep registrations alive until all transfers stop and unregistration succeeds."""

    def __init__(self, name: str, backend: Any, device_index: int):
        self.name = name
        self.backend = backend
        self.device_index = device_index
        self._registered: list[tuple[int, int]] = []
        self._registration_started = False
        self._engine = None

    def set_device(self) -> None:
        torch.npu.set_device(self.device_index)

    @property
    def requires_exists_before_put(self) -> bool:
        return getattr(self.backend, "requires_exists_before_put", True)

    @property
    def store(self):
        return self.backend.store

    def ensure_initialized(self) -> None:
        ensure = getattr(self.backend, "ensure_initialized", None)
        if ensure is not None:
            ensure()

    def validate_layerwise_support(self) -> None:
        self.backend.validate_layerwise_support()

    def exists(self, keys):
        return self.backend.exists(keys)

    def batch_is_exist(self, keys):
        return self.backend.batch_is_exist(keys)

    def batch_get_key_info(self, keys):
        return self.backend.batch_get_key_info(keys)

    def batch_alloc(self, keys, sizes, lease_ttl_ms=0):
        return self.backend.batch_alloc(keys, sizes, lease_ttl_ms)

    def batch_add_lease(self, keys, lease_ttl_ms=0):
        return self.backend.batch_add_lease(keys, lease_ttl_ms)

    def batch_remove_lease(self, keys):
        return self.backend.batch_remove_lease(keys)

    def batch_write_finish(self, keys, results):
        return self.backend.batch_write_finish(keys, results)

    def batch_put_start(self, keys, sizes):
        return self.backend.batch_put_start(keys, sizes)

    def batch_copy_put(self, keys, all_buffers, all_sizes, all_dst_offsets):
        return self.backend.batch_copy_put(keys, all_buffers, all_sizes, all_dst_offsets)

    def batch_commit(self, keys):
        return self.backend.batch_commit(keys)

    def batch_revoke(self, keys):
        return self.backend.batch_revoke(keys)

    def batch_get_start(self, keys):
        return self.backend.batch_get_start(keys)

    def batch_copy_get(self, keys, all_buffers, all_sizes, all_src_offsets):
        return self.backend.batch_copy_get(keys, all_buffers, all_sizes, all_src_offsets)

    def batch_get_end(self, keys):
        return self.backend.batch_get_end(keys)

    def get(self, keys, addresses, sizes):
        return self.backend.get(keys, addresses, sizes)

    def put(self, keys, addresses, sizes):
        return self.backend.put(keys, addresses, sizes)

    def register_buffer(self, pointers: list[int], lengths: list[int]) -> None:
        if self._registration_started:
            raise RuntimeError("Transfer buffers are already registered")
        if len(pointers) != len(lengths):
            raise ValueError("Buffer pointers and lengths must have equal sizes")
        self._registration_started = True
        self.set_device()
        if self.name == "memcache" and getattr(self.backend, "_lazy_init", False) is True:
            self.backend.register_buffer(pointers, lengths)
            self._registered.extend(zip(pointers, lengths))
            return
        if self.name == "yuanrong":
            self.backend.register_buffer(pointers, lengths)
            return
        if self.name == "mooncake":
            if self.backend._use_fabric_mem:
                return
            if not self.backend._use_store_independent_te:
                from vllm.utils.network_utils import get_ip

                from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

                self._engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        try:
            for pointer, length in zip(pointers, lengths):
                if self._engine is not None:
                    try:
                        result = self._engine.register_memory(pointer, length, f"npu:{self.device_index}")
                    except TypeError as exc:
                        raise RuntimeError(
                            "AscendStore multiprocess transfers with Mooncake's shared transfer engine require "
                            "mooncake-transfer-engine-npu>=0.3.12 "
                            "(register_memory(address, length, location))"
                        ) from exc
                else:
                    result = self.backend.store.register_buffer(pointer, length)
                if result not in (None, 0):
                    raise RuntimeError(f"Transfer buffer registration failed: {result}")
                self._registered.append((pointer, length))
        except BaseException:
            self.unregister_buffer()
            raise

    def unregister_buffer(self) -> None:
        errors = []
        if self.name == "memcache" and getattr(self.backend, "_store_initialized", True) is False:
            self._registered.clear()
            return
        for pointer, length in reversed(self._registered.copy()):
            try:
                if self._engine is not None:
                    result = self._engine.unregister_memory(pointer)
                elif self.name == "memcache":
                    result = self.backend.store.unregister_buffer(pointer, length)
                else:
                    result = self.backend.store.unregister_buffer(pointer)
                if result not in (None, 0):
                    raise RuntimeError(f"Buffer unregistration returned {result}")
                self._registered.remove((pointer, length))
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError(f"Transfer buffer unregistration failed: {errors}")

    def close(self) -> None:
        self.unregister_buffer()
        close = getattr(self.backend.store, "close", None)
        if callable(close):
            close()
