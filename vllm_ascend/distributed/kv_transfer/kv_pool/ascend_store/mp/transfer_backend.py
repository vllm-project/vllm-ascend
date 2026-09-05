"""Device selection and buffer ownership for the private transfer process."""

from typing import Any

import torch


def create_transfer_backend(name: str, device_index: int) -> "TransferBackend":
    # Only the selected SDK is imported; no distributed groups are initialized
    # in this process. These constructors do not consume ParallelConfig unless
    # Mooncake SSD offload needs its process-global rank.
    if name == "mooncake":
        from ..backend.mooncake_backend import MooncakeBackend, MooncakeStoreConfig

        if MooncakeStoreConfig.load_from_env().enable_ssd_offload:
            raise NotImplementedError("Multiprocess transfer does not yet support Mooncake SSD offload")
        backend = MooncakeBackend(None)  # type: ignore[arg-type]
    elif name == "memcache":
        from ..backend.memcache_backend import MemcacheBackend

        backend = MemcacheBackend(None, local_rank=device_index)  # type: ignore[assignment, arg-type]
    elif name == "yuanrong":
        from ..backend.yuanrong_backend import YuanrongBackend

        backend = YuanrongBackend(None)  # type: ignore[assignment, arg-type]
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

    def ensure_initialized(self) -> None:
        ensure = getattr(self.backend, "ensure_initialized", None)
        if ensure is not None:
            ensure()

    def exists(self, keys):
        return self.backend.exists(keys)

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
        if self.name == "yuanrong":
            if self.backend._needs_dev_mem_pregister:
                raise NotImplementedError("YuanRong pre-registered device memory cannot be safely released here")
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
